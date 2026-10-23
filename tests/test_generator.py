"""Unit tests for v4/generator.py: ConditionalBatchNorm2d, SelfAttention2d,
UpsampleBlock, and the Generator network as a whole.
"""

import pytest
import torch

from v4.generator import (
    ConditionalBatchNorm2d,
    Generator,
    SelfAttention2d,
    UpsampleBlock,
)


def make_generator_inputs(batch_size, noise_dim=128, cond_dim=82, num_districts=71, num_states=36):
    z = torch.randn(batch_size, noise_dim)
    cond_vector = torch.zeros(batch_size, cond_dim)
    district_ids = torch.randint(0, num_districts, (batch_size,))
    state_ids = torch.randint(0, num_states, (batch_size,))
    return z, cond_vector, district_ids, state_ids


class TestConditionalBatchNorm2d:
    def test_output_shape_matches_input(self):
        cbn = ConditionalBatchNorm2d(num_features=16, cond_dim=10)
        x = torch.randn(4, 16, 24, 1)
        c = torch.randn(4, 10)
        out = cbn(x, c)
        assert out.shape == x.shape

    def test_one_hot_condition_reduces_to_plain_batchnorm(self):
        # weight[:num_features] is init'd to all-ones and weight[num_features:]
        # to all-zeros, so for a one-hot condition vector (columns sum to 1),
        # gamma == 1 and beta == 0 everywhere -> output == unconditional BN(x).
        cbn = ConditionalBatchNorm2d(num_features=8, cond_dim=5)
        x = torch.randn(4, 8, 12, 1)
        c = torch.zeros(4, 5)
        c[:, 0] = 1.0
        out = cbn(x, c)
        expected = cbn.bn(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_zero_condition_vector_zeroes_the_output(self):
        # A non-obvious consequence of the same init scheme: an all-zero
        # condition vector drives both gamma and beta to zero, not just beta.
        cbn = ConditionalBatchNorm2d(num_features=8, cond_dim=5)
        x = torch.randn(4, 8, 12, 1)
        c = torch.zeros(4, 5)
        out = cbn(x, c)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


class TestSelfAttention2d:
    def test_output_shape_preserved(self):
        attn = SelfAttention2d(in_channels=32)
        x = torch.randn(4, 32, 24, 1)
        out = attn(x)
        assert out.shape == x.shape

    def test_acts_as_identity_before_training(self):
        # gamma is a learnable scalar initialized to zero, so at init the
        # attention branch must contribute nothing: out == x exactly.
        attn = SelfAttention2d(in_channels=16)
        x = torch.randn(4, 16, 12, 1)
        out = attn(x)
        assert torch.equal(out, x)


class TestUpsampleBlock:
    def test_doubles_the_time_dimension(self):
        block = UpsampleBlock(in_channels=256, out_channels=128, cond_dim=107)
        x = torch.randn(4, 256, 12, 1)
        c = torch.randn(4, 107)
        out = block(x, c)
        assert out.shape == (4, 128, 24, 1)

    def test_output_is_finite(self):
        block = UpsampleBlock(in_channels=64, out_channels=32, cond_dim=20)
        out = block(torch.randn(2, 64, 6, 1), torch.randn(2, 20))
        assert torch.isfinite(out).all()


class TestGenerator:
    def test_output_shape(self, batch_size):
        gen = Generator()
        out = gen(*make_generator_inputs(batch_size))
        assert out.shape == (batch_size, 9, 48, 1)

    def test_output_is_within_wgan_range(self, batch_size):
        gen = Generator()
        out = gen(*make_generator_inputs(batch_size))
        assert out.min().item() >= -1.0 - 1e-5
        assert out.max().item() <= 1.0 + 1e-5

    @pytest.mark.parametrize("hard", [True, False])
    def test_gumbel_softmax_output_is_categorical_per_timeslot(self, batch_size, hard):
        # Channel dim (dim=1, the 9 ICATUS divisions) must sum to 1 at every
        # one of the 48 half-hour slots, for both the hard (one-hot) and soft
        # (relaxed) Gumbel-Softmax modes.
        gen = Generator()
        out = gen(*make_generator_inputs(batch_size), hard=hard)
        probs = (out + 1.0) / 2.0  # undo the [0,1] -> [-1,1] rescale
        totals = probs.sum(dim=1)
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-4)

    def test_batch_size_one_edge_case(self):
        gen = Generator()
        out = gen(*make_generator_inputs(1))
        assert out.shape == (1, 9, 48, 1)

    def test_supports_non_default_conditioning_dimensions(self):
        gen = Generator(
            cond_dim=20, num_districts=5, num_states=3, district_embed_dim=4, state_embed_dim=2
        )
        z = torch.randn(2, 128)
        cond_vector = torch.zeros(2, 20)
        district_ids = torch.randint(0, 5, (2,))
        state_ids = torch.randint(0, 3, (2,))
        out = gen(z, cond_vector, district_ids, state_ids)
        assert out.shape == (2, 9, 48, 1)

    def test_district_id_out_of_range_raises(self, batch_size):
        gen = Generator(num_districts=71)
        z, cond_vector, _, state_ids = make_generator_inputs(batch_size)
        out_of_range_ids = torch.full((batch_size,), 71)  # valid range is [0, 70]
        with pytest.raises(IndexError):
            gen(z, cond_vector, out_of_range_ids, state_ids)

    def test_attention_projection_gradients_are_zero_at_init(self, batch_size):
        """Documents a real, non-obvious property of SelfAttention2d: gamma
        is initialized to 0 (a SAGAN-style trick so attention starts as a
        no-op), so on the very first backward pass query/key/value receive
        EXACTLY zero gradient. gamma itself still gets a live gradient, so
        this is a transient one-step cold start, not a permanent dead
        gradient like the hard-logit-masking failure mode in the README."""
        gen = Generator()
        out = gen(*make_generator_inputs(batch_size))
        out.sum().backward()
        assert gen.attn1.gamma.grad is not None
        assert gen.attn1.gamma.grad.item() != 0.0
        assert gen.attn1.query.weight.grad.abs().sum().item() == 0.0
        assert gen.attn1.key.weight.grad.abs().sum().item() == 0.0
        assert gen.attn1.value.weight.grad.abs().sum().item() == 0.0

    def test_gradients_flow_to_every_parameter_after_one_training_step(self, batch_size):
        """Regression guard for the 'Dead-Gradient' failure mode the README
        describes. Attention's query/key/value are zero-gradient only at
        init (see test above); once gamma moves off 0 after one optimizer
        step, every parameter -- including the attention projections --
        must receive a live, finite gradient."""
        gen = Generator()
        inputs = make_generator_inputs(batch_size)

        def step():
            out = gen(*inputs)
            out.sum().backward()

        step()
        with torch.no_grad():
            for param in gen.parameters():
                if param.grad is not None:
                    param -= 0.05 * param.grad
        gen.zero_grad()
        step()

        for name, param in gen.named_parameters():
            assert param.grad is not None, f"{name} received no gradient at all"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"
            assert param.grad.abs().sum().item() > 0.0, f"{name} has a dead (all-zero) gradient"

    def test_weight_init_produces_finite_parameters(self):
        gen = Generator()
        for name, param in gen.named_parameters():
            assert torch.isfinite(param).all(), f"{name} contains NaN/Inf right after init"

    def test_forward_is_deterministic_given_the_same_seed(self, batch_size):
        gen = Generator()
        inputs = make_generator_inputs(batch_size)

        torch.manual_seed(123)
        out1 = gen(*inputs, hard=False)
        torch.manual_seed(123)
        out2 = gen(*inputs, hard=False)
        assert torch.allclose(out1, out2)
