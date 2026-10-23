"""Unit tests for v4/critic.py: ConditionalInstanceNorm2d, SelfAttention2d,
DownsampleBlock, the Critic network, and the WGAN-GP loss helpers
(compute_gradient_penalty, critic_loss, generator_loss).
"""

import torch

from v4.critic import (
    Critic,
    ConditionalInstanceNorm2d,
    DownsampleBlock,
    SelfAttention2d,
    compute_gradient_penalty,
    critic_loss,
    generator_loss,
)
from v4.generator import Generator


def make_critic_inputs(batch_size, cond_dim=82, num_districts=71, num_states=36):
    diary = torch.randn(batch_size, 9, 48, 1)
    cond_vector = torch.zeros(batch_size, cond_dim)
    district_ids = torch.randint(0, num_districts, (batch_size,))
    state_ids = torch.randint(0, num_states, (batch_size,))
    return diary, cond_vector, district_ids, state_ids


class TestConditionalInstanceNorm2d:
    def test_output_shape_matches_input(self):
        cin = ConditionalInstanceNorm2d(num_features=16, cond_dim=10)
        x = torch.randn(4, 16, 24, 1)
        c = torch.randn(4, 10)
        out = cin(x, c)
        assert out.shape == x.shape

    def test_one_hot_condition_reduces_to_plain_instancenorm(self):
        # Same init scheme as the generator's ConditionalBatchNorm2d: a
        # one-hot condition vector drives gamma to 1 and beta to 0.
        cin = ConditionalInstanceNorm2d(num_features=8, cond_dim=5)
        x = torch.randn(4, 8, 12, 1)
        c = torch.zeros(4, 5)
        c[:, 0] = 1.0
        out = cin(x, c)
        expected = cin.norm(x)
        assert torch.allclose(out, expected, atol=1e-5)


class TestSelfAttention2d:
    def test_output_shape_preserved(self):
        attn = SelfAttention2d(in_channels=32)
        x = torch.randn(4, 32, 12, 1)
        out = attn(x)
        assert out.shape == x.shape

    def test_acts_as_identity_before_training(self):
        attn = SelfAttention2d(in_channels=16)
        x = torch.randn(4, 16, 6, 1)
        out = attn(x)
        assert torch.equal(out, x)


class TestDownsampleBlock:
    def test_halves_the_time_dimension(self):
        block = DownsampleBlock(in_channels=116, out_channels=64, cond_dim=107)
        x = torch.randn(4, 116, 48, 1)
        c = torch.randn(4, 107)
        out = block(x, c)
        assert out.shape == (4, 64, 24, 1)

    def test_output_is_finite(self):
        block = DownsampleBlock(in_channels=32, out_channels=16, cond_dim=10)
        out = block(torch.randn(2, 32, 12, 1), torch.randn(2, 10))
        assert torch.isfinite(out).all()


class TestCritic:
    def test_output_shape(self, batch_size):
        critic = Critic()
        score = critic(*make_critic_inputs(batch_size))
        assert score.shape == (batch_size, 1)

    def test_batch_size_one_edge_case(self):
        critic = Critic()
        score = critic(*make_critic_inputs(1))
        assert score.shape == (1, 1)

    def test_spectral_norm_wraps_conv_and_output_layers(self):
        critic = Critic()
        # nn.utils.spectral_norm reparametrizes the weight into weight_orig +
        # a weight_u buffer computed via power iteration on every forward.
        assert hasattr(critic.down1.conv, "weight_orig")
        assert hasattr(critic.down1.conv, "weight_u")
        assert hasattr(critic.output, "weight_orig")

    def test_attention_projection_gradients_are_zero_at_init(self, batch_size):
        """Same transient cold-start documented in test_generator.py: the
        critic's SelfAttention2d also starts with gamma=0, so query/key/value
        get exactly zero gradient on the very first backward pass."""
        critic = Critic()
        score = critic(*make_critic_inputs(batch_size))
        score.sum().backward()
        assert critic.attn1.gamma.grad is not None
        assert critic.attn1.gamma.grad.item() != 0.0
        assert critic.attn1.query.weight.grad.abs().sum().item() == 0.0

    def test_gradients_flow_to_every_parameter_after_one_training_step(self, batch_size):
        """Regression guard mirroring the generator's dead-gradient test:
        after one optimizer step moves attention's gamma off 0, every
        learnable Critic parameter must get a finite, non-zero gradient."""
        critic = Critic()
        inputs = make_critic_inputs(batch_size)

        def step():
            score = critic(*inputs)
            score.sum().backward()

        step()
        with torch.no_grad():
            for param in critic.parameters():
                if param.grad is not None:
                    param -= 0.05 * param.grad
        critic.zero_grad()
        step()

        for name, param in critic.named_parameters():
            assert param.grad is not None, f"{name} received no gradient at all"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"
            assert param.grad.abs().sum().item() > 0.0, f"{name} has a dead (all-zero) gradient"

    def test_weight_init_produces_finite_parameters(self):
        critic = Critic()
        for name, param in critic.named_parameters():
            assert torch.isfinite(param).all(), f"{name} contains NaN/Inf right after init"


class TestComputeGradientPenalty:
    def test_is_always_nonnegative(self, batch_size):
        critic = Critic()
        real, cond_vector, district_ids, state_ids = make_critic_inputs(batch_size)
        fake = torch.randn_like(real)
        gp = compute_gradient_penalty(
            critic, real, fake, cond_vector, district_ids, state_ids, torch.device("cpu")
        )
        assert gp.item() >= 0.0

    def test_scales_linearly_with_lambda_gp(self, batch_size):
        # eval() freezes spectral_norm's power-iteration buffers; in train
        # mode each critic(...) call mutates them in place, so the two
        # calls below would see slightly different effective weights.
        critic = Critic().eval()
        real, cond_vector, district_ids, state_ids = make_critic_inputs(batch_size)
        fake = torch.randn_like(real)

        torch.manual_seed(7)
        gp_default = compute_gradient_penalty(
            critic, real, fake, cond_vector, district_ids, state_ids,
            torch.device("cpu"), lambda_gp=10.0,
        )
        torch.manual_seed(7)
        gp_scaled = compute_gradient_penalty(
            critic, real, fake, cond_vector, district_ids, state_ids,
            torch.device("cpu"), lambda_gp=20.0,
        )
        assert torch.allclose(gp_scaled, gp_default * 2.0, atol=1e-4)

    def test_supports_double_backward_into_critic_parameters(self, batch_size):
        # WGAN-GP requires create_graph=True so that backpropagating through
        # the gradient penalty itself updates the critic's own parameters.
        critic = Critic()
        real, cond_vector, district_ids, state_ids = make_critic_inputs(batch_size)
        fake = torch.randn_like(real)
        gp = compute_gradient_penalty(
            critic, real, fake, cond_vector, district_ids, state_ids, torch.device("cpu")
        )
        gp.backward()
        grad_norms = [p.grad.abs().sum().item() for p in critic.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert any(g > 0.0 for g in grad_norms)


class TestLossFormulas:
    def test_critic_loss_matches_wgan_gp_formula(self):
        real_scores = torch.tensor([1.0, 2.0, 3.0])
        fake_scores = torch.tensor([0.5, 0.5, 0.5])
        gp = torch.tensor(0.25)
        loss = critic_loss(real_scores, fake_scores, gp)
        expected = fake_scores.mean() - real_scores.mean() + gp
        assert torch.allclose(loss, expected)

    def test_generator_loss_matches_negative_mean_score(self):
        fake_scores = torch.tensor([1.0, -2.0, 3.0])
        loss = generator_loss(fake_scores)
        assert torch.allclose(loss, -fake_scores.mean())


class TestFullWganGpStep:
    def test_gradients_flow_through_generator_and_critic_together(self, batch_size):
        """End-to-end smoke test wiring generator.py and critic.py together
        exactly as v4/train.py would use them for a single critic + generator
        step, guarding against a dead-gradient regression in either network.
        Runs the critic step twice: attention's gamma starts at 0 (see the
        cold-start tests above), so the first pass alone isn't a valid probe
        for permanently-dead gradients -- the second pass, after gamma has
        moved, is."""
        gen = Generator()
        critic = Critic()
        real, cond_vector, district_ids, state_ids = make_critic_inputs(batch_size)
        z = torch.randn(batch_size, gen.noise_dim)

        def critic_step():
            fake = gen(z, cond_vector, district_ids, state_ids)
            real_scores = critic(real, cond_vector, district_ids, state_ids)
            fake_scores = critic(fake.detach(), cond_vector, district_ids, state_ids)
            gp = compute_gradient_penalty(
                critic, real, fake.detach(), cond_vector, district_ids, state_ids,
                torch.device("cpu"),
            )
            critic_loss(real_scores, fake_scores, gp).backward()

        critic_step()
        with torch.no_grad():
            for param in critic.parameters():
                if param.grad is not None:
                    param -= 0.05 * param.grad
        critic.zero_grad()
        critic_step()

        # critic.output.bias is a mathematically permanent exception: with
        # equal-sized real/fake batches it contributes identically to every
        # score, so it cancels exactly in mean(fake) - mean(real), and the
        # gradient-penalty term doesn't depend on an additive bias either.
        # It is a structurally dead parameter under this loss, not a bug.
        for name, param in critic.named_parameters():
            if name == "output.bias":
                assert param.grad is not None and param.grad.abs().sum().item() == 0.0
                continue
            assert param.grad is not None and param.grad.abs().sum().item() > 0.0, (
                f"critic.{name} has a dead gradient after a WGAN-GP critic step"
            )

        critic.zero_grad()

        def generator_step():
            fake = gen(z, cond_vector, district_ids, state_ids)
            generator_loss(critic(fake, cond_vector, district_ids, state_ids)).backward()

        generator_step()
        with torch.no_grad():
            for param in gen.parameters():
                if param.grad is not None:
                    param -= 0.05 * param.grad
        gen.zero_grad()
        generator_step()

        for name, param in gen.named_parameters():
            assert param.grad is not None and param.grad.abs().sum().item() > 0.0, (
                f"generator.{name} has a dead gradient after a WGAN-GP generator step"
            )
