"""
TUS-GAN — Generator (v5.1 / Cell D)
====================================
Transformer-Based Conditional Generator with Soft Penalty Enforcement (Unmasked Logits).
Corresponds to Cell D in the 2x2 Experimental Design of TUS-GAN V2.

Architecture:
  - Temporal Transformer Block (Multi-Head Self-Attention over time)
  - Unmasked Logits in Training: Allows full gradient flow through Gumbel-Softmax
  - Differentiable Soft Penalty compatibility (returns soft probability distribution)
  - Optional inference-time deterministic masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Helper: Conditional Batch Normalisation
# ─────────────────────────────────────────────────────────────


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.affine = nn.Linear(cond_dim, 2 * num_features)
        nn.init.ones_(self.affine.weight[:num_features])
        nn.init.zeros_(self.affine.weight[num_features:])
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.affine(c)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * self.bn(x) + beta


# ─────────────────────────────────────────────────────────────
# Helper: Temporal Transformer Block
# ─────────────────────────────────────────────────────────────


class TemporalTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.channels = channels
        self.pos_embed = nn.Parameter(torch.randn(1, 48, channels) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=channels * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, 1) where H is time steps
        B, C, H, W = x.shape
        # Reshape to (B, H, C) for transformer
        seq = x.squeeze(-1).permute(0, 2, 1)
        # Apply transformer
        seq = seq + self.pos_embed[:, :H, :]
        out_seq = self.transformer(seq)
        # Reshape back to (B, C, H, 1)
        out = out_seq.permute(0, 2, 1).unsqueeze(-1)
        return x + out  # Residual connection


# ─────────────────────────────────────────────────────────────
# Helper: One Upsampling Block (Residual + CBN)
# ─────────────────────────────────────────────────────────────


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(4, 1),
            stride=(2, 1),
            padding=(1, 0),
            bias=False,
        )
        self.cbn = ConditionalBatchNorm2d(out_channels, cond_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Shortcut link for residual learning
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ConditionalBatchNorm2d(out_channels, cond_dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        res = self.shortcut[0](x)
        res = self.shortcut[1](res)
        res = self.shortcut[2](res, c)

        out = self.conv_t(x)
        out = self.cbn(out, c)
        out = self.act(out)

        return out + res


# ─────────────────────────────────────────────────────────────
# Main Generator (v5.1: Transformer Backbone + Soft Penalties)
# ─────────────────────────────────────────────────────────────


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int = 128,
        cond_dim: int = 83,
        num_districts: int = 71,
        num_states: int = 36,
        district_embed_dim: int = 16,
        state_embed_dim: int = 8,
        base_channels: int = 256,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.base_channels = base_channels

        # Learned Embeddings
        self.district_embed = nn.Embedding(num_districts, district_embed_dim)
        self.state_embed = nn.Embedding(num_states, state_embed_dim)

        # Full conditioning vector includes: OH vector + District + State
        full_cond_dim = cond_dim + district_embed_dim + state_embed_dim
        self.full_cond_dim = full_cond_dim

        # Backbone
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + full_cond_dim, base_channels * 12 * 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.start_time = 12
        self.start_ch = base_channels

        self.up1 = UpsampleBlock(base_channels, base_channels // 2, full_cond_dim)
        self.transformer1 = TemporalTransformerBlock(base_channels // 2, num_layers=2, nhead=4)
        self.up2 = UpsampleBlock(base_channels // 2, base_channels // 4, full_cond_dim)

        self.out_conv = nn.Conv2d(
            base_channels // 4, 9, kernel_size=(3, 1), stride=1, padding=(1, 0)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear) and not hasattr(m, "_is_cbn_affine"):
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        z: torch.Tensor,
        cond_vector: torch.Tensor,
        district_ids: torch.Tensor,
        state_ids: torch.Tensor,
        temp: float = 1.0,
        hard: bool = True,
        return_soft: bool = False,
        apply_hard_mask: bool = False,  # False during training (soft penalty), True at inference if desired
    ):
        # Embeddings
        d_emb = self.district_embed(district_ids)
        s_emb = self.state_embed(state_ids)

        # Full condition
        c = torch.cat([cond_vector, d_emb, s_emb], dim=1)

        # Initial map
        h = self.fc(torch.cat([z, c], dim=1))
        h = h.view(-1, self.start_ch, self.start_time, 1)

        # Upsample with Temporal Transformer
        h = self.up1(h, c)
        h = self.transformer1(h)
        h = self.up2(h, c)

        # Output logits
        logits = self.out_conv(h)  # (B, 9, 48, 1)

        # INFERENCE ONLY / OPTIONAL MASKING
        # In v5.1 training, logits remain unmasked so gradients flow freely.
        if apply_hard_mask:
            is_child = cond_vector[:, 0] == 1
            if is_child.any():
                logits[is_child, 0, :, :] = -1e9

        # Apply Gumbel-Softmax along division dimension (dim=1)
        y_hard = F.gumbel_softmax(logits, tau=temp, hard=True, dim=1)
        y_soft = F.gumbel_softmax(logits, tau=temp, hard=False, dim=1)

        output_hard = 2.0 * y_hard - 1.0
        output_soft = 2.0 * y_soft - 1.0

        if return_soft:
            return output_hard, output_soft
        return output_hard if hard else output_soft


if __name__ == "__main__":
    BATCH = 4
    G = Generator()
    z = torch.randn(BATCH, 128)
    cv = torch.zeros(BATCH, 83)
    cv[0, 0] = 1.0  # Test child profile
    di = torch.randint(0, 71, (BATCH,))
    si = torch.randint(0, 36, (BATCH,))
    
    # Training forward pass (unmasked logits, returns both hard and soft for loss)
    fake_hard, fake_soft = G(z, cv, di, si, return_soft=True)
    print(f"Hard Output shape: {fake_hard.shape}")
    print(f"Soft Output shape: {fake_soft.shape}")
    assert fake_hard.shape == (BATCH, 9, 48, 1)
    assert fake_soft.shape == (BATCH, 9, 48, 1)
    
    # Verify soft output contains non-zero probability mass on channel 0 to permit soft gradients
    soft_probs = (fake_soft.squeeze(-1) + 1.0) / 2.0
    print(f"Child employment soft mass (gradient enabled): {soft_probs[0, 0, :].mean().item():.4f}")
    print("v5.1 Generator Smoke Test Passed ✓")
