"""
TUS-GAN — Critic (v5)
======================
Conditional Critic for WGAN-GP that scores diary sequences for
realism given the respondent's demographic conditioning vector.

v5 Updates:
  - Transformer-Based Temporal Block (Self-Attention over time).
  - Contrastive Projection Head for InfoNCE Loss (Demographic Entanglement).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# Helper: Conditional Instance Normalisation (CIN)
# ─────────────────────────────────────────────────────────────


class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.affine = nn.Linear(cond_dim, 2 * num_features)
        nn.init.ones_(self.affine.weight[:num_features])
        nn.init.zeros_(self.affine.weight[num_features:])
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.affine(c)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * self.norm(x) + beta


# ─────────────────────────────────────────────────────────────
# Helper: Temporal Transformer Block
# ─────────────────────────────────────────────────────────────


class TemporalTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_layers: int = 1, nhead: int = 4):
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
        # Reshape to (B, H, C)
        seq = x.squeeze(-1).permute(0, 2, 1)
        seq = seq + self.pos_embed[:, :H, :]
        out_seq = self.transformer(seq)
        out = out_seq.permute(0, 2, 1).unsqueeze(-1)
        return x + out


# ─────────────────────────────────────────────────────────────
# Helper: Downsampling Block (Residual + CIN + Spectral Norm)
# ─────────────────────────────────────────────────────────────


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(4, 1),
                stride=(2, 1),
                padding=(1, 0),
                bias=False,
            )
        )
        self.cin = ConditionalInstanceNorm2d(out_channels, cond_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Shortcut path for downsampling
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
            ConditionalInstanceNorm2d(out_channels, cond_dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Residual path
        res = self.shortcut[0](x)
        res = self.shortcut[1](res)
        res = self.shortcut[2](res, c)

        # Main path
        out = self.conv(x)
        out = self.cin(out, c)
        out = self.act(out)

        return out + res


# ─────────────────────────────────────────────────────────────
# Main Critic
# ─────────────────────────────────────────────────────────────


class Critic(nn.Module):
    def __init__(
        self,
        cond_dim: int = 83,
        num_districts: int = 71,
        num_states: int = 36,
        district_embed_dim: int = 16,
        state_embed_dim: int = 8,
        base_channels: int = 64,
        proj_dim: int = 64,  # Dimension for InfoNCE contrastive projection
    ):
        super().__init__()

        self.district_embed = nn.Embedding(num_districts, district_embed_dim)
        self.state_embed = nn.Embedding(num_states, state_embed_dim)

        full_cond_dim = cond_dim + district_embed_dim + state_embed_dim
        in_ch = 9 + full_cond_dim

        self.down1 = DownsampleBlock(in_ch, base_channels, full_cond_dim)
        self.down2 = DownsampleBlock(base_channels, base_channels * 2, full_cond_dim)

        # Replace SelfAttention2d with Transformer
        self.transformer1 = TemporalTransformerBlock(base_channels * 2, num_layers=2, nhead=4)

        self.down3 = DownsampleBlock(base_channels * 2, base_channels * 4, full_cond_dim)

        flat_dim = base_channels * 4 * 6 * 1

        # WGAN-GP Real/Fake Score Output
        self.output = nn.Linear(flat_dim, 1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Contrastive Projection Heads (InfoNCE)
        self.feat_proj = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(base_channels * 2, proj_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(full_cond_dim, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(base_channels * 2, proj_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight_orig'):
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, diary, cond_vector, district_ids, state_ids, return_proj=False):
        d_emb = self.district_embed(district_ids)
        s_emb = self.state_embed(state_ids)
        c = torch.cat([cond_vector, d_emb, s_emb], dim=1)

        c_spatial = c.unsqueeze(-1).unsqueeze(-1)
        c_spatial = c_spatial.expand(-1, -1, diary.size(2), diary.size(3))
        x = torch.cat([diary, c_spatial], dim=1)

        x = self.down1(x, c)
        x = self.down2(x, c)
        x = self.transformer1(x)
        x = self.down3(x, c)

        x_flat = x.view(x.size(0), -1)
        score = self.output(x_flat)

        if return_proj:
            x_pooled = self.temporal_pool(x.squeeze(-1)).squeeze(-1)
            f_proj = self.feat_proj(x_pooled)
            c_proj = self.cond_proj(c)
            # L2 normalize for cosine similarity in InfoNCE
            f_proj = F.normalize(f_proj, p=2, dim=1)
            c_proj = F.normalize(c_proj, p=2, dim=1)
            return score, f_proj, c_proj

        return score


def compute_gradient_penalty(critic, real, fake, cond, dist, state, device, lambda_gp=10.0):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    x_hat = eps * real.detach() + (1 - eps) * fake.detach()
    x_hat.requires_grad_(True)

    # We only need the score for gradient penalty, so return_proj=False
    score_hat = critic(x_hat, cond, dist, state, return_proj=False)

    gradients = torch.autograd.grad(
        outputs=score_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(score_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_norm = gradients.view(B, -1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1.0) ** 2).mean()


def critic_loss(real_scores, fake_scores, gp):
    return fake_scores.mean() - real_scores.mean() + gp


def generator_loss(fake_scores):
    return -fake_scores.mean()


if __name__ == "__main__":
    BATCH = 4
    D = Critic()
    real = torch.randn(BATCH, 9, 48, 1)
    fake = torch.randn(BATCH, 9, 48, 1)
    cv = torch.zeros(BATCH, 83)
    di = torch.randint(0, 71, (BATCH,))
    si = torch.randint(0, 36, (BATCH,))

    scores = D(real, cv, di, si)
    print(f"Scores shape: {scores.shape}")

    scores, f_proj, c_proj = D(real, cv, di, si, return_proj=True)
    print(f"Scores shape: {scores.shape}")
    print(f"Feat Proj shape: {f_proj.shape}")
    print(f"Cond Proj shape: {c_proj.shape}")

    gp = compute_gradient_penalty(D, real, fake, cv, di, si, torch.device("cpu"))
    print(f"GP: {gp.item():.4f}")
    print("Smoke test passed ✓")
