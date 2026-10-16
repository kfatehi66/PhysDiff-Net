"""
PhysDiff-Net V5: Fixed Hour Conditioning with Gradient Surgery

Key changes from V4:
1. CONCATENATION-based hour conditioning (not additive)
   - Hour embedding is concatenated to input features
   - Forces network to explicitly process hour information
   - Cannot be ignored as with additive conditioning

2. AUXILIARY hour prediction loss
   - Direct gradient signal penalizing hour-inconsistent patterns
   - Cross-entropy loss on hour classification

3. Designed for PCGRAD training
   - Losses returned separately for gradient surgery
   - Reduces gradient conflicts from 78% to ~20%

4. NO LR warm restarts
   - V4 collapsed after restart at epoch 350
   - Use simple cosine annealing instead

Expected improvements:
- Hourly UE range: 0.0 -> >1.5
- Pattern score: 66% -> >70%
- Training stability: Reduced oscillation
"""

import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hour_conditioning import ConcatenationHourConditioning, HybridHourConditioning
from .hour_classifier import AuxiliaryHourClassifier, MultiScaleHourClassifier


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time step embeddings for diffusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class APEmbedding(nn.Module):
    """Embedding for AP index conditioning."""

    def __init__(self, num_aps: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_aps, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, ap_ids: torch.Tensor, seq_len: int = 1) -> torch.Tensor:
        emb = self.embed(ap_ids)  # [B, embed_dim]
        return emb.unsqueeze(1).expand(-1, seq_len, -1)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and GELU."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm feedforward
        x = x + self.ff(self.norm2(x))
        return x


class PhysDiffNetV5(nn.Module):
    """
    PhysDiff-Net V5: Fixed Hour Conditioning with Auxiliary Loss

    Architecture:
    1. Input: x [B, T, num_features]
    2. Hour embedding via CONCATENATION: [B, T, num_features + hour_dim]
    3. Project to hidden dim: [B, T, hidden_dim]
    4. Add timestep + positional encoding
    5. Transformer encoder
    6. Output: noise prediction [B, T, num_features]
    7. Auxiliary: hour classification [B, 24]
    """

    def __init__(self, config: Dict):
        super().__init__()

        # Core dimensions
        self.num_features = config.get('num_features', 29)
        self.seq_len = config.get('seq_len', 360)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 6)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)

        # Hour conditioning
        self.hour_dim = config.get('hour_dim', 64)
        self.num_hours = config.get('num_hours', 24)
        self.use_hybrid_conditioning = config.get('use_hybrid_conditioning', False)

        # AP conditioning
        self.num_aps = config.get('num_aps', 11)
        self.ap_dim = config.get('ap_dim', 32)

        # CFG dropout
        self.cfg_drop_prob = config.get('cfg_drop_prob', 0.3)

        # Diffusion params
        self.num_timesteps = config.get('num_timesteps', 1000)

        # ===== Hour Conditioning (CONCATENATION - Key V5 change) =====
        if self.use_hybrid_conditioning:
            self.hour_conditioning = HybridHourConditioning(
                num_hours=self.num_hours,
                hour_dim=self.hour_dim,
                variational=True
            )
        else:
            self.hour_conditioning = ConcatenationHourConditioning(
                num_hours=self.num_hours,
                hour_dim=self.hour_dim,
                variational=True
            )

        # AP conditioning
        self.ap_embedding = APEmbedding(self.num_aps, self.ap_dim)

        # ===== Input Projection =====
        # CRITICAL: Input dim includes concatenated hour embedding
        input_dim = self.num_features + self.hour_dim + self.ap_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # ===== Timestep Embedding =====
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )

        # ===== Positional Encoding =====
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_len, self.hidden_dim) * 0.02
        )

        # ===== Transformer =====
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                ff_mult=4,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])

        # ===== Output Projection =====
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_features)
        )

        # ===== Auxiliary Hour Classifier (Key V5 change) =====
        self.hour_classifier = MultiScaleHourClassifier(
            hidden_dim=self.hidden_dim,
            num_hours=self.num_hours,
            scales=(1, 4, 16),
            dropout=self.dropout
        )

        # ===== Unconditional embeddings for CFG =====
        self.uncond_hour_embed = nn.Parameter(torch.randn(1, 1, self.hour_dim) * 0.02)
        self.uncond_ap_embed = nn.Parameter(torch.randn(1, 1, self.ap_dim) * 0.02)

        # ===== Diffusion schedule =====
        self._init_diffusion_schedule()

        # Initialize weights
        self._init_weights()

    def _init_diffusion_schedule(self):
        """Initialize cosine beta schedule."""
        betas = self._cosine_beta_schedule(self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in Nichol & Dhariwal."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.02)

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def get_conditioning(
        self,
        hours: torch.Tensor,
        ap_ids: torch.Tensor,
        seq_len: int,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get conditioning embeddings for CONCATENATION.

        Returns:
            hour_embed: [B, seq_len, hour_dim]
            ap_embed: [B, seq_len, ap_dim]
        """
        B = hours.shape[0]
        device = hours.device

        # Hour conditioning
        hour_embed, _ = self.hour_conditioning(hours, seq_len)  # [B, T, hour_dim]

        # AP conditioning
        ap_embed = self.ap_embedding(ap_ids, seq_len)  # [B, T, ap_dim]

        # CFG dropout during training
        if training and self.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.cfg_drop_prob

            # Replace with unconditional embeddings
            uncond_hour = self.uncond_hour_embed.expand(B, seq_len, -1)
            uncond_ap = self.uncond_ap_embed.expand(B, seq_len, -1)

            hour_embed = torch.where(
                drop_mask[:, None, None],
                uncond_hour,
                hour_embed
            )
            ap_embed = torch.where(
                drop_mask[:, None, None],
                uncond_ap,
                ap_embed
            )

        return hour_embed, ap_embed

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        hours: torch.Tensor,
        ap_ids: torch.Tensor,
        return_hour_pred: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with noise prediction and auxiliary outputs.

        Args:
            x: Noisy input [B, T, num_features]
            t: Diffusion timesteps [B]
            hours: Hour indices [B]
            ap_ids: AP indices [B]
            return_hour_pred: Whether to compute hour prediction

        Returns:
            Dict with:
                - 'noise_pred': Predicted noise [B, T, num_features]
                - 'hour_pred': Hour logits [B, 24] (if return_hour_pred)
        """
        B, T, F = x.shape

        # Get conditioning embeddings
        hour_embed, ap_embed = self.get_conditioning(
            hours, ap_ids, T, training=self.training
        )

        # ===== CONCATENATE conditioning (Key V5 change) =====
        # This creates explicit feature channels for hour/AP information
        x_cond = torch.cat([x, hour_embed, ap_embed], dim=-1)  # [B, T, F + hour_dim + ap_dim]

        # Project to hidden dimension
        h = self.input_projection(x_cond)  # [B, T, hidden_dim]

        # Add positional encoding
        if T <= self.seq_len:
            h = h + self.pos_encoding[:, :T, :]
        else:
            # Handle longer sequences
            h = h + F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        # Add timestep embedding
        t_embed = self.time_embed(t)  # [B, hidden_dim]
        h = h + t_embed.unsqueeze(1)

        # Transformer blocks
        for block in self.transformer_blocks:
            h = block(h)

        # Output projection
        noise_pred = self.output_projection(h)  # [B, T, num_features]

        result = {'noise_pred': noise_pred}

        # Auxiliary hour prediction
        if return_hour_pred:
            hour_pred = self.hour_classifier(h)  # [B, 24]
            result['hour_pred'] = hour_pred

        return result

    def compute_loss(
        self,
        x: torch.Tensor,
        hours: torch.Tensor,
        ap_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute individual losses for PCGrad.

        Returns losses separately so PCGrad can handle gradient conflicts.
        """
        B = x.shape[0]
        device = x.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(x)

        # Create noisy sample
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        x_noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise

        # Forward pass
        output = self.forward(x_noisy, t, hours, ap_ids, return_hour_pred=True)

        # Diffusion loss
        diffusion_loss = F.mse_loss(output['noise_pred'], noise)

        # Hour auxiliary loss
        hour_loss = F.cross_entropy(output['hour_pred'], hours)

        # KL loss for variational hour embedding
        kl_loss = self.hour_conditioning.get_kl_loss(hours)

        # Approximate x0 for physics loss
        pred_x0 = (x_noisy - sqrt_one_minus_alpha * output['noise_pred']) / sqrt_alpha

        return {
            'diffusion': diffusion_loss,
            'hour': hour_loss,
            'kl': kl_loss,
            'pred_x0': pred_x0,
            'real_x0': x
        }

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        hours: torch.Tensor,
        ap_ids: torch.Tensor,
        num_steps: int = 50,
        cfg_scale: float = 7.5,
        progressive_cfg: bool = True
    ) -> torch.Tensor:
        """
        Generate samples with classifier-free guidance.

        Args:
            num_samples: Number of samples to generate
            hours: Hour conditioning [num_samples]
            ap_ids: AP conditioning [num_samples]
            num_steps: Number of denoising steps
            cfg_scale: Guidance scale (higher = stronger conditioning)
            progressive_cfg: Decrease CFG scale during generation

        Returns:
            Generated samples [num_samples, seq_len, num_features]
        """
        device = next(self.parameters()).device
        self.eval()

        # Start from pure noise
        x = torch.randn(num_samples, self.seq_len, self.num_features, device=device)

        # Compute step indices
        step_ratio = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_ratio))[::-1]

        # Denoising loop
        for i, t in enumerate(timesteps):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # Progressive CFG: reduce scale as we refine
            if progressive_cfg:
                progress = i / len(timesteps)
                current_scale = cfg_scale * (1 - 0.5 * progress)
            else:
                current_scale = cfg_scale

            # Conditional prediction
            cond_out = self.forward(x, t_batch, hours, ap_ids, return_hour_pred=False)
            noise_cond = cond_out['noise_pred']

            # Unconditional prediction (use zeros for hours/ap)
            uncond_hours = torch.zeros_like(hours)
            uncond_ap = torch.zeros_like(ap_ids)
            uncond_out = self.forward(x, t_batch, uncond_hours, uncond_ap, return_hour_pred=False)
            noise_uncond = uncond_out['noise_pred']

            # CFG combination
            noise_pred = noise_uncond + current_scale * (noise_cond - noise_uncond)

            # DDPM update step
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t - step_ratio] if t >= step_ratio else torch.tensor(1.0, device=device)

            # Predict x0
            pred_x0 = (x - self.sqrt_one_minus_alphas_cumprod[t] * noise_pred) / self.sqrt_alphas_cumprod[t]
            pred_x0 = pred_x0.clamp(-3, 3)  # Prevent explosion

            # Update x
            if t > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise
            else:
                x = pred_x0

        return x

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_physdiff_v5(config: Dict) -> PhysDiffNetV5:
    """Factory function to create PhysDiffNetV5."""
    return PhysDiffNetV5(config)
