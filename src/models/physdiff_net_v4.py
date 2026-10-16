"""
PhysDiff-Net V4: Variational Conditioning with Diversity Preservation

Key innovations over V3:
1. Variational Embeddings (learn mu, sigma instead of fixed vectors)
2. KL regularization to prevent embedding collapse
3. Lighter conditioning injection (scaled additive)
4. Higher CFG drop probability (0.3) for better unconditional baseline
5. Built on V2's physics-compliant architecture

This addresses V3's "variance collapse" where outputs became deterministic.
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time step embeddings for diffusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class VariationalEmbedding(nn.Module):
    """
    Variational conditioner that outputs distribution parameters.

    Instead of: embed(hour) -> fixed vector
    Now: embed(hour) -> (mu, sigma) -> sample z = mu + sigma * epsilon

    This prevents the "deterministic lookup" problem from V3.
    The KL divergence regularizes towards N(0,1), preventing collapse.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        min_std: float = 0.05,
        max_std: float = 0.3
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.min_std = min_std
        self.max_std = max_std

        # Mean embedding
        self.mu_embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Log-variance embedding (learn log(sigma^2) for numerical stability)
        self.logvar_embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Initialize logvar to small values (sigma around 0.1)
        nn.init.constant_(self.logvar_embedding.weight, -2.0)

        # Initialize mu with small random values
        nn.init.normal_(self.mu_embedding.weight, mean=0.0, std=0.1)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
        return_params: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input indices [B]
            training: If True, sample from distribution. If False, use mean.
            return_params: If True, also return (mu, std) for KL loss

        Returns:
            z: Sampled embedding [B, D]
            (optional) mu, std: Distribution parameters
        """
        mu = self.mu_embedding(x)  # [B, D]
        logvar = self.logvar_embedding(x)  # [B, D]

        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-4.0, max=2.0)
        std = torch.exp(0.5 * logvar)

        # Clamp std to reasonable range
        std = torch.clamp(std, min=self.min_std, max=self.max_std)

        if training:
            # Reparameterization trick: z = mu + std * epsilon
            eps = torch.randn_like(mu)
            z = mu + std * eps
        else:
            # At inference, use mean for deterministic output
            z = mu

        if return_params:
            return z, mu, std
        return z

    def kl_divergence(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from N(mu, sigma^2) to N(0, 1).
        Used as regularization to prevent embedding collapse.

        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        logvar = 2 * torch.log(std + 1e-8)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - std.pow(2), dim=-1)
        return kl.mean()


class TemporalConditioner(nn.Module):
    """
    Combines discrete hour embedding with continuous time encoding.
    Uses variational embedding for the discrete part.
    """

    def __init__(self, embed_dim: int = 64, min_std: float = 0.05, max_std: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim

        # Variational hour embedding (0-23)
        self.hour_embedding = VariationalEmbedding(
            num_embeddings=24,
            embedding_dim=embed_dim // 2,
            min_std=min_std,
            max_std=max_std
        )

        # Continuous hour encoding (sin/cos)
        self.continuous_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2)
        )

        # Combine discrete + continuous
        self.combine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(
        self,
        hour: torch.Tensor,
        training: bool = True,
        return_params: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            hour: Hour indices [B] (0-23)
            training: If True, sample from variational embedding
            return_params: If True, return KL params

        Returns:
            Temporal embedding [B, embed_dim]
        """
        # Discrete variational embedding
        if return_params:
            disc_embed, mu, std = self.hour_embedding(hour, training=training, return_params=True)
        else:
            disc_embed = self.hour_embedding(hour, training=training)

        # Continuous encoding
        hour_float = hour.float()
        hour_sin = torch.sin(2 * math.pi * hour_float / 24)
        hour_cos = torch.cos(2 * math.pi * hour_float / 24)
        cont_input = torch.stack([hour_sin, hour_cos], dim=-1)
        cont_embed = self.continuous_encoder(cont_input)

        # Combine
        combined = torch.cat([disc_embed, cont_embed], dim=-1)
        output = self.combine(combined)

        if return_params:
            return output, mu, std
        return output


class APConditioner(nn.Module):
    """
    Variational AP embedding that learns typical load patterns per AP.
    """

    def __init__(
        self,
        num_aps: int = 11,
        embed_dim: int = 64,
        min_std: float = 0.05,
        max_std: float = 0.3,
        ap_load_prior: Optional[list] = None
    ):
        super().__init__()

        self.ap_embedding = VariationalEmbedding(
            num_embeddings=num_aps,
            embedding_dim=embed_dim,
            min_std=min_std,
            max_std=max_std
        )

        # Initialize with prior knowledge about AP load ranking if provided
        if ap_load_prior is not None:
            with torch.no_grad():
                max_load = max(ap_load_prior)
                for ap_id, load in enumerate(ap_load_prior):
                    # Scale mean embedding by relative load
                    scale = load / max_load
                    self.ap_embedding.mu_embedding.weight[ap_id] *= scale

    def forward(
        self,
        ap_id: torch.Tensor,
        training: bool = True,
        return_params: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            ap_id: AP indices [B] (0-10)
            training: If True, sample from variational embedding
            return_params: If True, return KL params
        """
        return self.ap_embedding(ap_id, training=training, return_params=return_params)


class ConditionalTransformerBlock(nn.Module):
    """
    Transformer block with cross-attention for conditioning injection.
    Uses lighter conditioning than V3's dense cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_scale: float = 0.3
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.self_norm = nn.LayerNorm(hidden_dim)

        # Conditioning projection (lighter than V3)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_norm = nn.LayerNorm(hidden_dim)

        # Learnable conditioning scale (initialized small)
        self.cond_scale = nn.Parameter(torch.tensor(cond_scale))

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, hidden_dim] noisy features
            condition: [B, cond_dim] conditioning vector (or None for unconditional)
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.self_norm(x + attn_out)

        # Conditioning injection (additive, scaled)
        if condition is not None:
            cond = self.cond_proj(condition)  # [B, hidden_dim]
            cond = cond * self.cond_scale
            x = self.cond_norm(x + cond.unsqueeze(1))  # Add to all positions

        # Feedforward
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        return x


class PhysDiffNetV4(nn.Module):
    """
    V4: Variational conditioning with diversity preservation.

    Key differences from V3:
    1. Variational embeddings (mu, sigma) instead of fixed vectors
    2. Lighter conditioning injection (scaled additive, not cross-attention)
    3. KL regularization on embeddings
    4. Higher CFG drop probability (0.3 vs 0.1)
    5. Built on V2's proven architecture

    Architecture:
    - Pretrained TS2Vec encoder (frozen)
    - Variational hour + AP conditioners
    - Transformer denoiser with light conditioning
    - Cosine noise schedule
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        model_config = config.get('model', {})

        # Core dimensions
        self.input_dim = model_config.get('input_dim', 29)
        self.hidden_dim = model_config.get('hidden_dim', 256)
        self.latent_dim = model_config.get('latent_dim', 128)
        self.cond_dim = model_config.get('cond_dim', 64)
        self.num_aps = config.get('data', {}).get('num_aps', 11)

        # V4-specific parameters
        self.embedding_min_std = model_config.get('embedding_min_std', 0.05)
        self.embedding_max_std = model_config.get('embedding_max_std', 0.3)
        self.cfg_drop_prob = model_config.get('cfg_drop_prob', 0.3)
        self.cond_scale_init = model_config.get('cond_scale_init', 0.3)

        # Diffusion parameters
        diffusion_config = config.get('diffusion', {})
        self.num_timesteps = diffusion_config.get('num_timesteps', 1000)

        # ================== Encoder (from V2, frozen) ==================
        self._init_encoder(config)

        # ================== Diffusion Time Embedding ==================
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        # ================== V4: Variational Conditioners ==================
        self.temporal_cond = TemporalConditioner(
            embed_dim=self.cond_dim,
            min_std=self.embedding_min_std,
            max_std=self.embedding_max_std
        )

        self.ap_cond = APConditioner(
            num_aps=self.num_aps,
            embed_dim=self.cond_dim,
            min_std=self.embedding_min_std,
            max_std=self.embedding_max_std
        )

        # Combined conditioning dimension
        self.total_cond_dim = self.cond_dim * 2  # hour + AP

        # Condition combiner
        self.cond_combiner = nn.Sequential(
            nn.Linear(self.total_cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # ================== Denoiser ==================
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        num_layers = model_config.get('num_layers', 4)
        num_heads = model_config.get('num_heads', 4)
        dropout = model_config.get('dropout', 0.1)

        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(
                hidden_dim=self.hidden_dim,
                cond_dim=self.hidden_dim,  # After combining
                num_heads=num_heads,
                dropout=dropout,
                cond_scale=self.cond_scale_init
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )

        # ================== Noise Schedule (Cosine) ==================
        self.register_buffer('betas', self._cosine_beta_schedule(self.num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                           torch.sqrt(1 - self.alphas_cumprod))

        # Null condition for CFG
        self.register_buffer('null_cond', torch.zeros(self.hidden_dim))

    def _init_encoder(self, config: Dict) -> None:
        """Initialize encoder wrapper for compatibility with existing code."""
        # Simple passthrough - we'll use data directly without encoder in V4
        # The encoder representation comes from the conditioning instead
        self.use_encoder = config.get('model', {}).get('use_encoder', False)

        if self.use_encoder:
            from .encoder import TS2VecWrapper
            encoder_config = config.get('model', {}).get('encoder', {})
            self.encoder = TS2VecWrapper(
                input_dims=self.input_dim,
                output_dims=encoder_config.get('output_dims', 320),
                hidden_dims=encoder_config.get('hidden_dims', 64),
                depth=encoder_config.get('depth', 10),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            # Freeze encoder
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule from improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def get_condition(
        self,
        hour: Optional[torch.Tensor],
        ap_id: Optional[torch.Tensor],
        training: bool = True,
        drop_prob: float = 0.0
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get combined conditioning vector with optional dropout for CFG.

        Args:
            hour: Hour indices [B]
            ap_id: AP indices [B]
            training: Training mode
            drop_prob: Probability to drop conditioning (for CFG)

        Returns:
            condition: Combined condition [B, hidden_dim] or None
            aux_outputs: Dict with KL terms
        """
        aux_outputs = {}

        if hour is None or ap_id is None:
            return None, aux_outputs

        batch_size = hour.shape[0]
        device = hour.device

        # CFG: Randomly drop conditioning during training
        if training and drop_prob > 0:
            drop_mask = torch.rand(batch_size, device=device) < drop_prob
            if drop_mask.all():
                return None, aux_outputs
        else:
            drop_mask = None

        # Get variational embeddings with KL terms
        hour_emb, hour_mu, hour_std = self.temporal_cond(
            hour, training=training, return_params=True
        )
        ap_emb, ap_mu, ap_std = self.ap_cond(
            ap_id, training=training, return_params=True
        )

        # Store KL divergence for loss
        aux_outputs['kl_hour'] = self.temporal_cond.hour_embedding.kl_divergence(hour_mu, hour_std)
        aux_outputs['kl_ap'] = self.ap_cond.ap_embedding.kl_divergence(ap_mu, ap_std)

        # Combine embeddings
        combined = torch.cat([hour_emb, ap_emb], dim=-1)  # [B, 2*cond_dim]
        condition = self.cond_combiner(combined)  # [B, hidden_dim]

        # Apply drop mask for CFG training
        if drop_mask is not None and drop_mask.any():
            null_cond = self.null_cond.expand(batch_size, -1)
            condition = torch.where(
                drop_mask.unsqueeze(-1),
                null_cond,
                condition
            )

        return condition, aux_outputs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        hour: Optional[torch.Tensor] = None,
        ap_id: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict noise given noisy input, timestep, and optional conditions.

        Args:
            x: Noisy input [B, T, input_dim]
            t: Diffusion timestep [B]
            hour: Hour indices [B] (optional)
            ap_id: AP indices [B] (optional)
            training: Training mode flag

        Returns:
            noise_pred: Predicted noise [B, T, input_dim]
            aux_outputs: Dictionary with KL terms for loss computation
        """
        # Time embedding
        t_emb = self.time_embed(t)  # [B, hidden_dim]

        # Project input
        h = self.input_proj(x)  # [B, T, hidden_dim]

        # Add time embedding to all positions
        h = h + t_emb.unsqueeze(1)

        # Get conditioning with CFG dropout during training
        drop_prob = self.cfg_drop_prob if training else 0.0
        condition, aux_outputs = self.get_condition(
            hour, ap_id, training=training, drop_prob=drop_prob
        )

        # Apply transformer blocks with conditioning
        for block in self.blocks:
            h = block(h, condition)

        # Output projection
        noise_pred = self.output_proj(h)

        return noise_pred, aux_outputs

    def compute_loss(
        self,
        x: torch.Tensor,
        hour: torch.Tensor,
        ap_id: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute diffusion training loss.

        Args:
            x: Clean input [B, T, input_dim]
            hour: Hour indices [B]
            ap_id: AP indices [B]
            noise: Optional pre-generated noise

        Returns:
            loss: Diffusion loss
            aux_outputs: Dictionary with all loss components
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Sample noise
        if noise is None:
            noise = torch.randn_like(x)

        # Forward diffusion: add noise
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x_noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise

        # Predict noise
        noise_pred, aux_outputs = self.forward(x_noisy, t, hour, ap_id, training=True)

        # MSE loss
        diffusion_loss = F.mse_loss(noise_pred, noise)
        aux_outputs['diffusion_loss'] = diffusion_loss

        return diffusion_loss, aux_outputs

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        seq_length: int,
        hour: torch.Tensor,
        ap_id: torch.Tensor,
        num_inference_steps: int = 50,
        cfg_scale: float = 7.5,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate samples with classifier-free guidance.

        Args:
            num_samples: Number of samples to generate
            seq_length: Sequence length
            hour: Hour conditioning [num_samples]
            ap_id: AP conditioning [num_samples]
            num_inference_steps: Number of denoising steps
            cfg_scale: Guidance scale (higher = stronger conditioning)
            device: Device to generate on

        Returns:
            Generated samples [num_samples, seq_length, input_dim]
        """
        self.eval()

        # Start from pure noise
        x = torch.randn(num_samples, seq_length, self.input_dim, device=device)

        # Compute step indices for inference (uniform spacing)
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_ratio))[::-1]

        # Get conditioning vectors (no dropout during inference)
        cond, _ = self.get_condition(hour, ap_id, training=False, drop_prob=0.0)
        uncond = self.null_cond.expand(num_samples, -1)

        # Reverse diffusion with CFG
        for i, t in enumerate(timesteps):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # Predict noise (conditional)
            noise_cond, _ = self.forward(x, t_batch, hour, ap_id, training=False)

            # Predict noise (unconditional) - pass None for conditions
            noise_uncond, _ = self.forward(x, t_batch, None, None, training=False)

            # CFG: Amplify conditional direction
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # DDPM reverse step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            # Add noise for non-final steps
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / alpha.sqrt()) * (
                x - (beta / (1 - alpha_cumprod).sqrt()) * noise_pred
            ) + beta.sqrt() * noise

        return x

    def generate_scenario(
        self,
        num_windows: int,
        hours: list,
        num_aps: int = 11,
        seq_length: int = 360,
        cfg_scale: float = 7.5,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate a full scenario with all APs and specified hours.

        Args:
            num_windows: Number of windows to generate
            hours: List of hours for each window (or single hour to repeat)
            num_aps: Number of APs
            seq_length: Timesteps per window
            cfg_scale: CFG scale
            device: Device

        Returns:
            [num_windows, seq_length, num_aps, input_dim] tensor
        """
        self.eval()

        if isinstance(hours, int):
            hours = [hours] * num_windows

        all_windows = []

        for window_idx in range(num_windows):
            hour = hours[window_idx % len(hours)]

            # Generate all APs for this window
            hour_tensor = torch.full((num_aps,), hour, device=device, dtype=torch.long)
            ap_ids = torch.arange(num_aps, device=device)

            # Generate
            generated = self.generate(
                num_samples=num_aps,
                seq_length=seq_length,
                hour=hour_tensor,
                ap_id=ap_ids,
                cfg_scale=cfg_scale,
                device=device
            )  # [num_aps, seq_length, input_dim]

            # Reshape: [num_aps, seq_len, features] -> [seq_len, num_aps, features]
            generated = generated.permute(1, 0, 2)
            all_windows.append(generated)

        # Stack: [num_windows, seq_length, num_aps, features]
        return torch.stack(all_windows)

    def get_num_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_physdiff_v4(config: Dict) -> PhysDiffNetV4:
    """Factory function to create PhysDiffNetV4 from config."""
    return PhysDiffNetV4(config)
