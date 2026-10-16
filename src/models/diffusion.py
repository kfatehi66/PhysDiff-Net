"""Conditional diffusion model for time series generation.

This module implements a conditional diffusion model that generates
synthetic WiFi network time series conditioned on scenario parameters.
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class NoiseScheduler:
    """Noise scheduling for diffusion process.

    Handles the forward (noising) and reverse (denoising) diffusion
    process with configurable schedules.

    Attributes:
        num_timesteps: Number of diffusion steps
        beta_start: Starting noise level
        beta_end: Ending noise level
        schedule: Type of schedule ('linear', 'cosine', 'quadratic')
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: str = "cuda",
    ):
        """Initialize the noise scheduler.

        Args:
            num_timesteps: Number of diffusion steps
            beta_start: Starting noise level
            beta_end: Ending noise level
            schedule: Schedule type
            device: Device for tensors
        """
        self.num_timesteps = num_timesteps
        self.device = device

        # Compute beta schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            betas = self._cosine_schedule(num_timesteps)
        elif schedule == "quadratic":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas.to(device)

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule as proposed in improved DDPM.

        Args:
            timesteps: Number of timesteps
            s: Small offset to prevent singularity

        Returns:
            Beta values for each timestep
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean data (forward diffusion).

        Args:
            x_0: Clean data of shape (batch, seq, features)
            t: Timestep indices of shape (batch,)
            noise: Optional pre-generated noise

        Returns:
            Tuple of (noisy data, noise used)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        while len(sqrt_alpha_cumprod.shape) < len(x_0.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

        return x_t, noise

    def get_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity for v-prediction parameterization.

        Args:
            x_0: Clean data
            noise: Noise
            t: Timestep indices

        Returns:
            Velocity
        """
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]

        while len(sqrt_alpha_cumprod.shape) < len(x_0.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        return sqrt_alpha_cumprod * noise - sqrt_one_minus_alpha_cumprod * x_0


class ConditionalDiffusion(nn.Module):
    """Conditional diffusion model for time series generation.

    Uses a U-Net-like architecture conditioned on scenario embeddings
    and optionally on TS2Vec representations.

    Attributes:
        input_dims: Number of input features
        hidden_dims: Hidden dimension
        condition_dims: Dimension of conditioning vector
        num_timesteps: Number of diffusion steps
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int = 256,
        condition_dims: int = 64,
        time_embed_dims: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize the conditional diffusion model.

        Args:
            input_dims: Number of input/output features
            hidden_dims: Hidden layer dimension
            condition_dims: Dimension of condition embeddings
            time_embed_dims: Dimension of time embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.condition_dims = condition_dims
        self.time_embed_dims = time_embed_dims

        # Input projection
        self.input_proj = nn.Linear(input_dims, hidden_dims)

        # Time embedding (sinusoidal)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dims),
            nn.Linear(time_embed_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, hidden_dims),
        )

        # Condition embedding
        self.condition_proj = nn.Linear(condition_dims, hidden_dims)

        # Transformer-based denoising network
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dims=hidden_dims,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, input_dims),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting noise.

        Args:
            x: Noisy input of shape (batch, seq_length, input_dims)
            t: Diffusion timestep of shape (batch,)
            condition: Optional condition of shape (batch, condition_dims)

        Returns:
            Predicted noise of shape (batch, seq_length, input_dims)
        """
        batch_size, seq_length, _ = x.shape

        # Project input
        h = self.input_proj(x)  # (batch, seq, hidden)

        # Add time embedding (broadcast across sequence)
        t_emb = self.time_embed(t)  # (batch, hidden)
        h = h + t_emb.unsqueeze(1)

        # Add condition embedding if provided
        if condition is not None:
            c_emb = self.condition_proj(condition)  # (batch, hidden)
            h = h + c_emb.unsqueeze(1)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h)

        # Project to output
        output = self.output_proj(h)

        return output

    def sample(
        self,
        scheduler: NoiseScheduler,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Sample from the diffusion model.

        Args:
            scheduler: Noise scheduler instance
            shape: Shape of output (batch, seq_length, features)
            condition: Optional conditioning
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            device: Device for sampling

        Returns:
            Generated samples
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Compute step size
        step_size = scheduler.num_timesteps // num_inference_steps

        # Denoising loop
        for i in range(num_inference_steps - 1, -1, -1):
            t = torch.full((shape[0],), i * step_size, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.forward(x, t, condition)

            # Classifier-free guidance
            if guidance_scale != 1.0 and condition is not None:
                noise_uncond = self.forward(x, t, None)
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            # DDPM update step
            alpha = scheduler.alphas[t[0]]
            alpha_cumprod = scheduler.alphas_cumprod[t[0]]
            beta = scheduler.betas[t[0]]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Compute x_{t-1}
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            ) + torch.sqrt(beta) * noise

        return x


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        """Initialize embeddings.

        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for timesteps.

        Args:
            t: Timestep tensor of shape (batch,)

        Returns:
            Embeddings of shape (batch, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class TransformerBlock(nn.Module):
    """Transformer block for diffusion denoising."""

    def __init__(
        self,
        hidden_dims: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        """Initialize transformer block.

        Args:
            hidden_dims: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            mlp_ratio: MLP expansion ratio
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dims)
        self.attn = nn.MultiheadAttention(
            hidden_dims, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_dims)
        mlp_hidden = int(hidden_dims * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: Input tensor of shape (batch, seq, hidden)

        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x
