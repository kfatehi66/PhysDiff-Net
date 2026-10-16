"""
V5 Hour Conditioning Module: Concatenation-based Approach

Key insight from V4 failure analysis:
- Additive conditioning (hour_emb + x) was too weak
- Hour information was washed out by diffusion gradients
- Network could ignore hour conditioning entirely

V5 Solution:
- Concatenate hour embedding as additional feature channels
- Forces network to explicitly process hour information
- Creates stronger gradient signal for hour awareness
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConcatenationHourConditioning(nn.Module):
    """
    V5 Hour Conditioning: Concatenation instead of Addition

    This forces the network to explicitly process hour information
    as feature channels rather than treating it as a bias shift.

    The concatenated hour embedding becomes part of the input tensor,
    creating a direct dependency that cannot be ignored.
    """

    def __init__(
        self,
        num_hours: int = 24,
        hour_dim: int = 64,
        variational: bool = True,
        min_std: float = 0.05,
        max_std: float = 0.3
    ):
        super().__init__()
        self.num_hours = num_hours
        self.hour_dim = hour_dim
        self.variational = variational
        self.min_std = min_std
        self.max_std = max_std

        # Base hour embedding
        self.hour_embed = nn.Embedding(num_hours, hour_dim)
        nn.init.normal_(self.hour_embed.weight, mean=0.0, std=0.1)

        if variational:
            # Variational parameters for sampling diversity
            self.hour_mu = nn.Linear(hour_dim, hour_dim)
            self.hour_logvar = nn.Linear(hour_dim, hour_dim)
            nn.init.xavier_uniform_(self.hour_mu.weight)
            nn.init.zeros_(self.hour_mu.bias)
            nn.init.xavier_uniform_(self.hour_logvar.weight)
            nn.init.constant_(self.hour_logvar.bias, -2.0)  # Start with small variance

    def forward(
        self,
        hours: torch.Tensor,
        seq_len: int = 1,
        return_params: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get hour embedding for concatenation.

        Args:
            hours: Hour indices [B]
            seq_len: Sequence length to broadcast to
            return_params: If True, also return (mu, std) for KL loss

        Returns:
            hour_emb: Hour embedding [B, seq_len, hour_dim]
            (optional) params: (mu, std) tensors for KL computation
        """
        # Get base embedding
        h = self.hour_embed(hours)  # [B, hour_dim]

        mu, std = None, None

        if self.variational:
            mu = self.hour_mu(h)
            logvar = self.hour_logvar(h)

            # Clamp logvar for stability
            logvar = torch.clamp(logvar, min=-4.0, max=2.0)
            std = torch.exp(0.5 * logvar)
            std = torch.clamp(std, min=self.min_std, max=self.max_std)

            if self.training:
                # Reparameterization trick
                eps = torch.randn_like(std)
                h = mu + eps * std
            else:
                # Use mean during inference
                h = mu

        # Broadcast to sequence length: [B, hour_dim] -> [B, seq_len, hour_dim]
        h = h.unsqueeze(1).expand(-1, seq_len, -1)

        if return_params and self.variational:
            return h, (mu, std)
        return h, None

    def get_kl_loss(self, hours: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence from N(0, 1) prior.

        This regularizes the variational hour embeddings to prevent
        embedding collapse while maintaining diversity.
        """
        if not self.variational:
            return torch.tensor(0.0, device=hours.device)

        h = self.hour_embed(hours)
        mu = self.hour_mu(h)
        logvar = self.hour_logvar(h)
        logvar = torch.clamp(logvar, min=-4.0, max=2.0)

        # KL(N(mu, sigma) || N(0, 1))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()


class TemporalHourEncoding(nn.Module):
    """
    Additional temporal encoding that captures cyclical hour patterns.

    Uses sinusoidal encoding similar to positional encoding in transformers,
    but for the 24-hour cycle.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Learnable projection for temporal encoding
        self.projection = nn.Linear(2, dim)  # sin + cos

    def forward(self, hours: torch.Tensor) -> torch.Tensor:
        """
        Generate temporal encoding for hours.

        Args:
            hours: Hour indices [B]

        Returns:
            encoding: Temporal encoding [B, dim]
        """
        # Normalize to [0, 2*pi]
        angle = (hours.float() / 24.0) * 2 * math.pi

        # Sin and cos for cyclical encoding
        sin_enc = torch.sin(angle).unsqueeze(-1)
        cos_enc = torch.cos(angle).unsqueeze(-1)

        # Concatenate and project
        temporal = torch.cat([sin_enc, cos_enc], dim=-1)
        return self.projection(temporal)


class HybridHourConditioning(nn.Module):
    """
    Combines learnable embeddings with cyclical temporal encoding.

    This provides both:
    - Learnable hour-specific features (via embedding)
    - Smooth cyclical transitions (via sinusoidal encoding)
    """

    def __init__(
        self,
        num_hours: int = 24,
        hour_dim: int = 64,
        variational: bool = True
    ):
        super().__init__()

        # Learnable embedding (2/3 of dim)
        embed_dim = (hour_dim * 2) // 3
        self.embed_conditioning = ConcatenationHourConditioning(
            num_hours=num_hours,
            hour_dim=embed_dim,
            variational=variational
        )

        # Temporal encoding (1/3 of dim)
        temporal_dim = hour_dim - embed_dim
        self.temporal_encoding = TemporalHourEncoding(temporal_dim)

        self.hour_dim = hour_dim

    def forward(
        self,
        hours: torch.Tensor,
        seq_len: int = 1
    ) -> Tuple[torch.Tensor, None]:
        """
        Get combined hour conditioning.
        """
        # Get learnable embedding
        embed, _ = self.embed_conditioning(hours, seq_len)  # [B, seq_len, embed_dim]

        # Get temporal encoding
        temporal = self.temporal_encoding(hours)  # [B, temporal_dim]
        temporal = temporal.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, temporal_dim]

        # Concatenate
        combined = torch.cat([embed, temporal], dim=-1)  # [B, seq_len, hour_dim]

        return combined, None

    def get_kl_loss(self, hours: torch.Tensor) -> torch.Tensor:
        return self.embed_conditioning.get_kl_loss(hours)
