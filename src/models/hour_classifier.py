"""
Auxiliary Hour Classifier for V5

Key insight from V4 failure:
- No explicit gradient signal forcing hour-awareness
- Network could ignore hour conditioning with impunity

V5 Solution:
- Add auxiliary hour prediction head
- Cross-entropy loss creates direct gradient path
- Penalizes model for generating hour-inconsistent patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AuxiliaryHourClassifier(nn.Module):
    """
    Auxiliary head to predict input hour from generated/denoised features.

    This creates a direct gradient path that penalizes the model when
    generated patterns don't reflect the conditioning hour.

    Architecture: Pool -> LayerNorm -> MLP -> Softmax(24)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_hours: int = 24,
        dropout: float = 0.1,
        pool_method: str = 'mean'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hours = num_hours
        self.pool_method = pool_method

        # Multi-layer classifier for robust hour prediction
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_hours)
        )

        # Optional attention pooling
        if pool_method == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict hour from hidden representations.

        Args:
            h: Hidden states [B, T, hidden_dim]

        Returns:
            hour_logits: Hour prediction logits [B, num_hours]
        """
        # Pool over sequence dimension
        if self.pool_method == 'mean':
            pooled = h.mean(dim=1)  # [B, hidden_dim]
        elif self.pool_method == 'max':
            pooled = h.max(dim=1)[0]
        elif self.pool_method == 'attention':
            attn_weights = self.attention_pool(h)  # [B, T, 1]
            pooled = (h * attn_weights).sum(dim=1)  # [B, hidden_dim]
        elif self.pool_method == 'first_last':
            pooled = (h[:, 0] + h[:, -1]) / 2
        else:
            pooled = h.mean(dim=1)

        # Classify
        logits = self.classifier(pooled)
        return logits


class MultiScaleHourClassifier(nn.Module):
    """
    Multi-scale hour classifier that looks at different temporal scales.

    Inspired by the observation that hourly patterns manifest at different
    temporal granularities (e.g., morning rush vs gradual afternoon decline).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_hours: int = 24,
        scales: tuple = (1, 4, 16),  # Different pooling window sizes
        dropout: float = 0.1
    ):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)

        # Classifier per scale
        self.scale_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in scales
        ])

        # Combined classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4 * len(scales), hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_hours)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale hour prediction.

        Args:
            h: Hidden states [B, T, hidden_dim]

        Returns:
            hour_logits: Hour prediction logits [B, num_hours]
        """
        B, T, D = h.shape
        scale_features = []

        for scale, classifier in zip(self.scales, self.scale_classifiers):
            if scale == 1:
                # Global mean
                pooled = h.mean(dim=1)
            else:
                # Local pooling with stride
                if T >= scale:
                    # Reshape for strided pooling
                    n_windows = T // scale
                    h_trimmed = h[:, :n_windows * scale, :]
                    h_reshaped = h_trimmed.view(B, n_windows, scale, D)
                    pooled = h_reshaped.mean(dim=2).mean(dim=1)  # [B, D]
                else:
                    pooled = h.mean(dim=1)

            scale_feat = classifier(pooled)  # [B, D//4]
            scale_features.append(scale_feat)

        # Concatenate and classify
        combined = torch.cat(scale_features, dim=-1)  # [B, D//4 * num_scales]
        logits = self.final_classifier(combined)

        return logits


class ContrastiveHourClassifier(nn.Module):
    """
    Hour classifier using contrastive learning approach.

    Instead of direct classification, learns hour-specific prototypes
    and classifies based on similarity.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_hours: int = 24,
        prototype_dim: int = 64,
        temperature: float = 0.1
    ):
        super().__init__()
        self.temperature = temperature
        self.num_hours = num_hours

        # Projection to prototype space
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, prototype_dim)
        )

        # Learnable hour prototypes
        self.hour_prototypes = nn.Parameter(
            torch.randn(num_hours, prototype_dim) * 0.02
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Classify hour via prototype similarity.

        Args:
            h: Hidden states [B, T, hidden_dim]

        Returns:
            hour_logits: Hour prediction logits [B, num_hours]
        """
        # Pool and project
        pooled = h.mean(dim=1)  # [B, hidden_dim]
        projected = self.projector(pooled)  # [B, prototype_dim]

        # Normalize
        projected = F.normalize(projected, p=2, dim=-1)
        prototypes = F.normalize(self.hour_prototypes, p=2, dim=-1)

        # Cosine similarity as logits
        logits = torch.matmul(projected, prototypes.t()) / self.temperature

        return logits
