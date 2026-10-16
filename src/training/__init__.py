"""Training utilities for PhysDiff-Net."""

from .trainer import Trainer
from .losses import CombinedLoss, DiffusionLoss, PhysDiffLoss
from .losses_v4 import (
    V4CombinedLoss,
    VarianceMatchingLoss,
    TemporalContrastiveLoss,
    GradientOrthogonalizer,
    ImprovedPhysicsLossV4,
    compute_real_feature_stds,
)
from .losses_v5 import V5LossFunction, EvaluationMetrics

__all__ = [
    "Trainer",
    "CombinedLoss",
    "DiffusionLoss",
    "PhysDiffLoss",
    "V4CombinedLoss",
    "VarianceMatchingLoss",
    "TemporalContrastiveLoss",
    "GradientOrthogonalizer",
    "ImprovedPhysicsLossV4",
    "compute_real_feature_stds",
    # V5 exports
    "V5LossFunction",
    "EvaluationMetrics",
]
