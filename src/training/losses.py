"""Loss functions for PhysDiff-Net training.

This module provides loss functions for training including:
- Diffusion loss (noise prediction MSE)
- Combined loss with physics constraints
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Standard diffusion training loss.

    Computes mean squared error between predicted and actual noise.
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize diffusion loss.

        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute diffusion loss.

        Args:
            noise_pred: Predicted noise of shape (batch, seq, features)
            noise_target: Target noise of same shape
            weights: Optional per-sample weights

        Returns:
            Loss value
        """
        loss = F.mse_loss(noise_pred, noise_target, reduction="none")

        if weights is not None:
            loss = loss * weights.view(-1, 1, 1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """Combined loss with diffusion and physics components.

    Aggregates multiple loss terms with configurable weights.
    """

    def __init__(
        self,
        diffusion_weight: float = 1.0,
        physics_weight: float = 0.1,
        reconstruction_weight: float = 0.0,
    ):
        """Initialize combined loss.

        Args:
            diffusion_weight: Weight for diffusion loss
            physics_weight: Weight for physics constraint losses
            reconstruction_weight: Weight for optional reconstruction loss
        """
        super().__init__()

        self.diffusion_weight = diffusion_weight
        self.physics_weight = physics_weight
        self.reconstruction_weight = reconstruction_weight

        self.diffusion_loss = DiffusionLoss()

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        physics_loss: Optional[torch.Tensor] = None,
        reconstruction_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss.

        Args:
            noise_pred: Predicted noise
            noise_target: Target noise
            physics_loss: Pre-computed physics loss (optional)
            reconstruction_loss: Reconstruction loss (optional)

        Returns:
            Tuple of (total_loss, dict of individual losses)
        """
        losses = {}

        # Diffusion loss
        diff_loss = self.diffusion_loss(noise_pred, noise_target)
        losses["diffusion"] = diff_loss
        total_loss = self.diffusion_weight * diff_loss

        # Physics loss
        if physics_loss is not None:
            losses["physics"] = physics_loss
            total_loss = total_loss + self.physics_weight * physics_loss

        # Reconstruction loss
        if reconstruction_loss is not None and self.reconstruction_weight > 0:
            losses["reconstruction"] = reconstruction_loss
            total_loss = total_loss + self.reconstruction_weight * reconstruction_loss

        losses["total"] = total_loss

        return total_loss, losses


class VelocityLoss(nn.Module):
    """Velocity prediction loss for v-prediction diffusion.

    Alternative parameterization that can improve training stability.
    """

    def __init__(self):
        """Initialize velocity loss."""
        super().__init__()

    def forward(
        self,
        velocity_pred: torch.Tensor,
        velocity_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity prediction loss.

        Args:
            velocity_pred: Predicted velocity
            velocity_target: Target velocity

        Returns:
            Loss value
        """
        return F.mse_loss(velocity_pred, velocity_target)


class PerceptualLoss(nn.Module):
    """Perceptual loss using learned representations.

    Compares samples in representation space rather than raw feature space.
    """

    def __init__(self, encoder: nn.Module):
        """Initialize perceptual loss.

        Args:
            encoder: Encoder to use for computing representations
        """
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            generated: Generated samples
            target: Target samples

        Returns:
            Loss value
        """
        # Get representations
        with torch.no_grad():
            target_repr = self.encoder(target)

        generated_repr = self.encoder(generated)

        return F.mse_loss(generated_repr, target_repr)


class PhysDiffLoss(nn.Module):
    """Physics-informed diffusion loss with lambda warmup and normalization.

    Key features:
    1. Diffusion loss (noise prediction MSE) - main objective
    2. Physics losses with warmup - gradually increase during training
    3. Loss normalization - brings all losses to similar scale

    This ensures:
    - Early training: Model learns good reconstructions (diffusion dominates)
    - Later training: Physics constraints gradually enforced
    - Balanced gradients: All losses normalized to similar scale
    """

    def __init__(
        self,
        lambda_capacity: float = 0.0001,
        lambda_interference: float = 0.0001,
        lambda_temporal: float = 0.0001,
        lambda_spatial: float = 0.0001,
        warmup_epochs: int = 50,
        max_lambda: float = 0.01,
        max_devices_per_ap: int = 512,
        ue_count_idx: int = 0,
        rssi_idx: int = 8,
    ):
        """Initialize PhysDiff loss.

        Args:
            lambda_capacity: Initial weight for capacity loss
            lambda_interference: Initial weight for interference loss
            lambda_temporal: Initial weight for temporal smoothness loss
            lambda_spatial: Initial weight for spatial correlation loss
            warmup_epochs: Number of epochs to warm up lambdas
            max_lambda: Maximum lambda value after warmup
            max_devices_per_ap: Maximum devices per AP (capacity constraint)
            ue_count_idx: Index of ue_count feature in tensor
            rssi_idx: Index of avg_rssi feature in tensor
        """
        super().__init__()

        self.base_lambdas = {
            'capacity': lambda_capacity,
            'interference': lambda_interference,
            'temporal': lambda_temporal,
            'spatial': lambda_spatial,
        }
        self.warmup_epochs = warmup_epochs
        self.max_lambda = max_lambda

        # Feature indices
        self.ue_count_idx = ue_count_idx
        self.rssi_idx = rssi_idx
        self.max_devices = max_devices_per_ap

        # For loss normalization (set after first batch)
        self.register_buffer('capacity_scale', torch.tensor(1.0))
        self.register_buffer('interference_scale', torch.tensor(1.0))
        self.register_buffer('temporal_scale', torch.tensor(1.0))
        self.register_buffer('spatial_scale', torch.tensor(1.0))
        self.initialized = False

    def get_lambda(self, name: str, epoch: int) -> float:
        """Get lambda value with warmup schedule.

        Args:
            name: Loss component name
            epoch: Current epoch

        Returns:
            Lambda value for this epoch
        """
        base = self.base_lambdas[name]
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = epoch / self.warmup_epochs
            return base + (self.max_lambda - base) * scale
        return self.max_lambda

    def capacity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Capacity constraint: penalize exceeding max devices per AP.

        Args:
            x: Predicted data of shape (batch, seq, aps, features) or (batch, seq, features)

        Returns:
            Capacity violation loss
        """
        # Get ue_count feature
        if x.dim() == 4:
            ue_count = x[..., self.ue_count_idx]  # (batch, seq, aps)
        else:
            ue_count = x[..., self.ue_count_idx]  # (batch, seq)

        # Penalize values exceeding max
        violations = F.relu(ue_count - self.max_devices)
        return violations.pow(2).mean()

    def interference_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Interference constraint: penalize poor signal quality.

        Args:
            x: Predicted data

        Returns:
            Interference loss
        """
        # Get RSSI feature
        if x.dim() == 4:
            rssi = x[..., self.rssi_idx]  # (batch, seq, aps)
        else:
            rssi = x[..., self.rssi_idx]  # (batch, seq)

        # Penalize very poor signal (below -85 dBm normalized)
        # In normalized space, this corresponds to very negative values
        poor_signal = F.relu(-rssi - 1.5)  # Threshold in normalized space
        return poor_signal.pow(2).mean()

    def temporal_smoothness_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Temporal smoothness: penalize large changes between timesteps.

        Args:
            x: Predicted data

        Returns:
            Temporal smoothness loss
        """
        # Compute differences between consecutive timesteps
        if x.dim() == 4:
            diffs = x[:, 1:, :, :] - x[:, :-1, :, :]  # (batch, seq-1, aps, features)
        else:
            diffs = x[:, 1:, :] - x[:, :-1, :]  # (batch, seq-1, features)

        return diffs.pow(2).mean()

    def spatial_correlation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial correlation: penalize large differences between adjacent APs.

        Args:
            x: Predicted data of shape (batch, seq, aps, features)

        Returns:
            Spatial correlation loss
        """
        if x.dim() != 4:
            # Cannot compute spatial loss without AP dimension
            return torch.tensor(0.0, device=x.device)

        # Compute variance across APs (should be correlated, not random)
        ap_var = x.var(dim=2)  # (batch, seq, features)
        return ap_var.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            pred: Predicted noise or reconstructed data
            target: Target noise or clean data
            epoch: Current training epoch

        Returns:
            Dictionary of losses including 'total'
        """
        # Diffusion reconstruction loss (main objective)
        L_diffusion = F.mse_loss(pred, target)

        # Physics losses
        L_capacity = self.capacity_loss(pred)
        L_interference = self.interference_loss(pred)
        L_temporal = self.temporal_smoothness_loss(pred)
        L_spatial = self.spatial_correlation_loss(pred)

        # Initialize loss scales on first batch
        if not self.initialized:
            self.capacity_scale = L_capacity.detach().clamp(min=1e-8)
            self.interference_scale = L_interference.detach().clamp(min=1e-8)
            self.temporal_scale = L_temporal.detach().clamp(min=1e-8)
            self.spatial_scale = L_spatial.detach().clamp(min=1e-8)
            self.initialized = True

        # Get current lambdas
        lam_cap = self.get_lambda('capacity', epoch)
        lam_int = self.get_lambda('interference', epoch)
        lam_temp = self.get_lambda('temporal', epoch)
        lam_spat = self.get_lambda('spatial', epoch)

        # Normalized and weighted physics losses
        L_physics = (
            lam_cap * (L_capacity / self.capacity_scale) +
            lam_int * (L_interference / self.interference_scale) +
            lam_temp * (L_temporal / self.temporal_scale) +
            lam_spat * (L_spatial / self.spatial_scale)
        )

        # Total loss
        total_loss = L_diffusion + L_physics

        return {
            'total': total_loss,
            'diffusion': L_diffusion,
            'physics': L_physics,
            'capacity': L_capacity,
            'interference': L_interference,
            'temporal': L_temporal,
            'spatial': L_spatial,
            'physics_weight': torch.tensor(lam_cap),
        }
