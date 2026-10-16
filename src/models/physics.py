"""Physics constraint losses for WiFi network data generation.

This module implements physics-informed loss functions that ensure
generated synthetic data respects real-world network constraints:
- Capacity limits (max devices per AP)
- Interference patterns (channel utilization limits)
- Spatial correlations (nearby APs should correlate)
- Temporal smoothness (rate of change limits)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstraints(nn.Module):
    """Combined physics constraint losses.

    Aggregates multiple physics constraints with configurable weights.

    Attributes:
        capacity_loss: Capacity constraint loss
        interference_loss: Interference constraint loss
        spatial_loss: Spatial correlation loss
        temporal_loss: Temporal smoothness loss
    """

    def __init__(
        self,
        config: Dict,
        ap_distances: Optional[torch.Tensor] = None,
    ):
        """Initialize physics constraints.

        Args:
            config: Configuration dictionary with physics settings
            ap_distances: Optional matrix of AP distances (num_aps, num_aps)
        """
        super().__init__()

        self.config = config
        physics_config = config.get("physics", {})

        # Initialize individual losses
        cap_config = physics_config.get("capacity", {})
        if cap_config.get("enabled", True):
            self.capacity_loss = CapacityLoss(
                max_devices=cap_config.get("max_devices_per_ap", 512),
                weight=cap_config.get("weight", 1.0),
            )
        else:
            self.capacity_loss = None

        int_config = physics_config.get("interference", {})
        if int_config.get("enabled", True):
            self.interference_loss = InterferenceLoss(
                max_utilization=int_config.get("max_channel_utilization", 1.0),
                weight=int_config.get("weight", 0.5),
            )
        else:
            self.interference_loss = None

        spatial_config = physics_config.get("spatial", {})
        if spatial_config.get("enabled", True) and ap_distances is not None:
            self.spatial_loss = SpatialCorrelationLoss(
                ap_distances=ap_distances,
                correlation_radius=spatial_config.get("correlation_radius_m", 100.0),
                weight=spatial_config.get("weight", 0.3),
            )
        else:
            self.spatial_loss = None

        temp_config = physics_config.get("temporal", {})
        if temp_config.get("enabled", True):
            self.temporal_loss = TemporalSmoothnessLoss(
                max_rate=temp_config.get("max_rate_of_change", 0.5),
                weight=temp_config.get("weight", 0.2),
            )
        else:
            self.temporal_loss = None

    def forward(
        self,
        generated: torch.Tensor,
        real: Optional[torch.Tensor] = None,
        feature_indices: Optional[Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined physics loss.

        Args:
            generated: Generated data of shape (batch, seq, features)
            real: Optional real data for reference
            feature_indices: Mapping from feature names to indices

        Returns:
            Tuple of (total_loss, dict of individual losses)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=generated.device)

        # Default feature indices if not provided
        if feature_indices is None:
            feature_indices = {
                "connected_devices": 0,
                "channel_utilization": 3,
            }

        # Capacity loss
        if self.capacity_loss is not None:
            device_idx = feature_indices.get("connected_devices", 0)
            devices = generated[:, :, device_idx]
            cap_loss = self.capacity_loss(devices)
            losses["capacity"] = cap_loss
            total_loss = total_loss + cap_loss

        # Interference loss
        if self.interference_loss is not None:
            util_idx = feature_indices.get("channel_utilization", 3)
            utilization = generated[:, :, util_idx]
            int_loss = self.interference_loss(utilization)
            losses["interference"] = int_loss
            total_loss = total_loss + int_loss

        # Spatial correlation loss
        if self.spatial_loss is not None:
            spatial_loss = self.spatial_loss(generated, real)
            losses["spatial"] = spatial_loss
            total_loss = total_loss + spatial_loss

        # Temporal smoothness loss
        if self.temporal_loss is not None:
            temp_loss = self.temporal_loss(generated)
            losses["temporal"] = temp_loss
            total_loss = total_loss + temp_loss

        return total_loss, losses


class CapacityLoss(nn.Module):
    """Penalizes generated data that exceeds AP capacity limits.

    Ensures the number of connected devices per AP stays within
    physical limits (typically 512 for enterprise APs).
    """

    def __init__(
        self,
        max_devices: int = 512,
        weight: float = 1.0,
        soft_margin: float = 0.9,
    ):
        """Initialize capacity loss.

        Args:
            max_devices: Maximum allowed devices per AP
            weight: Loss weight
            soft_margin: Start penalizing at this fraction of max
        """
        super().__init__()
        self.max_devices = max_devices
        self.weight = weight
        self.soft_threshold = soft_margin * max_devices

    def forward(self, devices: torch.Tensor) -> torch.Tensor:
        """Compute capacity violation loss.

        Args:
            devices: Device counts of shape (batch, seq) or (batch, seq, 1)

        Returns:
            Scalar loss value
        """
        if devices.dim() == 3:
            devices = devices.squeeze(-1)

        # Hard violation: exceeds max
        hard_violations = F.relu(devices - self.max_devices)
        hard_loss = (hard_violations ** 2).mean()

        # Soft violation: approaching max (quadratic ramp)
        soft_violations = F.relu(devices - self.soft_threshold)
        soft_loss = (soft_violations ** 2).mean() * 0.1

        return self.weight * (hard_loss + soft_loss)


class InterferenceLoss(nn.Module):
    """Penalizes unrealistic channel utilization patterns.

    Ensures channel utilization stays within physical limits
    and follows realistic patterns (e.g., not sustained at 100%).
    """

    def __init__(
        self,
        max_utilization: float = 1.0,
        weight: float = 0.5,
        sustained_threshold: float = 0.95,
        sustained_window: int = 3,
    ):
        """Initialize interference loss.

        Args:
            max_utilization: Maximum channel utilization (0-1)
            weight: Loss weight
            sustained_threshold: Threshold for sustained high usage
            sustained_window: Window for detecting sustained usage
        """
        super().__init__()
        self.max_utilization = max_utilization
        self.weight = weight
        self.sustained_threshold = sustained_threshold
        self.sustained_window = sustained_window

    def forward(self, utilization: torch.Tensor) -> torch.Tensor:
        """Compute interference loss.

        Args:
            utilization: Channel utilization of shape (batch, seq) or (batch, seq, 1)

        Returns:
            Scalar loss value
        """
        if utilization.dim() == 3:
            utilization = utilization.squeeze(-1)

        # Exceeding 100% utilization is physically impossible
        over_max = F.relu(utilization - self.max_utilization)
        over_max_loss = (over_max ** 2).mean()

        # Negative utilization is impossible
        under_min = F.relu(-utilization)
        under_min_loss = (under_min ** 2).mean()

        # Penalize sustained near-saturation (unrealistic)
        if utilization.shape[1] >= self.sustained_window:
            # Use average pooling to detect sustained high usage
            kernel_size = self.sustained_window
            avg_util = F.avg_pool1d(
                utilization.unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ).squeeze(1)

            sustained_violations = F.relu(avg_util - self.sustained_threshold)
            sustained_loss = (sustained_violations ** 2).mean() * 0.5
        else:
            sustained_loss = torch.tensor(0.0, device=utilization.device)

        return self.weight * (over_max_loss + under_min_loss + sustained_loss)


class SpatialCorrelationLoss(nn.Module):
    """Ensures nearby APs have correlated behavior.

    APs that are physically close should show similar traffic
    patterns due to shared user populations.
    """

    def __init__(
        self,
        ap_distances: torch.Tensor,
        correlation_radius: float = 100.0,
        weight: float = 0.3,
    ):
        """Initialize spatial correlation loss.

        Args:
            ap_distances: Distance matrix of shape (num_aps, num_aps)
            correlation_radius: Radius within which APs should correlate (meters)
            weight: Loss weight
        """
        super().__init__()
        self.register_buffer("ap_distances", ap_distances)
        self.correlation_radius = correlation_radius
        self.weight = weight

        # Create neighbor mask (APs within radius)
        self.register_buffer(
            "neighbor_mask",
            (ap_distances > 0) & (ap_distances < correlation_radius)
        )

        # Expected correlation decreases with distance
        # Using exponential decay
        decay_rate = 0.01  # Correlation decay rate
        self.register_buffer(
            "expected_correlation",
            torch.exp(-decay_rate * ap_distances) * self.neighbor_mask.float()
        )

    def forward(
        self,
        generated: torch.Tensor,
        real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute spatial correlation loss.

        Args:
            generated: Generated data of shape (batch, num_aps, seq, features)
                      or (num_aps, seq, features) for single batch
            real: Optional real data for reference

        Returns:
            Scalar loss value
        """
        # Handle different input shapes
        if generated.dim() == 3:
            generated = generated.unsqueeze(0)

        batch_size, num_aps, seq_len, num_features = generated.shape

        # Check if we have enough APs
        if num_aps != self.ap_distances.shape[0]:
            # Skip if dimensions don't match
            return torch.tensor(0.0, device=generated.device)

        # Compute correlation matrix for generated data
        # Use the first feature (typically connected_devices)
        gen_feature = generated[:, :, :, 0]  # (batch, num_aps, seq)

        # Normalize
        gen_normalized = gen_feature - gen_feature.mean(dim=-1, keepdim=True)
        gen_std = gen_normalized.std(dim=-1, keepdim=True).clamp(min=1e-6)
        gen_normalized = gen_normalized / gen_std

        # Compute pairwise correlations
        # (batch, num_aps, num_aps)
        correlations = torch.bmm(
            gen_normalized,
            gen_normalized.transpose(1, 2)
        ) / seq_len

        # Compare with expected correlations
        expected = self.expected_correlation.unsqueeze(0)  # (1, num_aps, num_aps)
        diff = (correlations - expected) * self.neighbor_mask.unsqueeze(0).float()

        loss = (diff ** 2).sum() / (self.neighbor_mask.sum() * batch_size + 1e-6)

        return self.weight * loss


class TemporalSmoothnessLoss(nn.Module):
    """Penalizes unrealistic rapid changes in metrics.

    Network metrics should change smoothly over time; sudden
    jumps are physically unrealistic.
    """

    def __init__(
        self,
        max_rate: float = 0.5,
        weight: float = 0.2,
    ):
        """Initialize temporal smoothness loss.

        Args:
            max_rate: Maximum allowed rate of change (per timestep, normalized)
            weight: Loss weight
        """
        super().__init__()
        self.max_rate = max_rate
        self.weight = weight

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        """Compute temporal smoothness loss.

        Args:
            generated: Generated data of shape (batch, seq, features)

        Returns:
            Scalar loss value
        """
        # Compute differences between consecutive timesteps
        diffs = torch.diff(generated, dim=1)

        # Penalize changes exceeding max_rate
        violations = F.relu(torch.abs(diffs) - self.max_rate)
        loss = (violations ** 2).mean()

        return self.weight * loss
