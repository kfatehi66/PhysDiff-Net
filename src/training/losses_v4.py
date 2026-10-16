"""
V4 Loss Functions for PhysDiff-Net

Key innovations:
1. Variance-Matching Loss: Prevents mode collapse by matching per-feature std
2. Temporal Contrastive Loss: Separates hour representations
3. Gradient Orthogonalization: Handles conflicting objectives
4. Physics Losses: From V2, preserved and enhanced

These losses address V3's variance collapse where all outputs became near-identical.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceMatchingLoss(nn.Module):
    """
    Match per-feature standard deviation to real data statistics.

    L_var = sum_f || std(x_hat_f) - sigma_real_f ||^2

    This prevents mode collapse by explicitly requiring diversity.
    If the generated data has low variance, this loss will be high.
    """

    def __init__(
        self,
        target_stds: Optional[torch.Tensor] = None,
        weight: float = 1.0,
        min_ratio: float = 0.5,
        max_ratio: float = 1.5
    ):
        """
        Args:
            target_stds: Target std for each feature [F]
            weight: Loss weight
            min_ratio: Minimum acceptable variance ratio (generated/real)
            max_ratio: Maximum acceptable variance ratio (generated/real)
        """
        super().__init__()
        if target_stds is not None:
            self.register_buffer('target_stds', target_stds)
        else:
            self.target_stds = None
        self.weight = weight
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def set_target_stds(self, stds: torch.Tensor):
        """Set target stds from real data (call once during init)."""
        self.register_buffer('target_stds', stds)

    def forward(
        self,
        generated: torch.Tensor,
        real: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            generated: Generated samples [B, T, F] or [B, F]
            real: Optional real samples for dynamic target computation

        Returns:
            loss: Variance matching loss
            stats: Dictionary with variance statistics
        """
        # Flatten to [N, F]
        if generated.dim() == 3:
            generated_flat = generated.reshape(-1, generated.shape[-1])
        else:
            generated_flat = generated

        # Compute per-feature std across batch
        gen_stds = generated_flat.std(dim=0)  # [F]

        # Get target stds
        if real is not None:
            if real.dim() == 3:
                real_flat = real.reshape(-1, real.shape[-1])
            else:
                real_flat = real
            target_stds = real_flat.std(dim=0)
        elif self.target_stds is not None:
            target_stds = self.target_stds
        else:
            # Default: assume normalized data with std=1
            target_stds = torch.ones_like(gen_stds)

        # Compute variance ratio
        # Use larger epsilon to avoid explosion when target_stds ≈ 0
        # Also mask out features with very low target std (they're effectively constant)
        min_target_std = 0.01
        valid_mask = target_stds > min_target_std

        if valid_mask.sum() == 0:
            # All features have near-zero variance, skip
            return torch.tensor(0.0, device=generated.device), {'mean_var_ratio': 0.0, 'skipped': True}

        # Only compute ratio on valid features
        safe_target = torch.clamp(target_stds, min=min_target_std)
        var_ratio = gen_stds / safe_target

        # Mask invalid features (don't penalize them)
        var_ratio_valid = var_ratio[valid_mask]
        mean_var_ratio = var_ratio_valid.mean().item()

        # Loss: penalize if ratio is outside acceptable range
        # We want min_ratio <= var_ratio <= max_ratio
        below_min = F.relu(self.min_ratio - var_ratio_valid)
        above_max = F.relu(var_ratio_valid - self.max_ratio)
        loss = (below_min.pow(2) + above_max.pow(2)).mean()

        stats = {
            'mean_var_ratio': mean_var_ratio,
            'min_var_ratio': var_ratio_valid.min().item(),
            'max_var_ratio': var_ratio_valid.max().item(),
            'valid_features': valid_mask.sum().item(),
        }

        return self.weight * loss, stats


class TemporalContrastiveLoss(nn.Module):
    """
    Contrastive loss to ensure different hours produce distinct representations.

    - Same hour samples should be similar (positive pairs)
    - Different hour samples should be different (negative pairs)

    This prevents the "mean-averaging" effect where all hours collapse to similar outputs.
    Uses InfoNCE-style contrastive learning.
    """

    def __init__(self, temperature: float = 0.1, weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(
        self,
        representations: torch.Tensor,
        hour_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            representations: Sample representations [B, D]
            hour_labels: Hour indices [B]

        Returns:
            loss: Contrastive loss
            stats: Dictionary with contrast statistics
        """
        batch_size = representations.shape[0]

        if batch_size < 4:
            return torch.tensor(0.0, device=representations.device), {'contrast_valid': False}

        # L2 normalize representations
        representations = F.normalize(representations, dim=1)

        # Compute similarity matrix [B, B]
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Create positive mask (same hour = positive pair)
        hour_labels = hour_labels.view(-1, 1)
        positive_mask = (hour_labels == hour_labels.T).float()

        # Remove self-similarity from positive mask
        positive_mask.fill_diagonal_(0)

        # Check if we have any positive pairs
        num_positives = positive_mask.sum(dim=1)
        has_positives = num_positives > 0

        if not has_positives.any():
            return torch.tensor(0.0, device=representations.device), {'contrast_valid': False}

        # For numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

        # Compute log-softmax denominator (exclude self)
        exp_sim = torch.exp(sim_matrix)
        exp_sim.fill_diagonal_(0)
        sum_exp = exp_sim.sum(dim=1, keepdim=True) + 1e-8

        # Log probabilities
        log_prob = sim_matrix - torch.log(sum_exp)

        # Compute mean log probability for positive pairs
        pos_log_prob = (log_prob * positive_mask).sum(dim=1)
        pos_log_prob = pos_log_prob / num_positives.clamp(min=1)

        # Loss: negative mean log probability (only for samples with positives)
        loss = -pos_log_prob[has_positives].mean()

        # Statistics
        avg_pos_sim = (sim_matrix * positive_mask).sum() / positive_mask.sum().clamp(min=1)
        neg_mask = 1 - positive_mask
        neg_mask.fill_diagonal_(0)
        avg_neg_sim = (sim_matrix * neg_mask).sum() / neg_mask.sum().clamp(min=1)

        stats = {
            'contrast_valid': True,
            'avg_pos_sim': avg_pos_sim.item() * self.temperature,
            'avg_neg_sim': avg_neg_sim.item() * self.temperature,
            'num_positives_avg': num_positives.float().mean().item(),
        }

        return self.weight * loss, stats


class GradientOrthogonalizer:
    """
    Orthogonalize conflicting gradients between loss terms.

    If grad(L_physics) opposes grad(L_diffusion), project L_physics gradient
    onto the plane normal to L_diffusion gradient.

    This prevents physics constraints from "washing out" data fidelity,
    and vice versa.

    Based on gradient surgery techniques from multi-task learning.
    """

    def __init__(self, conflict_threshold: float = 0.0):
        """
        Args:
            conflict_threshold: Cosine similarity threshold below which
                               we consider gradients conflicting (negative = opposing)
        """
        self.conflict_threshold = conflict_threshold
        self.stats = {'conflicts': 0, 'total': 0}

    def reset_stats(self):
        """Reset conflict statistics."""
        self.stats = {'conflicts': 0, 'total': 0}

    def orthogonalize(
        self,
        grad_primary: torch.Tensor,
        grad_secondary: torch.Tensor
    ) -> torch.Tensor:
        """
        Project secondary gradient to be orthogonal to primary if conflicting.

        Args:
            grad_primary: Gradient of primary loss (e.g., diffusion)
            grad_secondary: Gradient of secondary loss (e.g., physics)

        Returns:
            Adjusted secondary gradient
        """
        # Flatten gradients
        g1 = grad_primary.flatten()
        g2 = grad_secondary.flatten()

        # Compute cosine similarity
        dot = torch.dot(g1, g2)
        norm1 = torch.norm(g1) + 1e-8
        norm2 = torch.norm(g2) + 1e-8
        cos_sim = dot / (norm1 * norm2)

        self.stats['total'] += 1

        # If gradients conflict (negative cosine similarity)
        if cos_sim < self.conflict_threshold:
            self.stats['conflicts'] += 1

            # Project g2 onto plane normal to g1
            # g2_orth = g2 - (g2 . g1 / ||g1||^2) * g1
            projection = (dot / (norm1 ** 2)) * g1
            g2_orthogonal = g2 - projection

            return g2_orthogonal.view_as(grad_secondary)

        return grad_secondary

    def apply_to_gradients(
        self,
        grads_primary: Dict[str, torch.Tensor],
        grads_secondary: Dict[str, torch.Tensor],
        model: nn.Module
    ) -> None:
        """
        Apply gradient orthogonalization to model parameters.

        Args:
            grads_primary: Dict of parameter name -> gradient (from primary loss)
            grads_secondary: Dict of parameter name -> gradient (from secondary loss)
            model: The model (will modify its .grad attributes)
        """
        for name, param in model.named_parameters():
            if param.grad is not None and name in grads_primary and name in grads_secondary:
                # Orthogonalize secondary gradient
                orth_grad = self.orthogonalize(grads_primary[name], grads_secondary[name])
                # Set combined gradient
                param.grad = grads_primary[name] + orth_grad

    def get_stats(self) -> Dict[str, float]:
        """Return conflict statistics."""
        if self.stats['total'] > 0:
            conflict_rate = self.stats['conflicts'] / self.stats['total']
        else:
            conflict_rate = 0.0
        return {
            'conflict_rate': conflict_rate,
            'total_checks': self.stats['total'],
            'total_conflicts': self.stats['conflicts']
        }


class ImprovedPhysicsLossV4(nn.Module):
    """
    Physics-based loss from V2, preserved and enhanced for V4.

    Enforces:
    - SNR-density negative correlation (more users -> lower SNR)
    - Traffic-density positive correlation (more users -> more traffic)
    - Temporal variance (prevent flat daily profiles)
    - Capacity constraints

    Key difference from V2: integrates with V4's combined loss framework.
    """

    def __init__(
        self,
        weight: float = 0.1,
        warmup_epochs: int = 150,
        feature_indices: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        self.weight = weight
        self.warmup_epochs = warmup_epochs
        self.warmup_start = 0.001

        # Feature indices (default for our WiFi data)
        self.indices = feature_indices or {
            'ue_count': 0,
            'active_ue': 1,
            'total_bytes': 4,
            'total_packets': 5,
            'avg_rssi': 8,
            'avg_snr': 9,
        }

    def get_lambda_scale(self, epoch: int) -> float:
        """Gradual warmup of physics losses."""
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            scale = self.warmup_start + (1.0 - self.warmup_start) * (1 - np.cos(np.pi * progress)) / 2
            return scale
        return 1.0

    def snr_density_loss(self, generated: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        CRITICAL: Enforce negative correlation between density and SNR.
        More users -> more interference -> lower SNR
        """
        ue_count = generated[..., self.indices['ue_count']]
        avg_snr = generated[..., self.indices['avg_snr']]

        # Flatten for correlation
        ue_flat = ue_count.reshape(-1)
        snr_flat = avg_snr.reshape(-1)

        # Compute Pearson correlation
        ue_centered = ue_flat - ue_flat.mean()
        snr_centered = snr_flat - snr_flat.mean()

        ue_std = torch.sqrt((ue_centered ** 2).sum() + 1e-8)
        snr_std = torch.sqrt((snr_centered ** 2).sum() + 1e-8)

        correlation = (ue_centered * snr_centered).sum() / (ue_std * snr_std)

        # Loss: Penalize positive correlation (we want negative)
        # Target: correlation < -0.1
        loss = F.relu(correlation + 0.1)

        return loss, correlation.item()

    def traffic_load_loss(self, generated: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Enforce positive correlation between density and traffic.
        More users -> more traffic
        """
        ue_count = generated[..., self.indices['ue_count']]
        total_bytes = generated[..., self.indices['total_bytes']]

        ue_flat = ue_count.reshape(-1)
        traffic_flat = total_bytes.reshape(-1)

        ue_centered = ue_flat - ue_flat.mean()
        traffic_centered = traffic_flat - traffic_flat.mean()

        ue_std = torch.sqrt((ue_centered ** 2).sum() + 1e-8)
        traffic_std = torch.sqrt((traffic_centered ** 2).sum() + 1e-8)

        correlation = (ue_centered * traffic_centered).sum() / (ue_std * traffic_std)

        # Loss: Penalize negative correlation (we want positive)
        # Target: correlation > 0.2
        loss = F.relu(-correlation + 0.2)

        return loss, correlation.item()

    def temporal_variance_loss(self, generated: torch.Tensor) -> torch.Tensor:
        """
        Enforce temporal variation - prevent flat daily profiles.
        Penalize low variance across time dimension.
        """
        ue_count = generated[..., self.indices['ue_count']]

        # Compute variance across time dimension
        if ue_count.dim() == 3:  # [B, T, APs]
            time_var = ue_count.var(dim=1)  # [B, APs]
            mean_val = ue_count.mean(dim=1).abs() + 1e-8
        elif ue_count.dim() == 2:  # [B, T]
            time_var = ue_count.var(dim=1)  # [B]
            mean_val = ue_count.mean(dim=1).abs() + 1e-8
        else:
            return torch.tensor(0.0, device=generated.device)

        # Coefficient of variation squared
        cv_squared = time_var / (mean_val ** 2 + 1e-8)

        # Target: CV^2 should be at least 0.1
        target_cv_squared = 0.1
        loss = F.relu(target_cv_squared - cv_squared).mean()

        return loss

    def forward(
        self,
        generated: torch.Tensor,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all physics losses with warmup scaling.

        Args:
            generated: Generated samples [B, T, F]
            epoch: Current training epoch

        Returns:
            Dictionary with all physics loss components
        """
        scale = self.get_lambda_scale(epoch)

        # Individual losses
        snr_loss, snr_corr = self.snr_density_loss(generated)
        traffic_loss, traffic_corr = self.traffic_load_loss(generated)
        temporal_loss = self.temporal_variance_loss(generated)

        # Weighted sum with warmup scaling
        total_physics = scale * self.weight * (
            0.2 * snr_loss +       # SNR-density (most important)
            0.1 * traffic_loss +   # Traffic-load
            0.1 * temporal_loss    # Temporal variance
        )

        return {
            'total': total_physics,
            'snr_density_loss': snr_loss,
            'snr_density_corr': snr_corr,
            'traffic_load_loss': traffic_loss,
            'traffic_load_corr': traffic_corr,
            'temporal_var_loss': temporal_loss,
            'lambda_scale': scale
        }


class V4CombinedLoss(nn.Module):
    """
    Combined loss for V4 training.

    L_total = L_diffusion + lambda_var * L_variance + lambda_contrast * L_contrastive
              + lambda_physics * L_physics + lambda_kl * L_kl

    Provides unified interface for all V4 loss components.
    """

    def __init__(
        self,
        target_stds: Optional[torch.Tensor] = None,
        feature_indices: Optional[Dict[str, int]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()

        # Default weights
        self.weights = weights or {
            'diffusion': 1.0,
            'variance': 1.0,
            'contrastive': 0.5,
            'physics': 0.1,
            'kl': 0.01
        }

        # Loss components
        self.variance_loss = VarianceMatchingLoss(
            target_stds=target_stds,
            weight=1.0,
            min_ratio=0.5,
            max_ratio=1.5
        )
        self.contrastive_loss = TemporalContrastiveLoss(temperature=0.1, weight=1.0)
        self.physics_loss = ImprovedPhysicsLossV4(
            weight=1.0,
            feature_indices=feature_indices
        )

        # Gradient orthogonalizer
        self.orthogonalizer = GradientOrthogonalizer(conflict_threshold=0.0)

    def set_target_stds(self, stds: torch.Tensor):
        """Set target stds for variance loss (call once with real data stats)."""
        self.variance_loss.set_target_stds(stds)

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        generated_samples: Optional[torch.Tensor] = None,
        real_samples: Optional[torch.Tensor] = None,
        representations: Optional[torch.Tensor] = None,
        hour_labels: Optional[torch.Tensor] = None,
        aux_outputs: Optional[Dict[str, torch.Tensor]] = None,
        epoch: int = 0,
        timesteps: Optional[torch.Tensor] = None,
        variance_t_threshold: int = 200
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            noise_pred: Predicted noise [B, T, F]
            noise_target: Target noise [B, T, F]
            generated_samples: Generated samples for variance loss [B, T, F]
            real_samples: Real samples for reference [B, T, F]
            representations: Sample representations for contrastive [B, D]
            hour_labels: Hour indices for contrastive [B]
            aux_outputs: Dict with KL terms from model forward pass
            epoch: Current training epoch
            timesteps: Diffusion timesteps [B] - used to filter high-noise samples
            variance_t_threshold: Only compute variance/physics loss when t < threshold

        Returns:
            Dictionary with all loss components and total
        """
        losses = {}
        stats = {}

        # 1. Diffusion loss (MSE on noise prediction)
        losses['diffusion'] = F.mse_loss(noise_pred, noise_target)

        # 2. Variance matching loss
        # Only compute on low-timestep samples where x0_pred is meaningful
        # At high t, x0_pred = (x_noisy - sqrt_1-alpha * noise_pred) / sqrt_alpha explodes
        if generated_samples is not None and timesteps is not None:
            # Filter to low-timestep samples
            low_t_mask = timesteps < variance_t_threshold
            if low_t_mask.sum() > 0:
                gen_low_t = generated_samples[low_t_mask]
                real_low_t = real_samples[low_t_mask] if real_samples is not None else None
                var_loss, var_stats = self.variance_loss(gen_low_t, real_low_t)
                losses['variance'] = var_loss
                stats.update({f'var_{k}': v for k, v in var_stats.items()})
                stats['variance_samples'] = low_t_mask.sum().item()
            else:
                losses['variance'] = torch.tensor(0.0, device=noise_pred.device)
                stats['variance_samples'] = 0
        elif generated_samples is not None:
            # No timesteps provided - compute on all (legacy behavior)
            var_loss, var_stats = self.variance_loss(generated_samples, real_samples)
            losses['variance'] = var_loss
            stats.update({f'var_{k}': v for k, v in var_stats.items()})
        else:
            losses['variance'] = torch.tensor(0.0, device=noise_pred.device)

        # 3. Temporal contrastive loss
        if representations is not None and hour_labels is not None:
            contrast_loss, contrast_stats = self.contrastive_loss(representations, hour_labels)
            losses['contrastive'] = contrast_loss
            stats.update({f'contrast_{k}': v for k, v in contrast_stats.items()})
        else:
            losses['contrastive'] = torch.tensor(0.0, device=noise_pred.device)

        # 4. Physics losses
        # Also filter by timestep like variance loss
        if generated_samples is not None and timesteps is not None:
            low_t_mask = timesteps < variance_t_threshold
            if low_t_mask.sum() > 0:
                gen_low_t = generated_samples[low_t_mask]
                physics_results = self.physics_loss(gen_low_t, epoch=epoch)
                losses['physics'] = physics_results['total']
                losses['snr_density'] = physics_results['snr_density_loss']
                stats['snr_density_corr'] = physics_results['snr_density_corr']
                stats['traffic_load_corr'] = physics_results['traffic_load_corr']
                stats['physics_lambda_scale'] = physics_results['lambda_scale']
                stats['physics_samples'] = low_t_mask.sum().item()
            else:
                losses['physics'] = torch.tensor(0.0, device=noise_pred.device)
                losses['snr_density'] = torch.tensor(0.0, device=noise_pred.device)
                stats['physics_samples'] = 0
        elif generated_samples is not None:
            # No timesteps provided - compute on all (legacy behavior)
            physics_results = self.physics_loss(generated_samples, epoch=epoch)
            losses['physics'] = physics_results['total']
            losses['snr_density'] = physics_results['snr_density_loss']
            stats['snr_density_corr'] = physics_results['snr_density_corr']
            stats['traffic_load_corr'] = physics_results['traffic_load_corr']
            stats['physics_lambda_scale'] = physics_results['lambda_scale']
        else:
            losses['physics'] = torch.tensor(0.0, device=noise_pred.device)
            losses['snr_density'] = torch.tensor(0.0, device=noise_pred.device)

        # 5. KL divergence (regularization on variational embeddings)
        if aux_outputs is not None:
            kl_hour = aux_outputs.get('kl_hour', torch.tensor(0.0, device=noise_pred.device))
            kl_ap = aux_outputs.get('kl_ap', torch.tensor(0.0, device=noise_pred.device))
            losses['kl'] = kl_hour + kl_ap
            stats['kl_hour'] = kl_hour.item() if isinstance(kl_hour, torch.Tensor) else kl_hour
            stats['kl_ap'] = kl_ap.item() if isinstance(kl_ap, torch.Tensor) else kl_ap
        else:
            losses['kl'] = torch.tensor(0.0, device=noise_pred.device)

        # Weighted total
        losses['total'] = (
            self.weights['diffusion'] * losses['diffusion'] +
            self.weights['variance'] * losses['variance'] +
            self.weights['contrastive'] * losses['contrastive'] +
            self.weights['physics'] * losses['physics'] +
            self.weights['kl'] * losses['kl']
        )

        # Add stats to output
        losses['_stats'] = stats

        return losses

    def get_orthogonalizer_stats(self) -> Dict[str, float]:
        """Get gradient orthogonalization statistics."""
        return self.orthogonalizer.get_stats()


def compute_real_feature_stds(dataloader) -> torch.Tensor:
    """
    Compute per-feature standard deviations from real data.
    Call this once during training initialization.

    Args:
        dataloader: PyTorch DataLoader with real data

    Returns:
        Tensor of per-feature stds [num_features]
    """
    all_data = []

    for batch in dataloader:
        x = batch['features']  # [B, T, A, F] or [B, T, F]
        if x.dim() == 4:
            # Flatten: [B, T, A, F] -> [B*T*A, F]
            x = x.reshape(-1, x.shape[-1])
        else:
            # Flatten: [B, T, F] -> [B*T, F]
            x = x.reshape(-1, x.shape[-1])
        all_data.append(x)

    all_data = torch.cat(all_data, dim=0)
    return all_data.std(dim=0)
