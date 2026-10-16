"""
V5 Loss Functions with Auxiliary Hour Loss

Key changes from V4:
1. AUXILIARY HOUR LOSS: Cross-entropy on hour prediction
   - Creates direct gradient path for hour awareness
   - lambda_hour = 0.5 (significant weight)

2. Designed for PCGRAD:
   - Individual losses returned separately
   - Gradient surgery handles conflicts

3. Physics loss warmup:
   - Gradual introduction prevents early instability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from scipy import stats
import numpy as np


class V5LossFunction(nn.Module):
    """
    Combined loss function for PhysDiff-Net V5.

    Components:
    1. Diffusion loss (MSE on noise prediction)
    2. Physics loss (SNR-density anti-correlation)
    3. Auxiliary hour loss (cross-entropy) - KEY V5 ADDITION
    4. Variance preservation loss
    5. KL divergence (for variational hour embedding)
    """

    def __init__(
        self,
        lambda_physics: float = 0.1,
        lambda_hour: float = 0.5,  # Significant weight for hour loss
        lambda_variance: float = 0.1,
        lambda_kl: float = 0.01,
        lambda_contrastive: float = 0.1,
        physics_warmup_epochs: int = 100,
        snr_feature_idx: int = 9,  # avg_snr in feature list
        load_feature_idx: int = 0,  # ue_count in feature list
        target_snr_corr: float = -0.3
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_hour = lambda_hour
        self.lambda_variance = lambda_variance
        self.lambda_kl = lambda_kl
        self.lambda_contrastive = lambda_contrastive
        self.physics_warmup_epochs = physics_warmup_epochs
        self.snr_idx = snr_feature_idx
        self.load_idx = load_feature_idx
        self.target_snr_corr = target_snr_corr

        # Hour cross-entropy loss
        self.hour_ce = nn.CrossEntropyLoss()

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        hour_pred: torch.Tensor,
        hour_target: torch.Tensor,
        generated: torch.Tensor,
        real: torch.Tensor,
        kl_loss: torch.Tensor,
        current_epoch: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            noise_pred: Predicted noise [B, T, F]
            noise_target: Target noise [B, T, F]
            hour_pred: Hour prediction logits [B, 24]
            hour_target: Target hours [B]
            generated: Approximate x0 [B, T, F]
            real: Real data [B, T, F]
            kl_loss: KL divergence from hour conditioning
            current_epoch: Current training epoch

        Returns:
            total_loss: Combined loss scalar
            loss_dict: Individual loss values for logging
        """
        # 1. Diffusion loss (MSE)
        diff_loss = F.mse_loss(noise_pred, noise_target)

        # 2. Auxiliary hour loss (Cross-entropy) - KEY V5 ADDITION
        hour_loss = self.hour_ce(hour_pred, hour_target)

        # 3. Physics loss with warmup
        warmup_factor = min(1.0, current_epoch / self.physics_warmup_epochs)
        physics_loss = self._compute_physics_loss(generated) * warmup_factor

        # 4. Variance preservation loss
        variance_loss = self._compute_variance_loss(generated, real)

        # 5. Contrastive loss (hour-based)
        contrastive_loss = self._compute_contrastive_loss(generated, hour_target)

        # 6. KL regularization
        kl = kl_loss * self.lambda_kl

        # Combine losses
        total_loss = (
            diff_loss +
            self.lambda_hour * hour_loss +
            self.lambda_physics * physics_loss +
            self.lambda_variance * variance_loss +
            self.lambda_contrastive * contrastive_loss +
            kl
        )

        loss_dict = {
            'total': total_loss.item(),
            'diffusion': diff_loss.item(),
            'hour': hour_loss.item(),
            'physics': physics_loss.item(),
            'variance': variance_loss.item(),
            'contrastive': contrastive_loss.item(),
            'kl': kl.item()
        }

        return total_loss, loss_dict

    def get_individual_losses(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        hour_pred: torch.Tensor,
        hour_target: torch.Tensor,
        generated: torch.Tensor,
        real: torch.Tensor,
        kl_loss: torch.Tensor,
        current_epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Return individual losses for PCGrad.

        PCGrad needs separate loss tensors to compute gradients individually.
        """
        # 1. Diffusion loss
        diff_loss = F.mse_loss(noise_pred, noise_target)

        # 2. Hour auxiliary loss
        hour_loss = self.lambda_hour * self.hour_ce(hour_pred, hour_target)

        # 3. Physics loss
        warmup_factor = min(1.0, current_epoch / self.physics_warmup_epochs)
        physics_loss = self.lambda_physics * self._compute_physics_loss(generated) * warmup_factor

        # 4. Variance loss
        variance_loss = self.lambda_variance * self._compute_variance_loss(generated, real)

        return {
            'diffusion': diff_loss,
            'hour': hour_loss,
            'physics': physics_loss,
            'variance': variance_loss
        }

    def _compute_physics_loss(self, generated: torch.Tensor) -> torch.Tensor:
        """
        Compute SNR-density anti-correlation loss.

        Target: Generated data should have negative correlation between
        SNR and load (density), matching real-world physics where
        higher network load degrades signal quality.
        """
        if generated.dim() == 3:
            # [B, T, F] -> flatten time
            snr = generated[:, :, self.snr_idx].flatten()
            load = generated[:, :, self.load_idx].flatten()
        else:
            snr = generated[:, self.snr_idx]
            load = generated[:, self.load_idx]

        # Normalize
        snr_norm = (snr - snr.mean()) / (snr.std() + 1e-8)
        load_norm = (load - load.mean()) / (load.std() + 1e-8)

        # Pearson correlation
        correlation = (snr_norm * load_norm).mean()

        # Penalize if correlation > target (should be negative)
        physics_loss = F.relu(correlation - self.target_snr_corr)

        return physics_loss

    def _compute_variance_loss(
        self,
        generated: torch.Tensor,
        real: torch.Tensor
    ) -> torch.Tensor:
        """
        Variance preservation loss.

        Ensures generated data maintains similar variance to real data.
        """
        gen_var = generated.var(dim=(0, 1)).mean()
        real_var = real.var(dim=(0, 1)).mean()

        # Target ratio = 1.0
        ratio = gen_var / (real_var + 1e-8)
        variance_loss = (ratio - 1.0).abs()

        return variance_loss

    def _compute_contrastive_loss(
        self,
        generated: torch.Tensor,
        hours: torch.Tensor
    ) -> torch.Tensor:
        """
        Hour-based contrastive loss.

        Encourages samples from different hours to have different patterns.
        """
        B = generated.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=generated.device)

        # Pool features
        pooled = generated.mean(dim=1)  # [B, F]

        # Normalize
        pooled_norm = F.normalize(pooled, p=2, dim=-1)

        # Pairwise similarity
        sim_matrix = torch.matmul(pooled_norm, pooled_norm.t())  # [B, B]

        # Create hour similarity mask (1 if same hour, 0 otherwise)
        hours_expanded = hours.unsqueeze(1)  # [B, 1]
        same_hour = (hours_expanded == hours_expanded.t()).float()

        # Contrastive loss: same hour should be similar, different hours different
        # Loss = mean(sim where different hour) - mean(sim where same hour)
        diff_hour_mask = 1 - same_hour - torch.eye(B, device=generated.device)
        same_hour_mask = same_hour - torch.eye(B, device=generated.device)

        if diff_hour_mask.sum() > 0:
            diff_hour_sim = (sim_matrix * diff_hour_mask).sum() / diff_hour_mask.sum()
        else:
            diff_hour_sim = torch.tensor(0.0, device=generated.device)

        if same_hour_mask.sum() > 0:
            same_hour_sim = (sim_matrix * same_hour_mask).sum() / same_hour_mask.sum()
        else:
            same_hour_sim = torch.tensor(0.0, device=generated.device)

        # Want: diff_hour_sim << same_hour_sim
        # Loss: max(0, diff_hour_sim - same_hour_sim + margin)
        margin = 0.2
        contrastive_loss = F.relu(diff_hour_sim - same_hour_sim + margin)

        return contrastive_loss


class EvaluationMetrics:
    """
    Compute evaluation metrics for V5.
    """

    def __init__(
        self,
        snr_feature_idx: int = 9,
        load_feature_idx: int = 0
    ):
        self.snr_idx = snr_feature_idx
        self.load_idx = load_feature_idx

    def compute_all(
        self,
        generated: np.ndarray,
        real: np.ndarray,
        gen_hours: np.ndarray,
        real_hours: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # 1. AP Variance Ratio
        gen_var = np.var(generated, axis=(0, 1)).mean()
        real_var = np.var(real, axis=(0, 1)).mean()
        metrics['ap_variance_ratio'] = float(gen_var / (real_var + 1e-8))
        metrics['ap_variance_pass'] = 0.8 <= metrics['ap_variance_ratio'] <= 1.2

        # 2. SNR-Density Correlation
        if generated.ndim == 3:
            gen_snr = generated[:, :, self.snr_idx].flatten()
            gen_load = generated[:, :, self.load_idx].flatten()
        else:
            gen_snr = generated[:, self.snr_idx]
            gen_load = generated[:, self.load_idx]

        try:
            corr, _ = stats.pearsonr(gen_snr, gen_load)
            metrics['snr_density_corr'] = float(corr)
        except:
            metrics['snr_density_corr'] = 0.0
        metrics['snr_density_pass'] = metrics['snr_density_corr'] < -0.3

        # 3. Hourly UE Range (KEY V5 METRIC)
        hourly_means = []
        for h in range(24):
            mask = gen_hours == h
            if mask.sum() > 0:
                if generated.ndim == 3:
                    mean_load = generated[mask, :, self.load_idx].mean()
                else:
                    mean_load = generated[mask, self.load_idx].mean()
                hourly_means.append(float(mean_load))

        if hourly_means:
            metrics['hourly_ue_range'] = max(hourly_means) - min(hourly_means)
        else:
            metrics['hourly_ue_range'] = 0.0
        metrics['hourly_ue_pass'] = metrics['hourly_ue_range'] > 1.5

        # 4. Pattern Score
        checks = sum([
            metrics['ap_variance_pass'],
            metrics['snr_density_pass'],
            metrics['hourly_ue_pass']
        ])
        metrics['checks_passed'] = checks
        metrics['total_checks'] = 3
        metrics['pattern_score'] = (checks / 3) * 100

        return metrics

    def compute_hour_accuracy(
        self,
        hour_pred: np.ndarray,
        hour_target: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute hour prediction accuracy metrics.
        """
        pred_hours = hour_pred.argmax(axis=1)
        accuracy = (pred_hours == hour_target).mean()

        # Top-3 accuracy
        top3_pred = np.argsort(hour_pred, axis=1)[:, -3:]
        top3_acc = np.mean([hour_target[i] in top3_pred[i] for i in range(len(hour_target))])

        return {
            'hour_accuracy': float(accuracy),
            'hour_top3_accuracy': float(top3_acc)
        }
