"""
PCGrad: Projecting Conflicting Gradients

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)

Key insight from V4 failure:
- Gradient conflicts reached 78.5% by end of training
- Physics loss and diffusion loss were fighting each other
- This caused training instability and pattern score oscillation

V5 Solution:
- When two gradients conflict (negative dot product), project one onto
  the normal plane of the other
- This removes the conflicting component while preserving non-conflicting signal
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import random


class PCGrad:
    """
    Projecting Conflicting Gradients (PCGrad) optimizer wrapper.

    When gradients from different losses conflict (negative dot product),
    project them to remove the conflicting component.
    """

    @staticmethod
    def compute_grad(
        loss: torch.Tensor,
        model: nn.Module,
        retain_graph: bool = True
    ) -> List[Optional[torch.Tensor]]:
        """
        Compute gradients for a single loss.

        Args:
            loss: Scalar loss tensor
            model: Model to compute gradients for
            retain_graph: Whether to retain computation graph

        Returns:
            List of gradient tensors (or None for unused parameters)
        """
        grads = torch.autograd.grad(
            loss,
            model.parameters(),
            retain_graph=retain_graph,
            allow_unused=True
        )
        return [g.clone() if g is not None else None for g in grads]

    @staticmethod
    def project_conflicting(
        grad_a: torch.Tensor,
        grad_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Project grad_a to remove component conflicting with grad_b.

        If dot(grad_a, grad_b) < 0, subtract the projection of grad_a
        onto grad_b to make them non-conflicting.

        Args:
            grad_a: Gradient to modify
            grad_b: Gradient to project against

        Returns:
            Modified grad_a with conflicting component removed
        """
        if grad_a is None or grad_b is None:
            return grad_a

        flat_a = grad_a.flatten()
        flat_b = grad_b.flatten()

        # Check for conflict
        dot = torch.dot(flat_a, flat_b)

        if dot < 0:
            # Project out the conflicting component
            # grad_a' = grad_a - (grad_a . grad_b / ||grad_b||^2) * grad_b
            norm_b_sq = torch.dot(flat_b, flat_b) + 1e-8
            proj = dot / norm_b_sq
            flat_a = flat_a - proj * flat_b

        return flat_a.view_as(grad_a)

    @staticmethod
    def apply_pcgrad(
        losses: List[torch.Tensor],
        model: nn.Module,
        reduction: str = 'sum'
    ) -> Dict[str, float]:
        """
        Apply PCGrad to multiple losses and set model gradients.

        Args:
            losses: List of loss tensors
            model: Model to optimize
            reduction: How to combine projected gradients ('sum' or 'mean')

        Returns:
            Dictionary with conflict statistics
        """
        # Compute gradients for each loss
        all_grads = [PCGrad.compute_grad(loss, model) for loss in losses]
        num_losses = len(losses)
        num_params = len(all_grads[0])

        # Track conflicts
        total_conflicts = 0
        total_pairs = 0

        final_grads = []

        for param_idx in range(num_params):
            grads_for_param = [grads[param_idx] for grads in all_grads]

            # Skip if all None
            if all(g is None for g in grads_for_param):
                final_grads.append(None)
                continue

            # Random order for projection (as per paper)
            indices = list(range(num_losses))
            random.shuffle(indices)

            projected = []
            for i in indices:
                g_i = grads_for_param[i]
                if g_i is None:
                    continue

                g_proj = g_i.clone()

                # Project against all other gradients
                for j in indices:
                    if i != j and grads_for_param[j] is not None:
                        g_j = grads_for_param[j]

                        # Check for conflict
                        flat_i = g_proj.flatten()
                        flat_j = g_j.flatten()
                        dot = torch.dot(flat_i, flat_j)

                        if dot < 0:
                            total_conflicts += 1
                            g_proj = PCGrad.project_conflicting(g_proj, g_j)

                        total_pairs += 1

                projected.append(g_proj)

            # Combine projected gradients
            if projected:
                if reduction == 'sum':
                    final_grads.append(sum(projected))
                else:  # mean
                    final_grads.append(sum(projected) / len(projected))
            else:
                final_grads.append(None)

        # Set gradients on model parameters
        for param, grad in zip(model.parameters(), final_grads):
            if grad is not None:
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad = grad  # Overwrite

        conflict_rate = total_conflicts / max(total_pairs, 1)
        return {
            'conflict_rate': conflict_rate,
            'total_conflicts': total_conflicts,
            'total_pairs': total_pairs
        }


class PCGradOptimizer:
    """
    Wrapper that applies PCGrad before optimizer step.

    Usage:
        optimizer = torch.optim.AdamW(model.parameters())
        pcgrad_opt = PCGradOptimizer(optimizer, model)

        losses = [loss1, loss2, loss3]
        pcgrad_opt.step(losses)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        reduction: str = 'sum'
    ):
        self.optimizer = optimizer
        self.model = model
        self.reduction = reduction
        self.conflict_history = []

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, losses: List[torch.Tensor]) -> Dict[str, float]:
        """
        Apply PCGrad and take optimizer step.

        Args:
            losses: List of loss tensors

        Returns:
            Conflict statistics
        """
        self.zero_grad()

        stats = PCGrad.apply_pcgrad(
            losses,
            self.model,
            reduction=self.reduction
        )

        self.conflict_history.append(stats['conflict_rate'])

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return stats

    def get_average_conflict_rate(self, window: int = 100) -> float:
        """Get average conflict rate over recent steps."""
        if not self.conflict_history:
            return 0.0
        recent = self.conflict_history[-window:]
        return sum(recent) / len(recent)


class GradientMonitor:
    """
    Monitor gradient conflicts and magnitudes during training.

    Useful for debugging and understanding loss interactions.
    """

    def __init__(self, loss_names: List[str]):
        self.loss_names = loss_names
        self.history = {
            'conflicts': [],
            'magnitudes': {name: [] for name in loss_names},
            'angles': []
        }

    def log_gradients(
        self,
        losses: List[torch.Tensor],
        model: nn.Module
    ) -> Dict[str, float]:
        """
        Log gradient statistics without modifying them.

        Returns:
            Dictionary of gradient statistics
        """
        all_grads = [PCGrad.compute_grad(loss, model) for loss in losses]

        # Flatten all gradients
        flat_grads = []
        for grads in all_grads:
            flat = torch.cat([
                g.flatten() for g in grads if g is not None
            ])
            flat_grads.append(flat)

        # Compute pairwise angles
        num_losses = len(losses)
        angles = []
        conflicts = 0

        for i in range(num_losses):
            for j in range(i + 1, num_losses):
                cos_sim = torch.nn.functional.cosine_similarity(
                    flat_grads[i].unsqueeze(0),
                    flat_grads[j].unsqueeze(0)
                ).item()
                angle = torch.acos(torch.clamp(torch.tensor(cos_sim), -1, 1)).item()
                angles.append(angle)

                if cos_sim < 0:
                    conflicts += 1

        # Log magnitudes
        magnitudes = {
            name: flat_grads[i].norm().item()
            for i, name in enumerate(self.loss_names)
        }

        stats = {
            'conflict_pairs': conflicts,
            'total_pairs': num_losses * (num_losses - 1) // 2,
            'conflict_rate': conflicts / max(1, num_losses * (num_losses - 1) // 2),
            'avg_angle': sum(angles) / len(angles) if angles else 0,
            'magnitudes': magnitudes
        }

        self.history['conflicts'].append(stats['conflict_rate'])
        self.history['angles'].append(stats['avg_angle'])
        for name, mag in magnitudes.items():
            self.history['magnitudes'][name].append(mag)

        return stats
