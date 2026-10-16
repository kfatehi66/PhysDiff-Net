#!/usr/bin/env python3
"""
PhysDiff-Net V5 Training Script

Key differences from V4:
1. Uses PCGrad for gradient surgery
2. Auxiliary hour prediction loss
3. NO LR warm restarts (caused V4 collapse)
4. Early stopping based on pattern score
"""

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.physdiff_net_v5 import PhysDiffNetV5
from src.models.pcgrad import PCGrad, PCGradOptimizer, GradientMonitor
from src.training.losses_v5 import V5LossFunction, EvaluationMetrics


def load_data(config: Dict) -> tuple:
    """Load and prepare training data."""
    print("Loading data...")

    train_path = Path(config.get('train_data', 'data/processed/train_normalized.npz'))
    val_path = Path(config.get('val_data', 'data/processed/val_normalized.npz'))

    train_data = np.load(train_path, allow_pickle=True)
    val_data = np.load(val_path, allow_pickle=True)

    # Get features (windows in this dataset)
    train_features = train_data['windows']  # [N, T, APs, features]
    val_features = val_data['windows']

    # Extract hours from timestamps
    train_timestamps = train_data['timestamps']
    val_timestamps = val_data['timestamps']

    train_hours = np.array([
        int(str(ts).split('T')[1].split(':')[0])
        for ts in train_timestamps
    ])
    val_hours = np.array([
        int(str(ts).split('T')[1].split(':')[0])
        for ts in val_timestamps
    ])

    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")
    print(f"Hours range: {train_hours.min()}-{train_hours.max()}")

    # Reshape: [N, T, APs, F] -> [N, T, APs*F]
    N, T, num_aps, num_features = train_features.shape

    # For V5, we train per-AP (simplifies conditioning)
    # Create expanded dataset with AP index
    print("Expanding dataset per AP...")

    train_samples = []
    train_hours_expanded = []
    train_ap_ids = []

    for ap_idx in range(num_aps):
        ap_features = train_features[:, :, ap_idx, :]  # [N, T, F]
        train_samples.append(ap_features)
        train_hours_expanded.append(train_hours)
        train_ap_ids.append(np.full(len(train_hours), ap_idx))

    train_samples = np.concatenate(train_samples, axis=0)
    train_hours_expanded = np.concatenate(train_hours_expanded)
    train_ap_ids = np.concatenate(train_ap_ids)

    # Same for validation
    val_samples = []
    val_hours_expanded = []
    val_ap_ids = []

    for ap_idx in range(num_aps):
        ap_features = val_features[:, :, ap_idx, :]
        val_samples.append(ap_features)
        val_hours_expanded.append(val_hours)
        val_ap_ids.append(np.full(len(val_hours), ap_idx))

    val_samples = np.concatenate(val_samples, axis=0)
    val_hours_expanded = np.concatenate(val_hours_expanded)
    val_ap_ids = np.concatenate(val_ap_ids)

    print(f"Expanded train samples: {train_samples.shape}")
    print(f"Expanded val samples: {val_samples.shape}")

    # Convert to tensors
    train_tensors = (
        torch.tensor(train_samples, dtype=torch.float32),
        torch.tensor(train_hours_expanded, dtype=torch.long),
        torch.tensor(train_ap_ids, dtype=torch.long)
    )
    val_tensors = (
        torch.tensor(val_samples, dtype=torch.float32),
        torch.tensor(val_hours_expanded, dtype=torch.long),
        torch.tensor(val_ap_ids, dtype=torch.long)
    )

    # Create datasets
    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    data_info = {
        'num_features': num_features,
        'seq_len': T,
        'num_aps': num_aps,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset)
    }

    return train_loader, val_loader, data_info


def evaluate_model(
    model: PhysDiffNetV5,
    val_loader: DataLoader,
    device: torch.device,
    num_eval_samples: int = 1000,
    cfg_scale: float = 7.5
) -> Dict[str, float]:
    """
    Evaluate model and compute metrics.
    """
    model.eval()

    # Generate samples for each hour
    all_generated = []
    all_hours = []

    samples_per_hour = num_eval_samples // 24

    with torch.no_grad():
        for hour in range(24):
            hours = torch.full((samples_per_hour,), hour, dtype=torch.long, device=device)
            ap_ids = torch.randint(0, model.num_aps, (samples_per_hour,), device=device)

            generated = model.generate(
                num_samples=samples_per_hour,
                hours=hours,
                ap_ids=ap_ids,
                num_steps=50,
                cfg_scale=cfg_scale
            )

            all_generated.append(generated.cpu().numpy())
            all_hours.extend([hour] * samples_per_hour)

    generated_np = np.concatenate(all_generated, axis=0)
    hours_np = np.array(all_hours)

    # Get real samples for comparison
    real_samples = []
    real_hours = []

    for batch in val_loader:
        features, hours, ap_ids = batch
        real_samples.append(features.numpy())
        real_hours.extend(hours.numpy())
        if len(real_samples) * features.shape[0] >= num_eval_samples:
            break

    real_np = np.concatenate(real_samples, axis=0)[:num_eval_samples]
    real_hours_np = np.array(real_hours[:num_eval_samples])

    # Compute metrics
    evaluator = EvaluationMetrics()
    metrics = evaluator.compute_all(generated_np, real_np, hours_np, real_hours_np)

    return metrics


def train_epoch(
    model: PhysDiffNetV5,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: V5LossFunction,
    device: torch.device,
    epoch: int,
    use_pcgrad: bool = True,
    grad_monitor: Optional[GradientMonitor] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    """
    model.train()
    epoch_losses = []
    conflict_rates = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, (features, hours, ap_ids) in enumerate(pbar):
        features = features.to(device)
        hours = hours.to(device)
        ap_ids = ap_ids.to(device)

        optimizer.zero_grad()

        # Compute individual losses
        loss_output = model.compute_loss(features, hours, ap_ids)

        # Get physics and variance losses
        physics_loss = loss_fn._compute_physics_loss(loss_output['pred_x0'])
        variance_loss = loss_fn._compute_variance_loss(
            loss_output['pred_x0'],
            loss_output['real_x0']
        )

        # Get hour prediction for auxiliary loss
        with torch.no_grad():
            # Get hidden states for hour prediction
            t = torch.randint(0, model.num_timesteps, (features.shape[0],), device=device)
            noise = torch.randn_like(features)
            sqrt_alpha = model.sqrt_alphas_cumprod[t][:, None, None]
            sqrt_one_minus = model.sqrt_one_minus_alphas_cumprod[t][:, None, None]
            x_noisy = sqrt_alpha * features + sqrt_one_minus * noise

        # Forward for hour prediction
        output = model(x_noisy, t, hours, ap_ids, return_hour_pred=True)

        # Individual losses for PCGrad
        individual_losses = {
            'diffusion': loss_output['diffusion'],
            'hour': loss_fn.lambda_hour * F.cross_entropy(output['hour_pred'], hours),
            'physics': loss_fn.lambda_physics * physics_loss * min(1.0, epoch / loss_fn.physics_warmup_epochs),
            'variance': loss_fn.lambda_variance * variance_loss,
            'kl': loss_fn.lambda_kl * loss_output['kl']
        }

        if use_pcgrad:
            # Apply PCGrad
            loss_list = [
                individual_losses['diffusion'],
                individual_losses['hour'],
                individual_losses['physics'],
                individual_losses['variance']
            ]

            stats = PCGrad.apply_pcgrad(loss_list, model)
            conflict_rates.append(stats['conflict_rate'])

            # Add KL (doesn't conflict with others typically)
            individual_losses['kl'].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss = sum(l.item() for l in loss_list) + individual_losses['kl'].item()
        else:
            # Standard training
            total_loss = sum(individual_losses.values())
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss = total_loss.item()

        loss_dict = {k: v.item() for k, v in individual_losses.items()}
        loss_dict['total'] = total_loss
        epoch_losses.append(loss_dict)

        pbar.set_postfix({
            'loss': f"{total_loss:.4f}",
            'hour': f"{loss_dict['hour']:.4f}",
            'phys': f"{loss_dict['physics']:.4f}"
        })

    # Average losses
    avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()}

    if conflict_rates:
        avg_losses['conflict_rate'] = np.mean(conflict_rates)

    return avg_losses


def main(config_path: str = 'config/v5_training.yaml'):
    """Main training function."""
    print("=" * 70)
    print("PhysDiff-Net V5 Training")
    print("Fixed Hour Conditioning with Gradient Surgery")
    print("=" * 70)

    # Load config
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {}

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Directories
    checkpoint_dir = Path('checkpoints')
    log_dir = Path('logs')
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Load data
    train_loader, val_loader, data_info = load_data(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Update config with data info
    config.update(data_info)

    # Initialize model
    print("\nInitializing model...")
    model = PhysDiffNetV5(config).to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Loss function
    loss_fn = V5LossFunction(
        lambda_physics=config.get('lambda_physics', 0.1),
        lambda_hour=config.get('lambda_hour', 0.5),
        lambda_variance=config.get('lambda_variance', 0.1),
        lambda_kl=config.get('lambda_kl', 0.01),
        lambda_contrastive=config.get('lambda_contrastive', 0.1),
        physics_warmup_epochs=config.get('physics_warmup_epochs', 100)
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01),
        betas=(0.9, 0.999)
    )

    # Scheduler - NO WARM RESTARTS (caused V4 collapse)
    num_epochs = config.get('num_epochs', 600)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=config.get('min_lr', 1e-6)
    )

    # Gradient monitor
    grad_monitor = GradientMonitor(['diffusion', 'hour', 'physics', 'variance'])

    # Training history
    history = {
        'train_losses': [],
        'eval_metrics': [],
        'config': config
    }

    best_pattern_score = 0
    best_epoch = 0
    use_pcgrad = config.get('use_pcgrad', True)
    eval_every = config.get('eval_every', 25)
    early_stop_patience = config.get('early_stop_patience', 100)
    epochs_without_improvement = 0

    print(f"\n{'='*60}")
    print(f"Starting V5 Training: {num_epochs} epochs")
    print(f"PCGrad enabled: {use_pcgrad}")
    print(f"Key V5 changes:")
    print(f"  - Concatenation hour conditioning (not additive)")
    print(f"  - Auxiliary hour loss (lambda={loss_fn.lambda_hour})")
    print(f"  - No LR warm restarts")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn,
            device, epoch, use_pcgrad, grad_monitor
        )

        scheduler.step()

        history['train_losses'].append({
            'epoch': epoch + 1,
            'losses': train_losses,
            'lr': scheduler.get_last_lr()[0]
        })

        # Log
        conflict_str = f", Conflicts: {train_losses.get('conflict_rate', 0):.1%}" if use_pcgrad else ""
        print(f"\nEpoch {epoch+1}/{num_epochs}: "
              f"Total={train_losses['total']:.4f}, "
              f"Diff={train_losses['diffusion']:.4f}, "
              f"Hour={train_losses['hour']:.4f}, "
              f"Phys={train_losses['physics']:.4f}"
              f"{conflict_str}")

        # Evaluate periodically
        if (epoch + 1) % eval_every == 0:
            print(f"\n  === Evaluation ===")

            metrics = evaluate_model(
                model, val_loader, device,
                num_eval_samples=config.get('num_eval_samples', 1000),
                cfg_scale=config.get('cfg_scale', 7.5)
            )

            history['eval_metrics'].append({
                'epoch': epoch + 1,
                'metrics': metrics
            })

            print(f"  Pattern Score: {metrics['pattern_score']:.1f}% (target >70%)")
            print(f"  Hourly UE Range: {metrics['hourly_ue_range']:.3f} (target >1.5)")
            print(f"  AP Variance Ratio: {metrics['ap_variance_ratio']:.3f} (target 0.8-1.2)")
            print(f"  SNR-Density Corr: {metrics['snr_density_corr']:.3f} (target <-0.3)")
            print(f"  Checks Passed: {metrics['checks_passed']}/{metrics['total_checks']}")

            # Save best model
            if metrics['pattern_score'] > best_pattern_score:
                best_pattern_score = metrics['pattern_score']
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'config': config
                }, checkpoint_dir / 'diffusion_v5_best.pt')

                print(f"  -> New best model saved! Pattern Score: {best_pattern_score:.1f}%")
            else:
                epochs_without_improvement += eval_every

            # Early stopping check
            if epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping: No improvement for {epochs_without_improvement} epochs")
                break

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_dir / f'diffusion_v5_epoch_{epoch+1}.pt')

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'config': config
    }, checkpoint_dir / 'diffusion_v5_final.pt')

    # Save history
    with open(log_dir / 'v5_training_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("V5 TRAINING COMPLETE")
    print(f"Best Pattern Score: {best_pattern_score:.1f}% (Epoch {best_epoch})")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"History saved to: {log_dir / 'v5_training_history.json'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train PhysDiff-Net V5')
    parser.add_argument('--config', type=str, default='config/v5_training.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    main(args.config)
