"""Training loop for PhysDiff-Net.

This module implements the main training loop with support for:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- TensorBoard logging
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models import PhysDiffNet


class Trainer:
    """Trainer for PhysDiff-Net model.

    Handles the complete training loop including optimization,
    logging, checkpointing, and validation.

    Attributes:
        model: PhysDiff-Net model
        config: Training configuration
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        writer: TensorBoard writer
    """

    def __init__(
        self,
        model: PhysDiffNet,
        config: Dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the trainer.

        Args:
            model: PhysDiff-Net model to train
            config: Training configuration dictionary
            train_loader: Training data loader
            val_loader: Optional validation data loader
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        training_config = config.get("training", {})

        # Setup checkpoint directory
        self.checkpoint_dir = Path(
            checkpoint_dir or training_config.get("checkpoint_dir", "checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        self.epochs = training_config.get("epochs", 500)
        self.gradient_accumulation_steps = training_config.get(
            "gradient_accumulation_steps", 1
        )
        self.save_every = training_config.get("save_every", 10)
        self.log_every = training_config.get("log_every", 100)

        # Device
        self.device = config.get("hardware", {}).get("device", "cuda")
        self.model.to(self.device)

        # Mixed precision
        self.use_amp = config.get("hardware", {}).get("mixed_precision", True)
        self.scaler = GradScaler() if self.use_amp else None

        # Setup optimizer
        self._setup_optimizer(training_config)

        # Setup scheduler
        self._setup_scheduler(training_config)

        # Setup logging
        log_dir = Path(training_config.get("tensorboard_dir", "logs/tensorboard"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        # Early stopping
        early_config = training_config.get("early_stopping", {})
        self.early_stopping_enabled = early_config.get("enabled", True)
        self.patience = early_config.get("patience", 50)
        self.min_delta = early_config.get("min_delta", 1e-4)
        self.patience_counter = 0

    def _setup_optimizer(self, training_config: Dict) -> None:
        """Setup optimizer.

        Args:
            training_config: Training configuration
        """
        opt_config = training_config.get("optimizer", {})
        opt_type = opt_config.get("type", "adamw").lower()

        lr = opt_config.get("lr", 1e-4)
        weight_decay = opt_config.get("weight_decay", 0.01)
        betas = tuple(opt_config.get("betas", [0.9, 0.999]))

        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        elif opt_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=betas,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

    def _setup_scheduler(self, training_config: Dict) -> None:
        """Setup learning rate scheduler.

        Args:
            training_config: Training configuration
        """
        sched_config = training_config.get("scheduler", {})
        sched_type = sched_config.get("type", "cosine").lower()

        warmup_epochs = sched_config.get("warmup_epochs", 10)
        min_lr = sched_config.get("min_lr", 1e-6)

        if sched_type == "cosine":
            # Cosine annealing with warmup
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - warmup_epochs,
                eta_min=min_lr,
            )
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            )
        elif sched_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get("step_size", 100),
                gamma=sched_config.get("gamma", 0.5),
            )
            self.warmup_scheduler = None
        else:
            self.scheduler = None
            self.warmup_scheduler = None

    def train(self) -> Dict[str, float]:
        """Run the full training loop.

        Returns:
            Dictionary of final metrics
        """
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}, Mixed Precision: {self.use_amp}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate()
                self._log_metrics(val_metrics, prefix="val", epoch=epoch)

                # Check for improvement
                val_loss = val_metrics.get("loss", float("inf"))
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best.pt")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.early_stopping_enabled and self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Log training metrics
            self._log_metrics(train_metrics, prefix="train", epoch=epoch)

            # Update learning rate
            if self.warmup_scheduler and epoch < 10:
                self.warmup_scheduler.step()
            elif self.scheduler:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Print progress
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Loss: {train_metrics['loss']:.4f} - "
                f"LR: {lr:.2e}"
            )

        # Save final checkpoint
        self._save_checkpoint("final.pt")

        self.writer.close()

        return {
            "final_loss": train_metrics["loss"],
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.current_epoch + 1,
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        epoch_losses = {
            "loss": 0.0,
            "diffusion": 0.0,
            "physics": 0.0,
        }
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    loss, loss_dict = self.model.compute_loss(batch)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                loss, loss_dict = self.model.compute_loss(batch)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Log to tensorboard
                if self.global_step % self.log_every == 0:
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(
                            f"train_step/{key}",
                            value.item() if torch.is_tensor(value) else value,
                            self.global_step,
                        )

            # Accumulate losses
            for key in epoch_losses:
                if key in loss_dict:
                    val = loss_dict[key]
                    epoch_losses[key] += val.item() if torch.is_tensor(val) else val
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss_dict.get("total", loss_dict.get("loss", 0)).item()
            })

        # Average losses
        return {k: v / num_batches for k, v in epoch_losses.items()}

    def _validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_losses = {
            "loss": 0.0,
            "diffusion": 0.0,
            "physics": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                if self.use_amp:
                    with autocast():
                        loss, loss_dict = self.model.compute_loss(batch)
                else:
                    loss, loss_dict = self.model.compute_loss(batch)

                for key in val_losses:
                    if key in loss_dict:
                        val = loss_dict[key]
                        val_losses[key] += val.item() if torch.is_tensor(val) else val
                num_batches += 1

        return {k: v / num_batches for k, v in val_losses.items()}

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str,
        epoch: int,
    ) -> None:
        """Log metrics to TensorBoard.

        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for metric names (train/val)
            epoch: Current epoch
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}_epoch/{key}", value, epoch)

    def _save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Resumed from epoch {self.current_epoch}")
