"""PhysDiff-Net: Physics-Informed Diffusion Network for WiFi Data Generation.

This is the main model architecture combining:
1. TS2Vec encoder for time series representation learning
2. Conditional diffusion model for generation
3. Physics-informed losses for constraint satisfaction
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from .encoder import TS2VecWrapper
from .diffusion import ConditionalDiffusion, NoiseScheduler
from .physics import PhysicsConstraints


class PhysDiffNet(nn.Module):
    """PhysDiff-Net: Physics-Informed Diffusion Network.

    Combines representation learning, conditional diffusion, and
    physics constraints for realistic WiFi network data generation.

    Architecture:
    1. Encoder: TS2Vec learns hierarchical time series representations
    2. Diffusion: Conditional diffusion generates synthetic sequences
    3. Physics: Constraint losses ensure physical realism

    Attributes:
        encoder: TS2Vec encoder for representations
        diffusion: Conditional diffusion model
        scheduler: Noise scheduler for diffusion
        physics: Physics constraint losses
        scenario_embedder: Embeds scenario parameters
    """

    def __init__(
        self,
        config: Dict,
        ap_distances: Optional[torch.Tensor] = None,
    ):
        """Initialize PhysDiff-Net.

        Args:
            config: Configuration dictionary with model settings
            ap_distances: Optional AP distance matrix for spatial constraints
        """
        super().__init__()

        self.config = config
        model_config = config.get("model", {})

        # Extract dimensions from config
        encoder_config = model_config.get("encoder", {})
        self.input_dims = encoder_config.get("input_dims", 28)
        self.hidden_dims = encoder_config.get("hidden_dims", 64)
        self.output_dims = encoder_config.get("output_dims", 320)

        diffusion_config = model_config.get("diffusion", {})
        self.num_timesteps = diffusion_config.get("num_timesteps", 1000)

        conditioning_config = model_config.get("conditioning", {})
        self.scenario_embed_dim = conditioning_config.get("scenario_embed_dim", 64)

        # Device
        self.device = config.get("hardware", {}).get("device", "cuda")

        # Initialize components
        self._init_encoder(encoder_config)
        self._init_diffusion(diffusion_config)
        self._init_scheduler(diffusion_config)
        self._init_physics(config, ap_distances)
        self._init_scenario_embedder()

    def _init_encoder(self, encoder_config: Dict) -> None:
        """Initialize the TS2Vec encoder.

        Args:
            encoder_config: Encoder configuration
        """
        self.encoder = TS2VecWrapper(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=encoder_config.get("depth", 10),
            device=self.device,
        )

    def _init_diffusion(self, diffusion_config: Dict) -> None:
        """Initialize the diffusion model.

        Args:
            diffusion_config: Diffusion configuration
        """
        unet_config = diffusion_config.get("unet", {})

        self.diffusion = ConditionalDiffusion(
            input_dims=self.input_dims,
            hidden_dims=unet_config.get("channels", [64, 128, 256, 512])[0],
            condition_dims=self.output_dims + self.scenario_embed_dim,
            time_embed_dims=128,
            num_layers=unet_config.get("num_res_blocks", 2) * 2,
            num_heads=8,
            dropout=unet_config.get("dropout", 0.1),
        )

    def _init_scheduler(self, diffusion_config: Dict) -> None:
        """Initialize the noise scheduler.

        Args:
            diffusion_config: Diffusion configuration
        """
        self.scheduler = NoiseScheduler(
            num_timesteps=diffusion_config.get("num_timesteps", 1000),
            beta_start=diffusion_config.get("beta_start", 0.0001),
            beta_end=diffusion_config.get("beta_end", 0.02),
            schedule=diffusion_config.get("beta_schedule", "linear"),
            device=self.device,
        )

    def _init_physics(
        self,
        config: Dict,
        ap_distances: Optional[torch.Tensor],
    ) -> None:
        """Initialize physics constraints.

        Args:
            config: Full configuration
            ap_distances: Optional AP distance matrix
        """
        self.physics = PhysicsConstraints(config, ap_distances)

    def _init_scenario_embedder(self) -> None:
        """Initialize scenario parameter embedder."""
        # Scenario parameters: density_multiplier, temporal_patterns, etc.
        num_scenario_params = 8  # Adjust based on scenario config
        self.scenario_embedder = nn.Sequential(
            nn.Linear(num_scenario_params, self.scenario_embed_dim),
            nn.GELU(),
            nn.Linear(self.scenario_embed_dim, self.scenario_embed_dim),
        )

    def encode(
        self,
        x: torch.Tensor,
        encoding_window: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode time series to representations.

        Args:
            x: Input data of shape (batch, seq_length, features)
            encoding_window: 'full_series' for single vector, None for per-timestamp

        Returns:
            Encoded representations
        """
        return self.encoder.encode(x, encoding_window=encoding_window)

    def diffuse(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion (add noise).

        Args:
            x_0: Clean data of shape (batch, seq, features)
            t: Timestep indices
            noise: Optional pre-generated noise

        Returns:
            Tuple of (noisy data, noise used)
        """
        return self.scheduler.add_noise(x_0, t, noise)

    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise for denoising.

        Args:
            x_t: Noisy data
            t: Timestep indices
            condition: Optional conditioning vector

        Returns:
            Predicted noise
        """
        return self.diffusion(x_t, t, condition)

    def decode(self, representations: torch.Tensor) -> torch.Tensor:
        """Decode representations back to time series.

        For PhysDiff-Net, decoding happens through the diffusion
        sampling process rather than a separate decoder.

        Args:
            representations: Encoded representations

        Returns:
            Decoded time series (via diffusion sampling)
        """
        # In practice, use generate() for full decoding
        # This is a placeholder for direct decoding if needed
        raise NotImplementedError(
            "Use generate() for decoding via diffusion sampling"
        )

    def generate(
        self,
        num_samples: int,
        seq_length: int,
        scenario_params: torch.Tensor,
        reference_data: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.5,
    ) -> torch.Tensor:
        """Generate synthetic time series.

        Args:
            num_samples: Number of samples to generate
            seq_length: Length of sequences
            scenario_params: Scenario parameters of shape (num_samples, num_params)
            reference_data: Optional reference for encoder conditioning
            num_inference_steps: Denoising steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            Generated samples of shape (num_samples, seq_length, features)
        """
        device = next(self.parameters()).device

        # Embed scenario parameters
        scenario_embed = self.scenario_embedder(scenario_params)  # (batch, scenario_dim)

        # Get encoder representations if reference provided
        if reference_data is not None:
            encoder_repr = self.encode(reference_data, encoding_window="full_series")
            # Combine with scenario embedding
            condition = torch.cat([encoder_repr, scenario_embed], dim=-1)
        else:
            # Use zero encoder representation
            zero_repr = torch.zeros(
                num_samples, self.output_dims, device=device
            )
            condition = torch.cat([zero_repr, scenario_embed], dim=-1)

        # Generate via diffusion sampling
        shape = (num_samples, seq_length, self.input_dims)
        samples = self.diffusion.sample(
            scheduler=self.scheduler,
            shape=shape,
            condition=condition,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )

        return samples

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        scenario_params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute training loss.

        Args:
            batch: Batch dictionary with 'features' key
            scenario_params: Optional scenario parameters

        Returns:
            Tuple of (total_loss, dict of individual losses)
        """
        x_0 = batch["features"]  # (batch, seq, features)
        batch_size = x_0.shape[0]
        device = x_0.device

        losses = {}

        # Sample random timesteps
        t = torch.randint(
            0, self.scheduler.num_timesteps,
            (batch_size,),
            device=device,
        )

        # Add noise
        x_t, noise = self.diffuse(x_0, t)

        # Get conditioning
        if scenario_params is not None:
            encoder_repr = self.encode(x_0, encoding_window="full_series")
            scenario_embed = self.scenario_embedder(scenario_params)
            condition = torch.cat([encoder_repr, scenario_embed], dim=-1)
        else:
            condition = None

        # Predict noise
        noise_pred = self.denoise(x_t, t, condition)

        # Diffusion loss (noise prediction)
        diffusion_loss = nn.functional.mse_loss(noise_pred, noise)
        losses["diffusion"] = diffusion_loss

        # Physics losses on predicted clean data
        # Reconstruct x_0 estimate from noise prediction
        alpha_cumprod = self.scheduler.alphas_cumprod[t]
        while len(alpha_cumprod.shape) < len(x_t.shape):
            alpha_cumprod = alpha_cumprod.unsqueeze(-1)

        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)

        physics_loss, physics_breakdown = self.physics(x_0_pred)
        losses["physics"] = physics_loss
        losses.update({f"physics_{k}": v for k, v in physics_breakdown.items()})

        # Combine losses with weights
        loss_weights = self.config.get("training", {}).get("loss_weights", {})
        total_loss = (
            loss_weights.get("diffusion", 1.0) * diffusion_loss +
            loss_weights.get("physics", 0.1) * physics_loss
        )
        losses["total"] = total_loss

        return total_loss, losses

    def fit_encoder(
        self,
        train_data: Union[np.ndarray, torch.Tensor],
        n_epochs: int = 200,
        verbose: bool = True,
    ) -> "PhysDiffNet":
        """Pre-train the encoder on training data.

        Args:
            train_data: Training data
            n_epochs: Number of epochs
            verbose: Print progress

        Returns:
            Self for chaining
        """
        print("Pre-training TS2Vec encoder...")
        self.encoder.fit(
            train_data,
            n_epochs=n_epochs,
            verbose=verbose,
        )
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config,
            "model_state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

        # Save encoder separately (TS2Vec has its own format)
        encoder_path = path.parent / f"{path.stem}_encoder"
        self.encoder.save(encoder_path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: str = "cuda",
    ) -> "PhysDiffNet":
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load to

        Returns:
            Loaded model
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)

        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # Load encoder
        encoder_path = path.parent / f"{path.stem}_encoder"
        if encoder_path.exists():
            model.encoder.load(encoder_path)

        return model

    def to(self, device: Union[str, torch.device]) -> "PhysDiffNet":
        """Move model to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        super().to(device)
        self.device = str(device)

        # Update scheduler device
        self.scheduler = NoiseScheduler(
            num_timesteps=self.scheduler.num_timesteps,
            device=str(device),
        )

        return self
