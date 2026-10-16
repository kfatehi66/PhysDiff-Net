"""TS2Vec encoder wrapper for time series representation learning.

This module provides a wrapper around the TS2Vec self-supervised time series
encoder. The TS2Vec implementation should be cloned from:
https://github.com/yuezhihan/ts2vec

into the external/ts2vec/ directory.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class TS2VecWrapper(nn.Module):
    """Wrapper for TS2Vec time series encoder.

    Handles loading the TS2Vec model from the external directory
    and provides a consistent interface for the PhysDiff-Net pipeline.

    The TS2Vec model learns hierarchical representations through
    contrastive learning at different scales.

    Attributes:
        input_dims: Number of input features
        output_dims: Dimension of output representations
        hidden_dims: Hidden layer dimension
        depth: Number of encoder layers
        model: The underlying TS2Vec model
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        device: str = "cuda",
        batch_size: int = 16,
        ts2vec_path: Optional[str] = None,
    ):
        """Initialize the TS2Vec wrapper.

        Args:
            input_dims: Number of input features
            output_dims: Dimension of output representations
            hidden_dims: Hidden layer dimension
            depth: Number of encoder layers
            device: Device to use ('cuda' or 'cpu')
            batch_size: Batch size for training and encoding
            ts2vec_path: Path to TS2Vec repo (default: external/ts2vec)
        """
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.device = device
        self.batch_size = batch_size

        # Setup path to TS2Vec
        if ts2vec_path is None:
            # Default to external/ts2vec relative to project root
            project_root = Path(__file__).parent.parent.parent
            ts2vec_path = project_root / "external" / "ts2vec"
        else:
            ts2vec_path = Path(ts2vec_path)

        self.ts2vec_path = ts2vec_path
        self.model = None
        self._ts2vec_available = False

        # Try to load TS2Vec
        self._setup_ts2vec()

    def _setup_ts2vec(self) -> None:
        """Setup TS2Vec by adding to path and importing."""
        if not self.ts2vec_path.exists():
            print(f"Warning: TS2Vec not found at {self.ts2vec_path}")
            print("Please clone: git clone https://github.com/yuezhihan/ts2vec.git external/ts2vec")
            self._create_fallback_encoder()
            return

        # Add TS2Vec to Python path
        ts2vec_str = str(self.ts2vec_path)
        if ts2vec_str not in sys.path:
            sys.path.insert(0, ts2vec_str)

        try:
            from ts2vec import TS2Vec

            self.model = TS2Vec(
                input_dims=self.input_dims,
                output_dims=self.output_dims,
                hidden_dims=self.hidden_dims,
                depth=self.depth,
                device=self.device,
                batch_size=self.batch_size,
            )
            self._ts2vec_available = True
            print(f"TS2Vec loaded successfully from {self.ts2vec_path}")

        except ImportError as e:
            print(f"Warning: Could not import TS2Vec: {e}")
            self._create_fallback_encoder()

    def _create_fallback_encoder(self) -> None:
        """Create a simple fallback encoder when TS2Vec is not available."""
        print("Using fallback TCN encoder")

        self.model = FallbackTCNEncoder(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
        )
        self._ts2vec_available = False

    def fit(
        self,
        train_data: Union[np.ndarray, torch.Tensor],
        n_epochs: int = 200,
        verbose: bool = True,
    ) -> "TS2VecWrapper":
        """Fit the encoder on training data.

        Args:
            train_data: Training data of shape (n_samples, seq_length, n_features)
            n_epochs: Number of training epochs
            verbose: Whether to print training progress

        Returns:
            Self for chaining
        """
        if isinstance(train_data, torch.Tensor):
            train_data = train_data.cpu().numpy()

        if self._ts2vec_available:
            # Note: TS2Vec uses batch_size from initialization, not fit()
            self.model.fit(
                train_data,
                n_epochs=n_epochs,
                verbose=verbose,
            )
        else:
            # Fallback training
            self.model.fit(train_data, n_epochs=n_epochs, batch_size=self.batch_size)

        return self

    def encode(
        self,
        data: Union[np.ndarray, torch.Tensor],
        encoding_window: Optional[str] = None,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Encode time series data to representations.

        Args:
            data: Input data of shape (n_samples, seq_length, n_features)
            encoding_window: 'full_series' for single vector per series,
                           None for per-timestamp representations
            batch_size: Batch size for encoding

        Returns:
            Encoded representations
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if self._ts2vec_available:
            representations = self.model.encode(
                data,
                encoding_window=encoding_window,
                batch_size=batch_size,
            )
        else:
            representations = self.model.encode(data)

        return torch.tensor(representations, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for integration with PyTorch training.

        Args:
            x: Input tensor of shape (batch, seq_length, n_features)

        Returns:
            Encoded representations
        """
        if self._ts2vec_available:
            # TS2Vec expects numpy, we need to handle this carefully
            x_np = x.detach().cpu().numpy()
            representations = self.model.encode(x_np)
            return torch.tensor(representations, dtype=torch.float32, device=x.device)
        else:
            return self.model(x)

    def save(self, path: Union[str, Path]) -> None:
        """Save the encoder state.

        Args:
            path: Path to save the encoder
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._ts2vec_available:
            self.model.save(str(path))
        else:
            torch.save(self.model.state_dict(), path)

    def load(self, path: Union[str, Path]) -> "TS2VecWrapper":
        """Load encoder state.

        Args:
            path: Path to load from

        Returns:
            Self for chaining
        """
        path = Path(path)

        if self._ts2vec_available:
            self.model.load(str(path))
        else:
            self.model.load_state_dict(torch.load(path))

        return self


class FallbackTCNEncoder(nn.Module):
    """Fallback TCN-based encoder when TS2Vec is not available.

    A simplified temporal convolutional network for time series encoding.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
    ):
        """Initialize the fallback encoder.

        Args:
            input_dims: Number of input features
            output_dims: Dimension of output representations
            hidden_dims: Hidden layer dimension
            depth: Number of convolutional layers
        """
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims

        # Input projection
        self.input_proj = nn.Linear(input_dims, hidden_dims)

        # TCN layers with dilated convolutions
        self.tcn_layers = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** (i % 4)  # Cycle through dilations
            self.tcn_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dims, hidden_dims,
                        kernel_size=3, padding=dilation, dilation=dilation
                    ),
                    nn.BatchNorm1d(hidden_dims),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dims, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_length, n_features)

        Returns:
            Encoded representations of shape (batch, seq_length, output_dims)
        """
        # Project input
        x = self.input_proj(x)  # (batch, seq, hidden)

        # Transpose for conv1d: (batch, hidden, seq)
        x = x.transpose(1, 2)

        # Apply TCN layers with residual connections
        for layer in self.tcn_layers:
            x = x + layer(x)

        # Transpose back: (batch, seq, hidden)
        x = x.transpose(1, 2)

        # Project to output dimension
        x = self.output_proj(x)

        return x

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode numpy data.

        Args:
            data: Input data of shape (n_samples, seq_length, n_features)

        Returns:
            Encoded representations
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(data, dtype=torch.float32)
            representations = self.forward(x)
            return representations.numpy()

    def fit(
        self,
        train_data: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 16,
    ) -> None:
        """Simple self-supervised training (reconstruction-based).

        Args:
            train_data: Training data
            n_epochs: Number of epochs
            batch_size: Batch size
        """
        # Simple reconstruction-based training as fallback
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_data, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Decoder for reconstruction
        decoder = nn.Linear(self.output_dims, self.input_dims)

        for epoch in range(n_epochs):
            total_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()
                encoded = self.forward(batch)
                reconstructed = decoder(encoded)
                loss = nn.functional.mse_loss(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(loader):.4f}")
