"""
1D ResNet-based imputer for time-series missing value imputation.

Architecture justification (from Phase 1 literature review):
- Residual connections solve vanishing gradients in deep networks, allowing
  the model to learn complex nonlinear consumption patterns (He et al., 2016)
- 1D convolutions efficiently capture local temporal motifs — daily/weekly
  periodicity in smart meter data (DRes-CNN literature, 2024-2025)
- Dilated convolutions expand the receptive field without increasing parameters,
  capturing both short-term (hours) and medium-term (days) dependencies
- Parallel processing (unlike sequential RNNs) enables faster training
- Missingness mask and time features are input channels, similar to
  how GRU-D uses mask as input

Architecture:
    Conv1D stem (kernel=7) → 4 × ResBlock (with dilation) → Conv1D projection → output
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ResBlock1D(nn.Module):
    """
    Residual block for 1D time-series data.

    Two convolutional layers with BatchNorm and ReLU, plus a skip connection.
    If in_channels != out_channels, a 1×1 conv adjusts the skip.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # same padding

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection projection if dimensions change
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            out: (batch, out_channels, seq_len)
        """
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity  # residual connection
        out = self.relu(out)
        return out


class ResNetImputer(nn.Module):
    """
    1D ResNet for time-series imputation.

    Input:  (batch, window_size, input_features)
    Output: (batch, window_size) — reconstructed kWh values for the full window

    Architecture:
        - Stem: Conv1d(input_features → 64, kernel=7)
        - 4 Residual Blocks with increasing then decreasing channels and dilation
        - Projection: Conv1d(last_channels → 1, kernel=1)
    """

    def __init__(
        self,
        input_size: int = config.INPUT_FEATURES,
        channels: list = None,
        kernel_size: int = config.RESNET_KERNEL_SIZE,
        stem_kernel: int = config.RESNET_STEM_KERNEL,
        dilations: list = None,
    ):
        super().__init__()
        self.model_name = "ResNet"

        if channels is None:
            channels = config.RESNET_CHANNELS  # [64, 128, 256, 128]
        if dilations is None:
            dilations = config.RESNET_DILATIONS  # [1, 2, 4, 1]

        # Stem convolution
        stem_padding = (stem_kernel - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, channels[0], stem_kernel, padding=stem_padding, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        blocks = []
        in_ch = channels[0]
        for out_ch, dil in zip(channels, dilations):
            blocks.append(ResBlock1D(in_ch, out_ch, kernel_size, dil))
            in_ch = out_ch
        self.res_blocks = nn.Sequential(*blocks)

        # Projection to single output channel
        self.projection = nn.Conv1d(channels[-1], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, input_features)
        Returns:
            out: (batch, window_size) — predicted kWh values
        """
        # Conv1D expects (batch, channels, seq_len), so transpose
        x = x.transpose(1, 2)  # (batch, input_features, window_size)

        x = self.stem(x)            # (batch, 64, window_size)
        x = self.res_blocks(x)      # (batch, 128, window_size)
        x = self.projection(x)      # (batch, 1, window_size)

        return x.squeeze(1)          # (batch, window_size)
