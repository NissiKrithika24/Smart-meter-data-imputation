"""
Bidirectional LSTM-based imputer for time-series missing value imputation.

Design choices informed by literature:
- Bidirectional processing captures both past and future context (inspired by BRITS)
- Missingness mask as an explicit input feature (inspired by GRU-D)
- Sequence-to-sequence architecture: full window in → full window out
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LSTMImputer(nn.Module):
    """
    Bidirectional LSTM for time-series imputation.

    Input:  (batch, window_size, input_features)
    Output: (batch, window_size) — reconstructed kWh values for the full window
    """

    def __init__(
        self,
        input_size: int = config.INPUT_FEATURES,
        hidden_size: int = config.RNN_HIDDEN_SIZE,
        num_layers: int = config.RNN_NUM_LAYERS,
        dropout: float = config.RNN_DROPOUT,
    ):
        super().__init__()
        self.model_name = "LSTM"

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bidirectional → output is 2 * hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window_size, input_features)
        Returns:
            out: (batch, window_size) — predicted kWh values
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 2*hidden)
        out = self.fc(lstm_out)     # (batch, seq_len, 1)
        return out.squeeze(-1)      # (batch, seq_len)
