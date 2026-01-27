import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMMultiHorizon(nn.Module):
    """(B,T,F) -> (B,H)"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(cfg.hidden_size) if cfg.use_layernorm else nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.head_hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_size, cfg.horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]          # (B, hidden)
        h_last = self.norm(h_last)
        return self.head(h_last)  # (B, H)

