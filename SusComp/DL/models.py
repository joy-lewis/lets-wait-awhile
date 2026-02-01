import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMMultiHorizon(nn.Module):
    """(B,T,F) -> (B,H)"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.hist_input_size,
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


#
# class LSTMMultiHorizonWithFuture(nn.Module):
#     """
#     Inputs:
#       x_hist: (B, L, F_hist)
#       x_fut:  (B, H, F_fut)   # forecast covariates aligned to horizon steps
#     Output:
#       y_hat:  (B, H)
#     """
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#
#         self.lstm = nn.LSTM(
#             input_size=cfg.hist_input_size,    # <-- only history features
#             hidden_size=cfg.hidden_size,
#             num_layers=cfg.num_layers,
#             dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
#             batch_first=True,
#         )
#
#         self.norm = nn.LayerNorm(cfg.hidden_size) if cfg.use_layernorm else nn.Identity()
#
#         # Per-horizon MLP that maps [h_last, x_fut_k] -> y_k
#         in_dim = cfg.hidden_size + cfg.fut_input_size
#         self.fusion = nn.Sequential(
#             nn.Linear(in_dim, cfg.head_hidden_size),
#             nn.ReLU(),
#             nn.Dropout(cfg.head_dropout),
#             nn.Linear(cfg.head_hidden_size, 1),
#         )
#
#     def forward(self, x_hist: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
#         # x_hist: (B, L, F_hist)
#         # x_fut:  (B, H, F_fut)
#         _, (h_n, _) = self.lstm(x_hist)
#         h_last = h_n[-1]                 # (B, hidden)
#         h_last = self.norm(h_last)
#
#         B, H, Ff = x_fut.shape
#         h_rep = h_last.unsqueeze(1).expand(B, H, h_last.size(-1))  # (B, H, hidden)
#
#         z = torch.cat([h_rep, x_fut], dim=-1)  # (B, H, hidden + F_fut)
#         y_hat = self.fusion(z).squeeze(-1)     # (B, H)
#         return y_hat