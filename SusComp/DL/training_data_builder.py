import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict
import pandas as pd


def time_features_from_datetime_parts(hour: torch.Tensor, dow: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
    """
    hour: (N,) int tensor [0..23]
    dow:  (N,) int tensor [0..6]
    doy:  (N,) int tensor [1..366]
    returns: (N, 6) float tensor [sin/cos hour, sin/cos dow, sin/cos doy]
    """
    hour = hour.float()
    dow = dow.float()
    doy = doy.float()

    two_pi = 2.0 * math.pi

    h_sin = torch.sin(two_pi * hour / 24.0)
    h_cos = torch.cos(two_pi * hour / 24.0)

    d_sin = torch.sin(two_pi * dow / 7.0)
    d_cos = torch.cos(two_pi * dow / 7.0)

    y_sin = torch.sin(two_pi * (doy - 1.0) / 365.25)
    y_cos = torch.cos(two_pi * (doy - 1.0) / 365.25)

    return torch.stack([h_sin, h_cos, d_sin, d_cos, y_sin, y_cos], dim=-1)


class WindowedForecastDataset(Dataset):
    """
    Each sample:
      x: (lookback_steps, F_num + F_time)
      y: (horizon_steps,)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        lookback_steps: int,
        horizon_steps: int,
        start_t: int,
        end_t: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        scale_y: bool = False,
        y_mean: float = 0.0,
        y_std: float = 1.0,
    ):
        assert isinstance(df.index, pd.DatetimeIndex), "df.index must be a DatetimeIndex"
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback_steps = lookback_steps
        self.horizon_steps = horizon_steps
        self.start_t = start_t
        self.end_t = end_t

        self.x_mean = x_mean.astype(np.float32)
        self.x_std = np.where(x_std == 0, 1.0, x_std).astype(np.float32)

        self.scale_y = scale_y
        self.y_mean = float(y_mean)
        self.y_std = float(y_std if y_std != 0 else 1.0)

        # valid forecast start indices t in [start_t, end_t)
        self.ts = np.arange(start_t, end_t, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = int(self.ts[idx])

        # Input window: [t-lookback, t)
        x_window = self.df.iloc[t - self.lookback_steps : t]
        x_num = x_window[self.feature_cols].to_numpy(dtype=np.float32)  # (L, F_num)
        x_num = (x_num - self.x_mean) / self.x_std

        # Time features from index
        dt_index = x_window.index
        hour = torch.from_numpy(dt_index.hour.to_numpy(dtype=np.int64))
        dow = torch.from_numpy(dt_index.dayofweek.to_numpy(dtype=np.int64))
        doy = torch.from_numpy(dt_index.dayofyear.to_numpy(dtype=np.int64))
        x_time = time_features_from_datetime_parts(hour, dow, doy).numpy().astype(np.float32)  # (L, 6)

        x = np.concatenate([x_num, x_time], axis=-1)  # (L, F_num+6)

        # Target horizon: [t, t+horizon)
        y = self.df.iloc[t : t + self.horizon_steps][self.target_col].to_numpy(dtype=np.float32)  # (H,)
        if self.scale_y:
            y = (y - self.y_mean) / self.y_std

        return torch.from_numpy(x), torch.from_numpy(y)



#Utilities: frequency, splits, scaling
def infer_base_freq(index: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = index.to_series().diff().dropna()
    if len(diffs) == 0:
        raise ValueError("Need at least 2 timestamps to infer frequency.")
    # robust: median difference
    return diffs.median()


def train_val_test_split_indices(
    n: int,
    lookback_steps: int,
    horizon_steps: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Dict[str, Tuple[int, int]]:
    """
    Split by forecast start index t (chronological).
    We create ranges of t such that y=[t, t+H) is inside data length.
    """
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1

    # valid t must satisfy: t-lookback >= 0 and t+horizon <= n
    t_min = lookback_steps
    t_max_excl = n - horizon_steps
    if t_max_excl <= t_min:
        raise ValueError("Not enough data for the requested lookback/horizon.")

    valid_count = t_max_excl - t_min
    train_end = t_min + int(valid_count * train_frac)
    val_end = train_end + int(valid_count * val_frac)

    # (start_t, end_t) ranges for forecast start times
    splits = {
        "train": (t_min, train_end),
        "val": (train_end, val_end),
        "test": (val_end, t_max_excl),
    }
    return splits


def compute_scalers(df: pd.DataFrame, feature_cols: list[str], target_col: str, train_t_range: Tuple[int, int],
                    lookback_steps: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute X scaler on the *training history* up to end of train range.
    We use data up to train_end (exclusive). This avoids leaking val/test stats.
    """
    train_start_t, train_end_t = train_t_range

    # Use feature rows that appear in training windows:
    # windows cover indices [t-lookback, t), so overall training history spans
    # [train_start_t - lookback, train_end_t)
    hist_start = max(0, train_start_t - lookback_steps)
    hist_end = train_end_t

    x_hist = df.iloc[hist_start:hist_end][feature_cols].to_numpy(dtype=np.float32)
    x_mean = x_hist.mean(axis=0)
    x_std = x_hist.std(axis=0) + 1e-6

    y_hist = df.iloc[hist_start:hist_end][target_col].to_numpy(dtype=np.float32)
    y_mean = float(y_hist.mean())
    y_std = float(y_hist.std() + 1e-6)

    return x_mean, x_std, y_mean, y_std

