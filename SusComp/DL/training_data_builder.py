import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict
import pandas as pd

############ TIME FEATURES #############
########################################
def time_features_from_datetime_parts(hour: torch.Tensor, dow: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
    """
    We encode the full date+time to a continuous numeric value so the model can handle it better
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


############ WEATHER FEATURES ##############
############################################


########## TRAINING BATCH BUILDER ##############
################################################
class WindowedForecastDataset(Dataset):
    """
    Each training sample:
      x: (lookback_steps, num_features + 6 time-features)
      y: (horizon_steps,)

    `ts` contains the forecast start indices (iloc positions) you want to use.
    If you want "daily at 03:00 forecast start", build `ts` that way in your split
    (anchored_train_val_test_split_indices) and pass it here.
    """
    def __init__(
            self,
            df: pd.DataFrame,
            feature_cols: list[str],
            target_col: str,
            lookback_steps: int,
            horizon_steps: int,
            ts: np.ndarray,
            x_mean: np.ndarray,
            x_std: np.ndarray,
            scale_y: bool = False,
            y_mean: float = 0.0,
            y_std: float = 1.0,
            # optional safety check:
            anchor_hour: int | None = None,
            anchor_minute: int = 0,
    ):
        assert isinstance(df.index, pd.DatetimeIndex), "df.index must be a DatetimeIndex"

        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback_steps = int(lookback_steps)
        self.horizon_steps = int(horizon_steps)

        self.x_mean = x_mean.astype(np.float32)
        self.x_std = np.where(x_std == 0, 1.0, x_std).astype(np.float32)

        self.scale_y = bool(scale_y)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std if y_std != 0 else 1.0)

        ts = np.asarray(ts, dtype=np.int64)
        if ts.size == 0:
            raise ValueError("Empty ts passed to dataset.")

        # ---- validate and filter ts so windows are always valid ----
        # valid t must satisfy: t-lookback >= 0 and t+horizon <= len(df)
        n = len(df)
        valid_mask = (ts - self.lookback_steps >= 0) & (ts + self.horizon_steps <= n)
        ts_valid = ts[valid_mask]

        if ts_valid.size == 0:
            raise ValueError(
                "After applying lookback/horizon constraints, no valid ts remain. "
                f"(lookback_steps={self.lookback_steps}, horizon_steps={self.horizon_steps}, n={n})"
            )

        # Optional: ensure ts is anchored (e.g. hour==3)
        if anchor_hour is not None:
            idx = df.index[ts_valid]
            anchor_mask = (idx.hour == anchor_hour) & (idx.minute == anchor_minute)
            ts_valid = ts_valid[anchor_mask]
            if ts_valid.size == 0:
                raise ValueError(
                    f"After applying anchor constraint, no valid ts remain for "
                    f"{anchor_hour:02d}:{anchor_minute:02d}."
                )

        self.ts = ts_valid

    def __len__(self) -> int:
        return len(self.ts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = int(self.ts[idx])

        # Input window: [t-lookback, t)
        x_window = self.df.iloc[t - self.lookback_steps : t]

        x_num = x_window[self.feature_cols].to_numpy(dtype=np.float32)  # (L, F)
        x_num = (x_num - self.x_mean) / self.x_std

        # Time features from index
        dt_index = x_window.index
        hour = torch.from_numpy(dt_index.hour.to_numpy(dtype=np.int64))
        dow = torch.from_numpy(dt_index.dayofweek.to_numpy(dtype=np.int64))
        doy = torch.from_numpy(dt_index.dayofyear.to_numpy(dtype=np.int64))
        x_time = time_features_from_datetime_parts(hour, dow, doy).numpy().astype(np.float32)  # (L, 6)

        x = np.concatenate([x_num, x_time], axis=-1)  # (L, F+6)

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


def anchored_train_val_test_split_indices(
        df: pd.DataFrame,
        lookback_steps: int,
        horizon_steps: int,
        anchor_hour: int = 3,
        anchor_minute: int = 0,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
):
    n = len(df)
    t_min = lookback_steps
    t_max_excl = n - horizon_steps
    if t_max_excl <= t_min:
        raise ValueError("Not enough data for the requested lookback/horizon.")

    cand = np.arange(t_min, t_max_excl, dtype=np.int64)
    idx = df.index[cand]
    mask = (idx.hour == anchor_hour) & (idx.minute == anchor_minute)
    ts = cand[mask]  # all valid daily forecast start indices

    if len(ts) < 10:
        raise ValueError(f"Too few anchored samples: {len(ts)}. Check anchor time and data frequency.")

    n_valid = len(ts)
    train_end = int(n_valid * train_frac)
    val_end = train_end + int(n_valid * val_frac)

    return {
        "train": ts[:train_end],
        "val": ts[train_end:val_end],
        "test": ts[val_end:],
    }


def compute_scalers_from_ts(
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        train_ts: np.ndarray,          # anchored forecast start indices (iloc positions)
        lookback_steps: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute scalers using ONLY the portion of the series that is seen by training windows.

    Training windows are built from forecast starts t in train_ts.
    Each window uses x rows [t-lookback, t), so the union of all training input rows is:
        [min(train_ts) - lookback_steps, max(train_ts))
    (end is exclusive, consistent with iloc slicing).

    This avoids leaking validation/test statistics.
    """
    train_ts = np.asarray(train_ts, dtype=np.int64)
    if train_ts.size == 0:
        raise ValueError("train_ts is empty; cannot compute scalers.")

    train_start_t = int(train_ts.min())
    train_end_t_excl = int(train_ts.max())  # exclusive end for history slice is max(t)

    hist_start = max(0, train_start_t - lookback_steps)
    hist_end = train_end_t_excl

    if hist_end <= hist_start:
        raise ValueError(f"Invalid scaler slice: hist_start={hist_start}, hist_end={hist_end}")

    x_hist = df.iloc[hist_start:hist_end][feature_cols].to_numpy(dtype=np.float32)
    x_mean = x_hist.mean(axis=0)
    x_std = x_hist.std(axis=0) + 1e-6

    y_hist = df.iloc[hist_start:hist_end][target_col].to_numpy(dtype=np.float32)
    y_mean = float(y_hist.mean())
    y_std = float(y_hist.std() + 1e-6)

    return x_mean, x_std, y_mean, y_std