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
# class WindowedForecastDataset(Dataset):
#     """
#     Each training sample:
#       x: (lookback_steps, num_features + 6 time-features)
#       y: (horizon_steps,)
#
#     `ts` contains the forecast start indices (iloc positions) you want to use.
#     If you want "daily at 03:00 forecast start", build `ts` that way in your split
#     (anchored_train_val_test_split_indices) and pass it here.
#     """
#     def __init__(
#             self,
#             df: pd.DataFrame,
#             feature_cols: list[str],
#             target_col: str,
#             lookback_steps: int,
#             horizon_steps: int,
#             ts: np.ndarray,
#             x_mean: np.ndarray,
#             x_std: np.ndarray,
#             scale_y: bool = False,
#             y_mean: float = 0.0,
#             y_std: float = 1.0,
#             # optional safety check:
#             anchor_hour: int | None = None,
#             anchor_minute: int = 0,
#     ):
#         assert isinstance(df.index, pd.DatetimeIndex), "df.index must be a DatetimeIndex"
#
#         self.df = df
#         self.feature_cols = feature_cols
#         self.target_col = target_col
#         self.lookback_steps = int(lookback_steps)
#         self.horizon_steps = int(horizon_steps)
#
#         self.x_mean = x_mean.astype(np.float32)
#         self.x_std = np.where(x_std == 0, 1.0, x_std).astype(np.float32)
#
#         self.scale_y = bool(scale_y)
#         self.y_mean = float(y_mean)
#         self.y_std = float(y_std if y_std != 0 else 1.0)
#
#         ts = np.asarray(ts, dtype=np.int64)
#         if ts.size == 0:
#             raise ValueError("Empty ts passed to dataset.")
#
#         # ---- validate and filter ts so windows are always valid ----
#         # valid t must satisfy: t-lookback >= 0 and t+horizon <= len(df)
#         n = len(df)
#         valid_mask = (ts - self.lookback_steps >= 0) & (ts + self.horizon_steps <= n)
#         ts_valid = ts[valid_mask]
#
#         if ts_valid.size == 0:
#             raise ValueError(
#                 "After applying lookback/horizon constraints, no valid ts remain. "
#                 f"(lookback_steps={self.lookback_steps}, horizon_steps={self.horizon_steps}, n={n})"
#             )
#
#         # Optional: ensure ts is anchored (e.g. hour==3)
#         if anchor_hour is not None:
#             idx = df.index[ts_valid]
#             anchor_mask = (idx.hour == anchor_hour) & (idx.minute == anchor_minute)
#             ts_valid = ts_valid[anchor_mask]
#             if ts_valid.size == 0:
#                 raise ValueError(
#                     f"After applying anchor constraint, no valid ts remain for "
#                     f"{anchor_hour:02d}:{anchor_minute:02d}."
#                 )
#
#         self.ts = ts_valid
#
#     def __len__(self) -> int:
#         return len(self.ts)
#
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         t = int(self.ts[idx])
#
#         # Input window: [t-lookback, t)
#         x_window = self.df.iloc[t - self.lookback_steps : t]
#
#         x_num = x_window[self.feature_cols].to_numpy(dtype=np.float32)  # (L, F)
#         x_num = (x_num - self.x_mean) / self.x_std
#
#         # Time features from index
#         dt_index = x_window.index
#         hour = torch.from_numpy(dt_index.hour.to_numpy(dtype=np.int64))
#         dow = torch.from_numpy(dt_index.dayofweek.to_numpy(dtype=np.int64))
#         doy = torch.from_numpy(dt_index.dayofyear.to_numpy(dtype=np.int64))
#         x_time = time_features_from_datetime_parts(hour, dow, doy).numpy().astype(np.float32)  # (L, 6)
#
#         x = np.concatenate([x_num, x_time], axis=-1)  # (L, F+6)
#
#         # Target horizon: [t, t+horizon)
#         y = self.df.iloc[t : t + self.horizon_steps][self.target_col].to_numpy(dtype=np.float32)  # (H,)
#         if self.scale_y:
#             y = (y - self.y_mean) / self.y_std
#
#         return torch.from_numpy(x), torch.from_numpy(y)
#


from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class WindowedForecastDatasetWithFuture(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            hist_feature_cols: list[str],
            fut_feature_cols: list[str],
            target_col: str,
            lookback_steps: int,
            horizon_steps: int,
            ts: np.ndarray,
            x_hist_mean: np.ndarray,
            x_hist_std: np.ndarray,
            x_fut_mean: np.ndarray,
            x_fut_std: np.ndarray,
            scale_y: bool = False,
            y_mean: float = 0.0,
            y_std: float = 1.0,
            anchor_hour: int | None = 3,
            anchor_minute: int = 0,
            add_time_features_to_future: bool = True,
            zero_future: bool = False,   # <- recommend default False
    ):
        assert isinstance(df.index, pd.DatetimeIndex), "df.index must be a DatetimeIndex"
        self.df = df
        self.hist_feature_cols = list(hist_feature_cols)
        self.fut_feature_cols = list(fut_feature_cols)
        self.target_col = target_col
        self.lookback_steps = int(lookback_steps)
        self.horizon_steps = int(horizon_steps)

        ts = np.asarray(ts, dtype=np.int64)
        if ts.size == 0:
            raise ValueError("Empty ts passed to dataset.")

        n = len(df)
        valid = (ts - self.lookback_steps >= 0) & (ts + self.horizon_steps <= n)
        ts = ts[valid]
        if ts.size == 0:
            raise ValueError("No valid ts after lookback/horizon filtering.")

        if anchor_hour is not None:
            idx = df.index[ts]
            mask = (idx.hour == anchor_hour) & (idx.minute == anchor_minute)
            ts = ts[mask]
            if ts.size == 0:
                raise ValueError("No ts match the anchor time after filtering.")

        self.ts = ts

        self.x_hist_mean = x_hist_mean.astype(np.float32)
        self.x_hist_std = np.where(x_hist_std == 0, 1.0, x_hist_std).astype(np.float32)

        self.x_fut_mean = x_fut_mean.astype(np.float32)
        self.x_fut_std = np.where(x_fut_std == 0, 1.0, x_fut_std).astype(np.float32)

        self.scale_y = bool(scale_y)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std if y_std != 0 else 1.0)

        self.zero_future = bool(zero_future)
        self.add_time_features_to_future = bool(add_time_features_to_future)

        self.fut_feat_dim = len(self.fut_feature_cols) + (6 if self.add_time_features_to_future else 0)

    def __len__(self) -> int:
        return len(self.ts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.ts[idx])

        # ----- history -----
        hist = self.df.iloc[t - self.lookback_steps : t]
        xh_num = hist[self.hist_feature_cols].to_numpy(dtype=np.float32)
        xh_num = (xh_num - self.x_hist_mean) / self.x_hist_std

        dt = hist.index
        hour = torch.from_numpy(dt.hour.to_numpy(dtype=np.int64))
        dow  = torch.from_numpy(dt.dayofweek.to_numpy(dtype=np.int64))
        doy  = torch.from_numpy(dt.dayofyear.to_numpy(dtype=np.int64))
        xh_time = time_features_from_datetime_parts(hour, dow, doy).numpy().astype(np.float32)

        x_hist = np.concatenate([xh_num, xh_time], axis=-1)

        # ----- future covariates -----
        if self.zero_future:
            x_fut = np.zeros((self.horizon_steps, self.fut_feat_dim), dtype=np.float32)
            fut_target = self.df.iloc[t : t + self.horizon_steps]  # for y only
        else:
            fut_target = self.df.iloc[t : t + self.horizon_steps]
            xf_num = fut_target[self.fut_feature_cols].to_numpy(dtype=np.float32)
            xf_num = (xf_num - self.x_fut_mean) / self.x_fut_std

            if self.add_time_features_to_future:
                dtf = fut_target.index
                hour_f = torch.from_numpy(dtf.hour.to_numpy(dtype=np.int64))
                dow_f  = torch.from_numpy(dtf.dayofweek.to_numpy(dtype=np.int64))
                doy_f  = torch.from_numpy(dtf.dayofyear.to_numpy(dtype=np.int64))
                xf_time = time_features_from_datetime_parts(hour_f, dow_f, doy_f).numpy().astype(np.float32)
                x_fut = np.concatenate([xf_num, xf_time], axis=-1)
            else:
                x_fut = xf_num

        # ----- target -----
        y = fut_target[self.target_col].to_numpy(dtype=np.float32)
        if self.scale_y:
            y = (y - self.y_mean) / self.y_std

        return torch.from_numpy(x_hist), torch.from_numpy(x_fut), torch.from_numpy(y)

#Utilities: frequency, splits, scaling
def infer_base_freq(index: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = index.to_series().diff().dropna()
    if len(diffs) == 0:
        raise ValueError("Need at least 2 timestamps to infer frequency.")
    # robust: median difference
    return diffs.median()


## Train: 2021–2023 + Jan 1, 2024 00:00 through Jun 30, 2024 23:59:59
## Val: Jul 1, 2024 00:00 through Dec 31, 2024 23:59:59
## Test: all of 2025

def anchored_train_val_test_split_indices(
    df: pd.DataFrame,
    lookback_steps: int,
    horizon_steps: int,
    anchor_hour: int = 3,
    anchor_minute: int = 0,
):
    # Ensure datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        t = df.index
    else:
        t = pd.to_datetime(df.index, errors="coerce")
        if pd.isna(t).any():
            bad = int(pd.isna(t).sum())
            raise ValueError(f"{bad} index values could not be parsed as datetimes.")
        df = df.copy()
        df.index = pd.DatetimeIndex(t)
        t = df.index

    n = len(df)
    t_min = lookback_steps
    t_max_excl = n - horizon_steps
    if t_max_excl <= t_min:
        raise ValueError("Not enough data for the requested lookback/horizon.")

    cand = np.arange(t_min, t_max_excl, dtype=np.int64)
    tt = t[cand]  # DatetimeIndex slice at candidate starts

    anchored = (tt.hour == anchor_hour) & (tt.minute == anchor_minute)

    train_mask = (
        (tt.year >= 2021) & (tt.year <= 2023) |
        ((tt.year == 2024) & (tt.month <= 6))
    )
    val_mask = (tt.year == 2024) & (tt.month >= 7)
    test_mask = (tt.year == 2025)

    train_idx = cand[anchored & train_mask]
    val_idx   = cand[anchored & val_mask]
    test_idx  = cand[anchored & test_mask]

    if len(train_idx) < 1 or len(val_idx) < 1 or len(test_idx) < 1:
        raise ValueError(
            f"Split produced too few samples: "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}. "
            f"Check anchor time, data coverage, and frequency."
        )

    return {"train": train_idx, "val": val_idx, "test": test_idx}

# def compute_scalers_from_ts(
#         df: pd.DataFrame,
#         feature_cols: list[str],
#         target_col: str,
#         train_ts: np.ndarray,          # anchored forecast start indices (iloc positions)
#         lookback_steps: int,
# ) -> Tuple[np.ndarray, np.ndarray, float, float]:
#     """
#     Compute scalers using ONLY the portion of the series that is seen by training windows.
#
#     Training windows are built from forecast starts t in train_ts.
#     Each window uses x rows [t-lookback, t), so the union of all training input rows is:
#         [min(train_ts) - lookback_steps, max(train_ts))
#     (end is exclusive, consistent with iloc slicing).
#
#     This avoids leaking validation/test statistics.
#     """
#     train_ts = np.asarray(train_ts, dtype=np.int64)
#     if train_ts.size == 0:
#         raise ValueError("train_ts is empty; cannot compute scalers.")
#
#     train_start_t = int(train_ts.min())
#     train_end_t_excl = int(train_ts.max())  # exclusive end for history slice is max(t)
#
#     hist_start = max(0, train_start_t - lookback_steps)
#     hist_end = train_end_t_excl
#
#     if hist_end <= hist_start:
#         raise ValueError(f"Invalid scaler slice: hist_start={hist_start}, hist_end={hist_end}")
#
#     x_hist = df.iloc[hist_start:hist_end][feature_cols].to_numpy(dtype=np.float32)
#     x_mean = x_hist.mean(axis=0)
#     x_std = x_hist.std(axis=0) + 1e-6
#
#     y_hist = df.iloc[hist_start:hist_end][target_col].to_numpy(dtype=np.float32)
#     y_mean = float(y_hist.mean())
#     y_std = float(y_hist.std() + 1e-6)
#
#     return x_mean, x_std, y_mean, y_std

from typing import Tuple
import numpy as np
import pandas as pd

def compute_scalers_hist_and_fut(
        df: pd.DataFrame,
        hist_feature_cols: list[str],
        fut_feature_cols: list[str],
        target_col: str,
        train_ts: np.ndarray,
        lookback_steps: int,
        horizon_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    train_ts = np.asarray(train_ts, dtype=np.int64)
    if train_ts.size == 0:
        raise ValueError("train_ts empty")

    t0 = int(train_ts.min())
    t1 = int(train_ts.max())

    # history rows used: [t-lookback, t)
    hist_start = max(0, t0 - lookback_steps)
    hist_end   = t1  # exclusive

    # future rows used: [t, t+horizon)
    fut_start = t0
    fut_end   = min(len(df), t1 + horizon_steps)

    xh = df.iloc[hist_start:hist_end][hist_feature_cols].to_numpy(np.float32)
    xh_mean = xh.mean(axis=0)
    xh_std  = xh.std(axis=0) + 1e-6

    xf = df.iloc[fut_start:fut_end][fut_feature_cols].to_numpy(np.float32)
    xf_mean = xf.mean(axis=0)
    xf_std  = xf.std(axis=0) + 1e-6

    y_hist = df.iloc[hist_start:hist_end][target_col].to_numpy(np.float32)
    y_mean = float(y_hist.mean())
    y_std  = float(y_hist.std() + 1e-6)

    return xh_mean, xh_std, xf_mean, xf_std, y_mean, y_std