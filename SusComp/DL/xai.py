# xai_perm_importance.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from datetime import timedelta

from training_data_builder import (
    anchored_train_val_test_split_indices,
    compute_scalers_hist_and_fut,
    WindowedForecastDatasetWithFuture,
)
from models import LSTMMultiHorizon


TIME_FEATURE_NAMES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]


@torch.no_grad()
def save_multi_horizon_test_predictions_csv(
    *,
    model: nn.Module,
    loader: DataLoader,
    pred_index: pd.DatetimeIndex,
    out_csv_path: str | Path,
    device: str,
    col_prefix: str = "pred_t+",
    col_suffix: str = "h",
    log_every_batches: int = 0,
) -> pd.DataFrame:
    """
    Runs model predictions for all samples in `loader` and saves a wide CSV:
      index = pred_index (forecast start times)
      columns = pred_t+01h ... pred_t+24h (or H inferred from output)

    Returns the predictions dataframe.
    """
    model.eval()
    out_csv_path = Path(out_csv_path)

    preds = []
    t0 = time.time()

    n_batches = len(loader)
    for bi, (x_hist, x_fut, y) in enumerate(loader):
        x_hist = x_hist.to(device)
        y_hat = model(x_hist)

        # handle (B, H, 1) -> (B, H) if needed
        if y_hat.ndim == 3 and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)

        preds.append(y_hat.detach().cpu().numpy())

        if log_every_batches and ((bi + 1) % log_every_batches == 0 or (bi + 1) == n_batches):
            elapsed = time.time() - t0
            print(f"[pred] batch {bi+1}/{n_batches} | elapsed={timedelta(seconds=int(elapsed))}")

    yhat = np.vstack(preds) if preds else np.empty((0, 0), dtype=np.float32)

    if yhat.shape[0] != len(pred_index):
        raise RuntimeError(
            f"Pred count mismatch: yhat has {yhat.shape[0]} rows but pred_index has {len(pred_index)}."
        )

    horizon = yhat.shape[1]
    pred_cols = [f"{col_prefix}{h:02d}{col_suffix}" for h in range(1, horizon + 1)]

    preds_df = pd.DataFrame(yhat, index=pred_index, columns=pred_cols)
    preds_df.index.name = "Time"

    preds_df.to_csv(out_csv_path)
    elapsed = time.time() - t0
    print(f"[pred] Saved predictions: {out_csv_path} | rows={len(preds_df)} | horizon={horizon} | elapsed={timedelta(seconds=int(elapsed))}")

    return preds_df


@torch.no_grad()
def _predict_all_test(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    """Returns y_hat for all samples in loader as (N, H)."""
    model.eval()
    preds = []
    for x_hist, x_fut, y in loader:
        x_hist = x_hist.to(device)
        y_hat = model(x_hist)              # expected shape (B, H)
        preds.append(y_hat.detach().cpu().numpy())
    return np.vstack(preds) if preds else np.empty((0, 0), dtype=np.float32)


@torch.no_grad()
def _eval_loss(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        max_batches: Optional[int] = None,
        log_every: int = 20,
) -> float:
    model.eval()
    losses: List[float] = []
    t0 = time.time()
    n_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)

    print(f"[eval] Running baseline loss over ~{n_batches} batches...")

    for bi, (x_hist, x_fut, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x_hist = x_hist.to(device)
        y      = y.to(device)

        y_hat = model(x_hist)
        loss = loss_fn(y_hat, y)
        losses.append(float(loss.item()))

        if log_every and ((bi + 1) % log_every == 0 or (bi + 1) == n_batches):
            elapsed = time.time() - t0
            print(f"[eval] batch {bi+1}/{n_batches} | mean_loss={np.mean(losses):.6f} | elapsed={timedelta(seconds=int(elapsed))}")

    out = float(np.mean(losses)) if losses else float("nan")
    elapsed = time.time() - t0
    print(f"[eval] Done. baseline_loss={out:.6f} | elapsed={timedelta(seconds=int(elapsed))}")
    return out


@torch.no_grad()
def _permutation_importance(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        n_features: int,
        n_repeats: int = 5,
        max_batches: Optional[int] = None,
        seed: int = 42,
        log_every_batches: int = 50,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Returns:
      baseline_loss: float
      mean_delta: (F,) mean(permuted_loss - baseline)
      std_delta:  (F,) std over repeats
    """
    rng = np.random.default_rng(seed)

    print(f"[perm] Starting permutation importance | features={n_features} | repeats={n_repeats} | device={device}")
    t_global = time.time()

    baseline = _eval_loss(model, loader, loss_fn, device, max_batches=max_batches, log_every=log_every_batches)

    mean_delta = np.zeros(n_features, dtype=np.float64)
    std_delta  = np.zeros(n_features, dtype=np.float64)

    repeat_seeds = rng.integers(low=0, high=2**31 - 1, size=n_repeats, dtype=np.int64)

    n_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)

    for f in range(n_features):
        t_feat = time.time()
        print(f"\n[perm] Feature {f+1}/{n_features} ...")
        deltas = []

        for r in range(n_repeats):
            t_rep = time.time()
            torch.manual_seed(int(repeat_seeds[r]))  # controls torch.randperm

            losses = []
            for bi, (x_hist, x_fut, y) in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break

                x_hist = x_hist.to(device)
                x_fut  = x_fut.to(device)
                y      = y.to(device)

                perm = torch.randperm(x_hist.size(0), device=device)
                x_perm = x_hist.clone()
                x_perm[:, :, f] = x_hist[perm, :, f]

                y_hat = model(x_perm)
                loss = loss_fn(y_hat, y)
                losses.append(float(loss.item()))

                if log_every_batches and ((bi + 1) % log_every_batches == 0 or (bi + 1) == n_batches):
                    elapsed = time.time() - t_rep
                    print(f"[perm]   f={f+1}/{n_features} r={r+1}/{n_repeats} batch {bi+1}/{n_batches} | mean_loss={np.mean(losses):.6f} | elapsed={timedelta(seconds=int(elapsed))}")

            perm_loss = float(np.mean(losses)) if losses else float("nan")
            delta = perm_loss - baseline
            deltas.append(delta)

            rep_elapsed = time.time() - t_rep
            print(f"[perm]   repeat {r+1}/{n_repeats} done | perm_loss={perm_loss:.6f} | Δ={delta:+.6f} | elapsed={timedelta(seconds=int(rep_elapsed))}")

        deltas = np.asarray(deltas, dtype=np.float64)
        mean_delta[f] = float(np.nanmean(deltas))
        std_delta[f]  = float(np.nanstd(deltas))

        feat_elapsed = time.time() - t_feat
        done = f + 1
        total_elapsed = time.time() - t_global
        # crude ETA
        avg_per_feat = total_elapsed / done
        eta = avg_per_feat * (n_features - done)

        print(f"[perm] Feature {f+1}/{n_features} summary | meanΔ={mean_delta[f]:+.6f} ±{std_delta[f]:.6f} | "
              f"feat_elapsed={timedelta(seconds=int(feat_elapsed))} | "
              f"ETA≈{timedelta(seconds=int(eta))}")

    total_elapsed = time.time() - t_global
    print(f"\n[perm] Done permutation importance | elapsed={timedelta(seconds=int(total_elapsed))}")
    return baseline, mean_delta, std_delta

def run_xai_permutation(
        df: pd.DataFrame,
        hist_feature_cols: Sequence[str],
        fut_feature_cols: Sequence[str],
        target_col: str,
        cfg,
        model_pth_path: str | Path = "best_model.pth",
        lookback_steps: int = 30 * 24,
        horizon_steps: int = 24,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "mps",
        n_repeats: int = 5,
        max_batches: Optional[int] = 100,
        seed: int = 42,
        out_dir: str | Path = "xai",
        make_plot: bool = True,
        anchor_hour: int = 3,
        anchor_minute: int = 0,
) -> pd.DataFrame:
    """
    Permutation importance runner consistent with the anchored (03:00) dataset pipeline.

    Produces a dataframe with:
      feature, importance_mean (Δloss), importance_std, baseline_loss

    Saves:
      out_dir/permutation_importance.csv
      out_dir/permutation_importance.png (optional)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(hist_feature_cols+fut_feature_cols)

    # sanity checks
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex.")

    splits = anchored_train_val_test_split_indices(df, lookback_steps, horizon_steps, anchor_hour=3, anchor_minute=0)

    xh_mean, xh_std, xf_mean, xf_std, y_mean, y_std = compute_scalers_hist_and_fut(
        df, hist_feature_cols, fut_feature_cols, target_col,
        splits["train"], lookback_steps, horizon_steps
    )

    test_ds = WindowedForecastDatasetWithFuture(
        df=df,
        hist_feature_cols=hist_feature_cols,
        fut_feature_cols=fut_feature_cols,
        target_col=target_col,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        ts=splits["test"],
        x_hist_mean=xh_mean, x_hist_std=xh_std,
        x_fut_mean=xf_mean,  x_fut_std=xf_std,
        scale_y=False, y_mean=y_mean, y_std=y_std,
        anchor_hour=3, anchor_minute=0,
        add_time_features_to_future=True,
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # feature names as seen by the model input
    feature_names = hist_feature_cols + TIME_FEATURE_NAMES
    n_input_features = len(feature_names)

    # Optional: consistency check vs cfg
    if hasattr(cfg, "input_size") and int(cfg.input_size) != n_input_features:
        raise ValueError(
            f"cfg.input_size={cfg.input_size} but dataset provides {n_input_features} features "
            f"(len(feature_cols)={len(feature_cols)} + 6 time features)."
        )

    # load model
    model = LSTMMultiHorizon(cfg).to(device)
    state = torch.load(str(model_pth_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    loss_fn = nn.SmoothL1Loss()

    baseline, mean_delta, std_delta = _permutation_importance(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        n_features=n_input_features,
        n_repeats=n_repeats,
        max_batches=max_batches,
        seed=seed,
    )

    # results df
    res = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean_delta_loss": mean_delta,
            "importance_std_delta_loss": std_delta,
        }
    )
    res["baseline_loss"] = baseline
    res = res.sort_values("importance_mean_delta_loss", ascending=False).reset_index(drop=True)

    csv_path = out_dir / "permutation_importance.csv"
    res.to_csv(csv_path, index=False)

    print(f"Baseline test loss: {baseline:.6f}")
    print(f"Saved: {csv_path}")
    print("\nTop features (higher Δloss = more important):")
    for i in range(min(15, len(res))):
        r = res.iloc[i]
        print(
            f"{i+1:02d}. {r['feature']:<20s}  "
            f"Δloss={r['importance_mean_delta_loss']:+.6f}  ±{r['importance_std_delta_loss']:.6f}"
        )

    if make_plot:
        top_k = min(20, len(res))
        plt.figure(figsize=(10, 6))
        plt.barh(
            res.loc[: top_k - 1, "feature"][::-1],
            res.loc[: top_k - 1, "importance_mean_delta_loss"][::-1],
            xerr=res.loc[: top_k - 1, "importance_std_delta_loss"][::-1],
        )
        plt.xlabel("Permutation importance (Δ loss)")
        plt.title("Permutation Feature Importance (Test set)")
        plt.tight_layout()
        fig_path = out_dir / "permutation_importance.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved: {fig_path}")


    return res

def plot_random_test_forecasts(
        model,
        test_ds,
        df,
        target_col: str,
        device: str,
        n: int = 5,
        seed: int = 42,
        out_dir: str = "plots",
        show: bool = False,
        anchor_hour: int = 3,
        anchor_minute: int = 0,
        strict_anchor_check: bool = True,
) -> None:
    """
    Plots n random forecasts vs ground truth from the test set.
    Enforces that forecast start time is anchored (default 03:00).
    """
    rng = np.random.default_rng(seed)
    n = min(n, len(test_ds))
    if n <= 0:
        print("No test samples to plot.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Optional: verify the dataset is anchored
    if strict_anchor_check:
        if not hasattr(test_ds, "ts"):
            raise ValueError("test_ds has no attribute `ts`. Cannot verify anchored start times.")
        ts = np.asarray(test_ds.ts, dtype=np.int64)
        if ts.size == 0:
            raise ValueError("test_ds.ts is empty.")
        times = df.index[ts]
        ok = (times.hour == anchor_hour) & (times.minute == anchor_minute)
        if not bool(np.all(ok)):
            bad = times[~ok][:5]
            raise ValueError(
                f"test_ds contains non-anchored start times. "
                f"Expected {anchor_hour:02d}:{anchor_minute:02d}. Examples: {list(bad)}"
            )

    model.eval()
    idxs = rng.choice(len(test_ds), size=n, replace=False)

    for k, ds_idx in enumerate(idxs):
        x_hist, x_fut, y = test_ds[ds_idx]     # x_hist: (L, F), y: (H,)
        t = int(test_ds.ts[ds_idx])            # forecast start iloc index in df

        horizon_len = int(y.shape[0])
        horizon_times = df.index[t: t + horizon_len]

        # Extra per-sample guard
        if strict_anchor_check:
            start_dt = horizon_times[0]
            if not (start_dt.hour == anchor_hour and start_dt.minute == anchor_minute):
                raise ValueError(
                    f"Sample start time is not anchored: start={start_dt} "
                    f"(expected {anchor_hour:02d}:{anchor_minute:02d})"
                )

        with torch.no_grad():
            x_hist_in = x_hist.unsqueeze(0).to(device)   # (1, L, F)
            y_hat = model(x_hist_in).squeeze(0).detach().cpu().numpy()

        y_true = y.detach().cpu().numpy()

        plt.figure()
        plt.plot(horizon_times, y_true, label="Ground truth")
        plt.plot(horizon_times, y_hat, label="Prediction")
        plt.title(f"Test forecast #{k+1} | start={horizon_times[0]}")
        plt.xlabel("Time")
        plt.ylabel(target_col)
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()

        out_path = Path(out_dir) / f"test_forecast_{k+1:02d}.png"
        plt.savefig(out_path, dpi=150)
        if show:
            plt.show()
        plt.close()

    print(f"Saved {n} forecast plot to: {out_dir}/")



def run_horizon_forecast(df: pd.DataFrame,
        hist_feature_cols: Sequence[str],
        fut_feature_cols: Sequence[str],
        target_col: str,
        cfg,
        model_pth_path: str | Path = "best_model.pth",
        lookback_steps: int = 30 * 24,
        horizon_steps: int = 24,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "mps",
        n_repeats: int = 5,
        max_batches: Optional[int] = 100,
        seed: int = 42,
        out_dir: str | Path = "xai",
        make_plot: bool = True,
        anchor_hour: int = 3,
        anchor_minute: int = 0,
        rand_seed: int = 42,):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(hist_feature_cols + fut_feature_cols)

    # sanity checks
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex.")

    splits = anchored_train_val_test_split_indices(df, lookback_steps, horizon_steps, anchor_hour=3, anchor_minute=0)

    xh_mean, xh_std, xf_mean, xf_std, y_mean, y_std = compute_scalers_hist_and_fut(
        df, hist_feature_cols, fut_feature_cols, target_col,
        splits["train"], lookback_steps, horizon_steps
    )

    test_ds = WindowedForecastDatasetWithFuture(
        df=df,
        hist_feature_cols=hist_feature_cols,
        fut_feature_cols=fut_feature_cols,
        target_col=target_col,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        ts=splits["test"],
        x_hist_mean=xh_mean, x_hist_std=xh_std,
        x_fut_mean=xf_mean, x_fut_std=xf_std,
        scale_y=False, y_mean=y_mean, y_std=y_std,
        anchor_hour=3, anchor_minute=0,
        add_time_features_to_future=True,
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # feature names as seen by the model input
    feature_names = hist_feature_cols + TIME_FEATURE_NAMES
    n_input_features = len(feature_names)

    # Optional: consistency check vs cfg
    if hasattr(cfg, "input_size") and int(cfg.input_size) != n_input_features:
        raise ValueError(
            f"cfg.input_size={cfg.input_size} but dataset provides {n_input_features} features "
            f"(len(feature_cols)={len(feature_cols)} + 6 time features)."
        )

    # load model
    model = LSTMMultiHorizon(cfg).to(device)
    state = torch.load(str(model_pth_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    loss_fn = nn.SmoothL1Loss()

    # Forecast start times (one per test sample)
    pred_index = df.index[test_ds.ts]

    # Optional: plot a few random test forecasts (saved as PNGs)
    plot_random_test_forecasts(
        model=model,
        test_ds=test_ds,
        df=df,
        target_col=target_col,
        device=device,
        n=5,
        seed=rand_seed,
        out_dir=str(out_dir / "forecast_plots"),
        show=False,
        anchor_hour=anchor_hour,
        anchor_minute=anchor_minute,
        strict_anchor_check=True,
    )

    save_multi_horizon_test_predictions_csv(
        model=model,
        loader=test_loader,
        pred_index=pred_index,
        out_csv_path=out_dir / "test_predictions_24h.csv",
        device=device,
        log_every_batches=20,  # set 0 to disable
    )