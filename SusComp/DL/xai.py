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

from training_data_builder import (
    anchored_train_val_test_split_indices,
    compute_scalers_from_ts,
    WindowedForecastDataset,
)
from models import LSTMMultiHorizon


TIME_FEATURE_NAMES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]


@torch.no_grad()
def _eval_loss(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        max_batches: Optional[int] = None,
) -> float:
    model.eval()
    losses: List[float] = []
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


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
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Returns:
      baseline_loss: float
      mean_delta: (F,) mean(permuted_loss - baseline)
      std_delta:  (F,) std over repeats

    Permutes feature f across batch dimension (keeps time dimension intact).
    """
    rng = np.random.default_rng(seed)

    baseline = _eval_loss(model, loader, loss_fn, device, max_batches=max_batches)

    mean_delta = np.zeros(n_features, dtype=np.float64)
    std_delta = np.zeros(n_features, dtype=np.float64)

    repeat_seeds = rng.integers(low=0, high=2**31 - 1, size=n_repeats, dtype=np.int64)

    for f in range(n_features):
        deltas = []
        for r in range(n_repeats):
            torch.manual_seed(int(repeat_seeds[r]))  # controls torch.randperm

            losses = []
            for bi, (x, y) in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break

                x = x.to(device)
                y = y.to(device)

                perm = torch.randperm(x.size(0), device=device)
                x_perm = x.clone()
                x_perm[:, :, f] = x[perm, :, f]

                y_hat = model(x_perm)
                loss = loss_fn(y_hat, y)
                losses.append(float(loss.item()))

            perm_loss = float(np.mean(losses)) if losses else float("nan")
            deltas.append(perm_loss - baseline)

        deltas = np.asarray(deltas, dtype=np.float64)
        mean_delta[f] = float(np.nanmean(deltas))
        std_delta[f] = float(np.nanstd(deltas))

    return baseline, mean_delta, std_delta


def run_xai_permutation(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        cfg,
        model_pth_path: str | Path = "best_model.pth",
        lookback_steps: int = 30 * 24,
        horizon_steps: int = 24,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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

    feature_cols = list(feature_cols)

    # sanity checks
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex.")

    # ✅ FIX 1: call split with df, not len(df)
    splits = anchored_train_val_test_split_indices(
        df=df,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        anchor_hour=anchor_hour,
        anchor_minute=anchor_minute,
    )

    # scalers from TRAIN ts only
    x_mean, x_std, y_mean, y_std = compute_scalers_from_ts(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        train_ts=splits["train"],
        lookback_steps=lookback_steps,
    )

    # ✅ FIX 2: use TEST split for test importance (unless you intentionally want val)
    test_ds = WindowedForecastDataset(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        ts=splits["test"],               # <- was splits["val"]
        x_mean=x_mean,
        x_std=x_std,
        scale_y=False,
        y_mean=y_mean,
        y_std=y_std,
        # optional safety check that ts is anchored:
        anchor_hour=anchor_hour,
        anchor_minute=anchor_minute,
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # feature names as seen by the model input
    feature_names = feature_cols + TIME_FEATURE_NAMES
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