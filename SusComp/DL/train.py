from training_data_builder import anchored_train_val_test_split_indices, compute_scalers_hist_and_fut, WindowedForecastDatasetWithFuture
from models import LSTMMultiHorizon
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict
import pandas as pd
import copy
import matplotlib.pyplot as plt
from pathlib import Path
from eval import plot_random_test_forecasts, plot_loss


def run_training(
    df: pd.DataFrame,
    hist_feature_cols: list[str],
    fut_feature_cols: list[str],
    target_col: str,
    lookback_steps: int = 30*24, # 30 days and 24 hours/day
    horizon_steps: int = 24, # 24 hours
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cfg=None,
    zero_future=True
):

    # Make sure required columns exist
    missing = [c for c in hist_feature_cols + fut_feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    splits = anchored_train_val_test_split_indices(df, lookback_steps, horizon_steps, anchor_hour=3, anchor_minute=0)

    print("Training started with config:\n"
          f"hist_feature_cols: {hist_feature_cols}\n"
          f"fut_feature_cols: {fut_feature_cols}\n"
          f"target_col: {target_col}\n"
          f"lookback_steps: {lookback_steps}\n"
          f"horizon_steps: {horizon_steps}\n"
          f"batch_size: {batch_size}\n"
          f"lr: {lr}\n"
          f"epochs: {epochs}\n"
          f"device: {device}\n")

    xh_mean, xh_std, xf_mean, xf_std, y_mean, y_std = compute_scalers_hist_and_fut(
        df, hist_feature_cols, fut_feature_cols, target_col,
        splits["train"], lookback_steps, horizon_steps
    )

    train_ds = WindowedForecastDatasetWithFuture(
        df=df,
        hist_feature_cols=hist_feature_cols,
        fut_feature_cols=fut_feature_cols,
        target_col=target_col,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        ts=splits["train"],
        x_hist_mean=xh_mean, x_hist_std=xh_std,
        x_fut_mean=xf_mean,  x_fut_std=xf_std,
        scale_y=False, y_mean=y_mean, y_std=y_std,
        anchor_hour=3, anchor_minute=0,
        add_time_features_to_future=True,
        zero_future=zero_future,
    )

    val_ds = WindowedForecastDatasetWithFuture(
        df=df,
        hist_feature_cols=hist_feature_cols,
        fut_feature_cols=fut_feature_cols,
        target_col=target_col,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        ts=splits["val"],
        x_hist_mean=xh_mean, x_hist_std=xh_std,
        x_fut_mean=xf_mean,  x_fut_std=xf_std,
        scale_y=False, y_mean=y_mean, y_std=y_std,
        anchor_hour=3, anchor_minute=0,
        add_time_features_to_future=True,
        zero_future=zero_future,
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
        zero_future=zero_future,
    )

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print("train samples:", len(train_ds), "val samples:", len(val_ds), "test samples:", len(test_ds))
    print("train batches:", len(train_loader), "val batches:", len(val_loader))

    # Model
    model = LSTMMultiHorizon(cfg).to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=14, eta_min=5e-5)

    loss_fn = nn.SmoothL1Loss()  # Huber loss

    def eval_loader(best_model, loader: DataLoader) -> float:
        best_model.eval()
        losses = []
        with torch.no_grad():
            for xv_hist, xv_fut, yv in loader:
                xv_hist = xv_hist.to(device)
                yv      = yv.to(device)

                yv_hat = model(xv_hist)
                loss_v = loss_fn(yv_hat, yv)
                losses.append(loss_v.item())
        return float(np.mean(losses)) if losses else float("nan")

    # Train
    best_val_loss = float("inf")
    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x_hist, x_fut, y in train_loader:
            if epoch==1:
                print(f"x_hist shape: {x_hist.shape}, x_fut shape: {x_fut.shape} y shape: {y.shape}")
            x_hist = x_hist.to(device)
            y      = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x_hist)
            loss = loss_fn(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            if math.isnan(loss.item()):
                raise ValueError("NaN loss encountered during training")
            if math.isinf(loss.item()):
                raise ValueError("Inf loss encountered during training")

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = eval_loader(model, val_loader)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Decay LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_model.pth")
            print(f"🎉NEW BEST - Epoch: {epoch} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | lr: {current_lr:.5f}")
        else:
            print(f"Epoch: {epoch} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | lr: {current_lr:.5f}")

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_loss = eval_loader(model, test_loader)
    print(f"Test loss: {test_loss:.5f}")


    # Eval
    plot_random_test_forecasts(
        model=model,
        test_ds=test_ds,
        df=df,
        target_col=target_col,
        device=device,
        n=5,                # <-- set how many random instances you want
        seed=42,
        out_dir="plots",
        show=False,
        anchor_hour=3,
        anchor_minute=0
    )

    plot_loss(train_loss_list, val_loss_list, test_loss)

if __name__ == "__main__":
    pass