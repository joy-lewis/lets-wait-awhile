from training_data_builder import train_val_test_split_indices, compute_scalers, WindowedForecastDataset
from models import LSTMMultiHorizon
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict
import pandas as pd
import copy



def run_training(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback_steps: int = 30*24, # 30 days and 24 hours/day
    horizon_steps: int = 24, # 24 hours
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cfg=None,
):

    # Make sure required columns exist
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    splits = train_val_test_split_indices(len(df), lookback_steps, horizon_steps)

    x_mean, x_std, y_mean, y_std = compute_scalers(
        df, feature_cols, target_col, splits["train"], lookback_steps
    )

    # Datasets
    train_ds = WindowedForecastDataset(
        df=df, feature_cols=feature_cols, target_col=target_col,
        lookback_steps=lookback_steps, horizon_steps=horizon_steps,
        start_t=splits["train"][0], end_t=splits["train"][1],
        x_mean=x_mean, x_std=x_std,
        scale_y=False, y_mean=y_mean, y_std=y_std
    )
    val_ds = WindowedForecastDataset(
        df=df, feature_cols=feature_cols, target_col=target_col,
        lookback_steps=lookback_steps, horizon_steps=horizon_steps,
        start_t=splits["val"][0], end_t=splits["val"][1],
        x_mean=x_mean, x_std=x_std,
        scale_y=False, y_mean=y_mean, y_std=y_std
    )
    test_ds = WindowedForecastDataset(
        df=df, feature_cols=feature_cols, target_col=target_col,
        lookback_steps=lookback_steps, horizon_steps=horizon_steps,
        start_t=splits["test"][0], end_t=splits["test"][1],
        x_mean=x_mean, x_std=x_std,
        scale_y=False, y_mean=y_mean, y_std=y_std
    )

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = LSTMMultiHorizon(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()  # good default for regression; swap to MSELoss if you want

    def eval_loader(best_model, loader: DataLoader) -> float:
        best_model.eval()
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                y_hat = best_model(x)
                loss = loss_fn(y_hat, y)
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    # Train
    best_val_loss = float("inf")
    best_val_model = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_model.pth")
            print(f"\n🎉NEW BEST: Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}\n")
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_loss = eval_loader(model, test_loader)
    print(f"Test loss: {test_loss:.5f}")

    return {
        "model": model,
        "df_regular": df,
        "scalers": {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std},
        "splits": splits,
        "steps": {"lookback_steps": lookback_steps, "horizon_steps": horizon_steps},
    }


if __name__ == "__main__":
    pass