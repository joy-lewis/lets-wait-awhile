import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch


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
        x_hist, x_fut, y = test_ds[ds_idx]              # x: (L, F), y: (H,)
        t = int(test_ds.ts[ds_idx])         # forecast start iloc index in df

        horizon_len = int(y.shape[0])
        horizon_times = df.index[t : t + horizon_len]

        # Extra per-sample guard (useful even if strict_anchor_check=False)
        if strict_anchor_check:
            start_dt = horizon_times[0]
            if not (start_dt.hour == anchor_hour and start_dt.minute == anchor_minute):
                raise ValueError(
                    f"Sample start time is not anchored: start={start_dt} "
                    f"(expected {anchor_hour:02d}:{anchor_minute:02d})"
                )

        with torch.no_grad():
            x_hist_in = x_hist.unsqueeze(0).to(device)   # (1, L, F)
            x_fut_in = x_fut.unsqueeze(0).to(device)
            y_hat = model(x_hist_in, x_fut_in).squeeze(0).detach().cpu().numpy()

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


def plot_loss(
        train_losses,
        val_losses,
        test_loss,
        out_path: str = "plots/loss_curve.png",
        title: str = "Training & Validation Loss",
        show: bool = False,
) -> None:
    """
    Plots train/val loss curves and saves to a PNG.
    """
    if len(train_losses) == 0 or len(val_losses) == 0:
        print("No losses to plot.")
        return

    n = min(len(train_losses), len(val_losses))
    epochs = list(range(1, n + 1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = f"{title}, Test loss: {test_loss:.5f}"

    plt.figure()
    plt.plot(epochs, list(train_losses)[:n], label="Train loss")
    plt.plot(epochs, list(val_losses)[:n], label="Val loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

    print(f"Saved loss curve to: {out_path}")


import numpy as np
import torch

@torch.no_grad()
def permutation_importance(
        model,
        loader,
        loss_fn,
        device,
        n_features: int,
        n_repeats: int = 5,
        max_batches: int | None = None,
):
    model.eval()

    # 1) baseline loss
    base_losses = []
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        base_losses.append(loss_fn(y_hat, y).item())
    baseline = float(np.mean(base_losses))

    # 2) permute each feature
    importances = np.zeros(n_features, dtype=np.float64)

    for f in range(n_features):
        deltas = []
        for r in range(n_repeats):
            losses = []
            for bi, (x, y) in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break
                x = x.to(device)
                y = y.to(device)

                # permute feature f across batch
                perm = torch.randperm(x.size(0), device=device)
                x_perm = x.clone()
                x_perm[:, :, f] = x[perm, :, f]

                y_hat = model(x_perm)
                losses.append(loss_fn(y_hat, y).item())

            deltas.append(float(np.mean(losses)) - baseline)

        importances[f] = float(np.mean(deltas))

    return baseline, importances