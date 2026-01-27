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
) -> None:
    """
    Plots n random 24-horizon forecasts vs ground truth from the test set.
    Saves figures to out_dir as PNGs.
    """
    rng = np.random.default_rng(seed)
    n = min(n, len(test_ds))
    if n <= 0:
        print("No test samples to plot.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model.eval()
    idxs = rng.choice(len(test_ds), size=n, replace=False)

    for k, ds_idx in enumerate(idxs):
        x, y = test_ds[ds_idx]                  # x: (L, F), y: (H,)
        t = int(test_ds.ts[ds_idx])             # forecast start index in df (iloc index)

        # Build horizon datetime index for the plot
        horizon_len = y.shape[0]
        horizon_times = df.index[t : t + horizon_len]

        # Predict
        with torch.no_grad():
            x_in = x.unsqueeze(0).to(device)    # (1, L, F)
            y_hat = model(x_in).squeeze(0).detach().cpu().numpy()  # (H,)

        y_true = y.detach().cpu().numpy()

        # Plot
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