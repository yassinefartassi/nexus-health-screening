"""
Plots per-round FedAvg convergence (train/eval loss + accuracy) from a CSV
log written by fl/server.py -- the convergence chart for the thesis.

    python -m fl.plot_convergence fl/logs/dermatology_fedavg_20260710_120000.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def plot_convergence(csv_path: Path, out_path: Optional[Path] = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    out_path = out_path or csv_path.with_suffix(".png")

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))

    ax_acc.plot(df["round"], df["train_accuracy"], marker="o", label="train (weighted avg)")
    ax_acc.plot(df["round"], df["eval_accuracy"], marker="o", label="eval (weighted avg)")
    ax_acc.set_xlabel("FL round")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy per round")
    ax_acc.legend()

    ax_loss.plot(df["round"], df["train_loss"], marker="o", label="train")
    ax_loss.plot(df["round"], df["eval_loss"], marker="o", label="eval")
    ax_loss.set_xlabel("FL round")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss per round")
    ax_loss.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FL convergence from a fl/server.py CSV log")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    plot_convergence(args.csv_path, args.out)


if __name__ == "__main__":
    main()
