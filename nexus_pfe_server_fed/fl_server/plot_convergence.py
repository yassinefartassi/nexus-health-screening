"""
Plots accuracy/loss vs. FL round from a CSV log written by server.py --
the convergence chart this whole PoC is meant to produce (proof that FedAvg
aggregation actually improves the model round over round).

    python -m fl_server.plot_convergence logs/dermatology_fedavg_20260711_143000.csv
    python -m fl_server.plot_convergence logs/ocular_fedavg_20260711_150000.csv

Works unchanged for either module -- the CSV schema (round, timestamp,
eval_loss, eval_accuracy, n_clients, comm_bytes) is identical regardless of
which framework produced it (see server.py's FederatingFedAvg).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def plot_convergence(csv_path: Path, out_path: Optional[Path] = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    out_path = out_path or csv_path.with_suffix(".png")

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))

    ax_acc.plot(df["round"], df["eval_accuracy"], marker="o", color="tab:blue")
    ax_acc.set_xlabel("FL round (0 = pretrained baseline)")
    ax_acc.set_ylabel("Held-out accuracy")
    ax_acc.set_title("Accuracy vs. round")
    ax_acc.grid(alpha=0.3)

    ax_loss.plot(df["round"], df["eval_loss"], marker="o", color="tab:orange")
    ax_loss.set_xlabel("FL round (0 = pretrained baseline)")
    ax_loss.set_ylabel("Held-out loss")
    ax_loss.set_title("Loss vs. round")
    ax_loss.grid(alpha=0.3)

    # CSV filenames are "<module>_fedavg_<timestamp>.csv" (see server.py) -- pull the module
    # name back out for the title instead of hardcoding one module.
    module_name = csv_path.stem.split("_fedavg_")[0]
    fig.suptitle(f"NEXUS_PFE {module_name} FL convergence (server-side held-out evaluation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FL convergence from a server.py CSV log")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    plot_convergence(args.csv_path, args.out)


if __name__ == "__main__":
    main()
