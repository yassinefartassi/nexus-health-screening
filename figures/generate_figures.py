"""
Generates the NEW report figures (architecture diagram, FL round flow, FL
convergence, rPPG pipeline illustration) programmatically with matplotlib, so
they're re-runnable and never hand-drawn/fabricated. Figures that already
exist as clean, real outputs from training/evaluation (confusion matrices,
learning curves) are copied in as-is by copy_existing_assets() rather than
regenerated -- see that function's comments for exactly which and why.

Run from the repo root: python figures/generate_figures.py
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
DPI = 150


def _box(ax, xy, w, h, text, facecolor="#EAF2FB", edgecolor="#4C72B0", fontsize=9.5):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.3,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return box


def _arrow(ax, start, end, text=None, color="#555555", style="-|>", connectionstyle="arc3,rad=0.0", fontsize=8):
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14, color=color, linewidth=1.3,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize, color=color,
                 backgroundcolor="white")


def system_architecture():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("NEXUS_PFE system architecture", fontsize=14, pad=14)

    # Camera / sensor sources (top)
    _box(ax, (0.3, 6.8), 2.2, 0.9, "Phone camera\n(mobile)", "#FFF3D6", "#C9971A")
    _box(ax, (2.7, 6.8), 2.2, 0.9, "Webcam / USB cam\n(desktop, RPi5)", "#FFF3D6", "#C9971A")
    _box(ax, (5.1, 6.8), 2.2, 0.9, "Pi Camera Module\n(RPi5, rpicam)", "#FFF3D6", "#C9971A")

    # Four deployments (middle)
    deployments = [
        (0.2, 4.6, "nexus_pfe\n(Streamlit desktop,\nUbuntu/Windows)"),
        (3.2, 4.6, "nexus_pfe_mobile\n(Flutter, Android,\non-device ONNX Runtime)"),
        (6.2, 4.6, "nexus_pfe_raspberrypi\n(RPi5 edge, SSH-deployed,\nsame inference code)"),
        (9.2, 4.6, "nexus_pfe_server_fed\n(FL server, Windows,\nFlower / FedAvg)"),
    ]
    for x, y, label in deployments:
        _box(ax, (x, y), 2.5, 1.3, label, "#EAF2FB", "#4C72B0")

    # Shared inference stack (below the three client deployments)
    _box(ax, (0.2, 2.7), 5.5, 0.8,
         "Shared inference: dermatology (EfficientNetV2-S, ONNX) + ocular (ResNet-18, ONNX)\n"
         "+ rPPG (CHROM/POS, no trainable weights)", "#F1EAFB", "#7B52AB", fontsize=8.5)
    _box(ax, (6.2, 2.7), 2.5, 0.8, "On-device inference only\n(same ONNX files)", "#F1EAFB", "#7B52AB", fontsize=8.5)

    # Camera -> deployment arrows
    _arrow(ax, (1.4, 6.8), (1.4, 5.9))
    _arrow(ax, (3.8, 6.8), (4.4, 5.9))
    _arrow(ax, (6.2, 6.8), (7.4, 5.9))

    # Deployment -> shared inference
    _arrow(ax, (1.4, 4.6), (2.5, 3.5))
    _arrow(ax, (4.4, 4.6), (3.5, 3.5))
    _arrow(ax, (7.4, 4.6), (7.4, 3.5))

    # FL topology: weight-only arrows between clients and FL server
    _box(ax, (2.5, 0.3), 6.5, 1.0,
         "Federated learning (dermatology + ocular only, modular / independently versioned): "
         "clients train locally on-device,\nONLY MODEL WEIGHTS are sent to the FL server -- raw images never leave the device that captured them",
         "#FDEAEA", "#C0392B", fontsize=8.5)
    _arrow(ax, (1.4, 4.6), (4.0, 1.3), color="#C0392B", connectionstyle="arc3,rad=-0.2")
    _arrow(ax, (7.4, 4.6), (6.5, 1.3), color="#C0392B", connectionstyle="arc3,rad=0.15")
    _arrow(ax, (10.45, 4.6), (9.0, 1.3), text="weights", color="#C0392B", connectionstyle="arc3,rad=-0.2")

    legend_handles = [
        mpatches.Patch(facecolor="#FFF3D6", edgecolor="#C9971A", label="Camera / sensor source"),
        mpatches.Patch(facecolor="#EAF2FB", edgecolor="#4C72B0", label="Deployment"),
        mpatches.Patch(facecolor="#F1EAFB", edgecolor="#7B52AB", label="Shared inference stack"),
        mpatches.Patch(facecolor="#FDEAEA", edgecolor="#C0392B", label="Federated learning (weights only)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, -0.02), fontsize=8, ncol=2, frameon=False)

    fig.tight_layout()
    out = FIGURES_DIR / "system_architecture.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def fl_round_flow():
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=DPI)
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.6, 3.9)
    ax.axis("off")
    ax.set_title("Federated learning round flow (per modality: dermatology OR ocular)", fontsize=13, pad=10)

    steps = [
        (0.3, 1.9, 1.9, 1.2, "Client A\n(e.g. desktop)\nlocal data, never\nleaves device"),
        (0.3, 0.0, 1.9, 1.2, "Client B\n(e.g. RPi5)\nlocal data, never\nleaves device"),
        (3.0, 0.95, 2.0, 1.2, "Local training\n1 epoch, on-device"),
        (5.8, 0.95, 2.2, 1.2, "FL server\nFedAvg aggregation\n(nexus_pfe_server_fed)"),
        (8.7, 1.9, 2.0, 1.2, "New global\nmodel (v{round})\n+ ONNX export"),
        (8.7, 0.0, 2.0, 1.2, "Redistribute\nweights to all\nclients"),
    ]
    for x, y, w, h, label in steps:
        _box(ax, (x, y), w, h, label, fontsize=8.5)

    _arrow(ax, (2.2, 2.5), (3.0, 2.0), text="weights")
    _arrow(ax, (2.2, 0.6), (3.0, 1.3), text="weights")
    _arrow(ax, (5.0, 1.55), (5.8, 1.55), text="local\nupdate")
    _arrow(ax, (8.0, 1.9), (8.7, 2.5))
    _arrow(ax, (8.0, 1.3), (8.7, 0.6))
    ax.text(5.5, -0.4,
            "Next round: the redistributed global model (bottom-right) becomes every client's starting point again.",
            ha="center", fontsize=8.5, color="#C0392B", style="italic")

    ax.text(5.5, 3.6, "Only model weight arrays cross the network at every arrow above -- raw images never do.",
            ha="center", fontsize=10, color="#C0392B", fontweight="bold")
    ax.text(5.5, 3.25, "Each modality (dermatology, ocular) has its own independently-versioned global model; rPPG is never federated (no trainable weights).",
            ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    out = FIGURES_DIR / "fl_round_flow.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def fl_convergence():
    """Plots the REAL round-accuracy CSVs from the local FL smoke test run
    during this hardening pass (2026-07-12), one line per modality. If the
    CSVs aren't present (round was never run), this figure is skipped --
    never fabricated."""
    csv_dir = REPO_ROOT / "nexus_pfe_server_fed" / "logs"
    csv_files = sorted(csv_dir.glob("dermatology_fedavg_*.csv")) + sorted(csv_dir.glob("ocular_fedavg_*.csv"))
    if not csv_files:
        print("No FL round CSV logs found -- skipping fl_convergence.png (pending: no FL round has been run).")
        return

    # Use the most recent CSV per modality.
    latest = {}
    for path in csv_files:
        modality = "dermatology" if path.name.startswith("dermatology") else "ocular"
        if modality not in latest or path.stat().st_mtime > latest[modality].stat().st_mtime:
            latest[modality] = path

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=DPI)
    colors = {"dermatology": "#4C72B0", "ocular": "#55A868"}
    for modality, path in latest.items():
        rounds, accs = [], []
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rounds.append(int(row["round"]))
                accs.append(float(row["eval_accuracy"]))
        ax.plot(rounds, accs, marker="o", label=f"{modality} ({path.name})", color=colors.get(modality))

    ax.set_xlabel("FL round")
    ax.set_ylabel("Centralized held-out accuracy")
    ax.set_title("FL convergence -- local 2-client smoke test\n(SYNTHETIC held-out data, tiny scale -- proves plumbing, not model quality)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = FIGURES_DIR / "fl_convergence.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out} (from real, if tiny/synthetic-data, FL round logs: {[p.name for p in latest.values()]})")


def rppg_pipeline():
    """Schematic (raw RGB -> CHROM/POS -> filtered -> BPM) combined with a
    real signal trace pulled from the webapp benchmark's run against the
    fixed synthetic test video, if that benchmark has been run; otherwise
    the schematic alone."""
    import json

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), dpi=DPI)
    fig.suptitle("rPPG signal pipeline: raw RGB → CHROM/POS → bandpass filter → BPM", fontsize=12)

    for ax in axes:
        ax.set_xticks([])

    # Stage 1: raw RGB trace (illustrative sinusoid + noise, same shape as real pipeline input)
    import numpy as np
    t = np.linspace(0, 10, 300)
    raw = 0.5 * np.sin(2 * np.pi * (72 / 60) * t) + 0.15 * np.random.default_rng(0).normal(size=len(t)) + 3
    axes[0].plot(t, raw, color="#C0392B", linewidth=0.9)
    axes[0].set_title("1. Raw mean-RGB\n(face ROI per frame)", fontsize=9)
    axes[0].set_yticks([])

    # Stage 2: CHROM/POS combination (illustrative)
    chrom = np.sin(2 * np.pi * (72 / 60) * t) + 0.05 * np.random.default_rng(1).normal(size=len(t))
    axes[1].plot(t, chrom, color="#7B52AB", linewidth=0.9)
    axes[1].set_title("2. CHROM / POS\nchrominance combination", fontsize=9)
    axes[1].set_yticks([])

    # Stage 3: bandpass filtered
    filtered = np.sin(2 * np.pi * (72 / 60) * t)
    axes[2].plot(t, filtered, color="#4C72B0", linewidth=1.1)
    axes[2].set_title("3. Zero-phase Butterworth\nbandpass (0.7-4 Hz)", fontsize=9)
    axes[2].set_yticks([])

    # Stage 4: FFT / BPM peak
    freqs = np.fft.rfftfreq(len(filtered), d=t[1] - t[0]) * 60
    power = np.abs(np.fft.rfft(filtered * np.hanning(len(filtered)))) ** 2
    valid = (freqs >= 40) & (freqs <= 180)
    axes[3].plot(freqs[valid], power[valid], color="#55A868", linewidth=1.1)
    peak_bpm = freqs[valid][np.argmax(power[valid])]
    axes[3].axvline(peak_bpm, color="#C0392B", linestyle="--", linewidth=1)
    axes[3].set_title(f"4. FFT peak search\n→ BPM ≈ {peak_bpm:.0f}", fontsize=9)
    axes[3].set_xlabel("BPM")
    axes[3].set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    caption = "Illustrative signal shapes"
    webapp_results_path = REPO_ROOT / "benchmark" / "webapp-benchmark" / "results" / "webapp_results.json"
    if webapp_results_path.exists():
        with open(webapp_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rppg = data.get("rppg", {})
        if rppg.get("bpm_mean") is not None:
            caption = (f"Illustrative signal shapes -- real measured result on the fixed synthetic test video "
                       f"(target 72 BPM injected): extracted BPM = {rppg['bpm_mean']:.1f} "
                       f"(±{rppg['bpm_std']:.2f} across {rppg['bpm_latency_ms']['n']} runs), "
                       f"latency {rppg['bpm_latency_ms']['mean']:.0f}ms/run")
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8, style="italic", color="#555555")

    out = FIGURES_DIR / "rppg_pipeline.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def copy_existing_assets():
    """These already exist as clean, real, publication-quality outputs from
    actual training/evaluation runs -- copied in as-is rather than
    regenerated, since regenerating would mean re-deriving them from the
    same underlying data with no quality improvement, adding risk of
    transcription error for zero benefit."""
    pairs = [
        (REPO_ROOT / "dermatho-assets" / "learning_curves.png", FIGURES_DIR / "dermatology_learning_curves.png"),
        (REPO_ROOT / "dermatho-assets" / "confusion_matrix.png", FIGURES_DIR / "dermatology_confusion_matrix.png"),
        (REPO_ROOT / "eyediseas-assets" / "confusion_matrix.png", FIGURES_DIR / "ocular_confusion_matrix.png"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")
        else:
            print(f"WARNING: {src} not found, skipping")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    copy_existing_assets()
    system_architecture()
    fl_round_flow()
    fl_convergence()
    rppg_pipeline()
    print("\nNote: latency_comparison.png and prediction_agreement_matrix.png are produced by "
          "benchmark/compare_results.py, not this script -- run that separately (or after this).")


if __name__ == "__main__":
    main()
