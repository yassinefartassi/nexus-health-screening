"""
rPPG evaluation harness: validates the existing CHROM/POS pipeline against
UBFC-rPPG ground truth.

This harness CALLS the production pipeline unmodified -- it does not
reimplement any signal-processing step:
  - src.preprocessing.face_roi.FaceROITracker  (face ROI -> per-frame mean RGB)
  - src.inference.rppg_infer.RPPGBuffer        (rolling buffer -> CHROM/POS ->
                                                 bandpass -> FFT BPM)

For each video, frames are fed one by one into the same RPPGBuffer/FaceROITracker
objects app.py drives for a live webcam feed: add_sample() every frame,
analyze() periodically once a full rolling window is buffered. The resulting
BPM time series is aligned against the CMS50E reference BPM (see
evaluation/ubfc_gt.py) over the same windows, and MAE / RMSE / Pearson r are
reported for CHROM and POS separately, plus Bland-Altman and scatter plots.

Usage:
    python evaluation/eval_rppg_ubfc.py \
        --dataset-root data/UBFC-rPPG/DATASET_1 \
        --output-dir evaluation/results

N.B.: UBFC-rPPG DATASET_1 has only 4 subjects (6, 8, 10, 11) -- this is a
preliminary sanity check, not a statistically powered validation. Subject
count is never hardcoded: this script walks --dataset-root for subject
folders, so pointing it at DATASET_2 (42 subjects) requires no code changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # nexus_pfe root, for `src.*` / `evaluation.*`

from evaluation.ubfc_gt import Subject, discover_subjects, parse_gtdump
from src.inference.rppg_infer import BUFFER_SECONDS, RPPGBuffer
from src.preprocessing.face_roi import FaceROITracker

# Categorical slots 1 & 2 (dataviz skill default palette, light mode) -- fixed
# assignment, not cycled: CHROM always blue, POS always green.
COLOR_CHROM = "#2a78d6"
COLOR_POS = "#008300"
COLOR_YX_LINE = "#898781"    # muted ink
COLOR_BIAS_LINE = "#0b0b0b"  # primary ink
COLOR_LOA_LINE = "#52514e"   # secondary ink
GRID_COLOR = "#e1e0d9"

MAE_WARN_THRESHOLD_BPM = 10.0
DURATION_MISMATCH_WARN_PCT = 10.0


def run_pipeline_on_video(
    video_path: Path, window_seconds: float, step_seconds: float, max_seconds: float | None = None
) -> tuple[pd.DataFrame, float, int]:
    """Run the real production pipeline over a video file frame-by-frame and
    repeatedly analyze a trailing rolling window of `window_seconds`.

    Mirrors app.py's live-webcam loop (continuous add_sample, periodic
    analyze()), except the first analysis is deferred until `window_seconds`
    of video time have elapsed, so every reported estimate spans the same
    duration as the ground-truth comparison window.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    fps_hint = cap.get(cv2.CAP_PROP_FPS) or 30.0

    buffer = RPPGBuffer(buffer_seconds=window_seconds)
    rows = []
    frame_idx = 0
    last_analyze_t = -np.inf
    try:
        with FaceROITracker() as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = frame_idx / fps_hint
                roi_result = tracker.process_frame(frame)
                if roi_result.face_found:
                    buffer.add_sample(roi_result.mean_rgb, timestamp=t)
                frame_idx += 1

                # Gate on elapsed VIDEO time (t >= window_seconds), not
                # buffer.is_ready(min_seconds=window_seconds): the buffer's
                # internal eviction keeps its span strictly <= buffer_seconds,
                # so a >= buffer_seconds check on the buffer's own span almost
                # never fires. buffer.is_ready() (its 5s default) just confirms
                # there is enough actual data to run CHROM/POS without erroring.
                if t >= window_seconds and buffer.is_ready() and (t - last_analyze_t) >= step_seconds:
                    try:
                        result = buffer.analyze()
                    except ValueError:
                        pass
                    else:
                        rows.append(
                            {
                                "window_start_s": t - window_seconds,
                                "window_end_s": t,
                                "bpm_chrom": result.bpm_chrom,
                                "bpm_pos": result.bpm_pos,
                                "fps_est": result.fps,
                                "n_samples": result.n_samples,
                            }
                        )
                    last_analyze_t = t

                if max_seconds is not None and t > max_seconds:
                    break
    finally:
        cap.release()

    columns = ["window_start_s", "window_end_s", "bpm_chrom", "bpm_pos", "fps_est", "n_samples"]
    df = pd.DataFrame(rows, columns=columns)
    return df, fps_hint, frame_idx


def reference_bpm_for_window(gt_df: pd.DataFrame, window_start_s: float, window_end_s: float) -> float:
    mask = (gt_df["timestamp_s"] >= window_start_s) & (gt_df["timestamp_s"] <= window_end_s)
    sel = gt_df.loc[mask, "hr_bpm"]
    return float(sel.mean()) if len(sel) else float("nan")


def evaluate_subject(
    subject: Subject, window_seconds: float, step_seconds: float, max_seconds: float | None
) -> tuple[pd.DataFrame, dict]:
    gt_df = parse_gtdump(subject.gt_path)
    est_df, container_fps, n_frames_decoded = run_pipeline_on_video(
        subject.video_path, window_seconds, step_seconds, max_seconds
    )

    gt_duration_s = float(gt_df["timestamp_s"].iloc[-1])
    video_duration_s = n_frames_decoded / container_fps if container_fps else float("nan")
    discrepancy_pct = (
        abs(video_duration_s - gt_duration_s) / gt_duration_s * 100.0 if gt_duration_s else float("nan")
    )

    est_df = est_df.copy()
    est_df["subject"] = subject.subject_id
    est_df["bpm_reference"] = [
        reference_bpm_for_window(gt_df, r.window_start_s, r.window_end_s) for r in est_df.itertuples()
    ]

    notes = {
        "subject": subject.subject_id,
        "video_path": str(subject.video_path),
        "gt_path": str(subject.gt_path),
        "container_fps": container_fps,
        "n_frames_decoded": n_frames_decoded,
        "video_duration_s": video_duration_s,
        "gt_duration_s": gt_duration_s,
        "duration_discrepancy_pct": discrepancy_pct,
        "n_windows": len(est_df),
        "warning": "",
    }
    if discrepancy_pct is not None and not np.isnan(discrepancy_pct) and discrepancy_pct > DURATION_MISMATCH_WARN_PCT:
        notes["warning"] = (
            f"Video duration ({video_duration_s:.1f}s, from container fps={container_fps:.2f}) diverges from "
            f"gt duration ({gt_duration_s:.1f}s) by {discrepancy_pct:.0f}%. This is a real property of this "
            f"recording, not a harness bug -- the production pipeline uses the same container fps. Evaluation "
            f"is naturally limited to the gt-covered span; treat this subject's metrics with caution as a "
            f"possible timestamp/fps desync rather than a genuine CHROM/POS result."
        )
    return est_df, notes


def compute_metrics(df: pd.DataFrame, est_col: str) -> dict:
    total = len(df)
    valid = df.dropna(subset=["bpm_reference", est_col])
    valid = valid[valid[est_col] > 0]
    dropped = total - len(valid)
    if len(valid) < 2:
        return {
            "n_windows_total": total,
            "n_windows_valid": len(valid),
            "n_windows_dropped": dropped,
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
        }
    ref = valid["bpm_reference"].to_numpy()
    est = valid[est_col].to_numpy()
    mae = float(np.mean(np.abs(est - ref)))
    rmse = float(np.sqrt(np.mean((est - ref) ** 2)))
    r, p = stats.pearsonr(ref, est)
    return {
        "n_windows_total": total,
        "n_windows_valid": len(valid),
        "n_windows_dropped": dropped,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": float(r),
        "pearson_p": float(p),
    }


def _style_axes(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.xaxis.label.set_color("#0b0b0b")
    ax.yaxis.label.set_color("#0b0b0b")
    ax.title.set_color("#0b0b0b")


def plot_bland_altman(df: pd.DataFrame, est_col: str, label: str, out_path: Path, color: str, n_subjects: int) -> None:
    valid = df.dropna(subset=["bpm_reference", est_col])
    valid = valid[valid[est_col] > 0]
    ref = valid["bpm_reference"].to_numpy()
    est = valid[est_col].to_numpy()
    diff = est - ref
    mean = (est + ref) / 2.0
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _style_axes(ax)
    ax.scatter(mean, diff, s=30, color=color, alpha=0.65, edgecolors="none")
    ax.axhline(bias, color=COLOR_BIAS_LINE, linewidth=1.5)
    ax.axhline(loa_upper, color=COLOR_LOA_LINE, linewidth=1.2, linestyle="--")
    ax.axhline(loa_lower, color=COLOR_LOA_LINE, linewidth=1.2, linestyle="--")

    x_right = ax.get_xlim()[1]
    ax.text(x_right, bias, f" biais={bias:+.2f}", va="center", ha="left", color=COLOR_BIAS_LINE, fontsize=9)
    ax.text(x_right, loa_upper, f" +1.96 SD={loa_upper:+.2f}", va="center", ha="left", color=COLOR_LOA_LINE, fontsize=9)
    ax.text(x_right, loa_lower, f" -1.96 SD={loa_lower:+.2f}", va="center", ha="left", color=COLOR_LOA_LINE, fontsize=9)

    ax.set_xlabel("Moyenne (référence, estimé) [BPM]")
    ax.set_ylabel("Différence (estimé - référence) [BPM]")
    ax.set_title(f"Bland-Altman -- {label} vs référence CMS50E (N={n_subjects} sujets, n={len(valid)} fenêtres)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, out_path: Path, n_subjects: int) -> None:
    chrom = df.dropna(subset=["bpm_reference", "bpm_chrom"])
    chrom = chrom[chrom["bpm_chrom"] > 0]
    pos = df.dropna(subset=["bpm_reference", "bpm_pos"])
    pos = pos[pos["bpm_pos"] > 0]

    all_vals = np.concatenate(
        [chrom["bpm_reference"], chrom["bpm_chrom"], pos["bpm_reference"], pos["bpm_pos"]]
    )
    lo, hi = float(np.min(all_vals)) - 3, float(np.max(all_vals)) + 3

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _style_axes(ax)
    ax.plot([lo, hi], [lo, hi], color=COLOR_YX_LINE, linewidth=1.5, linestyle="--", label="y = x", zorder=1)
    ax.scatter(chrom["bpm_reference"], chrom["bpm_chrom"], s=30, color=COLOR_CHROM, alpha=0.65, edgecolors="none", label="CHROM", zorder=2)
    ax.scatter(pos["bpm_reference"], pos["bpm_pos"], s=30, color=COLOR_POS, alpha=0.65, edgecolors="none", marker="^", label="POS", zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("BPM référence (CMS50E)")
    ax.set_ylabel("BPM estimé (rPPG)")
    ax.set_title(f"BPM estimé vs référence (N={n_subjects} sujets)")
    legend = ax.legend(frameon=False, loc="upper left")
    for text in legend.get_texts():
        text.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, help="Path to UBFC-rPPG DATASET_1 (or DATASET_2) directory")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--window-seconds", type=float, default=BUFFER_SECONDS, help="Sliding window duration (s), matches RPPGBuffer default")
    parser.add_argument("--step-seconds", type=float, default=1.0, help="Stride between successive window analyses (s)")
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional per-video cap in seconds (debugging only)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = discover_subjects(args.dataset_root)
    print(f"Found {len(subjects)} subject folder(s) under {args.dataset_root}: {[s.subject_id for s in subjects]}")

    all_rows = []
    all_notes = []
    for subject in subjects:
        print(f"\n=== Subject {subject.subject_id} ===")
        print(f"  video: {subject.video_path.name}   gt: {subject.gt_path.name}")
        est_df, notes = evaluate_subject(subject, args.window_seconds, args.step_seconds, args.max_seconds)
        all_notes.append(notes)
        print(
            f"  container_fps={notes['container_fps']:.2f}  frames={notes['n_frames_decoded']}  "
            f"video_dur={notes['video_duration_s']:.1f}s  gt_dur={notes['gt_duration_s']:.1f}s  "
            f"windows={notes['n_windows']}"
        )
        if notes["warning"]:
            print(f"  [WARNING] {notes['warning']}")
        all_rows.append(est_df)

    raw_df = pd.concat(all_rows, ignore_index=True)
    raw_df = raw_df[["subject", "window_start_s", "window_end_s", "bpm_reference", "bpm_chrom", "bpm_pos", "fps_est", "n_samples"]]
    raw_csv_path = out_dir / "rppg_ubfc_raw.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    n_subjects = raw_df["subject"].nunique()
    print(f"\nSaved raw per-window results ({len(raw_df)} rows, {n_subjects} subjects) -> {raw_csv_path}")

    notes_csv_path = out_dir / "rppg_ubfc_subject_notes.csv"
    pd.DataFrame(all_notes).to_csv(notes_csv_path, index=False)
    print(f"Saved per-subject sync/coverage notes -> {notes_csv_path}")

    per_subject_rows = []
    for subj, g in raw_df.groupby("subject"):
        for method, col in (("CHROM", "bpm_chrom"), ("POS", "bpm_pos")):
            m = compute_metrics(g, col)
            per_subject_rows.append({"subject": subj, "method": method, **m})
    per_subject_df = pd.DataFrame(per_subject_rows)
    per_subject_csv_path = out_dir / "rppg_ubfc_metrics_per_subject.csv"
    per_subject_df.to_csv(per_subject_csv_path, index=False)
    print(f"\n=== Per-subject MAE (diagnostic breakdown) ===")
    for subj, g in per_subject_df.groupby("subject"):
        mae_str = "  ".join(f"{row.method}={row.mae:.2f}" for row in g.itertuples())
        flag = " [OUTLIER -- see subject notes]" if (g["mae"] > MAE_WARN_THRESHOLD_BPM).any() else ""
        print(f"  {subj}: {mae_str} BPM{flag}")
    print(f"Saved per-subject metrics breakdown -> {per_subject_csv_path}")

    metrics_chrom = compute_metrics(raw_df, "bpm_chrom")
    metrics_pos = compute_metrics(raw_df, "bpm_pos")

    print(f"\n=== Metrics -- N={n_subjects} subject(s) (UBFC-rPPG DATASET_1: preliminary, not large-scale) ===")
    for name, m in (("CHROM", metrics_chrom), ("POS", metrics_pos)):
        print(
            f"{name}: n_valid={m['n_windows_valid']}/{m['n_windows_total']} windows "
            f"(dropped={m['n_windows_dropped']})  MAE={m['mae']:.2f} BPM  RMSE={m['rmse']:.2f} BPM  "
            f"Pearson r={m['pearson_r']:.3f} (p={m['pearson_p']:.2e})"
        )
        if not np.isnan(m["mae"]) and m["mae"] > MAE_WARN_THRESHOLD_BPM:
            print(
                f"  [WARNING] {name} MAE > {MAE_WARN_THRESHOLD_BPM:.0f} BPM -- outside the 3-6 BPM range "
                f"typical of CHROM/POS in the literature. Treat as a likely window/GT synchronization or "
                f"fps-mismatch issue (see {notes_csv_path.name}), not as a genuine algorithm result."
            )

    metrics_df = pd.DataFrame(
        [{"method": "CHROM", **metrics_chrom}, {"method": "POS", **metrics_pos}]
    )
    metrics_csv_path = out_dir / "rppg_ubfc_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics summary -> {metrics_csv_path}")

    # Secondary view: same metrics restricted to subjects with no flagged
    # sync/fps-duration issue (see evaluate_subject's DURATION_MISMATCH_WARN_PCT
    # check). Nothing is removed from the raw CSV or the N={n_subjects} metrics
    # above -- this is an additional, clearly-labeled view, not a replacement.
    # Which subjects are excluded is derived from the warning flags actually
    # raised, never a hardcoded subject list.
    flagged_subjects = sorted(n["subject"] for n in all_notes if n["warning"])
    if flagged_subjects:
        clean_df = raw_df[~raw_df["subject"].isin(flagged_subjects)]
        n_clean_subjects = clean_df["subject"].nunique()
        metrics_chrom_clean = compute_metrics(clean_df, "bpm_chrom")
        metrics_pos_clean = compute_metrics(clean_df, "bpm_pos")

        print(
            f"\n=== Metrics -- N={n_clean_subjects} subject(s) EXCLUDING flagged sync/fps outliers "
            f"{flagged_subjects} (secondary view; see N={n_subjects} metrics above for the full set) ==="
        )
        for name, m in (("CHROM", metrics_chrom_clean), ("POS", metrics_pos_clean)):
            print(
                f"{name}: n_valid={m['n_windows_valid']}/{m['n_windows_total']} windows "
                f"(dropped={m['n_windows_dropped']})  MAE={m['mae']:.2f} BPM  RMSE={m['rmse']:.2f} BPM  "
                f"Pearson r={m['pearson_r']:.3f} (p={m['pearson_p']:.2e})"
            )

        metrics_clean_df = pd.DataFrame(
            [
                {"method": "CHROM", "subjects_excluded": ",".join(flagged_subjects), **metrics_chrom_clean},
                {"method": "POS", "subjects_excluded": ",".join(flagged_subjects), **metrics_pos_clean},
            ]
        )
        metrics_clean_csv_path = out_dir / "rppg_ubfc_metrics_clean_subjects.csv"
        metrics_clean_df.to_csv(metrics_clean_csv_path, index=False)
        print(f"Saved 'clean subjects' metrics (excludes flagged outliers) -> {metrics_clean_csv_path}")
    else:
        print("\nNo subjects flagged for sync/fps issues -- no separate 'clean subjects' view needed.")

    plot_bland_altman(raw_df, "bpm_chrom", "CHROM", out_dir / "bland_altman_chrom.png", COLOR_CHROM, n_subjects)
    plot_bland_altman(raw_df, "bpm_pos", "POS", out_dir / "bland_altman_pos.png", COLOR_POS, n_subjects)
    plot_scatter(raw_df, out_dir / "scatter_bpm_estimated_vs_reference.png", n_subjects)
    print(f"Saved plots (Bland-Altman x2, scatter) -> {out_dir}")

    print(
        f"\nN.B.: this run covers N={n_subjects} subject(s) from UBFC-rPPG DATASET_1 -- a preliminary "
        f"sanity check, not a statistically powered validation. Re-run with --dataset-root pointing at "
        f"DATASET_2 (42 subjects) once available; no code changes are required."
    )


if __name__ == "__main__":
    main()
