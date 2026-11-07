"""
Ground-truth loader for UBFC-rPPG DATASET_1.

Each subject folder (e.g. `6-gt/`) holds:
  - a video file: `vid.avi` or `vid-XXX.avi`
  - `gtdump.xmp`: a headerless CSV dumped by the CMS50E pulse-oximeter
    software, sampled at ~62.5 Hz (~16 ms/row), with columns:
        1. timestamp_ms   - milliseconds since recording start
        2. hr_bpm         - device-computed heart rate (beats/min)
        3. spo2           - blood oxygen saturation (%)
        4. ppg            - raw PPG waveform sample

Column order confirmed against the DATASET_1 files (6/8/10/11-gt): column 2
is stable and physiologically plausible (updates roughly once per beat)
while column 4 oscillates every row, consistent with the known UBFC-rPPG
DATASET_1 format.

Assumption (documented, not verified per-subject): the video and gtdump.xmp
clocks both start at t=0 when recording begins, as is standard for this
dataset. There is no independent sync signal in DATASET_1 to cross-check
this against.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

GT_COLUMNS = ["timestamp_ms", "hr_bpm", "spo2", "ppg"]


def parse_gtdump(path: str | Path) -> pd.DataFrame:
    """Parse a gtdump.xmp file into a DataFrame with an added timestamp_s column."""
    df = pd.read_csv(path, header=None, names=GT_COLUMNS)
    df["timestamp_s"] = df["timestamp_ms"] / 1000.0
    return df


@dataclass
class Subject:
    subject_id: str
    video_path: Path
    gt_path: Path


def discover_subjects(dataset_root: str | Path) -> List[Subject]:
    """Walk `dataset_root` for subject folders containing one gtdump.xmp and
    one *.avi file each. Does not assume a fixed subject count or naming
    scheme, so this also works against DATASET_2 (42 subjects) unchanged."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    subjects = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        gt_files = sorted(d.glob("gtdump.xmp"))
        vid_files = sorted(d.glob("*.avi"))
        if len(gt_files) == 1 and len(vid_files) == 1:
            subjects.append(Subject(subject_id=d.name, video_path=vid_files[0], gt_path=gt_files[0]))

    if not subjects:
        raise FileNotFoundError(
            f"No subject folders with both gtdump.xmp and a *.avi file found under {root}"
        )
    return subjects
