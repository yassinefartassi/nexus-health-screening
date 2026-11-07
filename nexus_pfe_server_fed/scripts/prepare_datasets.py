"""
One-time data curation script: builds the REAL class-per-subfolder train/test
directories fl_server/datasets/{dermatology,ocular}.py and
nexus_pfe/fl_client, nexus_pfe_raspberrypi/fl's data_partition.py expect,
from the raw source datasets on this machine. Run once (or whenever the
source datasets change); output is written under data/ in this repo.

Source datasets (not modified -- this script only reads them and copies a
subset of files elsewhere):
  - DermNet: <PFE_ROOT>/dermnet/train (23 classes, NO normal) and
             <PFE_ROOT>/dermnet/test  (24 classes, INCLUDING normal)
  - Ocular:  <PFE_ROOT>/Image Dataset on Eye Diseases Classification
             (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and
             SMOTE Validation/<Cataract|Conjunctivitis|Eyelid|Normal|Uveitis>
             (5 class folders, no train/test split)

What this script does, in order:
  1. HARD GATE: verifies sorted(target 11 dermatology class names) equals
     fl_server/dermatology_class_names.json exactly. If not, aborts before
     copying anything -- a mismatch here means a permuted label map, which
     silently produces near-random accuracy on an otherwise-correct model.
  2. Dermatology: copies the 10 non-"normal" curated classes from
     dermnet/train -> data/dermatology_curated/train/<class> and
     dermnet/test -> data/dermatology_curated/test/<class> unchanged (the
     source dataset's own train/test split is preserved for these classes).
     "normal" only exists in dermnet/test (300 images, no source in
     dermnet/train) -- see NORMAL ASYMMETRY below.
  3. Ocular: no source train/test split exists at all. Files named
     "aug_*.jpg" are the dataset's own augmented/synthetic images (verified
     by inspection: every class's "real" files are plain numeric names like
     "1.jpg", "106(1).jpg"; every class's synthetic files start "aug_").
     Real images only are split 80/20 train/test (seeded); "aug_*" synthetic
     images are added to the train split only, never test.
  4. Every file is validated (PIL open + full pixel decode) at copy time --
     files that fail are never copied, and are counted as "removed" in the
     per-class report. Non-image extensions and zero-byte files are skipped
     before ever attempting to open them.

NORMAL ASYMMETRY: dermnet/train has no "normal" class at all (DermNet-NZ's
own directory structure never included one; only the curated test split
added it). No other normal-skin source exists on this machine (searched
system-wide). So dermnet/test/normal (300 images) is the ONLY normal-skin
source available, and per the project's explicit resolution, it's what gets
split: NORMAL_SPLIT_SEED-seeded, shuffled, 80/20 -- NOT the same 20% held
back as "test" that the other 10 classes already have from DermNet's own
split (that split is fixed and untouched), but a comparable ~20% fraction so
"normal" behaves like every other class in both curated dirs.

Seeds: NORMAL_SPLIT_SEED and OCULAR_SPLIT_SEED are curation-time constants,
unrelated to (and intentionally different from) fl_client/data_partition.py's
non-IID partitioning seeds (42/43 there) -- this script decides which RAW
FILES exist in the curated train/test directories; the FL clients' Dirichlet
non-IID split then happens on top of whatever's in data/*_curated/train.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# Where the raw source datasets live (see DATA.md at the repo root).
# Defaults to <repo>/data/raw; override with NEXUS_RAW_DATA_DIR.
PFE_ROOT = Path(os.environ.get("NEXUS_RAW_DATA_DIR", PROJECT_ROOT.parent / "data" / "raw"))

DERMNET_TRAIN_SRC = PFE_ROOT / "dermnet" / "train"
DERMNET_TEST_SRC = PFE_ROOT / "dermnet" / "test"
OCULAR_SRC = (
    PFE_ROOT
    / "Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, "
    "Cataract, Eyelid) with Symptoms and SMOTE Validation"
)

DERMATOLOGY_OUT = PROJECT_ROOT / "data" / "dermatology_curated"
OCULAR_OUT = PROJECT_ROOT / "data" / "ocular_curated"

DERMATOLOGY_CLASS_NAMES_PATH = PROJECT_ROOT / "fl_server" / "dermatology_class_names.json"
OCULAR_CLASS_NAMES_PATH = PROJECT_ROOT / "fl_server" / "ocular_class_names.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# 10 non-"normal" curated classes -- exact folder names, must match both
# dermnet/{train,test}'s subfolder names AND dermatology_class_names.json.
DERMATOLOGY_CURATED_CLASSES = [
    "Acne and Rosacea Photos",
    "Atopic Dermatitis Photos",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Nail Fungus and other Nail Disease",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Warts Molluscum and other Viral Infections",
]
NORMAL_CLASS = "normal"

NORMAL_SPLIT_SEED = 42
NORMAL_TEST_FRACTION = 0.20  # matches the ~20% test fraction DermNet's own split uses for 9/10 curated classes

OCULAR_SPLIT_SEED = 42
OCULAR_TEST_FRACTION = 0.20
OCULAR_CLASSES = ["Cataract", "Conjunctivitis", "Eyelid", "Normal", "Uveitis"]


def is_valid_image(path: Path) -> bool:
    """PIL open + full pixel decode. verify() alone only checks structural
    validity without decoding pixel data, so a truncated JPEG can pass
    verify() and still fail here -- .load() forces the real decode."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def list_valid_source_files(directory: Path) -> tuple[list[Path], int, int]:
    """Returns (valid_image_paths, non_image_or_zero_byte_count, corrupt_count)."""
    valid: list[Path] = []
    skipped_non_image = 0
    skipped_corrupt = 0
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in IMAGE_EXTENSIONS or entry.stat().st_size == 0:
            skipped_non_image += 1
            continue
        if is_valid_image(entry):
            valid.append(entry)
        else:
            skipped_corrupt += 1
    return valid, skipped_non_image, skipped_corrupt


def copy_files(paths: list[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in paths:
        shutil.copy2(src, dest_dir / src.name)


def verify_class_order_gate() -> list[str]:
    target_11 = sorted(DERMATOLOGY_CURATED_CLASSES + [NORMAL_CLASS])
    with open(DERMATOLOGY_CLASS_NAMES_PATH, encoding="utf-8") as f:
        expected = json.load(f)
    print("=== Task 1 gate: curated class order vs dermatology_class_names.json ===")
    for i, (a, b) in enumerate(zip(target_11, expected)):
        marker = "OK" if a == b else "MISMATCH"
        print(f"  [{marker}] {i}: curated={a!r}  expected={b!r}")
    if target_11 != expected:
        print("\nABORT: sorted curated class list does NOT match dermatology_class_names.json.")
        print("This would silently permute labels and produce near-random accuracy on a correct model.")
        sys.exit(1)
    print("PASS: sorted curated class list matches dermatology_class_names.json exactly.\n")
    return expected


def curate_dermatology() -> None:
    print("=== Dermatology curation ===")
    report = []

    for class_name in DERMATOLOGY_CURATED_CLASSES:
        for split_name, src_root in (("train", DERMNET_TRAIN_SRC), ("test", DERMNET_TEST_SRC)):
            src_dir = src_root / class_name
            if not src_dir.is_dir():
                print(f"ABORT: expected source class dir missing: {src_dir}")
                sys.exit(1)
            valid, skipped_non_image, skipped_corrupt = list_valid_source_files(src_dir)
            before = len(valid) + skipped_non_image + skipped_corrupt
            copy_files(valid, DERMATOLOGY_OUT / split_name / class_name)
            report.append((class_name, split_name, before, len(valid), skipped_non_image, skipped_corrupt))
            print(
                f"  {class_name!r:60s} [{split_name:5s}] before={before:5d} "
                f"copied={len(valid):5d} non_image/zero_byte={skipped_non_image:3d} corrupt={skipped_corrupt:3d}"
            )

    # "normal": ONLY source is dermnet/test/normal (300 images) -- see module
    # docstring's NORMAL ASYMMETRY section. Validate first, THEN split, so the
    # 80/20 split is over genuinely-usable images only.
    normal_src_dir = DERMNET_TEST_SRC / NORMAL_CLASS
    valid, skipped_non_image, skipped_corrupt = list_valid_source_files(normal_src_dir)
    before = len(valid) + skipped_non_image + skipped_corrupt
    print(f"\n  'normal' source: {normal_src_dir}")
    print(f"    before={before} valid={len(valid)} non_image/zero_byte={skipped_non_image} corrupt={skipped_corrupt}")

    rng = random.Random(NORMAL_SPLIT_SEED)
    shuffled = valid[:]
    rng.shuffle(shuffled)
    n_test = round(len(shuffled) * NORMAL_TEST_FRACTION)
    test_files = shuffled[:n_test]
    train_files = shuffled[n_test:]

    copy_files(train_files, DERMATOLOGY_OUT / "train" / NORMAL_CLASS)
    copy_files(test_files, DERMATOLOGY_OUT / "test" / NORMAL_CLASS)
    report.append((NORMAL_CLASS, "train", before, len(train_files), skipped_non_image, skipped_corrupt))
    report.append((NORMAL_CLASS, "test", before, len(test_files), 0, 0))
    print(
        f"    seed={NORMAL_SPLIT_SEED} shuffle -> train={len(train_files)} ({1 - NORMAL_TEST_FRACTION:.0%}) "
        f"/ test={len(test_files)} ({NORMAL_TEST_FRACTION:.0%})"
    )

    train_total = sum(r[3] for r in report if r[1] == "train")
    test_total = sum(r[3] for r in report if r[1] == "test")
    total_corrupt = sum(r[5] for r in report)
    total_non_image = sum(r[4] for r in report)
    print(f"\nDermatology curated totals: train={train_total} test={test_total} "
          f"total_removed(non_image+corrupt)={total_non_image + total_corrupt} "
          f"(non_image={total_non_image}, corrupt={total_corrupt})\n")


def curate_ocular() -> None:
    print("=== Ocular curation (no source train/test split -- building one) ===")
    with open(OCULAR_CLASS_NAMES_PATH, encoding="utf-8") as f:
        expected = json.load(f)
    if OCULAR_CLASSES != expected:
        print(f"ABORT: OCULAR_CLASSES {OCULAR_CLASSES} != ocular_class_names.json {expected}")
        sys.exit(1)

    train_total = 0
    test_total = 0
    total_corrupt = 0
    total_non_image = 0

    for class_name in OCULAR_CLASSES:
        src_dir = OCULAR_SRC / class_name
        if not src_dir.is_dir():
            print(f"ABORT: expected source class dir missing: {src_dir}")
            sys.exit(1)
        valid, skipped_non_image, skipped_corrupt = list_valid_source_files(src_dir)
        before = len(valid) + skipped_non_image + skipped_corrupt

        real_files = [p for p in valid if not p.name.lower().startswith("aug_")]
        synthetic_files = [p for p in valid if p.name.lower().startswith("aug_")]

        rng = random.Random(OCULAR_SPLIT_SEED)
        shuffled_real = real_files[:]
        rng.shuffle(shuffled_real)
        n_test = round(len(shuffled_real) * OCULAR_TEST_FRACTION)
        test_files = shuffled_real[:n_test]           # real images only, never synthetic
        train_files = shuffled_real[n_test:] + synthetic_files  # remaining real + all synthetic

        copy_files(train_files, OCULAR_OUT / "train" / class_name)
        copy_files(test_files, OCULAR_OUT / "test" / class_name)

        train_total += len(train_files)
        test_total += len(test_files)
        total_corrupt += skipped_corrupt
        total_non_image += skipped_non_image

        print(
            f"  {class_name:15s} before={before:4d} valid={len(valid):4d} "
            f"(real={len(real_files):4d}, aug/synthetic={len(synthetic_files):4d}) "
            f"-> train={len(train_files):4d} (real {len(shuffled_real) - n_test} + aug {len(synthetic_files)}) "
            f"test={len(test_files):4d} (real only) "
            f"non_image/zero_byte={skipped_non_image} corrupt={skipped_corrupt}"
        )

    print(f"\nOcular curated totals: train={train_total} test={test_total} "
          f"total_removed(non_image+corrupt)={total_non_image + total_corrupt} "
          f"(non_image={total_non_image}, corrupt={total_corrupt})\n")


def main() -> None:
    verify_class_order_gate()
    curate_dermatology()
    curate_ocular()
    print("Done. Curated data:")
    print(f"  {DERMATOLOGY_OUT}")
    print(f"  {OCULAR_OUT}")


if __name__ == "__main__":
    main()
