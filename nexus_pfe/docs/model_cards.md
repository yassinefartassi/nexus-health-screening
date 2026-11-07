# Model Cards

Per-module accuracy, dataset, and known-limitation notes for the three
modules validated in this desktop test bench. See `README.md` for the
overall project structure and phase context.

---

## Ocular — eye disease classifier

- **Task**: 5-class image classification (Cataract, Conjunctivitis, Eyelid,
  Normal, Uveitis).
- **Model**: pretrained, exported to ONNX (`models/eye_disease_model.onnx`,
  self-contained — no external `.onnx.data` sidecar). Input: NCHW float32,
  224x224, ImageNet-normalized.
- **Labels**: `models/eye_disease_class_names.json`.
- **Reported accuracy**: ~95.9% across 5 classes (344-sample confusion
  matrix), all per-class F1 ≥ 0.93.
- **Known limitations**:
  - Not yet confirmed whether the reported accuracy was measured on a true
    held-out test set or a validation split — same rigor question applies
    here as to the dermatology model below.
  - Class imbalance in the original training set: Conjunctivitis/Uveitis
    (~42-43 samples) vs. Normal (94 samples).
  - Re-validation inside this specific pipeline (to catch any ONNX
    conversion issues) — not yet done.
  - Preprocessing mode (`imagenet` normalization) is inherited from the
    original training pipeline's assumed convention; not independently
    re-verified against the training notebook.

---

## Dermatology — skin disease classifier

- **Task**: 11-class image classification (10 curated non-cancerous skin
  conditions + normal): Acne and Rosacea Photos, Atopic Dermatitis Photos,
  Hair Loss Photos Alopecia and other Hair Diseases, Herpes HPV and other
  STDs Photos, Light Diseases and Disorders of Pigmentation, Nail Fungus and
  other Nail Disease, Psoriasis pictures Lichen Planus and related diseases,
  Scabies Lyme Disease and other Infestations and Bites, Tinea Ringworm
  Candidiasis and other Fungal Infections, Warts Molluscum and other Viral
  Infections, normal.
- **Model**: EfficientNetV2-S backbone, fine-tuned (42 epochs), exported to
  ONNX (`models/dermnet_model.onnx`). Input: raw `[0,255]` pixel values, no
  external normalization — the model's own `Rescaling` layer (a `Mul` by
  `1/128` and `Add` of `-1.0` immediately after the input) is baked into the
  ONNX graph and handles preprocessing internally.
- **Labels**: `models/dermnet_class_names.pkl` (renamed from the generic
  `class_names.pkl` — see README naming-collision note).
- **Reported accuracy**: 61.58% test accuracy (precision 0.79 / recall 0.46,
  macro-averaged). Source: `dermatho-assets/classification_report.txt` and
  `history.pkl`.
- **Known weak classes**: Psoriasis is the weakest class (precision 0.47,
  recall 0.33, F1 0.39), most confused with Tinea/Ringworm/Fungal Infections
  (27 of 150 psoriasis test images misclassified as that class), tied with
  Atopic Dermatitis and Warts/Molluscum (15 each).
- **Known limitations**:
  - Materially lower accuracy than the ocular model, most likely because 11
    visually-overlapping skin conditions are a harder discrimination problem
    than ocular's 5 well-separated ones, not necessarily a modeling defect.
  - Full per-class precision/recall breakdown from the original training run
    is not reproduced in this repo — refer to
    `dermatho-assets/classification_report.txt` and the training history
    (`models/history.pkl`) if deeper analysis is needed.

---

## rPPG — heart rate (BPM) & HRV from webcam video

- **Task**: contactless heart-rate and heart-rate-variability estimation
  from a webcam feed, no trained weights — deterministic signal processing.
- **Pipeline**:
  1. `src/preprocessing/face_roi.py` — MediaPipe FaceLandmarker (478-point
     face mesh) locates the face each frame and extracts mean R/G/B over
     three landmark-relative skin ROIs (forehead, left cheek, right cheek).
  2. `src/inference/rppg_infer.py` — buffers ~10s of per-frame RGB samples,
     runs both CHROM and POS (`src/inference/rppg_algorithms.py`) on the
     buffered trace, bandpass-filters each resulting BVP signal to 0.7-4 Hz
     (42-240 BPM), and extracts BPM via an FFT peak search restricted to a
     40-180 BPM plausibility window.
  3. HRV (SDNN, RMSSD) is computed from peak-to-peak intervals detected on
     the filtered POS signal.
- **Algorithms**: CHROM (De Haan & Jeanne, 2013) and POS (Wang et al., 2017),
  ported from [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) as
  pure functions operating on an already-extracted RGB trace (the toolbox's
  dataloader/training framework was not pulled in). See the license note
  below and the citation header in `rppg_algorithms.py`.
- **Validation status**:
  - Verified end-to-end (face ROI → buffer → CHROM/POS → BPM/HRV) on a
    synthetic test clip with a known injected pulse frequency
    (`tests/test_inference.py`) — confirms the pipeline runs without
    crashing and recovers a BPM within ~2 BPM of the injected frequency
    (limited by ~0.1 Hz / 6 BPM FFT resolution over a 10s window).
  - **Not yet benchmarked against real ground-truth HR** (pulse oximeter or
    manual pulse count) on an actual live webcam feed — this is the
    remaining item on the validation checklist in `README.md`.
- **Known limitations**:
  - Forehead ROI accuracy degrades for hairstyles with heavy fringe/bangs
    covering the forehead (verified during development against a test image
    with this exact issue) — the cheek ROIs are the more reliable signal
    source in that case.
  - BPM/HRV accuracy depends on stable lighting, minimal motion, and a
    front-facing pose; no motion compensation is implemented.
  - The rPPG Streamlit tab uses a simple blocking OpenCV capture loop (no
    `streamlit-webrtc`), so it is single-threaded UI-wise — acceptable for
    this desktop validation phase but not the final architecture.
  - HRV metrics (SDNN, RMSSD) can return `NaN` when too few clean peaks are
    detected in a given 10s window (e.g. noisy signal, poor lighting) — this
    is surfaced in the UI rather than silently defaulting to 0.

### Licensing

The rPPG CHROM/POS implementation is adapted from
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox), released under
the Responsible AI License (RAIL) — review the license terms
([LICENSE](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/LICENSE)) before any use beyond academic/research
purposes, and cite the toolbox's paper in any written deliverable that uses
it:

- De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based
  rPPG. *IEEE Transactions on Biomedical Engineering, 60*(10), 2878-2886.
- Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
  Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical
  Engineering, 64*(7), 1479-1491.
