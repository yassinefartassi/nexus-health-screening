# NEXUS_PFE — Multimodal Health Screening Test Bench

> **2026-07 hardening pass note**: this project previously shipped an unrelated,
> stale dermatology model (EfficientNetB0, 24 classes, ~39.9% test accuracy) with
> a double-normalization preprocessing bug (ImageNet mean/std applied on top of
> the model's own baked-in preprocessing). Both issues are now fixed: dermatology
> uses the canonical model from `dermatho-assets` — EfficientNetV2-S, 11 classes,
> ~61.6% test accuracy — with raw `[0,255]` pixel input (no external normalization).

Desktop/Ubuntu validation environment for the multimodal edge AI health-screening
platform. This is **Phase 1 of deployment testing**: prove all three models produce
correct, consistent output on a regular desktop/laptop before porting the same
pipeline to the Raspberry Pi 5 and, later, the Flutter mobile app.

```
Desktop / Ubuntu (this project)  →  Raspberry Pi 5  →  Flutter mobile app  →  Federated Learning
        (validate)                    (deploy)            (deploy)              (this project is also an FL client, see below)
```

## What this project does

A Streamlit UI that lets you run all three health-screening modules against a
webcam feed or uploaded images/video, so outputs can be sanity-checked on a normal
machine before touching embedded hardware:

| Module | Model | Input | Output | Status |
|---|---|---|---|---|
| **Ocular** | Pretrained ONNX (Kaggle-trained) | Eye image | Condition class + confidence | Model present |
| **Dermatology** | EfficientNetV2-S → ONNX, 11 classes (10 curated conditions + normal) | Skin image | Condition class + confidence, ~61.6% test accuracy | Model present |
| **rPPG (HRV/BPM)** | CHROM & POS algorithms (rPPG-Toolbox) | Webcam video / face ROI | Heart rate (BPM), HRV estimate (SDNN, RMSSD) | **Implemented** |

This project includes both the inference pipeline above **and** a federated
learning (FL) client (`fl_client/`) that trains the dermatology/ocular models
locally and sends only weight updates to `nexus_pfe_server_fed`'s FedAvg
server — see "Federated learning" below.

## Project structure

```
NEXUS_PFE/
├── README.md
├── requirements.txt
├── requirements-fl.txt               # extra deps for fl_client/ only (flwr, tensorflow, torch)
├── app.py                            # Streamlit entry point (4 tabs: ocular, dermatology, rPPG, Apprentissage fédéré)
├── fl_client/                        # federated learning CLIENT (see "Federated learning" below)
│   ├── client.py                     # Flower NumPyClient, `python -m fl_client.client`
│   ├── models.py                     # per-module model builders (mirrors fl_server/modules/)
│   ├── data_partition.py             # seeded non-IID Dirichlet split (simulated, not real users)
│   ├── torch_ops.py                  # PyTorch ops for the ocular client (get/set weights, fit, evaluate)
│   └── status_client.py              # HTTP client for the FL server's read-only status API
├── models/
│   ├── eye_disease_model.onnx        # ocular model (self-contained, no external .onnx.data sidecar)
│   ├── eye_disease_class_names.json  # 5 classes (Cataract, Conjunctivitis, Eyelid, Normal, Uveitis)
│   ├── dermnet_model.onnx            # dermatology model
│   ├── dermnet_class_names.pkl       # dermatology's 11 class labels (renamed from class_names.pkl)
│   ├── history.pkl                   # dermatology training history
│   └── mediapipe/
│       └── face_landmarker.task      # MediaPipe FaceLandmarker model bundle (auto-downloaded on first run)
├── src/
│   ├── inference/
│   │   ├── ocular_infer.py           # ONNX Runtime wrapper, ocular model
│   │   ├── dermatology_infer.py      # ONNX Runtime wrapper, dermatology model
│   │   ├── rppg_algorithms.py        # CHROM & POS, adapted from rPPG-Toolbox (pure functions)
│   │   └── rppg_infer.py             # buffers RGB traces, runs CHROM/POS, BPM + HRV extraction
│   ├── preprocessing/
│   │   ├── image_utils.py            # resize/normalize helpers shared by both ONNX models
│   │   └── face_roi.py               # MediaPipe-based face ROI tracking for rPPG
│   └── utils/
│       └── onnx_runner.py            # shared ONNX Runtime session loader
├── data/
│   └── sample_images/                # a few test images per module, for smoke-testing
├── tests/
│   └── test_inference.py             # confirms each model loads/produces valid output, rPPG runs end-to-end
└── docs/
    └── model_cards.md                # accuracy, dataset, known limitations per model
```

> **Naming collision, resolved**: dermatology's class labels used to live in
> a generically-named `class_names.pkl`, which risked being confused with
> ocular's label set. It's been renamed to `dermnet_class_names.pkl`, and
> ocular's 5-class list now has its own file, `eye_disease_class_names.json`.
> If you retrain either model, double check the label file order still matches
> the model's output layer order — a mismatch doesn't throw an error, it
> silently mislabels every prediction.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` (core dependencies):
```
streamlit
onnxruntime
opencv-python
numpy
pillow
mediapipe        # face ROI detection for rPPG
scipy            # signal filtering for CHROM/POS
requests          # "Apprentissage fédéré" tab polls the FL server's status API
```

Only needed if you're going to run the federated learning client
(`python -m fl_client.client`) from this machine, on top of the above:

```bash
pip install -r requirements-fl.txt   # flwr, tensorflow, torch, torchvision
```

## Running

```bash
streamlit run app.py
```

Opens a local UI with three tabs — one per module. Ocular and dermatology each
accept a webcam snapshot or a file upload and display the model's prediction
alongside confidence scores. rPPG streams the live webcam feed, overlays the
detected face ROI (forehead + both cheeks), and once ~10s of signal has been
buffered, shows a running BPM and HRV (SDNN/RMSSD) estimate with CHROM vs POS
plotted side by side. The rPPG tab uses a simple blocking OpenCV capture loop
(no `streamlit-webrtc`), so other tabs won't re-render while "Start webcam" is
checked.

## Validation checklist for this phase

Before moving to the Raspberry Pi 5, confirm on desktop:

- [ ] Ocular model loads via ONNX Runtime and returns expected output shape/classes
- [ ] Dermatology model loads via ONNX Runtime and returns expected output shape/classes
- [x] rPPG CHROM/POS pipeline extracts a plausible BPM end-to-end (see `tests/test_inference.py`)
- [ ] rPPG BPM validated against a pulse oximeter or manual pulse count on a live webcam feed (not yet done — the automated test only checks the pipeline against a synthetic clip, not real physiological ground truth)
- [ ] Same test image produces the same prediction across repeated runs (determinism check)
- [ ] Inference latency per module is logged (baseline to compare against RPi5 later)

## Federated learning

`fl_client/` is this app's **federated learning client** — a reusable Flower
`NumPyClient` that trains dermatology (TensorFlow/Keras, EfficientNetV2-S) or
ocular (PyTorch, ResNet-18) locally on a small, seeded, **simulated** non-IID
data partition, then sends only the resulting weight arrays to
`nexus_pfe_server_fed`'s FedAvg server. **No raw image is ever sent** — see
`fl_client/client.py`'s module docstring for the full privacy argument; this
is the same guarantee the FL tab in `app.py` explains to the user.

This is a **separate concern from the Streamlit app's inference tabs**:
`app.py` only ever loads the deployed `.onnx` files in `models/` for
inference, never trains anything itself, and `fl_client/` only ever trains
locally and talks to the FL server — the two share no runtime state. `app.py`
does read the FL server's real round history (via `fl_client/status_client.py`)
to render the "Apprentissage fédéré" tab, but that's a status read, not a
training trigger.

### Install

```bash
pip install -r requirements-fl.txt   # flwr, tensorflow, torch, torchvision — on top of requirements.txt
```

### Run a client against `nexus_pfe_server_fed`

1. Start the FL server on the server machine (see that repo's
   `docs/fl_setup.md`) and note its LAN IP.
2. From this machine:
   ```bash
   python -m fl_client.client --client-id 0 --server <server-ip>:8080 --model dermatology --num-clients 2
   ```
   `--model ocular` federates the ocular module instead. `--client-id` picks
   which seeded non-IID shard (`fl_client/data_partition.py`, Dirichlet
   split, fixed seed — reproducible, **not** a real multi-user database) this
   process trains on; run a second `--client-id 1` process (here or on
   another machine, e.g. `nexus_pfe_raspberrypi`) to reach the server's
   `--min-clients 2` default and let round 1 start.
3. Watch this process's console for `[web client N] fit -> loss=... acc=...`
   per round, and the server's console/CSV log for the aggregated,
   server-side held-out accuracy — that second number, not this client's own
   training accuracy, is the thesis convergence metric.

### End-to-end round, in one place

Server: loads the pretrained checkpoint as round 0 → waits for
`--min-clients` clients → FedAvg-aggregates each round's weight updates →
evaluates the aggregated model on its held-out set → versions it
(`models/<module>/global_v{round}.<ext>`) → **re-exports it to ONNX**
(`.onnx`, via `tf2onnx` for dermatology or `torch.onnx.export` for ocular).
That last step is not optional: `app.py`'s inference tabs only ever load the
`.onnx` files in this project's own `models/` directory, and those are
**not** automatically updated by an FL round on the server — an operator
must explicitly copy the server's freshly re-exported
`models/<module>/global_v{round}.onnx` over this project's
`models/dermnet_model.onnx` / `models/eye_disease_model.onnx` to actually
deploy an FL-improved model here. Skipping either the server's re-export
step or that copy silently leaves inference on stale weights.

### Multi-framework, by design

Dermatology (TensorFlow/Keras) and ocular (PyTorch) are federated
independently, each with its own versioned global model — this project never
converts one framework's weights into the other's. `fl_client/models.py`'s
`MODEL_REGISTRY` and `fl_server/modules/base.py`'s `FederatedModule`
abstraction (server-side) are what make this possible without duplicating
the round/training logic per framework. See
`nexus_pfe_server_fed/docs/fl_setup.md`'s "Mixed frameworks" section for the
full rationale.

### "Apprentissage fédéré" tab

`app.py`'s fourth tab explains the FedAvg cycle (client → weights → server →
aggregation → global model → redistribution), restates the privacy
guarantee, and polls `nexus_pfe_server_fed`'s read-only status API
(`fl_server/status_api.py`, default `http://127.0.0.1:8090`) to show **real**
round/accuracy numbers read from the server's own logs — if no round has run
yet, or the status API isn't reachable, it says so ("en attente
d'exécution") instead of showing a fabricated chart.

## Next phases (not in this project)

1. **Raspberry Pi 5**: same `src/inference/` modules, same ONNX files — confirm
   identical output to desktop on the same test inputs before trusting the
   embedded deployment. Also runs its own FL client, `nexus_pfe_raspberrypi/fl/`.
2. **Flutter mobile app**: models converted/bundled for on-device inference
   (ONNX Runtime Mobile or TFLite); UI rebuilt natively, this Streamlit app is
   desktop-only tooling and won't be ported directly. Its FL role is
   explicitly **not** on-device training — see `nexus_pfe_mobile/README.md`.

## Licensing note

The rPPG CHROM/POS implementation is adapted from
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox), released under the
Responsible AI License (RAIL) — review the license terms before any use beyond
academic/research purposes, and cite the toolbox's paper in any written
deliverable that uses it.

## Model status (as of this phase)

- **Ocular**: pretrained, self-contained `.onnx` present in `models/` (no external
  `.onnx.data` sidecar). Original training results: **~95.9% accuracy** across 5
  classes (Cataract, Conjunctivitis, Eyelid, Normal, Uveitis; 344-sample confusion
  matrix), all per-class F1 ≥ 0.93. Notably stronger than the dermatology model,
  likely because 5 well-separated classes are inherently easier than 11
  visually-overlapping skin conditions. Known caveats: confirm whether this was
  measured on a true held-out test set (same rigor as the dermatology model) or
  validation split, and note the class imbalance (Conjunctivitis/Uveitis ~42-43
  samples vs. Normal's 94). Re-validation inside this specific pipeline (to catch
  any ONNX conversion issues) — TODO.
- **Dermatology**: EfficientNetV2-S, 11 classes (10 curated non-cancerous skin
  conditions + normal), **61.58% test accuracy** (precision 0.79 / recall 0.46
  macro), trained 42 epochs, fine-tuned. See `docs/model_cards.md` for the full
  per-class precision/recall breakdown and known weak classes (Psoriasis is the
  weakest, most confused with Tinea/Ringworm/Fungal Infections). `.onnx` file
  confirmed present in `models/`.
- **rPPG**: CHROM & POS only, no trainable weights — deterministic signal
  processing. Module code implemented: `src/inference/rppg_algorithms.py`
  (CHROM/POS ported from rPPG-Toolbox), `src/preprocessing/face_roi.py`
  (MediaPipe FaceLandmarker-based ROI extraction), `src/inference/rppg_infer.py`
  (buffering, bandpass filtering, FFT-based BPM, peak-based HRV). Verified
  end-to-end on a synthetic test clip (`tests/test_inference.py`) with a known
  injected pulse frequency; **not yet benchmarked against real ground-truth HR**
  (pulse oximeter / manual count) — see validation checklist above.