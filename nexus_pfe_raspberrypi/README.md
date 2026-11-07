# NEXUS_PFE — Raspberry Pi 5 Deployment + Federated Learning PoC

> **2026-07 hardening pass note**: this project previously shipped a stale,
> mismatched dermatology model (EfficientNetB0, 24 classes, ~39.9% test
> accuracy) with a double-normalization preprocessing bug, while `fl/` trained
> toward a different, newer model. Both are now fixed: the deployed model
> (`models/dermnet_model.onnx`) and the FL training target are the same
> canonical model — EfficientNetV2-S, 11 classes, ~61.6% test accuracy, raw
> `[0,255]` preprocessing. See `docs/deployment.md` for what this means for FL
> checkpoint exports.

Phase 2 of the NEXUS_PFE multimodal health-screening platform: the same
three-module inference stack validated on Ubuntu desktop (`nexus_pfe`), now
running unmodified on a Raspberry Pi 5, plus a first federated learning (FL)
proof of concept for the dermatology module.

```
Desktop / Ubuntu (nexus_pfe)  →  Raspberry Pi 5 (this project)  →  Flutter mobile app
        (validate)                 (deploy + FL PoC)                   (deploy)
```

## What's new here vs. the desktop project

1. **Camera abstraction** (`src/camera/camera_source.py`) -- the exact same
   `src/inference/` and `src/preprocessing/` modules now run against a USB
   webcam, a phone IP camera (MJPEG/RTSP), or the RPi5's `rpicam`
   (`picamera2`/libcamera), selected via `NEXUS_CAMERA_SOURCE` -- inference
   code never knows or cares which one is active.
2. **MediaPipe -> OpenCV Haar-cascade fallback** (`src/preprocessing/face_roi.py`)
   -- MediaPipe has no wheel for Python 3.13 on ARM64, so it can't be
   installed on the RPi5 at all. `face_roi.py` detects this automatically
   (no flag) and falls back to `cv2.CascadeClassifier` with a vendored
   `models/haarcascades/haarcascade_frontalface_default.xml`, returning the
   same `FaceROIResult` shape either way so `rppg_infer.py` is unaffected.
   See `docs/deployment.md` section 1.
3. **Federated learning proof of concept** (`fl/`) -- a Flower FedAvg server
   + client wrapping the dermatology model, with non-IID simulated data
   shards (no real multi-device patient data exists yet). Only model weight
   updates cross the network; see `fl/client.py`'s privacy note. After each
   run, `fl/export_to_onnx.py` re-exports the aggregated Keras checkpoint to
   ONNX -- an explicit step, not automatic (see `docs/deployment.md` step 4).
4. `docs/deployment.md` -- SSH-deploying to the RPi5 and running one FL
   round end-to-end across the Ubuntu machine and the RPi5.

**Inference logic itself is untouched**: `src/inference/`,
`src/preprocessing/`, `src/utils/` are the same files as `nexus_pfe`, reused
as-is (see `docs/model_cards.md`, copied unmodified).

## Project structure

```
nexus_pfe_raspberrypi/
├── README.md
├── docs/
│   ├── deployment.md              # RPi5 SSH deploy + FL round walkthrough
│   └── model_cards.md             # unchanged from nexus_pfe
├── requirements.txt                # core inference + Streamlit (same as nexus_pfe)
├── requirements-fl.txt             # + Flower, TensorFlow, pandas, matplotlib
├── requirements-rpi5.txt           # RPi5-specific notes (picamera2, aarch64 wheels)
├── app.py                          # Streamlit entry point, camera source swapped in for rPPG
├── models/                         # same ONNX weights + class labels as nexus_pfe
│   └── haarcascades/                # vendored Haar cascade for the RPi5 face-ROI fallback
├── src/
│   ├── camera/
│   │   └── camera_source.py        # CameraSource: WebcamSource / IPCameraSource / RpicamSource + factory
│   ├── inference/                  # unchanged from nexus_pfe (ocular, dermatology, rPPG)
│   ├── preprocessing/              # face_roi.py: MediaPipe, auto-falls back to OpenCV Haar cascade
│   └── utils/                      # unchanged from nexus_pfe
├── fl/
│   ├── server.py                   # LOCAL-ONLY smoke-test FedAvg server (dermatology only) -- NOT the
│   │                                 # production server, see "Running federated learning" below
│   ├── client.py                   # Flower client (Ubuntu AND RPi5), wraps ONE model at a time (dermatology or ocular)
│   ├── models.py                   # per-module model builders, MODEL_REGISTRY (dermatology + ocular)
│   ├── torch_ops.py                # PyTorch get/set-weights/fit/evaluate free functions for the ocular client
│   ├── data_partition.py           # seeded non-IID Dirichlet shard split, real dir or synthetic fallback (both models)
│   ├── export_to_onnx.py           # re-exports a Keras dermatology checkpoint to ONNX (explicit step)
│   ├── plot_convergence.py         # plots the per-round CSV log
│   ├── dermatology_class_names.json
│   ├── ocular_class_names.json
│   └── logs/                       # per-round convergence CSV/PNG + .keras checkpoints (gitignored)
├── scripts/
│   ├── deploy_to_rpi.sh            # rsync this project to the RPi5 over SSH
│   ├── start_fl_server.sh
│   ├── start_fl_client.sh
│   └── run_fl_demo_local.sh        # 2-client FL smoke test on one machine
├── data/sample_images/             # unchanged from nexus_pfe
└── tests/
    └── test_inference.py           # unchanged from nexus_pfe (same modules, same assertions)
```

[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) is not vendored in
this repository -- it was reference-only and nothing in
`src/inference/rppg_algorithms.py` imports from it at runtime. The licensing
note below still applies.

## Setup

Ubuntu (or any x86_64 desktop):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt          # inference + Streamlit
pip install -r requirements-fl.txt        # only if this machine runs the FL server/client
streamlit run app.py
```

Raspberry Pi 5: see `docs/deployment.md` (SSH deploy, venv with
`--system-site-packages`, apt-installed `picamera2`) and
`requirements-rpi5.txt`.

## Running federated learning

**`fl/server.py` in this repo is a local-only, dermatology-only smoke-test
scaffold** (see the module docstring) -- useful for a quick single-machine
sanity check via `scripts/run_fl_demo_local.sh`, but it has no model
versioning/registry, no held-out evaluation, and no automatic ONNX export.
The **authoritative FL server is `nexus_pfe_server_fed/fl_server/server.py`**
(supports both dermatology and ocular, versions every round, evaluates on a
held-out set, and auto-exports ONNX). `fl/client.py` in this repo is the real
client and is weight-compatible with that server -- point it there for
anything beyond a local smoke test. See
`nexus_pfe_server_fed/docs/fl_setup.md` for the full walkthrough. Short
version, 3 terminals (one server + two clients, either model):

```bash
# Windows machine (nexus_pfe_server_fed): FL server
python -m fl_server.server --model dermatology --port 8080 --min-clients 2 --rounds 5

# Ubuntu or RPi5 (over SSH): client 0
python -m fl.client --client-id 0 --server <server-ip>:8080 --model dermatology --num-clients 2

# Ubuntu or RPi5 (over SSH): client 1
python -m fl.client --client-id 1 --server <server-ip>:8080 --model dermatology --num-clients 2
```

Ocular works the same way with `--model ocular` and a different port (the
server federates one module per process, see `fl_setup.md`).

For the local single-machine smoke test instead (this repo's own
`fl/server.py`, dermatology only):

```bash
scripts/start_fl_server.sh 5 2
scripts/start_fl_client.sh 0 127.0.0.1:8080 2
scripts/start_fl_client.sh 1 127.0.0.1:8080 2
```

Then plot convergence: `python -m fl.plot_convergence fl/logs/<the-csv>`.

**Resolved 2026-07**: the FL package and `app.py`'s dermatology tab previously
targeted different models (FL trained toward EfficientNetV2-S/11-class while
the deployed tab ran an older EfficientNetB0/24-class model). They now target
the same **EfficientNetV2-S** dermatology model (11 classes, raw `[0,255]`
preprocessing). A freshly FL-trained checkpoint exported via
`fl/export_to_onnx.py` is architecture-compatible with what's deployed, but
still requires the manual export/swap-in step described in
`docs/deployment.md` section 4 — it isn't wired in automatically.

## Model status

Same models/status as `nexus_pfe` (see `docs/model_cards.md`):

- **Ocular**: pretrained ONNX (self-contained, no external `.onnx.data`
  sidecar), ~95.9% accuracy, 5 classes.
- **Dermatology (deployed and FL target — now the same model)**:
  EfficientNetV2-S ONNX, ~61.6% test accuracy, 11 classes -- what `app.py`'s
  Dermatology tab runs today, and what `fl/` trains toward.
- **rPPG**: CHROM & POS, deterministic, no trainable weights -- never
  federated.

## Federated learning: proof-of-concept scope

2 clients (Ubuntu + RPi5), 1 federated model (dermatology), plain FedAvg,
non-IID shards simulated with a seeded Dirichlet split. Secure aggregation,
differential privacy, >2 clients, and federating ocular are noted as future
work in `docs/deployment.md` -- not implemented here, to keep this a PoC
rather than a production FL system.

## Licensing note

The rPPG CHROM/POS implementation is adapted from
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox), released under
the Responsible AI License (RAIL) -- review the license terms before any use
beyond academic/research purposes, and cite the toolbox's paper in any
written deliverable that uses it (see `docs/model_cards.md` for full
citations).
