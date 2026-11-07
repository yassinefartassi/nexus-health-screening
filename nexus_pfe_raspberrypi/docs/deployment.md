# Deployment: Raspberry Pi 5 + Federated Learning Proof of Concept

This doc covers the two things added on top of the desktop (`nexus_pfe`)
project: running the same Streamlit test bench on a Raspberry Pi 5, and
running one federated learning (FL) round across the Ubuntu machine and the
RPi5.

## 1. Face ROI: MediaPipe vs. the OpenCV Haar-cascade fallback

`src/preprocessing/face_roi.py` locates the face and extracts the
forehead/cheek skin ROI that `rppg_infer.py`'s CHROM/POS pipeline runs on.
It prefers MediaPipe's FaceLandmarker (478-point mesh, landmark-relative
ROIs), but **MediaPipe has no wheel for Python 3.13 on ARM64** -- so it
cannot be installed on a Raspberry Pi 5 running the OS's default Python at
all (confirmed on-device: `pip install mediapipe` has nothing to offer for
`cp313-manylinux*-aarch64`).

`face_roi.py` handles this automatically, no flag to set: it attempts
`import mediapipe` once at module load, and falls back to an OpenCV Haar
cascade (`cv2.CascadeClassifier` + `haarcascade_frontalface_default.xml`,
vendored at `models/haarcascades/` so it works with no network access at
runtime) whenever that import fails. Both backends return the exact same
`FaceROIResult` (`face_found`, `mean_rgb`, `overlay_bgr`), so
`rppg_infer.py` and `app.py` need zero changes either way -- the Ubuntu
desktop (MediaPipe, landmark-relative ROIs) and the RPi5 (Haar, fixed
proportional ROIs off the face bounding box) just get different precision
under the same interface.

**Pin note**: `requirements.txt` pins `opencv-python-headless<5` --
OpenCV 5.0.0 removed `cv2.CascadeClassifier` from its Python bindings
(replaced by the DNN-based `FaceDetectorYN`), which broke the fallback
during RPi5 bring-up. The 4.x line still ships it and has full aarch64
wheels, so nothing is lost by pinning.

## 2. Camera sources

Inference code (`src/inference/`, `src/preprocessing/face_roi.py`) is
unchanged from the desktop project and never talks to a camera directly --
it always receives a BGR frame handed to it. `src/camera/camera_source.py`
is the only thing that changes per machine, and it's picked by config/env,
not by editing code:

| `NEXUS_CAMERA_SOURCE` | Class | Extra config |
|---|---|---|
| `webcam` (default off-Pi) | `WebcamSource` | `NEXUS_CAMERA_INDEX` (default `0`) |
| `ip` | `IPCameraSource` | `NEXUS_CAMERA_URL`, e.g. `http://192.168.4.3:8080/video` (phone MJPEG) or `rtsp://192.168.4.3:554/stream` |
| `rpicam` (default on a detected RPi) | `RpicamSource` | none -- uses `picamera2` |

If `NEXUS_CAMERA_SOURCE` isn't set, `create_camera_source()` checks
`/proc/device-tree/model` and defaults to `rpicam` on a Raspberry Pi, `webcam`
otherwise. Example:

```bash
# Phone camera app streaming MJPEG, from either machine
export NEXUS_CAMERA_SOURCE=ip
export NEXUS_CAMERA_URL=http://192.168.4.3:8080/video
streamlit run app.py
```

## 3. SSH-deploying to the RPi5

The RPi5 is reached over SSH on the Ubuntu machine's hotspot. Code is pushed
manually (no CI): from the Ubuntu machine,

```bash
scripts/deploy_to_rpi.sh pi@<rpi-ip> ~/nexus_pfe_raspberrypi
```

This rsyncs the project over (excluding `venv/`, `__pycache__/`, FL log
output), skipping nothing needed for inference or FL -- `models/` (55MB of
ONNX weights) is included since `src/inference/` needs it on both machines.

Then, on the RPi5:

```bash
ssh pi@<rpi-ip>
cd ~/nexus_pfe_raspberrypi
python3 -m venv --system-site-packages venv   # --system-site-packages: picamera2 is apt-only
source venv/bin/activate
sudo apt install -y python3-picamera2 python3-libcamera   # only if using NEXUS_CAMERA_SOURCE=rpicam
pip install -r requirements.txt -r requirements-fl.txt -r requirements-rpi5.txt
```

See `requirements-rpi5.txt` for why `--system-site-packages` and the apt step
are both necessary, and platform notes on `onnxruntime`/`tensorflow` aarch64
wheels.

Run the same Streamlit app as the desktop:

```bash
NEXUS_CAMERA_SOURCE=rpicam streamlit run app.py --server.address 0.0.0.0
```

## 4. Federated learning: one round, two clients (Ubuntu + RPi5)

> **This walkthrough uses `fl/server.py`, this repo's local-only smoke-test
> server (dermatology only, no versioning/registry, no held-out eval).** For
> a real federation against the authoritative server, use
> `nexus_pfe_server_fed/fl_server/server.py` instead (supports both
> dermatology and ocular, versions every round, evaluates on a held-out set,
> auto-exports ONNX) -- `fl/client.py` here is weight-compatible with it and
> is the real client either way; see `nexus_pfe_server_fed/docs/fl_setup.md`.
> The steps below are still useful as a quick single-machine sanity check
> before pointing at the real server.

`fl/` is a proof-of-concept federated learning setup covering both the
dermatology (TF/Keras) and ocular (PyTorch) modules (see `fl/models.py`'s
`MODEL_REGISTRY` -- rPPG is never federated, it has no trainable weights).
**Only model weight updates cross the network -- raw images never leave the
device that captured them.** See the privacy note at the top of
`fl/client.py` for the full argument; the server only ever receives weight
arrays and scalar metrics (loss/accuracy/sample counts) from each client.

Both machines need `requirements.txt` + `requirements-fl.txt` installed (the
RPi5 also needs `requirements-rpi5.txt`).

> **Python version**: TensorFlow's latest release only ships wheels for
> Python 3.10-3.13. If a machine's default `python3` is newer (check
> `python3 --version` -- this was the case on the Ubuntu dev machine used to
> build this project, which only had 3.14 installed), create the FL venv
> with an older interpreter instead (pyenv, deadsnakes PPA, conda, ...). The
> base inference venv (`requirements.txt` only, no FL) has no such
> constraint.

### Step 1 -- start the server (Ubuntu machine)

```bash
scripts/start_fl_server.sh 5 2   # 5 rounds, wait for 2 clients
# equivalent to: python -m fl.server --rounds 5 --min-clients 2
```

The server binds `0.0.0.0:8080` by default (`FL_SERVER_ADDRESS` to
override) and blocks until 2 clients have connected, then runs FedAvg for
the requested number of rounds.

### Step 2 -- start client 0 (Ubuntu machine, separate terminal)

```bash
scripts/start_fl_client.sh 0 127.0.0.1:8080 2
# equivalent to: python -m fl.client --client-id 0 --server 127.0.0.1:8080 --num-clients 2
```

### Step 3 -- start client 1 (RPi5, over SSH)

```bash
ssh pi@<rpi-ip>
cd ~/nexus_pfe_raspberrypi && source venv/bin/activate
scripts/start_fl_client.sh 1 <ubuntu-hotspot-ip>:8080 2
```

`--client-id` determines which non-IID shard `fl/data_partition.py` hands
that client (seeded, reproducible -- the same `client_id` always gets the
same shard). Each client trains a few local epochs, sends only its updated
weights back, and the server FedAvg-aggregates both into a new global
model, round after round.

### Quick local smoke test (optional, before touching the RPi5)

To check the whole pipeline wires up correctly first, run both clients on
just the Ubuntu machine as a 2-client simulation:

```bash
scripts/run_fl_demo_local.sh 2   # 2 rounds, both clients on localhost
```

### Convergence logging + plotting

`fl/server.py` writes one CSV row per round (`round, train_loss,
train_accuracy, eval_loss, eval_accuracy, n_clients`, each weighted by
sample count) to `fl/logs/dermatology_fedavg_<timestamp>.csv`. Plot it:

```bash
python -m fl.plot_convergence fl/logs/dermatology_fedavg_20260710_143000.csv
```

This is the artifact for the thesis convergence chart.

### Step 4 -- re-export the aggregated model to ONNX (required, not automatic)

ONNX Runtime is inference-only, so `fl/` trains in Keras/TF, not ONNX. After
the server finishes its last round (Step 1's process exits), it saves the
aggregated weights to `fl/logs/<model>_fedavg_<timestamp>.keras` and prints
the exact next command. Run it:

```bash
python -m fl.export_to_onnx fl/logs/dermatology_fedavg_20260710_143000.keras
# writes fl/logs/dermatology_fedavg_20260710_143000.onnx
```

**This step is explicit on purpose.** `src/inference/dermatology_infer.py`
loads a static file (`models/dermnet_model.onnx`) and has no way to know a
new checkpoint exists -- skip this step and the Streamlit app silently keeps
serving the pre-FL weights, no error, no warning, indistinguishable from a
successful round unless you check the file timestamp. `fl/export_to_onnx.py`
requires `tf2onnx` (in `requirements-fl.txt`).

Wiring the exported ONNX file into the Streamlit dermatology tab (updating
`MODEL_PATH` and `CLASS_NAMES_PATH` in `src/inference/dermatology_infer.py`)
is still a separate, deliberate manual step -- not done automatically by the
export script -- but as of the 2026-07 hardening pass it's now an
architecture-compatible swap (11 classes -> 11 classes, same preprocessing
convention) rather than the class-set change it used to be.

## 5. Which dermatology model this actually trains (resolved 2026-07)

`fl/models.py` builds an **EfficientNetV2-S** model (frozen ImageNet
backbone + small trainable head), matching `dermnet_v2s_curated.ipynb` and
its 11-class label set (`fl/dermatology_class_names.json`).

**Previously**, the ONNX file the desktop `app.py`/
`src/inference/dermatology_infer.py` loaded (`models/dermnet_model.onnx`)
was a stale, unrelated EfficientNetB0/24-class model with ImageNet-normalized
preprocessing -- confirmed at the time by inspecting the ONNX graph
(`block1a_activation`, `efficientnetb0_*` node names) -- while `fl/` trained
toward the newer EfficientNetV2-S/11-class model. **Now**, `models/
dermnet_model.onnx` is the same canonical EfficientNetV2-S/11-class model
that `fl/` trains toward: the deployed model and the FL training target are
no longer mismatched. A freshly FL-trained checkpoint exported via Step 4
above is architecture-compatible with what's deployed; wiring it in still
requires the manual `MODEL_PATH`/`CLASS_NAMES_PATH` update above, once a
federated round has actually been validated against real (non-synthetic)
data.

This is also why `fl/data_partition.py` is explicit about **not** reusing
`src/preprocessing/image_utils.py`'s `"imagenet"` mode: that mode is correct
for the ocular model, but wrong for the dermatology EfficientNetV2-S model,
which expects raw `[0, 255]` pixel values (it has its own rescaling baked in
as the first Keras layers, carried through into the exported ONNX graph).
Mixing the two conventions doesn't error -- it just silently trains/predicts
on wrongly-scaled inputs (this was, in fact, exactly the double-normalization
bug fixed in the 2026-07 hardening pass).

## 6. Proof-of-concept scope and future work

By design, this PoC is deliberately small:

- **2 clients** (Ubuntu + RPi5), **1 federated model** (dermatology), plain
  **FedAvg**, **non-IID shards simulated via a seeded Dirichlet split**
  (`fl/data_partition.py`) since no real multi-device patient data exists
  yet.
- **Not included yet** (future work, not needed for the PoC): secure
  aggregation, differential privacy, more than 2 clients, federating the
  ocular model (structurally ready via `fl/models.py`'s `MODEL_REGISTRY`,
  just not implemented), and federating rPPG (won't happen -- it's
  deterministic signal processing with no trainable weights).
