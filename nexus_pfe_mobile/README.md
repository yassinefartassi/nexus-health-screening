# NEXUS PFE — Mobile

Flutter mobile counterpart to **NEXUS_PFE** (the Python/Streamlit desktop
prototype, `health_ai_suite`), targeting Android first with the codebase
structured to also run on iOS.

All three inference modules run **fully on-device**: model inference (ONNX
Runtime), face tracking (ML Kit), and signal processing (CHROM/POS, FFT, HRV)
all execute locally. No network call is made at any point during inference —
these three tabs have no backend.

| Module | What it does | Model / method |
|---|---|---|
| Ocular | Eye disease classification from a photo (5 classes) | `eye_disease_model_merged.onnx` via ONNX Runtime |
| Dermatology | Skin condition classification from a photo (24 classes) | `dermnet_model.onnx` via ONNX Runtime |
| rPPG | Contactless heart rate + HRV from the live camera | CHROM / POS, implemented in Dart |
| Apprentissage fédéré | Explains + shows real FL round history | Reads `nexus_pfe_server_fed`'s status API over HTTP — see below. **The one screen in this app that makes a network call, and it never sends an image.** |

---

## 1. Prerequisites

- Flutter SDK (this project was built against Flutter 3.44, Dart 3.12 —
  `flutter --version` should report similar or newer).
- Android Studio / Android SDK with an emulator or a physical device
  (**physical device strongly recommended** for the rPPG module — camera
  behavior on emulators is unreliable).
- For iOS: Xcode 16+, a Mac. iOS 16 is the minimum deployment target,
  required by the ONNX Runtime iOS bindings.

```bash
flutter pub get
```

## 2. Model files

```
assets/models/
├── eye_disease_model_merged.onnx     # single-file export, no external .onnx.data
├── eye_disease_class_names.json      # 5 classes
├── dermnet_model.onnx                # single-file export, no external .onnx.data
└── dermnet_class_names.json          # 24 classes
```

Both models must be **single-file exports** (no companion `.onnx.data`
external-weights file) — Flutter's asset bundle doesn't give ONNX Runtime a
real filesystem path it can find a sibling file next to, so a split export
will fail to load with a generic "Failed to load ONNX model at ..." error
that doesn't say why.

### If you regenerate `eye_disease_model_merged.onnx`

If you re-export the ocular model from the desktop prototype and end up with
a split `eye_disease_model.onnx` + `eye_disease_model.onnx.data` pair again
(this is what PyTorch's default ONNX exporter produces once the weights
exceed the 2GB protobuf-embedded limit, or once you explicitly pass
`save_as_external_data=True`), merge them into one self-contained file
before copying into `assets/models/`:

```python
import onnx

m = onnx.load("eye_disease_model.onnx", load_external_data=True)  # picks up the sibling .onnx.data
onnx.save_model(m, "eye_disease_model_merged.onnx", save_as_external_data=False)
```

Sanity-check the result actually has no external references and isn't
suspiciously small (a graph-only export with the weights still external is
~90KB; this model's real weights bring it to ~43MB):

```python
m2 = onnx.load("eye_disease_model_merged.onnx", load_external_data=False)
assert not any(t.data_location == onnx.TensorProto.EXTERNAL for t in m2.graph.initializer)
```

After swapping the file in, re-run `flutter pub get` (or hot-restart if
already running) so the asset bundle picks it up, then run the smoke test
below to confirm it actually loads on-device.

### ⚠️ Verify the dermatology class order

`dermnet_class_names.json` was carried over from the desktop prototype,
whose own README flagged that its class order was an unverified best guess
(standard alphabetical DermNet ordering), **not** confirmed against the
actual Kaggle training notebook's `class_indices` / `ImageFolder` order. If
dermatology predictions look scrambled — e.g. a confident "Healthy" call on
obviously diseased skin — this file is the first thing to check. The ocular
class list (`eye_disease_class_names.json`) came from a source that had
already been verified on desktop, so it should not need the same scrutiny.

## 3. Run

```bash
flutter run            # picks whatever device/emulator is connected
flutter build apk      # release Android build
```

On first launch, each screen requests camera permission when you first try
to use the camera (via `permission_handler`); the rPPG screen also handles
"permission denied" and "no camera" as explicit error states rather than
crashing.

### Testing the models actually load

`integration_test/model_smoke_test.dart` loads both bundled ONNX models from
the real asset bundle and runs a dummy inference through each, on a real
device/emulator. This is what catches a broken/unmerged asset or a wrong
per-model preprocessing mode at test time instead of at runtime on a user's
device — a plain `flutter test` (host VM, no platform channels) can't
exercise the actual ONNX Runtime plugin, so this needs a connected target:

```bash
flutter test integration_test/model_smoke_test.dart -d <device-id>
```

### Diagnosing a "Model failed to load" error

`OnnxService.load()` and `ImageClassificationView._loadModel()` both log the
full underlying exception (via `dart:developer.log`, error + stack trace) and
surface it in the on-screen error message — check `flutter run`'s console
output or `adb logcat` for the `OnnxService` / `ImageClassificationView` log
tags rather than trusting the generic-looking on-screen text alone. Two
concrete causes already seen in this project:
- Asset path in `ocular_inference.dart`/`dermatology_inference.dart` doesn't
  match a file that actually exists under `assets/models/`.
- The asset exists but still has unresolved external-data references (see
  "If you regenerate `eye_disease_model_merged.onnx`" above) — ONNX Runtime
  Mobile has no sibling file to resolve them against inside the Flutter
  asset bundle.

An unsupported-operator/opset mismatch would also surface here, but as of
this writing both bundled models use only standard, widely-supported ops
(ocular: opset 18, `Conv`/`Relu`/`MaxPool`/`Gemm`/etc.; dermatology: opset 15,
`Conv`/`Sigmoid`/`Softmax`/etc.) — neither has hit this failure mode.

---

## Swapping in updated models

The app is built to make re-training and re-exporting painless:

1. **Input resolution is auto-detected**, not hard-coded. `OnnxService`
   (`lib/services/onnx_service.dart`) reads the expected input shape
   directly from the ONNX graph at load time (falling back to 224×224 NCHW
   if the shape can't be introspected — this happens on iOS, where
   `getInputInfo()` isn't currently supported by `flutter_onnxruntime`). You
   do not need to edit any code just because a new model export uses a
   different input size.

2. **Normalization must match training/export exactly, and it's different
   per model — do not unify it.** See `lib/services/image_preprocessing.dart`
   for the full `PreprocessMode` enum; this is not something the code can
   infer from the ONNX graph, so getting it wrong doesn't throw an error —
   the model just silently scores pixels that don't look like anything it
   was trained on, and confidences come out as plausible-looking nonsense.

   - **Ocular** (`eye_disease_model_merged.onnx`, ResNet-style): standard
     **ImageNet normalization** — scale to `[0, 1]`, then
     `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
     `PreprocessMode.imagenet` in `ocular_inference.dart`.
   - **Dermatology** (`dermnet_model.onnx`, EfficientNetV2-S): this graph
     bakes its own input rescaling/normalization into the first layers
     (confirmed via `onnx` inspection — `Sub`/`Mul`/`Cast` ops right after
     the input node), so the Dart side must hand it **raw `[0, 255]`** pixel
     values with **no** normalization applied on top.
     `PreprocessMode.raw255` in `dermatology_inference.dart`. Applying
     ImageNet normalization here (as an earlier version of this app
     mistakenly did for both models) double-preprocesses the input —
     it doesn't crash, it just silently produces wrong predictions.

   If a future model swap uses yet another convention (e.g. plain `[0, 1]`
   or `[-1, 1]` scaling with no baked-in graph preprocessing), change the
   `PreprocessMode` passed in the relevant `*_inference.dart` file
   accordingly — but the ocular/dermatology split above must stay per-model.

3. **Class name order must match the model's output layer order exactly**
   (index *i* in the JSON must be the same class as output neuron *i*).
   Swap the `.json` file for a new one with the same array length as the
   model's output size — `OcularInferenceService`/`DermatologyInferenceService`
   both throw a loud error at load time if the count doesn't match what's
   expected (5 and 24 respectively), specifically to catch this class of
   mistake early rather than silently mislabeling predictions.

4. Both classifiers share one ONNX wrapper (`onnx_service.dart`) and one
   preprocessing function (`image_preprocessing.dart`) — there's no
   model-specific inference code to update beyond the asset paths and class
   count check in `ocular_inference.dart` / `dermatology_inference.dart`.

---

## rPPG: CHROM / POS implementation notes

`lib/services/rppg_algorithms.dart` is an **original Dart re-implementation**
of two published, peer-reviewed rPPG algorithms:

- **CHROM** — de Haan & Jeanne, *"Robust Pulse Rate From Chrominance-Based
  rPPG"*, IEEE Trans. Biomed. Eng., 2013.
- **POS** — Wang et al., *"Algorithmic Principles of Remote-PPG"*, IEEE
  Trans. Biomed. Eng., 2016 (Plane-Orthogonal-to-Skin).

During prototyping on desktop/Ubuntu, this project's Python implementation
(`NEXUS_PFE` / `health_ai_suite/rppg/algorithms.py`) used
**[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)** as the
reference implementation for validating these algorithms' equations.
rPPG-Toolbox is distributed under the **Responsible AI Source Code License
(RAIL) v1.1** (see `LICENSE` in that repository / http://licenses.ai/). No
rPPG-Toolbox source is vendored, copied, or translated into this app — the
Dart code here reimplements the published linear-combination (CHROM) and
projection (POS) equations directly from the papers, cross-checked for
numerical agreement against the desktop prototype during development. If
you redistribute this app, review rPPG-Toolbox's RAIL license yourself if
your use is influenced by having consulted that reference implementation.

**One deliberate deviation from the desktop prototype:** the desktop version
used `scipy.signal.butter(3, ..., output="sos")` + `sosfiltfilt` for the
bandpass filter. There's no equivalent filter-design routine in pure Dart,
so `bandpassFilter()` instead implements a single RBJ-cookbook
constant-skirt-gain biquad
(https://www.w3.org/TR/audio-eq-cookbook/), run forward then backward
("filtfilt" trick) for zero phase. Same intent (bandlimit the signal to
0.7–4 Hz before the FFT peak search), not a numerically identical filter —
documented inline in `rppg_algorithms.dart`.

HRV (SDNN, RMSSD) is a new addition beyond what the desktop prototype
computed: `computeHrv()` does simple local-maximum peak detection on the
filtered POS waveform (POS is the default algorithm, matching the desktop
prototype's `RPPG_DEFAULT_ALGO`), with a minimum-distance constraint derived
from `kMaxBpm` so noise can't register as spurious extra beats, then
computes standard deviation of NN intervals (SDNN) and root-mean-square of
successive differences (RMSSD) from the resulting inter-beat intervals.

### Known simplifications (documented, not accidental)

- **Device orientation**: `rppg_screen.dart` computes the ML Kit rotation
  compensation from `CameraController.value.deviceOrientation` using the
  standard formula (see the [google_mlkit_commons example
  app](https://pub.dev/packages/google_mlkit_commons)), so it's correct for
  portrait use on both front and back cameras. It has not been stress-tested
  in landscape.
- **Face detection is throttled**, not run on every frame (`RppgService`
  reuses the last known bounding box every other frame) — this keeps the RGB
  sampling rate close to the camera's native frame rate, since CHROM/POS's
  effective fps directly affects the FFT frequency resolution.
- The forehead ROI geometry (30–70% of face width, 8–25% of face height) is
  carried over unchanged from the desktop prototype's `rppg/processor.py`.

---

## Federated learning: what's real, what's simulated

This app has a fourth tab, **"Apprentissage fédéré"**
(`lib/screens/federated_learning_screen.dart`), that explains and shows how
NEXUS_PFE's federated learning works. Read this section before assuming it
does more (or less) than it does:

- **REAL**: the round-by-round history shown on that screen (round number,
  client count, held-out accuracy/loss) is fetched live, over HTTP, from
  `nexus_pfe_server_fed`'s read-only status API (`fl_server/status_api.py`)
  and reflects actual FedAvg rounds that actually ran on that server. If the
  API isn't reachable, or no round has run yet for a module, the screen says
  so explicitly ("en attente d'exécution") — it never fabricates a chart.
- **REAL**: the training that produces that history happens for real, in
  TensorFlow (dermatology) and PyTorch (ocular), in two other repos:
  `nexus_pfe/fl_client/` (the Streamlit/desktop web app) and
  `nexus_pfe_raspberrypi/fl/` (the Raspberry Pi 5 client).
- **SIMULATED / NOT IMPLEMENTED, on purpose**: **this mobile app does not
  train a model on-device and does not send any weight update.** On-device
  training in Flutter/Dart with backpropagation through EfficientNetV2-S or
  ResNet-18 is not realistic with any current mobile ML framework
  (`flutter_onnxruntime` and TFLite are inference-only on this stack). Rather
  than fake that with a progress bar that doesn't correspond to any real
  gradient step — which would misrepresent the thesis's actual FL results —
  the screen's "Participer à un round" card is an honest **trigger/handoff**:
  it copies the exact `python -m fl_client.client ...` command to the
  clipboard for the user to run on a real Python machine (the web app, or any
  computer on the same LAN). It does not run that command itself and does
  not pretend training happened on this device.

This mirrors the project brief's explicit instruction not to fabricate
on-device training and to document precisely what's real vs. simulated.

### Pointing the app at the FL server

The screen has an editable "Adresse de l'API de statut du serveur FL" field,
defaulting to `http://127.0.0.1:8090` (`FlStatusService.defaultBaseUrl` in
`lib/services/fl_status_service.dart`). On a physical device or a
non-loopback emulator this must be changed to the FL server's real LAN IP
(`nexus_pfe_server_fed`, `scripts\start_status_api.ps1`, port 8090 by
default) — `127.0.0.1` from the phone's perspective is the phone itself, not
your desktop. Both the FedAvg server (port 8080) and the status API (port
8090) need their own Windows Firewall inbound rules — see
`nexus_pfe_server_fed/docs/fl_setup.md`.

## Project structure

```
lib/
├── main.dart
├── models/
│   ├── classification_result.dart   # label + confidence, softmax fallback logic
│   ├── rppg_result.dart             # BPM/HRV/waveform bundle returned per cycle
│   └── fl_status.dart               # FL round history data classes (real server data only)
├── services/
│   ├── onnx_service.dart            # generic ONNX Runtime wrapper (shared)
│   ├── image_preprocessing.dart     # resize + per-model normalize (imagenet/raw255/0-1/-1-1)
│   ├── image_classifier_service.dart# interface both classifiers implement
│   ├── ocular_inference.dart
│   ├── dermatology_inference.dart
│   ├── rppg_algorithms.dart         # CHROM, POS, bandpass, FFT->BPM, HRV
│   ├── rppg_service.dart            # face tracking + RGB buffering + orchestration
│   ├── permission_service.dart
│   └── fl_status_service.dart       # HTTP client for the FL server's read-only status API
├── screens/
│   ├── home_screen.dart
│   ├── ocular_screen.dart
│   ├── dermatology_screen.dart
│   ├── rppg_screen.dart             # camera lifecycle, device rotation
│   └── federated_learning_screen.dart # FL cycle explainer + real status + real-vs-simulated
└── widgets/
    ├── image_classification_view.dart # shared picker/results UI (Ocular+Dermatology)
    ├── face_roi_overlay.dart
    └── waveform_sparkline.dart

assets/models/                        # ONNX models + class name JSON (see above)
integration_test/
└── model_smoke_test.dart               # loads both real models + dummy inference on-device
```

## Dependencies and why

| Package | Why |
|---|---|
| `flutter_onnxruntime` | On-device ONNX inference. Chosen over the older `onnxruntime` (gtbluesky) FFI-based package because it's actively maintained (last publish within weeks vs. ~2 years stale at the time of writing) and wraps the current ONNX Runtime Android/iOS bindings directly. |
| `camera` | Live camera preview + frame stream, used by the rPPG screen. |
| `google_mlkit_face_detection` | On-device face bounding box for the rPPG ROI. |
| `fftea` | Pure-Dart FFT, used to turn the filtered CHROM/POS waveform into a BPM estimate. |
| `permission_handler` | Camera permission request/status/settings-deeplink. |
| `image` | Decode + bilinear resize for classifier preprocessing. |
| `image_picker` | Camera-capture-or-gallery-picker UX for the Ocular/Dermatology screens. Not in the originally scoped dependency list but required to fulfill "camera capture or gallery picker" — the `camera` package alone streams live frames and can't present the OS's photo picker UI, and `image` only decodes bytes it doesn't already have. |
| `http` | The federated learning screen's *only* network call: polling `nexus_pfe_server_fed`'s read-only status API for real round history (`fl_status_service.dart`). Not used anywhere in the three inference tabs. |
| `integration_test` (dev) | Runs `integration_test/model_smoke_test.dart` on a real device/emulator so a broken ONNX asset or wrong preprocessing is caught by CI/local testing instead of by a user hitting "Model failed to load". Needs a real target because it exercises the actual `flutter_onnxruntime` platform plugin, which plain `flutter test` (headless host VM) can't do. |

---

## Disclaimer

This app is a research prototype built for a thesis appendix. It is **not a
certified medical device**. Ocular/Dermatology predictions and rPPG
BPM/HRV readings are estimates only and must not be used for clinical
decisions.
