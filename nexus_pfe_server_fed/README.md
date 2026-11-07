# nexus_pfe_server_fed

Federated learning **server** for NEXUS_PFE, a multimodal health-screening
thesis project. This machine (Windows 11) is only the FL server -- it
aggregates model weight updates from clients (an Ubuntu desktop and a
Raspberry Pi 5), it never runs inference and is never itself a client.

Modular, **framework-agnostic** design: each trainable module gets its own
independently-versioned global model, in its own native ML framework --
dermatology is TensorFlow/Keras, ocular is PyTorch. Flower's wire protocol
only ever moves `List[numpy.ndarray]` + scalar metrics, so one generic
round loop (`fl_server/server.py`) federates either module through the same
`FederatedModule` interface (`fl_server/modules/base.py`), without ever
converting one framework's weights into the other's. See
`docs/fl_setup.md`'s "Mixed frameworks" section. rPPG is deterministic
CHROM/POS signal processing with no trainable weights and is never
federated.

**Start here: [`docs/fl_setup.md`](docs/fl_setup.md)** -- the full runbook
(install, place the pretrained checkpoint(s), find this machine's LAN IP,
open the Windows Firewall port, start the server, point clients at it,
produce the convergence plot).

## Layout

```
fl_server/
  config.py                shared paths/constants
  modules/
    base.py                 FederatedModule interface (the framework-agnostic contract)
    dermatology.py            TF/Keras implementation (EfficientNetV2-S, 11 classes)
    ocular.py                  PyTorch implementation (ResNet-18 + Dropout+Linear head, 5 classes)
  datasets/
    dermatology.py             raw-[0,255] held-out test set loader
    ocular.py                   ImageNet-normalized held-out test set loader
  model_registry.py         versioned per-round checkpoint storage + manifest.json (framework-agnostic)
  onnx_export.py             tf2onnx (dermatology) + torch.onnx.export (ocular) helpers
  server.py                   Flower FedAvg server + round-by-round housekeeping (framework-agnostic)
  status_api.py                read-only HTTP JSON status API (round history) for client UIs
  plot_convergence.py         accuracy/loss vs. round chart from a CSV log
models/
  checkpoints/               place pretrained checkpoints here (.keras / .pt)
  dermatology/                 global_v<round>.keras / .onnx + manifest.json (generated)
  ocular/                        global_v<round>.pt / .onnx + manifest.json (generated)
logs/                        per-round CSV logs + convergence PNGs (generated)
scripts/
  start_server.ps1            the FedAvg server (gRPC, port 8080 by default)
  start_status_api.ps1         the read-only status API (HTTP/JSON, port 8090 by default)
docs/fl_setup.md
```

## Clients

This server only aggregates -- it has no client code of its own. Weight-
compatible FL clients live in the other NEXUS_PFE repos, all implementing
the same non-IID partitioning + per-module architecture contract:

- `nexus_pfe/fl_client/` -- the Streamlit/desktop web app's client (`python -m fl_client.client`).
- `nexus_pfe_raspberrypi/fl/` -- the Raspberry Pi 5 reference client.
- `nexus_pfe_mobile` -- **does not** train on-device (not realistic in Flutter/Dart today);
  its "Apprentissage fédéré" screen polls `status_api.py` to show real round
  history and hands the user the CLI command to run a real Python client
  instead. See that repo's README for the full "real vs simulated" breakdown.

## Monitoring

`fl_server/status_api.py` (`scripts\start_status_api.ps1`) is a small,
stdlib-only, **read-only** HTTP JSON endpoint -- separate from the FedAvg
gRPC server above -- that serves `models/<module>/manifest.json`'s real
round history so `nexus_pfe`'s and `nexus_pfe_mobile`'s "Apprentissage
fédéré" screens can show real current-round/accuracy numbers without either
client needing filesystem access to this machine. `GET /status` or
`GET /status/<dermatology|ocular>`. It never triggers a round and never
touches model weights.

## Privacy

Only model weight arrays and small scalar metrics cross the network --
never raw images or patient data, for either module. See `docs/fl_setup.md`
section 0 and the docstring at the top of `fl_server/server.py`.
Differential privacy and secure aggregation are future work, not
implemented in this PoC.
