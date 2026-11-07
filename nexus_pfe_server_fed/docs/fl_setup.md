# NEXUS_PFE federated learning: end-to-end setup (Windows server + 2 clients)

Proof-of-concept scope, on purpose: **2 clients** (an Ubuntu desktop and a
Raspberry Pi 5), plain **FedAvg**, **2 federated modules** (dermatology,
TensorFlow/Keras; ocular, PyTorch), each with its pretrained checkpoint
already placed under `models/checkpoints/` (see step 2). Clarity and a
working demo matter more here than production hardening -- see "Future
work" at the bottom for what's deliberately not built yet.

This machine (Windows 11) is **only the FL server**. It never runs
inference and it is never itself a client. It aggregates weight updates
from the two clients, evaluates the aggregated model on a held-out test
set, versions it, and re-exports it to ONNX -- that's it.

---

## 0. Privacy note (read this first -- it's the core thesis argument)

Flower's client/server protocol only ever exchanges **model weight arrays**
and small scalar metrics (loss, accuracy, sample counts). **Raw images
never cross the network in either direction**, for either module. Each
client (Ubuntu, RPi5) trains locally on its own device; only the resulting
parameter update is sent to this server to be FedAvg-aggregated. This
server never receives, stores, or has access to a single patient image.

Differential privacy and secure aggregation are **not** implemented in this
PoC -- see "Future work" below.

---

## 0.5. Mixed frameworks (read this before installing anything)

NEXUS_PFE's two federated modules use **different ML frameworks**:

| Module | Framework | Checkpoint | Architecture |
|---|---|---|---|
| dermatology | TensorFlow/Keras | `.keras` | EfficientNetV2-S + head, 11 classes |
| ocular | PyTorch | `.pt` | ResNet-18 (frozen) + `Dropout` + `Linear`, 5 classes |

Flower itself is framework-agnostic -- its wire protocol only moves
`List[numpy.ndarray]` and scalar metrics, never a `tf.Tensor` or
`torch.Tensor`. This server exploits that deliberately: every module
implements the same `FederatedModule` interface
(`fl_server/modules/base.py` -- build / load_checkpoint / save_checkpoint /
get_weights / set_weights / fit / evaluate / export_onnx), and
`fl_server/server.py`, `model_registry.py`, and `onnx_export.py` are
written once, entirely in terms of that interface. None of them import
`tensorflow` or `torch` directly, or branch on which framework a module
uses. This is a deliberate architectural point of the thesis: adding a
third federated module in a third framework would require zero changes to
the round loop, versioning, or ONNX export code -- only a new
`FederatedModule` implementation. **We do not convert one framework's
weights into the other's** -- a dermatology round only ever aggregates
among dermatology clients, an ocular round only among ocular clients.

**Practical consequence: every server and client machine needs BOTH
TensorFlow and PyTorch installed**, even though a given server process (or
a given client run) only exercises one module at a time -- see
`requirements.txt`. This is heavier than a single-framework setup,
especially on the Raspberry Pi 5:

> **If either framework won't install/run on the RPi5** (limited RAM/storage,
> no aarch64 wheel available, etc.), don't fight it for this PoC. The
> fallback is to run **both clients on the Ubuntu machine** (two `fl.client`
> processes, `--client-id 0` and `--client-id 1`, both pointed at this
> server) and treat the RPi5 as future/optional hardware. Flag this
> explicitly if you hit it -- don't silently skip a client or fake its
> participation.

---

## 1. Install on this machine

```powershell
cd nexus_pfe_server_fed
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Python version**: check `python --version` against the PyPI wheel
> coverage for the `tensorflow`/`torch` versions pinned in `requirements.txt`
> before creating the venv, and install a matching interpreter if needed
> (Windows typically defaults to whatever was last installed from
> python.org).
>
> **tf2onnx + Keras 3 risk**: `requirements.txt` currently pins
> `tensorflow>=2.20`, which defaults `tf.keras` to Keras 3 -- `tf2onnx` (the
> library dermatology's ONNX export goes through, see
> `fl_server/onnx_export.py`) was written against Keras 2 and has patchy
> Keras 3 support. If a dermatology round fails specifically at the ONNX
> export step, this is the likely cause; the pinned-down fallback
> (`tensorflow>=2.15,<2.16`, Python 3.9-3.11 only) is documented right above
> that line in `requirements.txt`.
>
> **CPU-only by default**: both TensorFlow and PyTorch only use the GPU on
> Windows if CUDA/cuDNN are separately installed and configured. That's
> fine here -- FedAvg aggregation and the small per-round held-out
> evaluation are not compute-heavy; only the *clients'* local training
> benefits much from a GPU.

## 2. Place the pretrained checkpoint

Round 0 must be the centrally-pretrained model, **not random weights**.
`server.py` looks for these exact filenames by default (override with
`--checkpoint <path>` / `FL_INIT_CHECKPOINT`):

```
models\checkpoints\dermatology_pretrained.keras
models\checkpoints\ocular_pretrained.pt
```

- **dermatology_pretrained.keras** is `dermnet_v2s_model.keras`, the output
  of `dermnet_v2s_curated.ipynb`.
- **ocular_pretrained.pt** is a PyTorch export of the currently-deployed
  ocular model (`nexus_pfe/models/eye_disease_model.onnx`). Its state_dict
  keys are what confirmed `fl_server/modules/ocular.py`'s architecture
  (ResNet-18 + `Dropout` + `Linear` head) -- see that file's docstring and
  `models/checkpoints/README.md`.

If you're setting this up on a fresh machine and one or both files aren't
there yet, `server.py` always accepts `--allow-random-init` as an escape
hatch for smoke-testing the pipeline before the real checkpoint is copied
over -- but any accuracy numbers from that run are meaningless for the
thesis.

## 3. Find this machine's LAN IP

Clients need this IP to dial in. In PowerShell or cmd:

```powershell
ipconfig
```

Look for the **IPv4 Address** under the network adapter you're actually
using (e.g. `Wireless LAN adapter Wi-Fi` or `Ethernet adapter Ethernet`) --
typically something like `192.168.1.23`. That's `<server-ip>` in every
command below. Both client machines must be on the same LAN/subnet (or
reachable via routing) as this IP.

## 4. Open the Windows Firewall port

**This is the single most likely setup failure.** Without an inbound rule,
Windows Firewall silently drops the incoming connection -- clients will
just hang trying to connect, with no useful error message on either side.

Open PowerShell **as Administrator** and run:

```powershell
New-NetFirewallRule -DisplayName "NEXUS_PFE FL Server" `
    -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
```

(Change `8080` if you're running the server on a different `--port` -- e.g.
if you run dermatology and ocular servers concurrently on two ports, add a
rule per port.) If you'd rather use the GUI: *Windows Defender Firewall
with Advanced Security* -> *Inbound Rules* -> *New Rule...* -> *Port* -> TCP,
specific local port `8080` -> *Allow the connection* -> apply to whichever
network profiles your LAN uses (Private is usually right; avoid enabling it
for Public networks).

To verify quickly from another machine on the LAN once the server is
running (step 5):

```bash
# from the Ubuntu machine or RPi5
nc -zv <server-ip> 8080          # or: telnet <server-ip> 8080
```

If that fails, re-check the firewall rule and that the server is actually
bound to `0.0.0.0` (the default) and not `127.0.0.1`.

## 5. Start the server

```powershell
scripts\start_server.ps1 -Model dermatology -Rounds 5 -MinClients 2
# equivalent to: python -m fl_server.server --model dermatology --rounds 5 --min-clients 2 --port 8080
```

(`-Model ocular` federates the ocular module instead -- on a different
`-Port` if run alongside a dermatology server.) The server binds
`0.0.0.0:8080` (configurable via `--port` / `FL_SERVER_PORT`) and blocks
until the requested number of clients have connected, then runs FedAvg for
the requested number of rounds. It prints the LAN address clients should
use.

## 6. Configure and start each client

Each client is built/maintained in its own repo, but all are weight-
compatible with this server -- same per-module architecture (EfficientNetV2-S
11-class dermatology, ResNet-18 5-class ocular), same label order, same
preprocessing convention (raw `[0,255]` dermatology / ImageNet-normalized
ocular):

- `nexus_pfe/fl_client/` -- the Streamlit/desktop web app's client.
- `nexus_pfe_raspberrypi/fl/` -- the Raspberry Pi 5 reference client.
- `nexus_pfe_mobile` -- **not a training client**. On-device training in
  Flutter/Dart isn't realistic with today's frameworks, so the mobile app's
  "Apprentissage fédéré" screen polls `status_api.py` (step 9) to show real
  round history and hands the user the CLI command to run one of the two
  real clients above instead. See that repo's README for the full writeup.

Point each real client at `<server-ip>:8080`:

```bash
# nexus_pfe (web app / desktop), client 0
cd nexus_pfe && pip install -r requirements-fl.txt
python -m fl_client.client --model dermatology --client-id 0 --server <server-ip>:8080 --num-clients 2

# nexus_pfe_raspberrypi (Raspberry Pi 5), client 1, reached over SSH
ssh pi@<rpi-ip>
cd ~/nexus_pfe_raspberrypi && source venv/bin/activate
python -m fl.client --model dermatology --client-id 1 --server <server-ip>:8080 --num-clients 2
```

`--client-id` picks which non-IID data shard a client trains on (seeded,
reproducible). Once both clients connect, the server starts round 1
automatically. If the RPi5 can't run one of the two frameworks (see section
0.5), run both `--client-id 0` and `--client-id 1` from `nexus_pfe/fl_client`
instead.

## 7. What happens each round

For each round: both clients train locally and send back their updated
weights -> the server FedAvg-aggregates them -> the aggregated model is:

1. **Evaluated** on this server's held-out test set
   (`fl_server/datasets/dermatology.py` or `.../ocular.py` depending on
   `--model` -- point `DERMNET_TEST_DIR`/`OCULAR_TEST_DIR` or `--test-dir`
   at a real curated held-out directory when you have one; falls back to a
   seeded synthetic set for pipeline smoke tests),
2. **Versioned** to `models/<module>/global_v<round>.<ext>` (`.keras` or
   `.pt`), tracked in `models/<module>/manifest.json` (round, timestamp,
   client count, accuracy, loss, communication bytes),
3. **Re-exported to ONNX** (`models/<module>/global_v<round>.onnx`, via
   `tf2onnx` for dermatology or `torch.onnx.export` for ocular -- see
   `fl_server/onnx_export.py`) -- this step is not optional; skipping it
   means deployed inference keeps serving stale weights while the native
   checkpoint keeps improving.

Round 0 (the pretrained baseline, no clients involved yet) is saved and
evaluated the same way, so the convergence plot has a true starting point.

Per-round metrics are also appended to
`logs/<module>_fedavg_<timestamp>.csv`.

## 8. Produce the convergence plot

```powershell
python -m fl_server.plot_convergence logs\dermatology_fedavg_20260711_143000.csv
```

Saves a PNG next to the CSV: accuracy and loss vs. round, starting at round
0 (pretrained baseline) through round N. Works unchanged for an ocular CSV.
This is the thesis deliverable proving FL actually improves the model over
the centrally-pretrained starting point.

## 9. Let the client UIs show real round history (status API)

`nexus_pfe`'s "Apprentissage fédéré" Streamlit tab and `nexus_pfe_mobile`'s
"Apprentissage fédéré" screen both display real round-by-round accuracy --
never invented numbers -- by polling a small **read-only** HTTP JSON
endpoint on this server, separate from the FedAvg gRPC server above:

```powershell
scripts\start_status_api.ps1
# equivalent to: python -m fl_server.status_api --port 8090
```

It just re-serves whatever is already in `models/<module>/manifest.json` as
JSON (`GET /status`, `GET /status/<dermatology|ocular>`) -- it never
triggers a round and never touches model weights. Safe to start/stop any
time, independently of whether a FedAvg round is currently running. Same
firewall caveat as step 4 applies if a client isn't on this machine:

```powershell
New-NetFirewallRule -DisplayName "NEXUS_PFE FL Status API" `
    -Direction Inbound -Protocol TCP -LocalPort 8090 -Action Allow
```

If no round has run yet for a module, the endpoint returns an empty
`"rounds"` list -- both client UIs render that as "waiting for a run", never
a fabricated chart.

---

## Adding a third federated module

Dermatology and ocular each get their **own independently-versioned global
model** in their **own framework** -- not one shared model or a forced
common framework. To add a third module (in whatever framework fits it
best):

1. Implement `FederatedModule` (`fl_server/modules/base.py`) for it --
   build / load_checkpoint / save_checkpoint / get_weights / set_weights /
   fit / evaluate / export_onnx -- see `modules/dermatology.py` (TF/Keras)
   and `modules/ocular.py` (PyTorch) as the two existing examples.
2. Add a held-out loader alongside `datasets/dermatology.py` /
   `datasets/ocular.py`.
3. Register it in `MODULE_REGISTRY` (`fl_server/modules/__init__.py`) and
   `HOLDOUT_LOADERS`/`HOLDOUT_ENV_VARS` (`fl_server/server.py`).

`model_registry.py`, `onnx_export.py`, and `server.py`'s round loop need
**no changes** -- that's the point of the interface.

rPPG is never federated -- it's deterministic CHROM/POS signal processing
with no trainable weights, so there's nothing to aggregate, and it has no
`MODULE_REGISTRY` entry.

## Future work (not in this PoC)

- Differential privacy (e.g. DP-SGD on the client side, gradient clipping +
  noise) -- currently only implicit privacy from "raw images never leave
  the device."
- Secure aggregation (e.g. Flower's built-in secure aggregation strategies)
  so even the server can't inspect individual client updates before
  averaging.
- More than 2 clients / real non-IID patient data instead of the seeded
  Dirichlet-split or synthetic fallback.
- An actual ocular *training* pipeline/notebook (unlike dermatology's
  `dermnet_v2s_curated.ipynb`) -- `ocular_pretrained.pt` is a PyTorch
  export of the already-deployed ocular model, not the output of a
  documented training run, so there's no notebook to point to if the
  architecture ever needs retracing.
