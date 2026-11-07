# Pretrained checkpoints

One per federated module, each in its own native framework format (see
`fl_server/modules/base.py` for why -- dermatology is TensorFlow/Keras,
ocular is PyTorch, and this project never converts one into the other).
`fl_server/server.py` looks for these exact filenames by default (override
with `--checkpoint <path>` / `FL_INIT_CHECKPOINT`):

```
dermatology_pretrained.keras
ocular_pretrained.pt
```

`server.py` loads whichever one matches `--model` as round-0 starting
weights (see `docs/fl_setup.md`) -- federated training here always starts
from the pretrained model, never from random weights, unless you explicitly
pass `--allow-random-init` for a pipeline smoke test.

**dermatology_pretrained.keras**: `dermnet_v2s_model.keras`, the output of
`dermnet_v2s_curated.ipynb` (EfficientNetV2-S, 11 classes, ~62% test
accuracy at time of writing).

**ocular_pretrained.pt**: a PyTorch export of the currently-deployed ocular
model (`nexus_pfe/models/eye_disease_model.onnx`). Its state_dict keys
(`layer1`-`layer4` with 2 BasicBlocks each, downsample only on layer2/3/4's
first block, `fc.1.weight` shaped `(5, 512)`) confirm the architecture
`fl_server/modules/ocular.py` builds: ResNet-18 + `fc = Sequential(Dropout,
Linear(512, 5))` -- see that file's docstring for how this was verified
directly from the checkpoint's keys (no separate training notebook exists
for ocular, unlike dermatology's).

Both files are large (100+ MB) and machine-local -- this directory is
gitignored except for this README.
