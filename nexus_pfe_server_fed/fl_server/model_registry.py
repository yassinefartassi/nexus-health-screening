"""
Versioned storage for global FL models -- the MODULAR part of the design.
Each federated module (dermatology, ocular) gets its own subdirectory under
models/ with its own independently-numbered rounds and its own
manifest.json. There is deliberately no single shared "global model":
adding a third module means adding a models/<name>/ directory with the same
shape, not touching any other module's history.

This file never imports tensorflow or torch and never branches on which
framework a module uses -- it saves checkpoints entirely through the
module's own save_checkpoint() (see modules/base.py), which is what makes
this registry work unchanged for both dermatology's .keras checkpoints and
ocular's .pt checkpoints.

    models/
      dermatology/
        manifest.json
        global_v0.keras   global_v0.onnx   (round 0 = pretrained baseline)
        global_v1.keras   global_v1.onnx
        ...
      ocular/
        manifest.json
        global_v0.pt      global_v0.onnx
        ...

manifest.json is a JSON list, one entry per round:
    {"round": 1, "timestamp": "...", "num_clients": 2,
     "accuracy": 0.61, "loss": 1.02, "comm_bytes": 123456,
     "checkpoint_path": "global_v1.keras", "onnx_path": "global_v1.onnx",
     "data_source": "real"}

data_source is "real" when the round's held-out accuracy/loss came from a
real curated test directory (--test-dir / DERMNET_TEST_DIR / OCULAR_TEST_DIR
pointed at an actual directory), "synthetic" when datasets/*.py fell back to
its seeded synthetic smoke-test set. This distinction matters because a
synthetic-sourced accuracy number is meaningless (the model was never
trained on flat-color noise) -- see datasets/dermatology.py and
datasets/ocular.py's fallback warning. Older manifests written before this
field existed simply don't have it; treat a missing key the same as
"unknown", not as "synthetic" or "real".
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fl_server.config import MODELS_DIR

if TYPE_CHECKING:
    from fl_server.modules.base import FederatedModule


class ModelRegistry:
    def __init__(self, module_name: str, base_dir: Path = MODELS_DIR):
        self.module_name = module_name
        self.module_dir = Path(base_dir) / module_name
        self.module_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.module_dir / "manifest.json"
        if not self.manifest_path.exists():
            self._write_manifest([])

    def _read_manifest(self) -> List[Dict[str, Any]]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, entries: List[Dict[str, Any]]) -> None:
        self.manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def save_round(
        self,
        round_num: int,
        module: "FederatedModule",
        model: Any,
        num_clients: int,
        accuracy: Optional[float],
        loss: Optional[float],
        comm_bytes: int,
        data_source: str = "unknown",
    ) -> Path:
        """Saves the aggregated global model for this round (via the
        module's own save_checkpoint(), so the on-disk format matches
        whatever framework `module` uses) and records it in the manifest.
        Returns the checkpoint path -- server.py immediately passes this to
        module.export_onnx() so the ONNX deployment artifact never lags
        behind the versioned checkpoint.

        data_source: "real" or "synthetic" -- see module docstring. Passed
        through by server.py, which is the one place that knows whether the
        held-out set it just evaluated on came from a real directory."""
        checkpoint_path = module.save_checkpoint(model, self.module_dir / f"global_v{round_num}")

        entry = {
            "round": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_clients": num_clients,
            "accuracy": accuracy,
            "loss": loss,
            "comm_bytes": comm_bytes,
            "checkpoint_path": checkpoint_path.name,
            "onnx_path": checkpoint_path.with_suffix(".onnx").name,
            "data_source": data_source,
        }
        entries = [e for e in self._read_manifest() if e["round"] != round_num]
        entries.append(entry)
        entries.sort(key=lambda e: e["round"])
        self._write_manifest(entries)
        return checkpoint_path

    def latest_entry(self) -> Optional[Dict[str, Any]]:
        entries = self._read_manifest()
        return entries[-1] if entries else None

    def latest_path(self) -> Optional[Path]:
        entry = self.latest_entry()
        return self.module_dir / entry["checkpoint_path"] if entry else None
