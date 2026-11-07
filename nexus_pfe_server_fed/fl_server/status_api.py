"""
Read-only HTTP status API -- NOT the Flower FedAvg server (that's server.py,
gRPC, port 8080 by default). This is a small stdlib-only JSON endpoint whose
only job is to let the two FL CLIENTS' UIs (nexus_pfe's Streamlit "Apprentissage
federe" tab and nexus_pfe_mobile's "Federated Learning" screen) show REAL
round-by-round numbers -- current round, client count, accuracy/loss per
round -- without either of those apps needing filesystem access to this
Windows machine's models/ and logs/ directories.

Why this exists: nexus_pfe (web) could, in principle, read
models/<module>/manifest.json directly if it happens to run on the same
machine or a shared filesystem -- but nexus_pfe_mobile (Flutter, Android/iOS)
never has that option, it can only speak HTTP over the LAN. Rather than have
the web UI and the mobile UI read the same round history two different ways,
both go through this one HTTP endpoint. It is intentionally read-only and
serves only what's already written to disk by server.py's FederatingFedAvg
(model_registry.py's manifest.json) -- it never triggers a round, never
touches model weights, and never receives images. If no round has run yet
for a module, it reports an empty list; the caller's job is to render that as
"waiting for a run", never a fabricated chart -- see docs/fl_setup.md.

Deliberately stdlib-only (http.server + json), no Flask/FastAPI dependency,
and does NOT import fl_server.modules (which would pull in tensorflow/torch
just to serve JSON) -- so this can run on a machine that only has the FL
round-history files, nothing else.

Usage:
    python -m fl_server.status_api --port 8090
    scripts\\start_status_api.ps1

Endpoints:
    GET /health           -> {"ok": true}
    GET /status           -> {"dermatology": {...}, "ocular": {...}}
    GET /status/<module>  -> {"module", "framework", "class_names", "rounds", "latest_round"}
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fl_server.config import (
    DEFAULT_HOST,
    DERMATOLOGY_CLASS_NAMES_PATH,
    MODELS_DIR,
    OCULAR_CLASS_NAMES_PATH,
)

DEFAULT_STATUS_PORT = 8090

# Framework label is metadata only, hardcoded here (not read from
# fl_server.modules) specifically so this file never imports tensorflow/torch
# -- see module docstring.
MODULES: Dict[str, Dict[str, Any]] = {
    "dermatology": {
        "framework": "TensorFlow/Keras (.keras)",
        "class_names_path": DERMATOLOGY_CLASS_NAMES_PATH,
    },
    "ocular": {
        "framework": "PyTorch (.pt)",
        "class_names_path": OCULAR_CLASS_NAMES_PATH,
    },
}


def _load_class_names(path) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _load_manifest(module_name: str) -> List[Dict[str, Any]]:
    manifest_path = MODELS_DIR / module_name / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return sorted(entries, key=lambda e: e.get("round", 0))


def module_status(module_name: str) -> Dict[str, Any]:
    meta = MODULES[module_name]
    rounds = _load_manifest(module_name)
    return {
        "module": module_name,
        "framework": meta["framework"],
        "class_names": _load_class_names(meta["class_names_path"]),
        "rounds": rounds,
        "latest_round": rounds[-1] if rounds else None,
    }


def full_status() -> Dict[str, Any]:
    return {name: module_status(name) for name in MODULES}


class StatusHandler(BaseHTTPRequestHandler):
    server_version = "NexusFLStatusAPI/1.0"

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Read-only, no credentials, no cookies -- open CORS is fine here and
        # lets a browser-based dev tool hit this endpoint directly too.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        path = urlparse(self.path).path.rstrip("/") or "/"

        if path == "/health":
            self._send_json({"ok": True})
            return

        if path == "/status":
            self._send_json(full_status())
            return

        if path.startswith("/status/"):
            module_name = path[len("/status/"):]
            if module_name not in MODULES:
                self._send_json(
                    {"error": f"unknown module '{module_name}', expected one of {list(MODULES)}"},
                    status=404,
                )
                return
            self._send_json(module_status(module_name))
            return

        self._send_json({"error": "not found", "endpoints": ["/health", "/status", "/status/<module>"]}, status=404)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        print(f"[status_api] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS_PFE FL read-only status API (round history, JSON)")
    parser.add_argument("--host", type=str, default=os.environ.get("FL_STATUS_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("FL_STATUS_PORT", str(DEFAULT_STATUS_PORT)))
    )
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), StatusHandler)
    print(f"[status_api] serving real FL round history on http://{args.host}:{args.port}")
    print(f"[status_api] GET /status  or  GET /status/<dermatology|ocular>")
    print(f"[status_api] reading manifests from {MODELS_DIR}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
