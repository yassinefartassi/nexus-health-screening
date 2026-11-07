"""
Thin HTTP client for nexus_pfe_server_fed's read-only status API
(fl_server/status_api.py). Used only by app.py's "Apprentissage federe" tab
to show REAL round-by-round history -- current round, client count,
accuracy/loss -- never fabricated numbers. If the status API isn't reachable
(server not running, wrong address, firewall), callers get None back and the
UI must render "waiting for a run", not a fake chart.

This is a monitoring read, not part of the FL protocol itself: Flower's own
gRPC traffic (fl_client/client.py <-> fl_server/server.py) is the only thing
that ever moves model weights. This module only ever GETs small JSON
summaries of what already happened.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

DEFAULT_STATUS_URL = os.environ.get("FL_STATUS_URL", "http://127.0.0.1:8090")


def fetch_status(base_url: str = DEFAULT_STATUS_URL, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """GET {base_url}/status -> {"dermatology": {...}, "ocular": {...}}, or
    None if the status API isn't reachable (nothing has run yet, wrong
    address, server offline, ...)."""
    import requests

    try:
        response = requests.get(f"{base_url.rstrip('/')}/status", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None
