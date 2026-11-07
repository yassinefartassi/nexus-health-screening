"""
MODULE_REGISTRY -- the modular FL design in one dict. server.py, given
just a module name on the command line (--model dermatology|ocular), looks
up the right FederatedModule implementation here and never needs to know
which ML framework that module uses (see base.py). Adding a third
federated module later means adding one FederatedModule subclass and one
line here, nothing else changes.
"""
from __future__ import annotations

from typing import Dict

from fl_server.modules.base import FederatedModule
from fl_server.modules.dermatology import DermatologyModule
from fl_server.modules.ocular import OcularModule

MODULE_REGISTRY: Dict[str, FederatedModule] = {
    "dermatology": DermatologyModule(),
    "ocular": OcularModule(),
}
