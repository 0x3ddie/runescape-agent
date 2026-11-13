"""Simulator tests (conftest.py loads the private tables).

PACK: the pack the native tests read, sim/native.py PACK (logs/native/train-8-nc1-cc0614f.pack, train-8-nc1-rv2's
inputs exported with trees), or for development only the pack the environment variable
SIM_TEST_PACK names: every pack goes stale at an oracle edit, so a working tree's tests read one exported from it."""
import os
from pathlib import Path

from sim.native import PACK as DEFAULT_PACK

PACK = Path(os.environ.get('SIM_TEST_PACK') or DEFAULT_PACK)
