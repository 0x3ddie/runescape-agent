"""Simulator tests read the private extracted tables when they exist (sim/build.py); without
them the table-backed tests skip, like the name-table tests of the main suite."""
import pytest

from sim.build import DATA


@pytest.fixture(scope='session')
def tables():
    if not DATA.exists():
        pytest.skip('simulator tables logs/sim/data/draynor-nc1.npz absent (docs/RUNNING.md says how to build them)')
    from sim.data import Tables
    return Tables.load()
