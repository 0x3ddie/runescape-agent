"""Tests read a small synthetic name table, never the Jagex-derived one (rl/names.py)."""
import numpy as np
import pytest

import names

ROWS = [('npc', 'guard'), ('npc', 'banker'), ('loc', 'tree'), ('loc', 'bank booth'), ('obj', 'logs'),
        ('obj', 'bronze axe'), ('obj', 'raw shrimps'), ('npc', 'fishing spot')]


@pytest.fixture(scope='session')
def name_table_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('names') / 'names.npz'
    kinds, rows = zip(*ROWS)
    vectors = np.random.default_rng(0).normal(size=(len(ROWS), 8)).astype(np.float16)
    np.savez(path, kind=np.array(kinds), name=np.array(rows), vector=vectors)
    return path


@pytest.fixture(autouse=True)
def name_table(name_table_path, monkeypatch):
    monkeypatch.setenv('RL_NAME_TABLE', str(name_table_path))
    names.name_table.cache_clear()
    yield names.name_table()
    names.name_table.cache_clear()
