"""names-v1: a thing's identity is its row in the name table, whose vectors a pretrained
text model made from the name and its examine text (scripts/name_vectors.py), so a name
never trained on ("Yew", "Maple tree") still lands near related ones.

It replaced hashing names into 4,000 buckets, where 520 of the cache's 1,625 loc and npc
names shared one ("willow" with "female slave"). The table is Jagex-derived and never in
git; its path is RL_NAME_TABLE, by default logs/names/name-vectors.npz.
"""
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

NAMES_V1 = 'names-v1'
DEFAULT_TABLE = Path(__file__).resolve().parents[1] / 'logs' / 'names' / 'name-vectors.npz'


class NameTable:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            kinds, names, vectors = data['kind'], data['name'], data['vector']
        if len(kinds) != len(names) or len(names) != len(vectors) or vectors.ndim != 2:
            raise ValueError(f'malformed name table {path}')
        self.rows = {(str(k), str(n)): i + 1 for i, (k, n) in enumerate(zip(kinds, names))}
        # Row 0 is no name, or one the cache does not hold: it carries no meaning.
        self.vectors = np.vstack([np.zeros((1, vectors.shape[1]), np.float32), vectors.astype(np.float32)])
        self.digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
        # What a policy saves to name the table it reads (baseline/policy.py).
        self.spec = (len(self.vectors), self.vectors.shape[1], self.digest)

    def row(self, kind, name):
        return self.rows.get((kind, str(name or '').strip().lower()), 0)


@lru_cache(maxsize=None)
def name_table(path=None):
    return NameTable(path or os.environ.get('RL_NAME_TABLE') or DEFAULT_TABLE)


def identities():
    """(npc, loc, obj, place) identity functions: rows of the names-v1 table."""
    table = name_table()
    return dict(npc=lambda name: table.row('npc', name), loc=lambda name: table.row('loc', name),
                obj=lambda name: table.row('obj', name), place=lambda kind, name: table.row(kind, name))
