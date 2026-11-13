"""The extracted Draynor tables (sim/build.py), indexed for the simulator.

Coordinates stay absolute game tiles. Grids cover the data area (`area`) on each level of
`levels`, one level after another, row-major in x then z, as sim/extract.ts wrote them. Path
tables are per supported tile (`source`), a tile of one level.
"""
from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path

import numpy as np

from sim.build import DATA, pins

# Collision flags (vendor engine routefinder/flags.ts).
FLAG_NPC, FLAG_NULL = 0x80000, 0x7fffffff
BLOCK = dict(W=0x240108, E=0x240180, S=0x240102, N=0x240120, SW=0x24010e, SE=0x240183, NW=0x240138, NE=0x2401e0,
             N_AND_SE=0x24013e, N_AND_SW=0x2401e3, NE_AND_W=0x24018f, SE_AND_W=0x2401f8)
# Path finder direction flags: the way back toward the source.
D_NORTH, D_EAST, D_SOUTH, D_WEST = 0x1, 0x2, 0x4, 0x8
SEARCH, HALF = 128, 64


@dataclass
class Tables:
    meta: dict
    arrays: dict
    bounded = False  # True for a walled-in map smaller than the search windows (sim/world.py)
    walkmap = None  # the C port's walk steps over frozen layout tables (sim/world.py LayoutTables)

    @classmethod
    def load(cls, path=DATA, check_pins=True):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'{path} is missing; sim/build.py says how each archive is made')
        with np.load(path) as data:
            meta = json.loads(data['meta'].tobytes())
            arrays = {k: data[k] for k in data.files if k != 'meta'}
        if check_pins:
            current = pins()
            if meta['pins'] != current:
                changed = sorted(k for k in current if meta['pins'].get(k) != current[k])
                raise ValueError(f'{path} was built from other inputs ({changed}); rebuild with sim.build')
        return cls(meta, arrays)

    # ---- geometry
    @cached_property
    def area(self):
        return self.meta['data']

    @cached_property
    def size(self):
        a = self.area
        return a['x1'] - a['x0'] + 1, a['z1'] - a['z0'] + 1

    @cached_property
    def levels(self):
        return self.meta['levels']

    def tile(self, level, x, z):
        """Index of (level, x, z) in the data-area grids; -1 outside them."""
        a, (w, h) = self.area, self.size
        if type(level) is int and type(x) is int and type(z) is int:  # the agent's and the path finder's queries
            inside = 0 <= level < len(self.levels) and a['x0'] <= x <= a['x1'] and a['z0'] <= z <= a['z1']
            return level * w * h + (x - a['x0']) * h + (z - a['z0']) if inside else -1
        level, x, z = np.asarray(level), np.asarray(x), np.asarray(z)
        inside = (level >= 0) & (level < len(self.levels)) & (x >= a['x0']) & (x <= a['x1']) & (z >= a['z0']) & (z <= a['z1'])
        return np.where(inside, level * w * h + (x - a['x0']) * h + (z - a['z0']), -1)

    @cached_property
    def collision(self):
        return self.arrays['collision']

    def flags(self, level, x, z):
        i = self.tile(level, x, z)
        return np.where(i >= 0, self.collision[np.maximum(i, 0)], FLAG_NULL)

    @cached_property
    def source_grid(self):
        """Data-area grids of supported-tile indices, -1 where the simulator has no path table."""
        grid = np.full(len(self.levels) * self.size[0] * self.size[1], -1, np.int32)
        sources = np.asarray(self.meta['sources'], np.int64)
        grid[self.tile(sources[:, 2], sources[:, 0], sources[:, 1])] = np.arange(len(sources), dtype=np.int32)
        return grid

    def source(self, level, x, z):
        i = self.tile(level, x, z)
        return int(self.source_grid[i]) if i >= 0 else -1

    @cached_property
    def parents(self):
        return self.arrays['parents'].reshape(-1, SEARCH, SEARCH)

    @cached_property
    def orders(self):
        r = self.meta['order_r']
        return self.arrays['orders'].reshape(-1, 2 * r + 1, 2 * r + 1)

    @cached_property
    def dists(self):
        r = self.meta['dist_r']
        return self.arrays['dists'].reshape(-1, 2 * r + 1, 2 * r + 1)

    @cached_property
    def walks(self):
        r = self.meta['walk_r']
        return self.arrays['walks'].reshape(-1, 2 * r + 1, 2 * r + 1)

    @cached_property
    def entity_reach(self):
        return self.arrays['entity_reach']

    @cached_property
    def snapshot(self):
        """SDK snapshot step masks (sim.paths.STEPS bits) over `snapshot_area`, by level, x-major."""
        a = self.meta['snapshot_area']
        return self.arrays['snapshot'].reshape(len(self.levels), a['x1'] - a['x0'] + 1, a['z1'] - a['z0'] + 1)

    @cached_property
    def sides(self):
        """walk.operable's open sides (W, E, S, N bits) of each data-area tile, on the snapshot."""
        return self.arrays['sides']

    @cached_property
    def los(self):
        r = self.meta['los_r']
        return self.arrays['los'].reshape(-1, 2 * r + 1, 2 * r + 1)

    def line_of_sight(self, level, ax, az, bx, bz):
        r = self.meta['los_r']
        i = self.tile(level, ax, az)
        if i < 0 or max(abs(bx - ax), abs(bz - az)) > r:
            raise KeyError(f'line of sight {level},{ax},{az} -> {bx},{bz} outside the tables')
        return bool(self.los[i, bx - ax + r, bz - az + r])

    # ---- things
    @cached_property
    def loc_types(self):
        return {int(k): v for k, v in self.meta['loc_types'].items()}

    @cached_property
    def npc_types(self):
        return {int(k): v for k, v in self.meta['npc_types'].items()}

    @cached_property
    def obj_types(self):
        return {int(k): v for k, v in self.meta['obj_types'].items()}

    @cached_property
    def obj_ids(self):
        return {v['debugname']: k for k, v in sorted(self.obj_types.items(), reverse=True)}

    def obj_id(self, debugname):
        return self.obj_ids[debugname]

    @cached_property
    def stairs(self):
        """Where the stair scripts jump: (script, loc level, x, z) -> (x, z, level) (sim/build.py stairs)."""
        return {(name, level, x, z): (tx, tz, tl) for name, cases in self.meta['content']['stairs'].items()
                for x, z, level, tx, tz, tl in cases}
