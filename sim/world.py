"""The compact training world: seeded ground-floor layouts and their static tables.
`generate(seed, tables)` draws one layout; `LayoutTables` gives sim/engine.World
and sim/agent.Agent the Tables interface over it, so the reference runs on it unchanged; `derive`
computes every per-layout table the engine and the agent read, as frozen arrays.

Declared world (WORLD, compact-world-v1). A 48 x 48 tile ground floor whose south-west tile is
ORIGIN (one 64-tile chunk; no zone-dependent content there). Everything outside it is the declared
boundary: collision flags NULL (FLAG_NULL), as the engine's unallocated zones, so no step, sight
line or reach crosses it. 48 tiles holds the scale of the one native site with all four activities
within 30 tiles: the place scan (32 tiles) covers most of a layout, the action scan (15)
does not, so the policy has to move, and a walk across it takes a fifth of the 240-tick horizon.
Each layout varies, within WORLD's ranges:

- water along one edge (collision FLOOR, as Draynor's sea), a shore whose depth wanders;
- net fishing spots (npc 0_46_49_saltfish) moving among the layout's shore coordinates, which
  replace that type's fishing_movement_enum (fishing_movement.rs2 picks from them);
- normal trees and oaks in one or two groves; copper, tin and iron rocks in a mine (at least one
  copper and one tin); chickens (npc types 41 and 1017) spawned in an area they wander from;
- in a declared fraction of layouts (WORLD['pen_fraction']) that area is a fenced pen with one
  closed double gate, Entrana's pen gate types (PEN_GATE, general_use/configs/gates.loc), which the
  gate scripts open and close (sim/engine.py gate, from gates.rs2) and the loc lifecycle reverts;
- boulders and bushes (blocking scenery with no option) and fences (wall collision with no loc,
  as the map's static walls), which change paths but never cut off a placed resource.

A pen layout has three collision states: closed (the boot map), open (after the gate script: the
closed leaves gone, the open stages standing outward), and the SDK snapshot, in which, as at
Draynor's doors and gates, no gate stands at all. Live queries (the path finder, reach, sight)
read the closed or open state; walk steps, operable sides and the long route read the snapshot.
Without a pen the three are the same map.

Every resource is operable from the layout's main walkable component in the closed state, the
chickens of a pen from it through the gate; a layout that fails a check is redrawn (`attempt`).
Start tiles (`start`) are the closed state's component tiles from which each goal's nearest target
is within WORLD['goal_steps'] steps: normal trees (wood, at any level), every shore coordinate the
spots move among (fish, the worst one), copper and tin (mine), a chicken spawn (melee; in a pen,
the walk to a gate leaf plus the walk from there to a chicken with the gate open). The episode
samples its start from them, before and independently of its goal.

Splits are disjoint seed ranges (SPLITS); a layout depends only on its seed (numpy PCG64 streams).
Whether a seed has a pen is drawn once from its own stream, so the fraction is pen_fraction.

Frozen format. Coordinates are layout-local (absolute = ORIGIN + local); a tile's index is
`x * H + z` (x-major, as the archive). One layout:

    flags          int32 [W, H]   boot collision, gate closed (engine flags, no entity bits)
    flags_open     int32 [W, H]   the gate open (equal to flags without a pen)
    flags_snapshot int32 [W, H]   the SDK snapshot: no gate (equal to flags without a pen)
    locs           int32 [L, 5]   type, x, z, shape, angle; in index order (World loc index); a pen's
                                  main and outer gate leaves are its last two rows
    gate           int32 [2]      loc rows of the main and outer leaf, -1 without a pen
    open_locs      int32 [G, 5]   the open stages the gate script adds, in its order (main, outer);
                                  G = 2 with a pen, else 0. World numbers them after the map's locs
    npcs           int32 [N, 4]   nid, type, x, z spawns, in nid order
    fish           int32 [K, 2]   x, z: the spot type's movement coordinates, in enum order
    walkable       bool  [W, H]   tiles reachable in either state: the supported tiles (sources)
    start          bool  [W, H]   start tiles
    goal_dist      int16 [4, W, H] steps to each goal's nearest target (GOALS order), -1 unreached

`derive(layout, tables)` adds (paths.py ports, the same functions the reference calls); the leading
axis of the live tables is the state, 0 closed and 1 open (identical without a pen):

    steps          uint8   [2, W, H]        canTravel step bits (paths.STEPS order)
    entity_reach   uint16  [2, W, H]        ReachStrategy for a 1x1 entity on the tile (paths.NEIGHBOURS bits)
    los            uint8   [2, W, H, 21, 21] LineValidator from the tile to offsets -10..10
    loc_reach      int16   [2, L + G, 24]   tiles that reach each loc of `locs` then `open_locs` (its own
                                            footprint included), -1 padded; a loc absent in a state is all -1
    order          uint16  [2, T, T]        findPath1 queue position of tile t from source s (65535 none)
    dist           uint8   [2, T, T]        its BFS depth (255 unreached)
    parent         uint8   [2, T, T]        its step back toward s (paths D_* bits; 99 at s; 0 unreached)
    snapshot_steps uint8   [W, H]           the snapshot's step bits
    sides          uint8   [W, H]           reach.ts walk.operable sides (W, E, S, N bits), on the snapshot
    snapshot_order, snapshot_dist, snapshot_parent  [T, T]  the same search on the snapshot: the SDK
                                            long route's tree (its search is findPath1's queue order) and
                                            walk.ts steps (snapshot_dist)

Rows of non-walkable sources are empty. `save_pool` stacks layouts along a first axis, padding
L, N, K and G with -1, with counts n_locs, n_npcs, n_fish, n_open_locs.

    .venv-sim/bin/python -m sim.world show 1000000
    .venv-sim/bin/python -m sim.world pool --split development --count 32 --output logs/sim/world/dev.npz [--derived]
    .venv-sim/bin/python -m sim.world pool --world village-world-v2 --split train --count 128 --first 20000 \
        --output logs/sim/world/train-128-20000-v2.npz      # village-world-v2 (VILLAGE_WORLD), plain: no derived tables
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path

import numpy as np

from baseline.resource_task import GOALS
from sim import paths
from sim.data import FLAG_NULL, HALF, Tables
from sim.engine import DOOR_OPEN, _gate_close_offset, _gate_pair

W = H = 48
ORIGIN = (2880, 3328)
FLOOR = 0x200000
WALK_BLOCKED = 0x240100  # FLOOR | 0x40000 | LOC: a tile no step enters
WORLD = dict(
    version='compact-world-v1', size=(W, H), origin=ORIGIN,
    water_depth=(4, 9), fish_spots=2, fish_coords=(4, 6), fish_spacing=3,
    trees=(3, 6), oaks=(1, 3), groves=(1, 2), grove_spread=6,
    rocks=(3, 6), iron=(0, 2), mine_spread=3,
    chickens=(3, 6), pen_radius=8, chicken_spread=2, pen_fraction=0.5, pen_half=(4, 5),
    obstacles=(2, 10), fences=(0, 3), fence_length=(4, 12), fence_gap=2,
    region_separation=10, goal_steps=40, min_starts=100, attempts=200,
)
# The search study's declared world: WORLD's draws and checks on the whole 64-tile chunk
# at ORIGIN, so trees can stand far enough beyond the 32-tile place scan that one 10-step compass move cannot reveal them;
# every pair of its tiles stays inside the path finder's 128-tile window. sim/search.py plays its wood goal there.
SEARCH_WORLD = dict(WORLD, version='search-world-v1', size=(64, 64))
# The supply study's declared world: SEARCH_WORLD's chunk with three groves whose
# centres stand at least grove_separation apart, so trees of different groves are at least 26 - 2 x 6 = 14 tiles apart
# (sim/search.py groves); 9-12 normal trees and 2-3 oaks shared among them. sim/search.py plays its wood goal there.
SUPPLY_WORLD = dict(SEARCH_WORLD, version='supply-world-v1', trees=(9, 12), oaks=(2, 3), groves=(3, 3), grove_separation=26)
# village-world-v2's layouts (accepted 13 November 2026): WORLD's draws on the whole 64-tile chunk at ORIGIN with region
# centres 20 apart and every source of a start within 60 steps, and five additions, each drawn only by this spec
# (WORLD's layouts and their draws do not change). Every value below the accepted contents is a proposal until it is
# frozen:
# - a field: one region of `field_plants` plants, all of one kind drawn per seed by `field_kinds` (loc debugname,
#   share), on distinct tiles within `field_half` of its centre; each plant has a Pick (sim/engine.py pick);
# - a pen barrier: a pen seed's pen is closed by PEN_DOOR in place of PEN_GATE in `door_fraction` of them;
# - a willow and a yew (trees that a low level cannot cut) in a grove, each in its own share of seeds;
# - a highwayman (HUNTER) in `hunter_fraction` of seeds, spawned on a tile of the start component `hunter_near`
#   (Chebyshev) from one region centre drawn among the pen's, the groves', the mine's and the field's, and only where the
#   share of start tiles within its reach (its type's wanderrange + huntrange, Chebyshev from the spawn) lies in
#   `hunter_share`: near some starts and not all;
# - plants, willows and yews must be operable from the start component, and each plant's own tile walkable (the pick
#   teleports the player onto it).
# Its layouts are VillageLayouts, whose class names the world: its size is the search and supply worlds'. The seed decides the pen
# (PEN_STREAM) as in WORLD, and the five draws of VARIANT_STREAM in order: door, field kind, willow, yew, highwayman.
VILLAGE_WORLD = dict(WORLD, version='village-world-v2', size=(64, 64), by_name=True, region_separation=20, goal_steps=60,
                     field_plants=(6, 16), field_half=3, field_kinds=(('cabbage', 0.5), ('potato', 0.25), ('onion', 0.25)),
                     door_fraction=0.5, willow_fraction=0.5, yew_fraction=0.5,
                     hunter_fraction=0.5, hunter_near=(4, 10), hunter_share=(0.2, 0.6))
WORLDS = {spec['version']: spec for spec in (WORLD, SEARCH_WORLD, SUPPLY_WORLD, VILLAGE_WORLD)}
SPLITS = dict(train=(0, 1_000_000), development=(1_000_000, 1_010_000), final=(2_000_000, 2_010_000))
TREE, OAK = 'tree', 'oaktree'
ROCKS = dict(copper=('copperrock1', 'copperrock2'), tin=('tinrock1',), iron=('ironrock1', 'ironrock2'))
SPOT = '0_46_49_saltfish'
CHICKENS = ('chicken', 'chicken_brown')
OBSTACLES = ('loc_444', 'loc_445', 'bush1')   # Boulder 2x2, Boulder 2x2, Bush 1x1
FENCE = dict(blockwalk=True, blockrange=False)  # a wall with no loc (paths.collision_masks shape 0)
PEN_GATE = ('loc_1551', 'loc_1553')  # main and outer leaf of Entrana's chicken pen gate, closed
PEN_DOOR = 'poordooropen'  # a one-leaf door, closed ([oploc1,_door_closed], doors.rs2): village-world-v2's other pen barrier
PEN_BARRIERS = (*PEN_GATE, PEN_DOOR)  # what closes a pen, closed
WILLOW, YEW = 'willowtree', 'yewtree'  # woodcutting levels 30 and 60 (trees.dbrow)
HUNTER = 'highwayman'  # hunts a player of up to twice its level within its hunt range (sim/engine.py npc_hunt_all)
WALL = 0  # shape: a straight wall
PEN_STREAM = 1 << 20  # the PCG64 stream (seed, PEN_STREAM) decides a seed's pen; attempts use (seed, attempt)
DISTRACT_STREAM = 1 << 21  # the stream (seed, DISTRACT_STREAM) places a development variant's additions (with_distractions)
VARIANT_STREAM = 1 << 22  # the stream (seed, VARIANT_STREAM) draws a VILLAGE_WORLD seed's variant (_variant)
# Development additions for a cause test, each of a kind the native start sets show and the simulator plays fully.
# npcs: goblins, attackable npcs that fight back and are not eligible enemies (Entrana's monks and Rimmington's goblins);
# the reference plays only one-tile npcs (a unicorn or a cow raises 'a large npc in melee' or 'a large npc approaching').
DISTRACTIONS = dict(npcs=dict(type='goblin', count=3, near=(3, 9)))
CENTREPIECE = 10
LOS_R = 10
NPC_BASE = 0


@dataclass(frozen=True, eq=False)
class Layout:
    seed: int
    attempt: int
    flags: np.ndarray
    flags_open: np.ndarray
    flags_snapshot: np.ndarray
    locs: np.ndarray
    gate: np.ndarray
    open_locs: np.ndarray
    npcs: np.ndarray
    fish: np.ndarray
    walkable: np.ndarray
    start: np.ndarray
    goal_dist: np.ndarray
    world = None  # not a field: the declared world a subclass names (VillageLayout), else None (world_of)

    def arrays(self):
        return {f.name: np.asarray(getattr(self, f.name)) for f in fields(self)}


class VillageLayout(Layout):
    """A VILLAGE_WORLD layout: Layout's fields, and a class that names its world, since its size is the search and
    supply worlds' (world_of). dataclasses.replace keeps the class."""
    world = 'village-world-v2'


LAYOUT_CLASSES = {VillageLayout.world: VillageLayout}  # a by_name world's layouts, by the world's name


# ---- drawing --------------------------------------------------------------------------------------
class _Types:
    def __init__(self, t):
        loc = {v['debugname']: k for k, v in t.loc_types.items()}
        npc = {v['debugname']: k for k, v in t.npc_types.items()}
        self.tree, self.oak = loc[TREE], loc[OAK]
        self.rocks = {kind: [loc[n] for n in names] for kind, names in ROCKS.items()}
        self.obstacles = [loc[n] for n in OBSTACLES]
        self.spot = npc[SPOT]
        self.chickens = [npc[n] for n in CHICKENS]
        self.gate = [loc[n] for n in PEN_GATE]
        self.door, self.willow, self.yew = loc[PEN_DOOR], loc[WILLOW], loc[YEW]
        self.plants = {name: loc[name] for name, _ in VILLAGE_WORLD['field_kinds']}
        self.hunter = npc[HUNTER]
        self.size = {k: (v['width'], v['length']) for k, v in t.loc_types.items()}


def _shift(a, dx, dz):
    """out[x + dx, z + dz] = a[x, z], zero where nothing lands."""
    out = np.zeros_like(a)
    w, h = a.shape
    out[max(dx, 0):w + min(dx, 0), max(dz, 0):h + min(dz, 0)] = a[max(-dx, 0):w + min(-dx, 0), max(-dz, 0):h + min(-dz, 0)]
    return out


def flood(steps, seeds):
    """Breadth-first steps from a set of tiles over canTravel's step bits, -1 where unreached. On
    tiles a step can enter, a step and its reverse are allowed alike (walls mark both sides), so
    this is also the distance from each tile to the nearest seed."""
    dist = np.full(steps.shape, -1, np.int16)
    frontier = seeds.copy()
    dist[frontier] = 0
    depth = 0
    while frontier.any():
        depth += 1
        reached = np.zeros_like(frontier)
        for k, (dx, dz) in enumerate(paths.STEPS):
            reached |= _shift(frontier & ((steps >> k) & 1).astype(bool), dx, dz)
        frontier = reached & (dist < 0)
        dist[frontier] = depth
    return dist


def _flags_fn(flags):
    w, h = flags.shape

    def f(x, z):
        return int(flags[x, z]) if 0 <= x < w and 0 <= z < h else FLAG_NULL
    return f


def _footprint(x, z, w, l):
    return [(x + i, z + j) for i in range(w) for j in range(l)]


def _variant(seed, spec, fenced):
    """A VILLAGE_WORLD seed's variant (its spec's comment), drawn once per seed and not per attempt, so each share is the
    spec's: whether its pen is closed by the door, its field's kind, a willow, a yew, a highwayman. {} for a spec without
    a field (WORLD and the others draw nothing here)."""
    if 'field_plants' not in spec:
        return {}
    u = np.random.default_rng([seed, VARIANT_STREAM]).random(5)
    shares = np.cumsum([share for _, share in spec['field_kinds']])
    kind = spec['field_kinds'][min(int(np.searchsorted(shares, u[1], side='right')), len(shares) - 1)][0]
    return dict(door=bool(fenced and u[0] < spec['door_fraction']), kind=kind, willow=bool(u[2] < spec['willow_fraction']),
                yew=bool(u[3] < spec['yew_fraction']), hunter=bool(u[4] < spec['hunter_fraction']))


class _Draw:
    """One attempt at a layout; `layout()` is None when a check fails."""

    def __init__(self, rng, types, fenced, spec, variant=None):
        self.rng, self.ty, self.fenced, self.spec, self.variant = rng, types, fenced, spec, variant or {}
        W, H = self.W, self.H = spec['size']
        self.water = np.zeros((W, H), bool)
        self.taken = np.zeros((W, H), bool)     # loc footprints
        self.kept = np.zeros((W, H), bool)      # no footprint may cover these (rings, pen, shore)
        self.locs, self.npcs, self.fish, self.fences, self.gate = [], [], [], [], []

    def r(self, bounds):
        return int(self.rng.integers(bounds[0], bounds[1] + 1))

    def draw_water(self):
        W, H = self.W, self.H
        side, depth = int(self.rng.integers(4)), self.r(self.spec['water_depth'])
        lo, hi = self.spec['water_depth']
        for i in range(W):
            depth = int(np.clip(depth + self.rng.integers(-1, 2), lo, hi))
            if side == 0:
                self.water[i, :depth] = True
            elif side == 1:
                self.water[i, H - depth:] = True
            elif side == 2:
                self.water[:depth, i] = True
            else:
                self.water[W - depth:, i] = True

    def draw_fish(self):
        W, H = self.W, self.H
        land = ~self.water
        shore = self.water & (_shift(land, 1, 0) | _shift(land, -1, 0) | _shift(land, 0, 1) | _shift(land, 0, -1))
        shore[[0, -1], :] = False
        shore[:, [0, -1]] = False
        candidates = [tuple(int(v) for v in c) for c in np.argwhere(shore)]
        self.rng.shuffle(candidates)
        want = self.r(self.spec['fish_coords'])
        for x, z in candidates:
            if all(max(abs(x - a), abs(z - b)) >= self.spec['fish_spacing'] for a, b in self.fish):
                self.fish.append((x, z))
                if len(self.fish) == want:
                    break
        for x, z in self.fish:  # the land a spot is fished from stays clear
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + dx < W and 0 <= z + dz < H:
                    self.kept[x + dx, z + dz] = True
        return len(self.fish) == want

    def centre(self, margin, away, separation=None):
        """A land tile at least `margin` from water and the edges and `separation` (default region_separation) from
        `away`."""
        W, H = self.W, self.H
        separation = separation or self.spec['region_separation']
        clear = flood(np.full((W, H), 0xff, np.uint8), self.water)  # tiles from the water, ignoring collision
        for _ in range(200):
            x, z = (int(v) for v in self.rng.integers(margin, W - margin, 2))
            if clear[x, z] >= margin and all(max(abs(x - a), abs(z - b)) >= separation for a, b in away):
                return x, z
        return None

    def free(self, x, z, w, l):
        W, H = self.W, self.H
        tiles = _footprint(x, z, w, l)
        return all(1 <= a < W - 1 and 1 <= b < H - 1 and not self.water[a, b] and not self.taken[a, b]
                   and not self.kept[a, b] for a, b in tiles)

    def place_loc(self, type_id, near, spread):
        W = self.W
        w, l = self.ty.size[type_id]
        for _ in range(200):
            if near is None:
                x, z = (int(v) for v in self.rng.integers(1, W - 1, 2))
            else:
                x, z = (int(near[i] + self.rng.integers(-spread, spread + 1)) for i in range(2))
            if self.free(x, z, w, l):
                for a, b in _footprint(x, z, w, l):
                    self.taken[a, b] = True
                self.kept[max(x - 1, 0):x + w + 1, max(z - 1, 0):z + l + 1] = True  # one tile around stays open
                self.locs.append((type_id, x, z, CENTREPIECE, 0))
                return True
        return False

    def draw_things(self):
        pen = self.centre(self.spec['pen_radius'] + 1, [])
        if pen is None:
            return False
        r = self.spec['pen_radius']
        self.kept[pen[0] - r:pen[0] + r + 1, pen[1] - r:pen[1] + r + 1] = True
        groves = []
        for _ in range(self.r(self.spec['groves'])):
            centre = self.centre(3, [pen] + groves, self.spec.get('grove_separation'))
            if centre is None:
                return False
            groves.append(centre)
        mine = self.centre(3, [pen] + groves)
        if mine is None:
            return False
        self.regions = dict(pen=[pen], grove=groves, mine=[mine])
        if self.variant and not self.draw_field([pen] + groves + [mine]):
            return False
        trees = [self.ty.tree] * self.r(self.spec['trees']) + [self.ty.oak] * self.r(self.spec['oaks'])
        for k, tree in enumerate(trees):
            if not self.place_loc(tree, groves[k % len(groves)], self.spec['grove_spread']):
                return False
        for k, tree in enumerate(t for t, here in ((self.ty.willow, self.variant.get('willow')),
                                                       (self.ty.yew, self.variant.get('yew'))) if here):
            if not self.place_loc(tree, groves[k % len(groves)], self.spec['grove_spread']):
                return False
        n = self.r(self.spec['rocks'])
        iron = min(self.r(self.spec['iron']), n - 2)
        kinds = ['copper', 'tin'] + ['iron'] * iron + [('copper', 'tin')[int(self.rng.integers(2))] for _ in range(n - 2 - iron)]
        for kind in kinds:
            options = self.ty.rocks[kind]
            if not self.place_loc(options[int(self.rng.integers(len(options)))], mine, self.spec['mine_spread']):
                return False
        for _ in range(self.r(self.spec['obstacles'])):
            if not self.place_loc(self.ty.obstacles[int(self.rng.integers(len(self.ty.obstacles)))], None, 0):
                return False
        spots = [(self.ty.spot, x, z) for x, z in self.fish[:self.spec['fish_spots']]]
        spread = self.spec['chicken_spread']
        chickens, tiles = self.r(self.spec['chickens']), set()
        while len(tiles) < chickens:
            tiles.add(tuple(int(pen[i] + self.rng.integers(-spread, spread + 1)) for i in range(2)))
        hens = [(self.ty.chickens[int(self.rng.integers(len(self.ty.chickens)))], x, z) for x, z in sorted(tiles)]
        self.npcs = [(NPC_BASE + k, t, x, z) for k, (t, x, z) in enumerate(spots + hens)]
        self.pen = pen
        if self.fenced:
            self.draw_pen(pen)
        return True

    def draw_field(self, away):
        """The field (VILLAGE_WORLD): its own region's centre, `field_plants` plants of the variant's kind on distinct
        free tiles within field_half of it (centrepieces, which block nothing), then the field and a tile around it kept
        clear of later locs."""
        h = self.spec['field_half']
        field = self.centre(h + 1, away)
        if field is None:
            return False
        self.regions['field'] = [field]
        tiles = [(field[0] + i, field[1] + j) for i in range(-h, h + 1) for j in range(-h, h + 1)]
        want, placed = self.r(self.spec['field_plants']), 0
        for k in self.rng.permutation(len(tiles)):
            x, z = tiles[int(k)]
            if placed < want and self.free(x, z, 1, 1):
                self.taken[x, z] = True
                self.locs.append((self.ty.plants[self.variant['kind']], x, z, CENTREPIECE, 0))
                placed += 1
        self.kept[field[0] - h - 1:field[0] + h + 2, field[1] - h - 1:field[1] + h + 2] = True
        return placed == want

    def draw_pen(self, pen):
        """Fences on the edges of the square of half-size pen_half around the pen centre, with a
        double gate in one side: the main leaf on a tile of that side, the outer one on its pair tile
        ([proc,get_pair_coord]), each a wall of the side's angle (0 west, 1 north, 2 east, 3 south),
        away from the corners; open, the leaves stand outward. A door variant (VILLAGE_WORLD) puts one door on the main
        leaf's tile in place of both leaves."""
        r, (cx, cz) = self.r(self.spec['pen_half']), pen
        side = int(self.rng.integers(4))
        along = int(self.rng.integers(-r + 1, r - 1))  # the main leaf's place along its side
        main = ((cx - r, cz + along), (cx + along, cz + r), (cx + r, cz - along), (cx - along, cz - r))[side]
        edges = [((cx - r, z), 0) for z in range(cz - r, cz + r + 1)] + [((x, cz + r), 1) for x in range(cx - r, cx + r + 1)] \
            + [((cx + r, z), 2) for z in range(cz - r, cz + r + 1)] + [((x, cz - r), 3) for x in range(cx - r, cx + r + 1)]
        if self.variant.get('door'):
            self.fences += [(x, z, angle) for (x, z), angle in edges if not (angle == side and (x, z) == main)]
            self.gate = [len(self.locs)]
            self.locs.append((self.ty.door, *main, WALL, side))
            return
        outer = _gate_pair(*main, side, False)
        self.fences += [(x, z, angle) for (x, z), angle in edges if not (angle == side and (x, z) in (main, outer))]
        self.gate = [len(self.locs), len(self.locs) + 1]
        self.locs += [(self.ty.gate[0], *main, WALL, side), (self.ty.gate[1], *outer, WALL, side)]

    def open_locs(self, t):
        """[proc,open_gate] from the closed main leaf: each leaf's next stage, turned a quarter, at
        the main leaf's tile moved one and two tiles outward ([proc,gate_set_close]). A door: [oploc1,_door_closed]'s
        next stage on the tile in front ([proc,door_open]), turned a quarter (sim/engine.py door)."""
        if not self.gate:
            return []
        if len(self.gate) == 1:
            type_id, x, z, shape, angle = self.locs[self.gate[0]]
            dx, dz = DOOR_OPEN[shape][angle]
            return [(t.loc_types[type_id]['params']['next_loc_stage'], x + dx, z + dz, shape, (angle + 1) % 4)]
        (main, x, z, _, angle), (outer, *_) = (self.locs[k] for k in self.gate)
        return [(t.loc_types[leaf]['params']['next_loc_stage'], x + dx, z + dz, WALL, (angle + 3) % 4)
                for leaf, (dx, dz) in ((main, _gate_close_offset(angle, 1)), (outer, _gate_close_offset(angle, 2)))]

    def draw_fences(self):
        W, H = self.W, self.H
        for _ in range(self.r(self.spec['fences'])):
            for _ in range(100):
                x, z = (int(v) for v in self.rng.integers(2, W - 2, 2))
                length, along_x = self.r(self.spec['fence_length']), bool(self.rng.integers(2))
                tiles = [(x + i, z) if along_x else (x, z + i) for i in range(length)]
                gap = int(self.rng.integers(length - self.spec['fence_gap'] + 1))
                r = self.spec['pen_radius']
                if all(2 <= a < W - 2 and 2 <= b < H - 2 and not self.water[a, b] and not self.taken[a, b]
                       and max(abs(a - self.pen[0]), abs(b - self.pen[1])) > r + 1 for a, b in tiles):
                    # a wall on the tile's north edge (angle 1) along x, its east edge (angle 2) along z
                    self.fences += [(a, b, 1 if along_x else 2) for i, (a, b) in enumerate(tiles)
                                    if not gap <= i < gap + self.spec['fence_gap']]
                    break

    def collision(self, t, state='closed'):
        """The map with the gate closed, open, or absent (the snapshot)."""
        flags = np.where(self.water, FLOOR, 0).astype(np.int32)
        walls = [(x, z, WALL, angle, FENCE) for x, z, angle in self.fences]
        locs = [loc for k, loc in enumerate(self.locs) if k not in self.gate or state == 'closed']
        for type_id, x, z, shape, angle in locs + (self.open_locs(t) if state == 'open' else []):
            walls.append((x, z, shape, angle, t.loc_types[type_id]))
        for x, z, shape, angle, loc_type in walls:
            for a, b, mask in paths.collision_masks(x, z, shape, angle, loc_type):
                flags[a, b] |= mask
        return flags


def _component(steps, flags):
    """The largest set of tiles mutually reachable by steps, among tiles a step can enter."""
    open_ = (flags & WALK_BLOCKED) == 0
    best = np.zeros_like(open_)
    left = open_.copy()
    while left.any():
        seed = np.zeros_like(left)
        seed[tuple(np.argwhere(left)[0])] = True
        part = (flood(steps, seed) >= 0) & open_
        if part.sum() > best.sum():
            best = part
        left &= ~part
    return best


def _reach(flags, things, t):
    """Tiles inside the layout that operate each thing: ('loc', type, x, z, shape, angle) or ('npc', x, z)."""
    f, (w, h) = _flags_fn(flags), flags.shape
    out = []
    for thing in things:
        if thing[0] == 'loc':
            _, type_id, x, z, shape, angle = thing
            lt = t.loc_types[type_id]
            tiles = paths.loc_reaching(f, x, z, lt['width'], lt['length'], angle, shape, lt['forceapproach'])
        else:
            tiles = paths.entity_reaching_live(f, thing[1], thing[2])
        out.append([(a, b) for a, b in tiles if 0 <= a < w and 0 <= b < h])
    return out


def _seeds(tiles_lists, within):
    seeds = np.zeros_like(within)
    for tiles in tiles_lists:
        for a, b in tiles:
            seeds[a, b] |= within[a, b]
    return seeds


def generate(seed, tables, spec=WORLD):
    """The layout of a seed in a declared world (WORLD, or SEARCH_WORLD): whether it has a pen (stream (seed,
    PEN_STREAM)), then the first attempt (PCG64 stream (seed, attempt)) that passes the checks. VILLAGE_WORLD: also its
    variant (_variant), and in a highwayman seed an attempt whose hunter finds no spawn (place_hunter) fails too."""
    types = _Types(tables)
    fenced = np.random.default_rng([seed, PEN_STREAM]).random() < spec['pen_fraction']  # once per seed, not per attempt
    variant = _variant(seed, spec, fenced)
    for attempt in range(spec['attempts']):
        d = _Draw(np.random.default_rng([seed, attempt]), types, fenced, spec, variant)
        d.draw_water()
        if not d.draw_fish() or not d.draw_things():
            continue
        d.draw_fences()
        flags, flags_open, flags_snapshot = (d.collision(tables, state) for state in ('closed', 'open', 'snapshot'))
        layout = _checked(seed, attempt, flags, flags_open, flags_snapshot, d.locs, d.gate, d.open_locs(tables), d.npcs, d.fish,
                          types, tables, spec)
        if layout is not None and variant.get('hunter'):
            layout = place_hunter(layout, d.rng, d.regions, types, tables, spec)
        if layout is not None:
            return layout
    raise RuntimeError(f'no layout for seed {seed} in {spec["attempts"]} attempts')


def hunter_reach(tables):
    """Chebyshev tiles from its spawn within which the highwayman can hunt a player: its wanderrange plus its huntrange."""
    nt = tables.npc_types[_Types(tables).hunter]
    return nt['wanderrange'] + nt['huntrange']


def place_hunter(layout, rng, regions, types, tables, spec):
    """The layout with the highwayman's spawn (VILLAGE_WORLD's comment) as its last npc, or None when no tile fits: a
    region drawn among `regions` (name -> centres, in the order pen, grove, mine, field), then the tiles of the start
    component (sim/combat.py outside) hunter_near from its centre with no npc spawn, in an order the attempt's stream
    draws; the first whose share of start tiles within hunter_reach lies in hunter_share."""
    import dataclasses
    centres = [c for name in ('pen', 'grove', 'mine', 'field') for c in regions[name]]
    cx, cz = centres[int(rng.integers(len(centres)))]
    lo, hi = spec['hunter_near']
    reached = (flood(paths.step_masks(layout.flags), layout.start) >= 0) & layout.walkable
    spawns = {(int(x), int(z)) for _, _, x, z in layout.npcs}
    tiles = [(int(x), int(z)) for x, z in np.argwhere(reached)
             if lo <= max(abs(x - cx), abs(z - cz)) <= hi and (int(x), int(z)) not in spawns]
    starts, reach = np.argwhere(layout.start), hunter_reach(tables)
    low, high = spec['hunter_share']
    for k in rng.permutation(len(tiles)):
        x, z = tiles[int(k)]
        share = float(np.mean(np.max(np.abs(starts - (x, z)), axis=1) <= reach))
        if low <= share <= high:
            row = (len(layout.npcs) + NPC_BASE, types.hunter, x, z)
            return dataclasses.replace(layout, npcs=np.vstack([layout.npcs, np.asarray([row], np.int32)]).astype(np.int32))
    return None


def _checked(seed, attempt, flags, flags_open, flags_snapshot, locs, gate, open_locs, npcs, fish, types, tables, spec):
    """A drawn map's layout (module docstring), or None when a check fails: every resource operable from the main
    component (the pen's chickens through the open gate), each goal's distances and at least min_starts start tiles."""
    W, H = flags.shape
    steps, steps_open = paths.step_masks(flags), paths.step_masks(flags_open)
    outside = _component(steps, flags)
    inside = (flood(steps_open, outside) >= 0) & ((flags_open & WALK_BLOCKED) == 0)
    walkable = outside | inside
    loc = lambda kinds: [('loc', ty, x, z, shape, angle) for k, (ty, x, z, shape, angle) in enumerate(locs)
                         if ty in kinds and k not in gate]
    hens = [('npc', x, z) for _, ty, x, z in npcs if ty in types.chickens]
    plants = set(types.plants.values())
    resources = loc({types.tree, types.oak, types.willow, types.yew} | plants | {r for rocks in types.rocks.values() for r in rocks})
    leaves = [('loc', *locs[k]) for k in gate]
    operable = lambda flags_, things, within: all(any(within[a, b] for a, b in tiles) for tiles in _reach(flags_, things, tables))
    if not (operable(flags, resources + [('npc', x, z) for x, z in fish] + leaves, outside)
            and operable(flags_open, hens, walkable)):
        return None
    if not all(outside[x, z] for ty, x, z, _, _ in locs if ty in plants):  # a pick teleports the player onto its plant
        return None
    goal_dist = np.zeros((len(GOALS), W, H), np.int16)
    targets = dict(wood=[loc({types.tree})], fish=[[('npc', x, z)] for x, z in fish],
                   mine=[loc(set(types.rocks['copper']) | set(types.rocks['tin']))])
    for g, goal in enumerate(GOALS[:3]):
        worst = None
        for group in targets[goal]:  # fish: every coordinate a spot may stand on must be in reach
            dist = np.where(outside, flood(steps, _seeds(_reach(flags, group, tables), outside)), -1)
            worst = dist if worst is None else np.where((worst < 0) | (dist < 0), -1, np.maximum(worst, dist))
        goal_dist[g] = worst
    to_hens = flood(steps_open, _seeds(_reach(flags_open, hens, tables), walkable))
    if leaves:  # to a leaf the gate opens from, then through the open gate
        melee = np.full((W, H), -1, np.int16)
        for a, b in {tile for tiles in _reach(flags, leaves, tables) for tile in tiles if outside[tile]}:
            at = np.zeros((W, H), bool)
            at[a, b] = True
            there = flood(steps, at)
            via = np.where((there >= 0) & (to_hens[a, b] >= 0), there + to_hens[a, b], -1)
            melee = np.where((melee < 0) | ((via >= 0) & (via < melee)), via, melee)
        goal_dist[3] = np.where(outside, melee, -1)
    else:
        goal_dist[3] = np.where(outside, to_hens, -1)
    start = outside & np.all((goal_dist >= 0) & (goal_dist <= spec['goal_steps']), axis=0)
    if start.sum() < spec['min_starts']:
        return None
    rows = lambda items: np.asarray(items, np.int32).reshape(-1, 5)
    return LAYOUT_CLASSES.get(spec['version'], Layout)(
        seed=int(seed), attempt=attempt, flags=flags, flags_open=flags_open, flags_snapshot=flags_snapshot, locs=rows(locs),
        gate=np.asarray((list(gate) + [-1, -1])[:2] if gate else [-1, -1], np.int32), open_locs=rows(open_locs),
        npcs=np.asarray(npcs, np.int32).reshape(-1, 4), fish=np.asarray(fish, np.int32).reshape(-1, 2), walkable=walkable,
        start=start, goal_dist=goal_dist)


def relocate(layout, tables, moved):
    """The layout with some locs moved: `moved` maps a row of `locs` to its new local (x, z). Their collision leaves the
    old footprint and takes the new one in every state; reach, distances and start tiles are recomputed with generate's
    checks, so the result is None when one fails. Everything else, npcs and the order of `locs` included, is unchanged
    (a development tool: the search study's twin worlds, sim/search.py)."""
    types, flags = _Types(tables), [layout.flags.copy(), layout.flags_open.copy(), layout.flags_snapshot.copy()]
    locs = [tuple(int(v) for v in row) for row in layout.locs]
    for k, (x, z) in moved.items():
        type_id, ox, oz, shape, angle = locs[k]
        lt = tables.loc_types[type_id]
        for state in flags:
            for a, b, mask in paths.collision_masks(ox, oz, shape, angle, lt):
                state[a, b] &= ~mask
            for a, b, mask in paths.collision_masks(x, z, shape, angle, lt):
                state[a, b] |= mask
        locs[k] = (type_id, int(x), int(z), shape, angle)
    gate = [int(k) for k in layout.gate if k >= 0]
    return _checked(layout.seed, layout.attempt, *flags, locs, gate, [tuple(int(v) for v in r) for r in layout.open_locs],
                    [tuple(int(v) for v in r) for r in layout.npcs], [tuple(int(v) for v in r) for r in layout.fish], types, tables,
                    world_of(layout))


def without_locs(layout, tables, removed):
    """The layout without the loc rows in `removed` (the supply study's scarce variant: trees of the start's grove taken
    away). Their collision leaves every state; reach and distances are recomputed with generate's checks, so the result is
    None when one fails. The count of general start tiles is not required: a supply episode starts on its own tiles (sim/
    search.py supply_sites), and fewer trees can only lengthen the walks it counts. Gate rows are renumbered; everything
    else is unchanged."""
    removed = set(int(k) for k in removed)
    gate = [int(k) for k in layout.gate if k >= 0]
    if removed & set(gate):
        raise ValueError('a gate leaf cannot be removed')
    types, flags = _Types(tables), [layout.flags.copy(), layout.flags_open.copy(), layout.flags_snapshot.copy()]
    locs = [tuple(int(v) for v in row) for row in layout.locs]
    for k in sorted(removed):
        type_id, x, z, shape, angle = locs[k]
        for state in flags:
            for a, b, mask in paths.collision_masks(x, z, shape, angle, tables.loc_types[type_id]):
                state[a, b] &= ~mask
    kept = [row for k, row in enumerate(locs) if k not in removed]
    gate = [k - sum(r < k for r in removed) for k in gate]
    return _checked(layout.seed, layout.attempt, *flags, kept, gate, [tuple(int(v) for v in r) for r in layout.open_locs],
                    [tuple(int(v) for v in r) for r in layout.npcs], [tuple(int(v) for v in r) for r in layout.fish], types, tables,
                    dict(world_of(layout), min_starts=0))


def with_distractions(layout, tables, kinds, around):
    """A development variant of a layout: the same layout plus the DISTRACTIONS of `kinds`, placed from the stream
    (layout seed, DISTRACT_STREAM) on walkable tiles `near` steps (Chebyshev) from `around` (a local tile, the trial's
    start), never on it. Npcs change no collision, so every other part of the layout, its start tiles included, is the
    same."""
    import dataclasses
    unknown = set(kinds) - set(DISTRACTIONS)
    if unknown:
        raise ValueError(f'no such distraction: {sorted(unknown)}')
    rng = np.random.default_rng([layout.seed, DISTRACT_STREAM])
    npcs = [tuple(row) for row in layout.npcs.tolist()]
    if 'npcs' in kinds:
        spec = DISTRACTIONS['npcs']
        ty = next(k for k, v in tables.npc_types.items() if v['debugname'] == spec['type'])
        lo, hi = spec['near']
        tiles = [(int(x), int(z)) for x, z in np.argwhere(layout.walkable)
                 if lo <= max(abs(x - around[0]), abs(z - around[1])) <= hi]
        first = max(n for n, *_ in npcs) + 1
        for k, i in enumerate(rng.choice(len(tiles), spec['count'], replace=False)):
            npcs.append((first + k, ty, *tiles[int(i)]))
    return dataclasses.replace(layout, npcs=np.asarray(npcs, np.int32).reshape(-1, 4))


def world_of(layout):
    """The layout's declared world when its class names it (VillageLayout), else the first declared
    world of the layout's size among those that do not. The search and supply worlds share a size and every setting
    _checked reads (size, goal_steps, min_starts); a pool names its world (save_pool's version)."""
    if layout.world is not None:
        return WORLDS[layout.world]
    found = [spec for spec in WORLDS.values() if spec['size'] == layout.flags.shape and not spec.get('by_name')]
    if not found:
        raise ValueError(f'no declared world of size {layout.flags.shape}')
    return found[0]


def split_seeds(split, count, first=0):
    lo, hi = SPLITS[split]
    if lo + first + count > hi:
        raise ValueError(f'the {split} split has {hi - lo} seeds')
    return list(range(lo + first, lo + first + count))


# ---- static tables ---------------------------------------------------------------------------------
class _Probe:
    """A flags function whose every test passes and is recorded: the (x, z, mask) checks a
    LineValidator ray makes, which depend only on its offset, not on the flags."""

    def __init__(self):
        self.checks = []

    def __call__(self, x, z):
        probe = self

        class Read:
            def __and__(self, mask):
                probe.checks.append((x, z, mask))
                return 0
        return Read()


def _sight_checks():
    checks = {}
    for dx in range(-LOS_R, LOS_R + 1):
        for dz in range(-LOS_R, LOS_R + 1):
            probe = _Probe()
            paths.line_of_sight(probe, 0, 0, dx, dz)
            checks[(dx, dz)] = probe.checks
    return checks


def live_tables(flags, locs, t):
    """One live state's per-tile tables: step bits, entity reach, sight lines, and the reach of each
    (type, x, z, shape, angle) of `locs` (None for a loc absent in the state)."""
    f, (W, H) = _flags_fn(flags), flags.shape
    entity = np.zeros((W, H), np.uint16)
    for x in range(W):
        for z in range(H):
            entity[x, z] = sum(1 << b for b, (dx, dz) in enumerate(paths.NEIGHBOURS)
                               if paths.reached(f, x + dx, z + dz, x, z, 1, 1, 0, -2, 0))
    pad = LOS_R + 1
    big = np.pad(flags, pad, constant_values=FLAG_NULL)
    los = np.zeros((W, H, 2 * LOS_R + 1, 2 * LOS_R + 1), np.uint8)
    for (dx, dz), checks in _SIGHT.items():
        ok = np.ones((W, H), bool)
        for cx, cz, mask in checks:
            ok &= (big[pad + cx:pad + cx + W, pad + cz:pad + cz + H] & mask) == 0
        los[:, :, dx + LOS_R, dz + LOS_R] = ok
    reach = np.full((len(locs), 24), -1, np.int16)
    for i, loc in enumerate(locs):
        if loc is None:
            continue
        type_id, x, z, shape, angle = (int(v) for v in loc)
        lt = t.loc_types[type_id]
        tiles = paths.loc_reaching(f, x, z, lt['width'], lt['length'], angle, shape, lt['forceapproach'])
        tiles = [a * H + b for a, b in tiles if 0 <= a < W and 0 <= b < H]
        reach[i, :len(tiles)] = tiles
    return dict(steps=paths.step_masks(flags), entity_reach=entity, los=los, loc_reach=reach)


def state_locs(layout):
    """The locs of each live state, rows of `locs` then `open_locs`: closed, the gate leaves stand and
    the open stages are absent; open, the other way round."""
    gate = set(int(k) for k in layout.gate if k >= 0)
    closed = [loc for loc in layout.locs] + [None] * len(layout.open_locs)
    opened = [None if k in gate else loc for k, loc in enumerate(layout.locs)] + [loc for loc in layout.open_locs]
    return closed, opened


def snapshot_tables(layout):
    """The snapshot's step bits and walk.operable sides (sim/extract.ts reads both from the SDK snapshot)."""
    W, H = layout.flags_snapshot.shape
    padded = np.pad(layout.flags_snapshot, 1, constant_values=FLAG_NULL)
    flagged = lambda dx, dz, bit: (padded[1 + dx:W + 1 + dx, 1 + dz:H + 1 + dz] & bit) != 0
    sides = np.zeros((W, H), np.uint8)
    for k, (dx, dz, side, own) in enumerate(((-1, 0, 0x8, 0x80), (1, 0, 0x80, 0x8), (0, -1, 0x2, 0x20), (0, 1, 0x20, 0x2))):
        sides |= ((~flagged(dx, dz, side) & ~flagged(0, 0, own)).astype(np.uint8) << k)
    return dict(snapshot_steps=paths.step_masks(layout.flags_snapshot), sides=sides)


_SIGHT = _sight_checks()


def tree_rows(steps, x, z):
    """paths.search from local (x, z) as rows over the layout: order, dist, parent."""
    tree, (W, H) = paths.search(steps, 0, 0, x, z, bounded=True), steps.shape
    ox, oz = HALF - x, HALF - z
    order = tree.orders[ox:ox + W, oz:oz + H]
    return (np.where(order < 0, 65535, order).astype(np.uint16), tree.dists[ox:ox + W, oz:oz + H].astype(np.uint8),
            tree.parents[ox:ox + W, oz:oz + H].astype(np.uint8))


def _window(order, dist, parent, x, z):
    """A paths.Tree in the search window's coordinates from layout rows."""
    W, H = order.shape
    ox, oz = HALF - x, HALF - z
    parents = np.zeros((paths.SEARCH, paths.SEARCH), np.uint8)
    orders = np.full((2 * HALF + 1, 2 * HALF + 1), 65535, np.uint16)
    dists = np.full((2 * HALF + 1, 2 * HALF + 1), 255, np.uint8)
    parents[ox:ox + W, oz:oz + H], orders[ox:ox + W, oz:oz + H], dists[ox:ox + W, oz:oz + H] = parent, order, dist
    return paths.Tree(parents, orders, HALF, dists, HALF, 65535)


def derive(layout, t):
    """Every per-layout table, frozen (module docstring)."""
    states = [(layout.flags, locs) for locs in state_locs(layout)]
    states[1] = (layout.flags_open, states[1][1])
    live = [live_tables(flags, locs, t) for flags, locs in states]
    out = {k: np.stack([table[k] for table in live]) for k in live[0]}
    out.update(snapshot_tables(layout))
    W, H = layout.flags.shape
    T = W * H
    out.update(order=np.full((2, T, T), 65535, np.uint16), dist=np.full((2, T, T), 255, np.uint8),
               parent=np.zeros((2, T, T), np.uint8),
               snapshot_order=np.full((T, T), 65535, np.uint16), snapshot_dist=np.full((T, T), 255, np.uint8),
               snapshot_parent=np.zeros((T, T), np.uint8))
    same = layout.gate[0] < 0
    for x, z in np.argwhere(layout.walkable):
        x, z, i = int(x), int(z), int(x) * H + int(z)
        snap = tree_rows(out['snapshot_steps'], x, z)
        out['snapshot_order'][i], out['snapshot_dist'][i], out['snapshot_parent'][i] = (r.ravel() for r in snap)
        for state in range(2):
            rows = snap if same else tree_rows(out['steps'][state], x, z)
            out['order'][state, i], out['dist'][state, i], out['parent'][state, i] = (r.ravel() for r in rows)
    return out


# ---- the Tables interface ----------------------------------------------------------------------------
SHARED = ('trigger_ids', 'param_defaults', 'content', 'named_enums', 'environment', 'loc_types', 'npc_types', 'obj_types')


class _PerSource:
    """t.parents[src], t.orders[src], t.dists[src] (a paths.Tree's window arrays, boot map), t.walks[src, i, j]
    (snapshot) over a layout, from frozen rows or computed on first use (least recently used kept)."""

    def __init__(self, owner, part):
        self.owner, self.part = owner, part

    def __getitem__(self, key):
        if self.part == 'walks':
            src, i, j = key
            r = self.owner.meta['walk_r']
            d = self.owner.window(src, snapshot=True).dist(i - r, j - r)
            return -1 if d == 255 else d
        return getattr(self.owner.window(key), self.part)


class LayoutTables(Tables):
    """sim.data.Tables over one layout: the archive's types and content, the layout's map and
    things at absolute coordinates (ORIGIN + local), and its tables from `derive` when given,
    else computed as queried (the same functions)."""
    bounded = True
    CACHE = 256

    def __init__(self, base: Tables, layout: Layout, derived=None):
        x0, z0 = ORIGIN
        W, H = self.W, self.H = layout.flags.shape
        area = dict(x0=x0, x1=x0 + W - 1, z0=z0, z1=z0 + H - 1)
        meta = {k: base.meta[k] for k in SHARED}
        spot = base.npc_types[_Types(base).spot]
        enums = dict(base.meta['enums'])
        enum = str(spot['params']['fishing_movement_enum'])
        enums[enum] = dict(enums[enum], values=[[k, x0 + int(x), z0 + int(z), 0] for k, (x, z) in enumerate(layout.fish)])
        if derived is None:
            static = live_tables(layout.flags, state_locs(layout)[0], base) | snapshot_tables(layout)
        else:
            static = {k: derived[k][0] if k in ('steps', 'entity_reach', 'los', 'loc_reach') else derived[k] for k in derived}
        locs = []
        for type_id, x, z, shape, angle in layout.locs:
            lt = base.loc_types[int(type_id)]
            locs.append(dict(type=int(type_id), x=x0 + int(x), z=z0 + int(z), level=0, shape=int(shape), angle=int(angle),
                             width=lt['width'], length=lt['length']))
        loc_reach = [[v for tile in row if tile >= 0 for v in (x0 + int(tile) // H, z0 + int(tile) % H)]
                     for row in static['loc_reach'][:len(layout.locs)]]
        sources = [[x0 + int(x), z0 + int(z), 0] for x, z in np.argwhere(layout.walkable)]
        meta.update(schema=world_of(layout)['version'], data=area, support=area, snapshot_area=area, levels=[0], enums=enums,
                    sources=sources, locs=locs, loc_reach=loc_reach, objs=[], multiway=[],
                    npcs=[dict(nid=int(n), type=int(ty), x=x0 + int(x), z=z0 + int(z), level=0,
                               size=base.npc_types[int(ty)]['size']) for n, ty, x, z in layout.npcs],
                    order_r=HALF, dist_r=HALF, walk_r=48, los_r=LOS_R)
        arrays = dict(collision=layout.flags.ravel().astype(np.int32), entity_reach=static['entity_reach'].ravel(),
                      sides=static['sides'].ravel(), los=static['los'].reshape(W * H, -1),
                      snapshot=static['snapshot_steps'][None])
        super().__init__(meta, arrays)
        self.layout, self.derived = layout, derived
        self.steps, self.snapshot_steps = static['steps'], static['snapshot_steps']
        for name in ('loc_types', 'npc_types', 'obj_types', 'obj_ids'):
            self.__dict__[name] = getattr(base, name)
        self._trees, self._snapshot = OrderedDict(), OrderedDict()

    def local(self, src):
        x, z, _ = self.meta['sources'][src]
        return x - ORIGIN[0], z - ORIGIN[1]

    @cached_property
    def walkmap(self):
        """The C port's walk steps over the frozen snapshot distances (sim/_paths.c WalkMap), or None: no port built,
        or no frozen tables (the rows are then computed as queried, in Python)."""
        if paths._C is None or self.derived is None:
            return None
        local = np.asarray([s[:2] for s in self.meta['sources']], np.int64) - np.asarray(ORIGIN, np.int64)
        return paths._C.WalkMap(np.ascontiguousarray(self.derived['snapshot_dist']), local, self.W, self.H, self.meta['walk_r'])

    def rows(self, src, snapshot=False):
        """The boot map's (closed) search from a source as layout rows, or the snapshot's."""
        if self.derived is not None:
            x, z = self.local(src)
            names = ('snapshot_order', 'snapshot_dist', 'snapshot_parent') if snapshot else ('order', 'dist', 'parent')
            return tuple((self.derived[k] if snapshot else self.derived[k][0])[x * self.H + z].reshape(self.W, self.H) for k in names)
        return tree_rows(self.snapshot_steps if snapshot else self.steps, *self.local(src))

    def window(self, src, snapshot=False):
        cache = self._snapshot if snapshot else self._trees
        return self._cached(cache, src, lambda: _window(*self.rows(src, snapshot), *self.local(src)))

    def _cached(self, cache, key, make):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = cache[key] = make()
        if len(cache) > self.CACHE:
            cache.popitem(last=False)
        return value

    @cached_property
    def parents(self):
        return _PerSource(self, 'parents')

    @cached_property
    def orders(self):
        return _PerSource(self, 'orders')

    @cached_property
    def dists(self):
        return _PerSource(self, 'dists')

    @cached_property
    def walks(self):
        return _PerSource(self, 'walks')

    @cached_property
    def stairs(self):
        return {}


# ---- pools ----------------------------------------------------------------------------------------
PADDED = dict(locs=0, npcs=0, fish=0, open_locs=0, loc_reach=1)  # variable-length arrays: the axis padded with -1


def save_pool(path, layouts, derived=None, version=None):
    """Stacked layouts (and their derived tables, when given), padding the variable-length arrays. version: the
    declared world, when its size alone does not name it (the supply world)."""
    rows = [layout.arrays() | (derived[k] if derived else {}) for k, layout in enumerate(layouts)]
    specs = {world_of(layout)['version'] for layout in layouts}
    if len(specs) != 1:
        raise ValueError(f'a pool holds one declared world, not {sorted(specs)}')
    found = specs.pop()
    if version is not None and (WORLDS[version]['size'] != WORLDS[found]['size']
                                or bool(WORLDS[version].get('by_name')) != bool(WORLDS[found].get('by_name'))):
        raise ValueError(f'{version} layouts are not {found} layouts')
    out = dict(version=np.asarray(version or found), origin=np.asarray(ORIGIN, np.int32))
    for name in rows[0]:
        values = [np.asarray(r[name]) for r in rows]
        if name in PADDED:
            axis = PADDED[name]
            longest = max(v.shape[axis] for v in values)
            out[f'n_{name}'] = np.asarray([v.shape[axis] for v in values], np.int32)
            pad = lambda v: [(0, longest - v.shape[axis]) if k == axis else (0, 0) for k in range(v.ndim)]
            values = [np.pad(v, pad(v), constant_values=-1) for v in values]
        out[name] = np.stack(values)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def load_pool(path):
    """Layouts (and derived tables, if saved) of a pool file, unpadded."""
    with np.load(path) as data:
        if str(data['version']) not in WORLDS:
            raise ValueError(f"{path} holds {data['version']}, not one of {sorted(WORLDS)}")
        arrays = {k: data[k] for k in data.files}
    cls = LAYOUT_CLASSES.get(str(arrays['version']), Layout)  # a by_name world's layouts carry its name
    names = [f.name for f in fields(Layout)]
    layouts, derived = [], []
    for k in range(len(arrays['seed'])):
        row = {}
        for name, value in arrays.items():
            if name in ('version', 'origin') or name.startswith('n_'):
                continue
            n = arrays[f'n_{name}'][k] if name in PADDED else None
            row[name] = value[k] if n is None else value[k][:n] if PADDED[name] == 0 else value[k][:, :n]
        layouts.append(cls(**{n: (int(row[n]) if n in ('seed', 'attempt') else row[n]) for n in names}))
        rest = {n: v for n, v in row.items() if n not in names}
        derived.append(rest or None)
    return layouts, derived


# ---- inspection ---------------------------------------------------------------------------------------
def render(layout, t):
    """Rows north first: ~ water, # loc, T tree, O oak, W willow, Y yew, c copper, t tin, i iron, G gate leaf, D pen
    door, p field plant, f fishing coordinate, h chicken spawn, H highwayman spawn, - and | fence edges (north or south,
    east or west), . start tile, , other walkable tile."""
    W, H = layout.flags.shape
    grid = [[' '] * W for _ in range(H)]
    for x in range(W):
        for z in range(H):
            f = int(layout.flags[x, z])
            grid[z][x] = ('~' if f & FLOOR else '#' if f & 0x100 else '.' if layout.start[x, z]
                          else ',' if layout.walkable[x, z] else ' ')
            if f & 0x22:
                grid[z][x] = '-'
            elif f & 0x88:
                grid[z][x] = '|'
    marks = {TREE: 'T', OAK: 'O', 'copperrock1': 'c', 'copperrock2': 'c', 'tinrock1': 't', 'ironrock1': 'i', 'ironrock2': 'i',
             PEN_GATE[0]: 'G', PEN_GATE[1]: 'G', WILLOW: 'W', YEW: 'Y', PEN_DOOR: 'D',
             **{name: 'p' for name, _ in VILLAGE_WORLD['field_kinds']}}
    for type_id, x, z, _, _ in layout.locs:
        mark = marks.get(t.loc_types[int(type_id)]['debugname'])
        if mark:
            grid[z][x] = mark
    for x, z in layout.fish:
        grid[z][x] = 'f'
    for _, type_id, x, z in layout.npcs:
        name = t.npc_types[int(type_id)]['debugname']
        if name in CHICKENS or name == HUNTER:
            grid[z][x] = 'h' if name in CHICKENS else 'H'
    return '\n'.join(''.join(row) for row in reversed(grid))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    show = sub.add_parser('show', help='draw one layout')
    show.add_argument('seed', type=int)
    show.add_argument('--world', choices=(WORLD['version'], VILLAGE_WORLD['version']), default=WORLD['version'],
                      help='the declared world whose layout to draw')
    pool = sub.add_parser('pool', help='generate and save a split')
    pool.add_argument('--world', choices=(WORLD['version'], VILLAGE_WORLD['version']), default=WORLD['version'],
                      help='the declared world of the layouts (the pool file records it)')
    pool.add_argument('--split', choices=SPLITS, required=True)
    pool.add_argument('--count', type=int, required=True)
    pool.add_argument('--first', type=int, default=0)
    pool.add_argument('--output', type=Path, required=True)
    pool.add_argument('--derived', action='store_true', help='add the per-source trees of both states and the snapshot')
    args = parser.parse_args()
    tables = Tables.load()
    spec = WORLDS[args.world]
    if args.command == 'show':
        layout = generate(args.seed, tables, spec)
        print(f'seed {layout.seed} attempt {layout.attempt}: {len(layout.locs)} locs, {len(layout.npcs)} npcs, '
              f'{int(layout.walkable.sum())} walkable, {int(layout.start.sum())} start tiles')
        print(render(layout, tables))
        return
    layouts = [generate(seed, tables, spec) for seed in split_seeds(args.split, args.count, args.first)]
    derived = [derive(layout, tables) for layout in layouts] if args.derived else None
    print(f'[world] wrote {save_pool(args.output, layouts, derived)}: {len(layouts)} {args.split} layouts')


if __name__ == '__main__':
    main()
