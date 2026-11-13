"""The native path finder (sim/native/paths.c) answers as the oracle: sim/paths.py, the World methods
around it (sim/engine.py:843-1014) and sim/_paths.c, the existing reference port.

The C functions are reached through a small test library: SHIM below includes paths.c (its static searches, reach and
sight too) and is built with the native build's flags and generated header next to pack.c and rng.c, then loaded with
ctypes. Every check calls the Python functions themselves:
  - random collision grids (Hypothesis, as sim/tests/test_paths.py): step masks, the queue-order tree, the bitset depths,
    end tiles and waypoints of growing trees, reach for every shape, line of sight, the client's route (act-v3's
    clicks and ACT), collision masks;
  - real layouts (the archive's meta, the compact pool's derived tables, the supply pool): the stored trees and snapshot
    rows of every source, collision changes with their changed tiles and the stored-or-live queries after them;
  - a replay of every path query a short CpuBatch goal-activity-v5 run makes, recorded by wrapping the oracle's
    functions (the oracle is not edited), answered by C worlds in the recorded collision state.
The private files are read where they exist and the tests that need them skip otherwise; nothing private is printed
beyond counts."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sysconfig

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from sim import native, paths
from sim.engine import Loc, Unsupported, World

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'logs/sim/data/draynor-nc1.npz'
COMPACT = ROOT / 'logs/sim/world/train-8-nc1-derived.npz'
SUPPLY = ROOT / 'logs/supply/pools/train-128.npz'
REF = paths.IMPLS['python']
ORIGIN = (2880, 3328)
VIAS = ('target', 'reach', 'fallback')  # rs.h RS_VIA_* order

SHIM = r'''
/* test access to sim/native/paths.c (sim/tests/test_native_paths.py) */
#include <stdlib.h>
#include "paths.c"

RsLayout *t_layout(int32_t W, int32_t H, int32_t x0, int32_t z0, const int32_t *flags_boot, const uint8_t *steps_boot,
                   const uint8_t *steps_snap, const uint16_t *entity_reach, const int16_t *tile_src, int32_t n_src,
                   const int16_t *src_tile, int32_t n_locs, const int16_t *loc_reach, const uint16_t *order0,
                   const uint8_t *dist0, const uint8_t *parent0, const uint8_t *snap_dist) {
    RsLayout *L = calloc(1, sizeof *L);
    L->W = W, L->H = H, L->T = W * H, L->x0 = x0, L->z0 = z0, L->n_src = n_src, L->n_locs = n_locs;
    L->flags_boot = flags_boot, L->steps_boot = steps_boot, L->steps_snap = steps_snap, L->entity_reach = entity_reach;
    L->tile_src = tile_src, L->src_tile = src_tile, L->loc_reach = loc_reach;
    L->order0 = order0, L->dist0 = dist0, L->parent0 = parent0, L->snap_dist = snap_dist;
    if (!snap_dist) rs_snap_reach(L);   /* as pack.c resolves a layout without stored walk rows */
    return L;
}

RsPack *t_pack(int32_t n, const RsLocType *types) {
    RsPack *P = calloc(1, sizeof *P);
    P->n_loc_types = n, P->loc_types = types;
    return P;
}

World *t_world(const RsPack *P, const RsLayout *L, int32_t n_stored) {
    World *w = calloc(1, sizeof *w);
    w->P = P, w->L = L, w->n_stored_locs = n_stored;
    memcpy(w->flags, L->flags_boot, (size_t)L->T * sizeof(int32_t));
    rs_paths_reset(w);
    return w;
}

void t_free(void *p) { free(p); }

int32_t t_const(int32_t k) { return k == 0 ? RS_WALK_R : k == 1 ? RS_LOS_R : -1; }

/* a recorded collision state: the live flags, the changed set against the boot flags, a new version if they changed */
void t_set_flags(World *w, const int32_t *flags) {
    const RsLayout *L = w->L;
    int any = 0;
    memset(w->changed, 0, sizeof w->changed);
    w->n_changed = 0;
    for (int32_t t = 0; t < L->T; t++) {
        any |= w->flags[t] != flags[t];
        w->flags[t] = flags[t];
        if (flags[t] != L->flags_boot[t]) w->changed[t >> 6] |= 1ull << (t & 63), w->n_changed++;
    }
    if (any && ++w->coll_ver == 0) w->coll_ver = 1;
}

int32_t t_state(const World *w, int32_t *flags, uint64_t *changed) {
    memcpy(flags, w->flags, (size_t)w->L->T * sizeof(int32_t));
    memcpy(changed, w->changed, sizeof w->changed);
    return w->n_changed;
}

void t_err(const World *w, int32_t *out) {
    out[0] = w->err.code, out[1] = w->err.exc, out[2] = w->err.reason;
    for (int i = 0; i < RS_ERR_ARGS; i++) out[3 + i] = w->err.arg[i];
}

Loc *t_loc(World *w, int32_t slot, int32_t index, int32_t base, int32_t type, int32_t x, int32_t z, int32_t shape,
           int32_t angle, int32_t width, int32_t length, int32_t base_shape, int32_t base_angle, int32_t level) {
    Loc *l = &w->locs[slot];
    memset(l, 0, sizeof *l);
    l->index = index, l->base = (int16_t)base, l->type = (int16_t)type, l->x = (int16_t)x, l->z = (int16_t)z;
    l->shape = (uint8_t)shape, l->angle = (uint8_t)angle, l->width = (uint8_t)width, l->length = (uint8_t)length;
    l->base_shape = (uint8_t)base_shape, l->base_angle = (uint8_t)base_angle, l->level = (uint8_t)level;
    l->serial = 1, l->active = 1;
    return l;
}

int t_route(World *w, int32_t sx, int32_t sz, const int32_t *xz, int32_t nr, int32_t dx, int32_t dz, int32_t rw,
            int32_t rh, int16_t *out, int32_t *n) {
    View v;
    RS_TRY(tree_view(w, sx, sz, &v));
    int32_t reach[RS_MAX_T];
    int m = 0, got;
    for (int32_t k = 0; k < nr; k++) {
        int32_t t = rs_tile(w->L, 0, xz[2 * k], xz[2 * k + 1]);
        if (t >= 0) reach[m++] = t;
    }
    route(w, &v, sx, sz, reach, m, dx, dz, rw, rh, (int16_t (*)[2])out, &got);
    *n = got;
    return 0;
}

/* Agent.route from (sx, sz) with the build area's origin (ox, oz): to a tile (arrival and loc NULL), by an explicit
   arrival (width, length, angle, loc_shape, forceapproach) or by a loc's (rs_loc_arrival) */
int t_client_route(World *w, int32_t sx, int32_t sz, int32_t ox, int32_t oz, int32_t x, int32_t z, const int16_t *arrival,
                   const Loc *loc, int32_t nearest, int16_t *out, int32_t *n, uint8_t *via, int32_t *found) {
    w->p.x = (int16_t)sx, w->p.z = (int16_t)sz, w->p.origin_x = (int16_t)ox, w->p.origin_z = (int16_t)oz;
    RsArrival a = {0};
    if (arrival) a = (RsArrival){arrival[0], arrival[1], arrival[2], arrival[3], arrival[4]};
    if (loc) a = rs_loc_arrival(w->P, loc);
    int got = 0, f = 0;
    int rc = rs_client_route(w, x, z, arrival || loc ? &a : NULL, nearest, (int16_t (*)[2])out, &got, via, &f);
    *n = got, *found = f;
    return rc;
}

void t_step_masks(const int32_t *flags, int32_t W, int32_t H, uint8_t *out) { step_masks(flags, W, H, out); }

void t_tree(const uint8_t *steps, int32_t W, int32_t H, int32_t s, uint16_t *order, uint8_t *dist, uint8_t *parent) {
    static RsTree tr;
    RsLayout L;
    memset(&L, 0, sizeof L);
    L.W = W, L.H = H, L.T = W * H;
    tree_start(&tr, &L, 0, s);
    tree_grow(&tr, steps, H, NULL);
    memcpy(order, tr.order, (size_t)L.T * 2), memcpy(dist, tr.dist, (size_t)L.T), memcpy(parent, tr.parent, (size_t)L.T);
}

void t_bitset_dist(const uint8_t *steps, int32_t W, int32_t H, int32_t s, uint8_t *dist) {
    static uint64_t M[RS_MAX_W + 2][8];
    static RsLayout L;
    RsSnapGrow g;
    memset(&L, 0, sizeof L);
    L.W = W, L.H = H, L.T = W * H;
    column_masks(steps, W, H, M);
    grow_start(&g, &L, s, dist);
    while (!g.done) grow_level(&g, (const uint64_t (*)[8])M, W, H, dist);
}

/* A runtime walk row of source s on a layout whose reach comes from representative tile r, read at `tiles` in order
   through walk_read (its search grown only as the reads need): -1 unreached. */
void t_walk_lazy(const uint8_t *steps, int32_t W, int32_t H, int32_t r, int32_t s, const int32_t *tiles, int32_t n,
                 int32_t *out) {
    static RsLayout L;
    RsPathCache *pc = calloc(1, sizeof *pc);
    int16_t start = (int16_t)r;
    memset(&L, 0, sizeof L);
    L.W = W, L.H = H, L.T = W * H, L.steps_snap = steps, L.n_start = 1, L.start_tile = &start;
    rs_snap_reach(&L);
    column_masks(steps, W, H, pc->snap_masks);
    pc->mask_L = &L;
    grow_start(&pc->snap_grow[0], &L, s, pc->snap_dist[0]);
    pc->walk_row = pc->snap_dist[0], pc->walk_grow = &pc->snap_grow[0], pc->walk_ox = s / H, pc->walk_oz = s % H;
    for (int32_t i = 0; i < n; i++) out[i] = walk_read(pc, &L, tiles[i] / H - s / H, tiles[i] % H - s % H);
    free(pc);
}

int t_reached(const int32_t *flags, int32_t W, int32_t H, int32_t x0, int32_t z0, int32_t sx, int32_t sz, int32_t dx,
              int32_t dz, int32_t width, int32_t length, int32_t angle, int32_t shape, int32_t access) {
    Grid g = {flags, x0, z0, W, H};
    return reached(&g, sx, sz, dx, dz, width, length, angle, shape, access);
}

int t_loc_reaching(const int32_t *flags, int32_t W, int32_t H, int32_t x0, int32_t z0, int32_t x, int32_t z,
                   int32_t width, int32_t length, int32_t angle, int32_t shape, int32_t access, int16_t *out, int32_t cap) {
    Grid g = {flags, x0, z0, W, H};
    int n;
    return loc_reaching_live(&g, x, z, width, length, angle, shape, access, (int16_t (*)[2])out, &n, cap) ? -1 : n;
}

int t_entity_reaching(const int32_t *flags, int32_t W, int32_t H, int32_t x0, int32_t z0, int32_t x, int32_t z) {
    Grid g = {flags, x0, z0, W, H};
    return entity_reaching_live(&g, x, z);
}

int t_los(const int32_t *flags, int32_t W, int32_t H, int32_t x0, int32_t z0, int32_t sx, int32_t sz, int32_t ex,
          int32_t ez) {
    Grid g = {flags, x0, z0, W, H};
    return line_of_sight(&g, sx, sz, ex, ez);
}

/* the stored tables of sim/world.py live_tables, from the live functions: sight (21 x 21 offsets) and entity reach */
void t_los_table(const int32_t *flags, int32_t W, int32_t H, uint8_t *out) {
    Grid g = {flags, 0, 0, W, H};
    for (int32_t x = 0; x < W; x++)
        for (int32_t z = 0; z < H; z++)
            for (int32_t dx = -RS_LOS_R; dx <= RS_LOS_R; dx++)
                for (int32_t dz = -RS_LOS_R; dz <= RS_LOS_R; dz++)
                    *out++ = (uint8_t)line_of_sight(&g, x, z, x + dx, z + dz);
}

void t_entity_table(const int32_t *flags, int32_t W, int32_t H, uint16_t *out) {
    Grid g = {flags, 0, 0, W, H};
    for (int32_t x = 0; x < W; x++)
        for (int32_t z = 0; z < H; z++) *out++ = entity_reaching_live(&g, x, z);
}

int t_collision_masks(const RsPack *P, int32_t x, int32_t z, int32_t shape, int32_t angle, int32_t type, int32_t *out) {
    int32_t mx[16], mz[16], mask[16];
    int n = rs_collision_masks(P, x, z, shape, angle, type, mx, mz, mask);
    for (int k = 0; k < n; k++) out[3 * k] = mx[k], out[3 * k + 1] = mz[k], out[3 * k + 2] = mask[k];
    return n;
}
'''

P_, I32 = ctypes.c_void_p, ctypes.c_int32
SIGNATURES = dict(
    t_layout=(P_, [I32] * 4 + [P_] * 5 + [I32, P_, I32] + [P_] * 5), t_pack=(P_, [I32, P_]), t_world=(P_, [P_, P_, I32]),
    t_free=(None, [P_]), t_const=(I32, [I32]), t_set_flags=(None, [P_, P_]), t_state=(I32, [P_, P_, P_]),
    t_err=(None, [P_, P_]), t_loc=(P_, [P_] + [I32] * 13), t_route=(ctypes.c_int, [P_, I32, I32, P_, I32] + [I32] * 4 + [P_, P_]),
    t_step_masks=(None, [P_, I32, I32, P_]), t_tree=(None, [P_, I32, I32, I32, P_, P_, P_]),
    t_bitset_dist=(None, [P_, I32, I32, I32, P_]), t_walk_lazy=(None, [P_, I32, I32, I32, I32, P_, I32, P_]),
    t_reached=(ctypes.c_int, [P_] + [I32] * 13),
    t_loc_reaching=(ctypes.c_int, [P_] + [I32] * 11 + [P_, I32]), t_entity_reaching=(ctypes.c_int, [P_] + [I32] * 6),
    t_los=(ctypes.c_int, [P_] + [I32] * 8), t_los_table=(None, [P_, I32, I32, P_]),
    t_entity_table=(None, [P_, I32, I32, P_]), t_collision_masks=(ctypes.c_int, [P_] + [I32] * 5 + [P_]),
    rs_path_to_tile=(ctypes.c_int, [P_, I32, I32, I32, I32, P_, P_]),
    t_client_route=(ctypes.c_int, [P_] + [I32] * 6 + [P_, P_, I32, P_, P_, P_, P_]),
    rs_entity_reaching=(ctypes.c_int, [P_, I32, I32, P_]),
    rs_loc_reaching=(ctypes.c_int, [P_, P_, P_, P_, ctypes.c_int]),
    rs_line_of_sight=(ctypes.c_int, [P_, I32, I32, I32, I32, P_]),
    rs_walk_steps=(ctypes.c_int, [P_, I32, I32, I32, I32, I32, P_]),
    rs_walk_approach=(ctypes.c_int, [P_, I32, I32, I32, I32, I32, P_]),
    rs_loc_collision=(ctypes.c_int, [P_, P_, ctypes.c_int]),
    rs_changed_in=(ctypes.c_int, [P_, I32, I32, I32, I32, I32]), rs_window_changed=(ctypes.c_int, [P_, I32, I32, I32]),
)


def ptr(a):
    return None if a is None else a.ctypes.data


@pytest.fixture(scope='module')
def lib(tmp_path_factory):
    """The shim with paths.c, pack.c (errors) and rng.c, built as the extension is (native.FLAGS, rs_gen.h)."""
    header = native.generated_header()
    digest = hashlib.sha256(SHIM.encode() + header.encode() + ' '.join(native.FLAGS).encode())
    for name in ('paths.c', 'pack.c', 'rng.c', 'rs.h'):
        digest.update((native.SOURCE / name).read_bytes())
    out = native.build_dir() / f'_native_paths_test-{digest.hexdigest()[:16]}.so'
    if not out.exists():
        work = tmp_path_factory.mktemp('shim')
        (work / 'rs_gen.h').write_text(header)
        (work / 'shim.c').write_text(SHIM)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_name(f'{out.name}.{os.getpid()}.tmp')
        cc = (sysconfig.get_config_var('CC') or 'cc').split()
        command = [*cc, *native.FLAGS, '-shared', '-I', str(work), '-I', str(native.SOURCE), str(work / 'shim.c'),
                   str(native.SOURCE / 'pack.c'), str(native.SOURCE / 'rng.c'), '-o', str(temp), '-lz']
        done = subprocess.run(command, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-3000:]
        os.replace(temp, out)
    so = ctypes.CDLL(str(out))
    for name, (restype, argtypes) in SIGNATURES.items():
        fn = getattr(so, name)
        fn.restype, fn.argtypes = restype, argtypes
    return so


@pytest.fixture(scope='module')
def reasons():
    module = native.load()
    return module.reasons(), module.exceptions()


# ---- outcomes in one form for both sides ---------------------------------------------------------------------------
def py_outcome(fn, *args):
    """('ok', value) or (exception class, message) of an oracle call."""
    try:
        return 'ok', fn(*args)
    except paths.OutsideTables as error:
        return 'EXC_OUTSIDE', str(error)
    except KeyError as error:
        return 'EXC_KEYERROR', str(error.args[0])
    except Unsupported as error:
        return 'EXC_UNSUPPORTED', str(error)


class CWorld:
    """A C World on a layout, with the arrays it points at kept alive."""

    def __init__(self, lib, reasons, P, layout, n_stored=0):
        self.lib, self.reasons, self.layout = lib, reasons, layout
        self.w = lib.t_world(P, layout.ptr, n_stored)
        self.out, self.n = np.zeros(2 * 500, np.int16), np.zeros(1, np.int32)
        self.flags = None  # the last set_flags array, which the C world copied

    def __del__(self):
        self.lib.t_free(self.w)

    def error(self):
        e = np.zeros(9, np.int32)
        self.lib.t_err(self.w, ptr(e))
        rows, exceptions = self.reasons
        _, exc, fmt, kinds = rows[e[2]]
        return exceptions[e[1]], fmt.format(*[int(v) for v in e[3:3 + len(kinds)]])

    def call(self, fn, *args, result=None):
        rc = fn(self.w, *args)
        return self.error() if rc else ('ok', result() if result else None)

    def waypoints(self):
        return [(int(self.out[2 * i]), int(self.out[2 * i + 1])) for i in range(int(self.n[0]))]

    def path_to_tile(self, sx, sz, x, z):
        return self.call(self.lib.rs_path_to_tile, sx, sz, x, z, ptr(self.out), ptr(self.n), result=self.waypoints)

    def client_route(self, sx, sz, origin, dest, arrival=None, loc=None, nearest=False):
        """Agent.route as (turn points, via) or None; arrival (width, length, angle, loc_shape, forceapproach) or loc
        (t_loc fields of a loc whose arrival rs_loc_arrival makes)."""
        a = None if arrival is None else np.asarray(arrival, np.int16)
        loc = None if loc is None else self.loc(loc)
        via, found = np.zeros(1, np.uint8), np.zeros(1, np.int32)
        return self.call(self.lib.t_client_route, sx, sz, *origin, *dest, ptr(a), loc, int(nearest), ptr(self.out),
                         ptr(self.n), ptr(via), ptr(found), result=lambda: (self.waypoints(), VIAS[int(via[0])])
                         if found[0] else None)

    def loc(self, fields, slot=63):
        return self.lib.t_loc(self.w, slot, *fields)

    def route(self, sx, sz, reaching, dest, rect):
        xz = np.asarray(reaching, np.int32).reshape(-1)
        return self.call(self.lib.t_route, sx, sz, ptr(xz), len(reaching), dest[0], dest[1], rect[0], rect[1],
                         ptr(self.out), ptr(self.n), result=self.waypoints)

    def entity_reaching(self, x, z):
        mask = np.zeros(1, np.uint16)
        return self.call(self.lib.rs_entity_reaching, x, z, ptr(mask), result=lambda: [
            (x + dx, z + dz) for b, (dx, dz) in enumerate(paths.NEIGHBOURS) if int(mask[0]) >> b & 1])

    def loc_reaching(self, fields):
        loc = self.loc(fields)
        return self.call(self.lib.rs_loc_reaching, loc, ptr(self.out), ptr(self.n), 500, result=self.waypoints)

    def line_of_sight(self, ax, az, bx, bz):
        v = np.zeros(1, np.int32)
        return self.call(self.lib.rs_line_of_sight, ax, az, bx, bz, ptr(v), result=lambda: bool(v[0]))

    def walk_steps(self, src, sx, sz, x, z):
        v = np.zeros(1, np.int32)
        return self.call(self.lib.rs_walk_steps, src, sx, sz, x, z, ptr(v), result=lambda: int(v[0]))

    def walk_approach(self, src, sx, sz, x, z):
        v = np.zeros(1, np.int32)
        return self.call(self.lib.rs_walk_approach, src, sx, sz, x, z, ptr(v), result=lambda: int(v[0]))

    def window_changed(self, sx, sz):
        return bool(self.lib.rs_window_changed(self.w, 0, sx, sz))

    def changed_in(self, level, x0, x1, z0, z1):
        return bool(self.lib.rs_changed_in(self.w, level, x0, x1, z0, z1))

    def loc_collision(self, fields, add):
        return self.call(self.lib.rs_loc_collision, self.loc(fields, 62), int(add))

    def set_flags(self, flags):
        self.flags = np.ascontiguousarray(flags, np.int32)
        self.lib.t_set_flags(self.w, ptr(self.flags))

    def state(self):
        flags, changed = np.zeros(self.layout.T, np.int32), np.zeros(64, np.uint64)
        n = self.lib.t_state(self.w, ptr(flags), ptr(changed))
        bits = np.unpackbits(changed.view(np.uint8), bitorder='little')[:self.layout.T]
        return flags, set(np.flatnonzero(bits).tolist()), n


class CLayout:
    """An RsLayout over numpy arrays (layout rows x-major, tiles x * H + z)."""

    def __init__(self, lib, W, H, x0, z0, flags, steps_boot, steps_snap, walkable=None, entity_reach=None,
                 loc_reach=None, order0=None, dist0=None, parent0=None, snap_dist=None):
        self.lib, self.W, self.H, self.T, self.x0, self.z0 = lib, W, H, W * H, x0, z0
        c = lambda a, dt: None if a is None else np.ascontiguousarray(np.asarray(a).reshape(-1), dt)
        self.flags, self.steps_boot, self.steps_snap = c(flags, np.int32), c(steps_boot, np.uint8), c(steps_snap, np.uint8)
        walkable = np.ones((W, H), bool) if walkable is None else walkable
        tiles = [x * H + z for x, z in np.argwhere(walkable)]
        self.src_tile = np.asarray(tiles, np.int16)
        self.tile_src = np.full(self.T, -1, np.int16)
        self.tile_src[self.src_tile] = np.arange(len(tiles), dtype=np.int16)
        self.entity_reach, self.loc_reach = c(entity_reach, np.uint16), c(loc_reach, np.int16)
        self.n_locs = 0 if loc_reach is None else len(loc_reach)
        self.order0, self.dist0, self.parent0 = c(order0, np.uint16), c(dist0, np.uint8), c(parent0, np.uint8)
        self.snap_dist = c(snap_dist, np.uint8)
        self.ptr = lib.t_layout(W, H, x0, z0, ptr(self.flags), ptr(self.steps_boot), ptr(self.steps_snap),
                                ptr(self.entity_reach), ptr(self.tile_src), len(tiles), ptr(self.src_tile), self.n_locs,
                                ptr(self.loc_reach), ptr(self.order0), ptr(self.dist0), ptr(self.parent0),
                                ptr(self.snap_dist))

    def __del__(self):
        self.lib.t_free(self.ptr)


def loc_types_pack(lib, types):
    """t_pack over RsLocType records of {id: loc type dict} (the fields paths.c reads)."""
    dt = native.record_dtype('RsLocType')
    rows = np.zeros(max(types) + 1, dt)
    for i, t in types.items():
        rows['present'][i] = 1
        for name in ('blockwalk', 'blockrange', 'active', 'width', 'length', 'forceapproach'):
            rows[name][i] = int(t[name])
    return rows, lib.t_pack(len(rows), ptr(rows))


# ---- random collision grids ----------------------------------------------------------------------------------------
FLAG_CHOICES = (0, paths.LOC, paths.LOC | paths.LOC_PROJ, paths.WALL_NORTH, paths.WALL_EAST, paths.WALL_SOUTH,
                paths.WALL_WEST, paths.WALL_NORTH | paths.WALL_EAST, paths.WALL_SOUTH | paths.WALL_WEST | paths.PROJ_SOUTH,
                paths.PROJ_NORTH, paths.PROJ_EAST, paths.PROJ_WEST, 0x1, 0x4, 0x10, 0x40, 0x200000, 0x40000,
                paths.WALL_EAST | paths.PROJ_EAST)


def random_grid(seed, blocked, small=False):
    """A collision grid of random wall, loc, projectile and floor flags (some tiles any 31-bit value)."""
    rng = np.random.default_rng(seed)
    W, H = (int(rng.integers(1, 17)), int(rng.integers(1, 17))) if small else (int(rng.integers(1, 65)), int(rng.integers(1, 65)))
    flags = np.where(rng.random((W, H)) < blocked, rng.choice(FLAG_CHOICES, (W, H)), 0)
    wild = rng.random((W, H)) < 0.02
    flags = np.where(wild, rng.integers(0, 0x7fffffff, (W, H)), flags).astype(np.int32)
    return rng, flags


def grid_fn(flags, x0, z0):
    """sim/world.py _flags_fn at an origin: FLAG_NULL off the grid."""
    W, H = flags.shape
    return lambda x, z: int(flags[x - x0, z - z0]) if 0 <= x - x0 < W and 0 <= z - z0 < H else 0x7fffffff


def ref_rows(search, steps, x, z):
    """sim/world.py tree_rows with a given search: order (65535 none), dist and parent as layout rows."""
    tree, (W, H) = search(steps, 0, 0, x, z, bounded=True), steps.shape
    ox, oz = paths.HALF - x, paths.HALF - z
    order = tree.orders[ox:ox + W, oz:oz + H]
    return (np.where(order < 0, 65535, order).astype(np.uint16), tree.dists[ox:ox + W, oz:oz + H].astype(np.uint8),
            tree.parents[ox:ox + W, oz:oz + H].astype(np.uint8))


def c_rows(lib, steps, x, z):
    W, H = steps.shape
    s = np.ascontiguousarray(steps, np.uint8)
    order, dist, parent = np.zeros(W * H, np.uint16), np.zeros(W * H, np.uint8), np.zeros(W * H, np.uint8)
    lib.t_tree(ptr(s), W, H, x * H + z, ptr(order), ptr(dist), ptr(parent))
    return order.reshape(W, H), dist.reshape(W, H), parent.reshape(W, H)


def c_bitset(lib, steps, x, z):
    W, H = steps.shape
    s, dist = np.ascontiguousarray(steps, np.uint8), np.zeros(W * H, np.uint8)
    lib.t_bitset_dist(ptr(s), W, H, x * H + z, ptr(dist))
    return dist.reshape(W, H)


def test_the_radii_are_layout_tables_own(lib):
    """rs.h RS_WALK_R and RS_LOS_R are LayoutTables' meta walk_r and los_r (sim/world.py)."""
    from sim import world
    import inspect
    source = inspect.getsource(world.LayoutTables.__init__)
    assert 'walk_r=48' in source and 'los_r=LOS_R' in source
    assert (lib.t_const(0), lib.t_const(1)) == (48, world.LOS_R)


@settings(deadline=None, max_examples=200)
@given(st.integers(0, 2 ** 32 - 1), st.floats(0, 0.8))
def test_step_masks_and_trees_are_the_reference_search(lib, seed, blocked):
    """step_masks, the queue-order tree (order, dist, parent of every tile) and the bitset depths from random tiles
    equal paths.step_masks and paths.search (the Python reference and sim/_paths.c)."""
    rng, flags = random_grid(seed, blocked)
    W, H = flags.shape
    steps = paths.step_masks(flags)
    ours = np.zeros(W * H, np.uint8)
    lib.t_step_masks(ptr(np.ascontiguousarray(flags)), W, H, ptr(ours))
    assert np.array_equal(ours.reshape(W, H), steps)
    searches = [REF['search']] + ([paths._c_search] if paths._C is not None else [])
    for _ in range(3):
        x, z = int(rng.integers(W)), int(rng.integers(H))
        mine = c_rows(lib, steps, x, z)
        for search in searches:
            for a, b in zip(mine, ref_rows(search, steps, x, z)):
                assert np.array_equal(a, b)
        assert np.array_equal(c_bitset(lib, steps, x, z), mine[1])


@settings(deadline=None, max_examples=200)
@given(st.integers(0, 2 ** 32 - 1), st.floats(0, 0.8))
def test_walk_rows_searched_as_read_answer_as_the_whole_search(lib, seed, blocked):
    """A runtime snapshot row whose search runs only as far as its reads need, with the layout's reach (from a random
    representative tile) answering tiles outside it at once for a source in its strongly connected set: reads in a
    random order from random sources (the representative itself among them) equal the whole reference search's depths,
    unreached tiles included."""
    rng, flags = random_grid(seed, blocked)
    W, H = flags.shape
    steps = np.ascontiguousarray(paths.step_masks(flags), np.uint8)
    for k in range(4):
        r = int(rng.integers(W * H))
        s = r if k == 0 else int(rng.integers(W * H))
        full = ref_rows(REF['search'], steps, s // H, s % H)[1].reshape(-1).astype(np.int32)
        tiles = rng.permutation(W * H)[:int(rng.integers(1, W * H + 1))].astype(np.int32)
        out = np.zeros(len(tiles), np.int32)
        lib.t_walk_lazy(ptr(steps), W, H, r, s, ptr(tiles), len(tiles), ptr(out))
        assert np.array_equal(out, np.where(full[tiles] == 255, -1, full[tiles]))


@settings(deadline=None, max_examples=120)
@given(st.integers(0, 2 ** 32 - 1), st.floats(0, 0.7))
def test_end_tiles_and_waypoints_of_growing_trees_answer_as_the_reference(lib, reasons, seed, blocked):
    """World.tree's stored search (a source, nothing changed) queried for random reaching tiles, destinations and
    rectangles, from a few sources in turn: the C tree grows only to the first reach tile and resumes across queries,
    and every answer equals paths.end_tile + paths.backtrack on the whole reference tree. A tile that is no source is
    refused as paths.source_index refuses it."""
    rng, flags = random_grid(seed, blocked)
    W, H = flags.shape
    x0, z0 = ORIGIN
    steps = paths.step_masks(flags)
    walkable = rng.random((W, H)) < 0.9
    layout = CLayout(lib, W, H, x0, z0, flags, steps, steps, walkable=walkable)
    world = CWorld(lib, reasons, None, layout)
    sources = [tuple(int(v) for v in s) for s in np.argwhere(walkable)]
    starts = [sources[int(rng.integers(len(sources)))] for _ in range(3)] if sources else []
    trees = {s: REF['search'](steps, x0, z0, s[0] + x0, s[1] + z0, True) for s in starts}
    for _ in range(25):
        if not starts:
            break
        s = starts[int(rng.integers(len(starts)))]
        sx, sz, tree = s[0] + x0, s[1] + z0, trees[s]
        dest = (sx + int(rng.integers(-40, 41)), sz + int(rng.integers(-40, 41)))
        rect = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        reaching = [(dest[0] + int(rng.integers(-2, 3)), dest[1] + int(rng.integers(-2, 3)))
                    for _ in range(int(rng.integers(0, 7)))]
        expected = REF['backtrack'](tree, sx, sz, REF['end_tile'](tree, sx, sz, reaching, dest, rect))
        assert world.route(sx, sz, reaching, dest, rect) == ('ok', expected)
        if rng.random() < 0.3:
            assert world.path_to_tile(sx, sz, *dest) == ('ok', paths.path_to_tile(tree, sx, sz, *dest))
    blocked_tiles = np.argwhere(~walkable)
    if len(blocked_tiles):
        x, z = (int(v) for v in blocked_tiles[0])
        assert world.path_to_tile(x + x0, z + z0, x0, z0) == ('EXC_OUTSIDE', f'no path table for 0,{x + x0},{z + z0}')
    assert world.path_to_tile(x0 - 1, z0, x0, z0) == ('EXC_OUTSIDE', f'no path table for 0,{x0 - 1},{z0}')


SHAPES = (-3, -2, -1, *range(0, 24))


@settings(deadline=None, max_examples=150)
@given(st.integers(0, 2 ** 32 - 1), st.floats(0, 0.9), st.sampled_from(SHAPES), st.integers(0, 3), st.integers(1, 4),
       st.integers(1, 4), st.integers(0, 15))
def test_reach_answers_as_the_reference_for_every_shape(lib, seed, blocked, shape, angle, width, length, access):
    """paths.reached over the whole reach box of a destination, loc_reaching and entity_reaching_live, on random flags
    (off-grid tiles FLAG_NULL)."""
    rng, flags = random_grid(seed, blocked, small=True)
    W, H = flags.shape
    x0, z0 = int(rng.integers(-20, 3000)), int(rng.integers(-20, 3000))
    f, c = grid_fn(flags, x0, z0), np.ascontiguousarray(flags)
    dx, dz = x0 + int(rng.integers(-2, W + 2)), z0 + int(rng.integers(-2, H + 2))
    for sx in range(dx - 6, dx + width + 7):
        for sz in range(dz - 6, dz + length + 7):
            ours = lib.t_reached(ptr(c), W, H, x0, z0, sx, sz, dx, dz, width, length, angle, shape, access)
            assert bool(ours) == paths.reached(f, sx, sz, dx, dz, width, length, angle, shape, access), (sx, sz)
    out = np.zeros(2 * 512, np.int16)
    n = lib.t_loc_reaching(ptr(c), W, H, x0, z0, dx, dz, width, length, angle, shape, access, ptr(out), 512)
    assert [(int(out[2 * i]), int(out[2 * i + 1])) for i in range(n)] == \
        paths.loc_reaching(f, dx, dz, width, length, angle, shape, access)
    mask = lib.t_entity_reaching(ptr(c), W, H, x0, z0, dx, dz)
    assert [(dx + a, dz + b) for k, (a, b) in enumerate(paths.NEIGHBOURS) if mask >> k & 1] == \
        paths.entity_reaching_live(f, dx, dz)


@settings(deadline=None, max_examples=150)
@given(st.integers(0, 2 ** 32 - 1), st.floats(0, 0.6))
def test_line_of_sight_answers_as_the_reference(lib, seed, blocked):
    """LineValidator's ray between random tiles on and around the grid, at origins that make the ray's scaled
    coordinates negative (Python's floor shift) as well as at game coordinates."""
    rng, flags = random_grid(seed, blocked)
    W, H = flags.shape
    x0, z0 = (int(rng.integers(-80, 10)), int(rng.integers(-80, 10))) if rng.random() < 0.5 else ORIGIN
    f, c = grid_fn(flags, x0, z0), np.ascontiguousarray(flags)
    for _ in range(40):
        sx, sz, ex, ez = (x0 + int(rng.integers(-3, W + 3)), z0 + int(rng.integers(-3, H + 3)),
                          x0 + int(rng.integers(-3, W + 3)), z0 + int(rng.integers(-3, H + 3)))
        assert bool(lib.t_los(ptr(c), W, H, x0, z0, sx, sz, ex, ez)) == paths.line_of_sight(f, sx, sz, ex, ez)


def py_client_route(flags, x0, z0, sx, sz, origin, dest, arrival, nearest):
    """paths.client_route over paths.client_search (bounded) as (turn points, via) or None; an answer outside the
    tables is Unsupported, as World.path makes it."""
    cflags = paths.client_flags(flags)
    try:
        tree = paths.client_search(paths.client_steps(cflags), cflags, x0, z0, sx, sz, paths.scene_base(*origin), True)
    except paths.OutsideTables as error:
        raise Unsupported(str(error))
    route = paths.client_route(tree, sx, sz, dest, arrival, nearest)
    return None if route is None else (list(route), route.via)


@settings(deadline=None, max_examples=150)
@given(st.integers(0, 2 ** 32 - 1), st.floats(0, 0.7))
def test_the_client_route_answers_as_the_reference(lib, reasons, seed, blocked):
    """Agent.route on random bounded grids: to tiles on and off the layout with and without the 3 x 3 fallback, to npcs
    (1 x 1) and to locs of every shape (rs_loc_arrival's rotated size and forceapproach, or the wall shape), from
    build-area origins that leave part of the layout outside the scene or the source outside it."""
    rng, flags = random_grid(seed, blocked)
    W, H = flags.shape
    x0, z0 = ORIGIN
    steps = paths.step_masks(flags)
    layout = CLayout(lib, W, H, x0, z0, flags, steps, steps)
    width, length, fa = int(rng.integers(1, 4)), int(rng.integers(1, 4)), int(rng.integers(16))
    rows, P = loc_types_pack(lib, {1: dict(blockwalk=1, blockrange=1, active=1, width=width, length=length,
                                           forceapproach=fa)})
    world = CWorld(lib, reasons, P, layout)
    for _ in range(8):
        sx, sz = x0 + int(rng.integers(W)), z0 + int(rng.integers(H))
        origin = (sx + int(rng.integers(-60, 61)), sz + int(rng.integers(-60, 61)))
        x, z = x0 + int(rng.integers(-3, W + 3)), z0 + int(rng.integers(-3, H + 3))
        kind, nearest = int(rng.integers(3)), bool(rng.random() < 0.5)
        if kind == 0:
            got = world.client_route(sx, sz, origin, (x, z), nearest=nearest)
            arrival = None
        elif kind == 1:
            got = world.client_route(sx, sz, origin, (x, z), arrival=(1, 1, 0, 0, 0))
            arrival, nearest = paths.ENTITY_ARRIVAL, False
        else:
            shape, angle = int(rng.choice([0, 1, 2, 3, 9, 4, 5, 6, 7, 8, 10, 11, 22])), int(rng.integers(4))
            got = world.client_route(sx, sz, origin, (x, z), loc=(0, 0, 1, x, z, shape, angle, width, length, shape,
                                                                    angle, 0))
            arrival, nearest = paths.loc_arrival(shape, angle, width, length, fa), False
        assert got == py_outcome(py_client_route, flags, x0, z0, sx, sz, origin, (x, z), arrival, nearest)
    del rows


def test_collision_masks_answer_as_the_reference(lib):
    """paths.collision_masks for every shape and angle of synthetic loc types (blockwalk, blockrange, active, sizes)."""
    types, k = {}, 0
    for blockwalk in (0, 1):
        for blockrange in (0, 1):
            for active in (0, 1):
                for width in range(1, 5):
                    for length in range(1, 5):
                        types[k] = dict(blockwalk=blockwalk, blockrange=blockrange, active=active, width=width,
                                        length=length, forceapproach=0)
                        k += 1
    rows, P = loc_types_pack(lib, types)
    out = np.zeros(48, np.int32)
    for type_id, t in types.items():
        for shape in range(23):
            for angle in range(4):
                n = lib.t_collision_masks(P, 3000, 3300, shape, angle, type_id, ptr(out))
                assert [tuple(int(v) for v in out[3 * i:3 * i + 3]) for i in range(n)] == \
                    paths.collision_masks(3000, 3300, shape, angle, t)
    lib.t_free(P)


# ---- real layouts ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def base():
    """sim.data.Tables over the archive's meta only: LayoutTables, World and CpuBatch read meta, never the Draynor
    arrays, so the 400 MB of them stay unloaded (the archive's pin check fails without vendor/rs-sdk)."""
    if not ARCHIVE.exists():
        pytest.skip('simulator archive absent (docs/RUNNING.md)')
    from sim.data import Tables
    with np.load(ARCHIVE) as data:
        meta = json.loads(data['meta'].tobytes())
    return Tables(meta, {})


@pytest.fixture(scope='module')
def pack(lib, base):
    rows, P = loc_types_pack(lib, base.loc_types)
    yield P
    lib.t_free(P)


def load_pool_rows(path, picks, keys=None):
    """sim/world.py load_pool for some layouts of a pool (and of the derived arrays, only `keys` when given), reading
    one array at a time (the derived trees of all eight compact layouts are 170 MB per array)."""
    from dataclasses import fields
    from sim.world import PADDED, Layout
    names = [f.name for f in fields(Layout)]
    rows = [{} for _ in picks]
    with np.load(path) as data:
        counts = {k: data[k] for k in data.files if k.startswith('n_')}
        for name in data.files:
            if name in ('version', 'origin') or name.startswith('n_') or (keys is not None and name not in names
                                                                         and name not in keys):
                continue
            value = data[name]
            for row, k in zip(rows, picks):
                n = counts[f'n_{name}'][k] if name in PADDED else None
                row[name] = np.array(value[k] if n is None else value[k][:n] if PADDED[name] == 0 else value[k][:, :n])
            del value
    layouts = [Layout(**{n: (int(r[n]) if n in ('seed', 'attempt') else r[n]) for n in names}) for r in rows]
    derived = [{n: v for n, v in r.items() if n not in names} or None for r in rows]
    return layouts, derived


@pytest.fixture(scope='module')
def compact():
    """Two layouts of the compact derived pool: one with a pen, one without."""
    if not COMPACT.exists():
        pytest.skip('training pool absent (docs/RUNNING.md)')
    with np.load(COMPACT) as data:
        gates = data['gate']
    pen, open_ = int(np.flatnonzero(gates[:, 0] >= 0)[0]), int(np.flatnonzero(gates[:, 0] < 0)[0])
    return load_pool_rows(COMPACT, [pen, open_])


@pytest.fixture(scope='module')
def supply():
    if not SUPPLY.exists():
        pytest.skip('supply pool absent (docs/RUNNING.md)')
    from sim.world import load_pool
    return load_pool(SUPPLY)[0]


def layout_of(lib, lt):
    """The C layout of a LayoutTables: its boot flags and steps, snapshot steps, stored entity and loc reach, sources,
    and the derived state-0 trees and snapshot rows when it has them."""
    W, H, (x0, z0) = lt.W, lt.H, ORIGIN
    loc_reach = np.full((len(lt.meta['locs']), 24), -1, np.int16)
    for i, row in enumerate(lt.meta['loc_reach']):
        tiles = [(x - x0) * H + (z - z0) for x, z in zip(row[0::2], row[1::2])]
        loc_reach[i, :len(tiles)] = tiles
    d = lt.derived
    return CLayout(lib, W, H, x0, z0, lt.collision, lt.steps, lt.snapshot_steps, walkable=lt.layout.walkable,
                   entity_reach=lt.arrays['entity_reach'], loc_reach=loc_reach,
                   order0=None if d is None else d['order'][0], dist0=None if d is None else d['dist'][0],
                   parent0=None if d is None else d['parent'][0], snap_dist=None if d is None else d['snapshot_dist'])


def test_the_compact_pools_stored_trees_and_rows_are_this_search(lib):
    """The derived state-0 trees and snapshot rows a pack carries for the compact pool (sim/world.py derive) are the
    queue-order search on the boot steps and the bitset depths on the snapshot steps, for every source of all eight
    layouts: a C world without them (search on demand) answers as one with them. One array at a time (170 MB each)."""
    if not COMPACT.exists():
        pytest.skip('training pool absent (docs/RUNNING.md)')
    with np.load(COMPACT) as data:
        picks = list(range(len(data['seed'])))
    layouts, derived = load_pool_rows(COMPACT, picks, keys=('steps', 'snapshot_steps'))
    sources = [[(int(x), int(z)) for x, z in np.argwhere(l.walkable)] for l in layouts]
    for layout, d in zip(layouts, derived):
        assert np.array_equal(d['steps'][0], paths.step_masks(layout.flags))
        assert np.array_equal(d['snapshot_steps'], paths.step_masks(layout.flags_snapshot))
    checked = 0
    for key, part in (('order', 0), ('dist', 1), ('parent', 2), ('snapshot_dist', None)):
        with np.load(COMPACT) as data:
            rows = data[key]
        for k, (layout, d) in enumerate(zip(layouts, derived)):
            H = layout.flags.shape[1]
            for x, z in sources[k]:
                stored = rows[k, 0, x * H + z] if part is not None else rows[k, x * H + z]
                ours = c_rows(lib, d['steps'][0], x, z)[part] if part is not None else c_bitset(lib, d['snapshot_steps'], x, z)
                assert np.array_equal(ours.ravel(), stored), (key, k, x, z)
                checked += 1
        del rows
    assert checked == 4 * sum(map(len, sources))


def test_the_stored_sight_and_reach_tables_are_the_live_functions_on_the_boot_flags(lib, base, supply):
    """paths.c answers a stored sight line with the live ray, which holds because the stored table (sim/world.py
    live_tables: the conjunction of the ray's checks) is that ray on the boot flags; and it reads the stored entity and
    loc reach, which are the live functions on them (filtered to the layout). Checked for every tile, offset and map
    loc of all eight compact layouts (derived state 0) and of supply layouts (LayoutTables computes them)."""
    from sim.world import live_tables
    cases = []
    if COMPACT.exists():
        with np.load(COMPACT) as data:
            picks = list(range(len(data['seed'])))
        layouts, derived = load_pool_rows(COMPACT, picks, keys=('los', 'entity_reach', 'loc_reach'))
        cases += [(l, d['los'][0], d['entity_reach'][0], d['loc_reach'][0][:len(l.locs)]) for l, d in zip(layouts, derived)]
    for layout in supply[:3]:  # as LayoutTables builds them for a layout without derived tables (sim/world.py:749)
        static = live_tables(layout.flags, list(layout.locs), base)
        cases.append((layout, static['los'], static['entity_reach'], static['loc_reach']))
    for layout, los, entity, loc_reach in cases:
        W, H = layout.flags.shape
        flags = np.ascontiguousarray(layout.flags, np.int32)
        table = np.zeros(W * H * 441, np.uint8)
        lib.t_los_table(ptr(flags), W, H, ptr(table))
        assert np.array_equal(table.reshape(W, H, 21, 21), los)
        ent = np.zeros(W * H, np.uint16)
        lib.t_entity_table(ptr(flags), W, H, ptr(ent))
        assert np.array_equal(ent.reshape(W, H), entity)
        out = np.zeros(2 * 512, np.int16)
        for i, (type_id, x, z, shape, angle) in enumerate(layout.locs):
            t = base.loc_types[int(type_id)]
            n = lib.t_loc_reaching(ptr(flags), W, H, 0, 0, int(x), int(z), t['width'], t['length'], int(angle),
                                   int(shape), t['forceapproach'], ptr(out), 512)
            tiles = [int(out[2 * j]) * H + int(out[2 * j + 1]) for j in range(n)
                     if 0 <= out[2 * j] < W and 0 <= out[2 * j + 1] < H]
            assert tiles == [v for v in loc_reach[i].tolist() if v >= 0], i
    assert len(cases) >= 4


def test_every_source_of_a_supply_layout_searches_as_the_reference(lib, base, supply, reasons, pack):
    """A supply layout with a pen (no derived tables: trees and snapshot rows are searched as queried): every source's
    tree equals sim/_paths.c's (every 16th also the Python reference's), its snapshot depths the reference dists, walk
    steps and approaches through LayoutTables (sim/world.py _PerSource) equal the C world's, refusals included, and so
    do the client's routes (Agent.route over World.scene_tree) to tiles on and off the layout, with the fallback."""
    from sim.world import LayoutTables
    layout = next(l for l in supply if l.gate[0] >= 0)
    lt = LayoutTables(base, layout)
    py = World(lt)
    world = CWorld(lib, reasons, pack, layout_of(lib, lt), len(lt.meta['locs']))
    rng, routes = np.random.default_rng(5), 0
    searches = [paths._c_search if paths._C is not None else REF['search']]
    for src, (x, z, _) in enumerate(lt.meta['sources']):
        lx, lz = x - ORIGIN[0], z - ORIGIN[1]
        mine = c_rows(lib, lt.steps, lx, lz)
        for search in searches + ([REF['search']] if src % 16 == 0 else []):
            for a, b in zip(mine, ref_rows(search, lt.steps, lx, lz)):
                assert np.array_equal(a, b)
        assert np.array_equal(c_bitset(lib, lt.snapshot_steps, lx, lz), ref_rows(searches[0], lt.snapshot_steps, lx, lz)[1])
        if src % 4 == 0:
            for _ in range(6):
                tx, tz = x + int(rng.integers(-52, 53)), z + int(rng.integers(-52, 53))
                assert world.walk_steps(src, x, z, tx, tz) == py_outcome(paths.walk_steps, lt, src, x, z, tx, tz)
                assert world.walk_approach(src, x, z, tx, tz) == py_outcome(paths.walk_approach, lt, src, x, z, tx, tz)
        if src % 8 == 0:
            origin = (x + int(rng.integers(-24, 25)), z + int(rng.integers(-24, 25)))
            tx, tz = ORIGIN[0] + int(rng.integers(-4, lt.W + 4)), ORIGIN[1] + int(rng.integers(-4, lt.H + 4))
            nearest = bool(src % 16)
            route = py_outcome(py_world_route, py, x, z, origin, (tx, tz), None, nearest)
            assert world.client_route(x, z, origin, (tx, tz), nearest=nearest) == route
            routes += route[0] == 'ok' and route[1] is not None
    assert routes > 100


def py_world_route(py, sx, sz, origin, dest, arrival=None, nearest=False):
    """Agent.route on an oracle World from (sx, sz) with the build area's origin: (turn points, via) or None."""
    route = paths.client_route(py.scene_tree(0, sx, sz, paths.scene_base(*origin)), sx, sz, dest, arrival, nearest)
    return None if route is None else (list(route), route.via)


def loc_fields(loc):
    return (loc.index, loc.base, loc.type, loc.x, loc.z, loc.shape, loc.angle, loc.width, loc.length, loc.base_shape,
            loc.base_angle, loc.level)


def frozen(value):
    return [tuple(v) for v in value] if isinstance(value, list) else value


def test_collision_changes_and_the_queries_after_them_answer_as_the_world(lib, base, supply, compact, reasons, pack):
    """Random collision changes on pen layouts (map locs out and back, the open stages in and out, a map loc changed to
    a type of another forceapproach, everything reverted twice) through World.loc_collision and rs_loc_collision: the
    same live flags and changed tiles after each, and the same answers to window_changed, path_to_tile, the client's
    routes to a tile (with the fallback), an npc's tile and a loc, entity_reaching, loc_reaching and line_of_sight,
    stored or live, refusals included."""
    from sim.world import LayoutTables
    pens = [l for l in supply if l.gate[0] >= 0][:3] + [compact[0][0]]
    derived = [None, None, None, compact[1][0]]
    rng = np.random.default_rng(7)
    checked = dict(live=0, stored=0)
    for layout, d in zip(pens, derived):
        lt = LayoutTables(base, layout, d)
        py = World(lt)
        c = CWorld(lib, reasons, pack, layout_of(lib, lt), py.n_stored_locs)
        x0, z0 = ORIGIN
        opens = [Loc(len(py.locs) + k, int(t), int(t), x0 + int(x), z0 + int(z), int(s), int(a),
                     lt.loc_types[int(t)]['width'], lt.loc_types[int(t)]['length'], static=False)
                 for k, (t, x, z, s, a) in enumerate(layout.open_locs)]
        present = {id(l): True for l in py.locs} | {id(l): False for l in opens}
        sources = [tuple(s[:2]) for s in lt.meta['sources']]
        # a source beside the gate whose tree the C world keeps across the changes, and tiles in and around the pen
        gate = py.locs[int(layout.gate[0])]
        near = sorted(sources, key=lambda s: (max(abs(s[0] - gate.x), abs(s[1] - gate.z)), s))
        home, around = near[2], near[3:40:3]
        first = py.locs[0]
        bt = lt.loc_types[first.base]
        other = next(k for k, t in lt.loc_types.items() if t['forceapproach'] != bt['forceapproach']
                     and (t['width'], t['length']) == (bt['width'], bt['length']))

        def routes_from_home():
            for x, z in around:
                assert c.path_to_tile(*home, x, z) == py_outcome(lambda: frozen(py.path_to_tile(0, *home, x, z)))

        def toggle(loc, add):
            py.loc_collision(loc, add)
            assert c.loc_collision(loc_fields(loc), add) == ('ok', None)
            present[id(loc)] = add

        def revert():
            for loc in py.locs + opens:
                if present[id(loc)] != (loc in py.locs and loc.type == loc.base):
                    toggle(loc, not present[id(loc)])
            if first.type != first.base:
                first.type = first.base
                toggle(first, True)

        for step in range(16):
            if step in (6, 12):
                revert()
                assert not py.changed
            elif step == 8:  # a changed type: the stored reach no longer holds (World.loc_reaching's conditions)
                if present[id(first)]:
                    toggle(first, False)
                first.type = other
                toggle(first, True)
            else:
                candidates = [py.locs[int(k)] for k in layout.gate] + opens + [py.locs[int(rng.integers(len(py.locs)))]]
                loc = candidates[int(rng.integers(len(candidates)))]
                toggle(loc, not present[id(loc)])
            flags, changed, n = c.state()
            assert flags.tolist() == py._flags and n == len(changed)
            assert changed == {(x - x0) * lt.H + (z - z0) for _, x, z in py.changed}
            routes_from_home()
            for _ in range(6):
                sx, sz = sources[int(rng.integers(len(sources)))]
                assert c.window_changed(sx, sz) == py.window_changed(0, sx, sz)
                box = sorted(int(v) for v in rng.integers(-70, 70, 2) + sx), sorted(int(v) for v in rng.integers(-70, 70, 2) + sz)
                for level in (0, 1):
                    assert c.changed_in(level, *box[0], *box[1]) == py._changed_in(level, *box[0], *box[1])
                checked['live' if py.changed else 'stored'] += 1
                x, z = sx + int(rng.integers(-30, 31)), sz + int(rng.integers(-30, 31))
                assert c.path_to_tile(sx, sz, x, z) == py_outcome(lambda: frozen(py.path_to_tile(0, sx, sz, x, z)))
                origin = (sx + int(rng.integers(-24, 25)), sz + int(rng.integers(-24, 25)))
                assert c.client_route(sx, sz, origin, (x, z), nearest=True) == \
                    py_outcome(py_world_route, py, sx, sz, origin, (x, z), None, True)
                assert c.client_route(sx, sz, origin, (x, z), arrival=(1, 1, 0, 0, 0)) == \
                    py_outcome(py_world_route, py, sx, sz, origin, (x, z), paths.ENTITY_ARRIVAL)
                target = (py.locs + opens)[int(rng.integers(len(py.locs) + len(opens)))]
                t = lt.loc_types[target.type]
                arrival = paths.loc_arrival(target.shape, target.angle, t['width'], t['length'], t['forceapproach'])
                assert c.client_route(sx, sz, origin, (target.x, target.z), loc=loc_fields(target)) == \
                    py_outcome(py_world_route, py, sx, sz, origin, (target.x, target.z), arrival)
                assert c.loc_reaching(loc_fields(target)) == py_outcome(lambda: frozen(py.loc_reaching(target)))
                ex, ez = x0 + int(rng.integers(-2, lt.W + 2)), z0 + int(rng.integers(-2, lt.H + 2))
                assert c.entity_reaching(ex, ez) == py_outcome(lambda: frozen(py.entity_reaching(0, ex, ez)))
                bx, bz = sx + int(rng.integers(-12, 13)), sz + int(rng.integers(-12, 13))
                assert c.line_of_sight(sx, sz, bx, bz) == py_outcome(py.line_of_sight, 0, sx, sz, bx, bz)
                assert c.line_of_sight(ex, ez, sx, sz) == py_outcome(py.line_of_sight, 0, ex, ez, sx, sz)
            routes_from_home()  # home's tree stays cached across the next change
    assert checked['live'] > 100 and checked['stored'] > 50, checked


# ---- a replay of the path queries of a CpuBatch run ----------------------------------------------------------------
WORLD_QUERIES = ('path_to_tile', 'entity_reaching', 'loc_reaching', 'line_of_sight', 'window_changed')
MODULE_QUERIES = ('walk_steps', 'walk_approach')


def record(base, compact, supply, n=8, decisions=100, key=11):
    """Every path query of a goal-activity-v5 CpuBatch run under keyed random legal play (sim/tests/test_workers.py
    run): (name, layout tables index, collision state, args, outcome, agent). The World methods and the module's walk
    functions are wrapped for the run (monkeypatching, the oracle unedited). The client's routes are checked above and
    in sim/tests/test_native_agent.py: a route's tree is no query of its own."""
    from sim.engine import Rng
    from sim import agent as agent_module
    from sim.vec import CpuBatch
    layouts, derived = compact
    tables, index, log, current = [], {}, [], [None]

    def tables_index(t):
        if id(t) not in index:
            index[id(t)] = len(tables)
            tables.append(t)
        return index[id(t)]

    def state(w):
        return tuple(sorted(((x - w._x0) * w._h + (z - w._z0), w._flags[(x - w._x0) * w._h + (z - w._z0)])
                            for _, x, z in w.changed))

    def run(entry, fn, *args):
        try:
            out = fn(*args)
        except (paths.OutsideTables, KeyError, Unsupported) as error:
            text = str(error.args[0]) if isinstance(error, KeyError) else str(error)
            log.append(entry + ((type(error).__name__, text), current[0]))
            raise
        log.append(entry + (('ok', frozen(out)), current[0]))
        return out

    def world_method(name, original):
        def wrapped(self, *args):
            if name == 'loc_reaching':
                shown = args[:-1] + (loc_fields(args[-1]),)
            else:
                shown = args
            return run((name, tables_index(self.t), state(self), shown), original, self, *args)
        return wrapped

    def module_function(name, original):
        def wrapped(*args):
            t, *rest = args
            return run((name, tables_index(t), (), tuple(rest)), original, *args)
        return wrapped

    calls = [0]

    def observing(original):
        def wrapped(self, *args):
            calls[0] += 1
            current[0] = (id(self), calls[0])
            return original(self, *args)
        return wrapped

    with pytest.MonkeyPatch.context() as mp:
        for name in WORLD_QUERIES:
            mp.setattr(World, name, world_method(name, getattr(World, name)))
        for name in MODULE_QUERIES:
            mp.setattr(paths, name, module_function(name, getattr(paths, name)))
        mp.setattr(agent_module.Agent, 'project_items', observing(agent_module.Agent.project_items))
        env = CpuBatch(base, layouts, n, derived=derived, task='goal-activity-v5', search=(supply[:8], None))
        state_, obs = env.reset(key)
        rng = Rng(key)
        for step in range(decisions):
            legal = [np.flatnonzero(obs['mask'][b]) for b in range(n)]
            action = np.asarray([legal[b][int(rng(9, step, b) * len(legal[b]))] for b in range(n)], np.int32)
            state_, obs, *_ = env.step(state_, action)
    return tables, log


@pytest.fixture(scope='module')
def recorded(base, compact, supply):
    return record(base, compact, supply)


def snapshot_hits(tables, log, rows):
    """On the layouts without derived rows (supply, scarce variants), the share of observations whose first snapshot
    row (the walk queries' source) an LRU of `rows` rows per agent already holds."""
    caches, seen, hits, total = {}, set(), 0, 0
    for name, k, _, args, _, current in log:
        if name != 'walk_steps' or current is None or tables[k].derived is not None or current in seen:
            continue
        seen.add(current)
        cache, key = caches.setdefault(current[0], []), (k, args[0])
        total += 1
        if key in cache:
            hits += 1
            cache.remove(key)
        cache.append(key)
        del cache[:-rows]
    return hits / max(total, 1), total


def test_a_recorded_runs_path_queries_replay_in_c(lib, reasons, pack, recorded):
    """Each path query of the run, answered by a C world on its layout (compact with the derived tables, supply,
    scarce variants) in the recorded collision state: the same waypoints, reach, sight, walk steps, routes and
    refusals."""
    tables, log = recorded
    worlds = [CWorld(lib, reasons, pack, layout_of(lib, t), len(t.meta['locs'])) for t in tables]
    applied = [None] * len(tables)
    counts = {}
    for name, k, state, args, expected, _ in log:
        c = worlds[k]
        if name in WORLD_QUERIES and state != applied[k]:
            flags = np.asarray(tables[k].collision, np.int32).copy()
            for t, f in state:
                flags[t] = f
            c.set_flags(flags)
            applied[k] = state
        if name in ('path_to_tile', 'entity_reaching', 'line_of_sight', 'window_changed'):
            assert args[0] == 0
            args = args[1:]
        got = ('ok', c.window_changed(*args)) if name == 'window_changed' else getattr(c, name)(*args)
        assert got == expected, (name, args)
        kind = name + (' live' if state else '') + (' refused' if expected[0] != 'ok' else '')
        counts[kind] = counts.get(kind, 0) + 1
    hits = {rows: snapshot_hits(tables, log, rows) for rows in (1, 2, 4, 8)}
    print(f'\nreplayed {len(log)} queries on {len(tables)} layouts: {dict(sorted(counts.items()))}')
    print(f'snapshot rows held at {hits[1][1]} supply observations, by rows kept:',
          {r: round(h, 3) for r, (h, _) in hits.items()})
    # act-v3's agent walks by the client's route, so a run asks for no engine path or move window; the C tests above
    # check those queries on their own
    assert all(counts.get(name, 0) > 0 for name in ('walk_steps', 'walk_approach', 'entity_reaching', 'loc_reaching',
                                                     'line_of_sight'))
    assert len(tables) > 2  # compact, supply and (in this run) scarce variant layouts
