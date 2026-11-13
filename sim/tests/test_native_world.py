"""The native world core (sim/native/world.c) against the oracle it ports (sim/engine.py World, Player, Inventory and
_links), component by component (the port's design, section 5).

The C side is a harness library: every sim/native/*.c but module.c, built with the extension's flags, with the entry
points world.c calls into other pieces replaced by test doubles. paths.c's rs_loc_collision applies the masks of
sim.paths.collision_masks exactly as World.loc_collision does (engine.py:843-854); its reach and sight queries ask the
oracle world's own methods (World.entity_reaching, loc_reaching, line_of_sight), which hold the same state because the
C world cycles first and only player-phase queries reach them; scripts.c's rs_script_step runs small test programs
that the Python side runs as generators (pause, delay, open a chat and pause, close modals, enqueue). Everything else
is world.c itself. Both sides are compared after every step through the canonical record (rs_world_canon here,
world_canon below for the oracle).

The inventory, queue, stat and runner tests need no private data (a synthetic pack and an 8x8 open layout). The
movement, loc and interaction tests replay a compact pool layout and skip without the archive and the pool."""
from __future__ import annotations

import ctypes
from dataclasses import fields as dc_fields
import hashlib
from pathlib import Path
import subprocess
import weakref

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

from sim import engine, native, paths
from sim.engine import (APLOC1, APLOCU, LEVEL_XP, SKILLS, STACK_LIMIT, Inventory, Loc, Npc, Obj, Player,
                        World, _links, _unlink)

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'logs' / 'sim' / 'data' / 'draynor-nc1.npz'
POOL = ROOT / 'logs' / 'sim' / 'world' / 'train-8-nc1-derived.npz'
ORIGIN = (2880, 3328)
RS_NONE = -2 ** 31   # rs.h: Python's None in a pack record

HARNESS_C = r'''
#include <stdlib.h>
#include <string.h>
#include "rs.h"

/* ---- paths.c doubles: the oracle answers (sim.paths masks, the Python world's reach and sight) ------------------ */
typedef int (*masks_cb)(int32_t type, int32_t shape, int32_t angle, int32_t x, int32_t z, int32_t *out, int cap);
typedef int (*reach_cb)(int32_t x, int32_t z);
typedef int (*locreach_cb)(int32_t index, int16_t *out, int cap);
typedef int (*los_cb)(int32_t ax, int32_t az, int32_t bx, int32_t bz);
static masks_cb cb_masks;
static reach_cb cb_reach;
static locreach_cb cb_locreach;
static los_cb cb_los;
void t_callbacks(masks_cb m, reach_cb r, locreach_cb l, los_cb s) { cb_masks = m, cb_reach = r, cb_locreach = l, cb_los = s; }

void rs_paths_reset(World *w) { (void)w; }

/* World.loc_collision (engine.py:843-854) over the masks the oracle gives. */
int rs_loc_collision(World *w, const Loc *loc, int add) {
    int32_t out[3 * 64];
    int n = cb_masks(loc->type, loc->shape, loc->angle, loc->x, loc->z, out, 64);
    if (n < 0) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "test masks callback");
    const RsLayout *L = w->L;
    for (int i = 0; i < n; i++) {
        int32_t x = out[3 * i], z = out[3 * i + 1], mask = out[3 * i + 2];
        if (!(L->x0 <= x && x < L->x0 + L->W && L->z0 <= z && z < L->z0 + L->H))
            return RS_FAIL(&w->err, RS_E_COLLISION_OUTSIDE, x, z);
        int32_t t = (x - L->x0) * L->H + (z - L->z0);
        w->flags[t] = add ? (w->flags[t] | mask) : (int32_t)((uint32_t)w->flags[t] & ~(uint32_t)mask);
        int was = (int)(w->changed[t >> 6] >> (t & 63) & 1), now = w->flags[t] != L->flags_boot[t];
        if (now) w->changed[t >> 6] |= 1ull << (t & 63);
        else w->changed[t >> 6] &= ~(1ull << (t & 63));
        w->n_changed += now - was;
    }
    if (n) w->coll_ver++;
    return 0;
}

int rs_entity_reaching(World *w, int32_t x, int32_t z, uint16_t *mask) {
    int m = cb_reach(x, z);
    if (m < 0) return RS_FAIL(&w->err, RS_E_ENTITY_OUTSIDE, 0, x, z);
    *mask = (uint16_t)m;
    return 0;
}

int rs_loc_reaching(World *w, const Loc *loc, int16_t out[][2], int *n, int cap) {
    *n = cb_locreach(loc->index, &out[0][0], cap);
    if (*n < 0) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "test loc reach callback");
    return 0;
}

int rs_line_of_sight(World *w, int32_t ax, int32_t az, int32_t bx, int32_t bz, int *visible) {
    (void)w;
    *visible = cb_los(ax, az, bx, bz);
    return 0;
}

/* ---- scripts.c double: test programs of (op, a, b) triples ------------------------------------------------------ */
enum { T_END = 0, T_PAUSE, T_DELAY, T_CHAT, T_CLOSE, T_ENQ };
#define T_PROGS 32
#define T_OPS 12
static int32_t t_progs[T_PROGS][T_OPS][3];
static int32_t t_timer_progs[3];
void t_prog(int k, int n, const int32_t *ops) {
    memset(t_progs[k], 0, sizeof t_progs[k]);
    memcpy(t_progs[k], ops, (size_t)n * 3 * sizeof(int32_t));
}
void t_timer_prog(int i, int k) { t_timer_progs[i] = k; }

int rs_script_step(World *w, Frame *f, RsSusp *out) {
    int16_t s = f->script[0];
    int k = s == RS_S_STAT_REGEN ? t_timer_progs[0] : s == RS_S_STAT_BOOST_RESTORE ? t_timer_progs[1]
        : s == RS_S_HEALTH_REGEN ? t_timer_progs[2] : f->arg[0][0];
    for (;;) {
        const int32_t *op = t_progs[k][f->local[0]++];
        switch (op[0]) {
        case T_END: out->kind = RS_DONE; return 0;
        case T_PAUSE: f->pc[0]++; out->kind = RS_PAUSE; return 0;
        case T_DELAY: f->pc[0]++; out->kind = RS_DELAY; out->n = op[1]; return 0;
        case T_CHAT: rs_open_chat(w, op[1]); f->pc[0]++; out->kind = RS_PAUSE; return 0;
        case T_CLOSE: rs_close_modal(w, op[1]); break;
        case T_ENQ: RS_TRY(rs_enqueue(w, op[1] & 3, RS_S_PICKUP_OBJ, op[1] >> 2, 0, 0, op[2])); break;
        default: return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "test program op");
        }
    }
}

/* ---- npc.c doubles: npcs that do nothing on their turn (the oracle side disables World.npc_turn alike) -------- */
int rs_npc_turn(World *w, Npc *n) { (void)w, (void)n; return 0; }
int rs_npc_lifecycle(World *w, Npc *n) { (void)w, (void)n; return 0; }

/* ---- the driver ------------------------------------------------------------------------------------------------ */
static const RsNpcSpawn no_npcs[1];
World *t_world_new(void) { return calloc(1, sizeof(World)); }
void t_world_free(World *w) { free(w); }
void t_err(const World *w, int32_t *out) {
    out[0] = w->err.code, out[1] = w->err.exc, out[2] = w->err.reason;
    for (int i = 0; i < RS_ERR_ARGS; i++) out[3 + i] = w->err.arg[i];
}
const char *t_err_text(const World *w) { return w->err.text; }
int t_world_init(World *w, const RsPack *P, int layout, int64_t seed, int32_t start_tick, int32_t xp_rate) {
    return rs_world_init(w, P, &P->layouts[layout], seed, start_tick, no_npcs, 0, 0, xp_rate);
}
int t_world_init_npcs(World *w, const RsPack *P, int layout, int64_t seed, int32_t start_tick) {
    return rs_world_init(w, P, &P->layouts[layout], seed, start_tick, NULL, 0, 0, 25);
}
int t_set_interaction_npc(World *w, int32_t nid, int32_t op) {
    for (int i = 0; i < w->n_npcs; i++)
        if (w->npcs[i].nid == nid) return rs_set_interaction(w, rs_ref_npc(w, i), op);
    return -1;
}
void t_place_player(World *w, int32_t x, int32_t z) {   /* its build area around it, as after an arrival */
    rs_player_init(w);
    w->p.x = (int16_t)x, w->p.z = (int16_t)z, w->p.origin_x = (int16_t)x, w->p.origin_z = (int16_t)z;
    w->has_player = 1;
    rs_occupy(w, 0, x, z, 1, 1);
}
void t_queue(World *w, int n, const int16_t *xy) { rs_queue_waypoints(w, (const int16_t (*)[2])xy, n); }
int t_cycle(World *w) { return rs_world_cycle(w); }
int t_canon(const World *w, int64_t *buf, int cap) {
    int n = 0;
    RsErr e;
    return rs_world_canon(w, buf, cap, &n, &e) ? -1 : n;
}
void t_flags_all(const World *w, int32_t *out) { memcpy(out, w->flags, (size_t)w->L->T * sizeof *out); }
int32_t t_flags(const World *w, int32_t x, int32_t z, int npc) { return rs_flags(w, 0, x, z, npc); }
int t_can_travel(World *w, int32_t x, int32_t z, int32_t dx, int32_t dz, int32_t extra) {
    int ok;
    return rs_can_travel(w, 0, x, z, dx, dz, 1, extra, 0, &ok) ? -1 : ok;
}
int t_naive(World *w, int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t dw, int32_t dl, int16_t *out) {
    return rs_naive_path(w, 0, sx, sz, dx, dz, dw, dl, 1, out);
}
/* loc ops by slot (0, or -1 on an error); loc_add returns the slot or -1 */
int t_loc_op(World *w, int op, int slot, int32_t type, int32_t shape, int32_t angle, int32_t duration) {
    switch (op) {
    case 0: return -rs_change_loc(w, slot, type, shape, angle, duration);
    case 1: return -rs_remove_loc(w, slot, duration);
    case 2: return -rs_revert_loc(w, slot);
    default: rs_set_lifecycle(w, slot, duration); return 0;
    }
}
int t_loc_add(World *w, int32_t x, int32_t z, int32_t type, int32_t angle, int32_t shape, int32_t duration) {
    int slot;
    return rs_loc_add(w, 0, x, z, type, angle, shape, duration, &slot) ? -1 : slot;
}
int t_loc_slot(const World *w, int32_t index) {
    for (int s = 0; s < w->n_loc_slots; s++) if (w->locs[s].serial && w->locs[s].index == index) return s;
    return -1;
}
int t_set_interaction_loc(World *w, int slot, int32_t op) { return rs_set_interaction(w, rs_ref_loc(w, slot), op); }
int t_locs_query(const World *w, int which, int32_t x, int32_t z, int32_t radius, int32_t *out) {
    uint8_t s[RS_MAX_LOCS];
    int n = which == 0 ? rs_locs_near(w, 0, x, z, radius, s) : which == 1 ? rs_active_locs_at(w, 0, x, z, s)
        : which == 2 ? rs_loc_at(w, 0, x, z, s) : rs_locs_in_zone(w, 0, x, z, s);
    for (int i = 0; i < n; i++) out[i] = w->locs[s[i]].index;
    return n;
}
/* stats and draws */
void t_set_stats(World *w, const int16_t *levels, const int16_t *base, const int32_t *stats) {
    memcpy(w->p.levels, levels, sizeof w->p.levels);
    memcpy(w->p.base_levels, base, sizeof w->p.base_levels);
    memcpy(w->p.stats, stats, sizeof w->p.stats);
}
int t_stat(World *w, int op, int32_t stat, int32_t a, int32_t b, int cause) {
    switch (op) {
    case 0: return rs_stat_sub(w, stat, a, b);
    case 1: return rs_stat_add(w, stat, a, b);
    case 2: return rs_stat_heal(w, stat, a, b, cause);
    default: return rs_add_xp(w, stat, a);
    }
}
int32_t t_combat_level(const World *w) { return rs_combat_level(w); }
int64_t t_draw(const World *w, int op, int64_t n, int nkeys, const int64_t *keys, int32_t stat, int32_t low,
               int32_t high) {
    return op == 0 ? rs_random(w, n, nkeys, keys) : op == 1 ? rs_randominc(w, n, nkeys, keys)
        : rs_stat_random(w, stat, low, high, nkeys, keys);
}
void t_set_tick(World *w, int32_t tick) { w->tick = tick; }
/* inventories over a caller's slot array */
int64_t t_inv(const RsPack *P, Slot *s, int cap, int stackall, int op, int32_t obj, int64_t count) {
    switch (op) {
    case 0: return rs_inv_add(P, s, cap, stackall, obj, count);
    case 1: return rs_inv_remove(s, cap, obj, count);
    case 2: return rs_inv_total(s, cap, obj);
    case 3: return rs_inv_freespace(s, cap);
    default: return rs_inv_itemspace2(P, s, cap, stackall, obj, count);
    }
}
int t_move_item(World *w, int src, int dst, int32_t obj, int64_t count) { return rs_move_item(w, src, dst, obj, count); }
Slot *t_inventory(World *w, int which) { return which == 0 ? w->p.inv : which == 1 ? w->p.worn : w->p.bank; }
/* queues */
static ReqList *t_list(World *w, int kind) { return kind == 2 ? &w->p.engine : kind == 1 ? &w->p.weak : &w->p.queue; }
int t_enqueue(World *w, int kind, int32_t a0, int32_t delay) { return rs_enqueue(w, kind, RS_S_PICKUP_OBJ, a0, 0, 0, delay); }
/* One _links pass over a queue, with mutations (visit, op, label): 0 unlink the listed node of that label, 1 append a
   node of that label, 2 clear the list. Writes the labels visited; returns their number, -1 on an error. */
int t_links(World *w, int kind, int n_ops, const int32_t *ops, int32_t *visited, int cap) {
    ReqList *list = t_list(w, kind);
    RsLinks it;
    rs_links_begin(w, list, &it);
    int v = 0;
    for (int16_t node; (node = rs_links_next(w, &it)) >= 0; v++) {
        if (v < cap) visited[v] = w->req[node].arg[0];
        for (int i = 0; i < n_ops; i++) {
            if (ops[3 * i] != v) continue;
            int op = ops[3 * i + 1], label = ops[3 * i + 2];
            if (op == 0) {
                for (int16_t k = list->head; k >= 0; k = w->req[k].next)
                    if (w->req[k].arg[0] == label) { rs_req_unlink(w, list, k); break; }
            } else if (op == 1) {
                if (rs_enqueue(w, kind, RS_S_PICKUP_OBJ, label, 0, 0, 0)) return -1;
            } else {
                rs_req_clear(w, list);
            }
        }
    }
    rs_links_end(w);
    return v;
}
int t_list_labels(World *w, int kind, int32_t *out) {
    int n = 0;
    for (int16_t k = t_list(w, kind)->head; k >= 0; k = w->req[k].next) out[n++] = w->req[k].arg[0];
    return n;
}
int t_free_reqs(const World *w) {
    int n = 0;
    for (int16_t k = w->req_free; k >= 0; k = w->req[k].next) n++;
    return n;
}
/* the runner */
void t_set_timer(World *w, int i, int32_t interval, int32_t clock, int16_t script) {
    w->p.timer[i] = (Timer){interval, clock, script, 0, 0};
    if (w->p.n_timers < i + 1) w->p.n_timers = (uint8_t)(i + 1);
}
int t_resume(World *w) { return rs_resume(w); }
int t_script_mode(const World *w) { return w->p.script.serial ? w->p.script.mode : 0; }
void t_close_modal(World *w, int clear_weak) { rs_close_modal(w, clear_weak); }
/* ground objs: append returns the slot (-1 on an error); remove by list position; a held ref's state */
int t_obj_append(World *w, int32_t type, int32_t count, int32_t x, int32_t z, int32_t ticks) {
    int slot;
    return rs_obj_append(w, type, count, x, z, 0, 0, ticks, 1, &slot) ? -1 : slot;
}
void t_obj_remove_at(World *w, int at) { w->objs[w->obj_order[at]].active = 0; rs_obj_remove(w, w->obj_order[at]); }
static RsRef t_held;
void t_obj_hold(World *w, int at) { t_held = rs_ref_obj(w, w->obj_order[at]); }
int t_obj_held_valid(World *w) { return rs_get_obj(w, t_held) != NULL; }
'''

STUBBED = {'paths.c': ('rs_paths_reset', 'rs_loc_collision', 'rs_entity_reaching', 'rs_loc_reaching',
                       'rs_line_of_sight'),
           'scripts.c': ('rs_script_step',), 'npc.c': ('rs_npc_turn', 'rs_npc_lifecycle')}


# ---- the harness library ---------------------------------------------------------------------------------------
class RsErr(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('exc', ctypes.c_uint8), ('reason', ctypes.c_uint16),
                ('arg', ctypes.c_int32 * 6), ('text', ctypes.c_char_p)]


MASKS = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                         ctypes.POINTER(ctypes.c_int32), ctypes.c_int)
REACH = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int32, ctypes.c_int32)
LOCREACH = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int32, ctypes.POINTER(ctypes.c_int16), ctypes.c_int)
LOS = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)

_C = dict(p=ctypes.c_void_p, i=ctypes.c_int, i16=ctypes.c_int16, i32=ctypes.c_int32, i64=ctypes.c_int64,
          s=ctypes.c_char_p)
SIGNATURES = {
    'rs_pack_open': ('i', 's', 'p', 'p'), 'rs_pack_close': (None, 'p'),
    't_callbacks': (None, MASKS, REACH, LOCREACH, LOS), 't_prog': (None, 'i', 'i', 'p'), 't_timer_prog': (None, 'i', 'i'),
    't_world_new': ('p',), 't_world_free': (None, 'p'), 't_err': (None, 'p', 'p'), 't_err_text': ('s', 'p'),
    't_world_init': ('i', 'p', 'p', 'i', 'i64', 'i32', 'i32'), 't_place_player': (None, 'p', 'i32', 'i32'),
    't_queue': (None, 'p', 'i', 'p'), 't_cycle': ('i', 'p'), 't_canon': ('i', 'p', 'p', 'i'),
    't_flags_all': (None, 'p', 'p'), 't_flags': ('i32', 'p', 'i32', 'i32', 'i'),
    't_can_travel': ('i', 'p', 'i32', 'i32', 'i32', 'i32', 'i32'),
    't_naive': ('i', 'p', 'i32', 'i32', 'i32', 'i32', 'i32', 'i32', 'p'),
    't_loc_op': ('i', 'p', 'i', 'i', 'i32', 'i32', 'i32', 'i32'), 't_loc_add': ('i', 'p', 'i32', 'i32', 'i32', 'i32', 'i32', 'i32'),
    't_loc_slot': ('i', 'p', 'i32'), 't_set_interaction_loc': ('i', 'p', 'i', 'i32'),
    't_locs_query': ('i', 'p', 'i', 'i32', 'i32', 'i32', 'p'),
    't_set_stats': (None, 'p', 'p', 'p', 'p'), 't_stat': ('i', 'p', 'i', 'i32', 'i32', 'i32', 'i'),
    't_combat_level': ('i32', 'p'), 't_draw': ('i64', 'p', 'i', 'i64', 'i', 'p', 'i32', 'i32', 'i32'),
    't_set_tick': (None, 'p', 'i32'), 't_inv': ('i64', 'p', 'p', 'i', 'i', 'i', 'i32', 'i64'),
    't_move_item': ('i', 'p', 'i', 'i', 'i32', 'i64'), 't_inventory': ('p', 'p', 'i'),
    't_enqueue': ('i', 'p', 'i', 'i32', 'i32'), 't_links': ('i', 'p', 'i', 'i', 'p', 'p', 'i'),
    't_list_labels': ('i', 'p', 'i', 'p'), 't_free_reqs': ('i', 'p'),
    't_set_timer': (None, 'p', 'i', 'i32', 'i32', 'i16'), 't_resume': ('i', 'p'), 't_script_mode': ('i', 'p'),
    't_close_modal': (None, 'p', 'i'), 't_world_init_npcs': ('i', 'p', 'p', 'i', 'i64', 'i32'),
    't_set_interaction_npc': ('i', 'p', 'i32', 'i32'), 't_obj_append': ('i', 'p', 'i32', 'i32', 'i32', 'i32', 'i32'),
    't_obj_remove_at': (None, 'p', 'i'), 't_obj_hold': (None, 'p', 'i'), 't_obj_held_valid': ('i', 'p'),
}


def build_harness(out_dir):
    """Compile every sim/native/*.c but module.c (STUBBED entry points renamed away) and the driver into one library."""
    header = native.generated_header()
    digest = hashlib.sha256(header.encode() + HARNESS_C.encode() + ' '.join(native.FLAGS).encode())
    for path in sorted(native.SOURCE.glob('*.[ch]')):
        digest.update(path.name.encode() + path.read_bytes())
    lib = out_dir / f'world-harness-{digest.hexdigest()[:16]}.so'
    if lib.exists():
        return lib
    include = out_dir / 'include'
    include.mkdir(parents=True, exist_ok=True)
    (include / 'rs_gen.h').write_text(header)
    (out_dir / 'harness.c').write_text(HARNESS_C)
    objects = []
    for src in [*native.sources(), out_dir / 'harness.c']:
        if src.name == 'module.c':
            continue
        renames = [f'-D{name}={name}_ported' for name in STUBBED.get(src.name, ())]
        obj = out_dir / (src.stem + '.o')
        command = ['cc', *native.FLAGS, *renames, '-I', str(include), '-I', str(native.SOURCE), '-c', str(src), '-o', str(obj)]
        done = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr[-3000:]
        objects.append(str(obj))
    done = subprocess.run(['cc', '-shared', *map(str, objects), '-o', str(lib), '-lz'], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-3000:]
    return lib


@pytest.fixture(scope='session')
def lib(tmp_path_factory):
    c = ctypes.CDLL(str(build_harness(tmp_path_factory.mktemp('harness'))))
    for name, (res, *args) in SIGNATURES.items():
        f = getattr(c, name)
        f.restype = _C[res] if isinstance(res, str) else res
        f.argtypes = [_C[a] if isinstance(a, str) else a for a in args]
    return c


@pytest.fixture(scope='session')
def ids():
    """The module's tables by name: canonical fields, scripts and pack sections."""
    m = native.load()
    return dict(canon={name: k for k, name in enumerate(m.canon_fields())},
                script={name: k for k, (name, _, _) in enumerate(m.scripts())},
                sections=m.constants()['sections'])


def ptr(array):
    """The address of a numpy array the caller keeps alive across the call."""
    return ctypes.c_void_p(array.ctypes.data)


# ---- packs ------------------------------------------------------------------------------------------------------
class Strings:
    """The pack's string table (id 0 is '') as the oracle side needs it: text -> id, and .string for as_python."""
    def __init__(self, texts=()):
        self.texts, self.ids = [''], {'': 0}
        for text in texts:
            self.id(text)

    def id(self, text):
        if text not in self.ids:
            self.ids[text] = len(self.texts)
            self.texts.append(text)
        return self.ids[text]

    def string(self, k):
        return self.texts[k]


def record(name, n=1):
    return np.zeros(n, native.record_dtype(name))


def fill(row, **values):
    for name, value in values.items():
        row[name] = value


def write_pack(path, ids, strings, obj_types=None, loc_types=None, npc_types=None, layout=None, messages=()):
    """A pack with what world.c reads: CONSTS, type records, messages ((key, fill, text) rows) and one layout (a dict
    of the RsLayoutRec arrays: flags, tile_src, locs; the per-tile tables world.c does not read are zeros)."""
    sec = ids['sections']
    writer = native.PackWriter(path, {})
    consts = record('RsConsts')
    consts['level_xp'] = LEVEL_XP
    consts['members'] = 1
    writer.add(sec['CONSTS'], consts, record='RsConsts')
    for key, table, rec in (('OBJ_TYPES', obj_types, 'RsObjType'), ('LOC_TYPES', loc_types, 'RsLocType'),
                            ('NPC_TYPES', npc_types, 'RsNpcType')):
        if table is not None:
            writer.add(sec[key], table, record=rec)
    if messages:
        msgs, keys = record('RsMsg', len(messages)), record('RsMsgKeyRow', len(messages))
        for k, (key, value, text) in enumerate(sorted(messages)):
            msgs[k]['text'] = strings.id(text)
            keys[k] = (key, value, k)
        writer.add(sec['MSGS'], msgs, record='RsMsg')
        writer.add(sec['MSG_KEYS'], keys, record='RsMsgKeyRow')
    if layout is not None:
        heaps = dict(i32=[], i16=[], u16=[], u8=[])
        sizes = dict.fromkeys(heaps, 0)
        kinds = dict(i32=np.int32, i16=np.int16, u16=np.uint16, u8=np.uint8)

        def put(kind, array):
            a = np.ascontiguousarray(array, kinds[kind]).ravel()
            heaps[kind].append(a)
            sizes[kind] += a.size
            return sizes[kind] - a.size
        W, H = layout['flags'].shape
        T = W * H
        rec = record('RsLayoutRec')
        for name in rec.dtype.names:
            if rec.dtype[name].kind == 'i' and rec.dtype[name].shape == () and rec.dtype[name].itemsize == 8:
                rec[name] = -1
        src = np.asarray(layout['tile_src'], np.int16).ravel()
        locs = np.asarray(layout['locs'], np.int32).reshape(-1, 5)
        fill(rec, kind=0, W=W, H=H, T=T, x0=ORIGIN[0], z0=ORIGIN[1], n_src=int((src >= 0).sum()), n_locs=len(locs),
             gate=(-1, -1))
        rec['flags_boot'] = put('i32', layout['flags'])
        rec['steps_boot'], rec['steps_snap'], rec['sides'] = (put('u8', np.zeros(T)) for _ in range(3))
        rec['entity_reach'] = put('u16', np.zeros(T))
        rec['tile_src'] = put('i16', src)
        rec['src_tile'] = put('i16', np.flatnonzero(src >= 0))
        if len(locs):
            rec['locs'] = put('i32', locs)
            rec['loc_reach'] = put('i16', np.full((len(locs), 24), -1))
        npcs = np.asarray(layout.get('npcs', np.zeros((0, 4))), np.int32).reshape(-1, 4)
        if len(npcs):
            rec['n_npcs'], rec['npcs'] = len(npcs), put('i32', npcs)
        writer.add(sec['LAYOUTS'], rec, record='RsLayoutRec')
        for kind, parts in heaps.items():
            if parts:
                writer.add(sec[f'HEAP_{kind.upper()}'], np.concatenate(parts))
    blob = b''.join(text.encode() + b'\0' for text in strings.texts)
    offsets = np.cumsum([0] + [len(text.encode()) + 1 for text in strings.texts[:-1]]).astype(np.uint32)
    writer.add(sec['STRINGS'], blob)
    writer.add(sec['STR_OFF'], offsets)
    return writer.close()


def open_pack(lib, path):
    P, err = ctypes.c_void_p(), RsErr()
    assert lib.rs_pack_open(str(path).encode(), ctypes.byref(P), ctypes.byref(err)) == 0, err.text
    return P


# ---- the oracle's canonical record (rs.h RS_CANON_FIELDS, rs_world_canon) -----------------------------------------
CAUSES = dict(damage=0, eat=1, regen=2, levelup=3)
LOCATIONS = dict(inv=0, worn=1, bank=2)
EVENT_KINDS = dict(agent_hp=0, pickup=1, produce=2, transfer=3, remove=4)
MODES = dict(pausebutton=1, suspended=2, running=3)
TIMERS = ('stat_regen', 'stat_boost_restore', 'health_regen')


class Canon:
    """How the oracle's objects map to the record's ids: strings, script frames (a registry the test programs fill:
    generator -> (script id, [pc])) and queued requests by name."""
    def __init__(self, ids, strings):
        self.ids, self.strings, self.frames = ids, strings, weakref.WeakKeyDictionary()

    def script(self, name):
        return self.ids['script'][name]

    def frame(self, gen):
        script, box = self.frames[gen]
        return [script, box[0]]

    def request(self, r):
        if r.name == '[changestat,_]':
            return [self.script('RS_S_CHANGESTAT'), r.delay, 0, 0, 0]
        if r.name.startswith('[advancestat,'):
            return [self.script('RS_S_LEVELUP'), r.delay, SKILLS.index(r.name[13:-1]), 0, 0]
        assert r.name.startswith('prog'), r.name
        return [self.script('RS_S_PICKUP_OBJ'), r.delay, int(r.name[4:]), 0, 0]

    def record(self, w):
        out, F, sid = [], self.ids['canon'], self.strings.id

        def field(name, values):
            out.extend([F[name], len(values), *map(int, values)])

        def ref(t):
            if t is None:
                return [0, -1]
            if isinstance(t, Npc):
                return [1, t.nid]
            if isinstance(t, Loc):
                return [2, t.index]
            if isinstance(t, Obj):
                return [3, next((k for k, o in enumerate(w.objs) if o is t), -1)]
            return [4, 0]  # the player (rs.h canonical state)
        p = w.player if w.player is not None else w.pending_login[0] if w.pending_login else None
        field('RS_CF_TICK', [w.tick, w.n_stored_locs, w._next_loc, w.loc_version, w.pending_login is not None,
                             w.player is not None])
        if p is None:
            for name in ('PLAYER', 'PLAYER_PATH', 'PLAYER_TARGET', 'PLAYER_STATS', 'PLAYER_VARPS', 'PLAYER_LAST',
                         'PLAYER_INV', 'PLAYER_WORN', 'PLAYER_BANK', 'PLAYER_MODAL', 'PLAYER_SCRIPT', 'PLAYER_TIMERS',
                         'PLAYER_QUEUE', 'PLAYER_WEAK', 'PLAYER_ENGINE'):
                field(f'RS_CF_{name}', [])
        else:
            field('RS_CF_PLAYER', [p.x, p.z, p.level, p.instant, p.steps_taken, p.last_movement, p.delayed,
                                   p.delayed_until, p.protect, p.dead, p.afk_ready, p.origin_x, p.origin_z,
                                   p.allow_repath])
            field('RS_CF_PLAYER_PATH', [v for xz in p.path for v in xz])
            field('RS_CF_PLAYER_TARGET', ref(p.target) + [p.target_op, p.target_type, p.ap_range, p.ap_range_called]
                  + ref(p.next_target))
            field('RS_CF_PLAYER_STATS', [v for i in range(len(SKILLS)) for v in (p.levels[i], p.base_levels[i],
                                                                                 p.stats[i])])
            field('RS_CF_PLAYER_VARPS', [-1 if v is None else v for v in p.varps.values()])
            com = (-1, -1) if p.last_com == -1 else (sid(p.last_com[0]), p.last_com[1])
            field('RS_CF_PLAYER_LAST', [p.last_useitem, *com, p.last_item, p.last_slot, p.last_useslot])
            for name, inv in (('INV', p.inv), ('WORN', p.worn), ('BANK', p.bank)):
                field(f'RS_CF_PLAYER_{name}', [v for row in inv.listed() for v in row])
            bits = sum(bit for name, bit in (('main', 1), ('side', 2), ('chat', 4)) if name in p.modal_state)
            field('RS_CF_PLAYER_MODAL', [bits, -1 if p.modal_main is None else sid(p.modal_main),
                                         -1 if p.modal_chat is None else sid(p.modal_chat)]
                  + [v for iface, com in p.resume_buttons for v in (sid(iface), com)])
            field('RS_CF_PLAYER_SCRIPT', [0, 0] if p.script is None else [MODES[p.script[1]], 1, *self.frame(p.script[0])])
            field('RS_CF_PLAYER_TIMERS', [v for name, t in p.timers.items()
                                          for v in (self.script(f'RS_S_{name.upper()}'), t['kind'] == 'soft',
                                                    t['interval'], t['clock'])])
            for name, queue in (('QUEUE', p.queue), ('WEAK', p.weak_queue), ('ENGINE', p.engine_queue)):
                field(f'RS_CF_PLAYER_{name}', [v for r in queue for v in self.request(r)])
        mine = lambda v: v is not None and v == (p.name if p is not None else 'wd1a0')
        field('RS_CF_NPC', [v for n in w.npcs for v in (
            n.nid, n.type, n.base, n.x, n.z, n.level, n.active, n.delayed, n.delayed_until, n.mode,
            n.target is not None, n.timer_interval, n.timer_clock, n.hunt_clock, n.regen_clock, n.stuck, n.tele,
            n.observers, n.last_tick_x, n.last_tick_z, n.hunt_mode, n.huntrange, n.hunt_target is not None,
            n.lifecycle_tick, n.last_movement, n.life, bool(n.path), *(n.path[0] if n.path else (0, 0)),
            bool(n.heroes), sum(n.heroes.values()), n.vars['action_delay'], n.vars['lastcombat'],
            mine(n.vars['aggressive_player']), mine(n.vars['attacking']), n.vars['xp_mult'], n.vars['stunned'],
            *n.levels)])
        assert all(n.script is None and not n.queue for n in w.npcs), 'npc scripts and queues are npc.c records'
        field('RS_CF_NPC_SCRIPT', [v for n in w.npcs for v in (n.nid, 0, 0)])
        field('RS_CF_NPC_QUEUE', [v for n in w.npcs for v in (n.nid, 0)])
        field('RS_CF_NPC_EVENTS', [n.nid for n in w.npc_events])
        added = sorted({id(loc): loc for row in w.loc_at.values() for loc in row if not loc.static}.values(),
                       key=lambda loc: loc.index)
        field('RS_CF_LOCS', [v for loc in w.locs + added for v in (
            loc.index, loc.base, loc.type, loc.x, loc.z, loc.shape, loc.angle, loc.width, loc.length, loc.static,
            loc.active, loc.lifecycle_tick, loc.layer)])
        field('RS_CF_ZONES', [v for key in sorted(k for k, row in w.loc_zones.items() if row)
                              for v in (key[1], key[2], len(w.loc_zones[key]), *(loc.index for loc in w.loc_zones[key]))])
        field('RS_CF_LOC_EVENTS', [loc.index for loc in w.loc_events])
        field('RS_CF_OBJS', [v for o in w.objs for v in (o.type, o.count, o.x, o.z, o.level, o.static, o.ticks,
                                                          o.receiver is not None, o.active)])
        inside = lambda x, z: w._x0 <= x <= w._x1 and w._z0 <= z <= w._z1
        field('RS_CF_OCCUPIED', [v for (_, x, z), c in sorted(w.occupied.items()) if c and inside(x, z) for v in (x, z, c)])
        field('RS_CF_CHANGED', [v for _, x, z in sorted(w.changed) for v in (x, z)])
        chat = w.chat_log[-1] if w.chat_log else None
        text = w.text_log[-1] if w.text_log else None
        field('RS_CF_LOGS', [len(w.chat_log), sid(chat[1]) if chat else -1, chat[0] if chat else -1, len(w.text_log),
                             sid(text[1][0]) if text else -1, text[1][1] if text else -1, sid(text[2]) if text else -1,
                             len(w.events)])
        field('RS_CF_EVENT_LAST', self.event(*w.events[-1]) if w.events else [])
        return out

    def event(self, tick, kind, d):
        a8 = b8 = has_source = obj = count = v0 = v1 = v2 = 0
        if kind == 'agent_hp':
            a8, count, v0, v1 = CAUSES.get(d['cause'], 4), d['planned'], d['before'], d['after']
            has_source, v2 = d['source'] is not None, d['source'] or 0
        elif kind in ('pickup', 'produce'):
            obj, count = d['obj'], d['count']
            v0 = self.strings.id(d['script']) if kind == 'produce' else 0
        elif kind == 'transfer':
            a8, b8, obj, count, v0 = LOCATIONS[d['src']], LOCATIONS[d['dst']], d['obj'], d['count'], d['to']
        else:
            raise NotImplementedError(f'events of kind {kind} are npc.c and combat.c records')
        return [EVENT_KINDS[kind], tick, a8, b8, has_source, obj, count, v0, v1, v2, 0]


def c_canon(lib, w, cap=1 << 16):
    buf = np.zeros(cap, np.int64)
    n = lib.t_canon(w, ptr(buf), cap)
    assert n >= 0
    return buf[:n].tolist()


def decode(record, ids):
    """{field name: values} of a canonical record (the diff shows field names, not offsets)."""
    names, out, k = {v: n for n, v in ids['canon'].items()}, {}, 0
    while k < len(record):
        out[names[record[k]]] = record[k + 2:k + 2 + record[k + 1]]
        k += 2 + record[k + 1]
    return out


def assert_same(lib, w_c, w_py, canon, note=''):
    a, b = c_canon(lib, w_c), canon.record(w_py)
    if a != b:
        da, db = decode(a, canon.ids), decode(b, canon.ids)
        diff = {k: (da.get(k), db.get(k)) for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k)}
        raise AssertionError(f'{note}: C vs oracle differ in {diff}')


def c_error(lib, w, strings):
    """The oracle exception a C world's error renders as (sim/native.py as_python)."""
    out = np.zeros(9, np.int32)
    lib.t_err(w, ptr(out))
    text = lib.t_err_text(w)
    error = native.load().Error(*(int(v) for v in out[:3]), tuple(int(v) for v in out[3:]),
                                text.decode() if text else None, 0)
    return native.as_python(error, strings)


# ---- the tiny world: a synthetic pack, an 8x8 open layout, the tables World reads ----------------------------------
OBJS = dict(coins=(True, -1, -1), logs=(False, -1, -1), cert_logs=(True, 3, 1), raw_shrimp=(False, -1, -1),
            feather=(True, -1, -1), bones=(False, -1, -1))   # debugname: stackable, certtemplate, certlink
IFACES = ('message1', 'objbox1', 'doubleobjbox')


class TinyTables:
    """What World and Player read of a Tables for an 8x8 open layout at ORIGIN with no locs, npcs or objs."""
    def __init__(self):
        x0, z0 = ORIGIN
        self.meta = dict(locs=[], npcs=[], objs=[], environment=dict(members=True), multiway=[], content={})
        self.area = dict(x0=x0, x1=x0 + 7, z0=z0, z1=z0 + 7)
        self.size, self.levels, self.collision = (8, 8), [0], np.zeros(64, np.int32)
        self.obj_types = {k: dict(debugname=name, stackable=s, certtemplate=ct, certlink=cl)
                          for k, (name, (s, ct, cl)) in enumerate(OBJS.items())}
        self.npc_types, self.loc_types = {}, {}

    def source(self, level, x, z):
        a = self.area
        inside = level == 0 and a['x0'] <= x <= a['x1'] and a['z0'] <= z <= a['z1']
        return (x - a['x0']) * 8 + (z - a['z0']) if inside else -1


@pytest.fixture(scope='session')
def tiny(lib, ids, tmp_path_factory):
    strings = Strings(IFACES)
    objs = record('RsObjType', len(OBJS))
    for k, (name, (stack, ct, cl)) in enumerate(OBJS.items()):
        fill(objs[k], present=1, stackable=stack, certtemplate=ct, certlink=cl, debugname=strings.id(name))
    layout = dict(flags=np.zeros((8, 8), np.int32), tile_src=np.arange(64), locs=np.zeros((0, 5)))
    path = write_pack(tmp_path_factory.mktemp('tiny') / 'tiny.pack', ids, strings, obj_types=objs, layout=layout)
    return open_pack(lib, path), strings


class Pair:
    """A C world and an oracle world started alike (the player placed at (x, z), as Agent._arrive leaves it), without
    npcs or with the layout's."""
    def __init__(self, lib, P, tables, start_tick=6, xp_rate=25, player=(2883, 3331), seed=7, npcs=False):
        self.lib, self.w = lib, lib.t_world_new()
        if npcs:
            assert lib.t_world_init_npcs(self.w, P, 0, seed, start_tick) == 0
        else:
            assert lib.t_world_init(self.w, P, 0, seed, start_tick, xp_rate) == 0
        self.py = World(tables, seed=seed, start_tick=start_tick, npcs=None if npcs else [], xp_rate=xp_rate)
        if player is not None:
            lib.t_place_player(self.w, *player)
            p = self.p = Player(self.py, 'wd1a0')
            p.x, p.z = player
            p.origin_x, p.origin_z = player   # the client's scene (act-v3 routes) around the player
            self.py.player = p
            self.py._occupy(p, True)

    def close(self):
        self.lib.t_world_free(self.w)


def set_stats(lib, pair):
    """The oracle player's levels, base levels and xp into the C player."""
    p = pair.p
    levels, bases, xps = (np.asarray(p.levels, np.int16), np.asarray(p.base_levels, np.int16),
                          np.asarray(p.stats, np.int32))
    lib.t_set_stats(pair.w, ptr(levels), ptr(bases), ptr(xps))


# ---- Inventory ---------------------------------------------------------------------------------------------------
COUNTS = st.one_of(st.integers(0, 3), st.integers(0, 60), st.sampled_from([STACK_LIMIT, STACK_LIMIT - 1, 2 ** 30]),
                   st.integers(0, STACK_LIMIT))
INV_OPS = st.lists(st.tuples(st.sampled_from(['add', 'remove', 'total', 'freespace', 'itemspace2']),
                             st.integers(0, len(OBJS) - 1), COUNTS), max_size=60)


@settings(deadline=None, max_examples=400)
@given(st.sampled_from([(28, False), (14, False), (240, True), (3, False), (2, True)]), INV_OPS)
def test_inventory_sequences_equal_the_oracle(lib, tiny, shape, ops):
    """engine.Inventory's add (first fill for unstackables, even of 0; the first slot holding the obj, an occupied
    [id, 0] included; the stack cap), remove (slot order, freed at 0), total, freespace and itemspace2 (the first
    matching slot only), op by op, return values and slots."""
    P, _ = tiny
    cap, stackall = shape
    inv, t = Inventory(cap, stackall), TinyTables()
    slots = np.tile(np.asarray([-1, 0], np.int32), (cap, 1))
    for op, obj, count in ops:
        expected = {'add': lambda: inv.add(t, obj, count), 'remove': lambda: inv.remove(obj, count),
                    'total': lambda: inv.total(obj), 'freespace': inv.freespace,
                    'itemspace2': lambda: inv.itemspace2(t, obj, count)}[op]()
        code = ['add', 'remove', 'total', 'freespace', 'itemspace2'].index(op)
        assert lib.t_inv(P, ptr(slots), cap, stackall, code, obj, count) == expected, (op, obj, count)
        assert slots.tolist() == [[-1, 0] if it is None else it for it in inv.items], (op, obj, count)


@settings(deadline=None, max_examples=150)
@given(st.lists(st.tuples(st.sampled_from([(0, 1), (0, 2), (2, 0), (1, 0), (0, 0)]),
                          st.integers(0, len(OBJS) - 1), COUNTS), max_size=12),
       st.lists(st.tuples(st.integers(0, 2), st.integers(0, len(OBJS) - 1), st.integers(0, 40)), max_size=10))
def test_move_item_and_its_transfer_event_equal_the_oracle(lib, ids, tiny, moves, fills):
    """INV_MOVEITEM_UNCERT: what the source gives moves to the destination as its uncert obj (what does not fit
    vanishes, as in the oracle), with one transfer event per move that moved anything."""
    P, strings = tiny
    pair = Pair(lib, P, TinyTables())
    canon = Canon(ids, strings)
    try:
        names = (engine.INV, engine.WORN, engine.BANK)
        for where, obj, count in fills:
            inv = pair.p.inventory(names[where])
            inv.add(pair.py.t, obj, count)
            slots = np.ctypeslib.as_array(ctypes.cast(lib.t_inventory(pair.w, where), ctypes.POINTER(ctypes.c_int32)),
                                          (inv.capacity * 2,))
            slots[:] = np.asarray([v for it in inv.items for v in ((-1, 0) if it is None else it)], np.int32)
        for (src, dst), obj, count in moves:
            pair.py.move_item(pair.p, names[src], names[dst], obj, count)
            assert lib.t_move_item(pair.w, src, dst, obj, count) == 0
            assert_same(lib, pair.w, pair.py, canon, f'move {src}->{dst} {obj} x{count}')
    finally:
        pair.close()


# ---- queues and _links ------------------------------------------------------------------------------------------
class Node:
    def __init__(self, label):
        self.label = label


MUTATION = st.tuples(st.integers(0, 12), st.sampled_from([0, 0, 1, 2]), st.integers(0, 9))


@settings(deadline=None, max_examples=500)
@given(st.integers(0, 7), st.sampled_from([0, 1, 2]), st.lists(st.lists(MUTATION, max_size=8), min_size=1, max_size=4))
def test_queue_passes_visit_as_links_does(lib, tiny, n0, kind, passes):
    """engine._links over random mutations made during the pass (unlink any listed node, the one being visited and its
    successor included, append, clear): the successor is read at the visit, so an appended request waits for the next
    pass when it arrives as the last one runs, and a successor unlinked meanwhile is still visited, then the pass
    ends. Request slots unlinked in a pass are reused only after it."""
    P, _ = tiny
    pair = Pair(lib, P, TinyTables())
    try:
        items, label = [Node(k) for k in range(n0)], n0
        for k in range(n0):
            assert lib.t_enqueue(pair.w, kind, k, 0) == 0
        for ops in passes:
            fresh = []
            for visit, op, arg in ops:
                fresh.append((visit, op, label if op == 1 else arg))
                label += op == 1
            visited = []
            for node in _links(items):
                visited.append(node.label)
                for visit, op, arg in fresh:
                    if visit != len(visited) - 1:
                        continue
                    if op == 0:
                        hit = next((n for n in items if n.label == arg), None)
                        if hit is not None:
                            _unlink(items, hit)
                    elif op == 1:
                        items.append(Node(arg))
                    else:
                        items.clear()
            flat = np.asarray([v for row in fresh for v in row] or [0], np.int32)
            seen = np.zeros(256, np.int32)
            n = lib.t_links(pair.w, kind, len(fresh), ptr(flat), ptr(seen), 256)
            assert seen[:n].tolist() == visited
            listed = np.zeros(256, np.int32)
            m = lib.t_list_labels(pair.w, kind, ptr(listed))
            assert listed[:m].tolist() == [node.label for node in items]
            assert lib.t_free_reqs(pair.w) == native.load().constants()['RS_MAX_REQS'] - len(items)
    finally:
        pair.close()


# ---- stats, xp and the RuneScript draws --------------------------------------------------------------------------
LEVELS = st.lists(st.integers(0, 120), min_size=21, max_size=21)
BASES = st.lists(st.integers(1, 99), min_size=21, max_size=21)
XPS = st.lists(st.one_of(st.integers(0, 14_000_000), st.sampled_from(LEVEL_XP)), min_size=21, max_size=21)
STAT_OPS = st.lists(st.tuples(st.integers(0, 3), st.integers(0, 20), st.integers(-5, 400), st.integers(-10, 150),
                              st.sampled_from(['eat', 'regen', None])), max_size=12)
HEAL_CAUSES = {'eat': 1, 'regen': 2, None: 4}   # rs.h RS_HP_*


@settings(deadline=None, max_examples=250, suppress_health_check=[HealthCheck.too_slow])
@given(LEVELS, BASES, XPS, STAT_OPS, st.integers(1, 30), st.integers(0, 2000))
def test_stat_ops_and_add_xp_equal_the_oracle(lib, ids, tiny, levels, bases, xps, ops, rate, tick):
    """STAT_SUB/ADD/HEAL (Python's true division and truncation; a heal event for an HP gain; [changestat] whenever
    the computed level moved) and Player.addXp (the XP rate, the 2e9 cap, current level following base, the levelup
    event before [changestat] before [advancestat,<skill>] on the engine queue), op by op on the whole record."""
    P, strings = tiny
    pair = Pair(lib, P, TinyTables(), xp_rate=rate, start_tick=tick)
    canon = Canon(ids, strings)
    try:
        p = pair.p
        p.levels, p.base_levels, p.stats = list(levels), list(bases), list(xps)
        set_stats(lib, pair)
        assert lib.t_combat_level(pair.w) == p.combat_level()
        for op, stat, a, b, cause in ops:
            if op == 0:
                pair.py.stat_sub(p, stat, a, b)
            elif op == 1:
                pair.py.stat_add(p, stat, a, b)
            elif op == 2:
                pair.py.stat_heal(p, stat, a, b, cause=cause)
            else:
                pair.py.add_xp(p, stat, a)
            assert lib.t_stat(pair.w, op, stat, a, b, HEAL_CAUSES[cause]) == 0
            assert_same(lib, pair.w, pair.py, canon, f'stat op {op} {stat} {a} {b}')
            assert lib.t_combat_level(pair.w) == p.combat_level()
    finally:
        pair.close()


@settings(deadline=None, max_examples=400)
@given(st.integers(0, 2 ** 24), st.integers(0, 5000), st.lists(st.integers(-2 ** 40, 2 ** 40), max_size=6),
       st.integers(-10, 2 ** 31), st.integers(0, 20), st.integers(0, 255), st.integers(0, 255), st.integers(0, 120))
def test_runescript_draws_equal_the_oracle(lib, tiny, seed, tick, keys, n, stat, low, high, level):
    """world.random, randominc and stat_random on (R_SCRIPT, tick, *keys): int(draw * n) in double, floor division in
    the success threshold."""
    P, _ = tiny
    pair = Pair(lib, P, TinyTables(), seed=seed, start_tick=tick)
    try:
        pair.p.levels[stat] = level
        set_stats(lib, pair)
        k = np.asarray(keys or [0], np.int64)
        assert lib.t_draw(pair.w, 0, n, len(keys), ptr(k), 0, 0, 0) == pair.py.random(n, *keys)
        assert lib.t_draw(pair.w, 1, n, len(keys), ptr(k), 0, 0, 0) == pair.py.randominc(n, *keys)
        assert lib.t_draw(pair.w, 2, 0, len(keys), ptr(k), stat, low, high) == pair.py.stat_random(pair.p, stat, low,
                                                                                                   high, *keys)
    finally:
        pair.close()


# ---- ground objs ------------------------------------------------------------------------------------------------
OBJ_OPS = st.lists(st.one_of(
    st.tuples(st.just('drop'), st.integers(0, len(OBJS) - 1), st.integers(1, 5), st.integers(0, 7), st.integers(0, 7),
              st.sampled_from([-1, 0, 1, 2, 3, 5])),
    st.tuples(st.just('take'), st.integers(0, 40)), st.tuples(st.just('hold'), st.integers(0, 40)),
    st.tuples(st.just('cycle'), st.integers(1, 4))), max_size=30)


@settings(deadline=None, max_examples=200)
@given(OBJ_OPS)
def test_objs_keep_list_order_and_despawn_as_the_oracle(lib, ids, tiny, ops):
    """world.objs: drops append, a removal shifts the rest, process_zones counts each drop down over a snapshot of the
    list and removes it at 0 (a held reference then reads gone, as the oracle's obj turns inactive; a later drop
    reusing the slot is another obj)."""
    P, strings = tiny
    pair = Pair(lib, P, TinyTables())
    canon = Canon(ids, strings)
    held = None
    try:
        for op in ops:
            if op[0] == 'drop':
                _, obj, count, x, z, ticks = op
                x, z = ORIGIN[0] + x, ORIGIN[1] + z
                pair.py.objs.append(Obj(obj, count, x, z, static=False, ticks=ticks, receiver=pair.p.name))
                assert lib.t_obj_append(pair.w, obj, count, x, z, ticks) >= 0
            elif op[0] in ('take', 'hold') and pair.py.objs:
                at = op[1] % len(pair.py.objs)
                if op[0] == 'take':
                    obj = pair.py.objs[at]
                    obj.active = False
                    pair.py.objs.remove(obj)
                    lib.t_obj_remove_at(pair.w, at)
                else:
                    held = pair.py.objs[at]
                    lib.t_obj_hold(pair.w, at)
            elif op[0] == 'cycle':
                for _ in range(op[1]):
                    assert lib.t_cycle(pair.w) == 0
                    pair.py.cycle()
            assert_same(lib, pair.w, pair.py, canon, f'{op}')
            if held is not None:
                assert lib.t_obj_held_valid(pair.w) == held.is_valid()
    finally:
        pair.close()


# ---- the script runner, the queues and the timers in the player phase ------------------------------------------
END, PAUSE, DELAY, CHAT, CLOSE, ENQ = range(6)
QUEUE_KINDS = ('normal', 'weak', 'engine')


def program_ops(k, n_progs):
    """Ops of test program k: an enqueue names a later program (the chain ends), at most one per program."""
    plain = st.one_of(st.just((PAUSE, 0, 0)), st.tuples(st.just(DELAY), st.integers(0, 2), st.just(0)),
                      st.tuples(st.just(CHAT), st.integers(0, len(IFACES) - 1), st.just(0)),
                      st.tuples(st.just(CLOSE), st.integers(0, 1), st.just(0)))
    later = list(range(k + 1, n_progs))
    if not later:
        return st.lists(plain, max_size=4)
    enq = st.tuples(st.just(ENQ), st.tuples(st.sampled_from(later), st.integers(0, 2)).map(lambda a: a[0] * 4 + a[1]),
                    st.integers(0, 2))
    return st.tuples(st.lists(plain, max_size=3), st.lists(enq, max_size=1), st.lists(plain, max_size=2)).map(
        lambda parts: parts[0] + parts[1] + parts[2])


@st.composite
def runner_case(draw):
    n = draw(st.integers(1, 6))
    progs = [draw(program_ops(k, n)) for k in range(n)]
    timers = draw(st.lists(st.tuples(st.integers(1, 5), st.integers(0, n - 1)), max_size=3))
    initial = draw(st.lists(st.tuples(st.integers(0, 2), st.integers(0, n - 1), st.integers(0, 3)), max_size=4))
    between = draw(st.lists(st.one_of(st.just(('none',)), st.just(('resume',)), st.just(('close',)),
                                      st.tuples(st.just('enqueue'), st.integers(0, 2), st.integers(0, n - 1),
                                                st.integers(0, 2))), min_size=1, max_size=24))
    return progs, timers, initial, between


@settings(deadline=None, max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(runner_case())
def test_runner_queues_and_timers_equal_the_oracle(lib, ids, tiny, case):
    """execute / _run / resume (engine.py:1404-1440) with protect, delay and pausebutton, the queue, weak and engine
    passes (unlink-then-execute and execute-then-unlink, a weak clear inside a pass), timers in login order, and the
    between-cycle resume, close_modal and enqueue the agent makes, tick by tick over the whole record."""
    P, strings = tiny
    progs, timers, initial, between = case
    canon = Canon(ids, strings)
    for k, ops in enumerate(progs):   # a chat op opens the interface of that string id in C
        flat = np.asarray([v for op, a, b in ops for v in (op, strings.id(IFACES[a]) if op == CHAT else a, b)]
                          + [END, 0, 0], np.int32)
        lib.t_prog(k, len(ops) + 1, ptr(flat))
    pair = Pair(lib, P, TinyTables())
    w, py, p = pair.w, pair.py, pair.p

    def gen(k, script):
        box = [0]

        def body():
            for op, a, b in progs[k]:
                if op == CHAT:
                    p.open_chat(IFACES[a])
                if op in (PAUSE, CHAT):
                    box[0] += 1
                    yield ('pausebutton',)
                elif op == DELAY:
                    box[0] += 1
                    yield ('delay', a)
                elif op == CLOSE:
                    p.close_modal(bool(a))
                elif op == ENQ:
                    p.enqueue(f'prog{a >> 2}', factory(a >> 2), kind=QUEUE_KINDS[a & 3], delay=b)
        g = body()
        canon.frames[g] = (canon.script(script), box)
        return g

    def factory(k):
        return lambda: gen(k, 'RS_S_PICKUP_OBJ')
    try:
        for i, (interval, k) in enumerate(timers):
            name = TIMERS[i]
            p.set_timer(name, interval, (lambda k=k, name=name: gen(k, f'RS_S_{name.upper()}')))
            lib.t_timer_prog(i, k)
            lib.t_set_timer(w, i, interval, py.tick, canon.script(f'RS_S_{name.upper()}'))
        for kind, k, delay in initial:
            p.enqueue(f'prog{k}', factory(k), kind=QUEUE_KINDS[kind], delay=delay)
            assert lib.t_enqueue(w, kind, k, delay) == 0
        assert_same(lib, w, py, canon, 'start')
        for step, action in enumerate(between):
            if action[0] == 'resume' and p.script is not None and p.script[1] == 'pausebutton':
                py.resume(p)
                assert lib.t_resume(w) == 0
            elif action[0] == 'close':
                p.close_modal()
                lib.t_close_modal(w, 1)
            elif action[0] == 'enqueue':
                _, kind, k, delay = action
                p.enqueue(f'prog{k}', factory(k), kind=QUEUE_KINDS[kind], delay=delay)
                assert lib.t_enqueue(w, kind, k, delay) == 0
            assert_same(lib, w, py, canon, f'step {step} {action} (between cycles)')
            assert lib.t_cycle(w) == 0
            py.cycle()
            assert_same(lib, w, py, canon, f'step {step} {action} (after the cycle)')
    finally:
        pair.close()


# ---- a compact pool layout: movement, locs, interaction ---------------------------------------------------------
@pytest.fixture(scope='module')
def base():
    if not ARCHIVE.exists() or not POOL.exists():
        pytest.skip('archive or compact pool absent (logs/sim/data, logs/sim/world; docs/RUNNING.md)')
    from sim.workers import load_tables
    return load_tables(ARCHIVE)


def pool_layout(index):
    """One layout of the compact pool, its Layout fields only (the derived trees are not read)."""
    from sim import world
    names = [f.name for f in dc_fields(world.Layout)]
    row = {}
    with np.load(POOL) as data:
        for name in names:
            value = data[name][index]
            if name in world.PADDED:
                n = int(data[f'n_{name}'][index])
                value = value[:n] if world.PADDED[name] == 0 else value[:, :n]
            row[name] = value
    return world.Layout(**{n: int(row[n]) if n in ('seed', 'attempt') else row[n] for n in names})


class Layouts:
    """A pen layout of the pool as both sides see it: the oracle's LayoutTables and a pack of the same layout, with
    every loc type's size and triggers (RS_S_UNKNOWN wherever the oracle has a script, so C cannot silently skip one)
    and the messages default_op and the reach refusal write."""
    def __init__(self, lib, ids, base, path):
        from sim import world
        index = next(k for k in range(8) if pool_layout(k).gate[0] >= 0)
        self.layout = pool_layout(index)
        self.tables = world.LayoutTables(base, self.layout)
        t, self.strings = self.tables, Strings()
        W, H = self.layout.flags.shape
        types = sorted({int(v) for v in self.layout.locs[:, 0]} | {int(v) for v in self.layout.open_locs[:, 0]})
        self.types = types
        locs = record('RsLocType', max(t.loc_types) + 1)
        for k in types:
            lt = t.loc_types[k]
            row = locs[k]
            row['present'], row['width'], row['length'] = 1, lt['width'], lt['length']
            row['debugname'] = self.strings.id(lt['debugname'])
            row['next_loc_stage'] = lt['params'].get('next_loc_stage', RS_NONE)
            for slot in range(12):
                op = APLOC1 + slot if slot < 6 else APLOC1 + slot - 6 + engine.AP_TO_OP
                name = lt['triggers'].get(engine._trigger_name(t, op))
                row['trig_script'][slot] = ids['script']['RS_S_UNKNOWN'] if name else 0
                row['trig_name'][slot] = self.strings.id(name) if name else 0
        npc_types = sorted({int(v) for v in self.layout.npcs[:, 1]})
        npcs = record('RsNpcType', max(t.npc_types) + 1)
        for k in npc_types:
            nt, row = t.npc_types[k], npcs[k]
            fill(row, present=1, size=nt['size'], defaultmode=nt['defaultmode'], timer=nt['timer'],
                 huntmode=nt['huntmode'], huntrange=nt['huntrange'], stats=nt['stats'],
                 debugname=self.strings.id(nt['debugname']),
                 combat_xp_multiplier=nt['params'].get('combat_xp_multiplier',
                                                       t.meta['param_defaults']['combat_xp_multiplier']))
            for slot in range(12):
                op = engine.APNPC1 + slot if slot < 6 else engine.APNPC1 + slot - 6 + engine.AP_TO_OP
                name = nt['triggers'].get(engine._trigger_name(t, op))
                row['trig_script'][slot] = ids['script']['RS_S_UNKNOWN'] if name else 0
                row['trig_name'][slot] = self.strings.id(name) if name else 0
        messages = [(1000, k, f"No trigger for [oplocu,{t.loc_types[k]['debugname']}]") for k in types]
        messages += [(1001, k, f"No trigger for [opnpcu,{t.npc_types[k]['debugname']}]") for k in npc_types]
        messages += [(0, 0, t.meta['content']['texts']['dm_default']), (1002, 0, "I can't reach that!")]
        src = np.full(W * H, -1)
        for k, (x, z, _) in enumerate(t.meta['sources']):
            src[(x - ORIGIN[0]) * H + (z - ORIGIN[1])] = k
        self.path = write_pack(path, ids, self.strings, loc_types=locs, npc_types=npcs, messages=messages,
                               layout=dict(flags=self.layout.flags.astype(np.int32), tile_src=src, locs=self.layout.locs,
                                           npcs=self.layout.npcs))
        self.P = open_pack(lib, self.path)


@pytest.fixture(scope='module')
def pen(lib, ids, base):
    out = ROOT / 'logs' / 'native' / 'tests'
    pen = Layouts(lib, ids, base, out / f'world-pen-{hashlib.sha256(str(POOL).encode()).hexdigest()[:8]}.pack')
    yield pen
    lib.rs_pack_close(pen.P)
    pen.path.unlink(missing_ok=True)


class Oracle:
    """The callbacks the harness's paths doubles call: masks from sim.paths, reach and sight from the oracle world."""
    def __init__(self, lib, tables):
        self.world = None

        def masks(type_id, shape, angle, x, z, out, cap):
            rows = paths.collision_masks(x, z, shape, angle, tables.loc_types[type_id])
            assert len(rows) <= cap
            for k, (a, b, m) in enumerate(rows):
                out[3 * k], out[3 * k + 1], out[3 * k + 2] = a, b, m
            return len(rows)

        def reach(x, z):
            try:
                tiles = self.world.entity_reaching(0, x, z)
            except paths.OutsideTables:
                return -1
            return sum(1 << b for b, (dx, dz) in enumerate(paths.NEIGHBOURS) if (x + dx, z + dz) in tiles)

        def locreach(index, out, cap):
            loc = next(loc for loc in self.locs() if loc.index == index)
            tiles = self.world.loc_reaching(loc)
            assert len(tiles) <= cap
            for k, (a, b) in enumerate(tiles):
                out[2 * k], out[2 * k + 1] = a, b
            return len(tiles)

        def los(ax, az, bx, bz):
            return int(bool(self.world.line_of_sight(0, ax, az, bx, bz)))
        self.callbacks = (MASKS(masks), REACH(reach), LOCREACH(locreach), LOS(los))
        lib.t_callbacks(*self.callbacks)

    def locs(self):
        return world_locs(self.world)


def world_locs(w):
    """The oracle's map locs, then its added locs still on their tile (rs.h: an added loc leaves loc_at when removed)."""
    return list(w.locs) + [loc for row in w.loc_at.values() for loc in row if not loc.static]


def pen_pair(lib, pen, data, start, npcs=False):
    """A pair on the pen layout; the world seed and start tick vary, so keyed draws (900-902) vary too."""
    pair = Pair(lib, pen.P, pen.tables, player=start, npcs=npcs, seed=data.draw(st.integers(0, 2 ** 24 - 1)),
                start_tick=data.draw(st.integers(6, 400)))
    oracle = Oracle(lib, pen.tables)
    oracle.world = pair.py
    return pair, oracle


def start_tiles(pen):
    return [(ORIGIN[0] + int(x), ORIGIN[1] + int(z)) for x, z in np.argwhere(pen.layout.start)]


def queue_both(lib, pair, path):
    pair.p.queue_waypoints(path)
    xy = np.asarray([v for step in path for v in step] or [0], np.int16)
    lib.t_queue(pair.w, len(path), ptr(xy))


def cycle_both(lib, pair, canon, note):
    code = lib.t_cycle(pair.w)
    try:
        pair.py.cycle()
    except engine.Unsupported as error:
        assert code == 1 and str(c_error(lib, pair.w, canon.strings)) == str(error), note
        return False
    assert code == 0, (note, str(c_error(lib, pair.w, canon.strings)))
    assert_same(lib, pair.w, pair.py, canon, note)
    return True


def twin(lib, pair, strings, oracle, native_op):
    """An op on both sides: the oracle's value and C's code (>= 0), or None after the same Unsupported on both."""
    try:
        value = oracle()
    except engine.Unsupported as error:
        assert native_op() == -1 and str(c_error(lib, pair.w, strings)) == str(error)
        return None
    code = native_op()
    assert code >= 0, str(c_error(lib, pair.w, strings))
    return value, code


def flags_equal(lib, pair, pen):
    W, H = pen.layout.flags.shape
    out = np.zeros(W * H, np.int32)
    lib.t_flags_all(pair.w, ptr(out))
    assert out.tolist() == pair.py._flags


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(st.data())
def test_player_walks_recorded_paths_as_the_oracle(lib, ids, pen, data):
    """The player phase without scripts: recorded paths (the oracle's path_to_tile) queued on both sides, then
    process_movement one step a tick (diagonal, then x, then z; a reached waypoint popped), the source check, origin
    rebuilds, cleanup and occupancy, tick by tick; re-planned mid-walk. can_travel and naive_path agree tile by tile."""
    starts = start_tiles(pen)
    start = data.draw(st.sampled_from(starts))
    pair, _ = pen_pair(lib, pen, data, start)
    canon = Canon(ids, pen.strings)
    try:
        sources = [(x, z) for x, z, _ in pen.tables.meta['sources']]
        for leg in range(data.draw(st.integers(1, 3))):
            goal = data.draw(st.sampled_from(sources))
            path = pair.py.path(lambda: pair.py.path_to_tile(0, pair.p.x, pair.p.z, *goal))
            queue_both(lib, pair, path)
            for tick in range(data.draw(st.integers(1, 40))):
                assert cycle_both(lib, pair, canon, f'leg {leg} tick {tick}')
        x, z = pair.p.x, pair.p.z
        for dx, dz in ((dx, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)):
            for extra in (0, engine.FLAG_NPC):
                assert lib.t_can_travel(pair.w, x, z, dx, dz, extra) == pair.py.can_travel(0, x, z, dx, dz, 1, extra, 0)
        for _ in range(10):
            sx, sz = data.draw(st.sampled_from(sources))
            dx, dz = (sx, sz) if data.draw(st.booleans()) else data.draw(st.sampled_from(sources))
            out = np.zeros(2, np.int16)
            assert lib.t_naive(pair.w, sx, sz, dx, dz, 1, 1, ptr(out)) == 0
            assert tuple(out.tolist()) == pair.py.naive_path(0, sx, sz, dx, dz, 1, 1)[0]
    finally:
        pair.close()


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(st.data())
def test_loc_lifecycle_zones_and_events_equal_the_oracle(lib, ids, pen, data):
    """change_loc, remove_loc, revert_loc, set_lifecycle and LOC_ADD (the gate's open stages and arbitrary types on loc
    tiles) between cycles, then process_zones' countdowns (an added loc removed, a changed map loc reverted, a removed
    one added back), with the zone orders, loc_events order, the live flags and the changed tiles, tick by tick;
    the locs queries (locs_near, active_locs_at, loc_at, locs_in_zone) agree after each step; the player walks
    meanwhile."""
    pair, _ = pen_pair(lib, pen, data, data.draw(st.sampled_from(start_tiles(pen))))
    canon = Canon(ids, pen.strings)
    try:
        sources = [(x, z) for x, z, _ in pen.tables.meta['sources']]
        stage = [tuple(int(v) for v in row) for row in pen.layout.open_locs]
        for step in range(data.draw(st.integers(1, 16))):
            live = world_locs(pair.py)
            op = data.draw(st.sampled_from(['change', 'remove', 'revert', 'lifecycle', 'add', 'walk', 'none']))
            duration = data.draw(st.integers(-1, 12))
            if op in ('change', 'remove', 'revert', 'lifecycle'):
                tracked = [loc for loc in pair.py.loc_events if loc in live]   # re-tracking moves a loc to the tail
                loc = data.draw(st.sampled_from(tracked if tracked and data.draw(st.booleans()) else live))
                slot = lib.t_loc_slot(pair.w, loc.index)
                assert slot >= 0
                if op == 'revert' and not (loc.static and loc.active and loc.is_changed()):
                    continue                      # process_zones' only revert
                type_id = data.draw(st.sampled_from(pen.types))
                oracle = dict(change=lambda: pair.py.change_loc(loc, type_id, loc.shape, loc.angle, duration),
                              remove=lambda: pair.py.remove_loc(loc, duration), revert=lambda: pair.py.revert_loc(loc),
                              lifecycle=lambda: pair.py.set_lifecycle(loc, duration))[op]
                code = ('change', 'remove', 'revert', 'lifecycle').index(op)
                if twin(lib, pair, pen.strings, oracle,
                        lambda: lib.t_loc_op(pair.w, code, slot, type_id, loc.shape, loc.angle, duration)) is None:
                    return
            elif op == 'add':
                if data.draw(st.booleans()) and stage:
                    type_id, x, z, shape, angle = data.draw(st.sampled_from(stage))
                    x, z = ORIGIN[0] + x, ORIGIN[1] + z
                else:
                    loc = data.draw(st.sampled_from(live))
                    type_id, x, z, shape, angle = data.draw(st.sampled_from(pen.types)), loc.x, loc.z, loc.shape, loc.angle
                done = twin(lib, pair, pen.strings, lambda: pair.py.loc_add(0, x, z, type_id, angle, shape, duration),
                            lambda: lib.t_loc_add(pair.w, x, z, type_id, angle, shape, duration))
                if done is None:
                    return
                assert lib.t_loc_slot(pair.w, done[0].index) == done[1]
            elif op == 'walk':
                goal = data.draw(st.sampled_from(sources))
                path = pair.py.path(lambda: pair.py.path_to_tile(0, pair.p.x, pair.p.z, *goal))
                queue_both(lib, pair, path)
            assert_same(lib, pair.w, pair.py, canon, f'step {step} {op}')
            flags_equal(lib, pair, pen)
            for tick in range(data.draw(st.integers(0, 4))):
                if not cycle_both(lib, pair, canon, f'step {step} {op} tick {tick}'):
                    return
                flags_equal(lib, pair, pen)
            x, z = data.draw(st.sampled_from(sources))
            radius = data.draw(st.integers(0, 20))
            out = np.zeros(64, np.int32)
            n = lib.t_locs_query(pair.w, 0, x, z, radius, ptr(out))
            assert out[:n].tolist() == [loc.index for loc in pair.py.locs_near(0, x, z, radius)]
            loc = data.draw(st.sampled_from(live))
            for which, expected in ((1, pair.py.active_locs_at(0, loc.x, loc.z)),
                                    (2, pair.py.loc_at.get((0, loc.x, loc.z), [])),
                                    (3, pair.py.locs_in_zone(0, loc.x, loc.z))):
                n = lib.t_locs_query(pair.w, which, loc.x, loc.z, 0, ptr(out))
                assert out[:n].tolist() == [v.index for v in expected], which
    finally:
        pair.close()


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(st.data())
def test_interaction_without_scripts_equals_the_oracle(lib, ids, pen, data):
    """process_interaction on a loc target whose op has neither an AP nor an OP trigger: validation, the walk, the AP
    test (approach range and sight, ap_range -1), the OP-scenery rule (no steps this tick), default_op (the debug line
    and dm_default for a use, Unsupported for another op) and "I can't reach that!", tick by tick."""
    pair, _ = pen_pair(lib, pen, data, data.draw(st.sampled_from(start_tiles(pen))))
    canon = Canon(ids, pen.strings)
    try:
        t = pen.tables
        candidates = [(loc, op) for loc in pair.py.locs for op in (APLOCU, APLOC1, APLOC1 + 1, APLOC1 + 2)
                      if not pair.py.trigger(loc, op) and not pair.py.trigger(loc, op + engine.AP_TO_OP)]
        assert candidates
        for attempt in range(data.draw(st.integers(1, 3))):
            loc, op = data.draw(st.sampled_from(candidates))
            assert pair.p.set_interaction(loc, op)
            assert lib.t_set_interaction_loc(pair.w, lib.t_loc_slot(pair.w, loc.index), op) == 1
            if data.draw(st.booleans()):   # the walk act-v3's interact_loc sends: the client's route to the loc
                lt = t.loc_types[loc.type]
                arrival = paths.loc_arrival(loc.shape, loc.angle, lt['width'], lt['length'], lt['forceapproach'])
                route = paths.client_route(pair.py.player_scene(pair.p), pair.p.x, pair.p.z, (loc.x, loc.z), arrival)
                queue_both(lib, pair, [] if route is None else list(route))
            assert_same(lib, pair.w, pair.py, canon, f'attempt {attempt} set')
            for tick in range(data.draw(st.integers(1, 30))):
                if not cycle_both(lib, pair, canon, f'attempt {attempt} op {op} {t.loc_types[loc.type]["debugname"]} '
                                                    f'tick {tick}'):
                    return
    finally:
        pair.close()


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_npc_targets_without_scripts_equal_the_oracle(lib, ids, pen, monkeypatch, data):
    """process_interaction on the layout's npcs (World init's npcs in nid order, their [ai_spawn] and occupancy) for
    ops with neither an AP nor an OP trigger, the npcs standing still (npc turns are npc.c's: both sides skip them):
    path_to_pathing_target's naive repath and its random step from under the npc (keys 900, 901, 902), the AP test,
    ap_range -1, default_op (opnpcu's debug line and dm_default, Unsupported for another op), tick by tick."""
    monkeypatch.setattr(World, 'npc_turn', lambda self, npc: None)
    monkeypatch.setattr(World, 'npc_lifecycle', lambda self, npc: None)
    spawns = [(ORIGIN[0] + int(x), ORIGIN[1] + int(z)) for _, _, x, z in pen.layout.npcs]   # in the pen, or under one
    pair, _ = pen_pair(lib, pen, data, data.draw(st.sampled_from(spawns if data.draw(st.booleans()) else start_tiles(pen))),
                       npcs=True)
    canon = Canon(ids, pen.strings)
    try:
        assert_same(lib, pair.w, pair.py, canon, 'init')
        candidates = [(n, op) for n in pair.py.npcs for op in (engine.APNPCU, *range(engine.APNPC1, engine.APNPC1 + 5))
                      if not pair.py.trigger(n, op) and not pair.py.trigger(n, op + engine.AP_TO_OP)]
        assert candidates
        under = [(n, op) for n, op in candidates if (n.x, n.z) == (pair.p.x, pair.p.z)]
        uses = [(n, op) for n, op in candidates if op == engine.APNPCU]   # the op default_op answers without raising
        for attempt in range(data.draw(st.integers(1, 3))):
            pool = under if under and attempt == 0 else uses if uses and data.draw(st.booleans()) else candidates
            npc, op = data.draw(st.sampled_from(pool))
            assert pair.p.set_interaction(npc, op)
            assert lib.t_set_interaction_npc(pair.w, npc.nid, op) == 1
            for tick in range(data.draw(st.integers(1, 30))):
                if not cycle_both(lib, pair, canon, f'attempt {attempt} op {op} npc {npc.nid} tick {tick}'):
                    return
    finally:
        pair.close()
