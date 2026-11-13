"""The native npcs and combat (sim/native/npc.c, combat.c) against the oracle they port: sim/engine.py's
npcs (turns, lifecycle and respawn, hunting, wander, escape, the ai modes, the spot's move), the combat procs, the
death scripts run by sim/rs2.py through DeathContext, Player.applyDamage and the ground items, on the real exported
pack (sim/tests PACK) and the train-8 compact pool (the port's design, section 5).

The C side is a harness library: every sim/native/*.c but module.c, built with the extension's flags. The player's
scripts (the combat ones through rs_combat_script_step, CHANGESTAT, the stat timers, LEVELUP with its dialogs) and
P_OPNPC are scripts.c's, and the oracle runs its own. The agent's upkeep after each cycle (agent.py
keep_alive's observe_npcs, capture_text's drains) is done alike on both sides by the test. Everything else is the
port itself: world.c, paths.c on the pack's tables, npc.c and combat.c. After every
cycle both sides are compared through the canonical record (rs_world_canon; sim/native_harness.py Canon's world fields
for the oracle) and the multiset of that cycle's draws (key tuples and 24-bit values); a raise is compared by class and
text.

These read the archive, pool and pack in logs/ (docs/RUNNING.md) and skip without them. Failures report ids,
field names and numbers, never the archive's strings; Hypothesis draws only numbers (indices into tables the test
holds), so a failing example prints no private data."""
from __future__ import annotations

import ctypes
from dataclasses import fields as dc_fields
import hashlib
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

from sim import agent, combat, engine, native, native_harness as nh
from sim.engine import APNPC1, SKILLS, Player, World
from sim.tests import PACK

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / 'logs'
ARCHIVE = LOGS / 'sim' / 'data' / 'draynor-nc1.npz'
POOL = LOGS / 'sim' / 'world' / 'train-8-nc1-derived.npz'
ORIGIN = (2880, 3328)
NAME = 'wd1a0'
EXCLUSION_BITS = dict(random_events=1, clue_drops=2, loot_drops=4)   # rs.h RS_EXCLUDE_*
SLOW = dict(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])

HARNESS_C = r'''
#include <stdlib.h>
#include <string.h>
#include "rs.h"

/* ---- the driver --------------------------------------------------------------------------------------------- */
#define T_DRAWS (1 << 14)
static RsDraw t_draw_buf[T_DRAWS];
static RsTrace t_trace_buf;
World *t_world_new(void) { return calloc(1, sizeof(World)); }
void t_world_free(World *w) { free(w); }
int t_layout(const RsPack *P, int k, int64_t *seed) {
    if (k < 0 || k >= P->n_compact) return -1;
    *seed = P->layouts[P->compact0 + k].rec->seed;
    return P->compact0 + k;
}
int t_world_init(World *w, const RsPack *P, int layout, int64_t seed, int32_t tick, const int32_t *spawns, int n,
                 int excl) {
    int code = rs_world_init(w, P, &P->layouts[layout], seed, tick, n < 0 ? NULL : (const RsNpcSpawn *)spawns,
                             n < 0 ? 0 : n, (uint8_t)excl, 25);
    t_trace_buf = (RsTrace){.draws_cap = T_DRAWS, .draws = t_draw_buf};
    w->trace = &t_trace_buf;
    return code;
}
void t_err(const World *w, int32_t *out) {
    out[0] = w->err.code, out[1] = w->err.exc, out[2] = w->err.reason;
    for (int i = 0; i < RS_ERR_ARGS; i++) out[3 + i] = w->err.arg[i];
}
const char *t_err_text(const World *w) { return w->err.text; }
void t_place_player(World *w, int32_t x, int32_t z) {
    rs_player_init(w);
    w->p.x = (int16_t)x, w->p.z = (int16_t)z;
    w->has_player = 1;
    rs_occupy(w, 0, x, z, 1, 1);
}
int t_cycle(World *w) { return rs_world_cycle(w); }
int t_canon(const World *w, int64_t *buf, int cap) {
    int n = 0;
    RsErr e;
    return rs_world_canon(w, buf, cap, &n, &e) ? -1 : n;
}
/* the draws since the last call: per draw n, m, keys[RS_DRAW_KEYS]; -1 when the buffer overflowed */
int t_draws(int64_t *out, int cap) {
    int n = t_trace_buf.n_draws;
    if (t_trace_buf.truncated || n > cap) return -1;
    for (int i = 0; i < n; i++) {
        const RsDraw *d = &t_draw_buf[i];
        out[10 * i] = d->n, out[10 * i + 1] = d->m;
        for (int k = 0; k < RS_DRAW_KEYS; k++) out[10 * i + 2 + k] = d->key[k];
    }
    t_trace_buf.n_draws = 0;
    return n;
}
/* agent.py keep_alive's observe_npcs (254-260) and capture_text's drains (266-277), done alike on the oracle side */
void t_upkeep(World *w) {
    const Player *p = &w->p;
    for (int i = 0; w->has_player && i < w->n_npcs; i++) {
        Npc *n = &w->npcs[i];
        int dx = abs(p->x - n->x), dz = abs(p->z - n->z);
        n->observers = n->active && !n->tele && n->level == p->level && (dx > dz ? dx : dz) <= 15;
    }
    w->chat.captured = w->chat.n;
    w->text.n = 0;
}
void t_set_stats(World *w, const int16_t *levels, const int16_t *base, const int32_t *stats) {
    memcpy(w->p.levels, levels, sizeof w->p.levels);
    memcpy(w->p.base_levels, base, sizeof w->p.base_levels);
    memcpy(w->p.stats, stats, sizeof w->p.stats);
}
void t_set_slot(World *w, int which, int slot, int32_t id, int32_t count) {
    Slot *s = which == 0 ? w->p.inv : which == 1 ? w->p.worn : w->p.bank;
    s[slot] = (Slot){id, count};
}
void t_set_varp(World *w, int k, int32_t v) { w->p.varp[k] = v; }
void t_varps(const World *w, int32_t *out) { memcpy(out, w->p.varp, sizeof w->p.varp); }
void t_set_player(World *w, int protect, int delayed, int32_t delayed_until, int modal) {
    w->p.protect = (uint8_t)protect, w->p.delayed = (uint8_t)delayed, w->p.delayed_until = delayed_until;
    w->p.modal = (uint8_t)modal;
}
void t_set_timer(World *w, int i, int32_t interval, int32_t clock, int16_t script) {
    w->p.timer[i] = (Timer){interval, clock, script, 0, 0};
    if (w->p.n_timers < i + 1) w->p.n_timers = (uint8_t)(i + 1);
}
int t_combat_stat(World *w) { return rs_player_combat_stat(w); }
/* combat_level of each row of bases ([n][7]: attack, defence, strength, hitpoints, ranged, prayer, magic) */
void t_combat_levels(World *w, const int16_t *bases, int n, int32_t *out) {
    for (int i = 0; i < n; i++) {
        memcpy(w->p.base_levels, bases + 7 * i, 7 * sizeof(int16_t));
        out[i] = rs_combat_level(w);
    }
}
int t_set_interaction_npc(World *w, int32_t nid, int32_t op) {
    for (int i = 0; i < w->n_npcs; i++)
        if (w->npcs[i].nid == nid) return rs_set_interaction(w, rs_ref_npc(w, i), op);
    return -1;
}
void t_queue(World *w, int n, const int16_t *xy) { rs_queue_waypoints(w, (const int16_t (*)[2])xy, n); }
int t_npc_slot(const World *w, int32_t nid) {
    for (int i = 0; i < w->n_npcs; i++) if (w->npcs[i].nid == nid) return i;
    return -1;
}
void t_npc_set(World *w, int slot, int has_hero, int32_t hero_damage, int aggressive, int32_t last_movement,
               int32_t hp) {
    Npc *n = &w->npcs[slot];
    n->has_hero = (uint8_t)has_hero, n->hero_damage = has_hero ? hero_damage : 0;
    n->v_aggressive_player = (uint8_t)aggressive, n->last_movement = last_movement;
    n->levels[RS_NPC_HP] = (int16_t)hp;
}
int t_npc_enqueue(World *w, int slot, int16_t id, int32_t arg, int32_t delay, int source) {
    return rs_npc_enqueue(w, &w->npcs[slot], id, arg, delay, source);
}
/* rs_npc_validate_target for an npc placed at (x, z) in `mode` targeting the player (or not), the player at (px, pz)
   (no occupancy is read) */
int t_validate(World *w, int slot, int32_t x, int32_t z, int32_t mode, int target, int32_t px, int32_t pz) {
    Npc *n = &w->npcs[slot];
    n->x = (int16_t)x, n->z = (int16_t)z, n->mode = (int16_t)mode, n->target_player = (uint8_t)target;
    w->p.x = (int16_t)px, w->p.z = (int16_t)pz;
    return rs_npc_validate_target(w, n);
}
/* events [from, n) as native_harness.Canon.event rows (kind, tick, a8, b8, has_source, obj, count, v0..v3) */
int t_events(const World *w, int from, int64_t *out, int cap) {
    int k = 0;
    for (int i = from; i < w->n_events && k < cap; i++, k++) {
        const RsEvent *e = &w->ev[i];
        int64_t row[11] = {e->kind, e->tick, e->a8, e->b8, e->has_source, e->obj, e->count, e->v0, e->v1, e->v2, e->v3};
        memcpy(out + 11 * k, row, sizeof row);
    }
    return k;
}
/* chat lines [from, n) still in the ring as (tick, text string id) */
int t_chat(const World *w, int64_t from, int64_t *out, int cap) {
    int k = 0;
    for (int64_t i = from; i < w->chat.n && k < cap; i++, k++) {
        const RsChatLine *c = &w->chat.ring[i % RS_CHAT_RING];
        out[2 * k] = c->tick, out[2 * k + 1] = w->P->msgs[c->msg].text;
    }
    return k;
}
int t_obj_add(World *w, int32_t x, int32_t z, int32_t obj, int32_t count, int32_t duration) {
    return rs_obj_add(w, 0, x, z, obj, count, duration);
}
static RsRef t_held[16];
void t_obj_hold(World *w, int k, int at) { t_held[k] = rs_ref_obj(w, w->obj_order[at]); }
int t_take_held(World *w, int k) { return rs_take_obj(w, t_held[k]); }
'''

STUBBED = {}  # no entry point is replaced: scripts.c is the port's own
_C = dict(p=ctypes.c_void_p, i=ctypes.c_int, i16=ctypes.c_int16, i32=ctypes.c_int32, i64=ctypes.c_int64,
          s=ctypes.c_char_p)
SIGNATURES = {
    'rs_pack_open': ('i', 's', 'p', 'p'), 'rs_pack_close': (None, 'p'),
    't_world_new': ('p',), 't_world_free': (None, 'p'), 't_layout': ('i', 'p', 'i', 'p'),
    't_world_init': ('i', 'p', 'p', 'i', 'i64', 'i32', 'p', 'i', 'i'), 't_err': (None, 'p', 'p'),
    't_err_text': ('s', 'p'), 't_place_player': (None, 'p', 'i32', 'i32'), 't_cycle': ('i', 'p'),
    't_canon': ('i', 'p', 'p', 'i'), 't_draws': ('i', 'p', 'i'), 't_upkeep': (None, 'p'),
    't_set_stats': (None, 'p', 'p', 'p', 'p'), 't_set_slot': (None, 'p', 'i', 'i', 'i32', 'i32'),
    't_set_varp': (None, 'p', 'i', 'i32'), 't_varps': (None, 'p', 'p'), 't_set_player': (None, 'p', 'i', 'i', 'i32', 'i'),
    't_set_timer': (None, 'p', 'i', 'i32', 'i32', 'i16'), 't_combat_stat': ('i', 'p'),
    't_combat_levels': (None, 'p', 'p', 'i', 'p'), 't_set_interaction_npc': ('i', 'p', 'i32', 'i32'),
    't_queue': (None, 'p', 'i', 'p'), 't_npc_slot': ('i', 'p', 'i32'),
    't_npc_set': (None, 'p', 'i', 'i', 'i32', 'i', 'i32', 'i32'), 't_npc_enqueue': ('i', 'p', 'i', 'i16', 'i32', 'i32', 'i'),
    't_obj_add': ('i', 'p', 'i32', 'i32', 'i32', 'i32', 'i32'), 't_obj_hold': (None, 'p', 'i', 'i'),
    't_take_held': ('i', 'p', 'i'), 't_events': ('i', 'p', 'i', 'p', 'i'), 't_chat': ('i', 'p', 'i64', 'p', 'i'), 't_validate': ('i', 'p', 'i', 'i32', 'i32', 'i32', 'i', 'i32', 'i32'),
}


def build_harness(out_dir):
    """Every sim/native/*.c but module.c (STUBBED entry points renamed away) and the driver, as one library."""
    header = native.generated_header()
    digest = hashlib.sha256(header.encode() + HARNESS_C.encode() + ' '.join(native.FLAGS).encode())
    for path in sorted(native.SOURCE.glob('*.[ch]')):
        digest.update(path.name.encode() + path.read_bytes())
    lib = out_dir / f'npc-harness-{digest.hexdigest()[:16]}.so'
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
        command = ['cc', *native.FLAGS, *renames, '-I', str(include), '-I', str(native.SOURCE), '-c', str(src), '-o',
                   str(obj)]
        done = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr[-3000:]
        objects.append(str(obj))
    done = subprocess.run(['cc', '-shared', *objects, '-o', str(lib), '-lz'], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-3000:]
    return lib


def ptr(array):
    return ctypes.c_void_p(array.ctypes.data)


class RsErr(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('exc', ctypes.c_uint8), ('reason', ctypes.c_uint16),
                ('arg', ctypes.c_int32 * 6), ('text', ctypes.c_char_p)]


@pytest.fixture(scope='module')
def lib(tmp_path_factory):
    c = ctypes.CDLL(str(build_harness(tmp_path_factory.mktemp('npc-harness'))))
    for name, (res, *args) in SIGNATURES.items():
        f = getattr(c, name)
        f.restype = _C[res] if isinstance(res, str) else res
        f.argtypes = [_C[a] if isinstance(a, str) else a for a in args]
    return c


# ---- the oracle's record (native_harness.Canon's world fields) --------------------------------------------------
class WorldCanon(nh.Canon):
    """native_harness.Canon without the agent's and the episode's fields: the record rs_world_canon writes."""
    AGENT, TASK = SimpleNamespace(name=NAME), SimpleNamespace(active=False)

    def agent(self, put, world, agent):
        pass

    def world(self, world):
        out = []
        for name, values in nh.decode(self(world, self.AGENT, self.TASK)).items():
            if name != 'RS_CF_EPISODE':
                out += [self.id[name], len(values), *values.tolist()]
        return out


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


class Env:
    """The archive's tables, the pack (C and Python views), the pool's layouts as LayoutTables, the record."""

    def __init__(self, lib):
        for path in (ARCHIVE, POOL, PACK):
            if not path.exists():
                pytest.skip(f'{path} is absent: these tests need the archive, pool and pack (docs/RUNNING.md)')
        from sim.workers import load_tables
        self.lib, self.base = lib, load_tables(ARCHIVE)
        self.pack = native.load().Pack(str(PACK))
        self.canon = WorldCanon(nh.PackStrings(self.pack))
        self.P, err = ctypes.c_void_p(), RsErr()
        assert lib.rs_pack_open(str(PACK).encode(), ctypes.byref(self.P), ctypes.byref(err)) == 0, err.text
        self._layouts = {}
        t = self.base
        self.ids = {v['debugname']: k for k, v in t.npc_types.items()}
        self.drop_types = sorted(k for k, v in t.npc_types.items() if v['triggers'].get('AI_QUEUE3')
                                 and k in {self.ids[n] for n in ('chicken', 'chicken_brown', 'goblin', 'mugger',
                                                                  '0_46_49_saltfish')})
        cats = t.meta['content']['categories']
        trail = engine.trail_categories(t)
        self.clue_objs = sorted(k for k, o in t.obj_types.items() if o['category'] in trail)
        self.wieldable = sorted(k for k, o in t.obj_types.items() if o['wearpos'] is not None)
        self.bows = sorted(k for k in self.wieldable if t.obj_types[k]['category'] in (cats['weapon_bow'],
                                                                                         cats['weapon_crossbow']))
        self.ammo = sorted(k for k, o in t.obj_types.items() if o['category'] in (cats['arrows'], cats['bolts']))
        self.weapons = sorted(t.obj_ids[name] for name in ('bronze_dagger', 'bronze_sword', 'bronze_axe',
                                                            'bronze_pickaxe') if name in t.obj_ids)
        self.inv_objs = self.program_objs()

    def program_objs(self):
        """The objs the death programs' inv_total reads (the constant loaded into its obj register), so death worlds
        can hold them."""
        m, c = native.load(), nh.header()[0]
        sections = m.constants()['sections']
        ops = np.frombuffer(self.pack.section(sections['DROP_OPS']), native.record_dtype('RsDropOp'))
        objs = set()
        for i, op in enumerate(ops):
            if op['op'] == c['DOP_INV_TOTAL']:
                const = next((o for o in ops[i - 1::-1] if o['op'] == c['DOP_CONST'] and o['a'] == op['c']), None)
                if const is not None and const['b'] in self.base.obj_types:
                    objs.add(int(const['b']))
        return sorted(objs)

    def layout(self, k):
        if k not in self._layouts:
            from sim import world
            layout = pool_layout(k)
            seed = ctypes.c_int64()
            rec = self.lib.t_layout(self.P, k, ctypes.byref(seed))
            assert rec >= 0 and seed.value == layout.seed, f'pack compact layout {k} is not pool layout {k}'
            self._layouts[k] = (layout, world.LayoutTables(self.base, layout), rec)
        return self._layouts[k]

    def close(self):
        self.lib.rs_pack_close(self.P)


@pytest.fixture(scope='module')
def env(lib):
    e = Env(lib)
    yield e
    e.close()


# ---- a pair of worlds ---------------------------------------------------------------------------------------------
def c_error(env, w):
    out = np.zeros(9, np.int32)
    env.lib.t_err(w, ptr(out))
    text = env.lib.t_err_text(w)
    error = native.load().Error(*(int(v) for v in out[:3]), tuple(int(v) for v in out[3:]),
                                text.decode() if text else None, 0)
    return native.as_python(error, env.pack)


class Pair:
    """A C world and an oracle world started alike on compact layout k: the layout's npcs or `spawns` (World's npcs
    dicts), exclusions, seed and start tick; the oracle's Rng traced."""

    def __init__(self, env, k, seed, tick, spawns=None, exclusions=()):
        self.env, lib = env, env.lib
        layout, self.tables, rec = env.layout(k)
        self.layout = layout
        self.w = lib.t_world_new()
        bits = sum(EXCLUSION_BITS[name] for name in exclusions)
        if spawns is None:
            code = lib.t_world_init(self.w, env.P, rec, seed, tick, None, -1, bits)
        else:
            rows = np.asarray([[s['nid'], s['type'], s['x'], s['z'], s['level'], s['size']] for s in spawns] or [[0] * 6],
                              np.int32)
            code = lib.t_world_init(self.w, env.P, rec, seed, tick, ptr(rows), len(spawns), bits)
        assert code == 0, str(c_error(env, self.w))
        self.py = World(self.tables, seed=seed, start_tick=tick, npcs=spawns, exclusions=frozenset(exclusions))
        self.draws = []
        self.py.rng = nh.TracedRng(self.py.rng, lambda keys, value: self.draws.append(
            (tuple(int(k) for k in keys), int(value * 16777216))))
        self.p = None
        self.events_seen = self.chat_seen = 0
        self.stats = dict(ticks=0, kills=0, escapes=0, respawns=0, hunts=0, deaths=0, drops=0, excluded=0, spots=0,
                          raised=None)

    def close(self):
        self.env.lib.t_world_free(self.w)

    def place(self, x, z):
        self.env.lib.t_place_player(self.w, x, z)
        p = self.p = Player(self.py, NAME)
        p.x, p.z = x, z
        self.py.player = p
        self.py._occupy(p, True)

    def kit(self, levels, weapon, worn, hp=None, com_mode=0):
        """Levels (and xp of those levels), a weapon carried or worn, com_mode, the [login,_] timers and
        player_combat_stat on both sides."""
        p, lib, w = self.p, self.env.lib, self.w
        for i, level in enumerate(levels):
            p.levels[i] = p.base_levels[i] = level
            p.stats[i] = engine.xp_for(level)
        if hp is not None:
            p.levels[engine.HITPOINTS] = hp
        set_stats(lib, w, p)
        if weapon is not None:
            inv = p.worn if worn else p.inv
            slot = engine.WEARPOS_RHAND if worn else 0
            inv.items[slot] = [weapon, 1]
            lib.t_set_slot(w, 1 if worn else 0, slot, weapon, 1)
        self.varp('com_mode', com_mode)
        for i, name in enumerate(('stat_regen', 'stat_boost_restore', 'health_regen')):   # [login,_]
            fn = getattr(agent, name)
            p.set_timer(name, 100, (lambda fn=fn: fn(self.py, p)))
            lib.t_set_timer(w, i, 100, self.py.tick, self.env.canon.ids[f'RS_S_{name.upper()}'])
        engine.player_combat_stat(self.py, p)
        assert lib.t_combat_stat(w) == 0

    def varp(self, name, value):
        self.p.varps[name] = value
        self.env.lib.t_set_varp(self.w, self.env.canon.varps.index(name), -1 if value is None else value)

    def attack(self, npc):
        assert self.p.set_interaction(npc, APNPC1 + 1)
        assert self.env.lib.t_set_interaction_npc(self.w, npc.nid, APNPC1 + 1) == 1

    def walk(self, goal):
        py = self.py
        path = py.path(lambda: py.path_to_tile(0, self.p.x, self.p.z, *goal))
        self.p.queue_waypoints(path)
        xy = np.asarray([v for step in path for v in step] or [0], np.int16)
        self.env.lib.t_queue(self.w, len(path), ptr(xy))

    def same(self, note):
        a = np.zeros(1 << 16, np.int64)
        n = self.env.lib.t_canon(self.w, ptr(a), len(a))
        assert n >= 0
        got, expected = a[:n].tolist(), self.env.canon.world(self.py)
        if got != expected:
            raise AssertionError(f'{note}: C vs oracle differ in ' + '; '.join(nh.field_diff(got, expected)))
        # the streams since the last comparison (the record holds their counts and last entries only)
        py, lib, canon = self.py, self.env.lib, self.env.canon
        new = py.events[self.events_seen:]
        buf = np.zeros(11 * max(len(new), 1), np.int64)
        k = lib.t_events(self.w, self.events_seen, ptr(buf), len(new))
        assert buf[:11 * k].reshape(k, 11).tolist() == [canon.event(ev, NAME) for ev in new], f'{note}: events differ'
        lines = py.chat_log[self.chat_seen:]
        buf = np.zeros(2 * max(len(lines), 1), np.int64)
        k = lib.t_chat(self.w, self.chat_seen, ptr(buf), len(lines))
        assert buf[:2 * k].reshape(k, 2).tolist() == [[tick, canon.strings(text)] for tick, text in lines], \
            f'{note}: chat lines differ'
        self.events_seen, self.chat_seen = len(py.events), len(py.chat_log)

    def c_draws(self):
        buf = np.zeros(10 * (1 << 14), np.int64)
        n = self.env.lib.t_draws(ptr(buf), 1 << 14)
        assert n >= 0, 'the C draw buffer overflowed'
        rows = buf[:10 * n].reshape(n, 10)
        return [(tuple(int(k) for k in row[2:2 + row[0]]), int(row[1])) for row in rows]

    def cycle(self, note):
        """One cycle on both: the same raise (class and text), else the same record and draws; then the upkeep."""
        before = {(n.nid, n.life): (n.active, n.mode, n.levels[engine.NPC_HP], n.x, n.z) for n in self.py.npcs}
        events = len(self.py.events)
        code = self.env.lib.t_cycle(self.w)
        try:
            self.py.cycle()
        except Exception as error:
            assert code == 1, f'{note}: the oracle raised {type(error).__name__}, C did not'
            got = c_error(self.env, self.w)
            assert (type(got).__name__, str(got)) == (type(error).__name__, str(error)), note
            self.stats['raised'] = type(error).__name__
            return False
        assert code == 0, f'{note}: C raised {c_error(self.env, self.w)!r}, the oracle did not'
        self.same(note)
        got, expected = sorted(self.c_draws()), sorted(self.draws)
        assert got == expected, f'{note}: draws differ: C {len(got)}, oracle {len(expected)}, first ' \
                                f'{next((a, b) for a, b in zip(got + [None] * 9, expected + [None] * 9) if a != b)}'
        self.draws.clear()
        self.upkeep()
        self.count(before, events)
        return True

    def upkeep(self):
        p = self.p
        if p is not None:
            for npc in self.py.npcs:   # agent.py observe_npcs
                near = npc.active and not npc.tele and npc.level == p.level and agent.chebyshev(p.x, p.z, npc.x,
                                                                                                npc.z) <= 15
                npc.observers = 1 if near else 0
        self.py.text_log.clear()
        self.env.lib.t_upkeep(self.w)

    def count(self, before, events):
        """What the cycle reached (coverage for the tests' assertions)."""
        s = self.stats
        s['ticks'] += 1
        for tick, kind, d in self.py.events[events:]:
            s['kills'] += kind == 'npc_defeated'
            s['excluded'] += kind == 'excluded'
        for n in self.py.npcs:
            was = before.get((n.nid, n.life))
            if was is None:
                s['respawns'] += 1
            elif n.mode == engine.MODE_PLAYERESCAPE and (n.x, n.z) != was[3:]:
                s['escapes'] += 1
            if was is not None and n.mode == engine.MODE_OPPLAYER2 and was[1] != n.mode and n.hunt_mode == -1 \
                    and n.base == self.env.ids['mugger']:
                s['hunts'] += 1
            if n.base == self.env.ids['0_46_49_saltfish'] and was is not None and (n.x, n.z) != was[3:]:
                s['spots'] += 1
        s['drops'] = max(s['drops'], len(self.py.objs))
        s['deaths'] += self.p is not None and self.p.dead and s['deaths'] == 0


def set_stats(lib, w, p):
    """The oracle player's levels, base levels and xp into the C player (the arrays live across the call)."""
    levels, bases, xps = (np.asarray(p.levels, np.int16), np.asarray(p.base_levels, np.int16),
                          np.asarray(p.stats, np.int32))
    lib.t_set_stats(w, ptr(levels), ptr(bases), ptr(xps))


def start_tiles(layout):
    return [(ORIGIN[0] + int(x), ORIGIN[1] + int(z)) for x, z in np.argwhere(layout.start)]


def player_tile(pair, rng, kinds, near):
    """A start tile, or (near) a walkable tile beside an npc of `kinds` (inside a pen whose gate stays closed: the
    gate scripts are scripts.c's)."""
    if near:
        npcs = [n for n in pair.py.npcs if n.base in kinds]
        tiles = [(x, z) for x, z, _ in pair.tables.meta['sources']
                 if any(0 < max(abs(x - n.x), abs(z - n.z)) <= 2 for n in npcs)]
        if tiles:
            return rng.choice(tiles)
    return rng.choice(start_tiles(pair.layout))


def random_levels(rng, low=1, high=40):
    levels = [1] * len(SKILLS)
    for i in (engine.ATTACK, engine.DEFENCE, engine.STRENGTH, engine.RANGED, engine.PRAYER, engine.MAGIC):
        levels[i] = rng.randint(low, high)
    levels[engine.HITPOINTS] = rng.randint(max(low, 10), max(high, 10))
    return levels


# ---- combat_level and player_combat_stat -----------------------------------------------------------------------
def test_combat_level_over_every_level_combination(env, lib):
    """Player.combat_level (engine.py:362-369, float, decides mugger hunting at 12) against C's for every (defence,
    hitpoints, prayer) triple and every (attack, strength, ranged) triple (the others varied by a fixed mix), and every
    magic level: 1.94M level sets in all, the threshold's neighbourhood included."""
    w = lib.t_world_new()
    try:
        p = Player(None, NAME)
        grid = np.arange(1, 100)
        mix = lambda *v: 1 + (np.add.reduce([a * m for a, m in zip(v, (7, 13, 29))]) % 99)
        a, b, c = (x.ravel() for x in np.meshgrid(grid, grid, grid, indexing='ij'))
        rows = []
        # columns: attack, defence, strength, hitpoints, ranged, prayer, magic
        rows.append(np.stack([mix(a, b), a, mix(b, c), b, mix(c, a), c, mix(a, c)], 1))
        rows.append(np.stack([a, mix(a, b), b, mix(b, c) % 90 + 10, c, mix(c, a), mix(a, b, c)], 1))
        rows.append(np.stack([grid, grid[::-1], grid, np.full(99, 10), grid, grid, grid], 1))
        bases = np.ascontiguousarray(np.concatenate(rows), np.int16)
        out = np.zeros(len(bases), np.int32)
        lib.t_combat_levels(w, ptr(bases), len(bases), ptr(out))
        expected = np.zeros(len(bases), np.int32)
        for k, row in enumerate(bases.tolist()):
            p.base_levels[:7] = row
            expected[k] = p.combat_level()
        bad = np.flatnonzero(out != expected)
        assert not len(bad), f'{len(bad)} level sets differ, first {bases[bad[0]].tolist()}: {out[bad[0]]} vs ' \
                             f'{expected[bad[0]]}'
        assert (expected == 12).any() and (expected == 13).any()
    finally:
        lib.t_world_free(w)


@settings(max_examples=300, **SLOW)
@given(st.data())
def test_player_combat_stat_and_bonuses_over_random_worn_sets(env, data):
    """[proc,player_combat_stat] (engine.py:2335-2358) with equip_get_bonuses and the weapon's style table over random
    worn sets (any wieldable obj in any slot, bows and crossbows with ammunition in the quiver), levels and com_mode
    (past the table and negative included): every varp it writes, the clamped com_mode included."""
    pair = Pair(env, 0, 7, 6, spawns=[])
    try:
        pair.place(*start_tiles(pair.layout)[0])
        p, lib, w = pair.p, env.lib, pair.w
        rng = random.Random(data.draw(st.integers(0, 2 ** 32 - 1)))
        levels = [rng.randint(1, 120) for _ in SKILLS]
        p.levels[:] = levels
        set_stats(lib, w, p)
        for _ in range(data.draw(st.integers(0, 8))):
            slot = data.draw(st.integers(0, 13))
            obj = env.wieldable[data.draw(st.integers(0, len(env.wieldable) - 1))]
            p.worn.items[slot] = [obj, 1]
            lib.t_set_slot(w, 1, slot, obj, 1)
        if data.draw(st.booleans()):
            bow, ammo = rng.choice(env.bows), rng.choice(env.ammo)
            for slot, obj in ((engine.WEARPOS_RHAND, bow), (engine.WEARPOS_QUIVER, ammo)):
                p.worn.items[slot] = [obj, rng.randint(1, 50)]
                lib.t_set_slot(w, 1, slot, obj, p.worn.items[slot][1])
        pair.varp('com_mode', data.draw(st.integers(-3, 6)))
        try:
            engine.player_combat_stat(pair.py, p)
        except Exception as error:
            assert lib.t_combat_stat(w) == 1
            assert type(c_error(env, w)).__name__ == type(error).__name__
            return
        assert lib.t_combat_stat(w) == 0, str(c_error(env, w))
        got = np.zeros(len(env.canon.varps), np.int32)
        lib.t_varps(w, ptr(got))
        expected = [-1 if p.varps[v] is None else p.varps[v] for v in env.canon.varps]
        diff = [(name, int(a), b) for name, a, b in zip(env.canon.varps, got, expected) if a != b]
        assert not diff, diff
    finally:
        pair.close()


# ---- ground items ---------------------------------------------------------------------------------------------------
@settings(max_examples=120, **SLOW)
@given(st.data())
def test_obj_add_and_take_obj_equal_the_oracle(env, data):
    """OBJ_ADD (engine.py:1135-1152) and OBJ_TAKEITEM (1154-1164) against the oracle's, between cycles that count
    the drops down: stacks joining the receiver's stack on the tile up to STACK_LIMIT (their timers restarted), one obj
    per unstackable item, -1 objs and counts, a dummy obj's Unsupported, takes into a filling inventory (the pickup
    event's added count) and takes of objs that already left the list (a held reference to a despawned or taken obj)."""
    t = env.base
    stackable = sorted(k for k, o in t.obj_types.items() if o['stackable'] and not o['dummyitem'])
    single = sorted(k for k, o in t.obj_types.items() if not o['stackable'] and not o['dummyitem'])
    dummy = sorted(k for k, o in t.obj_types.items() if o['dummyitem'])
    pair = Pair(env, 0, data.draw(st.integers(0, 2 ** 24)), 6, spawns=[])
    try:
        x0, z0 = start_tiles(pair.layout)[0]
        pair.place(x0, z0)
        p, lib, w, py = pair.p, env.lib, pair.w, pair.py
        objs = [stackable[data.draw(st.integers(0, len(stackable) - 1))] for _ in range(3)]
        objs += [single[data.draw(st.integers(0, len(single) - 1))] for _ in range(2)]
        for slot in range(data.draw(st.sampled_from([0, 0, 25, 27, 28]))):   # a nearly or wholly full inventory
            obj = objs[3] if slot else objs[0]
            p.inv.items[slot] = [obj, engine.STACK_LIMIT - 3 if obj == objs[0] else 1]
            lib.t_set_slot(w, 0, slot, *p.inv.items[slot])
        held = []
        for step in range(data.draw(st.integers(1, 40))):
            op = data.draw(st.sampled_from(['add', 'add', 'add', 'take', 'hold', 'cycle']))
            if op == 'add' and len(py.objs) < 80:
                kind = data.draw(st.integers(0, 9))
                obj = -1 if kind == 0 else dummy[0] if kind == 1 and dummy else objs[data.draw(st.integers(0, 4))]
                big = t.obj_types.get(obj, {}).get('stackable')
                count = data.draw(st.sampled_from([-1, 0, 1, 2, 7, engine.STACK_LIMIT - 1, 2 ** 30] if big else
                                                  [-1, 0, 1, 2, 4]))
                tile = (x0 + data.draw(st.integers(0, 1)), z0)
                duration = data.draw(st.sampled_from([1, 3, 200]))
                try:
                    py.obj_add(p, 0, *tile, obj, count, duration)
                except engine.Unsupported as error:
                    assert lib.t_obj_add(w, *tile, obj, count, duration) == 1
                    assert str(c_error(env, w)) == str(error)
                    return
                assert lib.t_obj_add(w, *tile, obj, count, duration) == 0, str(c_error(env, w))
            elif op == 'hold' and py.objs and len(held) < 16:
                at = data.draw(st.integers(0, len(py.objs) - 1))
                held.append(py.objs[at])
                lib.t_obj_hold(w, len(held) - 1, at)
            elif op == 'take' and held:
                k = data.draw(st.integers(0, len(held) - 1))
                py.take_obj(p, held[k])
                assert lib.t_take_held(w, k) == 0, str(c_error(env, w))
            elif op == 'cycle':
                assert pair.cycle(f'step {step}')
                continue
            pair.same(f'step {step} {op}')
    finally:
        pair.close()


# ---- npc targets ----------------------------------------------------------------------------------------------------
@settings(max_examples=400, **SLOW)
@given(st.data())
def test_npc_validate_target_equals_the_oracle(env, data):
    """Npc.validateTarget + targetWithinMaxRange (engine.py:1225-1243) for every mode kind (op modes' square without
    its far corners, ap modes' maxrange + attackrange, escape's two conditions, the others' maxrange + 1), ranges from
    the spawn, the npc away from it, the player anywhere around, with and without a target."""
    pair = combat_pair(env, random.Random(data.draw(st.integers(0, 2 ** 16))), data.draw(st.integers(0, 7)))
    try:
        pair.place(*start_tiles(pair.layout)[0])
        npc = pair.py.npcs[data.draw(st.integers(0, len(pair.py.npcs) - 1))]
        nt = env.base.npc_types[npc.type]
        r = nt['maxrange'] + nt['attackrange'] + 3
        off = st.integers(-r, r)
        x, z = npc.start_x + data.draw(off), npc.start_z + data.draw(off)
        px, pz = npc.start_x + data.draw(off), npc.start_z + data.draw(off)
        mode = data.draw(st.sampled_from([0, 1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17]))
        target = data.draw(st.booleans())
        npc.x, npc.z, npc.mode, npc.target = x, z, mode, pair.p if target else None
        pair.p.x, pair.p.z = px, pz
        expected = bool(target and pair.py.npc_validate_target(npc, nt))
        slot = env.lib.t_npc_slot(pair.w, npc.nid)
        assert env.lib.t_validate(pair.w, slot, x, z, mode, target, px, pz) == expected
    finally:
        pair.close()


# ---- the death drop programs -----------------------------------------------------------------------------------
def death_case(env, rng, npc_type=None):
    """A random world for a death script: the npc's type (one of the five compiled roots), its hero and aggressor, the
    player's inventories (clues, the drop tables' objs, others), %trail_status, access (protect, delay, a chat),
    exclusions, the npc's last movement against the tick, the seed and the tick."""
    t = env.base
    pool = env.clue_objs + env.inv_objs * 8 + env.weapons + sorted(rng.sample(sorted(t.obj_types), 30))
    return dict(type=npc_type if npc_type is not None else rng.choice(env.drop_types), k=rng.randrange(8),
                seed=rng.randrange(2 ** 24), tick=rng.randrange(10, 5000),
                items=[[(rng.choice(pool), rng.randint(1, 3)) for _ in range(rng.choice((0, 0, 1, 3, 6)))]
                       for _ in range(3)],
                trail=rng.choice((0, rng.randrange(1 << 20))), protect=rng.random() < 0.3,
                delayed=rng.random() < 0.2, chat=rng.random() < 0.3,
                exclusions=rng.choice(((), ('clue_drops', 'loot_drops'), ('clue_drops',), ('loot_drops',),
                                       tuple(engine.EXCLUSIONS))),
                hero=rng.random() < 0.8, damage=rng.randint(1, 9), aggressive=rng.random() < 0.7,
                moved=rng.choice((-1, 0, 1, 2, 3, 40)), far=rng.random() < 0.3)


def run_death(env, case, monkey=None):
    """npc_death_script on both sides through the npc queue (an [3, 0, 0] request, as npc_default_damage queues it):
    up to 6 cycles, each compared; returns the pair's stats."""
    layout, tables, _ = env.layout(case['k'])
    t = env.base
    rows = [(int(n), int(ty), int(x), int(z)) for n, ty, x, z in layout.npcs]
    target = rows[len(rows) // 2][0]
    spawns = [dict(nid=n, type=case['type'] if n == target else ty, x=ORIGIN[0] + x, z=ORIGIN[1] + z, level=0,
                   size=t.npc_types[case['type'] if n == target else ty]['size']) for n, ty, x, z in rows]
    pair = Pair(env, case['k'], case['seed'], case['tick'], spawns=spawns, exclusions=case['exclusions'])
    try:
        tiles = start_tiles(layout)
        pair.place(*(tiles[-1] if case['far'] else tiles[0]))
        p, lib, w = pair.p, env.lib, pair.w
        for which, (inv, items) in enumerate(zip((p.inv, p.worn, p.bank), case['items'])):
            for obj, count in items:
                inv.add(t, obj, count)
            for slot, it in enumerate(inv.items):
                if it:
                    lib.t_set_slot(w, which, slot, it[0], it[1])
        pair.varp('trail_status', case['trail'])
        p.protect, p.delayed = case['protect'], case['delayed']
        p.delayed_until = case['tick'] + 2 if case['delayed'] else -1
        p.modal_state = {'chat'} if case['chat'] else set()
        lib.t_set_player(w, p.protect, p.delayed, p.delayed_until, 4 if case['chat'] else 0)
        npc = next(n for n in pair.py.npcs if n.nid == target)
        slot = lib.t_npc_slot(w, target)
        npc.heroes = {NAME: case['damage']} if case['hero'] else {}
        npc.vars['aggressive_player'] = NAME if case['aggressive'] else None
        npc.last_movement = case['tick'] - case['moved']
        lib.t_npc_set(w, slot, case['hero'], case['damage'], case['aggressive'], npc.last_movement,
                      npc.levels[engine.NPC_HP])
        pair.py.npc_queue(npc, 3, 0, 0)
        assert lib.t_npc_enqueue(w, slot, 3, 0, 0, 0) == 0
        pair.upkeep()
        pair.same('setup')
        for tick in range(6):
            if not pair.cycle(f'death of npc type {case["type"]}, tick {tick}'):
                break
        pair.stats['drops'] = len(pair.py.objs)
        return pair.stats
    finally:
        pair.close()


@settings(max_examples=40, **SLOW)
@given(st.integers(0, 2 ** 32 - 1))
def test_drop_programs_do_what_npc_death_script_does(env, key):
    """Each compiled death program (DOP_* on combat.c's machine, inside npc_death's suspensions) against
    npc_death_script with rs2.run and DeathContext on identical worlds, tick by tick: the npc's arrival and death
    delays, its removal, the drops (obj_add stacking, list order, receiver, timers), the excluded clue and loot events,
    %trail_status and %lastcombat, the closed modals and weak queue, the draws (1000 + k keys) and every raise
    (Unsupported without a hero, ZeroDivisionError from rs2's arithmetic)."""
    run_death(env, death_case(env, random.Random(key)))


def test_drop_programs_drop_exclude_and_raise_as_the_oracle(env):
    """A fixed sample of 150 death worlds per compiled program, each compared tick by tick as above, reaches drops,
    excluded drops and the oracle's raises (so the equivalence covers the tables, not only their refusals)."""
    rng = random.Random(20261026)
    totals = dict(drops=0, excluded=0, raised=0, kills=0)
    for npc_type in env.drop_types:
        drops = 0
        for _ in range(150):
            stats = run_death(env, death_case(env, rng, npc_type))
            drops += stats['drops']
            totals['excluded'] += stats['excluded']
            totals['raised'] += stats['raised'] is not None
        totals['drops'] += drops
        if env.base.npc_types[npc_type]['op'][1]:   # an attackable type drops something
            assert drops > 0, npc_type
    assert totals['excluded'] > 0 and totals['raised'] > 0, totals


# ---- npc turns on pen layouts and combat camps -----------------------------------------------------------------
def play(pair, rng, ticks, attack_at=None, targets=(), walk_every=0):
    """Cycle both sides `ticks` times (fewer after a raise): from tick `attack_at` on, an idle player attacks a living
    npc of a base type in `targets`, the nearest one; it walks to a random source every `walk_every` ticks."""
    sources = [(x, z) for x, z, _ in pair.tables.meta['sources']]
    for tick in range(ticks):
        if attack_at is not None and tick >= attack_at and pair.p is not None and pair.p.target is None \
                and not pair.p.delayed and tick % 7 == attack_at % 7:
            alive = [n for n in pair.py.npcs if n.active and n.base in targets and not n.delayed]
            if alive:
                p = pair.p
                pair.attack(min(alive, key=lambda n: (max(abs(n.x - p.x), abs(n.z - p.z)), n.nid)))
        if walk_every and tick % walk_every == walk_every - 1 and pair.p is not None and pair.p.target is None:
            pair.walk(rng.choice(sources))
        if not pair.cycle(f'tick {tick}'):
            return False
    return True


@settings(max_examples=25, **SLOW)
@given(st.integers(0, 2 ** 32 - 1))
def test_pen_npcs_turn_as_the_oracle(env, key):
    """The layout's npcs (spots and chickens) on a random compact layout, tick by tick: dormancy without a player
    (lifecycle only), the spot's move at boot (timer 1) and its later moves, wander draws and stuck teleports, then a
    player who walks and, half the time, attacks chickens: retaliation, escape at the retreat hitpoints, the attack
    loop, death, drops and respawn."""
    rng = random.Random(key)
    k = rng.randrange(8)
    pair = Pair(env, k, rng.randrange(2 ** 24), rng.randrange(6, 3000))
    try:
        pair.same('init')
        for tick in range(rng.choice((0, 2, 5))):      # no player yet: every npc is dormant
            assert pair.cycle(f'dormant tick {tick}')
        chickens = {env.ids['chicken'], env.ids['chicken_brown']}
        pair.place(*player_tile(pair, rng, chickens, rng.random() < 0.6))
        pair.kit(random_levels(rng), rng.choice(env.weapons + [None]), rng.random() < 0.6, com_mode=rng.randrange(4))
        play(pair, rng, rng.randint(60, 220), attack_at=rng.randrange(3, 30) if rng.random() < 0.6 else None,
             targets=chickens, walk_every=rng.choice((0, 0, 9, 25)))
    finally:
        pair.close()


def combat_pair(env, rng, k, mugger=True, exclusions=tuple(combat.COMBAT['exclusions'])):
    """A combat world on layout k (combat.combat_npcs: spots, a goblin at every chicken spawn, a mugger on an
    outside() tile away from the camp, as draw_combat draws them)."""
    layout, tables, _ = env.layout(k)
    muggers = []
    if mugger:
        camp = combat.camp(layout, env.base)
        cx, cz = np.mean(camp, axis=0)
        options = [(int(x), int(z)) for x, z in np.argwhere(combat.outside(layout))
                   if 6 <= max(abs(x - cx), abs(z - cz)) <= 12 and (int(x), int(z)) not in camp]
        muggers = [rng.choice(options)]
    spawns = combat.combat_npcs(layout, env.base, muggers)
    return Pair(env, k, rng.randrange(2 ** 24), rng.randrange(6, 3000), spawns=spawns, exclusions=exclusions)


@settings(max_examples=25, **SLOW)
@given(st.integers(0, 2 ** 32 - 1))
def test_combat_camp_npcs_hunt_and_fight_as_the_oracle(env, key):
    """A combat world (goblins and a mugger, combat's exclusions) tick by tick: the mugger's hunt (process_world's
    filter, huntrange, sight, combat level against vislevel * 2, the not-in-combat checks), its acquire and chase
    (naive paths, entity_reaching), npc_default_attack and playerhit_n_melee, combat_damage_player and the auto-retaliate
    on the normal queue behind it, the player's attacks on goblins, goblin and mugger deaths with clue and loot exclusions,
    respawns, and the player's death (player_death's delays and its Unsupported)."""
    rng = random.Random(key)
    pair = combat_pair(env, rng, rng.randrange(8), mugger=rng.random() < 0.8)
    try:
        pair.place(*player_tile(pair, rng, {env.ids['goblin']}, rng.random() < 0.4))
        weak = rng.random() < 0.6           # combat level <= 12: the mugger hunts
        levels = random_levels(rng, 1, 3 if weak else 30)
        pair.kit(levels, rng.choice(env.weapons + [None]), rng.random() < 0.6,
                 hp=rng.randint(1, levels[engine.HITPOINTS]), com_mode=rng.randrange(4))
        play(pair, rng, rng.randint(80, 260), attack_at=rng.randrange(3, 40) if rng.random() < 0.7 else None,
             targets={env.ids['goblin']}, walk_every=rng.choice((0, 0, 15)))
    finally:
        pair.close()


@settings(max_examples=20, **SLOW)
@given(st.integers(0, 2 ** 32 - 1), st.booleans())
def test_melee_exchange_until_a_kill_or_a_death(env, key, camp):
    """The player attacks one chicken (a pen) or goblin (a camp, no mugger) from the start and keeps at it: approach,
    player_combat_start_ap's range wait, the in-combat checks and their messages, hit and defence rolls, xp, heroes,
    npc_retaliate, npc_default_damage's events, the npc's attacks back and the player's damage, until the npc's death
    script, its drops and some ticks after (or the player's death)."""
    rng = random.Random(key)
    k = rng.randrange(8)
    pair = combat_pair(env, rng, k, mugger=False) if camp else Pair(env, k, rng.randrange(2 ** 24),
                                                                     rng.randrange(6, 3000))
    try:
        kinds = {env.ids['goblin']} if camp else {env.ids['chicken'], env.ids['chicken_brown']}
        pair.place(*player_tile(pair, rng, kinds, rng.random() < 0.8))
        levels = random_levels(rng, 1, 25)
        pair.kit(levels, rng.choice(env.weapons + [None]), rng.random() < 0.7, com_mode=rng.randrange(4),
                 hp=rng.randint(1, levels[engine.HITPOINTS]))
        npc = min((n for n in pair.py.npcs if n.base in kinds), key=lambda n: (max(abs(n.x - pair.p.x),
                                                                                  abs(n.z - pair.p.z)), n.nid))
        pair.attack(npc)
        for tick in range(400):
            if not pair.cycle(f'tick {tick}'):
                break
            if pair.p.target is None and npc.active and not npc.delayed and not pair.p.delayed and tick % 5 == 0:
                pair.attack(npc)
            if pair.stats['kills'] and npc.active and tick > 30:
                break
    finally:
        pair.close()


def test_fixed_runs_reach_the_npc_and_combat_paths(env):
    """A fixed set of runs of the three kinds above, each compared tick by tick, reaches what they claim to cover:
    kills, escapes, respawns, mugger hunts, spot moves, drops, excluded drops and a player death."""
    totals = dict.fromkeys(('ticks', 'kills', 'escapes', 'respawns', 'hunts', 'deaths', 'spots', 'excluded'), 0)
    drops = 0
    for key in range(14):
        rng = random.Random(1000 + key)
        k = key % 8
        if key % 2 == 0:
            pair = Pair(env, k, rng.randrange(2 ** 24), rng.randrange(6, 3000))
            targets = {env.ids['chicken'], env.ids['chicken_brown']}
        else:
            pair = combat_pair(env, rng, k)
            targets = {env.ids['goblin']}
        try:
            pair.place(*player_tile(pair, rng, targets, key % 4 != 1))
            levels = random_levels(rng, 1, 3 if key % 4 == 1 else 30)
            pair.kit(levels, rng.choice(env.weapons), True, hp=rng.randint(1, levels[engine.HITPOINTS]))
            play(pair, rng, 300, attack_at=2, targets=targets)
            for name in totals:
                totals[name] += pair.stats[name]
            drops += pair.stats['drops'] > 0
        finally:
            pair.close()
    print(totals, drops)
    for name in ('kills', 'escapes', 'respawns', 'hunts', 'spots', 'deaths'):
        assert totals[name] > 0, (name, totals)
    assert drops > 0, totals
