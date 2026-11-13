"""The native content scripts (sim/native/scripts.c) against the oracle scripts they port (sim/engine.py:1830-3260,
sim/agent.py:1396-1414), run to their ends in a C world and a Python world on the same layout (the port's design,
section 5).

The C side is a harness library: every sim/native/*.c but module.c, over the exported pack sim/tests PACK
(compact train-8 and the train-128 supply pool), with the real paths.c, world.c, scripts.c and combat.c (pickups,
drops and the combat stat run combat.c's code). Npc turns and lifecycles do nothing on both sides (the oracle's are
disabled alike), so the fixed scenarios find their fishing spot where they left it. Both
worlds are driven as the agent drives a world (agent.py interact_loc, interact_npc, use_on, pick_up, op_held,
drop_slot, use_on_held, resume_dialog) with the same waypoints, and cycled in lockstep. After every cycle and every
between-cycle op the World part of the canonical record (sim/native_harness.py Canon, fields RS_CF_TICK through
RS_CF_EVENT_LAST, rs_world_canon on the C side) must be equal, and so must the draws (key tuples and m, as multisets),
the events, the chat lines and the dialog texts written since the previous check. An Unsupported must be raised on
both sides with the same text.

Hypothesis draws the layout, the world seed, the start tick, the XP rate, the start levels (just under a level-up
often: 14/15 is the oak and anchovy unlock), the inventory (the kit's tools, food, bones, chicken loot, logs, ores and
weapons, a full inventory or none of a tool), the worn items, the action and eat delays, the three stat timers and the
sequence of actions. Nothing of the archive is printed: failing examples hold ints, and texts are compared by hash."""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zlib

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

from sim import agent as agent_mod, engine, native, native_harness as H
from sim.engine import APLOC1, APLOCU, APNPC1, APNPCU, APOBJ1, LEVEL_XP, SKILLS, Player, Unsupported, World
from sim.tests import PACK

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'logs' / 'sim' / 'data' / 'draynor-nc1.npz'
COMPACT_POOL = ROOT / 'logs' / 'sim' / 'world' / 'train-8-nc1-derived.npz'
SUPPLY_POOL = ROOT / 'logs' / 'supply' / 'pools' / 'train-128.npz'
ORIGIN = (2880, 3328)
NAME = 'wd1a0'
WC, FISH, MINE, HP = SKILLS.index('woodcutting'), SKILLS.index('fishing'), SKILLS.index('mining'), SKILLS.index('hitpoints')
TIMERS = ('stat_regen', 'stat_boost_restore', 'health_regen')

HARNESS_C = r'''
#include <stdlib.h>
#include <string.h>
#include "rs.h"

/* ---- npc.c doubles: npcs stand still (the oracle side disables World.npc_turn and npc_lifecycle alike), so the
   fixed scenarios find their fishing spot where they left it ------------------------------------------------------ */
int rs_npc_turn(World *w, Npc *n) { (void)w, (void)n; return 0; }
int rs_npc_lifecycle(World *w, Npc *n) { (void)w, (void)n; return 0; }

/* ---- the driver ------------------------------------------------------------------------------------------------ */
World *t_world_new(void) { return calloc(1, sizeof(World)); }
void t_world_free(World *w) {
    if (w->trace) free(w->trace->draws), free(w->trace);
    free(w);
}
void t_err(const World *w, int32_t *out) {
    out[0] = w->err.code, out[1] = w->err.exc, out[2] = w->err.reason;
    for (int i = 0; i < RS_ERR_ARGS; i++) out[3 + i] = w->err.arg[i];
}
const char *t_err_text(const World *w) { return w->err.text; }
int32_t t_pool(const RsPack *P, int which) {
    return which == 0 ? P->compact0 : which == 1 ? P->n_compact : which == 2 ? P->supply0 : P->n_supply;
}
/* World init on a layout record with its npcs, a draw trace kept across the init's reset. */
int t_world_init(World *w, const RsPack *P, int rec, int64_t seed, int32_t start_tick, int32_t xp_rate) {
    RsTrace *t = w->trace;
    if (!t) {
        t = calloc(1, sizeof *t);
        t->draws_cap = 1 << 14;
        t->draws = calloc((size_t)t->draws_cap, sizeof(RsDraw));
    }
    int rc = rs_world_init(w, P, &P->layouts[rec], seed, start_tick, NULL, 0, 0, xp_rate);
    w->trace = t, t->n_draws = 0, t->truncated = 0;
    return rc;
}
void t_place_player(World *w, int32_t x, int32_t z) {   /* its build area around it, as after an arrival */
    rs_player_init(w);
    w->p.x = (int16_t)x, w->p.z = (int16_t)z, w->p.origin_x = (int16_t)x, w->p.origin_z = (int16_t)z;
    w->has_player = 1;
    rs_occupy(w, 0, x, z, 1, 1);
}
/* levels, base levels, xp, inventory and worn (id, count) pairs, varps (RS_VARPS order) */
void t_set_player(World *w, const int16_t *levels, const int16_t *base, const int32_t *stats, const int32_t *inv,
                  const int32_t *worn, const int32_t *varps, int afk) {
    Player *p = &w->p;
    memcpy(p->levels, levels, sizeof p->levels);
    memcpy(p->base_levels, base, sizeof p->base_levels);
    memcpy(p->stats, stats, sizeof p->stats);
    for (int i = 0; i < RS_INV_CAP; i++) p->inv[i] = (Slot){inv[2 * i], inv[2 * i + 1]};
    for (int i = 0; i < RS_WORN_CAP; i++) p->worn[i] = (Slot){worn[2 * i], worn[2 * i + 1]};
    memcpy(p->varp, varps, sizeof p->varp);
    p->afk_ready = (uint8_t)afk;
}
void t_set_last(World *w, int32_t item, int32_t slot, int32_t useitem, int32_t useslot) {
    w->p.last_item = item, w->p.last_slot = slot, w->p.last_useitem = useitem, w->p.last_useslot = useslot;
}
void t_set_timer(World *w, int i, int32_t interval, int32_t clock, int16_t script, int kind) {
    w->p.timer[i] = (Timer){interval, clock, script, (uint8_t)kind, 0};
    if (w->p.n_timers < i + 1) w->p.n_timers = (uint8_t)(i + 1);
}
int t_cycle(World *w) { return rs_world_cycle(w); }
int t_canon(const World *w, int64_t *buf, int cap) {
    int n = 0;
    RsErr e;
    return rs_world_canon(w, buf, cap, &n, &e) ? -1 : n;
}
/* draws since the last call, rows (tick, n, m, keys[RS_DRAW_KEYS]); -1 when the trace overflowed */
int t_draws(World *w, int64_t *out, int cap) {
    RsTrace *t = w->trace;
    if (t->truncated || t->n_draws > cap) return -1;
    for (int i = 0; i < t->n_draws; i++) {
        const RsDraw *d = &t->draws[i];
        int64_t *row = out + i * (3 + RS_DRAW_KEYS);
        row[0] = d->tick, row[1] = d->n, row[2] = d->m;
        for (int k = 0; k < RS_DRAW_KEYS; k++) row[3 + k] = d->key[k];
    }
    int n = t->n_draws;
    t->n_draws = 0;
    return n;
}
/* events from `from`, rows (kind, tick, a8, b8, has_source, obj, count, v0, v1, v2, v3) */
int t_events(const World *w, int from, int32_t *out, int cap) {
    int n = 0;
    for (int i = from; i < w->n_events && n < cap; i++, n++) {
        const RsEvent *e = &w->ev[i];
        int32_t row[11] = {e->kind, e->tick, e->a8, e->b8, e->has_source, e->obj, e->count, e->v0, e->v1, e->v2, e->v3};
        memcpy(out + 11 * n, row, sizeof row);
    }
    return n;
}
static int32_t text_of(const World *w, int32_t msg) { return msg >= 0 && msg < w->P->n_msgs ? w->P->msgs[msg].text : -1; }
/* chat lines from line `from` (tick, text string id); -1 when a line left the ring */
int t_chat(const World *w, int64_t from, int32_t *out, int cap) {
    if (from < w->chat.n - RS_CHAT_RING) return -1;
    int n = 0;
    for (int64_t k = from; k < w->chat.n && n < cap; k++, n++) {
        const RsChatLine *c = &w->chat.ring[k % RS_CHAT_RING];
        out[2 * n] = c->tick, out[2 * n + 1] = text_of(w, c->msg);
    }
    return n;
}
int64_t t_chat_n(const World *w) { return w->chat.n; }
/* settext lines from `from` of this capture (tick, iface, com, text string id) */
int t_texts(const World *w, int from, int32_t *out, int cap) {
    int n = 0;
    for (int i = from; i < w->text.n && n < cap; i++, n++) {
        const RsTextLine *t = &w->text.line[i];
        out[4 * n] = t->tick, out[4 * n + 1] = t->iface, out[4 * n + 2] = t->com, out[4 * n + 3] = text_of(w, t->msg);
    }
    return n;
}
/* Agent.capture_text's effect on the world: chat read up to now, the text log drained */
void t_capture(World *w) { w->chat.captured = w->chat.n, w->text.n = 0; }
int t_loc_slot(const World *w, int32_t index) {
    for (int s = 0; s < w->n_loc_slots; s++) if (w->locs[s].serial && w->locs[s].index == index) return s;
    return -1;
}
void t_clear_pending(World *w) { rs_clear_pending_action(w); }
void t_queue(World *w, int n, const int16_t *xy) { rs_queue_waypoints(w, (const int16_t (*)[2])xy, n); }
int t_set_interaction_loc(World *w, int slot, int32_t op) { return rs_set_interaction(w, rs_ref_loc(w, slot), op); }
int t_set_interaction_npc(World *w, int32_t nid, int32_t op) {
    for (int i = 0; i < w->n_npcs; i++)
        if (w->npcs[i].nid == nid) return rs_set_interaction(w, rs_ref_npc(w, i), op);
    return -1;
}
int t_set_interaction_obj(World *w, int at, int32_t op) {
    return at < w->n_objs ? rs_set_interaction(w, rs_ref_obj(w, w->obj_order[at]), op) : -1;
}
/* World.run_held of an obj type's trigger slot (agent.op_held, drop_slot, use_on_held look the script up so) */
int t_run_held_obj(World *w, int32_t obj, int trig) {
    const RsObjType *t = rs_obj_type(w->P, obj);
    if (!t) return -1;
    return rs_run_held(w, t->trig_script[trig], t->trig_arg[trig], t->trig_name[trig]);
}
int t_resume(World *w) { return rs_resume(w); }
int t_combat_stat(World *w) { return rs_player_combat_stat(w); }
int t_execute(World *w, int16_t script, int32_t a0, int32_t a1) { return rs_execute(w, script, (RsRef){0}, a0, a1, 0, 1, 0); }
'''

STUBBED = {'npc.c': ('rs_npc_turn', 'rs_npc_lifecycle')}   # combat.c is the port's own

_C = dict(p=ctypes.c_void_p, i=ctypes.c_int, i16=ctypes.c_int16, i32=ctypes.c_int32, i64=ctypes.c_int64,
          s=ctypes.c_char_p)
SIGNATURES = {
    'rs_pack_open': ('i', 's', 'p', 'p'), 'rs_pack_close': (None, 'p'),
    't_world_new': ('p',), 't_world_free': (None, 'p'), 't_err': (None, 'p', 'p'), 't_err_text': ('s', 'p'),
    't_pool': ('i32', 'p', 'i'), 't_world_init': ('i', 'p', 'p', 'i', 'i64', 'i32', 'i32'),
    't_place_player': (None, 'p', 'i32', 'i32'), 't_set_player': (None, 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'i'),
    't_set_last': (None, 'p', 'i32', 'i32', 'i32', 'i32'), 't_set_timer': (None, 'p', 'i', 'i32', 'i32', 'i16', 'i'),
    't_cycle': ('i', 'p'), 't_canon': ('i', 'p', 'p', 'i'), 't_draws': ('i', 'p', 'p', 'i'),
    't_events': ('i', 'p', 'i', 'p', 'i'), 't_chat': ('i', 'p', 'i64', 'p', 'i'), 't_chat_n': ('i64', 'p'),
    't_texts': ('i', 'p', 'i', 'p', 'i'), 't_capture': (None, 'p'), 't_loc_slot': ('i', 'p', 'i32'),
    't_clear_pending': (None, 'p'), 't_queue': (None, 'p', 'i', 'p'), 't_set_interaction_loc': ('i', 'p', 'i', 'i32'),
    't_set_interaction_npc': ('i', 'p', 'i32', 'i32'), 't_set_interaction_obj': ('i', 'p', 'i', 'i32'),
    't_run_held_obj': ('i', 'p', 'i32', 'i'), 't_resume': ('i', 'p'), 't_combat_stat': ('i', 'p'),
    't_execute': ('i', 'p', 'i16', 'i32', 'i32'),
}


class RsErr(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('exc', ctypes.c_uint8), ('reason', ctypes.c_uint16),
                ('arg', ctypes.c_int32 * 6), ('text', ctypes.c_char_p)]


def build_harness(out_dir):
    """Compile every sim/native/*.c but module.c (STUBBED entry points renamed away) and the driver into one library."""
    header = native.generated_header()
    digest = hashlib.sha256(header.encode() + HARNESS_C.encode() + ' '.join(native.FLAGS).encode())
    for path in sorted(native.SOURCE.glob('*.[ch]')):
        digest.update(path.name.encode() + path.read_bytes())
    lib = out_dir / f'scripts-harness-{digest.hexdigest()[:16]}.so'
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
    done = subprocess.run(['cc', '-shared', *objects, '-o', str(lib), '-lz'], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-3000:]
    return lib


def ptr(array):
    return ctypes.c_void_p(array.ctypes.data)


def digest(text):
    """A text as the assertions show it: its length and crc32 (game text is private)."""
    return None if text is None else f'{len(text)} chars crc32 {zlib.crc32(text.encode()):08x}'


# ---- the session: pack, tables, layouts --------------------------------------------------------------------------
class Env:
    """The harness library, the C pack, the module's view of the same pack (strings, script ids) and the oracle's
    tables and layouts."""

    def __init__(self, lib):
        from sim import native_export as X
        self.lib = lib
        self.pack = native.load().Pack(str(PACK))
        if json.loads(self.pack.manifest).get('oracle_hash') != native.oracle_hash():
            pytest.skip(f'{PACK} was exported from another oracle: export it again')
        P, err = ctypes.c_void_p(), RsErr()
        assert lib.rs_pack_open(str(PACK).encode(), ctypes.byref(P), ctypes.byref(err)) == 0, err.text
        self.P = P
        self.pools = {'compact': (lib.t_pool(P, 0), lib.t_pool(P, 1)), 'supply': (lib.t_pool(P, 2), lib.t_pool(P, 3))}
        self.base = X.load_tables(ARCHIVE)
        self.layouts = {'compact': X.load_pool(COMPACT_POOL)[0], 'supply': X.load_pool(SUPPLY_POOL)[0][:4]}
        assert len(self.layouts['compact']) == self.pools['compact'][1]
        self.canon = H.Canon(H.PackStrings(self.pack))
        self.sid = self.canon.strings
        self.script = {name: k for k, (name, _, _) in enumerate(native.load().scripts())}
        self.script_names = {k: name for name, k in self.script.items()}
        self._tables = {}

    def tables(self, kind, k):
        from sim import world
        if (kind, k) not in self._tables:
            self._tables[(kind, k)] = world.LayoutTables(self.base, self.layouts[kind][k])
        return self._tables[(kind, k)]

    def obj(self, name):
        return self.base.obj_ids.get(name)


@pytest.fixture(scope='module')
def env(tmp_path_factory):
    for path in (ARCHIVE, COMPACT_POOL, SUPPLY_POOL, PACK):
        if not path.exists():
            pytest.skip(f'{path.relative_to(ROOT)} absent (docs/RUNNING.md)')
    lib = ctypes.CDLL(str(build_harness(tmp_path_factory.mktemp('scripts'))))
    for name, (res, *args) in SIGNATURES.items():
        f = getattr(lib, name)
        f.restype = _C[res] if isinstance(res, str) else res
        f.argtypes = [_C[a] if isinstance(a, str) else a for a in args]
    e = Env(lib)
    saved = World.npc_turn, World.npc_lifecycle
    World.npc_turn = World.npc_lifecycle = lambda self, npc: None   # still npcs on both sides (the C doubles)
    try:
        yield e
    finally:
        World.npc_turn, World.npc_lifecycle = saved
        lib.rs_pack_close(e.P)


# ---- a pair of worlds ------------------------------------------------------------------------------------------------
STUB_AGENT = SimpleNamespace(name=NAME, decision=0, charged=0, episode_ticks=0, last_mask_sum=0, pace=0, inv_page=0,
                             bank_page=0, bank_pages=1, style_stale=False, using=None, death_tick=None, visited=set(),
                             chat_seq=0, chat_covered=0, chat=[], option=None, option_end=None, refused={},
                             texts={})
STUB_TASK = SimpleNamespace(active=False)


class Pair:
    """A C world and an oracle world on one layout, the player placed on (x, z) as Agent._arrive leaves it, compared
    after every cycle and op (module docstring)."""

    def __init__(self, env, kind, k, seed, start_tick, xp_rate, tile):
        self.env, self.lib = env, env.lib
        self.tables = env.tables(kind, k)
        self.w = self.lib.t_world_new()
        rec = env.pools[kind][0] + k
        assert self.lib.t_world_init(self.w, env.P, rec, seed, start_tick, xp_rate) == 0, self.error_text()
        self.py = World(self.tables, seed=seed, start_tick=start_tick, npcs=None, xp_rate=xp_rate)
        self.draws = []
        py = self.py
        self.py.rng = H.TracedRng(py.rng, lambda keys, value: self.draws.append(
            (py.tick, tuple(int(v) for v in keys), int(value * 16777216))))
        self.lib.t_place_player(self.w, *tile)
        self.p = Player(py, NAME)
        self.p.x, self.p.z = tile
        self.p.origin_x, self.p.origin_z = tile   # the client's scene (act-v3 routes) around the player
        py.player = self.p
        py._occupy(self.p, True)
        self.events_seen = self.chat_seen = self.text_seen = 0
        self.cut = env.canon.id['RS_CF_AGENT']
        self.seen = set()       # the suspended frame chains checked, as ('SCRIPT:pc', ...) outermost first
        self.unsupported = None  # the text both sides raised, if they did
        self.keys = set()        # the script draw keys (R_SCRIPT, tick, k) made, by k

    def close(self):
        self.lib.t_world_free(self.w)

    # -- comparing
    def error(self):
        out = np.zeros(9, np.int32)
        self.lib.t_err(self.w, ptr(out))
        text = self.lib.t_err_text(self.w)
        e = native.load().Error(*(int(v) for v in out[:3]), tuple(int(v) for v in out[3:]),
                                text.decode() if text else None, 0)
        return native.as_python(e, self.env.pack)

    def error_text(self):
        e = self.error()
        return f'{type(e).__name__} {digest(str(e))}'

    def world_record(self):
        rec = self.env.canon(self.py, STUB_AGENT, STUB_TASK)
        k = 0
        while rec[k] != self.cut:
            k += 2 + int(rec[k + 1])
        return rec[:k]

    def check(self, note):
        cap = 1 << 17
        buf = np.zeros(cap, np.int64)
        n = self.lib.t_canon(self.w, ptr(buf), cap)
        assert n >= 0
        a, b = buf[:n], self.world_record()
        if self.p.script is not None:
            names = self.env.script_names
            self.seen.add(tuple(f'{names[k][5:]}:{pc}' for k, pc in H.frame_chain(self.p.script[0])))
        if not np.array_equal(a, b):
            raise AssertionError(f'{note}: C vs oracle records differ: {H.field_diff(a, b)}')
        rows = np.zeros((1 << 12, 3 + 8), np.int64)
        n = self.lib.t_draws(self.w, ptr(rows), len(rows))
        assert n >= 0, f'{note}: the C draw trace overflowed'
        c_draws = sorted((int(r[0]), tuple(int(v) for v in r[3:3 + int(r[1])]), int(r[2])) for r in rows[:n])
        assert c_draws == sorted(self.draws), f'{note}: draws differ: C {c_draws} oracle {sorted(self.draws)}'
        self.keys.update(keys[2] for _, keys, _ in self.draws if keys[0] == engine.R_SCRIPT and len(keys) > 2)
        self.draws.clear()
        py = self.py
        events = np.zeros((256, 11), np.int32)
        n = self.lib.t_events(self.w, self.events_seen, ptr(events), len(events))
        want = [self.env.canon.event(e, NAME) for e in py.events[self.events_seen:]]
        assert events[:n].tolist() == want, f'{note}: events differ'
        self.events_seen = len(py.events)
        chat = np.zeros((256, 2), np.int32)
        n = self.lib.t_chat(self.w, self.chat_seen, ptr(chat), len(chat))
        want = [[t, self.env.sid(text)] for t, text in py.chat_log[self.chat_seen:]]
        assert chat[:n].tolist() == want, f'{note}: chat differs'
        self.chat_seen = len(py.chat_log)
        texts = np.zeros((64, 4), np.int32)
        n = self.lib.t_texts(self.w, self.text_seen, ptr(texts), len(texts))
        want = [[t, self.env.sid(iface), com, self.env.sid(text)] for t, (iface, com), text in py.text_log[self.text_seen:]]
        assert texts[:n].tolist() == want, f'{note}: dialog texts differ'
        self.text_seen = len(py.text_log)

    def capture(self):
        """Agent.capture_text after a cycle: the text log drained (the chat cursor is kept by check)."""
        self.py.text_log.clear()
        self.text_seen = 0
        self.lib.t_capture(self.w)

    def raised(self, code, error, note):
        """Both sides raised: C's error renders as the oracle's Unsupported."""
        assert code == 1, f'{note}: the oracle raised {digest(str(error))}, C did not'
        mine = self.error()
        assert type(mine) is type(error) and str(mine) == str(error), \
            f'{note}: oracle {type(error).__name__} {digest(str(error))}, C {self.error_text()}'
        self.unsupported = str(error)

    def cycle(self, note):
        """One World.cycle on both sides: False when both raised the same Unsupported (the world stops there)."""
        code = self.lib.t_cycle(self.w)
        try:
            self.py.cycle()
        except Unsupported as error:
            self.raised(code, error, note)
            return False
        assert code == 0, f'{note}: C raised {self.error_text()}'
        self.check(note)
        self.capture()
        return True

    def op(self, py_op, c_op, note):
        """A between-cycle op (the agent's dispatch): False when both raised the same Unsupported."""
        try:
            py_op()
        except Unsupported as error:
            self.raised(c_op(), error, note)
            return False
        code = c_op()
        assert code == 0, f'{note}: C raised {self.error_text()}'
        self.check(note)
        return True

    # -- the player
    def sync_player(self):
        """The oracle player's levels, xp, inventories and varps into the C player."""
        p = self.p
        slots = lambda inv: np.asarray([v for it in inv.items for v in (it if it else (-1, 0))], np.int32)
        levels, base = np.asarray(p.levels, np.int16), np.asarray(p.base_levels, np.int16)
        stats = np.asarray(p.stats, np.int32)
        inv, worn = slots(p.inv), slots(p.worn)
        varps = np.asarray([-1 if p.varps[v] is None else p.varps[v] for v in H.header()[1]], np.int32)
        self.lib.t_set_player(self.w, ptr(levels), ptr(base), ptr(stats), ptr(inv), ptr(worn), ptr(varps),
                              int(p.afk_ready))
        self.lib.t_set_last(self.w, p.last_item, p.last_slot, p.last_useitem, p.last_useslot)

    def sync_last(self):
        p = self.p
        self.lib.t_set_last(self.w, p.last_item, p.last_slot, p.last_useitem, p.last_useslot)

    def set_timers(self, intervals):
        for i, (name, interval) in enumerate(zip(TIMERS, intervals)):
            fn = getattr(agent_mod, name)
            self.p.set_timer(name, interval, lambda fn=fn: fn(self.py, self.p))
            self.lib.t_set_timer(self.w, i, interval, self.py.tick, self.env.script[f'RS_S_{name.upper()}'], 0)

    def queue(self, path):
        self.p.queue_waypoints(path)
        xy = np.asarray([v for step in path for v in step] or [0], np.int16)
        self.lib.t_queue(self.w, len(path), ptr(xy))


# ---- the agent's dispatch, on both sides -------------------------------------------------------------------------
def approach(pair, target, entity):
    """The walk of agent.interact_loc / interact_npc / pick_up (act-v3): the client's route (Agent.route: a loc by its
    arrival, an npc beside its tile, an obj onto its tile, else beside it), the same turn points queued on both sides
    (a route that is only the player's tile queues none, as send_route); False when the oracle refuses the search
    (World.path's Unsupported: the agent would raise there)."""
    from sim import paths
    py, p = pair.py, pair.p
    if entity == 'loc':
        t = py.t.loc_types[target.type]
        arrivals = [paths.loc_arrival(target.shape, target.angle, t['width'], t['length'], t['forceapproach'])]
    else:
        arrivals = [paths.ENTITY_ARRIVAL] if entity == 'npc' else [None, paths.ENTITY_ARRIVAL]
    try:
        route = None
        for arrival in arrivals:
            route = route or paths.client_route(py.player_scene(p), p.x, p.z, (target.x, target.z), arrival)
    except Unsupported:
        return False
    if route is not None:
        pair.queue([] if list(route) == [(p.x, p.z)] else list(route))
    return True


def interact(pair, target, op, note, walk=True):
    """interact_loc / interact_npc / pick_up: approach, clear_pending_action, set_interaction(op). walk=False
    leaves out the approach (the player stays where it is)."""
    from sim.engine import Loc, Npc
    entity = 'loc' if isinstance(target, Loc) else 'npc' if isinstance(target, Npc) else 'obj'
    if walk and not approach(pair, target, entity):
        return
    pair.p.clear_pending_action()
    pair.lib.t_clear_pending(pair.w)
    ok = pair.p.set_interaction(target, op)
    if entity == 'loc':
        got = pair.lib.t_set_interaction_loc(pair.w, pair.lib.t_loc_slot(pair.w, target.index), op)
    elif entity == 'npc':
        got = pair.lib.t_set_interaction_npc(pair.w, target.nid, op)
    else:
        got = pair.lib.t_set_interaction_obj(pair.w, next(k for k, o in enumerate(pair.py.objs) if o is target), op)
    assert got == int(ok), note
    pair.sync_last()
    pair.check(note)


def run_held(pair, obj, trig, name, note):
    return pair.op(lambda: pair.py.run_held(pair.p, name), lambda: pair.lib.t_run_held_obj(pair.w, obj, trig), note)


def op_held(pair, slot, op, note):
    """agent.op_held: OPHELD{op+1} of the slot's obj with last_item and last_slot set."""
    p = pair.p
    item = p.inv.items[slot]
    if item is None:
        return True
    name = pair.py.t.obj_types[item[0]]['triggers'].get(f'OPHELD{op + 1}')
    if not name:
        return True
    p.last_item, p.last_slot = item[0], slot
    pair.sync_last()
    return run_held(pair, item[0], H.header()[0]['RS_OTRIG_HELD'] + op, name, note)


def use_on_held(pair, slot, using_slot, note):
    """agent.use_on_held: the OPHELDU lookup with its swaps, clear_pending_action, then the script found."""
    w, p = pair.py, pair.p
    item, used = p.inv.items[slot], p.inv.items[using_slot]
    if item is None or used is None:
        return True
    using = dict(id=used[0], slot=using_slot)
    p.last_item, p.last_slot, p.last_useitem, p.last_useslot = item[0], slot, using['id'], using['slot']

    def swap():
        p.last_item, p.last_useitem = p.last_useitem, p.last_item
        p.last_slot, p.last_useslot = p.last_useslot, p.last_slot

    def script(obj, category):
        name = w.t.obj_types[obj]['triggers'].get('OPHELDU')
        own = name is not None and not name.split(',')[1].startswith('_')
        return name if name and name != '[opheldu,_]' and own != category else None
    found, owner = script(item[0], False), item[0]
    if not found:
        found, owner = script(using['id'], False), using['id']
        swap()
    if not found and w.t.obj_types[item[0]]['category'] != -1:
        found, owner = script(item[0], True), item[0]
    if not found and w.t.obj_types[using['id']]['category'] != -1:
        found, owner = script(using['id'], True), using['id']
        swap()
    p.clear_pending_action()
    pair.lib.t_clear_pending(pair.w)
    pair.sync_last()
    if not found:
        pair.check(note)
        return True
    return run_held(pair, owner, H.header()[0]['RS_OTRIG_HELDU'], found, note)


def resume(pair, note):
    """agent.resume_dialog without choices: World.resume of a pausebutton script."""
    p = pair.p
    if p.script is None or p.script[1] != 'pausebutton' or p.resume_buttons:
        return True
    return pair.op(lambda: pair.py.resume(p), lambda: pair.lib.t_resume(pair.w), note)


# ---- the scenarios -------------------------------------------------------------------------------------------------
LEVELS = (1, 5, 10, 14, 15, 16, 20, 30)
ITEMS = ('bronze_axe', 'iron_axe', 'bronze_pickaxe', 'iron_pickaxe', 'net', 'fishing_rod', 'fishing_bait', 'shrimp',
         'bread', 'bones', 'feather', 'raw_chicken', 'logs', 'oak_logs', 'copper_ore', 'tin_ore', 'raw_shrimp',
         'coins', 'bronze_dagger', 'bronze_sword', 'bronze_sq_shield', 'bronze_med_helm', 'chefs_hat', 'brass_necklace')
TOOLS = ('bronze_axe', 'bronze_pickaxe', 'net')
# The scripts of each focus, by the oracle function's outer name (closures: mining, get_ore, gate).
FOCUS = dict(wood={'attempt_cut_tree', 'cut_tree', 'use_on_tree'}, mine={'mining', 'get_ore', 'prospect'},
             fish={'attempt_fish_shrimp', 'fish_shrimp', 'attempt_fish_sardine', 'use_on_saltfish'}, gate={'gate'})
COMBAT = (engine.player_combat_start, engine.player_combat_start_ap)


def xp_for(level):
    return 0 if level <= 1 else LEVEL_XP[level - 2]


def levels_strategy(data, stats):
    """Start levels of the gathering skills (and hitpoints), xp often just under the next level."""
    out = {}
    for stat in stats:
        level = data.draw(st.sampled_from(LEVELS), label=f'level {stat}')
        lo, hi = xp_for(level), xp_for(level + 1)
        under = data.draw(st.booleans(), label='under')
        xp = hi - data.draw(st.integers(1, min(400, hi - lo)), label='gap') if under and level < 99 else lo
        out[stat] = (level, xp)
    return out


def items_strategy(data, env):
    """An inventory (slot order) from ITEMS by id, with or without the focus's tool, sometimes full."""
    ids = [env.obj(name) for name in ITEMS if env.obj(name) is not None]
    tools = {env.obj(name) for name in TOOLS} - {None}
    picks = data.draw(st.lists(st.sampled_from(ids), max_size=10), label='items')
    if not data.draw(st.booleans(), label='no tools'):   # (Hypothesis draws False first)
        picks = [*tools, *picks]
    elif data.draw(st.booleans(), label='strip tools'):
        picks = [i for i in picks if i not in tools]
    fill = data.draw(st.sampled_from([0, 0, 0, 1, 2]), label='fill')  # 1: full, 2: one slot left
    junk = env.obj('logs')
    while fill and len(picks) < 28 - (fill - 1):
        picks.append(junk)
    rows = []
    for obj in picks[:28]:
        t = env.base.obj_types[obj]
        rows.append([obj, data.draw(st.integers(1, 40), label='stack') if t['stackable'] else 1])
    return rows


def wieldable(env):
    """Obj ids of ITEMS that have a wearpos (worn at the start sometimes)."""
    ids = [env.obj(name) for name in ITEMS if env.obj(name) is not None]
    return [i for i in ids if engine.wearpos(SimpleNamespace(t=env.base), i)]


def scripted(env, pair, target, op):
    """The oracle script an op on a target runs (None for none), its OP trigger first, as try_interact."""
    py = pair.py
    for trig in (op + engine.AP_TO_OP, op):
        name = py.trigger(target, trig)
        if name:
            return engine.SCRIPTS.get(name)
    return None


def loc_ops(pair, loc):
    """The ops interact_loc offers on a loc (visible ones), as APLOC1-based op ids."""
    raw = pair.py.t.loc_types[loc.type]['op'] or []
    return [APLOC1 + i for i, o in enumerate(raw) if o and o != 'hidden']


def npc_ops(pair, npc):
    raw = pair.py.t.npc_types[npc.type]['op'] or []
    return [APNPC1 + i for i, o in enumerate(raw) if o and o != 'hidden']


def focus_of(fn, focus):
    return fn is not None and fn.__qualname__.split('.')[0] in FOCUS.get(focus, ())


def live_locs(py):
    return [loc for loc in list(py.locs) + [l for row in py.loc_at.values() for l in row if not l.static] if loc.active]


def targets(env, pair, focus):
    """(target, op) pairs whose script is of the focus (all scripted, non-combat ones for 'mix')."""
    out = []
    for loc in live_locs(pair.py):
        for op in loc_ops(pair, loc):
            fn = scripted(env, pair, loc, op)
            if focus == 'mix' and fn is not None or focus_of(fn, focus):
                out.append((loc, op))
    for npc in pair.py.npcs:
        for op in npc_ops(pair, npc):
            fn = scripted(env, pair, npc, op)
            if fn not in COMBAT and (focus == 'mix' and fn is not None or focus_of(fn, focus)):
                out.append((npc, op))
    return out


def near_tile(pair, target, data):
    """A walkable tile within a few tiles of the target (the player starts there)."""
    sources = [(x, z) for x, z, _ in pair.tables.meta['sources']]
    near = [(x, z) for x, z in sources if max(abs(x - target.x), abs(z - target.z)) <= 4
            and (x, z) not in {(n.x, n.z) for n in pair.py.npcs}]
    return data.draw(st.sampled_from(near or sources), label='start tile')


def start(env, data, focus):
    """A pair on a drawn layout, the player near a target of the focus, with drawn levels, items, delays and timers."""
    pens = [k for k, L in enumerate(env.layouts['compact']) if L.gate[0] >= 0]
    if focus == 'wood' and data.draw(st.booleans(), label='supply'):
        kind, k = 'supply', data.draw(st.integers(0, len(env.layouts['supply']) - 1), label='supply layout')
    else:
        kind = 'compact'
        k = data.draw(st.sampled_from(pens) if focus == 'gate' else st.integers(0, len(env.layouts['compact']) - 1),
                      label='layout')
    seed = data.draw(st.integers(0, 2 ** 31 - 1), label='seed')
    start_tick = data.draw(st.one_of(st.integers(6, 300), st.integers(480, 499)), label='start tick')
    xp_rate = data.draw(st.sampled_from([25, 25, 25, 1, 5]), label='xp rate')
    tables = env.tables(kind, k)
    probe = World(tables, seed=seed, start_tick=start_tick, npcs=None, xp_rate=xp_rate)
    probe_pair = SimpleNamespace(py=probe, tables=tables)
    pool = targets(env, probe_pair, focus)
    sources = [(x, z) for x, z, _ in tables.meta['sources']]
    first = data.draw(st.sampled_from(pool), label='first target') if pool else None
    tile = near_tile(probe_pair, first[0], data) if first else data.draw(st.sampled_from(sources), label='start tile')
    pair = Pair(env, kind, k, seed, start_tick, xp_rate, tile)
    if first:   # the same loc or npc of the pair's oracle world
        target, op = first
        found = pair.py.npc_by_nid[target.nid] if isinstance(target, engine.Npc) else pair.py.locs[target.index]
        first = found, op
    p = pair.p
    stats = levels_strategy(data, (WC, FISH, MINE))
    for stat, (level, xp) in stats.items():
        p.levels[stat] = p.base_levels[stat] = level
        p.stats[stat] = xp
    p.levels[HP] = data.draw(st.integers(1, 10), label='hp')
    if data.draw(st.booleans(), label='boosted'):
        p.levels[WC] += data.draw(st.integers(-1, 2), label='boost')   # the timers restore it
    for slot, item in enumerate(items_strategy(data, env)):
        p.inv.items[slot] = item
    if data.draw(st.booleans(), label='worn'):
        obj = data.draw(st.sampled_from(wieldable(env)), label='worn obj')
        p.worn.items[env.base.obj_types[obj]['wearpos']] = [obj, 1]
    p.varps['action_delay'] = pair.py.tick + data.draw(st.integers(-4, 6), label='action delay')
    p.varps['eat_delay'] = pair.py.tick + data.draw(st.integers(-3, 3), label='eat delay')
    p.afk_ready = data.draw(st.integers(0, 30), label='afk') == 30
    pair.sync_player()
    assert pair.op(lambda: engine.player_combat_stat(pair.py, p), lambda: pair.lib.t_combat_stat(pair.w), 'login stat')
    if data.draw(st.booleans(), label='timers'):
        pair.set_timers([data.draw(st.integers(1, 12), label='interval') for _ in TIMERS])
    pair.check('start')
    return pair, first


def held_slots(pair):
    return [k for k, it in enumerate(pair.p.inv.items) if it]


def act(env, pair, data, focus, note):
    """One agent dispatch drawn for the focus; False when the world raised (both sides alike)."""
    p = pair.p
    weights = dict(target=6, use_target=2, held=2, drop=1, use_held=1, pickup=2 if pair.py.objs else 0, resume=2,
                   wait=1, other=1)
    if focus == 'held':
        weights.update(target=1, use_target=1, held=6, drop=3, use_held=3, pickup=4 if pair.py.objs else 0)
    kinds = [k for k, n in weights.items() for _ in range(n)]
    kind = data.draw(st.sampled_from(kinds), label='act')
    note = f'{note} {kind}'
    if kind in ('target', 'use_target', 'other'):
        pool = targets(env, pair, focus if kind != 'other' else 'mix')
        if kind == 'other':   # any op or use on a loc or npc near the player, scripted or not (default_op, refusals)
            pool = [(loc, op) for loc in live_locs(pair.py) if max(abs(loc.x - p.x), abs(loc.z - p.z)) <= 8
                    for op in (*loc_ops(pair, loc), APLOCU)]
            pool += [(npc, APNPCU) for npc in pair.py.npcs if max(abs(npc.x - p.x), abs(npc.z - p.z)) <= 8]
        near = [(t, op) for t, op in pool if max(abs(t.x - p.x), abs(t.z - p.z)) <= 10]
        if near and data.draw(st.integers(0, 3), label='far') < 3:
            pool = near
        if not pool:
            return True
        target, op = data.draw(st.sampled_from(pool), label='target')
        if kind == 'use_target':
            op = APLOCU if op >= APLOC1 else APNPCU
        if op in (APLOCU, APNPCU):
            slots = held_slots(pair)
            if not slots:
                return True
            p.last_useitem = p.inv.items[data.draw(st.sampled_from(slots), label='used slot')][0]
        if scripted(env, pair, target, op) in COMBAT:
            return True
        interact(pair, target, op, note)
        return True
    if kind == 'pickup':
        obj = data.draw(st.sampled_from(pair.py.objs), label='ground obj')
        ops = pair.py.t.obj_types[obj.type]['op'] or []
        interact(pair, obj, APOBJ1 + next((i for i, o in enumerate(ops) if o and 'take' in o.lower()), 2), note)
        return True
    slots = held_slots(pair)
    if kind == 'held' and slots:
        slot = data.draw(st.sampled_from(slots), label='slot')
        iop = pair.py.t.obj_types[p.inv.items[slot][0]]['iop'] or []
        ops = [i for i, o in enumerate(iop) if o and o != 'hidden']
        return op_held(pair, slot, data.draw(st.sampled_from(ops), label='iop'), note) if ops else True
    if kind == 'drop' and slots:
        slot = data.draw(st.sampled_from(slots), label='slot')
        if not pair.py.t.obj_types[p.inv.items[slot][0]]['triggers'].get('OPHELD5'):
            return True
        return op_held(pair, slot, 4, note)
    if kind == 'use_held' and len(slots) > 1:
        a, b = data.draw(st.lists(st.sampled_from(slots), min_size=2, max_size=2, unique=True), label='use slots')
        return use_on_held(pair, a, b, note)
    if kind == 'resume':
        return resume(pair, note)
    return True


def busy(p):
    """The player is in an interaction, a walk, a script or its delay, or has engine requests (a levelup) waiting."""
    return p.target is not None or bool(p.path) or p.script is not None or p.delayed or bool(p.engine_queue)


def settle(pair, data, note, budget):
    """Cycles while the player is busy (up to budget ticks), then a few more; a paused dialog is continued now and then.
    False when the world raised (both sides alike)."""
    extra = data.draw(st.integers(0, 4), label='extra ticks')
    for tick in range(budget):
        p = pair.p
        if p.script is not None and p.script[1] == 'pausebutton' and data.draw(st.integers(0, 3), label='wait') < 3:
            if not resume(pair, f'{note} tick {tick} resume'):
                return False
        if not pair.cycle(f'{note} tick {tick}'):
            return False
        if not busy(pair.p):
            extra -= 1
            if extra < 0:
                break
    return True


def run(env, data, focus, steps=(1, 6), budget=60):
    """A scenario: start, the first target's op, then drawn dispatches, each followed by cycles until the player is
    idle (every cycle compared), as the agent's options run their scripts to their ends."""
    pair, first = start(env, data, focus)
    try:
        for step in range(data.draw(st.integers(*steps), label='steps')):
            if step == 0 and first is not None:
                target, op = first
                interact(pair, target, op, 'step 0 first target')
            elif not act(env, pair, data, focus, f'step {step}'):
                return
            if not settle(pair, data, f'step {step}', budget):
                return
    finally:
        pair.close()


SETTINGS = dict(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


@settings(max_examples=100, **SETTINGS)
@given(st.data())
def test_woodcutting_equals_the_oracle(env, data):
    """attempt_cut_tree, cut_tree (op3 from the cadence), get_logs with its three keys, use_on_tree, on normal trees and
    oaks of compact and supply layouts: the level refusal at 14 against the oak's 15, the full inventory, no axe (on the
    second click for a level-0 tree), depletion into stumps, the afk Unsupported, and the levelup dialogs (with the oak
    unlock's second dialog) the xp brings."""
    run(env, data, 'wood', steps=(1, 4), budget=80)


@settings(max_examples=100, **SETTINGS)
@given(st.data())
def test_mining_equals_the_oracle(env, data):
    """mining first and continue for normal and fast rocks with get_ore (the gem roll, the normal rock's p_delay before
    the ore), prospect's two delays, the empty-rock message, and the mesbox refusals (full, level, no pickaxe) with
    their continues, through two frame levels."""
    run(env, data, 'mine', steps=(1, 4), budget=80)


@settings(max_examples=100, **SETTINGS)
@given(st.data())
def test_fishing_equals_the_oracle(env, data):
    """attempt_fish_shrimp, fish_shrimp, fish_roll (anchovies from 15), the Bait op's level and rod refusals (a rod
    with bait is not supported), and use_on_saltfish: four frame levels deep for a net-less use with a net elsewhere,
    the fish_full mesbox, the levelup dialogs."""
    run(env, data, 'fish', steps=(1, 4), budget=80)


@settings(max_examples=50, **SETTINGS)
@given(st.data())
def test_pen_gates_equal_the_oracle(env, data):
    """The four gate scripts on pen layouts: a main gate swings itself and its outer half, an outer half swings the main
    one beside it; opened stages close again; the added locs, zone orders, loc events, collision and changed tiles."""
    run(env, data, 'gate', steps=(1, 5), budget=20)


@settings(max_examples=150, **SETTINGS)
@given(st.data())
def test_held_scripts_and_pickup_equal_the_oracle(env, data):
    """wield and try_equip's swaps and conflicts, eat with its delays, heals and messages, bury_bones over its delay,
    drop_slot then pickup_obj (from the tile at once, from beside it after a delay, refused when full), and the three
    use_default scripts, with the stat timers and changestat running between."""
    run(env, data, 'held', steps=(2, 8), budget=12)


@settings(max_examples=100, **SETTINGS)
@given(st.data())
def test_mixed_play_equals_the_oracle(env, data):
    """Every scripted op of the layout and plain ops and uses (default_op), in any order, with levelups and timers."""
    run(env, data, 'mix', steps=(2, 8), budget=40)


# ---- fixed scenarios: each case the port must meet, run to its end on both sides ----------------------------------
def roles(env, kind, k):
    """The layout's targets by role, from the oracle's own trigger tables: [(loc index or npc nid, op)] for 'tree0' and
    'tree15' (by levelrequired), 'rock' and 'fastrock' (mining's get_ore closure), 'spot' (net fishing) and the closed
    'gate_main' and 'gate_outer' halves."""
    import inspect
    tables = env.tables(kind, k)
    probe = SimpleNamespace(py=World(tables, npcs=None), tables=tables)
    trees = tables.meta['content']['trees']
    out = {}
    for loc in probe.py.locs:
        fn = scripted(env, probe, loc, APLOC1)
        if fn is None:
            continue
        name = fn.__qualname__.split('.')[0]
        cells = inspect.getclosurevars(fn).nonlocals if fn.__closure__ else {}
        if name == 'attempt_cut_tree':
            role = 'tree0' if trees[tables.loc_types[loc.type]['debugname']]['levelrequired'] == 0 else 'tree15'
        elif name == 'mining':
            role = 'fastrock' if inspect.getclosurevars(cells['get_ore']).nonlocals['fast'] else 'rock'
        elif name == 'gate' and cells['opening']:
            role = 'gate_outer' if cells['outer'] else 'gate_main'
        else:
            continue
        out.setdefault(role, []).append(loc.index)
    for npc in probe.py.npcs:
        if scripted(env, probe, npc, APNPC1) is engine.attempt_fish_shrimp:
            out.setdefault('spot', []).append(npc.nid)
    return out


def resolve(pair, role, ident):
    return pair.py.npc_by_nid[ident] if role == 'spot' else pair.py.locs[ident]


def beside(pair, target, distance):
    """The first walkable tile (by distance, x, z) at a Chebyshev distance of `distance` or more from the target."""
    tiles = sorted((max(abs(x - target.x), abs(z - target.z)), x, z) for x, z, _ in pair.tables.meta['sources']
                   if (x, z) not in {(n.x, n.z) for n in pair.py.npcs})
    return next((x, z) for d, x, z in tiles if d >= distance)


def fixed(env, role, levels=None, items=(), kind='compact', k=0, seed=11, start_tick=40, xp_rate=25, distance=1,
          index=0, gap=None):
    """A pair with the player `distance` tiles from a target of the role, levels {stat: level} (xp `gap` under the
    next level when given), items [(name, count)] in slot order; the login's player_combat_stat run on both sides."""
    ident = roles(env, kind, k)[role][index]
    probe = SimpleNamespace(py=World(env.tables(kind, k), npcs=None), tables=env.tables(kind, k))
    tile = beside(probe, resolve(probe, role, ident), distance)
    pair = Pair(env, kind, k, seed, start_tick, xp_rate, tile)
    p = pair.p
    for stat, level in (levels or {}).items():
        p.levels[stat] = p.base_levels[stat] = level
        p.stats[stat] = xp_for(level + 1) - gap if gap else xp_for(level)
    for slot, (name, count) in enumerate(items):
        p.inv.items[slot] = [env.obj(name), count]
    pair.sync_player()
    assert pair.op(lambda: engine.player_combat_stat(pair.py, p), lambda: pair.lib.t_combat_stat(pair.w), 'login stat')
    return pair, resolve(pair, role, ident)


def drive(pair, target, op, note, budget=120, use=None, after=None):
    """The agent's op (or use of item `use`) on a target, then cycles until the player is idle, every pausebutton
    dialog continued at once; after(pair, tick) runs between cycles (a dispatch mid-walk). False when both raised."""
    if use is not None:
        pair.p.last_useitem = pair.env.obj(use)
    interact(pair, target, op, note)
    for tick in range(budget):
        p = pair.p
        if after is not None and not after(pair, tick):
            return False
        if p.script is not None and p.script[1] == 'pausebutton' and not resume(pair, f'{note} tick {tick} resume'):
            return False
        if not pair.cycle(f'{note} tick {tick}'):
            return False
        if not busy(pair.p):
            return True
    return True


def op_of(env, pair, target, fn):
    """The op id whose script is fn on that target (interact_loc/npc's visible ops)."""
    ops = loc_ops(pair, target) if isinstance(target, engine.Loc) else npc_ops(pair, target)
    return next(op for op in ops if scripted(env, pair, target, op) is fn)


KIT = [('bronze_axe', 1), ('net', 1), ('bronze_pickaxe', 1), ('shrimp', 1), ('shrimp', 1), ('bread', 1)]
FULL = [('logs', 1)] * 28


@pytest.mark.parametrize('kind, k, role, stat', [
    ('compact', 0, 'tree0', WC), ('compact', 3, 'tree0', WC), ('supply', 0, 'tree0', WC), ('supply', 2, 'tree0', WC),
    ('compact', 0, 'spot', FISH), ('compact', 5, 'spot', FISH), ('compact', 0, 'rock', MINE), ('compact', 3, 'rock', MINE)])
def test_levelup_dialogs_cross_the_unlock_level(env, kind, k, role, stat):
    """Level 14 one xp unit under 15: the first catch levels up to 15 (the oak, anchovy and iron unlocks); the levelup
    dialog's continue opens the unlock's objbox, doubleobjbox or mesbox, whose continue closes the chat; then gathering
    goes on (oaks for woodcutting: refused at 14, cut at 15)."""
    pair, target = fixed(env, role, {stat: 14}, KIT, kind=kind, k=k, gap=1, xp_rate=1)   # one level, not past 15
    try:
        op = APLOC1 if role != 'spot' else APNPC1
        if role == 'tree0':
            oak = resolve(pair, 'tree15', roles(env, kind, k)['tree15'][0])
            assert drive(pair, oak, APLOC1, 'oak at 14')
            assert pair.p.base_levels[WC] == 14
        assert drive(pair, target, op, 'to 15')
        assert pair.p.base_levels[stat] >= 15, 'the scenario did not level up'
        dialogs = {chain for chain in pair.seen if chain[0].startswith('LEVELUP')}
        assert ('LEVELUP:1',) in dialogs and any(len(chain) == 2 for chain in dialogs), sorted(dialogs)
        if role == 'tree0':
            assert drive(pair, oak, APLOC1, 'oak at 15')
        else:
            assert drive(pair, target, op, 'at 15')
    finally:
        pair.close()


REFUSALS = [
    # (role, levels, items, op script, use, expected frame chain or Unsupported, distance)
    ('tree0', {WC: 1}, [], 'attempt_cut_tree', None, None, 2),          # level 0: the first click skips the axe check
    ('tree15', {WC: 15}, [], 'attempt_cut_tree', None, None, 1),
    ('tree15', {WC: 14}, KIT, 'attempt_cut_tree', None, None, 1),
    ('tree0', {WC: 1}, KIT[:1] + FULL[:27], 'attempt_cut_tree', None, None, 1),
    ('tree0', {WC: 1}, KIT, 'attempt_cut_tree', 'bronze_axe', None, 3),  # use_on_tree: an axe starts op1
    ('tree0', {WC: 1}, KIT, 'attempt_cut_tree', 'logs', None, 1),        # anything else: the default message
    ('rock', {MINE: 1}, [], 'mining', None, ('MINE_FIRST_NORMAL:3', 'MESBOX:1'), 1),
    ('rock', {MINE: 1}, KIT[2:3] + FULL[:27], 'mining', None, ('MINE_FIRST_NORMAL:1', 'MESBOX:1'), 1),
    ('fastrock', {MINE: 1}, KIT, 'mining', None, ('MINE_FIRST_FAST:2', 'MESBOX:1'), 1),
    ('rock', {MINE: 1}, KIT, 'prospect', None, ('PROSPECT:2',), 1),
    ('spot', {FISH: 1}, [], 'attempt_fish_shrimp', None, ('ATTEMPT_FISH_SHRIMP:1', 'CHECK_FISH_EQUIPMENT:1', 'MESBOX:1'), 1),
    ('spot', {FISH: 1}, KIT[1:2] + FULL[:27], 'attempt_fish_shrimp', None, ('ATTEMPT_FISH_SHRIMP:2', 'MESBOX:1'), 1),
    ('spot', {FISH: 1}, KIT, 'attempt_fish_sardine', None, ('ATTEMPT_FISH_SARDINE:1', 'MESBOX:1'), 1),
    ('spot', {FISH: 5}, KIT, 'attempt_fish_sardine', None,
     ('ATTEMPT_FISH_SARDINE:2', 'CHECK_FISH_EQUIPMENT:1', 'MESBOX:1'), 1),
    ('spot', {FISH: 5}, [('fishing_rod', 1)], 'attempt_fish_sardine', None,
     ('ATTEMPT_FISH_SARDINE:2', 'CHECK_FISH_EQUIPMENT:2', 'MESBOX:1'), 1),
    ('spot', {FISH: 5}, [('fishing_rod', 1), ('fishing_bait', 5)], 'attempt_fish_sardine', None, 'unsupported', 1),
    ('spot', {FISH: 1}, KIT[1:2] + FULL[:27], 'attempt_fish_shrimp', 'net',
     ('USE_ON_SALTFISH:1', 'ATTEMPT_FISH_SHRIMP:2', 'MESBOX:1'), 1),
    ('spot', {FISH: 1}, KIT, 'attempt_fish_shrimp', 'logs', None, 1),
]


@pytest.mark.parametrize('case', range(len(REFUSALS)))
def test_refusals_and_uses_equal_the_oracle(env, case):
    """Each refusal in its script's order (level, space, tool, the bait and rod checks) and the use scripts: missing
    axe (on the second click of a level-0 tree), oak level, full inventory, no pickaxe, an iron rock's level, no net,
    the Bait op's level, rod and bait checks (a rod with bait is not supported), a net used on a spot with no space,
    an axe or logs used on a tree, logs on a spot; the frame chain each mesbox suspends in is reached."""
    role, levels, items, fn, use, expected, distance = REFUSALS[case]
    if any(env.obj(name) is None for name, _ in items):
        pytest.skip('an item the archive lacks')
    pair, target = fixed(env, role, levels, items, distance=distance)
    try:
        fn = {'mining': None}.get(fn, getattr(engine, fn, None))
        if use is not None:
            op = APLOCU if isinstance(target, engine.Loc) else APNPCU
        elif fn is None:   # mining's first-swing closure: op1
            op = APLOC1
        else:
            op = op_of(env, pair, target, fn)
        done = drive(pair, target, op, f'case {case}', use=use)
        if expected == 'unsupported':
            assert not done and pair.unsupported is not None
        else:
            assert done
            if expected is not None:
                assert expected in pair.seen, sorted(pair.seen)
    finally:
        pair.close()


def test_a_net_dropped_mid_walk_reaches_four_frame_levels(env):
    """A net used on a spot from afar, then dropped while the player walks: use_on_saltfish runs attempt_fish_shrimp,
    whose net check opens a mesbox, four levels deep (RS_FRAME_DEPTH), continued and closed."""
    pair, spot = fixed(env, 'spot', {FISH: 1}, KIT, distance=4)
    try:
        net = next(k for k, it in enumerate(pair.p.inv.items) if it and it[0] == env.obj('net'))
        dropped = []

        def drop(pair, tick):
            if dropped:
                return True
            dropped.append(tick)
            return op_held(pair, net, 4, 'drop the net')
        assert drive(pair, spot, APNPCU, 'net on spot', use='net', after=drop)
        assert ('USE_ON_SALTFISH:1', 'ATTEMPT_FISH_SHRIMP:1', 'CHECK_FISH_EQUIPMENT:1', 'MESBOX:1') in pair.seen, \
            sorted(pair.seen)
    finally:
        pair.close()


@pytest.mark.parametrize('role, stat', [('tree0', WC), ('tree15', WC), ('rock', MINE), ('fastrock', MINE)])
def test_depletion_and_respawn_equal_the_oracle(env, role, stat):
    """Gathering at level 30 until the tree turns into a stump or the rock empties (change_loc with the scaled respawn),
    then op1 and prospect on the empty rock (mine_none, prospect_empty), then idle until the respawn reverts it."""
    pair, target = fixed(env, role, {stat: 30}, KIT, xp_rate=1)
    try:
        base = target.type
        for attempt in range(6):
            assert drive(pair, target, APLOC1, f'gather {attempt}', budget=200)
            if target.type != base:
                break
        assert target.type != base, 'the target never depleted'
        if role in ('rock', 'fastrock'):
            assert drive(pair, target, APLOC1, 'empty rock')
            prospect = next(op for op in loc_ops(pair, target) if scripted(env, pair, target, op) is engine.prospect)
            assert drive(pair, target, prospect, 'prospect the empty rock')
        for tick in range(260):
            assert pair.cycle(f'respawn tick {tick}')
            if target.type == base:
                break
        assert target.type == base, 'the target never respawned'
    finally:
        pair.close()


def test_gem_rolls_equal_the_oracle(env):
    """get_ore's 1-in-gem_chance branch (key 4 == 1), which skips the Mining roll and draws the gem roll (key 6): seeds
    of a level-1 miner are tried until the oracle takes it, and that run is compared (a gem produced, or none)."""
    for seed in (2, 11, 40, 42, 44, *range(100, 140)):   # the first five took it in an oracle-only search
        pair, rock = fixed(env, 'rock', {MINE: 1}, [('bronze_pickaxe', 1)], seed=seed, xp_rate=1)
        try:
            for attempt in range(40):
                if 6 in pair.keys or not drive(pair, rock, APLOC1, f'seed {seed} attempt {attempt}', budget=40):
                    break
            if 6 in pair.keys:
                return
        finally:
            pair.close()
    raise AssertionError('no seed took the gem branch')


@pytest.mark.parametrize('k', [0, 2, 4, 6])
def test_pen_gates_open_and_close(env, k):
    """A main gate opened (both halves swing to their open stages), closed through its open stage, opened through the
    outer half and closed through the outer open stage, then the 500-tick lifecycle that restores the closed gate."""
    pair, gate = fixed(env, 'gate_main', {}, KIT, k=k)
    try:
        cats = pair.py.t.meta['content']['categories']

        def stage(category):
            return next(loc for loc in live_locs(pair.py) if pair.py.t.loc_types[loc.type]['category'] == cats[category])
        assert drive(pair, gate, APLOC1, 'open main')
        assert drive(pair, stage('gate_main_open'), APLOC1, 'close main')
        assert drive(pair, stage('gate_outer_closed'), APLOC1, 'open outer')
        assert drive(pair, stage('gate_outer_open'), APLOC1, 'close outer')
        assert pair.py._next_loc >= pair.py.n_stored_locs + 4, 'the gates did not swing twice'
        for tick in range(505):
            assert pair.cycle(f'lifecycle tick {tick}')
    finally:
        pair.close()


def test_held_items_and_pickup_equal_the_oracle(env):
    """Eat at low hitpoints (heal, messages) and again within eat_delay (refused silently); wield a dagger, then a sword
    (the dagger back to the first free slot), a shield and a helm; drop and pick up from the tile, and from beside it
    (a p_delay first); bury bones over their delay; the use_default scripts (feather, logs, a stab weapon)."""
    items = [('shrimp', 1), ('bread', 1), ('bronze_dagger', 1), ('bronze_sword', 1), ('bronze_sq_shield', 1),
             ('bronze_med_helm', 1), ('bones', 1), ('feather', 5), ('logs', 1), ('oak_logs', 1)]
    if any(env.obj(name) is None for name, _ in items):
        pytest.skip('an item the archive lacks')
    pair, _ = fixed(env, 'tree0', {HP: 10}, items, distance=3)
    try:
        p = pair.p
        p.levels[HP] = 3
        pair.sync_player()
        slot = lambda name: next(k for k, it in enumerate(p.inv.items) if it and it[0] == env.obj(name))
        held_op = lambda name, fn: next(k for k in range(5) if engine.held_script(
            pair.py, pair.py.t.obj_types[env.obj(name)]['triggers'].get(f'OPHELD{k + 1}', '')) is fn)

        def settle_all(note, n=4):
            for tick in range(n):
                if p.script is not None and p.script[1] == 'pausebutton':
                    assert resume(pair, f'{note} resume')
                assert pair.cycle(f'{note} tick {tick}')
        assert op_held(pair, slot('shrimp'), held_op('shrimp', engine.eat), 'eat shrimp')
        assert op_held(pair, slot('bread'), held_op('bread', engine.eat), 'eat bread at once')
        settle_all('after eating')
        assert op_held(pair, slot('bread'), held_op('bread', engine.eat), 'eat bread later')
        for name in ('bronze_dagger', 'bronze_sword', 'bronze_sq_shield', 'bronze_med_helm'):
            assert op_held(pair, slot(name), held_op(name, engine.wield), f'wield {name}')
            settle_all(f'after {name}', 1)
        assert any(e[1] == 'transfer' and e[2]['src'] == 'worn' for e in pair.py.events), 'no swap back'
        assert op_held(pair, slot('logs'), 4, 'drop logs')
        settle_all('dropped', 1)
        interact(pair, pair.py.objs[-1], APOBJ1 + 2, 'pick up on the tile')
        settle_all('picked', 2)
        assert op_held(pair, slot('logs'), 4, 'drop logs again')
        settle_all('dropped again', 1)
        here = (p.x, p.z)
        step = next((x, z) for x, z, _ in pair.tables.meta['sources'] if abs(x - here[0]) + abs(z - here[1]) == 1)
        pair.queue([step])
        settle_all('stepped', 2)
        interact(pair, next(o for o in pair.py.objs if (o.x, o.z) == here), APOBJ1 + 2, 'pick up from beside',
                 walk=False)
        settle_all('picked from beside', 4)
        assert ('PICKUP_OBJ:1',) in pair.seen, sorted(pair.seen)
        assert sum(e[1] == 'pickup' for e in pair.py.events) == 2
        assert op_held(pair, slot('bones'), held_op('bones', engine.bury_bones), 'bury')
        settle_all('buried', 3)
        assert ('BURY_BONES:1',) in pair.seen
        for a, b in (('feather', 'logs'), ('logs', 'oak_logs'), ('logs', 'bronze_sword')):
            if any(it and it[0] == env.obj(a) for it in p.inv.items) and any(it and it[0] == env.obj(b) for it in p.inv.items):
                assert use_on_held(pair, slot(a), slot(b), f'use {a} on {b}')
                settle_all(f'used {a}', 1)
    finally:
        pair.close()


@pytest.mark.parametrize('worn, wield, fill', [('bronze_dagger', 'bronze_sword', 27), ('bronze_dagger', 'bronze_sword', 26),
                                               ('bronze_sq_shield', 'bronze_sq_shield', 27)])
def test_equip_swaps_at_the_edge_of_a_full_inventory(env, worn, wield, fill):
    """try_equip's space rule (conflicts > freespace + 1 refuses): a worn weapon swapped for another with no free slot
    (the item coming off takes the slot the new one leaves), with one free slot, and two of one shield."""
    if env.obj(worn) is None or env.obj(wield) is None:
        pytest.skip('an item the archive lacks')
    pair, _ = fixed(env, 'tree0', {}, [(wield, 1)] + [('logs', 1)] * fill, distance=3)
    try:
        p = pair.p
        p.worn.items[env.base.obj_types[env.obj(worn)]['wearpos']] = [env.obj(worn), 1]
        pair.sync_player()
        assert op_held(pair, 0, next(k for k in range(5) if engine.held_script(
            pair.py, pair.py.t.obj_types[env.obj(wield)]['triggers'].get(f'OPHELD{k + 1}', '')) is engine.wield), 'wield')
        for tick in range(2):
            assert pair.cycle(f'tick {tick}')
        assert any(e[1] == 'transfer' for e in pair.py.events), 'the swap was refused'
    finally:
        pair.close()


def test_stat_timers_and_changestat_equal_the_oracle(env):
    """The three login timers with short intervals: stat_regen raises drained stats, stat_boost_restore lowers boosted
    ones (prayer and hitpoints excepted), health_regen heals a hitpoint and queues [changestat,_] each time."""
    pair, _ = fixed(env, 'tree0', {}, KIT)
    try:
        p = pair.p
        p.levels[WC], p.levels[FISH], p.levels[MINE] = 4, 0, 1
        p.base_levels[MINE] = p.base_levels[FISH] = 3
        p.levels[SKILLS.index('attack')] = 3
        p.levels[HP] = 4
        pair.sync_player()
        pair.set_timers([2, 3, 1])
        for tick in range(30):
            assert pair.cycle(f'tick {tick}')
        assert p.levels[HP] == 10 and p.levels[WC] == 1 and p.levels[MINE] == 3
    finally:
        pair.close()


def test_unported_scripts_raise_not_ported(env):
    """rs_script_step's contract: an RS_NP script raises RS_E_NOT_PORTED_SCRIPT with the trigger name it was run by
    (a gap, never an Unsupported). A failed execute leaves p.protect set, as the oracle's would: one world each."""
    name = env.sid('[oploc1,bankbooth]')
    for script in ('RS_S_BOOTH_USE', 'RS_S_DOOR_OPEN', 'RS_S_TALK_TO_BANKER'):
        pair = Pair(env, 'compact', 0, 1, 10, 25, (ORIGIN[0] + 10, ORIGIN[1] + 10))
        try:
            assert pair.lib.t_execute(pair.w, env.script[script], -1, name) == 1
            error = pair.error()
            assert isinstance(error, native.NotPorted) and str(error) == 'native port lacks script [oploc1,bankbooth]'
        finally:
            pair.close()
