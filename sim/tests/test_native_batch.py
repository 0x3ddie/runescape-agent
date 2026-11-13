"""The native episode, batch, canon and binding (sim/native/task.c's episode functions, batch.c, canon.c,
module.c's Batch, sim/native.py NativeBatch and the workers' native engine) against the oracle.

Here NativeBatch runs on a build whose agent and observation (agent.c, obs.c and text.c) are a replay double
(REPLAY_C): the oracle's CpuBatch plays keyed random legal actions
first, under sim/native_harness.py's PythonSide (its tick Rows and the harness Canon), and every Agent.start and
Agent.step it makes is recorded cycle by cycle where wait_ticks captures text (the world tick, the player's
hitpoints, death flag, tile and inventories, the events, the map locs) and at its end (the projection GoalEpisode.project
makes of it: rl/obs's 31 arrays, the task's three left unset; heldSel, bankOpen and the visited cells). The double's
rs_agent_start and rs_agent_step check the call they receive against the recorded one (the start's tile, kit and
levels, each step's action and the hitpoints it starts from) and apply the recorded states cycle by cycle, calling the
hooks agent.c's wait_ticks will call (rs_task_watch, then rs_trace_tick) and stopping when rs_ended says the episode
ended; its rs_project hands back the recorded projection. Everything else is the real C on the real pack: the keyed
draws, the World built for each start, the setup (the kit, the combat start's hitpoints, STYLE_0, the Wield slot),
arm, the watch, the payment and settlement, the ledgers, audit, facts and the SupplyTracker, the task's three
projected arrays, the batch loop with its auto-reset, unsupported flagging, evidence and output rows, the thread pool
and the tick traces. So this checks the episode code's own behavior against CpuBatch's outputs bit for bit and its
per-tick episode record against the harness Canon's; sim/native_harness.py runs the same comparisons on the real
agent.

Private data stays private: assertions report ids, counts and hashes, never strings; the pack, pools and archive are
read in place and skip when absent."""
import ctypes
import hashlib
import importlib.machinery
import importlib.util
import shlex
import subprocess
import sysconfig

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from sim import native
from sim import native_harness as H
from sim.build import DATA
from sim.tests import PACK

ROOT = DATA.parents[3]
POOL, SUPPLY = ROOT / 'logs/sim/world/train-8-nc1-derived.npz', ROOT / 'logs/supply/pools/train-128.npz'
KEY = 11
START, STEP = 0, 1  # replay call kinds
UNSUPPORTED = 'player left the supported area at 0,1,2'  # the injected Unsupported (rs.h RS_E_PLAYER_LEFT 0, 1, 2)

# The replay double: agent.c, obs.c and text.c's rs.h entry points, replaying the oracle's recorded episodes (module
# docstring). A tape per (global slot, episode): [n_calls, calls...]; a call: kind, action, raise (1: an Unsupported at
# once, nothing follows), then for START tile x, z, the kit (n, then obj, count) and the 7 levels, for STEP the
# hitpoints the step starts from; then n cycles, each: tick, hp, base hp, dead, x, z, events (n, 11 each),
# inventory, worn, bank (n, then slot, id, count each), map locs (n, then slot, type, active each); then the end: bank
# open, heldSel (n, then slot, id, count, name, option each), visited bits, visited cells off the layout, events after
# the last cycle (n, 11 each), and the projected row (RS_OBS_BYTES / 8 words).
REPLAY_C = r'''
#include <stdlib.h>
#include <string.h>
#include "rs.h"

#define RT_TAPES 8192
#define RT_SLOTS 256
typedef struct { int32_t slot, episode; int64_t n; int64_t *data; } Tape;
static Tape TAPES[RT_TAPES];
static int N_TAPES;
typedef struct { const int64_t *at, *end; int64_t calls; _Alignas(8) uint8_t proj[RS_OBS_BYTES]; } Play;
static Play PLAY[RT_SLOTS];

int rt_put(int32_t slot, int32_t episode, const int64_t *data, int64_t n) {
    if (N_TAPES >= RT_TAPES || slot < 0 || slot >= RT_SLOTS) return 1;
    Tape *t = &TAPES[N_TAPES++];
    t->slot = slot, t->episode = episode, t->n = n;
    t->data = malloc((size_t)n * sizeof *data);
    if (!t->data) return 1;
    memcpy(t->data, data, (size_t)n * sizeof *data);
    return 0;
}

void rt_clear(void) {
    for (int i = 0; i < N_TAPES; i++) free(TAPES[i].data);
    N_TAPES = 0;
    memset(PLAY, 0, sizeof PLAY);
}

static int bad(RsSlot *s, const char *what) { return RS_FAIL_TEXT(&s->w.err, RS_E_INTERNAL, what); }
#define NEXT(p) ((p)->at < (p)->end ? *(p)->at++ : (int64_t)INT32_MIN)

static void put_slots(Play *p, Slot *slots, int cap, int apply) {
    int64_t n = NEXT(p);
    if (apply)
        for (int i = 0; i < cap; i++) slots[i] = (Slot){-1, 0};
    for (int64_t k = 0; k < n; k++) {
        int64_t slot = NEXT(p), id = NEXT(p), count = NEXT(p);
        if (apply && slot >= 0 && slot < cap) slots[slot] = (Slot){(int32_t)id, (int32_t)count};
    }
}

static int put_events(RsSlot *s, Play *p, int apply) {
    int64_t n = NEXT(p);
    for (int64_t k = 0; k < n; k++) {
        int64_t v[11];
        for (int i = 0; i < 11; i++) v[i] = NEXT(p);
        RsEvent e = {.kind = (uint8_t)v[0], .tick = (int32_t)v[1], .a8 = (uint8_t)v[2], .b8 = (uint8_t)v[3],
                     .has_source = (uint8_t)v[4], .obj = (int32_t)v[5], .count = (int32_t)v[6], .v0 = (int32_t)v[7],
                     .v1 = (int32_t)v[8], .v2 = (int32_t)v[9], .v3 = (int32_t)v[10]};
        if (apply && rs_event(&s->w, &e)) return 1;
    }
    return 0;
}

static int play(RsSlot *s, int kind, int32_t action, const int16_t *start) {
    Play *p = &PLAY[s->slot];
    World *w = &s->w;
    if (p->calls-- <= 0) return bad(s, "replay: no call left");
    if (NEXT(p) != kind) return bad(s, "replay: another kind of call");
    int64_t want = NEXT(p), raise = NEXT(p);
    if (kind == 1 && want != action) return bad(s, "replay: another action");
    if (raise) return RS_FAIL(&w->err, RS_E_PLAYER_LEFT, 0, 1, 2);
    if (kind == 0) {
        for (int i = 0; i < 2; i++)
            if (NEXT(p) != start[i]) return bad(s, "replay: another start tile");
        int64_t n_kit = NEXT(p);
        if (n_kit != start[2]) return bad(s, "replay: another kit");
        for (int64_t k = 0; k < 2 * n_kit; k++)
            if (NEXT(p) != start[3 + k]) return bad(s, "replay: another kit");
        for (int k = 0; k < 7; k++)
            if (NEXT(p) != start[3 + 2 * n_kit + k]) return bad(s, "replay: other levels");
    } else if (NEXT(p) != w->p.levels[RS_SKILL_HITPOINTS]) {
        return bad(s, "replay: a step from other hitpoints");
    }
    int64_t cycles = NEXT(p);
    for (int64_t c = 0; c < cycles; c++) {
        int apply = !rs_ended(s);                      /* wait_ticks: ended() before each cycle */
        int64_t tick = NEXT(p), hp = NEXT(p), base = NEXT(p), dead = NEXT(p), x = NEXT(p), z = NEXT(p);
        if (put_events(s, p, apply)) return 1;
        put_slots(p, w->p.inv, RS_INV_CAP, apply);
        put_slots(p, w->p.worn, RS_WORN_CAP, apply);
        put_slots(p, w->p.bank, RS_BANK_CAP, apply);
        int64_t n_locs = NEXT(p);
        for (int64_t k = 0; k < n_locs; k++) {
            int64_t slot = NEXT(p), type = NEXT(p), active = NEXT(p);
            if (apply && slot >= 0 && slot < w->n_loc_slots) w->locs[slot].type = (int16_t)type, w->locs[slot].active = (uint8_t)active;
        }
        if (!apply) continue;
        w->tick = (int32_t)tick;
        w->p.levels[RS_SKILL_HITPOINTS] = (int16_t)hp, w->p.base_levels[RS_SKILL_HITPOINTS] = (int16_t)base;
        w->p.dead = (uint8_t)dead, w->p.x = (int16_t)x, w->p.z = (int16_t)z;
        rs_task_watch(s);
        rs_trace_tick(s);
    }
    s->a.obs.bank_open = (uint8_t)NEXT(p);
    s->a.obs.n_held = (int32_t)NEXT(p);
    for (int k = 0; k < s->a.obs.n_held && k < 8; k++) {
        RsHeld *h = &s->a.obs.held_sel[k];
        h->slot = (int8_t)NEXT(p), h->id = (int32_t)NEXT(p), h->count = (int32_t)NEXT(p);
        h->name = (int32_t)NEXT(p), h->option = (int32_t)NEXT(p);
    }
    s->a.visited = (uint64_t)NEXT(p);
    s->a.n_visited_off = (uint8_t)NEXT(p);   /* the count alone: the canon and facts read no cell */
    if (put_events(s, p, 1)) return 1;
    if (p->end - p->at < RS_OBS_BYTES / 8) return bad(s, "replay: a short tape");
    memcpy(p->proj, p->at, RS_OBS_BYTES);
    p->at += RS_OBS_BYTES / 8;
    return 0;
}

int rs_agent_start(RsSlot *s, int64_t seed, int32_t tile_x, int32_t tile_z, const int32_t *kit_obj,
                   const int32_t *kit_count, int n_kit, const int16_t levels[7]) {
    if (s->slot < 0 || s->slot >= RT_SLOTS || seed != 0) return bad(s, "replay: a slot past the double or seed");
    const Tape *t = NULL;
    for (int i = 0; i < N_TAPES && !t; i++)
        if (TAPES[i].slot == s->slot && TAPES[i].episode == s->episode) t = &TAPES[i];
    if (!t) return bad(s, "replay: no tape for this slot's episode");
    Play *p = &PLAY[s->slot];
    p->at = t->data + 1, p->end = t->data + t->n, p->calls = t->data[0];
    memset(&s->a, 0, sizeof s->a);
    rs_player_init(&s->w);
    s->w.has_player = 1;
    int16_t start[3 + 2 * RS_START_KIT_MAX + 7];
    start[0] = (int16_t)tile_x, start[1] = (int16_t)tile_z, start[2] = (int16_t)n_kit;
    for (int k = 0; k < n_kit; k++) start[3 + 2 * k] = (int16_t)kit_obj[k], start[4 + 2 * k] = (int16_t)kit_count[k];
    for (int k = 0; k < 7; k++) start[3 + 2 * n_kit + k] = levels[k];
    return play(s, 0, -1, start);
}

int rs_agent_step(RsSlot *s, int32_t action) { return play(s, 1, action, NULL); }

int rs_project(RsSlot *s, uint8_t *row) {
    memcpy(row, PLAY[s->slot].proj, RS_OBS_BYTES);
    return 0;
}

void rs_episode_start_feedback(RsRawObs *obs) { (void)obs; }  /* the recorded arm projection is episode_start's */
int rs_agent_login(World *w) { return RS_NOT_PORTED_HERE(&w->err); }
int rs_option_busy(RsSlot *s, int *busy) { (void)busy; return RS_NOT_PORTED_HERE(&s->w.err); }
int rs_build_obs(RsSlot *s, int32_t d, int done, int truncated, int is_dead) {
    (void)d, (void)done, (void)truncated, (void)is_dead;
    return RS_NOT_PORTED_HERE(&s->w.err);
}
int rs_project_items(RsSlot *s) { return RS_NOT_PORTED_HERE(&s->w.err); }
int rs_build_mask(RsSlot *s) { return RS_NOT_PORTED_HERE(&s->w.err); }
void rs_snapshot(const RsSlot *s, RsSnap *out) { (void)s, (void)out; RS_TRAP(); }
void rs_describe_action(const RsSlot *s, int32_t action, RsMeaning *out) { (void)s, (void)action, (void)out; RS_TRAP(); }
int rs_feedback(RsSlot *s, const RsMeaning *m, const RsSnap *b, const RsSnap *a, int running) {
    (void)m, (void)b, (void)a, (void)running;
    return RS_NOT_PORTED_HERE(&s->w.err);
}
int rs_capture_text(RsSlot *s) { return RS_NOT_PORTED_HERE(&s->w.err); }

/* rt_canon: the canonical record of a slot whose agent and episode are loaded from `in` (AGENT_STATE's order, Python
   encode_state) over an empty world on the pack's first compact layout holding only the npcs' nids. */
static const RsPack *RT_PACK;
int rt_open(const char *path) {
    RsErr err;
    return RT_PACK ? 0 : rs_pack_open(path, &RT_PACK, &err);
}

int rt_canon(const int64_t *in, int64_t n_in, int64_t *out, int cap) {
    RsSlot *s = calloc(1, sizeof *s);
    if (!s || !RT_PACK) return -1;
    const int64_t *at = in, *end = in + n_in;
#define V() (at < end ? *at++ : 0)
    World *w = &s->w;
    s->P = w->P = RT_PACK;
    w->L = &RT_PACK->layouts[RT_PACK->compact0];
    rs_player_init(w);
    w->tick = (int32_t)V();
    w->n_npcs = (uint8_t)V();
    for (int i = 0; i < w->n_npcs; i++) w->npcs[i].nid = (int16_t)V(), w->npcs[i].queue = (ReqList){-1, -1};
    Agent *a = &s->a;
    a->decision = (int32_t)V(), a->charged = (int32_t)V(), a->episode_ticks = (int32_t)V();
    a->last_mask_sum = (int32_t)V(), a->pace = (int32_t)V(), a->inv_page = (int32_t)V(), a->bank_page = (int32_t)V();
    a->bank_pages = (int32_t)V(), a->style_stale = (uint8_t)V();
    a->has_using = (uint8_t)V(), a->using_slot = (int32_t)V(), a->using_id = (int32_t)V();
    a->death_tick = (int32_t)V(), a->visited = (uint64_t)V(), a->n_visited_off = (uint8_t)V();
    a->chat_seq = (int32_t)V(), a->chat_covered = (int32_t)V(), a->n_chat = (int32_t)V();
    RsOption *o = &a->option;
    if ((o->has = (uint8_t)V())) {
        o->label = (uint8_t)V(), o->label_k = (uint8_t)V(), o->start = (int32_t)V(), o->last_x = (int16_t)V();
        o->last_z = (int16_t)V(), o->last_progress = (int32_t)V(), o->has_action = (uint8_t)V(), o->action = (int32_t)V();
        o->done = (uint8_t)V(), o->done_start = (int32_t)V();
        if ((o->has_act = (uint8_t)V())) {
            RsAct *act = &o->act;
            act->key.kind = (uint8_t)V(), act->key.id = (int32_t)V(), act->key.x = (int16_t)V(), act->key.z = (int16_t)V();
            act->key.level = (uint8_t)V(), act->key.option_lc = (int32_t)V(), act->xp = V(), act->last_xp = V();
            act->has_loc = (uint8_t)V(), act->loc_type = (int32_t)V(), act->loc_x = (int16_t)V();
            act->loc_z = (int16_t)V(), act->loc_level = (uint8_t)V();
            for (int pass = 0; pass < 2; pass++) {
                Slot *items = pass ? act->last_items : act->items;
                for (int i = 0; i < RS_INV_CAP; i++) items[i] = (Slot){-1, 0};
                for (int64_t k = 0, n = V(); k < n; k++) {
                    int64_t slot = V(), id = V(), count = V();
                    if (slot >= 0 && slot < RS_INV_CAP) items[slot] = (Slot){(int32_t)id, (int32_t)count};
                }
            }
        }
    }
    a->option_end.has = (uint8_t)V(), a->option_end.action = (int32_t)V(), a->option_end.reason = (uint8_t)V();
    a->option_end.refused = (uint8_t)V(), a->option_end.reported = (uint8_t)V();
    a->n_refused = (int32_t)V();
    for (int i = 0; i < a->n_refused && i < RS_REFUSED_CAP; i++) {
        RsRefused *r = &a->refused[i];
        r->key.kind = (uint8_t)V(), r->key.id = (int32_t)V(), r->key.x = (int16_t)V(), r->key.z = (int16_t)V();
        r->key.level = (uint8_t)V(), r->key.option_lc = (int32_t)V(), r->until = (int32_t)V();
    }
    a->n_texts = (int32_t)V();
    for (int i = 0; i < a->n_texts && i < RS_TEXTS_CAP; i++) {
        RsTextEntry *t = &a->texts[i];
        t->iface = (int32_t)V(), t->com = (int32_t)V(), t->msg = (int32_t)V(), t->tick = (int32_t)V();
    }
    Episode *ep = &s->ep;
    ep->armed = (uint8_t)V(), ep->active = (uint8_t)V(), ep->start_tick = (int32_t)V(), ep->end = (int32_t)V();
    ep->event_start = (int32_t)V(), ep->terminal = (uint8_t)V(), ep->term_reason = (uint8_t)V();
    ep->term_tick = (int32_t)V(), ep->decisions = (int32_t)V(), ep->carried_before = (int32_t)V();
    ep->banked = (int32_t)V();
    int64_t paid = V();
    memcpy(&ep->paid, &paid, sizeof paid);
#undef V
    int n = 0;
    RsErr err;
    memset(&err, 0, sizeof err);
    int failed = rs_canon(s, out, cap, &n, &err) || at != end;
    free(s);
    return failed ? -1 : n;
}

/* rt_world: the canonical record of a world rs_world_init makes on layout record `rec` (spawns: n x (nid, type, x, z,
   level, size), NULL for the layout's), after ops (7 ints each: kind, then its arguments; a slot < 0 names the
   -1-k-th loc an earlier LOC_ADD made): 0 remove (slot, duration), 1 change (slot, type, shape, angle, duration),
   2 revert (slot), 3 LOC_ADD (x, z, type, angle, shape, duration), 4 set_lifecycle (slot, duration), 5 add_loc (slot,
   duration). Returns the record's length, or -1 - the failed op's reason. */
int rt_world(int32_t rec, int64_t seed, int32_t start_tick, const int32_t *spawns, int n_spawns, int excl,
             const int32_t *ops, int n_ops, int64_t *out, int cap) {
    RsSlot *s = calloc(1, sizeof *s);
    if (!s || !RT_PACK) return -1;
    RsNpcSpawn own[RS_MAX_NPCS];
    for (int i = 0; i < n_spawns && i < RS_MAX_NPCS; i++)
        own[i] = (RsNpcSpawn){spawns[6 * i], spawns[6 * i + 1], spawns[6 * i + 2], spawns[6 * i + 3], spawns[6 * i + 4],
                              spawns[6 * i + 5]};
    s->P = RT_PACK;
    World *w = &s->w;
    int failed = rs_world_init(w, RT_PACK, &RT_PACK->layouts[rec], seed, start_tick, spawns ? own : NULL, n_spawns,
                               (uint8_t)excl, RS_XP_RATE);
    int added[64], n_added = 0;
    for (int k = 0; k < n_ops && !failed; k++) {
        const int32_t *o = &ops[7 * k];
        int slot = o[1] < 0 ? (-1 - o[1] < n_added ? added[-1 - o[1]] : 0) : o[1];
        switch (o[0]) {
        case 0: failed = rs_remove_loc(w, slot, o[2]); break;
        case 1: failed = rs_change_loc(w, slot, o[2], o[3], o[4], o[5]); break;
        case 2: failed = rs_revert_loc(w, slot); break;
        case 3: {
            int at = -1;
            failed = rs_loc_add(w, 0, o[1], o[2], o[3], o[4], o[5], o[6], &at);
            if (!failed && n_added < 64) added[n_added++] = at;
            break;
        }
        case 4: rs_set_lifecycle(w, slot, o[2]); break;
        case 5: failed = rs_add_loc(w, slot, o[2]); break;
        }
    }
    int n = 0;
    RsErr err;
    memset(&err, 0, sizeof err);
    if (failed) n = -1 - w->err.reason;
    else if (rs_canon(s, out, cap, &n, &err)) n = -1 - err.reason;
    free(s);
    return n;
}

/* rt_supply: rs_supply_observe over a supply pool layout's world (record supply0 + layout, grove `grove`), fed steps:
   per step tick, player x, z, n produce events (obj, count, tick each), n loc writes (slot, type, active each); then
   the tracker observes. Out: the logs (tick, local each) and per step the exhausted tick. Returns the log count. */
int rt_supply(int32_t layout, int32_t grove, const int32_t *steps, int n_steps, int32_t *logs, int cap,
              int32_t *exhausted) {
    RsSlot *s = calloc(1, sizeof *s);
    if (!s || !RT_PACK) return -1;
    const RsLayout *L = &RT_PACK->layouts[RT_PACK->supply0 + layout];
    s->P = RT_PACK;
    if (rs_world_init(&s->w, RT_PACK, L, 1, 0, NULL, 0, 0, RS_XP_RATE)) return -2;
    s->ep.goal = RS_GOAL_WOOD;
    RsSupply *t = &s->supply;
    t->active = 1, t->exhausted = -1;
    for (int32_t r = L->grove_off[grove]; r < L->grove_off[grove + 1] && t->n_pos < RS_GROVE_MAX; r++) {
        int32_t k = L->grove_row[r];
        t->pos_x[t->n_pos] = (int16_t)(L->x0 + L->locs[k * 5 + 1]);
        t->pos_z[t->n_pos++] = (int16_t)(L->z0 + L->locs[k * 5 + 2]);
    }
    const int32_t *at = steps;
    for (int k = 0; k < n_steps; k++) {
        s->w.tick = *at++, s->w.p.x = (int16_t)*at++, s->w.p.z = (int16_t)*at++;
        for (int32_t e = 0, n = *at++; e < n; e++, at += 3) {
            RsEvent ev = {.kind = RS_EV_PRODUCE, .obj = at[0], .count = at[1], .tick = at[2]};
            if (rs_event(&s->w, &ev)) return -3;
        }
        for (int32_t e = 0, n = *at++; e < n; e++, at += 3)
            s->w.locs[at[0]].type = (int16_t)at[1], s->w.locs[at[0]].active = (uint8_t)at[2];
        if (rs_supply_observe(s)) return -4;
        exhausted[k] = t->exhausted;
    }
    int n = t->n_logs;
    for (int i = 0; i < n && i < cap; i++) logs[2 * i] = t->log_tick[i], logs[2 * i + 1] = t->log_local[i];
    free(s);
    return n;
}
void rs_text_vector_chat(const RsPack *P, const int32_t *m, int n, int dim, float *out) {
    (void)P, (void)m, (void)n, (void)dim, (void)out;
    RS_TRAP();
}
void rs_text_vector_parts(const RsPack *P, const int32_t *m, int n, int dim, float *out) {
    (void)P, (void)m, (void)n, (void)dim, (void)out;
    RS_TRAP();
}
'''
REPLACED = ('agent.c', 'obs.c', 'text.c')  # the double stands for these files' rs.h entry points


def build_replay(out_dir):
    """The extension (module.c and every sim/native/*.c but REPLACED) with REPLAY_C, as sim/native.py builds it."""
    header = native.generated_header()
    sources = [path for path in native.sources() if path.name not in REPLACED]
    digest = hashlib.sha256(header.encode() + REPLAY_C.encode() + ' '.join(native.FLAGS).encode())
    for path in sorted(native.SOURCE.glob('*.[ch]')):
        digest.update(path.name.encode() + path.read_bytes())
    out = out_dir / f'_native_replay-{digest.hexdigest()[:16]}{sysconfig.get_config_var("EXT_SUFFIX")}'
    if out.exists():
        return out
    include = out_dir / 'include'
    include.mkdir(parents=True, exist_ok=True)
    (include / 'rs_gen.h').write_text(header)
    shim = out_dir / 'replay.c'
    shim.write_text(REPLAY_C)
    link = shlex.split(sysconfig.get_config_var('LDSHARED') or 'cc -shared')
    command = [*link, *native.FLAGS, '-I', sysconfig.get_paths()['include'], '-I', str(include), '-I',
               str(native.SOURCE), *map(str, sources), str(shim), '-o', str(out), '-lz', '-lpthread']
    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return out


@pytest.fixture(scope='module')
def replay(tmp_path_factory):
    """The replay build as sim._native for this module's tests (native.load() returns it), and its ctypes view."""
    path = build_replay(tmp_path_factory.mktemp('replay'))
    loader = importlib.machinery.ExtensionFileLoader('sim._native', str(path))
    spec = importlib.util.spec_from_file_location('sim._native', path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    lib = ctypes.CDLL(str(path))
    lib.rt_put.restype = ctypes.c_int
    lib.rt_put.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int64]
    lib.rt_clear.restype = None
    saved = native._MODULE
    native._MODULE = module
    native.log1p_table.cache_clear()
    yield module, lib
    native._MODULE = saved
    native.log1p_table.cache_clear()
    lib.rt_clear()


class Names:
    """The name table the pack's rows were exported with (its manifest's spec): this module's tests copy the oracle's
    projected ids, so the synthetic table the suite installs does not reach them."""
    def __init__(self, spec):
        self.spec = (spec['rows'], spec['dim'], spec['digest'])


@pytest.fixture(scope='module')
def inputs(replay):
    if not (DATA.exists() and POOL.exists() and SUPPLY.exists() and PACK.exists()):
        pytest.skip('archive, pools or pack absent (docs/RUNNING.md)')
    from sim.workers import load_tables
    from sim.world import load_pool
    _, meta, _ = native.read_pack(PACK)
    # the inputs check: the pack was exported from these pools (CpuBatch and the C side must play the same layouts)
    pack = native.open_pack(PACK, names=Names(meta['names']), inputs=dict(pool=POOL, supply_pool=SUPPLY))
    layouts, derived = load_pool(POOL)
    supply = load_pool(SUPPLY)
    assert (len(layouts), len(supply[0])) == (meta['counts']['compact'], meta['counts']['supply'])
    return dict(tables=load_tables(DATA), layouts=layouts, derived=derived, supply=supply, pack=pack)


# ---- the oracle's run, recorded ---------------------------------------------------------------------------------
CELL0 = (2880 // 8, 3328 // 8)  # every pool layout's origin (rs.h COORDINATES), in 8x8 visit cells


def listed(inv):
    rows = inv.listed()
    return [len(rows), *[v for row in rows for v in row]]


class Recorder:
    """The tapes of a CpuBatch run (REPLAY_C's format), one per (global slot, episode), from hooks on Agent.start,
    Agent.step, Agent.capture_text (each cycle, where the harness takes its Row), GoalEpisode.project and
    GoalEpisode.__init__. inject: {(slot, episode, decision)} whose step raises UNSUPPORTED before it runs (decision
    'setup': the STYLE_0 setup step); no_wield: {(slot, episode)} whose STYLE_0 observation offers no Wield."""

    def __init__(self, env, pack, monkeypatch, inject=(), no_wield=()):
        from sim.agent import A_STYLE0, Agent
        from sim.engine import Unsupported
        from sim.spec import OBSERVATION_SHAPES, START_LEVELS, dtype
        from sim.task import GoalEpisode
        self.env, self.inject, self.calls, self.of = env, set(inject), {}, {}
        no_wield = set(no_wield)
        self.strings = H.PackStrings(pack)
        self.canon = H.Canon(self.strings)
        layout = {row[0]: (row[5], row[6]) for row in native.load().constants()['obs']}
        task_keys = ('budget_vec', 'critic_vec', 'goal_vec')
        binding = []
        inner = env._episode

        def episode(b):
            binding.append((env.first + b, env.episodes[b], bool(env.spec['supply'] and env.goals[b] == 'wood')))
            try:
                return inner(b)
            finally:
                binding.pop()
        env._episode = episode
        init, start, step, capture, project = (GoalEpisode.__init__, Agent.start, Agent.step, Agent.capture_text,
                                               GoalEpisode.project)

        def episode_init(task, world, *args, **kwargs):
            init(task, world, *args, **kwargs)
            if binding:
                slot, ep, supply = binding[-1]
                self.calls[slot, ep] = []
                self.of[task.agent] = dict(key=(slot, ep), task=task, seen=0, supply=supply)

        def agent_start(agent, seed, cfg, tile, kit, levels):
            call = self.open(agent, START, -1)
            if call is not None:
                ids = [v for name, count in kit for v in (agent.world.t.obj_id(name), count)]
                call['start'] = [tile[0], tile[1], len(kit), *ids, *[levels[s] for s in START_LEVELS]]
            raw = start(agent, seed, cfg, tile, kit, levels)
            self.close(agent, raw)
            return raw

        def agent_step(agent, action):
            call = self.open(agent, STEP, action)
            setup = False
            if call is not None:
                task, key = self.of[agent]['task'], self.of[agent]['key']
                setup = not getattr(task, 'armed', False) and action == A_STYLE0
                if (*key, task.decisions if getattr(task, 'armed', False) else 'setup' if setup else None) in self.inject:
                    call['raise_'] = 1
                    raise Unsupported(UNSUPPORTED)
                call['hp'] = agent.player.hp
            raw = step(agent, action)
            if setup and self.of[agent]['key'] in no_wield:
                raw['heldSel'] = [h for h in raw['heldSel'] if h['option'] != 'Wield']
            self.close(agent, raw)
            return raw

        def capture_text(agent):
            mine = self.of.get(agent)
            if mine is not None and self.calls[mine['key']]:
                self.calls[mine['key']][-1]['cycles'].append(self.cycle(agent, mine))
            return capture(agent)

        def task_project(task):
            result = project(task)
            mine = self.of.get(task.agent)
            if mine is not None and self.calls[mine['key']]:
                row = np.full(sum(n for _, n in layout.values()), 0xFF, np.uint8)  # the task's arrays: C writes them
                for name in OBSERVATION_SHAPES:
                    if name not in task_keys:
                        off, n = layout[name]
                        row[off:off + n] = np.frombuffer(np.ascontiguousarray(result[name], dtype(name)).tobytes(),
                                                         np.uint8)
                self.calls[mine['key']][-1]['proj'] = row.view('<i8')
            return result
        monkeypatch.setattr(GoalEpisode, '__init__', episode_init)
        monkeypatch.setattr(Agent, 'start', agent_start)
        monkeypatch.setattr(Agent, 'step', agent_step)
        monkeypatch.setattr(Agent, 'capture_text', capture_text)
        monkeypatch.setattr(GoalEpisode, 'project', task_project)

    def open(self, agent, kind, action):
        mine = self.of.get(agent)
        if mine is None:
            return None
        call = dict(kind=kind, action=int(action), raise_=0, cycles=[], end=None, proj=None)
        self.calls[mine['key']].append(call)
        return call

    def events(self, agent, mine):
        world = agent.world
        rows = [v for ev in world.events[mine['seen']:] for v in self.canon.event(ev, agent.name)]
        n = len(world.events) - mine['seen']
        mine['seen'] = len(world.events)
        return [n, *rows]

    def cycle(self, agent, mine):
        world, p = agent.world, agent.player
        locs = [(k, loc.type, int(loc.active)) for k, loc in enumerate(world.locs)] if mine['supply'] else []
        return [world.tick, p.levels[3], p.base_levels[3], int(p.dead), p.x, p.z, *self.events(agent, mine),
                *listed(p.inv), *listed(p.worn), *listed(p.bank), len(locs), *[v for row in locs for v in row]]

    def close(self, agent, raw):
        mine = self.of.get(agent)
        if mine is None:
            return
        held = [v for h in raw['heldSel'] for v in (h['slot'], h['id'], h['count'], self.strings(h['name']),
                                                   self.strings(h['option']))]
        bits = off = 0
        for level, cx, cz in agent.visited:
            if level == 0 and 0 <= cx - CELL0[0] < 8 and 0 <= cz - CELL0[1] < 8:
                bits |= 1 << ((cx - CELL0[0]) * 8 + cz - CELL0[1])
            else:
                off += 1
        self.calls[mine['key']][-1]['end'] = [int(bool(raw.get('bankOpen'))), len(raw['heldSel']), *held,
                                              bits - (1 << 64) if bits >= 1 << 63 else bits, off,
                                              *self.events(agent, mine)]

    def tapes(self):
        """{(slot, episode): int64 tape} of every episode the run began."""
        words = native.load().constants()['RS_OBS_BYTES'] // 8
        out = {}
        for key, calls in self.calls.items():
            tape = [len(calls)]
            for c in calls:
                tape += [c['kind'], c['action'], c['raise_']]
                if c['raise_']:
                    continue
                tape += c['start'] if c['kind'] == START else [c['hp']]
                tape.append(len(c['cycles']))
                for cycle in c['cycles']:
                    tape += cycle
                tape += c['end']
                tape += list(c['proj']) if c['proj'] is not None else [0] * words
            out[key] = np.asarray(tape, np.int64)
        return out


def outputs_equal(name, a, b):
    """Bitwise equality of two step outputs (state, obs, reward, done, info) or two resets (state, obs)."""
    lines = []
    if not np.array_equal(a[0]['episode'], b[0]['episode']):
        lines.append(f'{name}: episodes {a[0]["episode"].tolist()} vs {b[0]["episode"].tolist()}')
    for key in a[1]:
        lines += H._array_diff(f'{name} obs {key}', a[1][key], b[1][key])
    if len(a) > 2:
        lines += H._array_diff(f'{name} reward', a[2], b[2]) + H._array_diff(f'{name} done', a[3], b[3])
        for key in a[4]:
            lines += H._array_diff(f'{name} info {key}', a[4][key], b[4][key])
    return lines


EPISODE_FIELDS = ('RS_CF_EPISODE', 'RS_CF_EVENT_LAST')


def rows_equal(name, ra, rb):
    """The C rows of one world in one call against the Python side's: their count, ticks, episodes and events, and the
    record's episode fields and event count (the world's other fields are the replay's, not C's); and each C row's crc
    is zlib.crc32 of its record."""
    import zlib
    if len(ra) != len(rb):
        return [f'{name}: {len(ra)} vs {len(rb)} cycles']
    lines = []
    for k, (x, y) in enumerate(zip(ra, rb)):
        if (x.tick, x.episode, x.events) != (y.tick, y.episode, y.events):
            lines.append(f'{name} cycle {k}: tick, episode or events differ ({x.tick} {x.episode} vs {y.tick} {y.episode})')
        if y.crc != zlib.crc32(y.record.tobytes()):
            lines.append(f'{name} cycle {k}: crc {y.crc:08x} is not the record\'s')
        da, db = H.decode(x.record), H.decode(y.record)
        for field in EPISODE_FIELDS:
            if not np.array_equal(da[field], db[field]):
                lines.append(f'{name} cycle {k} tick {x.tick}: {field} {da[field].tolist()} vs {db[field].tolist()}')
        if da['RS_CF_LOGS'][-1] != db['RS_CF_LOGS'][-1]:
            lines.append(f'{name} cycle {k}: {da["RS_CF_LOGS"][-1]} vs {db["RS_CF_LOGS"][-1]} events')
    return lines[:8]


def attempt(fn, *args):
    try:
        return fn(*args), None
    except Exception as error:
        return None, error


def python_run(inputs, monkeypatch, n, task, steps, goals=None, horizon=None, inject=(), first=0, scenario=None,
               source='random'):
    """CpuBatch under keyed actions (sim/native_harness.py make_source: random legal or biased), as a PythonSide with the pack's
    string ids (and a harness scenario start): per call its outputs (or the exception), the world Rows, the starts and
    the new evidence rows."""
    from baseline.resource_task import GOALS
    from sim.vec import CpuBatch
    kwargs = dict(derived=inputs['derived'], task=task, goals=goals or GOALS, first=first)
    if task == 'goal-activity-v5':
        kwargs['search'] = inputs['supply']
    if horizon is not None:
        kwargs['horizon'] = horizon
    env = CpuBatch(inputs['tables'], inputs['layouts'], n, **kwargs)
    side = H.PythonSide(env, canon=H.Canon(H.PackStrings(inputs['pack'])), scenario=scenario)
    recorder = Recorder(env, inputs['pack'], monkeypatch, inject)
    source, calls = H.make_source(source, KEY), []
    out, error = attempt(side.reset, KEY)
    calls.append(dict(out=out, error=error, rows=side.take(), starts=list(env.starts), evidence=list(env.evidence),
                      action=None))
    for step in range(steps):
        if error is not None:
            break
        action = source(step, out[1], first + np.arange(n), None)
        seen = len(env.evidence)
        out, error = attempt(side.step, out[0], action)
        calls.append(dict(out=out, error=error, rows=side.take(), starts=list(env.starts),
                          evidence=list(env.evidence[seen:]), action=action))
    monkeypatch.undo()
    return calls, recorder.tapes()


class Drawn:
    """A NativeSide's `python` for a scenario run replayed after the oracle's: CpuBatch.starts as they stood after
    each call (NativeSide reads env.starts after the Python side's call)."""
    def __init__(self):
        self.env, self.starts = self, None


def native_run(replay, inputs, calls, tapes, n, task, goals=None, horizon=None, threads=1, first=0, scenario=None):
    """NativeBatch on the replay build, fed the oracle's actions (and, with a scenario, its start records through
    NativeSide's begin): per call its outputs or exception, rows, starts and new evidence rows, as python_run."""
    module, lib = replay
    lib.rt_clear()
    for (slot, episode), tape in tapes.items():
        assert lib.rt_put(slot, episode, tape.ctypes.data, len(tape)) == 0
    batch = native.NativeBatch(inputs['pack'], n, first=first, task=task, goals=goals, horizon=horizon,
                               threads=threads)
    drawn = Drawn()
    side = H.NativeSide(batch, scenario=scenario, python=drawn)
    drawn.starts = calls[0]['starts']
    out, error = attempt(side.reset, KEY)
    mine = [dict(out=out, error=error, rows=side.take(), starts=batch.starts, evidence=list(batch.evidence))]
    for call in calls[1:]:
        if error is not None:
            break
        seen = len(batch.evidence)
        drawn.starts = call['starts']
        out, error = attempt(side.step, out[0], call['action'])
        mine.append(dict(out=out, error=error, rows=side.take(), starts=batch.starts,
                         evidence=list(batch.evidence[seen:])))
    return mine


def compare(python, mine, n, first=0):
    """Lines naming every difference of the two runs, call by call (empty: equal)."""
    lines = []
    for k, (a, b) in enumerate(zip(python, mine)):
        name = 'reset' if k == 0 else f'step {k - 1}'
        if (a['error'] is None) != (b['error'] is None) or (a['error'] is not None and (
                type(a['error']) is not type(b['error']) or str(a['error']) != str(b['error']))):
            lines.append(f'{name}: {type(a["error"]).__name__}: {a["error"]} vs {type(b["error"]).__name__}: '
                         f'{b["error"]}')
            break
        if a['error'] is not None:
            break
        lines += outputs_equal(name, a['out'], b['out'])
        for w in range(n):
            lines += rows_equal(f'{name} world {w}', a['rows'][w], b['rows'][w])
        done = np.ones(n, bool) if k == 0 else a['out'][3]
        for w in np.flatnonzero(done):
            if H._plain(a['starts'][w]) != H._plain(b['starts'][w]):
                lines.append(f'{name} world {w}: starts differ')
        if a['evidence'] or b['evidence']:
            ea = [H._plain({k2: v for k2, v in row.items() if k2 != 'observation'}) for row in a['evidence']]
            eb = [H._plain(row) for row in b['evidence']]
            if ea != eb:
                lines.append(f'{name}: evidence {ea} vs {eb}')
        if lines:
            break
    return lines


# ---- the tests ----------------------------------------------------------------------------------------------------
def ended(calls, goal_of):
    """{goal: [terminal codes]} of the episodes a run ended."""
    out = {}
    for call in calls[1:]:
        if call['out'] is not None:
            info = call['out'][4]
            for w in np.flatnonzero(call['out'][3]):
                out.setdefault(goal_of[w], []).append(int(info['terminal'][w]))
    return out


def test_a_v5_batch_equals_cpubatch_and_its_thread_count_changes_nothing(replay, inputs, monkeypatch):
    """goal-activity-v5 on 8 worlds (supply wood, fish, mine and sustained combat, two of each): every observation
    array, reward, done and info field (made, kills, facts), each start and every tick's episode record equal
    CpuBatch's, on one thread and on three."""
    from baseline.resource_task import GOALS
    calls, tapes = python_run(inputs, monkeypatch, 8, 'goal-activity-v5', 160)
    assert all(call['error'] is None for call in calls)
    goals = [GOALS[w % 4] for w in range(8)]
    done = ended(calls, goals)
    assert {'fish', 'mine', 'wood'} <= set(done), done  # every gathering goal ended an episode
    lines = compare(calls, native_run(replay, inputs, calls, tapes, 8, 'goal-activity-v5'), 8)
    assert not lines, lines[:8]
    lines = compare(calls, native_run(replay, inputs, calls, tapes, 8, 'goal-activity-v5', threads=3), 8)
    assert not lines, lines[:8]


@pytest.mark.parametrize('task, goal, scenario, steps, source', [
    ('goal-activity-v5', 'melee', None, 60, 'random'), ('goal-activity-v5', 'melee', 'danger', 30, 'random'),
    ('goal-activity-v5', 'wood', 'scarce', 70, 'random'), ('goal-activity-v5', 'fish', 'under', 50, 'random'),
    ('goal-activity-v3', 'melee', 'hurt', 550, 'biased')])  # the last: a death after a certified kill (step 540 since the auto-retaliate fix)
def test_specialists_and_scenario_starts_equal_cpubatch(replay, inputs, monkeypatch, task, goal, scenario, steps,
                                                        source):
    """A single-goal set on 8 worlds, its starts drawn (melee: both worn states, so the Wield setup step) or a
    harness scenario's (a mugger in every combat start; every supply start scarce; levels under the unlocks; low
    hitpoints with food, under play weighted to acts, so a death settles kills to 0), the
    native side beginning each episode from the oracle's start record (NativeSide's begin, rs_start_rec), equals
    CpuBatch's outputs, starts and per-tick episode records."""
    calls, tapes = python_run(inputs, monkeypatch, 8, task, steps, goals=(goal,), scenario=scenario, source=source)
    assert all(call['error'] is None for call in calls)
    starts = [start for call in calls for start in call['starts']]
    if goal == 'melee' and scenario is None:
        assert {s['combat']['worn'] for s in starts} == {False, True}
    if scenario == 'danger':
        assert all(s['combat']['muggers'] for s in starts)
    if scenario == 'scarce':
        assert {s['variant'] for s in starts} == {'scarce'}
    if scenario == 'hurt':
        infos = [call['out'][4] for call in calls[1:]]
        assert any(i['terminal'][w] == 2 and i['outcomes'][w] > 0 for i in infos for w in range(8))
    lines = compare(calls, native_run(replay, inputs, calls, tapes, 8, task, goals=(goal,), scenario=scenario), 8)
    assert not lines, lines[:8]


def test_an_unsupported_step_is_flagged_with_its_evidence_and_refused_after(replay, inputs, monkeypatch):
    """A step that reaches unsupported content (injected on both sides: the oracle's Agent.step raises, the double
    raises RS_E_PLAYER_LEFT at once) in a slice of a v2 batch (first 5, a 40-tick horizon so episodes end): the world
    is flagged with its goal and stale observation and nothing else, the evidence row is CpuBatch's (reason, global
    slot, key, episode, start, layout seed, decision, tick, action), the other worlds step on, and the next step fails
    with CpuBatch's RuntimeError."""
    calls, tapes = python_run(inputs, monkeypatch, 4, 'goal-activity-v2', 30, horizon=40, first=5,
                              inject={(6, 1, 2)})
    flagged = [k for k, call in enumerate(calls[1:]) if call['out'] is not None and call['out'][4]['unsupported'].any()]
    assert len(flagged) == 1 and calls[1 + flagged[0]]['out'][4]['unsupported'].tolist() == [False, True, False, False]
    assert calls[1 + flagged[0]]['out'][4]['goal'][1] == 2 and isinstance(calls[2 + flagged[0]]['error'], RuntimeError)
    assert calls[1 + flagged[0]]['evidence'][0]['reason'] == UNSUPPORTED
    lines = compare(calls, native_run(replay, inputs, calls, tapes, 4, 'goal-activity-v2', horizon=40, first=5), 4)
    assert not lines, lines[:8]


def put_tapes(replay, tapes):
    replay[1].rt_clear()
    for (slot, episode), tape in tapes.items():
        assert replay[1].rt_put(slot, episode, tape.ctypes.data, len(tape)) == 0


def test_refusals_and_repros_equal_the_oracle(replay, inputs, monkeypatch):
    """The oracle's refusals come out of the native batch as the same exception and text: a masked action
    (ValueError), hitpoints outside 1..base at a combat start and a Wield the setup observation does not offer
    (ValueError), and an Unsupported in a setup step, which reset raises rather than flags. A repro (the harness's
    start_episode: the episode counter set, then the drawn episode begun) equals CpuBatch's."""
    from sim import vec
    from sim.vec import CpuBatch
    pack = inputs['pack']
    env = CpuBatch(inputs['tables'], inputs['layouts'], 4, derived=inputs['derived'], task='goal-activity-v3')
    a = H.PythonSide(env, canon=H.Canon(H.PackStrings(pack)))
    recorder = Recorder(env, pack, monkeypatch, no_wield={(3, 7)})
    reset_a = a.reset(KEY)
    masked = [int(np.flatnonzero(reset_a[1]['mask'][0] == 0)[0])] + [int(np.flatnonzero(reset_a[1]['mask'][w])[0])
                                                                       for w in range(1, 4)]
    _, masked_a = attempt(a.step, reset_a[0], np.asarray(masked, np.int32))
    repro_a = a.start_episode(2, 4)
    repro_start = env.starts[2]
    draw, refused = vec.draw_combat, []
    for episode, edit in ((5, dict(hp=99)), (7, dict(worn=True))):  # a world 3 (melee) start, edited
        env.episodes[3] = episode
        monkeypatch.setattr(vec, 'draw_combat', lambda *args, edit=edit, **kw: {**draw(*args, **kw), **edit})
        refused.append((attempt(env._episode, 3)[1], env.starts[3]))
    monkeypatch.undo()
    (hp_a, hp_start), (wield_a, wield_start) = refused
    assert isinstance(masked_a, ValueError) and isinstance(hp_a, ValueError) and isinstance(wield_a, ValueError)
    put_tapes(replay, recorder.tapes())
    batch = native.NativeBatch(pack, 4, task='goal-activity-v3')
    b = H.NativeSide(batch)
    reset_b = b.reset(KEY)
    assert not outputs_equal('reset', reset_a, reset_b)
    _, masked_b = attempt(b.step, reset_b[0], np.asarray(masked, np.int32))
    repro_b = b.start_episode(2, 4)
    assert not outputs_equal('repro', repro_a, repro_b) and H._plain(batch.starts[2]) == H._plain(repro_start)
    batch.set_episode(3, 5)
    _, hp_b = attempt(batch.begin, 3, hp_start)
    batch.set_episode(3, 7)
    _, wield_b = attempt(batch.begin, 3, wield_start)
    for x, y in ((masked_a, masked_b), (hp_a, hp_b), (wield_a, wield_b)):
        assert type(x) is type(y) and str(x) == str(y), (type(y).__name__, str(y) == str(x))
    # an Unsupported in STYLE_0 at reset is raised, not flagged (vec.py; the port keeps the raise)
    env = CpuBatch(inputs['tables'], inputs['layouts'], 2, derived=inputs['derived'], horizon=40)
    a = H.PythonSide(env)  # before the recorder, which wraps the side's episode hook
    recorder = Recorder(env, pack, monkeypatch, inject={(1, 0, 'setup')})
    _, setup_a = attempt(a.reset, KEY)
    monkeypatch.undo()
    put_tapes(replay, recorder.tapes())
    _, setup_b = attempt(native.NativeBatch(pack, 2, task='goal-activity-v2', horizon=40).reset, KEY)
    assert type(setup_a).__name__ == type(setup_b).__name__ == 'Unsupported' and str(setup_a) == str(setup_b)


def test_a_v11_native_batch_refuses_another_pool_or_goal_set(inputs):
    """NativeBatch goal-activity-v11 makes CpuBatch's checks before it builds a batch: every goal set but mine is
    refused, and so is this pack, whose compact layouts (read from its LAYOUTS records) are the 8-layout pool's training
    seeds 0-7; v5 takes the same pack and goals."""
    from sim.world import world_of
    pack = inputs['pack']
    assert list(native.compact_layouts(pack)) == [(layout.seed, world_of(layout)['version']) for layout in inputs['layouts']]
    for goals in (None, ('wood',)):
        with pytest.raises(ValueError, match=r"plays goals \['mine'\]"):
            native.NativeBatch(pack, 4, task='goal-activity-v11', goals=goals)
    with pytest.raises(ValueError, match='plays compact-world-v1 train seeds 1000-1127 in order, not 8 layouts'):
        native.NativeBatch(pack, 4, task='goal-activity-v11', goals=('mine',))
    native.NativeBatch(pack, 4, task='goal-activity-v5', goals=('mine',))


def test_native_workers_write_their_slices_in_place(replay, inputs, monkeypatch):
    """sim/workers.py's worker with engine='native' (two of them, on threads of this process so they run the replay
    build) writes each slice of the shared buffer in place, and the buffer holds what CpuBatch returns for the whole
    batch; CpuWorkers refuses a native engine without a pack."""
    import threading
    from multiprocessing import Pipe, shared_memory
    import names
    from baseline.resource_task import GOALS
    from sim.workers import CpuWorkers, _worker, views
    with pytest.raises(ValueError, match='native'):
        CpuWorkers(4, 2, POOL, engine='native')
    calls, tapes = python_run(inputs, monkeypatch, 4, 'goal-activity-v2', 25, horizon=40)
    put_tapes(replay, tapes)
    _, meta, _ = native.read_pack(PACK)
    table = lambda: Names(meta['names'])  # the table the pack's rows come from (open_pack's check)
    table.cache_clear = lambda: None      # the suite's name table fixture clears it on teardown
    monkeypatch.setattr(names, 'name_table', table)
    block = shared_memory.SharedMemory(create=True, size=views(None, 4)[1])
    v, _ = views(block.buf, 4)
    conns, threads = [], []
    try:
        for k in range(2):
            parent, child = Pipe()
            thread = threading.Thread(target=_worker, args=(2 * k, 2, 4, block.name, None, str(POOL), 40,
                                                             'goal-activity-v2', child, None, GOALS, 'native', str(PACK)))
            thread.start()
            conns.append(parent)
            threads.append(thread)
        assert [conn.recv()[0] for conn in conns] == ['ready', 'ready']
        lines = []
        for k, call in enumerate(calls):
            if k:
                v['action'][:] = call['action']
            for conn in conns:
                conn.send(('reset', KEY) if k == 0 else ('step', None))
            replies = [conn.recv() for conn in conns]
            assert [r[0] for r in replies] == ['ok', 'ok'], [r[1] for r in replies]
            for name, value in call['out'][1].items():
                lines += H._array_diff(f'call {k} obs {name}', value, v[f'obs/{name}'])
            if k:
                _, _, reward, done, info = call['out']
                lines += H._array_diff(f'call {k} reward', reward, v['reward'])
                lines += H._array_diff(f'call {k} done', done, v['done'])
                for name, value in info.items():
                    lines += H._array_diff(f'call {k} info {name}', value, v[f'info/{name}'])
        assert not lines, lines[:8]
    finally:
        for conn in conns:
            conn.send(('close', None))
        for thread in threads:
            thread.join(timeout=30)
        del v
        block.close()
        block.unlink()


# ---- canon.c's agent and episode fields --------------------------------------------------------------------------
AGENT_FIELDS = ('RS_CF_AGENT', 'RS_CF_AGENT_OPTION', 'RS_CF_AGENT_ACT', 'RS_CF_AGENT_OPTION_END', 'RS_CF_AGENT_REFUSED',
                'RS_CF_AGENT_TEXTS', 'RS_CF_EPISODE')


def encode_state(world, agent, task, strings, message):
    """An Agent and GoalEpisode as the rs.h structs hold them (rt_canon's order): what agent.c and task.c keep, written
    from the oracle's objects as rs.h documents each field (labels and done tests by kind, affordance keys by
    their parts, item states as slots, texts by message id). Dict and set orders are kept: canon.c sorts."""
    import re
    from feedback_contract import END_REASONS
    e = H.header()[0]
    out = [world.tick, len(world.npcs), *[npc.nid for npc in world.npcs]]
    using = agent.using
    bits = off = 0
    for level, cx, cz in agent.visited:
        if level == 0 and 0 <= cx - CELL0[0] < 8 and 0 <= cz - CELL0[1] < 8:
            bits |= 1 << ((cx - CELL0[0]) * 8 + cz - CELL0[1])
        else:
            off += 1
    out += [agent.decision, agent.charged, agent.episode_ticks, agent.last_mask_sum, agent.pace, agent.inv_page,
            agent.bank_page, agent.bank_pages, int(agent.style_stale), *((1, using['slot'], using['id']) if using else
                                                                          (0, 0, 0)),
            -1 if agent.death_tick is None else agent.death_tick, bits - (1 << 64) if bits >= 1 << 63 else bits, off,
            agent.chat_seq, agent.chat_covered, len(agent.chat)]
    o = agent.option
    key = H.Canon(strings).key
    if o is None:
        out.append(0)
    else:
        label = re.fullmatch(r'(act|nav_click)_(\d+)', o['label'])
        kind, k = (f'RS_LABEL_{label[1].upper()}', int(label[2])) if label else ('RS_LABEL_PICK', 0)
        done = o['done'].__code__.co_qualname
        done_kind, done_start = {'Agent.act_done.<locals>.done': ('RS_DONE_ACT', 'start'),
                                 'Agent.pick_up.<locals>.<lambda>': ('RS_DONE_PICK', None),
                                 'Agent.click.<locals>.<lambda>': ('RS_DONE_WALK', None)}[done]
        done_start = H._cells(o['done'])['start'] if done_start else 0
        out += [1, e[kind], k, o['start'], *o['last_pos'], o['last_progress'], int('action' in o), o.get('action', 0),
                e[done_kind], done_start]
        act = o.get('act')
        if act is None:
            out.append(0)
        else:
            items = lambda state: [int(v) for part in state.split(',') if part for v in part.split(':')]
            now, last = items(act['items']), items(act['last_items'])
            out += [1, *key(act['key']), act['xp'], act['last_xp'], *((1, *act['loc']) if act['loc'] else (0, 0, 0, 0, 0)),
                    len(now) // 3, *now, len(last) // 3, *last]
    end = agent.option_end
    out += [0] * 5 if end is None else [1, end['action'], END_REASONS.index(end['reason']), int(end['refused']),
                                        int(end['reported'])]
    out += [len(agent.refused), *[v for k_, until in agent.refused.items() for v in (*key(k_), until)]]
    out += [len(agent.texts), *[v for (iface, com), (text, tick) in agent.texts.items()
                                for v in (strings(iface), com, message[strings(text)], tick)]]
    term = getattr(task, 'terminal', None)
    out += [int(getattr(task, 'armed', False)), int(task.active), getattr(task, 'start_tick', 0), getattr(task, 'end', 0),
            getattr(task, 'event_start', 0), int(term is not None),
            e[f"RS_TERM_{term['reason'].upper()}"] if term else 0, term['tick'] if term else 0,
            getattr(task, 'decisions', 0), getattr(task, 'carried_before', 0), getattr(task, 'banked', 0),
            H._bits(getattr(task, 'paid', 0.0))]
    return np.asarray(out, np.int64)


def test_canon_writes_the_agent_and_episode_fields_as_the_harness_canon(replay, inputs, monkeypatch):
    """canon.c's agent and episode part of the record (RS_CF_AGENT .. RS_CF_EPISODE) equals the harness Canon's on the
    oracle's own states: every tick of a v2 batch under biased play (act, held, use and dialog weighted, so act and
    click options, refusals and component texts occur), each loaded into the rs.h structs by their documented meaning.
    The world part is world.c's."""
    from sim.agent import Agent
    pack, (module, lib) = inputs['pack'], replay
    lib.rt_open.argtypes, lib.rt_canon.restype = [ctypes.c_char_p], ctypes.c_int
    lib.rt_canon.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
    assert lib.rt_open(str(PACK).encode()) == 0
    msgs = np.frombuffer(pack.section(module.constants()['sections']['MSGS']), native.record_dtype('RsMsg'))
    message = {}
    for k, text in enumerate(msgs['text'].tolist()):
        message.setdefault(text, k)
    strings = H.PackStrings(pack)
    canon = H.Canon(strings)
    from sim.vec import CpuBatch
    env = CpuBatch(inputs['tables'], inputs['layouts'], 4, derived=inputs['derived'])  # 240 ticks: refusals expire
    tasks = {}
    seen = dict(states=0, act=0, click=0, refused=0, expired=0, texts=0, ended=0)
    lines = []
    capture = Agent.capture_text

    def capture_text(agent):
        task = tasks.get(agent)
        if task is not None and len(lines) < 8:
            expected = H.decode(canon(agent.world, agent, task))
            state = encode_state(agent.world, agent, task, strings, message)
            out = np.zeros(1 << 16, np.int64)
            n = lib.rt_canon(state.ctypes.data, len(state), out.ctypes.data, len(out))
            assert n > 0
            got = H.decode(out[:n])
            lines.extend(f'tick {agent.world.tick}: {f}' for f in AGENT_FIELDS if not np.array_equal(expected[f], got[f]))
            o = agent.option or {}
            seen['states'] += 1
            seen['act'] += 'act' in o
            seen['click'] += o.get('label', '').startswith('nav_click')
            seen['refused'] += any(until > agent.world.tick for until in agent.refused.values())
            seen['expired'] += any(until == agent.world.tick for until in agent.refused.values())
            seen['texts'] += bool(agent.texts)
            seen['ended'] += agent.option_end is not None
        return capture(agent)
    from sim.task import GoalEpisode
    init = GoalEpisode.__init__

    def episode_init(task, world, *args, **kwargs):
        init(task, world, *args, **kwargs)
        tasks[task.agent] = task
    monkeypatch.setattr(GoalEpisode, '__init__', episode_init)
    monkeypatch.setattr(Agent, 'capture_text', capture_text)
    source = H.Biased(KEY)
    state, obs = env.reset(KEY)
    for step in range(60):
        state, obs, *_ = env.step(state, source(step, obs, np.arange(4), None))
    monkeypatch.undo()
    assert not lines, lines
    assert all(seen.values()), seen


WORLD_FIELDS = 'RS_CF_TICK', 'RS_CF_EVENT_LAST'  # the world's part of the record: these two and the fields between


def test_the_world_part_of_the_record_is_the_harness_canons(replay, inputs):
    """rs_canon's world fields (world.c rs_world_canon, RS_CF_TICK .. RS_CF_EVENT_LAST) equal the harness Canon's on
    worlds made as an episode makes them (rs_world_init: every compact layout, alone and with combat_npcs' goblins and
    a mugger, and supply layouts and scarce variants with their exclusions), before and after loc operations applied
    to both worlds alike (removal, change, revert, lifecycles, re-adding, LOC_ADD on an occupied and a free tile)."""
    from sim.combat import COMBAT, combat_npcs
    from sim.engine import Rng, Unsupported, World
    from sim.search import SUPPLY, groves
    from sim.task import GoalEpisode
    from sim.world import LayoutTables, without_locs
    pack, lib = inputs['pack'], replay[1]
    lib.rt_open.argtypes, lib.rt_world.restype = [ctypes.c_char_p], ctypes.c_int
    lib.rt_world.argtypes = [ctypes.c_int32, ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int,
                             ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    assert lib.rt_open(str(PACK).encode()) == 0
    canon, fields = H.Canon(H.PackStrings(pack)), H.contract()[0]
    world_fields = fields[fields.index(WORLD_FIELDS[0]):fields.index(WORLD_FIELDS[1]) + 1]
    tables, rng = inputs['tables'], Rng(KEY)
    sec = native.load().constants()['sections']
    recs = np.frombuffer(pack.section(sec['LAYOUTS']), native.record_dtype('RsLayoutRec'))
    variants = np.frombuffer(pack.section(sec['VARIANTS']), native.record_dtype('RsVariantKey'))
    supply, compact0, supply0 = inputs['supply'][0], np.flatnonzero(recs['kind'] == 0)[0], np.flatnonzero(recs['kind'] == 1)[0]
    worlds = [(compact0 + k, LayoutTables(tables, layout, inputs['derived'][k]), None, frozenset())
              for k, layout in enumerate(inputs['layouts'])]
    camp = next(k for k in range(len(inputs['layouts'])) if recs[compact0 + k]['n_camp'] > 0)
    worlds.append((compact0 + camp, worlds[camp][1], combat_npcs(inputs['layouts'][camp], tables, [(5, 5)]),
                   COMBAT['exclusions']))
    for j in (0, 1):
        worlds.append((supply0 + j, LayoutTables(tables, supply[j]), None, SUPPLY['exclusions']))
    for row in variants[:2]:  # the scarce variant of (layout, grove, kept row), as variant_layout makes it
        layout = supply[row['layout']]
        removed = [k for k in groves(layout, tables)[row['grove']] if k != row['keep']]
        worlds.append((row['rec'], LayoutTables(tables, without_locs(layout, tables, removed)), None,
                       SUPPLY['exclusions']))
    lines, operations = [], 0
    for rec, lt, npcs, exclusions in worlds:
        rec = int(rec)
        seed = int(rng(1, rec) * (1 << 24))
        bits = sum(1 << ('random_events', 'clue_drops', 'loot_drops').index(e) for e in exclusions)
        spawns = None if npcs is None else np.asarray([(n['nid'], n['type'], n['x'], n['z'], n['level'], n['size'])
                                                       for n in npcs], np.int32)
        world = World(lt, seed=seed, start_tick=6, npcs=npcs, exclusions=exclusions)
        task = GoalEpisode(world, {}, 'wood', horizon=240, scale=1.0)
        locs, ops = world.locs, []
        for step in range(10):
            k = int(rng(2, rec, step) * len(locs))
            other = locs[int(rng(3, rec, step) * len(locs))]
            kind = step % 6
            free = (world._x0 + 3 + step, world._z0 + 2 + step)
            op = [(0, k, 50, 0, 0, 0, 0), (1, k, other.type, other.shape, other.angle, 30, 0), (2, k, 0, 0, 0, 0, 0),
                  (3, *((locs[k].x, locs[k].z) if step % 2 else free), other.type, other.angle, other.shape, 40),
                  (4, k, 25, 0, 0, 0, 0), (5, k, 20, 0, 0, 0, 0)][kind]
            ops.append(op)
            try:
                if kind == 0:
                    world.remove_loc(locs[k], 50)
                elif kind == 1:
                    world.change_loc(locs[k], other.type, other.shape, other.angle, 30)
                elif kind == 2:
                    world.revert_loc(locs[k])
                elif kind == 3:
                    world.loc_add(0, op[1], op[2], other.type, other.angle, other.shape, 40)
                elif kind == 4:
                    world.set_lifecycle(locs[k], 25)
                else:
                    world.add_loc(locs[k], 20)
                expected = H.decode(canon(world, task.agent, task))
            except Unsupported:  # the oracle refuses the op (a collision off the data area): C must refuse it too
                expected = None
            flat = np.asarray(ops, np.int32)
            out = np.zeros(1 << 16, np.int64)
            n = lib.rt_world(rec, seed, 6, None if spawns is None else spawns.ctypes.data, 0 if spawns is None else
                             len(spawns), bits, flat.ctypes.data, len(ops), out.ctypes.data, len(out))
            operations += 1
            if expected is None:
                assert n < 0, (rec, step)
                break
            assert n > 0, (rec, step, -1 - n)
            got = H.decode(out[:n])
            lines += [f'record {rec} op {step}: {f}' for f in world_fields if not np.array_equal(expected[f], got[f])]
    assert not lines, lines[:8]
    assert operations >= 8 * len(worlds)


@pytest.fixture(scope='module')
def supply_world(inputs, replay):
    """What the tracker test reads of 4 supply layouts: each one's groves of the oracle and a stump-like loc type."""
    from sim.search import OAK, TREE, groves, supply_sites
    tables = inputs['tables']
    module, lib = replay
    lib.rt_open.argtypes = [ctypes.c_char_p]
    assert lib.rt_open(str(PACK).encode()) == 0
    lib.rt_supply.restype = ctypes.c_int
    lib.rt_supply.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_int, ctypes.c_void_p]
    layouts = inputs['supply'][0][:4]
    return dict(layouts=layouts, sites=[[g for g, _ in supply_sites(layout, tables)] for layout in layouts],
                groves=[groves(layout, tables) for layout in layouts],
                stump=min(k for k, v in tables.loc_types.items() if v['debugname'] not in (TREE, OAK)))


STEPS = st.lists(st.tuples(st.integers(0, 7), st.integers(-5, 5), st.integers(-5, 5), st.integers(1, 9),
                           st.lists(st.tuples(st.booleans(), st.integers(1, 3)), max_size=2),
                           st.lists(st.tuples(st.integers(0, 7), st.integers(0, 2)), max_size=2)), min_size=1, max_size=12)


def test_the_supply_tracker_equals_the_oracles(replay, inputs, supply_world):
    """rs_supply_observe against search.SupplyTracker itself, on a supply layout's grove: logs (one per produced item,
    its tick from arm, local by the player's tile at the observation within 3 tiles of any grove position), the grove's
    standing trees as loc types and activity change, and the first tick none stands; the gap that follows from them.
    Hypothesis draws only ints (layout, site, steps): the tables and layouts stay out of a failing example's repr."""
    from types import SimpleNamespace
    from baseline.resource_task import GOAL_FAMILIES
    from sim.search import SupplyTracker
    from sim.world import ORIGIN
    tables, lib, w = inputs['tables'], replay[1], supply_world
    wood, other = sorted(GOAL_FAMILIES['wood'])[0], sorted(GOAL_FAMILIES['fish'])[0]  # obj ids

    @settings(deadline=None, max_examples=120, database=None)
    @given(st.integers(0, 3), st.integers(0, 2), STEPS)
    def check(layout, site, steps):
        sites, layout_ = w['sites'][layout], w['layouts'][layout]
        grove = sites[site % len(sites)]
        rows = w['groves'][layout][grove]
        positions = [(int(layout_.locs[k, 1]), int(layout_.locs[k, 2])) for k in rows]
        locs = [SimpleNamespace(type=int(r[0]), active=True) for r in layout_.locs]
        loc_at = {}
        for k, r in enumerate(layout_.locs):
            loc_at.setdefault((0, ORIGIN[0] + int(r[1]), ORIGIN[1] + int(r[2])), []).append(locs[k])
        player = SimpleNamespace(x=0, z=0, levels=[1] * 21)
        world = SimpleNamespace(events=[], loc_at=loc_at, t=tables, tick=0)
        task = SimpleNamespace(world=world, agent=SimpleNamespace(player=player), start_tick=0, event_start=0, raw={})
        tracker = SupplyTracker(task, positions)
        flat, exhausted, tick = [], [], 0
        for at, dx, dz, dt, produced, changes in steps:
            tick += dt
            x, z = ORIGIN[0] + positions[at % len(positions)][0] + dx, ORIGIN[1] + positions[at % len(positions)][1] + dz
            world.tick, player.x, player.z = tick, x, z
            flat += [tick, x, z, len(produced)]
            for is_wood, count in produced:
                world.events.append((tick - 1, 'produce', dict(obj=wood if is_wood else other, count=count)))
                flat += [wood if is_wood else other, count, tick - 1]
            flat.append(len(changes))
            for row, kind in changes:
                k = rows[row % len(rows)]
                locs[k].type, locs[k].active = (w['stump'], True) if kind == 0 else (int(layout_.locs[k, 0]), kind == 2)
                flat += [k, locs[k].type, int(locs[k].active)]
            tracker.observe()
            exhausted.append(-1 if tracker.exhausted is None else tracker.exhausted)
        flat = np.asarray(flat, np.int32)
        logs, ex = np.zeros(2 * 256, np.int32), np.zeros(len(steps), np.int32)
        n = lib.rt_supply(layout, grove, flat.ctypes.data, len(steps), logs.ctypes.data, 256, ex.ctypes.data)
        assert n >= 0, n
        assert [tuple(r) for r in logs[:2 * n].reshape(-1, 2).tolist()] == [(t, int(local)) for t, local in tracker.logs]
        assert ex.tolist() == exhausted
    check()
