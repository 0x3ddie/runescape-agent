"""The native agent (sim/native/agent.c) against the oracle it ports (sim/agent.py Agent): Agent.start with its login
and arrival ticks, Agent.step with its paced chunks, option_busy, act-v3's clicks and the loc, npc and obj ops by the
client's route, and dispatch for every action family, compared through the canonical record after every world cycle and after every step (the port's design,
section 5).

The C side is a harness library: every sim/native/*.c but module.c, built with the extension's flags, on the real
exported pack (sim/tests PACK) and the real paths.c, world.c, npc.c, combat.c and text.c: npcs take their
turns and hunt on both sides, player_combat_stat is combat.c's and capture_text is text.c's. The other entry points
agent.c calls into the pieces beside it are test doubles, and the oracle side is given the same doubles:
  - scripts.c: the three [login,_] timers and [changestat,_] run as the oracle does (agent.py:1396-1414, engine.py
    changestat); every content script (loc, npc, obj and held triggers) runs the test's mode (MODES), as the oracle's
    double of it does: it finishes at once, opens the chat (RsConsts.s_message1) and pauses once, delays once,
    takes one hitpoint, adds an item and xp (levels gained queue [advancestat], whose levelup finishes at once), or
    adds a coin to a stack;
  - obs.c: the raw observation's parts agent.c reads (invSel, heldSel, actSel, the mask, the act targets
    and scanned objs, using, bank pages) are injected from the oracle agent's own
    observation of the same step; feedback marks option_end reported and publishes chat as Agent.feedback does;
  - task.c: GoalEpisode._watch's terminal test (death, then horizon); canon.c: the agent's and episode's fields of
    the record as sim/native_harness.py Canon lays them out, appended to rs_world_canon's.
These doubles fix the scenarios (content by mode, the episode the test arms); the harness diff (python -m sim.native
diff) runs agent.c with the real scripts, observation, episode and canon.

Needs the archive, the compact and supply pools and the exported pack; skips without them. No private string is
printed: records hold string ids, and failures name fields, ids and ticks only."""
from __future__ import annotations

import copy
import ctypes
import hashlib
from pathlib import Path
import subprocess

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

from sim import engine, native
from sim.tests import PACK

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'logs' / 'sim' / 'data' / 'draynor-nc1.npz'
POOL = ROOT / 'logs' / 'sim' / 'world' / 'train-8-nc1-derived.npz'
SUPPLY = ROOT / 'logs' / 'supply' / 'pools' / 'train-128.npz'
pytestmark = pytest.mark.skipif(not all(f.exists() for f in (ARCHIVE, POOL, SUPPLY, PACK)),
                                reason='needs the archive, the compact and supply pools and the exported pack')

HARNESS_C = r'''
#include <stdlib.h>
#include <string.h>
#include "rs.h"

/* ---- scripts.c double: the timers and changestat as the oracle, content scripts per mode ----------------------- */
/* 0: a content script finishes at once; 1: it opens the chat and pauses once (as `mesbox`); 2: it delays t_delay
   ticks once (as `prospect`); 3: it takes one hitpoint (stat_sub) and finishes (as `hurt`); 4: it adds one t_obj to
   the inventory and 40 woodcutting xp (add_xp, levels gained included) and finishes (as `gain`); 5: it adds one
   t_coins (a stack: the slot's count changes) and finishes (as `item`). [advancestat]'s levelup finishes at once in
   every mode (the oracle's levelup is `noop`). */
static int t_mode, t_delay, t_obj, t_coins;
void t_set_mode(int mode, int delay, int obj, int coins) {
    t_mode = mode, t_delay = delay, t_obj = obj, t_coins = coins;
}
int rs_script_step(World *w, Frame *f, RsSusp *out) {
    Player *p = &w->p;
    out->kind = RS_DONE;
    switch (f->script[0]) {
    case RS_S_STAT_REGEN:
        for (int i = 0; i < RS_N_SKILLS; i++)
            if (p->base_levels[i] > p->levels[i] && i != RS_SKILL_PRAYER && i != RS_SKILL_HITPOINTS)
                RS_TRY(rs_stat_add(w, i, 1, 0));
        return 0;
    case RS_S_STAT_BOOST_RESTORE:
        for (int i = 0; i < RS_N_SKILLS; i++)
            if (p->base_levels[i] < p->levels[i] && i != RS_SKILL_PRAYER && i != RS_SKILL_HITPOINTS)
                RS_TRY(rs_stat_sub(w, i, 1, 0));
        return 0;
    case RS_S_HEALTH_REGEN: return rs_stat_heal(w, RS_SKILL_HITPOINTS, 1, 0, RS_HP_REGEN);
    case RS_S_CHANGESTAT: return rs_player_combat_stat(w);
    case RS_S_LEVELUP: return 0;
    default:
        if (t_mode == 1 && f->pc[0] == 0) {
            rs_open_chat(w, w->P->consts->s_message1);
            f->script[0] = RS_S_MESBOX, f->pc[0] = 1;
            out->kind = RS_PAUSE;
        } else if (t_mode == 2 && f->pc[0] == 0) {
            f->script[0] = RS_S_PROSPECT, f->pc[0] = 1;
            out->kind = RS_DELAY, out->n = t_delay;
        } else if (t_mode == 3) {
            return rs_stat_sub(w, RS_SKILL_HITPOINTS, 1, 0);
        } else if (t_mode == 4) {
            (void)rs_inv_add(w->P, p->inv, RS_INV_CAP, 0, t_obj, 1);
            return rs_add_xp(w, RS_SKILL_WOODCUTTING, 40);
        } else if (t_mode == 5) {
            (void)rs_inv_add(w->P, p->inv, RS_INV_CAP, 0, t_coins, 1);
        }
        return 0;
    }
}

/* ---- task.c double: GoalEpisode._watch's check (task.py:162-176), holdings not kept ----------------------------- */
void rs_task_watch(RsSlot *s) {
    Episode *e = &s->ep;
    if (e->terminal || !e->armed) return;
    if (s->w.p.levels[RS_SKILL_HITPOINTS] <= 0 || s->w.p.dead)
        e->terminal = 1, e->term_reason = RS_TERM_DEATH, e->term_tick = s->w.tick;
    else if (s->w.tick >= e->end) e->terminal = 1, e->term_reason = RS_TERM_HORIZON, e->term_tick = e->end;
}

/* ---- obs.c doubles: the oracle's observation parts, injected (t_obs_set) and applied by project_items -------- */
static int32_t t_obs[1024];
static int t_obs_len;
void t_obs_set(const int32_t *buf, int n) { memcpy(t_obs, buf, (size_t)n * sizeof *buf); t_obs_len = n; }
static int loc_slot_of(const World *w, int32_t index) {
    for (int s = 0; s < w->n_loc_slots; s++) if (w->locs[s].serial && w->locs[s].index == index) return s;
    return -1;
}
static int npc_slot_of(const World *w, int32_t nid) {
    for (int s = 0; s < w->n_npcs; s++) if (w->npcs[s].nid == nid) return s;
    return -1;
}
int t_obs_apply(RsSlot *s) {
    Agent *a = &s->a;
    World *w = &s->w;
    RsRawObs *o = &a->obs;
    const int32_t *v = t_obs;
    for (int k = 0; k < 8; k++, v += 4)
        o->inv_sel[k] = (RsItem){.slot = (int16_t)v[0], .id = v[1], .count = v[2], .name = v[3]};
    o->n_held = *v++;
    for (int k = 0; k < 8; k++, v += 5)
        o->held_sel[k] = (RsHeld){.slot = (int8_t)v[0], .id = v[1], .count = v[2], .name = v[3], .option = v[4]};
    o->n_act = *v++;
    for (int k = 0; k < 16; k++, v += 7)
        o->act_sel[k] = (RsAff){.kind = (uint8_t)v[0], .level = (uint8_t)v[1], .x = (int16_t)v[2], .z = (int16_t)v[3],
                                .id = v[4], .name = v[5], .option = v[6]};
    for (int k = 0; k < RS_N_ACTIONS; k++) o->mask[k] = (int8_t)*v++;
    a->n_act_targets = *v++;
    for (int k = 0; k < 16; k++, v += 2) {
        int slot = v[0] == RS_REF_NPC ? npc_slot_of(w, v[1]) : v[0] == RS_REF_LOC ? loc_slot_of(w, v[1])
                 : v[0] == RS_REF_OBJ && v[1] >= 0 && v[1] < w->n_objs ? w->obj_order[v[1]] : -1;
        if (k >= a->n_act_targets) { a->act_targets[k] = (RsRef){0}; continue; }
        if (slot < 0) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "test act target");
        a->act_targets[k] = v[0] == RS_REF_NPC ? rs_ref_npc(w, slot) : v[0] == RS_REF_LOC ? rs_ref_loc(w, slot)
                          : rs_ref_obj(w, slot);
    }
    a->has_using = (uint8_t)v[0], a->using_slot = v[1], a->using_id = v[2], a->using_name = v[3], v += 4;
    a->bank_page = v[0], a->bank_pages = v[1], v += 2;
    a->n_scan_objs = *v++;
    for (int k = 0; k < a->n_scan_objs; k++) a->scan_objs[k] = w->obj_order[*v++];
    a->has_obs = 1;
    return (int)(v - t_obs) == t_obs_len ? 0 : RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "test observation length");
}
int rs_build_obs(RsSlot *s, int32_t decision, int done, int truncated, int is_dead) {
    s->a.obs.decision = decision, s->a.obs.done = (uint8_t)done, s->a.obs.truncated = (uint8_t)truncated;
    s->a.obs.is_dead = (uint8_t)is_dead, s->a.obs.tick = s->w.tick;
    return 0;
}
int rs_project_items(RsSlot *s) { return t_obs_apply(s); }
int rs_build_mask(RsSlot *s) { (void)s; return 0; }
void rs_snapshot(const RsSlot *s, RsSnap *out) { memset(out, 0, sizeof *out); out->tick = s->w.tick; }
void rs_describe_action(const RsSlot *s, int32_t action, RsMeaning *out) {
    (void)s;
    memset(out, 0, sizeof *out);
    out->action = action;
}
int rs_feedback(RsSlot *s, const RsMeaning *meaning, const RsSnap *before, const RsSnap *after, int running) {
    (void)meaning, (void)before, (void)after, (void)running;
    if (s->a.option_end.has && !s->a.option_end.reported) s->a.option_end.reported = 1;
    s->a.chat_covered = s->a.chat_seq;   /* publish_chat */
    return 0;
}

/* ---- canon.c doubles: the record after each cycle (rs_trace_tick) and on demand (t_canon) ----------------------- */
typedef struct { int64_t *buf; int cap, n, at; } Em;
static void ev(Em *e, int64_t v) { if (e->n < e->cap) e->buf[e->n] = v; e->n++; }
static void open_field(Em *e, int field) { ev(e, field); e->at = e->n; ev(e, 0); }
static void close_field(Em *e) { if (e->at < e->cap) e->buf[e->at] = e->n - e->at - 1; }
static int lex(const int64_t *a, const int64_t *b, int k) {
    for (int i = 0; i < k; i++) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
static void sort_rows(int64_t *rows, int n, int k) {
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && lex(&rows[(j - 1) * k], &rows[j * k], k) > 0; j--)
            for (int c = 0; c < k; c++) {
                int64_t t = rows[(j - 1) * k + c];
                rows[(j - 1) * k + c] = rows[j * k + c], rows[j * k + c] = t;
            }
}
static void listed(Em *e, const Slot *s) {
    int n = 0;
    for (int i = 0; i < RS_INV_CAP; i++) n += s[i].id >= 0;
    ev(e, n);
    for (int i = 0; i < RS_INV_CAP; i++) if (s[i].id >= 0) ev(e, i), ev(e, s[i].id), ev(e, s[i].count);
}
int t_canon(const RsSlot *s, int64_t *buf, int cap) {
    int n = 0;
    RsErr err;
    if (rs_world_canon(&s->w, buf, cap, &n, &err)) return -1;
    Em e = {buf, cap, n, 0};
    const Agent *a = &s->a;
    const World *w = &s->w;
    const RsPack *P = w->P;
    const RsOption *o = &a->option;
    open_field(&e, RS_CF_AGENT);
    int64_t v[] = {a->decision, a->charged, a->episode_ticks, a->last_mask_sum, a->pace, a->inv_page, a->bank_page,
                   a->bank_pages, a->style_stale, a->has_using ? a->using_slot : -1, a->has_using ? a->using_id : -1,
                   a->death_tick, __builtin_popcountll(a->visited) + a->n_visited_off, a->chat_seq, a->chat_covered,
                   a->n_chat};
    for (size_t i = 0; i < sizeof v / sizeof *v; i++) ev(&e, v[i]);
    close_field(&e);
    open_field(&e, RS_CF_AGENT_OPTION);
    if (!o->has) ev(&e, 0);
    else {
        int64_t u[] = {1, o->label, o->label_k, o->start, o->last_x, o->last_z, o->last_progress,
                       o->has_action ? o->action : -1, o->done, o->done == RS_DONE_ACT ? o->done_start : 0};
        for (size_t i = 0; i < sizeof u / sizeof *u; i++) ev(&e, u[i]);
    }
    close_field(&e);
    open_field(&e, RS_CF_AGENT_ACT);
    if (!o->has || !o->has_act) ev(&e, 0);
    else {
        const RsAct *c = &o->act;
        int64_t u[] = {1, c->key.kind, c->key.id, c->key.x, c->key.z, c->key.level, c->key.option_lc, c->xp,
                       c->last_xp};
        for (size_t i = 0; i < sizeof u / sizeof *u; i++) ev(&e, u[i]);
        if (c->has_loc) ev(&e, 1), ev(&e, c->loc_type), ev(&e, c->loc_x), ev(&e, c->loc_z), ev(&e, c->loc_level);
        else ev(&e, 0);
        listed(&e, c->items);
        listed(&e, c->last_items);
    }
    close_field(&e);
    open_field(&e, RS_CF_AGENT_OPTION_END);
    if (!a->option_end.has) ev(&e, 0);
    else ev(&e, 1), ev(&e, a->option_end.action), ev(&e, a->option_end.reason), ev(&e, a->option_end.refused),
         ev(&e, a->option_end.reported);
    close_field(&e);
    int64_t rows[RS_REFUSED_CAP * 7];
    int nr = 0;
    for (int i = 0; i < a->n_refused; i++) {
        const RsRefused *r = &a->refused[i];
        if (r->until <= w->tick) continue;
        int64_t u[7] = {r->key.kind, r->key.id, r->key.x, r->key.z, r->key.level, r->key.option_lc, r->until};
        memcpy(&rows[7 * nr++], u, sizeof u);
    }
    sort_rows(rows, nr, 7);
    open_field(&e, RS_CF_AGENT_REFUSED);
    for (int i = 0; i < 7 * nr; i++) ev(&e, rows[i]);
    close_field(&e);
    nr = 0;
    for (int i = 0; i < a->n_texts; i++) {
        const RsTextEntry *t = &a->texts[i];
        int64_t u[5] = {P->str_rank[t->iface], t->com, t->iface, P->msgs[t->msg].text, t->tick};
        memcpy(&rows[5 * nr++], u, sizeof u);
    }
    sort_rows(rows, nr, 5);
    open_field(&e, RS_CF_AGENT_TEXTS);
    for (int i = 0; i < nr; i++)
        ev(&e, rows[5 * i + 2]), ev(&e, rows[5 * i + 1]), ev(&e, rows[5 * i + 3]), ev(&e, rows[5 * i + 4]);
    close_field(&e);
    const Episode *p = &s->ep;
    int64_t paid;
    memcpy(&paid, &p->paid, sizeof paid);
    open_field(&e, RS_CF_EPISODE);
    int64_t u[] = {p->armed, p->active, p->start_tick, p->end, p->event_start, p->terminal, p->term_reason,
                   p->term_tick,
                   p->decisions, p->carried_before, p->banked, paid};
    for (size_t i = 0; i < sizeof u / sizeof *u; i++) ev(&e, u[i]);
    close_field(&e);
    return e.n > cap ? -2 : e.n;
}
#define T_RECORD 16384
static int64_t *t_rows;
static int t_rows_n, t_rows_cap, t_rows_bad;
void rs_trace_tick(RsSlot *s) {
    int64_t rec[T_RECORD];
    int n = t_canon(s, rec, T_RECORD);
    if (n < 0) { t_rows_bad = 1; return; }
    if (t_rows_n + n + 1 > t_rows_cap) {
        int cap = 2 * (t_rows_n + n + 1);
        int64_t *grown = realloc(t_rows, (size_t)cap * sizeof *t_rows);
        if (!grown) { t_rows_bad = 1; return; }
        t_rows = grown, t_rows_cap = cap;
    }
    t_rows[t_rows_n++] = n;
    memcpy(t_rows + t_rows_n, rec, (size_t)n * sizeof *rec);
    t_rows_n += n;
}
/* the rows since the last t_start / t_step: [n, record...] each; returns the int64 count (-1 if a record failed) */
int t_rows_get(int64_t *out, int cap) {
    if (t_rows_bad) return -1;
    if (t_rows_n <= cap) memcpy(out, t_rows, (size_t)t_rows_n * sizeof *out);
    return t_rows_n;
}

/* ---- the driver ------------------------------------------------------------------------------------------------ */
RsSlot *t_slot_new(const RsPack *P, int32_t jitter, int32_t points, int32_t max_decisions, int32_t max_ticks) {
    RsSlot *s = calloc(1, sizeof *s);
    RsConfig *c = calloc(1, sizeof *c);
    c->pace_jitter = jitter, c->decision_points = points;
    c->max_decisions = max_decisions, c->max_episode_ticks = max_ticks;
    s->P = P, s->cfg = c;
    return s;
}
void t_slot_free(RsSlot *s) { free((void *)s->cfg); free(s); }
int t_compact0(const RsPack *P) { return P->compact0; }
int t_world(RsSlot *s, int rec, int64_t seed) {
    memset(&s->ep, 0, sizeof s->ep);
    return rs_world_init(&s->w, s->P, &s->P->layouts[rec], seed, RS_WORLD_START_TICK, NULL, 0, 0, RS_XP_RATE);
}
static void rows_reset(void) { t_rows_n = 0, t_rows_bad = 0; }
int t_start(RsSlot *s, int64_t seed, int32_t x, int32_t z, const int32_t *obj, const int32_t *count, int n,
            const int16_t *levels) {
    rows_reset();
    return rs_agent_start(s, seed, x, z, obj, count, n, levels);
}
int t_step(RsSlot *s, int32_t action) { rows_reset(); return rs_agent_step(s, action); }
int t_busy(RsSlot *s) { int busy; rows_reset(); return rs_option_busy(s, &busy) ? -1 : busy; }
void t_arm(RsSlot *s, int32_t horizon) {
    Episode *e = &s->ep;
    e->armed = e->active = 1;
    e->start_tick = s->w.tick, e->end = s->w.tick + horizon, e->event_start = s->w.n_events;
}
int t_ended(const RsSlot *s) { return rs_ended(s); }
void t_clear_waypoints(RsSlot *s) { rs_clear_waypoints(&s->w); }
void t_open_chat(RsSlot *s, int32_t iface) { rs_open_chat(&s->w, iface); }
void t_set_decision_points(RsSlot *s, int on) { s->a.decision_points = (uint8_t)on; }
int t_supply0(const RsPack *P) { return P->supply0; }
void t_set_mask_sum(RsSlot *s, int32_t sum) { s->a.last_mask_sum = sum; }
void t_err(const RsSlot *s, int32_t *out) {
    out[0] = s->w.err.code, out[1] = s->w.err.exc, out[2] = s->w.err.reason;
    for (int i = 0; i < RS_ERR_ARGS; i++) out[3 + i] = s->w.err.arg[i];
}
const char *t_err_text(const RsSlot *s) { return s->w.err.text; }
int t_inv_add(RsSlot *s, int32_t obj, int32_t count) {
    return (int)rs_inv_add(s->P, s->w.p.inv, RS_INV_CAP, 0, obj, count);
}
int t_obj_append(RsSlot *s, int32_t type, int32_t count, int32_t x, int32_t z, int32_t ticks) {
    int slot;
    return rs_obj_append(&s->w, type, count, x, z, 0, 0, ticks, 1, &slot) ? -1 : slot;
}
/* The player delayed until a tick, as a script's delay leaves it; an npc out of the player's npc list. */
void t_delay_player(RsSlot *s, int32_t until) { s->w.p.delayed = 1, s->w.p.delayed_until = until; }
int t_npc_unseen(RsSlot *s, int32_t nid) {
    int k = npc_slot_of(&s->w, nid);
    if (k < 0) return -1;
    s->w.npcs[k].observers = 0;
    return 0;
}
'''

STUBBED = {'scripts.c': ('rs_script_step',), 'task.c': ('rs_task_watch',),
           'obs.c': ('rs_build_obs', 'rs_project_items', 'rs_build_mask', 'rs_snapshot', 'rs_describe_action',
                     'rs_feedback'),
           'canon.c': ('rs_trace_tick',)}

_C = dict(p=ctypes.c_void_p, i=ctypes.c_int, i16=ctypes.c_int16, i32=ctypes.c_int32, i64=ctypes.c_int64,
          s=ctypes.c_char_p)
SIGNATURES = {
    'rs_pack_open': ('i', 's', 'p', 'p'), 't_set_mode': (None, 'i', 'i', 'i', 'i'),
    't_obs_set': (None, 'p', 'i'), 't_obs_apply': ('i', 'p'), 't_canon': ('i', 'p', 'p', 'i'),
    't_rows_get': ('i', 'p', 'i'), 't_slot_new': ('p', 'p', 'i32', 'i32', 'i32', 'i32'), 't_slot_free': (None, 'p'),
    't_compact0': ('i', 'p'), 't_world': ('i', 'p', 'i', 'i64'),
    't_start': ('i', 'p', 'i64', 'i32', 'i32', 'p', 'p', 'i', 'p'),
    't_step': ('i', 'p', 'i32'), 't_busy': ('i', 'p'), 't_arm': (None, 'p', 'i32'), 't_ended': ('i', 'p'),
    't_set_mask_sum': (None, 'p', 'i32'), 't_err': (None, 'p', 'p'), 't_err_text': ('s', 'p'),
    't_inv_add': ('i', 'p', 'i32', 'i32'), 't_clear_waypoints': (None, 'p'), 't_supply0': ('i', 'p'),
    't_open_chat': (None, 'p', 'i32'), 't_set_decision_points': (None, 'p', 'i'),
    't_obj_append': ('i', 'p', 'i32', 'i32', 'i32', 'i32', 'i32'), 't_delay_player': (None, 'p', 'i32'),
    't_npc_unseen': ('i', 'p', 'i32'),
}


def build_harness(out_dir):
    """Compile every sim/native/*.c but module.c (STUBBED entry points renamed away) and the driver into one library."""
    header = native.generated_header()
    digest = hashlib.sha256(header.encode() + HARNESS_C.encode() + ' '.join(native.FLAGS).encode())
    for path in sorted(native.SOURCE.glob('*.[ch]')):
        digest.update(path.name.encode() + path.read_bytes())
    lib = out_dir / f'agent-harness-{digest.hexdigest()[:16]}.so'
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


# ---- the oracle's doubles ---------------------------------------------------------------------------------------
MODE = dict(kind='noop', iface=None, delay=70)


# The content scripts' doubles, one per mode. A suspending one is named for an rs.h script with that suspension
# (RS_S_MESBOX, RS_S_PROSPECT), so the canon names its frame (script, pc 1) as the C double renames it.
def noop(world, p, *_):
    yield from ()


def mesbox(world, p, *_):
    p.open_chat(MODE['iface'])
    yield ('pausebutton',)


def prospect(world, p, *_):
    yield ('delay', MODE['delay'])


def hurt(world, p, *_):
    world.stat_sub(p, engine.HITPOINTS, 1, 0)
    yield from ()


def gain(world, p, *_):
    p.inv.add(world.t, MODE['obj'], 1)
    world.add_xp(p, engine.SKILLS.index('woodcutting'), 40)
    yield from ()


def item(world, p, *_):
    p.inv.add(world.t, MODE['coins'], 1)
    yield from ()


MODES = dict(noop=(noop, 0), dialog=(mesbox, 1), delay=(prospect, 2), hurt=(hurt, 3), gain=(gain, 4), item=(item, 5))


def content(world, p, arg):
    """Every content script (loc, npc, obj and held triggers) of the oracle: the mode's double."""
    return MODES[MODE['kind']][0](world, p, arg)


def set_mode(env, kind, delay=70):
    """Content scripts on both sides (MODES): done at once, a dialog, a delay, or a hitpoint lost."""
    MODE['kind'], MODE['iface'], MODE['delay'] = kind, env.pack.string(MODE['iface_id']), delay
    MODE['obj'], MODE['coins'] = env.tables.obj_id('logs'), env.tables.obj_id('coins')
    env.lib.t_set_mode(MODES[kind][1], delay, MODE['obj'], MODE['coins'])


@pytest.fixture
def oracle(env, monkeypatch):
    """The oracle with the C side's doubles: still npcs, content scripts by MODE (noop unless a test asks)."""
    set_mode(env, 'noop')
    for name in list(engine.SCRIPTS):
        monkeypatch.setitem(engine.SCRIPTS, name, content)
    held = engine.held_script
    monkeypatch.setattr(engine, 'held_script', lambda world, name: None if held(world, name) is None else content)
    monkeypatch.setattr(engine, 'levelup', noop)   # [advancestat]'s script: as the C double's RS_S_LEVELUP
    yield
    set_mode(env, 'noop')


# ---- the shared inputs ----------------------------------------------------------------------------------------
class Env:
    """The oracle's tables and pool, the pack (C pointer and string ids) and the harness library."""

    def __init__(self, lib):
        from baseline import config
        from sim import native_harness as H
        from sim.workers import load_tables
        from sim.world import LayoutTables, load_pool
        self.lib = lib
        self.tables = load_tables(ARCHIVE)  # check_pins=False: the archive's pin check fails without vendor/rs-sdk
        self.layouts, derived = load_pool(POOL)
        self.lts = [LayoutTables(self.tables, layout, derived[k]) for k, layout in enumerate(self.layouts)]
        self.supply, _ = load_pool(SUPPLY)   # the supply pool's 64-tile layouts (no derived tables: live trees)
        self.supply_lts = {}
        self.cfg = dict(config.TASK_CFG)
        module = native.load()
        self.pack = module.Pack(str(PACK))
        self.sid = H.PackStrings(self.pack)
        self.canon = H.Canon(strings=self.sid)
        self.enums, self.varps = H.header()
        consts = np.frombuffer(self.pack.section(module.constants()['sections']['CONSTS']),
                               native.record_dtype('RsConsts'))[0]
        MODE['iface_id'] = int(consts['s_message1'])
        self.P = ctypes.c_void_p()
        err = (ctypes.c_uint8 * 64)()
        assert lib.rs_pack_open(str(PACK).encode(), ctypes.byref(self.P), err) == 0
        self.c0, self.s0 = lib.t_compact0(self.P), lib.t_supply0(self.P)

    def tables_of(self, start):
        """The oracle's LayoutTables and the pack's layout record of a start (compact, or supply when it says so)."""
        if not start.get('supply_pool'):
            return self.lts[start['layout']], self.c0 + start['layout']
        from sim.search import SUPPLY as SPEC
        from sim.world import LayoutTables
        k = start['layout']
        if k not in self.supply_lts:
            lt = self.supply_lts[k] = LayoutTables(self.tables, self.supply[k])
            lt.CACHE = SPEC['cache']   # as CpuBatch builds its supply tables (a pure memo)
        return self.supply_lts[k], self.s0 + k


@pytest.fixture(scope='module')
def env(tmp_path_factory):
    lib = ctypes.CDLL(str(build_harness(tmp_path_factory.mktemp('agent'))))
    for name, (res, *args) in SIGNATURES.items():
        f = getattr(lib, name)
        f.restype = _C[res] if isinstance(res, str) else res
        f.argtypes = [_C[a] if isinstance(a, str) else a for a in args]
    return Env(lib)


def ptr(array):
    return ctypes.c_void_p(array.ctypes.data)


class Mismatch(AssertionError):
    pass


KIND = dict(loc=0, npc=1, obj=2)


class Pair:
    """One episode on both sides: the oracle's World, GoalEpisode and Agent (its rows taken where capture_text
    starts, the rs.h hash point) and the C slot."""

    def __init__(self, env, start, label='', cfg=None):
        from sim.engine import World
        from sim.task import WORLD_START_TICK, GoalEpisode
        self.env, self.start, self.label, self.history, self.ticks = env, start, label, [], 0
        self.cfg = cfg = dict(env.cfg, **(cfg or {}))
        tables, rec = env.tables_of(start)
        self.world = World(tables, seed=start['world_seed'], start_tick=WORLD_START_TICK)
        self.task = GoalEpisode(self.world, cfg, 'wood', horizon=240, scale=1.0)
        self.agent = self.task.agent
        self.rows = []
        capture = self.agent.capture_text

        def hooked():
            self.rows.append(env.canon(self.world, self.agent, self.task))
            capture()
        self.agent.capture_text = hooked
        self.s = env.lib.t_slot_new(env.P, cfg.get('paceJitterTicks', 0), int(bool(cfg.get('decisionPoints'))),
                                    cfg.get('maxDecisions', 1500), cfg.get('maxEpisodeTicks', 20000))
        assert env.lib.t_world(self.s, rec, start['world_seed']) == 0

    def close(self):
        self.env.lib.t_slot_free(self.s)

    # -- the injected observation (obs.c's part) --------------------------------------------------------------
    def obs_buffer(self):
        a, w, sid, e = self.agent, self.world, self.env.sid, self.env.enums
        obs, buf = a.last_observation, []
        for it in obs['invSel']:
            buf += [it['slot'], it['id'], it['count'], sid(it['name'])]
        for key, n, row in (('heldSel', 8, lambda h: [h['slot'], h['id'], h['count'], sid(h['name']),
                                                      sid(h['option'])]),
                            ('actSel', 16, lambda c: [KIND[c['kind']], c['level'], c['x'], c['z'], c['id'],
                                                      sid(c['name']), sid(c['option'])])):
            entries = obs[key]
            buf.append(len(entries))
            width = len(row(entries[0])) if entries else {'heldSel': 5, 'actSel': 7}[key]
            for k in range(n):
                buf += row(entries[k]) if k < len(entries) else [0] * width
        buf += [int(m) for m in obs['mask']]
        buf.append(len(a.act_targets))
        for k in range(16):
            t = a.act_targets[k] if k < len(a.act_targets) else None
            if t is None:
                buf += [0, 0]
            elif isinstance(t, engine.Npc):
                buf += [e['RS_REF_NPC'], t.nid]
            elif isinstance(t, engine.Loc):
                buf += [e['RS_REF_LOC'], t.index]
            else:
                buf += [e['RS_REF_OBJ'], next(i for i, o in enumerate(w.objs) if o is t)]
        buf += [1, a.using['slot'], a.using['id'], sid(a.using['name'])] if a.using else [0, 0, 0, 0]
        buf += [a.bank_page, a.bank_pages]
        positions = [next(i for i, o in enumerate(w.objs) if o is s['obj']) for s in a._scan['objs']]
        buf += [len(positions), *positions]
        return np.asarray(buf, np.int32)

    def inject(self, now=False):
        buf = self.obs_buffer()
        self.env.lib.t_obs_set(ptr(buf), len(buf))
        if now:
            assert self.env.lib.t_obs_apply(self.s) == 0

    def refresh(self):
        """The oracle agent's observation made again (after a test edit of the state), injected into C now."""
        a = self.agent
        obs = a.build_obs(dict(decision=a.decision, done=False, truncated=False, isDead=False))
        a.project_items(obs)
        obs['mask'] = a.mask(obs)
        a.last_mask_sum = sum(obs['mask'])
        self.inject(now=True)
        self.env.lib.t_set_mask_sum(self.s, a.last_mask_sum)
        self.check_state('refresh')

    # -- comparison -------------------------------------------------------------------------------------------
    def c_canon(self):
        buf = np.zeros(1 << 15, np.int64)
        n = self.env.lib.t_canon(self.s, ptr(buf), len(buf))
        assert n >= 0, 'the C record failed'
        return buf[:n].copy()

    def c_rows(self):
        buf = np.zeros(1 << 22, np.int64)
        n = self.env.lib.t_rows_get(ptr(buf), len(buf))
        assert 0 <= n <= len(buf), 'a C row failed'
        rows, k = [], 0
        while k < n:
            m = int(buf[k])
            rows.append(buf[k + 1:k + 1 + m].copy())
            k += 1 + m
        return rows

    def fail(self, what, a, b):
        from sim.native_harness import field_diff
        raise Mismatch(f'{self.label} {what} after actions {self.history[-12:]}: '
                       + '; '.join(field_diff(a, b)))

    def check_rows(self, what):
        py, c = self.rows, self.c_rows()
        self.rows = []
        self.ticks += len(py)
        if len(py) != len(c):
            raise Mismatch(f'{self.label} {what}: {len(py)} oracle ticks, {len(c)} native ticks '
                           f'(actions {self.history[-12:]})')
        for k, (a, b) in enumerate(zip(py, c)):
            if not np.array_equal(a, b):
                self.fail(f'{what}, tick row {k}', a, b)

    def check_state(self, what):
        a, b = self.env.canon(self.world, self.agent, self.task), self.c_canon()
        if not np.array_equal(a, b):
            self.fail(f'{what}, end state', a, b)

    def error(self):
        out = (ctypes.c_int32 * 9)()
        self.env.lib.t_err(self.s, out)
        text = self.env.lib.t_err_text(self.s)

        class E:
            args = (out[0], out[1], out[2], tuple(out[3:9]), None if text is None else text.decode(), -1)
        return native.as_python(E, self.env.pack)

    def both(self, what, python, c):
        """Run one call on each side: equal outcomes (a return, or the same exception class and text), rows, state."""
        try:
            python()
            raised = None
        except engine.Unsupported as error:
            raised = error
        status = c()
        if raised is not None:
            assert status == 1, f'{self.label} {what}: the oracle raised Unsupported, native returned {status}'
            native_error = self.error()
            assert type(native_error) is type(raised) and str(native_error) == str(raised), \
                f'{self.label} {what}: {type(native_error).__name__} vs {type(raised).__name__}'
        else:
            assert status == 0, f'{self.label} {what}: native failed with {self.error()!r}'
        self.check_rows(what)
        self.check_state(what)
        return raised

    # -- the episode ----------------------------------------------------------------------------------------
    def begin(self):
        """Agent.start on both sides (as GoalEpisode.start calls it), then the STYLE setup step."""
        from sim.spec import START_KIT, START_LEVELS
        start, t = self.start, self.world.t
        kit = list(START_KIT) + [(start['weapon'], 1)]
        obj = np.asarray([t.obj_id(name) for name, _ in kit], np.int32)
        count = np.asarray([c for _, c in kit], np.int32)
        levels = np.asarray([start['levels'][stat] for stat in START_LEVELS], np.int16)
        tile = start['tile']

        def py():
            self.agent.start(0, copy.deepcopy(self.cfg), tile, kit, start['levels'])
            self.inject()
        raised = self.both('start', py, lambda: self.env.lib.t_start(self.s, 0, tile[0], tile[1], ptr(obj), ptr(count),
                                                                      len(kit), ptr(levels)))
        assert raised is None
        return self

    def step(self, action):
        self.history.append(int(action))

        def py():
            self.agent.step(int(action))
            self.inject()
        return self.both(f'step {len(self.history)} ({action})', py, lambda: self.env.lib.t_step(self.s, int(action)))

    def arm(self, horizon):
        """GoalEpisode._arm's state as the canon and the watch read it (task.py:208-229), without the projection."""
        task, world = self.task, self.world
        task.terminal, task.armed, task.active = None, True, True
        task.start_tick, task.end = world.tick, world.tick + horizon
        task.event_start, task.decisions, task.paid = len(world.events), 0, 0.0
        task.banked, task.carried_before, task.bank_was_open = 0, 0, False
        task.initial = task.holdings()
        task._watch()
        self.agent.ended = lambda: task.terminal is not None
        self.env.lib.t_arm(self.s, horizon)
        self.check_state('arm')

    def busy(self):
        """Agent.option_busy called directly on both sides, between decisions: equal results and state."""
        py = self.agent.option_busy()
        c = self.env.lib.t_busy(self.s)
        assert c in (0, 1) and bool(c) == py, f'{self.label} option_busy: oracle {py}, native {c}'
        self.check_state('option_busy')

    def offer_click(self, k):
        """Click slot k offered in both last observations (its mask bit), even where its target lies off the layout
        (no route, or the 3 x 3 fallback onto it). Returns its action."""
        from act_contract import CLICK_ROWS
        action = CLICK_ROWS[k % len(CLICK_ROWS)]
        self.agent.last_observation['mask'][action] = 1
        self.inject(now=True)
        return action

    def add_items(self, name, n):
        obj = self.world.t.obj_id(name)
        for _ in range(n):
            self.agent.player.inv.add(self.world.t, obj, 1)
            assert self.env.lib.t_inv_add(self.s, obj, 1) == 1
        return obj

    def add_ground(self, obj, count, dx, dz, ticks=300):
        p = self.agent.player
        x, z = p.x + dx, p.z + dz
        if self.world.t.source(p.level, x, z) < 0:
            return
        self.world.objs.append(engine.Obj(obj, count, x, z, static=False, ticks=ticks, receiver=p.name))
        assert self.env.lib.t_obj_append(self.s, obj, count, x, z, ticks) >= 0

    def delay(self, ticks):
        """The player delayed on both sides for `ticks`, as a script's delay leaves it (delayed_until)."""
        p = self.agent.player
        p.delayed, p.delayed_until = True, self.world.tick + ticks
        self.env.lib.t_delay_player(self.s, p.delayed_until)
        self.check_state('delay')

    def unsee(self, npc):
        """An npc out of the player's npc list on both sides (observers 0), until the next tick's keep_alive."""
        npc.observers = 0
        assert self.env.lib.t_npc_unseen(self.s, npc.nid) == 0
        self.check_state('unseen')

    def clear_waypoints(self):
        self.agent.player.clear_waypoints()
        self.env.lib.t_clear_waypoints(self.s)
        self.check_state('clear_waypoints')

    def open_chat(self):
        """A chat interface (the mode's, RsConsts.s_message1) opened on both players, as a dialog script opens one."""
        self.agent.player.open_chat(MODE['iface'])
        self.env.lib.t_open_chat(self.s, MODE['iface_id'])
        self.check_state('open_chat')

    def bounds(self):
        w = self.world
        return w._x0, w._x1, w._z0, w._z1

    def ended(self):
        ended = self.task.terminal is not None
        assert ended == bool(self.env.lib.t_ended(self.s))
        return ended

    def legal(self):
        return np.flatnonzero(np.asarray(self.agent.last_observation['mask']))


def starts(env, key, n, episode=0, supply=False):
    """draw_start's keyed starts on the compact pool, or on the supply pool's layouts (its start tiles)."""
    from sim.task import draw_start
    if not supply:
        return [draw_start(key, slot, episode, env.layouts) for slot in range(n)]
    return [dict(draw_start(key, slot, episode, env.supply), supply_pool=True) for slot in range(n)]


# Biased keyed play: weights by action family, so act, held, use and dialog actions are tried often.
WEIGHT = np.ones(96)
WEIGHT[1:17] = 6      # act
WEIGHT[17:29] = 3     # click
WEIGHT[29:37] = 4     # held
WEIGHT[37:45] = 3     # drop
WEIGHT[61] = 3        # close
WEIGHT[62] = 6        # dialog continue
WEIGHT[76:84] = 3     # use
WEIGHT[84:96] = 3     # click


def pick(rng, legal, biased=True):
    w = WEIGHT[legal] if biased else np.ones(len(legal))
    return int(rng.choice(legal, p=w / w.sum()))


def family(action):
    from sim.agent import family as fam
    return fam(int(action))


def play(env, start, rng, decisions, horizon, label, setup=None, counts=None):
    """Start, the setup STYLE step, arm, then `decisions` biased random legal steps (stopping at the terminal or an
    Unsupported); `setup(pair)` edits both states after the start."""
    pair = Pair(env, start, label).begin()
    try:
        if setup is not None:
            setup(pair)
        pair.step(72)
        pair.arm(horizon)
        for _ in range(decisions):
            if pair.ended():
                break
            action = pick(rng, pair.legal())
            if counts is not None:
                counts[family(action)] = counts.get(family(action), 0) + 1
            if pair.step(action) is not None:
                break
        return pair
    finally:
        pair.close()


# ---- Agent.start -------------------------------------------------------------------------------------------------
def test_start_ticks_and_state(env, oracle):
    """Agent.start (agent.py:296-359): login at LOGIN_TILE, two ticks, levels, kit and coins, the jump, two ticks,
    the first observation at tick 10; every tick's record and the end state equal the oracle's, on every compact
    layout, then the STYLE setup step."""
    seen = set()
    for key in (3, 11, 29):
        for start in starts(env, key, 8):
            seen.add(start['layout'])
            pair = Pair(env, start, f'key {key} layout {start["layout"]}')
            try:
                pair.begin()
                assert pair.ticks == 4 and pair.world.tick == 10 and pair.agent.decision == 0
                pair.step(72)
            finally:
                pair.close()
    assert seen == set(range(len(env.layouts)))


# ---- lockstep play ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('kind', list(MODES))
@pytest.mark.parametrize('supply', [False, True], ids=['compact', 'supply'])
def test_random_play(env, oracle, kind, supply):
    """Biased keyed random legal play from real observations on both sides (6 worlds, 70 decisions each), every tick
    and every step compared, under each content-script mode, on the compact pool and on 64-tile supply layouts; the
    horizon cuts some episodes mid-decision, and in hurt mode some end in death. Every reachable family is
    dispatched."""
    set_mode(env, kind)
    seed = 7 + list(MODES).index(kind) + 10 * supply
    rng, counts = np.random.default_rng(seed), {}
    for k, start in enumerate(starts(env, 41 + seed, 6, supply=supply)):
        play(env, start, rng, 70, 160 + 40 * k, f'world {k} {kind}', counts=counts)
    # style: every episode's setup step; wait: test_clicks; close needs `using` or a dialog: dialog mode reaches it
    # here, test_families forces it otherwise
    wanted = {'act', 'click', 'held', 'use'} | ({'dialog', 'close'} if kind == 'dialog' else set())
    assert wanted <= set(counts), sorted(counts)


@settings(max_examples=10, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(key=st.integers(0, 2 ** 31 - 1), choices=st.lists(st.integers(0, 95), min_size=4, max_size=40),
       kind=st.sampled_from(list(MODES)), supply=st.booleans())
def test_action_lists(env, oracle, key, choices, kind, supply):
    """Hypothesis action lists (the action is legal[i % len(legal)]) from a keyed start."""
    set_mode(env, kind)
    (start,) = starts(env, key, 1, episode=key % 5, supply=supply)
    pair = Pair(env, start, f'key {key}').begin()
    try:
        pair.step(72)
        pair.arm(120)
        for i in choices:
            if pair.ended():
                break
            legal = pair.legal()
            if pair.step(int(legal[i % len(legal)])) is not None:
                break
    finally:
        pair.close()


# ---- families random play rarely reaches ------------------------------------------------------------------------
def test_families(env, oracle):
    """Drops (a full inventory), pick-ups (ground objs beside the player), the use-then-close family, clicks offered
    on any slot (inside the layout or not) and held actions, each forced at least once over keyed play on every
    layout."""
    from sim.agent import A_CLOSE_IF, A_USE0
    rng, counts = np.random.default_rng(5), {}
    for k, start in enumerate(starts(env, 67, 8)):
        pair = Pair(env, start, f'world {k}').begin()
        try:
            logs = pair.add_items('logs', 20)
            for dx, dz in ((0, 0), (1, 0), (0, -2), (3, 3)):
                pair.add_ground(logs, 1, dx, dz)
            pair.add_ground(pair.world.t.obj_id('coins'), 7, -1, 1)
            pair.refresh()
            pair.step(72)
            pair.arm(400)
            for i in range(40):
                if pair.ended():
                    break
                legal = pair.legal()
                if i % 9 == 4 and A_USE0 in legal:
                    action = A_USE0
                elif i % 9 == 5 and A_CLOSE_IF in legal:
                    action = A_CLOSE_IF
                elif i % 9 == 7:
                    action = pair.offer_click(int(rng.integers(24)))
                else:
                    action = pick(rng, legal)
                counts[family(action)] = counts.get(family(action), 0) + 1
                if action in range(1, 17) and pair.agent.last_observation['actSel'][action - 1]['kind'] == 'obj':
                    counts['take'] = counts.get('take', 0) + 1
                if pair.step(action) is not None:
                    break
        finally:
            pair.close()
    assert {'drop', 'close', 'click', 'take', 'held', 'use', 'act'} <= set(counts), sorted(counts)


OPS = ('click', 'click', 'wait', 'wait', 'wait', 'clear', 'continue', 'busy')


@settings(max_examples=30, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(key=st.integers(0, 2 ** 31 - 1), supply=st.booleans(), kind=st.sampled_from(list(MODES)),
       ops=st.lists(st.tuples(st.sampled_from(OPS), st.integers(0, 23)), min_size=4, max_size=40))
def test_clicks(env, oracle, key, supply, kind, ops):
    """act-v3's clicks (Agent.click: tryMove with the 3 x 3 fallback, then MoveClickHandler) on every slot, offered
    whether or not the target lies on the layout, mid-walk (from the server tile) and while a script delays the player
    (refused), between WAIT steps that continue the walk; walks cut short (both sides' waypoints cleared: the option
    ends done), a dialog continue offered with nothing suspended, and option_busy called directly between decisions.
    Every tick compared."""
    from sim.agent import A_DIALOG_CONTINUE
    set_mode(env, kind)
    (start,) = starts(env, key, 1, episode=key % 7, supply=supply)
    pair = Pair(env, start, f'key {key}').begin()
    try:
        pair.step(72)
        pair.arm(3000)
        for op, k in ops:
            if pair.ended():
                break
            if op == 'busy':
                pair.busy()
                continue
            if op == 'clear':
                pair.clear_waypoints()
                continue
            if op == 'continue':
                pair.agent.last_observation['mask'][A_DIALOG_CONTINUE] = 1
                pair.inject(now=True)
                action = A_DIALOG_CONTINUE
            elif op == 'wait':
                action = 0
            else:
                action = pair.offer_click(k)
            if pair.step(action) is not None:
                break
    finally:
        pair.close()


def test_a_delayed_click_and_an_act_on_an_npc_out_of_the_list_are_refused_alike(env, oracle):
    """The dispatch refusals in the C engine against the oracle, every tick compared, during a click walk
    (decision points off, so the walk is still running): an ACT on an npc that is not in the player's npc list is
    refused before any route, and a click while the player is delayed sends nothing and ends failed; the earlier walk's
    end is kept both times. Random play reaches neither on purpose."""
    from sim.agent import A_ACT0
    reached = dict(unseen=0, delayed=0)
    for key in range(800, 860):
        (start,) = starts(env, key, 1, supply=key % 2 == 1)
        pair = Pair(env, start, f'key {key}', cfg=dict(decisionPoints=False)).begin()
        try:
            if pair.step(72) is not None:
                continue
            pair.arm(600)
            p = pair.agent.player
            for k in (16, 20, 18, 22):   # 16 tiles north, south, east or west: a walk still running after one decision
                stopped = pair.step(pair.offer_click(k)) is not None
                if stopped or len(p.path) or pair.ended():
                    break
            # a walk with more than one decision's steps left (p.path holds turn points; the last is the walk's end), on
            # both sides alike (Pair.step returns the Unsupported both raised, else None)
            running = lambda: p.path and not pair.ended() and max(abs(p.path[-1][0] - p.x), abs(p.path[-1][1] - p.z)) > 3
            if stopped or not running():
                continue
            end = p.path[-1]
            obs = pair.agent.last_observation
            npcs = [k for k, a in enumerate(obs['actSel']) if a['kind'] == 'npc' and obs['mask'][A_ACT0 + k]]
            if npcs:
                npc = pair.agent.act_targets[npcs[0]]
                pair.unsee(npc)
                assert pair.step(A_ACT0 + npcs[0]) is None
                route = pair.agent.last_observation['route']
                assert route['refused'] == 'npc not in view' and route['waypoints'] == [], route
                assert p.target is None and p.path and p.path[-1] == end
                reached['unseen'] += 1
            if not running():
                continue
            end = p.path[-1]
            pair.delay(20)
            assert pair.step(pair.offer_click(4)) is None
            route, ended = pair.agent.last_observation['route'], pair.agent.last_observation['feedback']['ended']
            assert route['refused'] == 'delayed' and route['waypoints'] == [], route
            assert ended['reason'] == 'failed' and ended['refused'] and p.path and p.path[-1] == end
            reached['delayed'] += 1
        finally:
            pair.close()
    assert reached['unseen'] and reached['delayed'], reached


def test_a_dialog_opened_during_a_click_walk_ends_the_wait(env, oracle):
    """keep_waiting (agent.py) reads the dialog before option_busy: with a click walk running (dispatched in a short
    decision, decision points then turned on on both sides) and a chat opened, the decision ends at its first check on
    both sides with the walk still running. The soak once found the native side reading the dialog after option_busy
    and waiting on."""
    from sim.agent import K_TICKS
    stopped = 0
    for key in (500, 501, 594, 600, 608):
        (start,) = starts(env, key, 1, supply=True)
        pair = Pair(env, start, f'key {key}', cfg=dict(decisionPoints=False)).begin()
        try:
            pair.step(72)
            pair.arm(5000)
            p = pair.agent.player
            for k in range(16, 24):   # a 16-tile click with a route
                pair.step(pair.offer_click(k))
                if pair.agent.option is not None:
                    break
            pair.agent.cfg['decisionPoints'] = True
            env.lib.t_set_decision_points(pair.s, 1)
            for _ in range(2):
                if pair.agent.option is None or pair.ended():
                    break
                pair.open_chat()
                before = pair.ticks
                pair.step(0)
                stopped += (pair.ticks - before <= K_TICKS + pair.agent.pace_jitter and pair.agent.option is not None
                            and p.modal_chat is not None)
        finally:
            pair.close()
    assert stopped


@pytest.mark.parametrize('cfg', [dict(maxDecisions=6), dict(maxEpisodeTicks=45), dict(decisionPoints=False),
                                 dict(paceJitterTicks=2), dict(paceJitterTicks=0)], ids=str)
def test_cfg(env, oracle, cfg):
    """The agent's cfg (_login's copy): truncation by decisions or ticks, no decision points, other pace jitters."""
    rng = np.random.default_rng(3)
    for k, start in enumerate(starts(env, 83, 3)):
        pair = Pair(env, start, f'world {k} {cfg}', cfg=cfg).begin()
        try:
            pair.step(72)
            pair.arm(300)
            for _ in range(25):
                if pair.ended() or pair.step(pick(rng, pair.legal())) is not None:
                    break
        finally:
            pair.close()


def test_unsupported_held_script(env, oracle):
    """A held op whose trigger names a script the oracle has no port of raises the same Unsupported on both sides
    (agent.py:687 -> engine.py:1711-1716), mid-dispatch."""
    from sim.agent import A_HELD0
    module = native.load()
    types = np.frombuffer(env.pack.section(module.constants()['sections']['OBJ_TYPES']),
                          native.record_dtype('RsObjType'))
    unknown = [name for name, _, _ in module.scripts()].index('RS_S_UNKNOWN')
    held0 = env.enums['RS_OTRIG_HELD']
    found = [k for k in range(len(types)) if types[k]['present'] and not types[k]['stackable']
             and any(types[k]['iop_state'][j] == 1 and types[k]['trig_script'][held0 + j] == unknown for j in range(5))]
    assert found, 'the pack has no obj with an unported held trigger'
    (start,) = starts(env, 97, 1)
    pair = Pair(env, start, 'unsupported').begin()
    try:
        obj = int(found[len(found) // 2])
        pair.agent.player.inv.add(pair.world.t, obj, 1)
        assert env.lib.t_inv_add(pair.s, obj, 1) == 1
        pair.refresh()
        held = pair.agent.last_observation['heldSel']
        iop = pair.world.t.obj_types[obj]['iop'] or []
        op = lambda option: next(j for j, o in enumerate(iop) if o and o.lower() == option.lower())
        k = next(i for i, h in enumerate(held) if h['id'] == obj and types[obj]['trig_script'][held0 + op(h['option'])]
                 == unknown)
        raised = pair.step(A_HELD0 + k)
        assert isinstance(raised, engine.Unsupported)
    finally:
        pair.close()
