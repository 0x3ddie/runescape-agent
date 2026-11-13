"""The native observation and text (sim/native/obs.c, text.c) against the oracle they port: sim/agent.py's
scan, build_obs, project_items (the click slots among them), build_mask, snapshot, describe_action, feedback and
capture_text, and rl/obs.py project with its contracts (obs-v3, mask-v3, feedback-v2), bit for bit.

The C side is a harness library: every sim/native/*.c but module.c, built with the extension's flags, plus a driver
(DRIVER below); the weapon styles the observation reads are combat.c's. Worlds are built with
world.c's own calls (rs_world_init on the pack's layout record, rs_player_init, the loc ops, rs_obj_append) and set
equal to an oracle state; the agent's fields are set through the driver.

Every check calls the Python functions themselves:
  - text vectors of every message and of Hypothesis-drawn chats and dialogs against feedback_contract.text_vector;
  - capture_text over Hypothesis-drawn chat and component streams against Agent.capture_text;
  - rl/obs.project of raw observations recorded from short CpuBatch runs (v2, v3, v5), injected into RsRawObs;
  - CpuBatch states (v2, v3, v5, keyed biased play) transplanted into C worlds, with the agent's state perturbed:
    the raw observation field by field, the mask bit for bit, snapshot, describe_action for every action, feedback,
    and the projection of the C observation against the oracle's.
Names are rows of a synthetic names-v1 table holding every type name of the pack (written to a temporary file; the
pack's own rows are replaced by its rows), so a wrong kind or string shows. The archive, pools and the exported pack
(sim/tests PACK) are read where they exist and the tests skip otherwise; nothing private is printed beyond
ids and counts, and Hypothesis draws only ids."""
from __future__ import annotations

from copy import deepcopy
import ctypes
import hashlib
import os
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

from sim import native
from sim.tests import PACK

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'logs/sim/data/draynor-nc1.npz'
COMPACT = ROOT / 'logs/sim/world/train-8-nc1-derived.npz'
SUPPLY = ROOT / 'logs/supply/pools/train-128.npz'
ORIGIN = (2880, 3328)

DRIVER = r'''
#include <stdlib.h>
#include <string.h>
#include "rs.h"

/* ---- the pack ------------------------------------------------------------------------------------------------- */
void t_pack_info(const RsPack *P, int32_t *out) {
    out[0] = P->compact0, out[1] = P->n_compact, out[2] = P->supply0, out[3] = P->n_supply, out[4] = P->n_layouts;
    out[5] = P->n_strings, out[6] = P->n_msgs, out[7] = P->n_tokens;
}
void t_set_rows(RsPack *P, const int32_t *rows) { P->str_row = rows; }   /* the test's names-v1 rows */
int32_t t_variant(const RsPack *P, int32_t layout, int32_t grove, int32_t keep) { return rs_variant_find(P, layout, grove, keep); }
int32_t t_msg_text(const RsPack *P, int32_t msg) { return P->msgs[msg].text; }

/* ---- a slot and its world -------------------------------------------------------------------------------------- */
static RsConfig t_cfg;
RsSlot *t_slot_new(const RsPack *P, const float *log1p) {
    RsSlot *s = calloc(1, sizeof *s);
    s->P = P, s->cfg = &t_cfg, s->log1p = log1p;
    return s;
}
void t_slot_free(RsSlot *s) { free(s); }
void t_err(const RsSlot *s, int32_t *out) {
    out[0] = s->w.err.code, out[1] = s->w.err.exc, out[2] = s->w.err.reason;
    for (int i = 0; i < RS_ERR_ARGS; i++) out[3 + i] = s->w.err.arg[i];
}
const char *t_err_text(const RsSlot *s) { return s->w.err.text; }
/* a fresh world on layout record rec (spawns: nid, type, x, z, level, size rows; n < 0: the layout's) and a fresh
   agent as Agent.__init__ makes it (rs.h Agent) */
int t_world_init(RsSlot *s, int32_t rec, int64_t seed, const int32_t *spawns, int32_t n) {
    RsNpcSpawn sp[RS_MAX_NPCS];
    for (int i = 0; i < n; i++)
        sp[i] = (RsNpcSpawn){spawns[6 * i], spawns[6 * i + 1], spawns[6 * i + 2], spawns[6 * i + 3], spawns[6 * i + 4],
                             spawns[6 * i + 5]};
    memset(&s->a, 0, sizeof s->a);
    s->a.bank_pages = 1, s->a.death_tick = -1;
    return rs_world_init(&s->w, s->P, &s->P->layouts[rec], seed, 0, n < 0 ? NULL : sp, n < 0 ? 0 : n, 0, 25);
}
/* the player: x, z, level, modal bits, modal_main, modal_chat, script mode (0 none), n resume buttons, 4 x (iface,
   com), origin x, z, dead, tick */
void t_player(RsSlot *s, const int32_t *v) {
    World *w = &s->w;
    rs_player_init(w);
    Player *p = &w->p;
    p->x = (int16_t)v[0], p->z = (int16_t)v[1], p->level = (uint8_t)v[2];
    w->has_player = 1;
    rs_occupy(w, p->level, p->x, p->z, 1, 1);
    p->modal = (uint8_t)v[3], p->modal_main = v[4], p->modal_chat = v[5];
    if (v[6]) p->script.serial = ++w->next_serial, p->script.mode = (uint8_t)v[6], p->script.depth = 1;
    p->n_resume = (uint8_t)v[7];
    for (int k = 0; k < RS_RESUME_BUTTONS; k++) p->resume_iface[k] = v[8 + 2 * k], p->resume_com[k] = v[9 + 2 * k];
    p->origin_x = (int16_t)v[16], p->origin_z = (int16_t)v[17], p->dead = (uint8_t)v[18];
    w->tick = v[19];
}
void t_player_stats(RsSlot *s, const int16_t *levels, const int16_t *base, const int32_t *stats, const int32_t *varps) {
    memcpy(s->w.p.levels, levels, sizeof s->w.p.levels);
    memcpy(s->w.p.base_levels, base, sizeof s->w.p.base_levels);
    memcpy(s->w.p.stats, stats, sizeof s->w.p.stats);
    memcpy(s->w.p.varp, varps, sizeof s->w.p.varp);
}
/* inventory which (0 inv, 1 worn, 2 bank): (id, count) per slot, id -1 empty */
void t_inventory(RsSlot *s, int which, const int32_t *v) {
    Slot *slot = which == 0 ? s->w.p.inv : which == 1 ? s->w.p.worn : s->w.p.bank;
    int cap = which == 0 ? RS_INV_CAP : which == 1 ? RS_WORN_CAP : RS_BANK_CAP;
    for (int k = 0; k < cap; k++) slot[k].id = v[2 * k], slot[k].count = v[2 * k + 1];
}
/* npc slot k (nid order): nid (checked), x, z, level, active, tele, target is the player, delayed */
int t_npc(RsSlot *s, int k, const int32_t *v) {
    Npc *n = &s->w.npcs[k];
    if (k >= s->w.n_npcs || n->nid != v[0]) return -1;
    n->x = (int16_t)v[1], n->z = (int16_t)v[2], n->level = (uint8_t)v[3], n->active = (uint8_t)v[4];
    n->tele = (uint8_t)v[5], n->target_player = (uint8_t)v[6], n->delayed = (uint8_t)v[7];
    return 0;
}
int t_n_npcs(const RsSlot *s) { return s->w.n_npcs; }
/* map loc ops by slot (= Loc.index for map locs): 0 change, 1 remove; LOC_ADD returns the slot (-1 on an error) */
int t_loc_op(RsSlot *s, int op, int slot, int32_t type, int32_t shape, int32_t angle) {
    return op == 0 ? rs_change_loc(&s->w, slot, type, shape, angle, 500) : rs_remove_loc(&s->w, slot, 500);
}
int t_loc_add(RsSlot *s, int32_t x, int32_t z, int32_t type, int32_t angle, int32_t shape) {
    int slot;
    return rs_loc_add(&s->w, 0, x, z, type, angle, shape, 500, &slot) ? -1 : slot;
}
int t_n_stored(const RsSlot *s) { return s->w.n_stored_locs; }
/* the live flags (without entity bits) of the oracle; the changed set follows (World.changed) */
void t_live_flags(RsSlot *s, const int32_t *flags) {
    World *w = &s->w;
    memcpy(w->flags, flags, (size_t)w->L->T * sizeof *flags);
    memset(w->changed, 0, sizeof w->changed);
    w->n_changed = 0;
    for (int t = 0; t < w->L->T; t++)
        if (w->flags[t] != w->L->flags_boot[t]) w->changed[t >> 6] |= 1ull << (t & 63), w->n_changed++;
    w->coll_ver++;
}
int t_obj(RsSlot *s, int32_t type, int32_t count, int32_t x, int32_t z, int32_t level, int active) {
    int slot;
    if (rs_obj_append(&s->w, type, count, x, z, level, 0, 100, 1, &slot)) return -1;
    s->w.objs[slot].active = (uint8_t)active;
    return 0;
}
/* the agent: inv_page, bank_page, bank_pages, has_using, using slot, id, name, style_stale, chat_seq, chat_covered,
   visited (2 x 32 bits), option_end (has, action, reason, refused, reported), decision */
void t_agent(RsSlot *s, const int32_t *v) {
    Agent *a = &s->a;
    a->inv_page = v[0], a->bank_page = v[1], a->bank_pages = v[2], a->has_using = (uint8_t)v[3];
    a->using_slot = v[4], a->using_id = v[5], a->using_name = v[6], a->style_stale = (uint8_t)v[7];
    a->chat_seq = v[8], a->chat_covered = v[9];
    a->visited = (uint64_t)(uint32_t)v[10] | (uint64_t)(uint32_t)v[11] << 32;
    a->option_end = (RsOptionEnd){(uint8_t)v[12], (uint8_t)v[14], (uint8_t)v[15], (uint8_t)v[16], v[13]};
    a->decision = v[17];
}
void t_agent_lists(RsSlot *s, int n_ref, const int32_t *ref, int n_chat, const int32_t *chat, int n_texts,
                   const int32_t *texts) {
    Agent *a = &s->a;
    a->n_refused = n_ref;
    for (int i = 0; i < n_ref; i++) {
        const int32_t *r = ref + 7 * i;
        a->refused[i] = (RsRefused){{(uint8_t)r[0], (uint8_t)r[1], (int16_t)r[2], (int16_t)r[3], r[4], r[5]}, r[6]};
    }
    a->n_chat = n_chat;
    for (int i = 0; i < n_chat; i++) a->chat[i] = (RsChatKept){chat[3 * i], chat[3 * i + 1], chat[3 * i + 2]};
    a->n_texts = n_texts;
    for (int i = 0; i < n_texts; i++)
        a->texts[i] = (RsTextEntry){texts[4 * i], texts[4 * i + 1], texts[4 * i + 2], texts[4 * i + 3]};
}
void t_agent_state(const RsSlot *s, int32_t *out) {
    const Agent *a = &s->a;
    out[0] = a->inv_page, out[1] = a->bank_page, out[2] = a->bank_pages, out[3] = a->has_using, out[4] = a->chat_covered;
    out[5] = a->option_end.reported, out[6] = a->n_act_targets;
    for (int k = 0; k < a->n_act_targets; k++) {   /* kind, nid or type, x, z */
        RsRef r = a->act_targets[k];
        const World *w = &s->w;
        int32_t *o = out + 7 + 4 * k;
        o[0] = r.kind;
        if (r.kind == RS_REF_NPC) o[1] = w->npcs[r.slot].nid, o[2] = w->npcs[r.slot].x, o[3] = w->npcs[r.slot].z;
        else if (r.kind == RS_REF_LOC) o[1] = w->locs[r.slot].type, o[2] = w->locs[r.slot].x, o[3] = w->locs[r.slot].z;
        else o[1] = w->objs[r.slot].type, o[2] = w->objs[r.slot].x, o[3] = w->objs[r.slot].z;
    }
}

/* ---- the calls ------------------------------------------------------------------------------------------------- */
int t_observe(RsSlot *s, int32_t decision, int done, int truncated, int dead, int busy, int32_t action) {
    if (rs_build_obs(s, decision, done, truncated, dead) || rs_project_items(s)) return 1;
    s->a.obs.option_busy = (uint8_t)busy, s->a.obs.option_action = action;
    return 0;
}
int t_mask(RsSlot *s) { return rs_build_mask(s); }
static void snap_out(const RsSnap *v, int64_t *out) {
    out[0] = v->tick, out[1] = v->x, out[2] = v->z, out[3] = v->level, out[4] = v->hp, out[5] = v->xp;
    out[6] = v->bank_open, out[7] = v->dialog_open, out[8] = v->n_items;
    for (int i = 0; i < v->n_items; i++) out[9 + 2 * i] = v->item_obj[i], out[10 + 2 * i] = v->item_count[i];
}
static void snap_in(const int64_t *v, RsSnap *out) {
    memset(out, 0, sizeof *out);
    out->tick = (int32_t)v[0], out->x = (int16_t)v[1], out->z = (int16_t)v[2], out->level = (uint8_t)v[3];
    out->hp = (int32_t)v[4], out->xp = v[5], out->bank_open = (uint8_t)v[6], out->dialog_open = (uint8_t)v[7];
    out->n_items = (int32_t)v[8];
    for (int i = 0; i < out->n_items; i++) out->item_obj[i] = (int32_t)v[9 + 2 * i], out->item_count[i] = v[10 + 2 * i];
}
void t_snapshot(const RsSlot *s, int64_t *out) { RsSnap v; rs_snapshot(s, &v); snap_out(&v, out); }
void t_describe(const RsSlot *s, int32_t action, int32_t *out) {
    RsMeaning m;
    rs_describe_action(s, action, &m);
    out[0] = m.action, out[1] = m.family, out[2] = m.kind, out[3] = m.name, out[4] = m.option, out[5] = m.item;
}
/* feedback with a meaning (action, family, kind, name, option, item; NULL none) and a before snapshot (NULL none) */
int t_feedback(RsSlot *s, const int32_t *meaning, const int64_t *before, int running) {
    RsMeaning m;
    RsSnap b, after;
    if (meaning) m = (RsMeaning){meaning[0], (uint8_t)meaning[1], (uint8_t)meaning[2], meaning[3], meaning[4], meaning[5]};
    if (before) snap_in(before, &b);
    rs_snapshot(s, &after);
    return rs_feedback(s, meaning ? &m : NULL, before ? &b : NULL, &after, running);
}
int t_project(RsSlot *s, uint8_t *row) { return rs_project(s, row); }
/* the late log1p features of the last rs_project (rs.h RsLate) as (off, k, count) triples */
int t_late(const RsSlot *s, int64_t *out, int cap) {
    for (int i = 0; i < s->n_late_next && i < cap; i++)
        out[3 * i] = s->late_next[i].off, out[3 * i + 1] = s->late_next[i].k, out[3 * i + 2] = s->late_next[i].count;
    return s->n_late_next;
}
void t_start_feedback(RsSlot *s) { rs_episode_start_feedback(&s->a.obs); }
/* text capture: lines into the world's chat ring and text log, then capture_text */
int t_mes(RsSlot *s, int32_t tick, int32_t msg) { s->w.tick = tick; return rs_mes(&s->w, msg); }
int t_settext(RsSlot *s, int32_t tick, int32_t iface, int32_t com, int32_t msg) {
    s->w.tick = tick;
    return rs_settext(&s->w, iface, com, msg);
}
int t_capture(RsSlot *s) { return rs_capture_text(s); }
/* chat kept (msg, tick, seq)..., then chat_seq, chat_seen, captured, text log n, then texts (iface, com, msg, tick) */
int t_text_state(const RsSlot *s, int64_t *out) {
    const Agent *a = &s->a;
    int n = 0;
    out[n++] = a->n_chat;
    for (int i = 0; i < a->n_chat; i++) out[n++] = a->chat[i].msg, out[n++] = a->chat[i].tick, out[n++] = a->chat[i].seq;
    out[n++] = a->chat_seq, out[n++] = a->chat_seen, out[n++] = s->w.chat.captured, out[n++] = s->w.text.n;
    out[n++] = a->n_texts;
    for (int i = 0; i < a->n_texts; i++)
        out[n++] = a->texts[i].iface, out[n++] = a->texts[i].com, out[n++] = a->texts[i].msg, out[n++] = a->texts[i].tick;
    return n;
}

/* ---- RsRawObs as a flat int64 record (written by both sides; the Python side's is in encode_raw) ---------------- */
typedef struct { int64_t *b; int n, cap, dec; } Codec;
#define V(x) do { if (c->dec) (x) = c->b[c->n++]; else { if (c->n < c->cap) c->b[c->n] = (int64_t)(x); c->n++; } } while (0)
#define D(x) do { int64_t bits_; if (c->dec) { bits_ = c->b[c->n++]; memcpy(&(x), &bits_, 8); } \
                  else { memcpy(&bits_, &(x), 8); if (c->n < c->cap) c->b[c->n] = bits_; c->n++; } } while (0)
static void item(Codec *c, RsItem *it, int options) {
    V(it->slot); V(it->id); V(it->count); V(it->name);
    if (!options) return;
    V(it->n_opt);
    for (int k = 0; k < it->n_opt; k++) V(it->opt[k]);
}
static void codec(Codec *c, RsRawObs *o) {
    V(o->tick); V(o->decision); V(o->done); V(o->truncated); V(o->is_dead); V(o->bank_open); V(o->dialog_open);
    V(o->modal_open); V(o->option_busy); V(o->option_action); V(o->has_using);
    if (o->has_using) { V(o->using_.slot); V(o->using_.id); V(o->using_.name); }
    V(o->style_stale); V(o->style.mode); V(o->style.count); V(o->in_combat); V(o->inv_page); V(o->bank_page);
    V(o->bank_pages); V(o->n_dialog_options); V(o->px); V(o->pz); V(o->plevel); V(o->hp); V(o->max_hp);
    V(o->combat_level); V(o->target_index);
    for (int i = 0; i < RS_N_SKILLS; i++) { V(o->levels[i]); V(o->base_levels[i]); V(o->xp[i]); }
    V(o->n_inv);
    for (int i = 0; i < o->n_inv; i++) item(c, &o->inventory[i], 1);
    V(o->n_worn);
    for (int i = 0; i < o->n_worn; i++) { V(o->equipment[i].slot); V(o->equipment[i].id); V(o->equipment[i].name); }
    V(o->n_npcs);
    for (int i = 0; i < o->n_npcs; i++) {
        RsNpcObs *n = &o->npcs[i];
        V(n->nid); V(n->name); V(n->combat_level); V(n->dx); V(n->dz); V(n->distance); V(n->in_combat); V(n->attack_op);
    }
    for (int k = 0; k < 8; k++) item(c, &o->inv_sel[k], 1);
    for (int k = 0; k < 8; k++) item(c, &o->bank_sel[k], 0);
    V(o->n_act);
    for (int i = 0; i < o->n_act; i++) {
        RsAff *a = &o->act_sel[i];
        V(a->kind); V(a->level); V(a->x); V(a->z); V(a->dx); V(a->dz); V(a->distance); V(a->id); V(a->name);
        V(a->option); V(a->steps); V(a->count); V(a->combat_level);
    }
    V(o->n_held);
    for (int i = 0; i < o->n_held; i++) {
        RsHeld *h = &o->held_sel[i];
        V(h->slot); V(h->id); V(h->count); V(h->name); V(h->option);
    }
    for (int k = 0; k < RS_N_CLICK; k++) { V(o->clicks[k].inside); V(o->clicks[k].open); }
    V(o->n_banks);
    for (int i = 0; i < o->n_banks; i++) { V(o->bank_x[i]); V(o->bank_z[i]); }
    for (int k = 0; k < RS_N_ACTIONS; k++) V(o->mask[k]);
    V(o->fb_has_action);
    if (o->fb_has_action) {
        RsMeaning *m = &o->fb_meaning;
        V(m->action); V(m->family); V(m->kind); V(m->name); V(m->option); V(m->item);
    }
    V(o->fb_running); V(o->fb_has_ended);
    if (o->fb_has_ended) { V(o->fb_ended_action); V(o->fb_ended_reason); V(o->fb_ended_refused); }
    V(o->fb_ticks); V(o->fb_dx); V(o->fb_dz); V(o->fb_level_changed); V(o->fb_bank_opened); V(o->fb_dialog_opened);
    V(o->fb_xp); V(o->fb_hp); V(o->fb_gained); V(o->fb_lost);
    V(o->n_chat);
    for (int i = 0; i < o->n_chat; i++) { V(o->chat_msg[i]); V(o->chat_age[i]); }
    V(o->n_dialog_parts);
    for (int i = 0; i < o->n_dialog_parts; i++) V(o->dialog_part[i]);
    V(o->n_choices); V(o->has_attacker);
    if (o->has_attacker) { V(o->attacker_name); V(o->attacker_dx); V(o->attacker_dz); }
}
int t_encode(RsSlot *s, int64_t *buf, int cap) { Codec c = {buf, 0, cap, 0}; codec(&c, &s->a.obs); return c.n; }
int t_decode(RsSlot *s, const int64_t *buf, int n) {
    Codec c = {(int64_t *)buf, 0, n, 1};
    memset(&s->a.obs, 0, sizeof s->a.obs);
    codec(&c, &s->a.obs);
    return c.n;
}
'''
STUBBED = {}  # no entry point is replaced: combat.c's rs_weapon_styles is the port's own

_P, _I, _I16, _I32, _I64, _D, _F = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int16, ctypes.c_int32, ctypes.c_int64,
                                    ctypes.c_double, ctypes.c_float)
SIGNATURES = dict(
    rs_pack_open=(_I, [ctypes.c_char_p, _P, _P]), rs_pack_close=(None, [_P]), t_pack_info=(None, [_P, _P]),
    t_set_rows=(None, [_P, _P]), t_variant=(_I32, [_P, _I32, _I32, _I32]), t_msg_text=(_I32, [_P, _I32]),
    t_slot_new=(_P, [_P, _P]), t_slot_free=(None, [_P]), t_err=(None, [_P, _P]), t_err_text=(ctypes.c_char_p, [_P]),
    t_world_init=(_I, [_P, _I32, _I64, _P, _I32]), t_player=(None, [_P, _P]), t_player_stats=(None, [_P, _P, _P, _P, _P]),
    t_inventory=(None, [_P, _I, _P]), t_npc=(_I, [_P, _I, _P]), t_n_npcs=(_I, [_P]),
    t_loc_op=(_I, [_P, _I, _I, _I32, _I32, _I32]), t_loc_add=(_I, [_P, _I32, _I32, _I32, _I32, _I32]),
    t_n_stored=(_I, [_P]), t_live_flags=(None, [_P, _P]), t_obj=(_I, [_P, _I32, _I32, _I32, _I32, _I32, _I]),
    t_agent=(None, [_P, _P]), t_agent_lists=(None, [_P, _I, _P, _I, _P, _I, _P]),
    t_agent_state=(None, [_P, _P]), t_observe=(_I, [_P, _I32, _I, _I, _I, _I, _I32]), t_mask=(_I, [_P]),
    t_snapshot=(None, [_P, _P]), t_describe=(None, [_P, _I32, _P]), t_feedback=(_I, [_P, _P, _P, _I]),
    t_project=(_I, [_P, _P]), t_late=(_I, [_P, _P, _I]), t_start_feedback=(None, [_P]),
    t_mes=(_I, [_P, _I32, _I32]), t_settext=(_I, [_P, _I32, _I32, _I32, _I32]), t_capture=(_I, [_P]),
    t_text_state=(_I, [_P, _P]), t_encode=(_I, [_P, _P, _I]), t_decode=(_I, [_P, _P, _I]),
    rs_text_vector_chat=(None, [_P, _P, _I, _I, _P]), rs_text_vector_parts=(None, [_P, _P, _I, _I, _P]),
)


def ptr(array):
    """The address of a numpy array the caller keeps alive across the call."""
    return ctypes.c_void_p(array.ctypes.data)


def i32(values):
    return np.ascontiguousarray(values, np.int32)


def build_harness():
    """Every sim/native/*.c but module.c (STUBBED entry points renamed away) and the driver, in one library cached by
    its sources, header and flags."""
    header = native.generated_header()
    digest = hashlib.sha256(header.encode() + DRIVER.encode() + ' '.join(native.FLAGS).encode())
    for path in sorted(native.SOURCE.glob('*.[ch]')):
        digest.update(path.name.encode() + path.read_bytes())
    out = native.build_dir() / f'_native_obs_test-{digest.hexdigest()[:16]}'
    lib = out.with_suffix('.so')
    if lib.exists():
        return lib
    out.mkdir(parents=True, exist_ok=True)
    (out / 'rs_gen.h').write_text(header)
    (out / 'driver.c').write_text(DRIVER)
    objects = []
    for src in [*native.sources(), out / 'driver.c']:
        if src.name == 'module.c':
            continue
        renames = [f'-D{name}={name}_ported' for name in STUBBED.get(src.name, ())]
        obj = out / (src.stem + '.o')
        command = ['cc', *native.FLAGS, *renames, '-I', str(out), '-I', str(native.SOURCE), '-c', str(src), '-o', str(obj)]
        done = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr[-3000:]
        objects.append(str(obj))
    temp = lib.with_name(f'{lib.name}.{os.getpid()}.tmp')
    done = subprocess.run(['cc', '-shared', *objects, '-o', str(temp), '-lz'], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-3000:]
    os.replace(temp, lib)
    return lib


@pytest.fixture(scope='module')
def lib():
    c = ctypes.CDLL(str(build_harness()))
    for name, (res, args) in SIGNATURES.items():
        f = getattr(c, name)
        f.restype, f.argtypes = res, args
    return c


@pytest.fixture(scope='module')
def ids():
    m = native.load()
    return dict(reasons=m.reasons(), exceptions=m.exceptions(), constants=m.constants())


# ---- the pack, its strings, and a names-v1 table of every type name ----------------------------------------------
class Pack:
    """The exported pack opened in the harness: its string table (id <-> text), messages by text, layout ranges, and
    its names-v1 rows replaced by those of a synthetic table of every name the pack's types carry."""

    def __init__(self, lib, path, table_path):
        header, meta, sections = native.read_pack(path)
        if meta.get('oracle_hash') != native.oracle_hash():
            pytest.skip(f'{path.name} was exported from another oracle; export it again')
        sid = native.load().constants()['sections']
        blob, offsets = sections[sid['STRINGS']][3], sections[sid['STR_OFF']][3]
        self.texts = [blob[o:blob.index(b'\0', o)].decode() for o in offsets.tolist()]
        self.ids = {text: k for k, text in enumerate(self.texts)}
        msgs = np.frombuffer(sections[sid['MSGS']][3], native.record_dtype('RsMsg'))
        self.msg_of = {self.texts[int(t)]: k for k, t in enumerate(msgs['text'])}
        self.n_msgs = len(msgs)
        self.lib, self.P = lib, ctypes.c_void_p()
        err = (ctypes.c_byte * 64)()
        assert lib.rs_pack_open(str(path).encode(), ctypes.byref(self.P), err) == 0
        info = np.zeros(8, np.int32)
        lib.t_pack_info(self.P, ptr(info))
        self.compact0, self.n_compact, self.supply0, self.n_supply = (int(v) for v in info[:4])
        assert int(info[5]) == len(self.texts) and int(info[6]) == self.n_msgs
        # every type's names (name_of and the raw name) by kind: the synthetic table's rows, shuffled per kind
        names = {}
        for kind, section, record in (('loc', 'LOC_TYPES', 'RsLocType'), ('npc', 'NPC_TYPES', 'RsNpcType'),
                                      ('obj', 'OBJ_TYPES', 'RsObjType')):
            rows = np.frombuffer(sections[sid[section]][3], native.record_dtype(record))
            rows = rows[rows['present'] == 1]
            names[kind] = sorted({self.texts[int(k)].strip().lower() for k in np.concatenate([rows['name_of'], rows['name']])
                                  if self.texts[int(k)].strip()})
        rng = np.random.default_rng(5)
        kinds, rows_ = [], []
        for kind in ('loc', 'npc', 'obj'):
            for k in rng.permutation(len(names[kind])):
                kinds.append(kind), rows_.append(names[kind][k])
        np.savez(table_path, kind=np.array(kinds), name=np.array(rows_),
                 vector=rng.normal(size=(len(rows_), 4)).astype(np.float16))
        self.table_path = table_path
        from names import NameTable
        table = NameTable(table_path)
        self.rows = i32([[table.row(kind, text) for text in self.texts] for kind in ('loc', 'npc', 'obj')]).ravel()
        lib.t_set_rows(self.P, ptr(self.rows))
        self.log1p = native.log1p_table()

    def id(self, text):
        return self.ids[text]

    def opt(self, text):
        """A string's id, '' as 0 and None as -1 (modal ids)."""
        return -1 if text is None else self.ids[text]

    def msg(self, text):
        return self.msg_of[text]

    def string(self, k):
        return self.texts[k]


@pytest.fixture(scope='module')
def pack(lib, tmp_path_factory):
    if not PACK.exists():
        pytest.skip(f'the exported pack is absent (python -m sim.native export, {PACK})')
    p = Pack(lib, PACK, tmp_path_factory.mktemp('names') / 'names.npz')
    yield p
    lib.rs_pack_close(p.P)


@pytest.fixture
def names_all(pack, monkeypatch):
    """rl/names.py reading the synthetic table of every pack name (the conftest's eight-row table otherwise)."""
    import names
    monkeypatch.setenv('RL_NAME_TABLE', str(pack.table_path))
    names.name_table.cache_clear()
    yield
    names.name_table.cache_clear()


class Slot:
    """One RsSlot of the harness, freed with the object."""

    def __init__(self, lib, pack):
        self.lib, self.pack = lib, pack
        self.s = lib.t_slot_new(pack.P, ptr(pack.log1p))

    def __del__(self):
        self.lib.t_slot_free(self.s)

    def error(self, ids):
        """(exception name, reason name, text) of the slot's error."""
        out = np.zeros(3 + 6, np.int32)
        self.lib.t_err(self.s, ptr(out))
        text = self.lib.t_err_text(self.s)
        return (ids['exceptions'][out[1]], ids['reasons'][out[2]][0], text.decode() if text else '')

    def check(self, status, ids, what):
        assert status == 0, f'{what}: {self.error(ids)}'


@pytest.fixture(scope='module')
def tables():
    if not ARCHIVE.exists():
        pytest.skip('archive absent (logs/sim/data; docs/RUNNING.md)')
    from sim.workers import load_tables
    return load_tables(ARCHIVE)  # check_pins=False: the archive's pin check fails without vendor/rs-sdk


@pytest.fixture(scope='module')
def pools():
    if not COMPACT.exists() or not SUPPLY.exists():
        pytest.skip('the compact or supply pool is absent (logs/sim/world, logs/supply/pools)')
    from sim.world import load_pool
    layouts, derived = load_pool(COMPACT)
    supply, none = load_pool(SUPPLY)
    return layouts, derived, supply, none


# ---- the oracle's raw observation as the driver's flat record (DRIVER codec, field for field) --------------------
KIND = dict(loc=0, npc=1, obj=2)
MODES = dict(pausebutton=1, suspended=2, running=3)


def bits(value):
    return struct.unpack('<q', struct.pack('<d', float(value)))[0]


def dialog_parts(pk, agent):
    """The message ids of dialog_text's parts (agent.py:1030-1037): the open chat interface's nonblank component texts by
    com. Checked against the joined text by the callers."""
    chat = agent.player.modal_chat
    if chat is None:
        return []
    return [pk.msg(v[0]) for k, v in sorted(agent.texts.items(), key=lambda kv: kv[0][1]) if k[0] == chat and v[0].strip()]


def encode_raw(pk, o, parts):
    """(values, labels) of a raw observation dict in the driver's record order; parts: dialog_parts."""
    from act_contract import banks
    from feedback_contract import END_REASONS, FAMILIES
    out, labels = [], []

    def put(label, *values):
        for k, v in enumerate(values):
            out.append(int(v))
            labels.append(label if len(values) == 1 else f'{label}[{k}]')

    def item(label, it, options):
        put(label, it['slot'], it['id'], it['count'], pk.id(it['name']))
        if options:
            ops = it.get('options') or []
            put(f'{label}.options', len(ops), *[pk.id(x) for x in ops])
    p, using, style = o['player'], o['using'], o['style']
    put('head', o['tick'], o['decision'], o['done'], o['truncated'], p['isDead'], o['bankOpen'], o['dialogOpen'],
        o['modalOpen'], o.get('optionBusy', False), o.get('optionAction', -1), using is not None)
    if using:
        put('using', using['slot'], using['id'], pk.id(using['name']))
    put('style', style['stale'], style['mode'], style['count'])
    put('player', p['inCombat'], o['invPage'], o['bankPage'], o['bankPages'], len(o['dialogOptions']), p['x'], p['z'],
        p['level'], p['hp'], p['maxHp'], p['combatLevel'], p['targetIndex'])
    for s in o['skills']:
        put(f"skill {s['name']}", s['level'], s['baseLevel'], s['xp'])
    put('n_inv', len(o['inventory']))
    for k, it in enumerate(o['inventory']):
        item(f'inventory[{k}]', it, True)
    put('n_worn', len(o['equipment']))
    for k, it in enumerate(o['equipment']):
        put(f'equipment[{k}]', it['slot'], it['id'], pk.id(it['name']))
    put('n_npcs', len(o['npcs']))
    for k, n in enumerate(o['npcs']):
        put(f'npcs[{k}]', n['id'], pk.id(n['name']), n['combatLevel'], n['dx'], n['dz'], n['distance'], n['inCombat'],
            any('attack' in str(x).lower() for x in n['options'] or []))
    for k, it in enumerate(o['invSel']):
        item(f'invSel[{k}]', it, True)
    for k, it in enumerate(o['bankSel']):
        item(f'bankSel[{k}]', it, False)
    put('n_act', len(o['actSel']))
    for k, a in enumerate(o['actSel']):
        put(f'actSel[{k}]', KIND[a['kind']], a['level'], a['x'], a['z'], a['dx'], a['dz'], a['distance'], a['id'],
            pk.id(a['name']), pk.id(a['option']), a['steps'], a['count'], a['combatLevel'])
    put('n_held', len(o['heldSel']))
    for k, h in enumerate(o['heldSel']):
        put(f'heldSel[{k}]', h['slot'], h['id'], h['count'], pk.id(h['name']), pk.id(h['option']))
    for k, c in enumerate(o['clicks']):
        put(f'clicks[{k}]', c['inside'], c['open'])
    near = banks(o)
    put('n_banks', len(near))
    for k, tile in enumerate(near):
        put(f'banks[{k}]', *tile)
    put('mask', *o['mask'])
    fb = o['feedback']
    a, ended = fb['action'], fb['ended']
    put('fb.action', a is not None)
    if a is not None:
        put('fb.meaning', a['action'], FAMILIES.index(a['family']), KIND.get(a['kind'], 3), pk.id(a['name']),
            pk.id(a['option']), pk.id(a['item']))
    put('fb.running', fb['running'], ended is not None)
    if ended is not None:
        put('fb.ended', ended['action'], END_REASONS.index(ended['reason']), ended['refused'])
    put('fb.changes', fb['ticks'], fb['dx'], fb['dz'], fb['levelChanged'], fb['bankOpened'], fb['dialogOpened'], fb['xp'],
        fb['hp'], fb['gained'], fb['lost'])
    put('fb.chat', len(fb['chat']), *[v for c in fb['chat'] for v in (pk.msg(c['text']), c['age'])])
    put('fb.dialog', len(parts), *parts)
    put('fb.choices', len(fb['dialog']['choices']))
    attacker = o.get('attacker')
    put('attacker', attacker is not None)
    if attacker is not None:
        put('attacker', pk.id(attacker['name']), attacker['dx'], attacker['dz'])
    return out, labels


def c_record(lib, slot):
    buf = np.zeros(8192, np.int64)
    n = lib.t_encode(slot.s, ptr(buf), len(buf))
    assert n <= len(buf)
    return buf[:n].tolist()


def assert_record(c, py, labels, what):
    """The C record equals the oracle's: the first differing field by name (values are ids and ints)."""
    if c == py:
        return
    k = next((i for i, (a, b) in enumerate(zip(c, py)) if a != b), min(len(c), len(py)))
    near = labels[k] if k < len(labels) else 'past the end'
    raise AssertionError(f'{what}: field {near} (#{k}) differs: C {c[k:k + 6]} vs oracle {py[k:k + 6]} '
                         f'(lengths {len(c)}, {len(py)})')


def projected(lib, slot, ids, late=None):
    """The 31 arrays rs_project writes, by name, viewed from the packed row, its late log1p features filled as
    NativeBatch fills them (native.log1p_feature); `late`, a list, receives their (off, k, count)."""
    k = ids['constants']
    row = np.zeros(k['RS_OBS_BYTES'], np.uint8)
    assert lib.t_project(slot.s, ptr(row)) == 0, slot.error(ids)
    found = np.zeros((64, 3), np.int64)
    n = lib.t_late(slot.s, ptr(found), len(found))
    assert n <= len(found)
    for off, j, count in found[:n].tolist():
        assert row[off:off + 4].view(np.float32)[0] == 0.0
        row[off:off + 4] = np.frombuffer(native.log1p_feature(j, count).tobytes(), np.uint8)
    if late is not None:
        late.extend(found[:n].tolist())
    from sim.spec import OBSERVATION_SHAPES, dtype
    out = {}
    for key, _, _, _, n, off, nbytes in k['obs']:   # rs_gen.h RS_OBS_* (sim/native.py generated_header)
        if key not in ('budget_vec', 'critic_vec', 'goal_vec'):
            out[key] = row[off:off + nbytes].view(np.dtype(dtype(key))).reshape(OBSERVATION_SHAPES[key])
    return out


def assert_projection(c, raw, what):
    """rs_project's arrays bit for bit against rl/obs.project (float32 compared as uint32)."""
    from obs import project
    py = project(raw)
    assert set(py) == set(c), sorted(set(py) ^ set(c))
    for key, value in py.items():
        a, b = np.ascontiguousarray(c[key]), np.ascontiguousarray(value, c[key].dtype)
        if a.dtype == np.float32:
            a, b = a.view(np.uint32), b.view(np.uint32)
        if not np.array_equal(a, b):
            at = np.argwhere(a != b)[0].tolist()
            raise AssertionError(f'{what}: {key} differs at {at}: C {c[key][tuple(at)]!r} vs oracle {value[tuple(at)]!r}')


# ---- text vectors and text capture ---------------------------------------------------------------------------------
def c_vector(lib, pack, msgs, dim, parts):
    m, out = i32(msgs), np.zeros(dim, np.float32)
    (lib.rs_text_vector_parts if parts else lib.rs_text_vector_chat)(pack.P, ptr(m), len(msgs), dim, ptr(out))
    return out


def py_vector(pack, msgs, dim, parts):
    from feedback_contract import text_vector
    texts = [pack.string(pack_text(pack, m)) for m in msgs]
    return text_vector([' '.join(t.strip() for t in texts)] if parts else texts, dim)


def pack_text(pack, msg):
    return pack.lib.t_msg_text(pack.P, msg)


def same_bits(a, b):
    return np.array_equal(np.asarray(a, np.float32).view(np.uint32), np.asarray(b, np.float32).view(np.uint32))


def test_every_message_alone_has_its_text_vector(lib, pack):
    """Each message as one chat line and as a one-part dialog, at the text and the choice widths."""
    bad = [(m, dim, parts) for m in range(pack.n_msgs) for dim in (64, 16) for parts in (False, True)
           if not same_bits(c_vector(lib, pack, [m], dim, parts), py_vector(pack, [m], dim, parts))]
    assert not bad, f'{len(bad)} messages differ, first {bad[:5]}'


@settings(deadline=None, max_examples=300, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_chats_and_dialogs_have_their_text_vectors(lib, pack, data):
    """Several chat lines (each its own text: no pair crosses two) and several dialog parts (one joined text: pairs
    cross the joins, blank parts add nothing), drawn as message ids."""
    msgs = data.draw(st.lists(st.integers(0, pack.n_msgs - 1), max_size=12), label='messages')
    dim = data.draw(st.sampled_from((64, 16)), label='dim')
    for parts in (False, True):
        assert same_bits(c_vector(lib, pack, msgs, dim, parts), py_vector(pack, msgs, dim, parts)), (msgs, dim, parts)


def text_state(lib, slot):
    out = np.zeros(4096, np.int64)
    n = lib.t_text_state(slot.s, ptr(out))
    return out[:n].tolist()


@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_capture_text_keeps_the_chat_and_component_texts(lib, pack, ids, data):
    """Agent.capture_text over drawn ticks of chat lines and component texts (rewrites of a component included, bursts
    past CHAT_KEEP): the kept lines with their sequence numbers, the read marks and the texts map after every capture,
    and the text log drained."""
    from sim.agent import Agent
    ifaces = [pack.id(k) for k in sorted(pack.ids)[:6]]  # any string ids serve as interface names
    py_world = SimpleNamespace(chat_log=[], text_log=[])
    agent = Agent(py_world)
    agent.chat, agent.texts = [], {}
    slot = Slot(lib, pack)
    ticks = data.draw(st.lists(st.tuples(
        st.lists(st.integers(0, pack.n_msgs - 1), max_size=20),
        st.lists(st.tuples(st.integers(0, len(ifaces) - 1), st.integers(0, 4), st.integers(0, pack.n_msgs - 1)),
                 max_size=6)), min_size=1, max_size=12), label='ticks')
    for tick, (lines, texts) in enumerate(ticks):
        for m in lines:
            slot.check(lib.t_mes(slot.s, tick, m), ids, 'mes')
            py_world.chat_log.append((tick, pack.string(pack_text(pack, m))))
        for i, com, m in texts:
            slot.check(lib.t_settext(slot.s, tick, ifaces[i], com, m), ids, 'settext')
            py_world.text_log.append((tick, (ifaces[i], com), pack.string(pack_text(pack, m))))
        slot.check(lib.t_capture(slot.s), ids, 'capture')
        agent.capture_text()
        py = [len(agent.chat)] + [v for text, t, seq in agent.chat for v in (pack.msg(text), t, seq)]
        py += [agent.chat_seq, agent.chat_seen, len(py_world.chat_log), len(py_world.text_log), len(agent.texts)]
        c = text_state(lib, slot)
        n_texts = c[len(py) - 1]
        texts_c = {(c[len(py) + 4 * k], c[len(py) + 4 * k + 1]): (c[len(py) + 4 * k + 2], c[len(py) + 4 * k + 3])
                   for k in range(n_texts)}
        assert c[:len(py)] == py, tick
        assert texts_c == {k: (pack.msg(text), t) for k, (text, t) in agent.texts.items()}, tick


# ---- recorded CpuBatch runs ----------------------------------------------------------------------------------------
TASKS = ('goal-activity-v2', 'goal-activity-v3', 'goal-activity-v5')


def cpu_batch(tables, pools, task, n, supply_count=8):
    from sim.vec import CpuBatch
    layouts, derived, supply, none = pools
    search = (supply[:supply_count], none[:supply_count]) if task == 'goal-activity-v5' else None
    return CpuBatch(tables, layouts, n, derived=derived, task=task, search=search)


def play(env, key, steps, visit):
    """Keyed biased legal play (sim/native_harness.py Biased); visit(b, task, start, before) after the reset and after
    every step with world b's pre-step snapshot and meaning: an ended episode's terminal state (whose observation
    CpuBatch builds and discards), then the next episode's first (before None)."""
    from sim.agent import describe_action
    from sim.native_harness import Biased
    source = Biased(key)
    state, obs = env.reset(key)
    for b in range(env.n):
        visit(b, env.tasks[b], env.starts[b], None)
    for step in range(steps):
        action = source(step, obs, range(env.n), env.starts)
        before, old = [], list(zip(env.tasks, env.starts))
        for b, task in enumerate(env.tasks):
            agent = task.agent
            before.append((agent.snapshot(), describe_action(agent.last_observation, int(action[b]), agent.using)))
        state, obs, reward, done, info = env.step(state, action)
        assert not info['unsupported'].any(), env.evidence
        for b in range(env.n):
            if done[b]:
                visit(b, *old[b], before[b])
            visit(b, env.tasks[b], env.starts[b], None if done[b] else before[b])


@pytest.mark.parametrize('task', TASKS)
def test_projection_of_recorded_raw_observations(lib, pack, ids, tables, pools, names_all, task):
    """rl/obs.project of every raw observation a short keyed run projects (arm and step), injected into RsRawObs, equals
    rs_project bit for bit; the injected record reads back as written."""
    env = cpu_batch(tables, pools, task, 8)
    slot, seen = Slot(lib, pack), dict(n=0, chat=0, dialog=0, using=0, busy=0)

    def visit(b, t, start, _):
        raw, agent = t.raw, t.agent
        parts = dialog_parts(pack, agent)
        assert ' '.join(pack.string(pack_text(pack, m)).strip() for m in parts) == raw['feedback']['dialog']['text']
        values, labels = encode_raw(pack, raw, parts)
        buf = np.asarray(values, np.int64)
        assert lib.t_decode(slot.s, ptr(buf), len(buf)) == len(buf)
        assert_record(c_record(lib, slot), values, labels, f'{task} world {b} read back')
        assert_projection(projected(lib, slot, ids), raw, f'{task} world {b} decision {raw["decision"]}')
        seen['n'] += 1
        seen['chat'] += bool(raw['feedback']['chat'])
        seen['dialog'] += bool(parts)
        seen['using'] += raw['using'] is not None
        seen['busy'] += bool(raw.get('optionBusy'))
    play(env, 3, 60, visit)
    assert seen['n'] > 8 * 61 and seen['chat'] and seen['busy'] and seen['dialog'] and seen['using'], seen


def test_counts_past_the_log1p_table_project_as_the_oracle(lib, pack, ids, tables, pools, names_all):
    """Counts the 65,536-entry log1p tables do not hold project as rl/obs.project does (feedback XP is in tenths: seven
    anchovies at XP_RATE 25 pass the table): recorded raw observations with their item, page, held, wealth and
    feedback counts raised past it, up to 2**31 - 1 for a stack and 10**10 for XP, are late features of rs_project that
    native.log1p_feature fills, bit for bit."""
    env = cpu_batch(tables, pools, 'goal-activity-v5', 8)
    coins = tables.obj_id('coins')
    big = (65_536, 65_537, 99_999, 10_000_000, 2 ** 31 - 1)
    slot, seen = Slot(lib, pack), dict(n=0, late=0, held=0, wealth=0)

    def visit(b, t, start, _):
        raw, n = deepcopy(t.raw), seen['n']
        count = lambda j: big[(n + j) % len(big)]
        fb = raw['feedback']
        fb['xp'], fb['gained'], fb['lost'] = 10 ** 10 + n, count(1), count(2)
        page = {}
        for j, it in enumerate(raw['inventory']):
            it['count'] = page[it['slot']] = count(j)
        for it in raw['invSel']:
            if it['slot'] != -1:
                it['count'] = page[it['slot']]
        for h in raw['heldSel']:
            h['count'] = page[h['slot']]
        free = next((k for k in range(28) if k not in page), None)
        if free is not None:  # coins on a free slot: the wealth feature
            raw['inventory'].append(dict(slot=free, id=coins, count=count(4), name=tables.obj_types[coins]['name'],
                                         options=[]))
        values, _ = encode_raw(pack, raw, dialog_parts(pack, t.agent))
        buf = np.asarray(values, np.int64)
        assert lib.t_decode(slot.s, ptr(buf), len(buf)) == len(buf)
        late = []
        assert_projection(projected(lib, slot, ids, late), raw, f'world {b} decision {raw["decision"]}')
        seen['n'] += 1
        seen['late'] += len(late)
        seen['held'] += bool(raw['heldSel'])
        seen['wealth'] += free is not None
    play(env, 5, 50, visit)  # long enough for an episode to end (its terminal state is visited too)
    assert seen['n'] > 8 * 51 and seen['late'] >= 3 * seen['n'] and seen['held'] and seen['wealth'], seen


# ---- oracle states transplanted into C worlds -----------------------------------------------------------------------
def layout_record(lib, pack, env, start):
    """The pack's layout record of an episode's start: its compact or supply pool layout, or its scarce variant."""
    if not start.get('supply'):
        return pack.compact0 + start['layout']
    if start['variant'] == 'ample':
        return pack.supply0 + start['layout']
    from sim.search import scarce_rows
    tile = (start['tile'][0] - ORIGIN[0], start['tile'][1] - ORIGIN[1])
    _, keep = scarce_rows(env.search_layouts[start['layout']], env.tables,
                          env.supply_groves[start['layout']][start['grove']], tile)
    rec = lib.t_variant(pack.P, start['layout'], start['grove'], keep)
    assert rec >= 0, (start['layout'], start['grove'], keep)
    return rec


def transplant(lib, pack, ids, slot, rec, world, agent, spawns=None):
    """World b's oracle state into a fresh C world on layout record rec, through world.c's calls (init, player, loc
    ops, objs) and the driver's setters (npc positions and states, live flags, the agent's fields)."""
    s, p = slot.s, world.player
    rows = i32([[n['nid'], n['type'], n['x'], n['z'], n['level'], n['size']] for n in spawns] if spawns else np.zeros((0, 6)))
    slot.check(lib.t_world_init(s, rec, 0, ptr(rows), -1 if spawns is None else len(rows)), ids, 'world init')
    resume = [v for iface, com in p.resume_buttons for v in (pack.id(iface), com)]
    v = i32([p.x, p.z, p.level, sum(bit for name, bit in (('main', 1), ('side', 2), ('chat', 4)) if name in p.modal_state),
             pack.opt(p.modal_main), pack.opt(p.modal_chat), 0 if p.script is None else MODES[p.script[1]],
             len(p.resume_buttons)] + (resume + [0] * 8)[:8] + [p.origin_x, p.origin_z, p.dead, world.tick])
    lib.t_player(s, ptr(v))
    levels, base = np.asarray(p.levels, np.int16), np.asarray(p.base_levels, np.int16)
    stats, varps = i32(p.stats), i32([-1 if x is None else x for x in p.varps.values()])
    lib.t_player_stats(s, ptr(levels), ptr(base), ptr(stats), ptr(varps))
    for which, inv in enumerate((p.inv, p.worn, p.bank)):
        lib.t_inventory(s, which, ptr(i32([it if it else (-1, 0) for it in inv.items])))
    assert lib.t_n_npcs(s) == len(world.npcs)
    for k, n in enumerate(world.npcs):
        assert lib.t_npc(s, k, ptr(i32([n.nid, n.x, n.z, n.level, n.active, n.tele, n.target is p, n.delayed]))) == 0
    assert lib.t_n_stored(s) == world.n_stored_locs
    for k, loc in enumerate(world.locs):
        if loc.is_changed():
            slot.check(lib.t_loc_op(s, 0, k, loc.type, loc.shape, loc.angle), ids, 'change_loc')
        if not loc.active:
            slot.check(lib.t_loc_op(s, 1, k, 0, 0, 0), ids, 'remove_loc')
    added = sorted((loc for at in world.loc_at.values() for loc in at if loc.index >= world.n_stored_locs),
                   key=lambda loc: loc.index)
    for loc in added:
        assert lib.t_loc_add(s, loc.x, loc.z, loc.type, loc.angle, loc.shape) >= 0, slot.error(ids)
    T = world.t.size[0] * world.t.size[1]
    lib.t_live_flags(s, ptr(i32(world._flags[:T])))
    for o in world.objs:
        assert lib.t_obj(s, o.type, o.count, o.x, o.z, o.level, o.active) == 0
    from feedback_contract import END_REASONS
    x0, z0 = world.t.area['x0'] >> 3, world.t.area['z0'] >> 3
    visited = 0
    for level, cx, cz in agent.visited:
        assert level == 0 and 0 <= cx - x0 < 8 and 0 <= cz - z0 < 8
        visited |= 1 << ((cx - x0) * 8 + cz - z0)
    using, end = agent.using, agent.option_end
    picked = (using['slot'], using['id'], pack.id(using['name'])) if using else (0, 0, 0)
    ended = (end['action'], END_REASONS.index(end['reason']), end['refused'], end['reported']) if end else (0, 0, 0, 0)
    halves = np.asarray([visited & 0xffffffff, visited >> 32], np.uint32).view(np.int32).tolist()
    lib.t_agent(s, ptr(i32([agent.inv_page, agent.bank_page, agent.bank_pages, using is not None, *picked,
                             agent.style_stale, agent.chat_seq, agent.chat_covered, *halves, end is not None, *ended,
                             agent.decision])))
    refused = [[KIND[kind], int(level), int(x), int(z), int(tid), pack.id(op), until]
               for key, until in agent.refused.items() for kind, tid, x, z, level, op in [key.split(':', 5)]]
    chat = [[pack.msg(text), tick, seq] for text, tick, seq in agent.chat]
    texts = [[pack.id(iface), com, pack.msg(text), tick] for (iface, com), (text, tick) in agent.texts.items()]
    lists = [i32(x).reshape(-1) for x in (refused, chat, texts)]
    lib.t_agent_lists(s, len(refused), ptr(lists[0]), len(chat), ptr(lists[1]), len(texts), ptr(lists[2]))


# The agent fields the observation reads or writes, restored after a comparison so the run goes on as it would have.
AGENT_STATE = ('using', 'inv_page', 'bank_page', 'bank_pages', 'refused', 'chat_covered', 'option_end', 'style_stale',
               'texts', 'last_observation', 'act_targets', '_scan', '_scan_locs', '_loc_cache')
COPIED = ('refused', 'option_end', 'texts')


def perturb(rng, agent, world, blank):
    """States a run reaches rarely: a picked item (valid or stale), another page, refused affordances (active or expired),
    an unreported option end, unpublished chat lines, a stale style; and of the world, an inventory filled to 23..25
    stacks (the drop rows' threshold), an npc fighting the player (an attacker from more geometries), a
    pausebutton script without the chat modal, a blank component text on the open chat interface (`blank`: the pack's
    blank message). Returns the undo of the world's changes (the agent's are AGENT_STATE)."""
    p, last, undo = world.player, agent.last_observation, []
    if p.modal_chat is not None and rng.random() < 0.5:
        agent.texts[(p.modal_chat, int(rng.integers(8)))] = (blank, world.tick)
    if rng.random() < 0.15:
        items = list(p.inv.items)
        undo.append(lambda: setattr(p.inv, 'items', items))
        p.inv.items = list(items)
        logs, free = world.t.obj_id('logs'), [k for k, it in enumerate(items) if it is None]
        for k in free[:max(0, int(rng.integers(23, 26)) - (28 - len(free)))]:
            p.inv.items[k] = [logs, 1]
    if rng.random() < 0.3:
        near = [n for n in world.npcs if n.active and n.target is None and max(abs(n.x - p.x), abs(n.z - p.z)) <= 15]
        if near:
            n = near[int(rng.integers(len(near)))]
            undo.append(lambda: setattr(n, 'target', None))
            n.target = p
    if p.script is not None and p.script[1] == 'pausebutton' and rng.random() < 0.3:
        chat = p.modal_chat
        undo.append(lambda: setattr(p, 'modal_chat', chat))
        p.modal_chat = None
    if rng.random() < 0.3:
        held = p.inv.listed()
        if held:
            slot, obj, _ = held[int(rng.integers(len(held)))]
            stale = rng.random() < 0.2
            name = world.t.obj_types[obj]['name'] or world.t.obj_types[obj]['debugname']
            agent.using = dict(slot=(slot + 1) % 28 if stale else slot, id=obj, name=name)
    if rng.random() < 0.3:
        agent.inv_page = int(rng.integers(4))
    if rng.random() < 0.3 and last['actSel']:
        from sim.agent import affordance_key
        for a in last['actSel'][:int(rng.integers(1, 4))]:
            agent.refused[affordance_key(a)] = world.tick + int(rng.choice([-1, 0, 1, 50]))
    if agent.option_end is not None and rng.random() < 0.5:
        agent.option_end['reported'] = False
    if rng.random() < 0.5:
        agent.chat_covered = max(0, agent.chat_seq - int(rng.integers(4)))
    if rng.random() < 0.2:
        agent.style_stale = not agent.style_stale
    return undo


def compare_state(lib, pack, ids, slot, world, agent, before, meaning, what):
    """build_obs, project_items, the mask and feedback on both sides from equal states, as _finish runs them (the agent's
    option_busy left out: its result is given), then snapshot, describe_action for every action and the projection."""
    from sim.agent import describe_action
    p = world.player
    dead = p.hp <= 0 or p.dead
    busy, running = agent.option is not None and not dead, agent.option.get('action', -1) if agent.option else -1
    obs = agent.build_obs(dict(decision=agent.decision, done=dead, truncated=False, isDead=dead))
    agent.project_items(obs)
    obs['optionBusy'], obs['optionAction'] = busy, running
    obs['mask'] = agent.mask(obs)
    after = agent.snapshot()
    parts = dialog_parts(pack, agent)
    obs['feedback'] = agent.feedback(meaning, before, after, busy, agent.publish_chat(), agent.dialog_text())
    assert ' '.join(pack.string(pack_text(pack, m)).strip() for m in parts) == obs['feedback']['dialog']['text']
    slot.check(lib.t_observe(slot.s, agent.decision, dead, 0, dead, busy, running), ids, f'{what} observe')
    slot.check(lib.t_mask(slot.s), ids, f'{what} mask')
    from feedback_contract import FAMILIES
    m = None if meaning is None else i32([meaning['action'], FAMILIES.index(meaning['family']), KIND.get(meaning['kind'], 3),
                                          pack.id(meaning['name']), pack.id(meaning['option']), pack.id(meaning['item'])])
    b = None if before is None else np.asarray(snap_record(before), np.int64)
    slot.check(lib.t_feedback(slot.s, None if m is None else ptr(m), None if b is None else ptr(b), busy), ids,
               f'{what} feedback')
    values, labels = encode_raw(pack, obs, parts)
    assert_record(c_record(lib, slot), values, labels, what)
    out = np.zeros(7 + 4 * 16, np.int32)
    lib.t_agent_state(slot.s, ptr(out))
    targets = []
    for t in agent.act_targets:
        kind = 1 if type(t).__name__ == 'Npc' else 2 if type(t).__name__ == 'Loc' else 3
        targets += [kind, t.nid if kind == 1 else t.type, t.x, t.z]
    py = [agent.inv_page, agent.bank_page, agent.bank_pages, agent.using is not None, agent.chat_covered,
          bool(agent.option_end and agent.option_end['reported']), len(agent.act_targets), *targets]
    assert out[:len(py)].tolist() == py, f'{what}: agent state {out[:len(py)].tolist()} vs {py}'
    snap = np.zeros(9 + 2 * 42, np.int64)
    lib.t_snapshot(slot.s, ptr(snap))
    c_items = sorted(zip(snap[9:9 + 2 * int(snap[8]):2].tolist(), snap[10:10 + 2 * int(snap[8]):2].tolist()))
    assert snap[:9].tolist() + [v for kv in c_items for v in kv] == snap_record(after), what
    got = np.zeros(6, np.int32)
    for action in (a for a in range(96) if a != 71):   # RETREAT has no family under act-v3
        lib.t_describe(slot.s, action, ptr(got))
        d = describe_action(agent.last_observation, action, agent.using)
        want = [action, FAMILIES.index(d['family']), KIND.get(d['kind'], 3), pack.id(d['name']), pack.id(d['option']),
                pack.id(d['item'])]
        assert got.tolist() == want, f'{what}: describe_action({action})'
    return obs


def snap_record(snap):
    items = sorted(snap['items'].items())
    return [snap['tick'], snap['x'], snap['z'], snap['level'], snap['hp'], snap['xp'], snap['bankOpen'], snap['dialogOpen'],
            len(items), *[v for kv in items for v in kv]]


@pytest.mark.parametrize('task', TASKS)
def test_transplanted_states_observe_as_the_oracle(lib, pack, ids, tables, pools, names_all, task):
    """Every world state of a short keyed run (biased play: acts, held items, uses, dialogs), perturbed, built into a C
    world: the raw observation field by field, the mask bit for bit, the agent's resulting fields, snapshot, every
    action's describe_action, feedback with the step's meaning and before snapshot, and the C observation's projection
    against rl/obs.project of the oracle's."""
    from sim.combat import combat_npcs
    env = cpu_batch(tables, pools, task, 8)
    slot, rng = Slot(lib, pack), np.random.default_rng(TASKS.index(task))
    blank = next(text for text in pack.msg_of if not text.strip())
    seen = dict(n=0, live=0, dialog=0, using=0, ended=0, chat=0, objs=0, attacker=0, closed=0, refused=0, dead=0, xp=0,
                drop=0)

    def visit(b, t, start, pre):
        world, agent = t.world, t.agent
        saved = {k: deepcopy(getattr(agent, k)) if k in COPIED else getattr(agent, k) for k in AGENT_STATE}
        undo = perturb(rng, agent, world, blank) if agent.last_observation is not None else []
        spawns = combat_npcs(env.layouts[start['layout']], env.tables, start['combat']['muggers']) \
            if 'combat' in start else None
        transplant(lib, pack, ids, slot, layout_record(lib, pack, env, start), world, agent, spawns)
        before, meaning = pre if pre is not None else (None, None)
        what = f'{task} world {b} episode {env.episodes[b]} decision {agent.decision}'
        obs = compare_state(lib, pack, ids, slot, world, agent, before, meaning, what)
        assert_projection(projected(lib, slot, ids), obs, what)
        # the arm's quiet feedback in place (feedback_contract.episode_start), on both observations
        from feedback_contract import episode_start
        quiet = episode_start(obs)
        lib.t_start_feedback(slot.s)
        values, labels = encode_raw(pack, quiet, dialog_parts(pack, agent))
        assert_record(c_record(lib, slot), values, labels, f'{what} episode_start')
        seen['dead'] += obs['player']['isDead']
        seen['xp'] = max(seen['xp'], obs['feedback']['xp'])
        seen['n'] += 1
        seen['live'] += world.window_changed(world.player.level, world.player.x, world.player.z)
        seen['dialog'] += bool(obs['feedback']['dialog']['text'])
        seen['using'] += obs['using'] is not None
        seen['ended'] += obs['feedback']['ended'] is not None
        seen['chat'] += bool(obs['feedback']['chat'])
        seen['objs'] += any(a['kind'] == 'obj' for a in obs['actSel'])
        seen['attacker'] += obs['attacker'] is not None
        seen['closed'] += any(not c['open'] for c in obs['clicks'])
        seen['refused'] += any(until > world.tick for until in agent.refused.values())
        seen['drop'] += any(obs['mask'][37:45])
        for k, value in saved.items():
            setattr(agent, k, value)
        for fn in undo:
            fn()
    play(env, 5, 80, visit)
    print(task, seen)
    assert seen['n'] > 8 * 81 and all(seen[k] for k in ('live', 'dialog', 'using', 'ended', 'chat', 'objs', 'attacker', 'drop',
                                                        'closed')), seen


def fresh_agent(world, x, z, visited):
    from sim.agent import Agent
    from sim.engine import Player
    p = Player(world, 'wd1a0')
    p.x, p.z = x, z
    world.player = p
    agent = Agent(world)
    agent.player, agent.chat, agent.visited = p, [], set(visited)
    return agent


def c_raised(slot, ids, status):
    """(exception class name, text) of a failed C call, rendered as sim/native.py renders it for Python."""
    assert status != 0
    out = np.zeros(9, np.int32)
    slot.lib.t_err(slot.s, ptr(out))
    text = slot.lib.t_err_text(slot.s)
    error = SimpleNamespace(args=(int(out[0]), int(out[1]), int(out[2]), out[3:].tolist(), text.decode() if text else '', 0))
    return ids['exceptions'][out[1]], native.render(error, slot.pack)


def test_refusals_raise_as_the_oracle(lib, pack, ids, tables, pools, names_all):
    """The raises an engine state can reach, with the oracle's class and text: the player off every source
    (Unsupported), a combat style below 0 (ValueError), an occupied page slot of count 0 (the projection's ValueError);
    and the one gap: dialog choices, whose texts RsRawObs does not carry (NOT_PORTED, never a guess)."""
    from obs import project
    from sim.engine import Unsupported, World
    from sim.world import LayoutTables
    layouts, derived, _, _ = pools
    lt = LayoutTables(tables, layouts[0], derived[0])
    world, slot = World(lt), Slot(lib, pack)
    W, H = lt.W, lt.H
    off = next((ORIGIN[0] + x, ORIGIN[1] + z) for x in range(W) for z in range(H) if lt.source(0, ORIGIN[0] + x, ORIGIN[1] + z) < 0)
    on = lt.meta['sources'][0][:2]

    def fresh(x, z):
        agent = fresh_agent(world, x, z, [])
        transplant(lib, pack, ids, slot, pack.compact0, world, agent)
        return agent
    agent = fresh(*off)
    obs = agent.build_obs(dict(decision=1, done=False, truncated=False, isDead=False))
    with pytest.raises(Unsupported) as raised:
        agent.project_items(obs)
    assert c_raised(slot, ids, lib.t_observe(slot.s, 1, 0, 0, 0, 0, -1)) == ('EXC_UNSUPPORTED', str(raised.value))

    agent = fresh(*on)
    world.player.varps['com_mode'] = -1
    transplant(lib, pack, ids, slot, pack.compact0, world, agent)
    obs = agent.build_obs(dict(decision=1, done=False, truncated=False, isDead=False))
    agent.project_items(obs)
    with pytest.raises(ValueError) as raised:
        agent.mask(obs)
    assert lib.t_observe(slot.s, 1, 0, 0, 0, 0, -1) == 0
    assert c_raised(slot, ids, lib.t_mask(slot.s)) == ('EXC_VALUEERROR', str(raised.value))
    world.player.varps['com_mode'] = 0

    agent = fresh(*on)
    bones = world.t.obj_id('bones')
    world.player.inv.items[3] = [bones, 0]   # an [id, 0] stack (Inventory.add of 0 to a new stack)
    transplant(lib, pack, ids, slot, pack.compact0, world, agent)
    obs = agent.build_obs(dict(decision=1, done=False, truncated=False, isDead=False))
    agent.project_items(obs)
    obs['mask'] = agent.mask(obs)
    obs['feedback'] = agent.feedback(None, None, agent.snapshot(), False, agent.publish_chat(), agent.dialog_text())
    with pytest.raises(ValueError) as raised:
        project(obs)
    for call in (lib.t_observe(slot.s, 1, 0, 0, 0, 0, -1), lib.t_mask(slot.s), lib.t_feedback(slot.s, None, None, 0)):
        assert call == 0
    row = np.zeros(ids['constants']['RS_OBS_BYTES'], np.uint8)
    assert c_raised(slot, ids, lib.t_project(slot.s, ptr(row))) == ('EXC_VALUEERROR', str(raised.value))
    world.player.inv.items[3] = None

    agent = fresh(*on)
    world.player.script, world.player.resume_buttons = (iter(()), 'pausebutton'), [('', 1)]
    transplant(lib, pack, ids, slot, pack.compact0, world, agent)
    assert lib.t_observe(slot.s, 1, 0, 0, 0, 0, -1) == 0 and lib.t_mask(slot.s) == 0
    assert c_raised(slot, ids, lib.t_feedback(slot.s, None, None, 0))[0] == 'EXC_NOT_PORTED'
