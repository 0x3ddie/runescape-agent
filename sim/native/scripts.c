/* sim/native/scripts.c: the player content scripts of sim/engine.py and sim/agent.py as state machines, and
 * the player-script dispatcher rs_script_step (rs.h SCRIPT FRAMES). Each script function steps one level d of a Frame
 * from its pc to its next suspension or its end and names the Python it mirrors (file and function,
 * as rs.h). Combat scripts are combat.c's (rs_combat_script_step); npc scripts (move_fishing_spot) are npc.c's.
 *
 * Exactness notes the code relies on:
 *   - pc k >= 1 is the k-th suspension site (ast.Yield / ast.YieldFrom in source order, `yield from ()` excluded) of
 *     the Python function, as sim/native_harness.py pc_of numbers them; the site numbers are written beside each
 *     suspension below. A level that suspends is the innermost one: depth = d + 1, or deeper while a sub-level runs.
 *   - `yield from sub(...)` pushes level d + 1 (push) and steps it (sub); a sub that never suspends in Python (get_logs,
 *     whose only site is `yield from ()`) is a plain C call, since no dump can see it running.
 *   - Locals that live across a yield sit in the level's arg row (arg[d][3]); a sub-level's return value in Frame.ret.
 *   - Draws go through world.c's rs_random / rs_stat_random (keyed (R_SCRIPT, tick, k)); every draw Python makes is
 *     made here too, including those whose result cannot matter (random(0, 3)), because the trace compares key sets.
 *   - Text comes from rs_msg by (key, fill) with the exporter's fill codes (rs.h RS_TEXT_KEYS): world.text raising
 *     Unsupported for a key the archive lacks is rs_msg's RS_E_GAME_TEXT, at the point Python calls world.text.
 *   - A Python exception other than Unsupported at a site no declared content reaches (a KeyError, IndexError or
 *     TypeError) is RS_FATAL RS_E_INTERNAL naming the class, never a guess at a successor state. */
#include <string.h>

#include "rs.h"

static int step(World *w, Frame *f, int d, RsSusp *out);

/* ---- frame helpers ----------------------------------------------------------------------------------------------- */
static int done(RsSusp *out) {
    out->kind = RS_DONE, out->n = 0;
    return 0;
}

/* Level d suspends at its site k: ('pausebutton',) or ('delay', n). */
static int pause_at(Frame *f, int d, int k, RsSusp *out) {
    f->pc[d] = (uint8_t)k, f->depth = (uint8_t)(d + 1);
    out->kind = RS_PAUSE, out->n = 0;
    return 0;
}

static int delay_at(Frame *f, int d, int k, int32_t n, RsSusp *out) {
    f->pc[d] = (uint8_t)k, f->depth = (uint8_t)(d + 1);
    out->kind = RS_DELAY, out->n = n;
    return 0;
}

/* The generator `sub(...)` of a `yield from` at level d: level d + 1 at pc 0 with its arguments. */
static int push(World *w, Frame *f, int d, int16_t script, RsRef ref, int32_t a0) {
    if (d + 2 > RS_FRAME_DEPTH) return RS_OVERFLOW(&w->err, "RS_FRAME_DEPTH script levels");
    f->depth = (uint8_t)(d + 2);
    f->script[d + 1] = script, f->pc[d + 1] = 0, f->ref[d + 1] = ref;
    memset(f->arg[d + 1], 0, sizeof f->arg[d + 1]);
    f->arg[d + 1][0] = a0;
    return 0;
}

/* One step of the `yield from` at level d's site k: *fin is 1 when the sub returned (its level popped, its value in
   f->ret), else level d stays suspended at k with the sub's suspension in out. */
static int sub(World *w, Frame *f, int d, int k, RsSusp *out, int *fin) {
    RS_TRY(step(w, f, d + 1, out));
    if (out->kind == RS_DONE) {
        f->depth = (uint8_t)(d + 1);
        *fin = 1;
    } else {
        f->pc[d] = (uint8_t)k;
        *fin = 0;
    }
    return 0;
}

/* push + sub, then the script returns when the sub does (`yield from sub(...); return`). */
static int tail_call(World *w, Frame *f, int d, int k, int16_t script, RsRef ref, int32_t a0, RsSusp *out) {
    int fin;
    RS_TRY(push(w, f, d, script, ref, a0));
    RS_TRY(sub(w, f, d, k, out, &fin));
    return fin ? done(out) : 0;
}

/* Resume the sub of site k, then return when it does. */
static int resume_tail(World *w, Frame *f, int d, int k, RsSusp *out) {
    int fin;
    RS_TRY(sub(w, f, d, k, out, &fin));
    return fin ? done(out) : 0;
}

static int bad_pc(World *w) { return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a script resumed off its suspension sites"); }

/* ---- small world reads ------------------------------------------------------------------------------------------- */
static const RsConsts *K(const World *w) { return w->P->consts; }

/* world.mes(world.text(key, **fill)) */
static int mes_text(World *w, int key, int32_t fill) {
    int32_t msg;
    RS_TRY(rs_text_msg(w, key, fill, &msg));
    return rs_mes(w, msg);
}

/* p.worn.items[WEARPOS_RHAND][0] if p.worn.items[WEARPOS_RHAND] else None (-1) */
static int32_t rhand(const World *w) {
    const Slot *s = &w->p.worn[RS_WEARPOS_RHAND];
    return s->id >= 0 ? s->id : -1;
}

static int loc_of(World *w, RsRef ref, Loc **l, const RsLocType **lt) {
    *l = rs_get_loc(w, ref);
    if (!*l) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a script's loc slot no longer holds it");
    *lt = rs_loc_type(w->P, (*l)->type);
    if (!*lt) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a loc of a type the pack lacks");
    return 0;
}

/* world.t.obj_types[obj] (a KeyError for an id the archive lacks, -1 included) */
static int obj_of(World *w, int32_t obj, const RsObjType **ot) {
    *ot = rs_obj_type(w->P, obj);
    if (!*ot) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an obj type the archive lacks (a KeyError)");
    return 0;
}

/* p.inv.items[slot] with Python's indexing (a negative slot counts from the end; past the end an IndexError) */
static int inv_slot(World *w, int32_t slot, Slot **s) {
    int32_t i = slot < 0 ? slot + RS_INV_CAP : slot;
    if (i < 0 || i >= RS_INV_CAP) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an inventory slot past its end (an IndexError)");
    *s = &w->p.inv[i];
    return 0;
}

static int transfer_event(World *w, int src, int dst, int32_t obj, int32_t count) {
    RsEvent e = {.tick = w->tick, .kind = RS_EV_TRANSFER, .a8 = (uint8_t)src, .b8 = (uint8_t)dst, .obj = obj,
                 .count = count, .v0 = obj};
    return rs_event(w, &e);
}

/* ---- P_OPLOC, P_OPNPC, INV_DELSLOT ------------------------------------------------------------------------------- */
/* P_OPLOC (engine.py): an op the type has (a 'hidden' op counts) stops the action, walks onto the loc when
   it is not operable, and sets the interaction. */
int rs_p_oploc(World *w, RsRef ref, int32_t op) {
    Loc *l;
    const RsLocType *lt;
    RS_TRY(loc_of(w, ref, &l, &lt));
    if (op < 1 || op > 5) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "p_oploc of an op outside 1..5");
    if (!lt->op[op - 1]) return 0;
    rs_stop_action(w);
    int ok;
    RS_TRY(rs_in_operable(w, ref, &ok));
    if (!ok) rs_queue_waypoint(w, l->x, l->z);
    rs_set_interaction(w, ref, RS_APLOC1 + op - 1);
    return 0;
}

/* P_OPNPC (engine.py). */
int rs_p_opnpc(World *w, RsRef ref, int32_t op) {
    if (ref.kind != RS_REF_NPC || ref.slot >= w->n_npcs) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "p_opnpc of no npc");
    const RsNpcType *nt = rs_npc_type(w->P, w->npcs[ref.slot].type);
    if (!nt) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an npc of a type the pack lacks");
    if (op < 1 || op > 5) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "p_opnpc of an op outside 1..5");
    if (!nt->op[op - 1]) return 0;
    rs_stop_action(w);
    rs_set_interaction(w, ref, RS_APNPC1 + op - 1);
    return 0;
}

/* inv_delslot (engine.py): the slot's items leave the holdings ('remove' event). */
int rs_inv_delslot(World *w, int32_t slot, int why, int32_t *obj, int32_t *count) {
    Slot *s;
    RS_TRY(inv_slot(w, slot, &s));
    if (s->id < 0) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "inv_delslot of an empty slot (a TypeError)");
    *obj = s->id, *count = s->count;
    *s = (Slot){-1, 0};
    RsEvent e = {.tick = w->tick, .kind = RS_EV_REMOVE, .a8 = (uint8_t)why, .obj = *obj, .count = *count};
    return rs_event(w, &e);
}

/* ---- dialogs (engine.py) ------------------------------------------------------------------------------- */
/* mesbox(text) (3117-3122) and objbox(text) (3125-3129): arg[0] the message; _one_page raises first. Site 1. */
static int s_box(World *w, Frame *f, int d, RsSusp *out, int32_t iface, int32_t com) {
    if (f->pc[d] == 1) return done(out);
    if (f->pc[d]) return bad_pc(w);
    int32_t msg = f->arg[d][0];
    if (msg < 0 || msg >= w->P->n_msgs) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a dialog of no message");
    if (w->P->msgs[msg].long_text) return RS_FAIL0(&w->err, RS_E_DIALOG_PAGES);
    RS_TRY(rs_settext(w, iface, com, msg));
    rs_open_chat(w, iface);
    return pause_at(f, d, 1, out);
}

/* doubleobjbox(text) (3132-3137): arg[0] the UNLOCKS row, whose lines are the text split on '|' (no _one_page). */
static int s_doubleobjbox(World *w, Frame *f, int d, RsSusp *out) {
    if (f->pc[d] == 1) return done(out);
    if (f->pc[d]) return bad_pc(w);
    const RsUnlock *u = &w->P->unlocks[f->arg[d][0]];
    int32_t iface = K(w)->s_doubleobjbox;
    for (int i = 0; i < u->n_lines; i++) RS_TRY(rs_settext(w, iface, 2 + i, u->msg[i]));
    rs_open_chat(w, iface);
    return pause_at(f, d, 1, out);
}

/* levelup(stat) (engine.py), run from the engine queue: the base level when it runs, its message, the two
   dialog lines, a continue, then the unlock's dialog if the level unlocks one. arg[0] the stat, arg[3] the level.
   Sites: 1 the pausebutton, 2 objbox, 3 doubleobjbox, 4 mesbox. */
static int s_levelup(World *w, Frame *f, int d, RsSusp *out) {
    const RsPack *P = w->P;
    int32_t stat = f->arg[d][0];
    if (stat < 0 || stat >= P->n_levelup) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "levelup of a stat outside SKILLS");
    const RsLevelupRow *row = &P->levelup[stat];
    switch (f->pc[d]) {
    case 0: {
        if (!row->present) return done(out);
        int32_t level = w->p.base_levels[stat];
        if (level < 0 || level >= 100) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "levelup of a level outside 0..99");
        f->arg[d][3] = level;
        RS_TRY(rs_mes(w, row->message));
        RS_TRY(rs_settext(w, row->iface, 1, row->dbl));
        RS_TRY(rs_settext(w, row->iface, 2, row->level_msg[level]));
        rs_open_chat(w, row->iface);
        return pause_at(f, d, 1, out);
    }
    case 1: {
        int32_t level = f->arg[d][3];
        if (!row->unlocks[level] || row->unlock[level] < 0) return done(out);
        const RsUnlock *u = &P->unlocks[row->unlock[level]];
        if (u->kind == RS_UNLOCK_OBJBOX) return tail_call(w, f, d, 2, RS_S_OBJBOX, (RsRef){0}, u->msg[0], out);
        if (u->kind == RS_UNLOCK_DOUBLEOBJBOX)
            return tail_call(w, f, d, 3, RS_S_DOUBLEOBJBOX, (RsRef){0}, row->unlock[level], out);
        return tail_call(w, f, d, 4, RS_S_MESBOX, (RsRef){0}, u->msg[0], out);
    }
    case 2: case 3: case 4:
        return resume_tail(w, f, d, f->pc[d], out);
    default:
        return bad_pc(w);
    }
}

/* ---- woodcutting (engine.py) --------------------------------------------------------------------------- */
/* tree_row (1830-1835) */
static int tree_row(World *w, const RsLocType *lt, const RsTreeRow **row) {
    if (lt->tree_row < 0) return RS_FAIL(&w->err, RS_E_TREE_ROW, lt->debugname);
    *row = &w->P->trees[lt->tree_row];
    return 0;
}

/* axe_checker (1841-1851): the best axe (AXES order) worn in the right hand or carried; -1 None. */
static int axe_checker(World *w, int print_mes, int32_t *axe, int *rank) {
    int32_t rh = rhand(w);
    for (int a = 0; a < 7; a++) {
        int32_t obj = K(w)->axes[a];
        if (obj >= 0 && (rh == obj || rs_inv_total(w->p.inv, RS_INV_CAP, obj) > 0)) {
            *axe = obj, *rank = a;
            return 0;
        }
    }
    *axe = -1, *rank = -1;
    if (print_mes) RS_TRY(mes_text(w, RS_TX_wc_noaxe, 0));
    return 0;
}

/* get_logs (1921-1940): one roll (key 2); a tree with respawnrate 0 draws its respawn (key 1) and always depletes
   (random(0, 3) is 0); the deplete roll (key 3) is made on every success. Never suspends: a plain call. */
static int get_logs(World *w, RsRef ref, const RsTreeRow *row, int rank) {
    if (!(row->axe_mask >> rank & 1u))
        return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "get_logs: an axe the tree's successchance lacks (a KeyError)");
    int32_t low = row->low[rank], high = row->high[rank];
    int64_t respawn = row->respawnrate, deplete = 8;
    if (respawn == 0) {
        respawn = 120 + rs_random(w, 80, RS_KEYS(1));
        deplete = 0;
    }
    int32_t product = row->product;
    if (product == RS_NONE) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "get_logs: a product the archive lacks (a KeyError)");
    if (rs_stat_random(w, RS_SKILL_WOODCUTTING, low, high, RS_KEYS(2))) {
        int32_t tree_type = w->locs[ref.slot].type;   /* the xp's src: the tree's loc type, before it depletes */
        RS_TRY(mes_text(w, RS_TX_wc_get, product));
        RS_TRY(rs_add_xp(w, RS_SKILL_WOODCUTTING, row->productexp, RS_XPC_PRODUCE, product, tree_type));
        int64_t added;
        RS_TRY(rs_produce(w, product, K(w)->s_produce_get_logs, &added));
        if (rs_random(w, deplete, RS_KEYS(3)) == 0) {
            respawn = rs_floordiv(respawn * (4000 - 1), 4000);   /* ~scale_by_playercount, one player */
            Loc *l;
            const RsLocType *lt;
            RS_TRY(loc_of(w, ref, &l, &lt));
            if (lt->next_loc_stage == RS_NONE)
                return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "get_logs: a tree without next_loc_stage (a KeyError)");
            return rs_change_loc(w, ref.slot, lt->next_loc_stage, l->shape, l->angle, (int32_t)respawn);
        }
    }
    return rs_p_oploc(w, ref, 3);
}

/* attempt_cut_tree, [oploc1,_tree] (1871-1898): level, space, afk, then the action_delay cadence; the first click on a
   tree of level 0 skips the axe check. Its one site (yield from get_logs) never suspends. */
static int s_attempt_cut_tree(World *w, Frame *f, int d, RsSusp *out) {
    if (f->pc[d]) return bad_pc(w);
    Player *p = &w->p;
    RsRef ref = f->ref[d];
    Loc *l;
    const RsLocType *lt;
    const RsTreeRow *row;
    RS_TRY(loc_of(w, ref, &l, &lt));
    RS_TRY(tree_row(w, lt, &row));
    int32_t level = row->levelrequired, axe;
    int rank;
    if (p->levels[RS_SKILL_WOODCUTTING] < level) {
        RS_TRY(mes_text(w, RS_TX_wc_level, level));
        return done(out);
    }
    if (rs_inv_freespace(p->inv, RS_INV_CAP) < 1) {
        RS_TRY(mes_text(w, RS_TX_wc_full, 0));
        return done(out);
    }
    if (p->afk_ready) {
        p->afk_ready = 0;
        return RS_FAIL0(&w->err, RS_E_AFK_WOODCUTTING);
    }
    if (p->varp[RS_VARP_action_delay] < w->tick) {
        if (level > 0) {
            RS_TRY(axe_checker(w, 1, &axe, &rank));
            if (axe < 0) return done(out);
        }
        p->varp[RS_VARP_action_delay] = w->tick + 3;
        RS_TRY(rs_p_oploc(w, ref, 1));
        return done(out);
    }
    RS_TRY(axe_checker(w, 1, &axe, &rank));
    if (axe < 0) return done(out);
    RS_TRY(mes_text(w, RS_TX_wc_swing, 0));
    if (p->varp[RS_VARP_action_delay] == w->tick) {
        RS_TRY(get_logs(w, ref, row, rank));
        return done(out);
    }
    RS_TRY(rs_p_oploc(w, ref, 3));
    return done(out);
}

/* cut_tree, [oploc3,_tree] (1901-1918): axe, level, space, then the cadence. */
static int s_cut_tree(World *w, Frame *f, int d, RsSusp *out) {
    if (f->pc[d]) return bad_pc(w);
    Player *p = &w->p;
    RsRef ref = f->ref[d];
    int32_t axe;
    int rank;
    RS_TRY(axe_checker(w, 1, &axe, &rank));
    if (axe < 0) return done(out);
    Loc *l;
    const RsLocType *lt;
    const RsTreeRow *row;
    RS_TRY(loc_of(w, ref, &l, &lt));
    RS_TRY(tree_row(w, lt, &row));
    if (p->levels[RS_SKILL_WOODCUTTING] < row->levelrequired) {
        RS_TRY(mes_text(w, RS_TX_wc_level, row->levelrequired));
        return done(out);
    }
    if (rs_inv_freespace(p->inv, RS_INV_CAP) < 1) {
        RS_TRY(mes_text(w, RS_TX_wc_full, 0));
        return done(out);
    }
    if (p->varp[RS_VARP_action_delay] < w->tick) {
        p->varp[RS_VARP_action_delay] = w->tick + 3;
    } else if (p->varp[RS_VARP_action_delay] == w->tick) {
        RS_TRY(get_logs(w, ref, row, rank));
        return done(out);
    }
    RS_TRY(rs_p_oploc(w, ref, 3));
    return done(out);
}

/* use_on_tree, [oplocu,_tree] (1943-1950): an item whose debugname is an axe's starts op1. */
static int s_use_on_tree(World *w, Frame *f, int d, RsSusp *out) {
    if (f->pc[d]) return bad_pc(w);
    const RsObjType *used;
    RS_TRY(obj_of(w, w->p.last_useitem, &used));
    int axe = 0;
    for (int a = 0; a < 7 && !axe; a++) {
        const RsObjType *t = rs_obj_type(w->P, K(w)->axes[a]);
        axe = t && t->debugname == used->debugname;
    }
    if (axe) RS_TRY(rs_p_oploc(w, f->ref[d], 1));
    else RS_TRY(mes_text(w, RS_TX_dm_default, 0));
    return done(out);
}

/* ---- mining (engine.py) -------------------------------------------------------------------------------- */
/* pickaxe_checker (1964-1973): the first compact pickaxe the current Mining level may use, worn or carried. */
static int pickaxe_checker(World *w, int32_t *pickaxe) {
    int32_t rh = rhand(w);
    *pickaxe = -1;
    for (int i = 0; i < K(w)->n_pickaxes; i++) {
        int32_t obj = K(w)->pickaxes[i];
        if (obj < 0) continue;
        const RsObjType *ot;
        RS_TRY(obj_of(w, obj, &ot));
        if (ot->levelrequire == RS_NONE)
            return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "pickaxe_checker: no levelrequire param (a KeyError)");
        if (w->p.levels[RS_SKILL_MINING] >= ot->levelrequire && (rh == obj || rs_inv_total(w->p.inv, RS_INV_CAP, obj) > 0)) {
            *pickaxe = obj;
            return 0;
        }
    }
    return 0;
}

/* The part of get_ore's success after its p_delay (2046-2051): deplete, produce, xp, message. */
static int ore_won(World *w, Frame *f, int d, int fast) {
    int32_t r = f->arg[d][0];
    const RsRockRow *row = &w->P->rocks[r];
    int64_t respawn = rs_floordiv((int64_t)row->respawn * (4000 - 1), 4000);   /* ~scale_by_playercount */
    Loc *l;
    const RsLocType *lt;
    RS_TRY(loc_of(w, f->ref[d], &l, &lt));
    if (lt->next_loc_stage_mining == RS_NONE)
        return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "get_ore: next_loc_stage_mining is None (a TypeError)");
    RS_TRY(rs_change_loc(w, f->ref[d].slot, lt->next_loc_stage_mining, l->shape, l->angle, (int32_t)respawn));
    int64_t added;
    RS_TRY(rs_produce(w, row->output, fast ? K(w)->s_produce_ore_fast : K(w)->s_produce_ore_normal, &added));
    RS_TRY(rs_add_xp(w, RS_SKILL_MINING, row->exp, RS_XPC_PRODUCE, row->output, f->arg[d][1]));
    return mes_text(w, RS_TX_mine_get, r);
}

/* get_ore(fast) (2022-2053), level d with ref the rock and arg[0] its ROCKS row: a gem roll (key 4 == 1, then key 6),
   else the Mining roll (key 5), whose success on a normal rock waits a p_delay(0) (site 1). */
static int s_get_ore(World *w, Frame *f, int d, RsSusp *out, int fast) {
    if (f->pc[d] == 1) {
        RS_TRY(ore_won(w, f, d, fast));
        return done(out);
    }
    if (f->pc[d]) return bad_pc(w);
    const RsPack *P = w->P;
    const RsRockRow *row = &P->rocks[f->arg[d][0]];
    f->arg[d][1] = w->locs[f->ref[d].slot].type;   /* rock = loc.type: the xp's src, read before the loc changes */
    int32_t name = fast ? K(w)->s_produce_ore_fast : K(w)->s_produce_ore_normal;
    const Slot *neck = &w->p.worn[RS_WEARPOS_FRONT];
    if (neck->id >= 0) {
        const RsObjType *t;
        RS_TRY(obj_of(w, neck->id, &t));
        if ((t->flags & RS_OF_GLORY) && w->members) return RS_FAIL0(&w->err, RS_E_GLORY);
    }
    if (row->output == RS_NONE) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "get_ore: an ore the archive lacks (a KeyError)");
    if (rs_random(w, K(w)->gem_chance, RS_KEYS(4)) == 1) {   /* random($chance) = ^true: a gem (or nothing) */
        int64_t roll = rs_random(w, K(w)->gem_roll, RS_KEYS(6));
        int32_t gem = -1;
        for (int g = 0; g < P->n_gems && gem < 0; g++)
            if (roll < P->gems[g].below) gem = P->gems[g].gem;
        if (gem >= 0) {
            int64_t added;
            RS_TRY(rs_produce(w, gem, name, &added));
            RS_TRY(mes_text(w, RS_TX_mine_gem, gem));
        }
    } else if (rs_stat_random(w, RS_SKILL_MINING, row->low, row->high, RS_KEYS(5))) {
        if (!fast) return delay_at(f, d, 1, 0, out);
        RS_TRY(ore_won(w, f, d, fast));
        return done(out);
    }
    RS_TRY(rs_p_oploc(w, f->ref[d], 3));
    return done(out);
}

/* mining(get_ore(fast), first).script (1976-2019): the refusals in the script's order, then the cadence. Sites:
   1-3 the mine_full, mine_level and mine_nopick mesboxes, 4 the first swing's get_ore, 5 the continued one's. */
static int s_mining(World *w, Frame *f, int d, RsSusp *out, int first, int fast) {
    switch (f->pc[d]) {
    case 0: break;
    case 1: case 2: case 3: case 4: case 5: return resume_tail(w, f, d, f->pc[d], out);
    default: return bad_pc(w);
    }
    Player *p = &w->p;
    RsRef ref = f->ref[d];
    Loc *l;
    const RsLocType *lt;
    RS_TRY(loc_of(w, ref, &l, &lt));
    if (first && lt->mining_rock_empty == 1) {
        if (lt->is_loc_4030) return RS_FAIL0(&w->err, RS_E_EMPTY_LOC_4030);
        RS_TRY(mes_text(w, RS_TX_mine_none, 0));
        return done(out);
    }
    if (lt->rock_row < 0) {
        RS_TRY(mes_text(w, RS_TX_dm_default, 0));
        return done(out);
    }
    int32_t r = lt->rock_row, msg;
    const RsRockRow *row = &w->P->rocks[r];
    if (rs_inv_freespace(p->inv, RS_INV_CAP) < 1) {
        RS_TRY(rs_text_msg(w, RS_TX_mine_full, r, &msg));
        return tail_call(w, f, d, 1, RS_S_MESBOX, (RsRef){0}, msg, out);
    }
    if (p->levels[RS_SKILL_MINING] < row->level) {
        RS_TRY(rs_text_msg(w, RS_TX_mine_level, row->level, &msg));
        return tail_call(w, f, d, 2, RS_S_MESBOX, (RsRef){0}, msg, out);
    }
    if (first && p->afk_ready) {
        p->afk_ready = 0;
        return RS_FAIL0(&w->err, RS_E_AFK_MINING);
    }
    int32_t pickaxe;
    RS_TRY(pickaxe_checker(w, &pickaxe));
    if (pickaxe < 0) {
        RS_TRY(rs_text_msg(w, RS_TX_mine_nopick, 0, &msg));
        return tail_call(w, f, d, 3, RS_S_MESBOX, (RsRef){0}, msg, out);
    }
    const RsObjType *pt;
    RS_TRY(obj_of(w, pickaxe, &pt));
    if (pt->mining_rate == RS_NONE) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a pickaxe without mining_rate (a KeyError)");
    int32_t rate = pt->mining_rate;
    int16_t ore = fast ? RS_S_GET_ORE_FAST : RS_S_GET_ORE_NORMAL;
    if (first) {
        if (p->varp[RS_VARP_action_delay] < w->tick) {
            p->varp[RS_VARP_action_delay] = w->tick + rate;
            RS_TRY(rs_p_oploc(w, ref, 1));
            return done(out);
        }
        RS_TRY(mes_text(w, RS_TX_mine_swing, 0));
        if (p->varp[RS_VARP_action_delay] == w->tick) return tail_call(w, f, d, 4, ore, ref, r, out);
    } else if (p->varp[RS_VARP_action_delay] < w->tick) {
        p->varp[RS_VARP_action_delay] = w->tick + rate;
    } else if (p->varp[RS_VARP_action_delay] == w->tick) {
        return tail_call(w, f, d, 5, ore, ref, r, out);
    }
    RS_TRY(rs_p_oploc(w, ref, 3));
    return done(out);
}

/* prospect, [oploc2,_mining_rock_*] (2061-2074): the message, p_delay(0) (site 1) and p_delay(1) (site 2), then what
   the rock holds when the second delay ends. */
static int s_prospect(World *w, Frame *f, int d, RsSusp *out) {
    switch (f->pc[d]) {
    case 0:
        RS_TRY(mes_text(w, RS_TX_prospect_start, 0));
        return delay_at(f, d, 1, 0, out);
    case 1:
        return delay_at(f, d, 2, 1, out);
    case 2: break;
    default: return bad_pc(w);
    }
    Loc *l;
    const RsLocType *lt;
    RS_TRY(loc_of(w, f->ref[d], &l, &lt));
    if (lt->mining_rock_empty == 1) RS_TRY(mes_text(w, RS_TX_prospect_empty, 0));
    else if (lt->rock_row < 0) RS_TRY(mes_text(w, RS_TX_dm_default, 0));
    else RS_TRY(mes_text(w, RS_TX_prospect_ore, lt->rock_row));
    return done(out);
}

/* ---- pen gates (engine.py) ----------------------------------------------------------------------------- */
/* _gate_close_offset (2101-2103) */
static void gate_close_offset(int32_t angle, int32_t mod, int32_t *x, int32_t *z) {
    static const int8_t D[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    *x = D[angle][0] * mod, *z = D[angle][1] * mod;
}

/* _gate_open_offset (2106-2111) */
static void gate_open_offset(int32_t angle, int outer, int32_t *x, int32_t *z) {
    static const int8_t D[4][2] = {{0, -1}, {-1, 0}, {0, 1}, {1, 0}};
    static const int8_t OUTER[4][2] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};
    *x = D[angle][0], *z = D[angle][1];
    if (outer) *x += OUTER[angle][0], *z += OUTER[angle][1];
}

/* _gate_pair (2114-2117): the tile whose zone holds the other gate. */
static void gate_pair(int32_t x, int32_t z, int32_t angle, int outer, int32_t *px, int32_t *pz) {
    int32_t s = outer ? -1 : 1;
    static const int8_t D[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    *px = x + D[angle][0] * s, *pz = z + D[angle][1] * s;
}

/* _gate_partner (2120-2127): the first active loc of the pair tile's zone (last first) within a tile of the gate in
   the category; -1 none. */
static int gate_partner(World *w, int slot, int outer, int32_t category, int *found) {
    const Loc *l = &w->locs[slot];
    if (l->angle > 3) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a gate angle outside 0..3 (an IndexError)");
    int32_t px, pz;
    gate_pair(l->x, l->z, l->angle, outer, &px, &pz);
    uint8_t zone[RS_ZONE_CAP];
    int n = rs_locs_in_zone(w, l->level, px, pz, zone);
    *found = -1;
    for (int i = 0; i < n; i++) {
        const Loc *o = &w->locs[zone[i]];
        int32_t dx = o->x - l->x, dz = o->z - l->z;
        if (dx < 0) dx = -dx;
        if (dz < 0) dz = -dz;
        const RsLocType *ot = rs_loc_type(w->P, o->type);
        if (!ot) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a loc of a type the pack lacks");
        if ((dx > dz ? dx : dz) <= 1 && ot->category == category) {
            *found = zone[i];
            return 0;
        }
    }
    return 0;
}

/* _swing_gate (2130-2144): the main gate and its outer half each go for LOC_DURATION, their next stages standing
   beside, turned a quarter. The partner search runs after the main gate's removal and its new stage. */
static int swing_gate(World *w, int slot, int opening) {
    Loc *l = &w->locs[slot];
    if (l->angle > 3) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a gate angle outside 0..3 (an IndexError)");
    int32_t angle = l->angle, x = l->x, z = l->z, turn = opening ? 3 : 1, mx, mz, ox, oz, stage;
    if (opening) {
        gate_close_offset(angle, 1, &mx, &mz);
        gate_close_offset(angle, 2, &ox, &oz);
    } else {
        gate_open_offset(angle, 0, &mx, &mz);
        gate_open_offset(angle, 1, &ox, &oz);
    }
    RS_TRY(rs_loc_next_stage(w, l, &stage));
    RS_TRY(rs_remove_loc(w, slot, RS_LOC_DURATION));
    int added, partner;
    RS_TRY(rs_loc_add(w, l->level, x + mx, z + mz, stage, (angle + turn) % 4, RS_WALL_STRAIGHT, RS_LOC_DURATION, &added));
    const RsConsts *k = K(w);
    RS_TRY(gate_partner(w, slot, 0, opening ? k->cat_gate_outer_closed : k->cat_gate_outer_open, &partner));
    if (partner < 0) return 0;
    RS_TRY(rs_loc_next_stage(w, &w->locs[partner], &stage));
    RS_TRY(rs_remove_loc(w, partner, RS_LOC_DURATION));
    return rs_loc_add(w, l->level, x + ox, z + oz, stage, (angle + turn) % 4, RS_WALL_STRAIGHT, RS_LOC_DURATION, &added);
}

/* gate(opening, outer).script (2147-2158): a main gate swings itself; an outer half swings the main gate beside it. */
static int s_gate(World *w, Frame *f, int d, RsSusp *out, int opening, int outer) {
    if (f->pc[d]) return bad_pc(w);
    Loc *l;
    const RsLocType *lt;
    RS_TRY(loc_of(w, f->ref[d], &l, &lt));
    int slot = f->ref[d].slot;
    if (!outer) {
        RS_TRY(swing_gate(w, slot, opening));
    } else {
        int main;
        const RsConsts *k = K(w);
        RS_TRY(gate_partner(w, slot, 1, opening ? k->cat_gate_main_closed : k->cat_gate_main_open, &main));
        if (main >= 0) RS_TRY(swing_gate(w, main, opening));
    }
    return done(out);
}

/* door(offsets, turn).script (engine.py:2183-2197): [oploc1,_door_closed] (DOOR_OPEN, turn 1) and [oploc1,_door_opened]
   (DOOR_CLOSE, turn 3): a shape other than a straight or diagonal wall is not supported; the stage stands on the tile at
   the offset of the door's shape and angle (rs_gen.h RS_DOOR_*), where a diagonal door with the player on it is not
   supported; then loc_param(next_loc_stage), loc_del(500) and loc_add of the stage turned a quarter, for 500 ticks. No
   suspension. The level, angle and shape are the loc's before its removal, which changes none of them. */
static int s_door(World *w, Frame *f, int d, RsSusp *out, int opening) {
    if (f->pc[d]) return bad_pc(w);
    Loc *l;
    const RsLocType *lt;
    RS_TRY(loc_of(w, f->ref[d], &l, &lt));
    int k = l->shape == RS_WALL_STRAIGHT ? 0 : l->shape == RS_WALL_DIAGONAL ? 1 : -1;
    if (k < 0) return RS_FAIL(&w->err, RS_E_DOOR_SHAPE, l->shape);
    if (l->angle > 3) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a door angle outside 0..3 (an IndexError)");
    int i = k * 4 + l->angle;
    int32_t x = l->x + (opening ? RS_DOOR_OPEN_DX : RS_DOOR_CLOSE_DX)[i];
    int32_t z = l->z + (opening ? RS_DOOR_OPEN_DZ : RS_DOOR_CLOSE_DZ)[i];
    if (w->p.x == x && w->p.z == z && l->shape == RS_WALL_DIAGONAL) return RS_FAIL0(&w->err, RS_E_DOOR_DIAGONAL);
    int32_t stage, level = l->level, angle = (l->angle + (opening ? 1 : 3)) % 4, shape = l->shape, added;
    RS_TRY(rs_loc_next_stage(w, l, &stage));
    RS_TRY(rs_remove_loc(w, f->ref[d].slot, RS_LOC_DURATION));
    RS_TRY(rs_loc_add(w, level, x, z, stage, angle, shape, RS_LOC_DURATION, &added));
    return done(out);
}

/* ---- field plants (engine.py:3205-3237) -------------------------------------------------------------------------- */
/* pick, [oploc2,cabbage|potato|onion]: the plant type's picks row (read at the start, as Python's name and row; a
   KeyError without one); after a step this cycle p_arrivedelay (site 1); with no free slot the full message, p_teleport
   onto the plant and p_delay(0) (site 2), then the end; else p_teleport and p_delay(0) (site 3), then loc_del for the
   respawn (~scale_by_playercount where the row says so), the overflow refusal, the item (a produce of
   [proc,pickup_loc_floor], which owes no XP) and the picked message. No draws. arg[2]: the plant's type. */
static int s_pick(World *w, Frame *f, int d, RsSusp *out) {
    Player *p = &w->p;
    Loc *l;
    const RsLocType *lt;
    RS_TRY(loc_of(w, f->ref[d], &l, &lt));
    switch (f->pc[d]) {
    case 0:
        if (!lt->has_pick) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "pick of a loc type with no picks row (a KeyError)");
        f->arg[d][2] = l->type;
        if (p->last_movement >= w->tick) return delay_at(f, d, 1, 0, out);   /* p_arrivedelay */
        break;
    case 1: break;
    case 2: return done(out);
    case 3: {
        const RsLocType *row = rs_loc_type(w->P, f->arg[d][2]);
        if (!row || !row->has_pick) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "pick of a loc type with no picks row");
        int64_t respawn = row->pick_respawn;
        if (row->pick_scaled)   /* scale(sub(4000, min(playercount, 2000)), 4000, $base) */
            respawn = rs_floordiv(respawn * (4000 - (RS_PLAYER_COUNT < 2000 ? RS_PLAYER_COUNT : 2000)), 4000);
        RS_TRY(rs_remove_loc(w, f->ref[d].slot, (int32_t)respawn));
        if (rs_inv_freespace(p->inv, RS_INV_CAP) == 0) return RS_FAIL0(&w->err, RS_E_PICK_OVERFLOW);
        if (row->pick_item == RS_NONE) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a picks row item the archive lacks (a KeyError)");
        int64_t added;
        RS_TRY(rs_produce(w, row->pick_item, K(w)->s_produce_pick, &added));
        RS_TRY(mes_text(w, RS_TX_pick, f->arg[d][2]));
        return done(out);
    }
    default: return bad_pc(w);
    }
    if (rs_inv_freespace(p->inv, RS_INV_CAP) == 0) {
        RS_TRY(mes_text(w, RS_TX_pick_full, f->arg[d][2]));
        RS_TRY(rs_teleport(w, l->x, l->z));
        return delay_at(f, d, 2, 0, out);
    }
    RS_TRY(rs_teleport(w, l->x, l->z));
    return delay_at(f, d, 3, 0, out);
}

/* ---- net fishing (engine.py:2175-2256) --------------------------------------------------------------------------- */
enum { EQUIP_NET = 0, EQUIP_FISHING_ROD };

/* check_fish_equipment(equipment) (2175-2184), arg[0] EQUIP_*: the tool's fail mesbox (site 1), the rod's bait one
   (site 2); Frame.ret its bool. */
static int s_check_fish_equipment(World *w, Frame *f, int d, RsSusp *out) {
    int fin;
    switch (f->pc[d]) {
    case 0: break;
    case 1: case 2:
        RS_TRY(sub(w, f, d, f->pc[d], out, &fin));
        if (fin) f->ret = 0;
        return fin ? done(out) : 0;
    default: return bad_pc(w);
    }
    const RsConsts *k = K(w);
    int rod = f->arg[d][0] == EQUIP_FISHING_ROD;
    int32_t tool = rod ? k->obj_fishing_rod : k->obj_net, msg;
    if (tool < 0) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "check_fish_equipment: a tool the archive lacks (a KeyError)");
    if (rs_inv_total(w->p.inv, RS_INV_CAP, tool) < 1) {
        RS_TRY(rs_text_msg(w, rod ? RS_TX_fish_fail_fishing_rod : RS_TX_fish_fail_net, 0, &msg));
        RS_TRY(push(w, f, d, RS_S_MESBOX, (RsRef){0}, msg));
        RS_TRY(sub(w, f, d, 1, out, &fin));
        if (fin) f->ret = 0;
        return fin ? done(out) : 0;
    }
    if (rod) {
        if (k->obj_fishing_bait < 0) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "check_fish_equipment: no fishing_bait (a KeyError)");
        if (rs_inv_total(w->p.inv, RS_INV_CAP, k->obj_fishing_bait) < 1) {
            RS_TRY(rs_text_msg(w, RS_TX_fish_no_bait, 0, &msg));
            RS_TRY(push(w, f, d, RS_S_MESBOX, (RsRef){0}, msg));
            RS_TRY(sub(w, f, d, 2, out, &fin));
            if (fin) f->ret = 0;
            return fin ? done(out) : 0;
        }
    }
    f->ret = 1;
    return done(out);
}

/* fish_roll(fish1, fish2) (2222-2233): anchovies first (key 10) when level >= 15, then shrimp (key 11). */
static int fish_roll(World *w, int anchovies) {
    for (int k = 0; k < 2; k++) {
        int r = k == 0 ? (anchovies ? 1 : -1) : 0;   /* FISH rows: 0 raw_shrimp, 1 raw_anchovies */
        if (r < 0) continue;
        const RsFishRow *s = &w->P->fish[r];
        if (rs_stat_random(w, RS_SKILL_FISHING, s->success_low, s->success_high, RS_KEYS(10 + k))) {
            int64_t added;
            RS_TRY(rs_produce(w, s->obj, K(w)->s_produce_fish_roll, &added));
            RS_TRY(rs_add_xp(w, RS_SKILL_FISHING, s->productexp, RS_XPC_PRODUCE, s->obj, -1));
            return rs_mes(w, s->message);
        }
    }
    return 0;
}

/* The net and space checks both fishing ops open with: site 1 the net check, site 2 the fish_full mesbox. *go is 1
   when the op body runs now (the checks passed this step). */
static int fish_checks(World *w, Frame *f, int d, RsSusp *out, int *go) {
    int fin;
    *go = 0;
    switch (f->pc[d]) {
    case 0: RS_TRY(push(w, f, d, RS_S_CHECK_FISH_EQUIPMENT, (RsRef){0}, EQUIP_NET)); break;
    case 1: break;
    case 2: return resume_tail(w, f, d, 2, out);
    default: return bad_pc(w);
    }
    RS_TRY(sub(w, f, d, 1, out, &fin));
    if (!fin) return 0;
    if (!f->ret) return done(out);
    if (rs_inv_freespace(w->p.inv, RS_INV_CAP) < 1) {
        int32_t msg;
        RS_TRY(rs_text_msg(w, RS_TX_fish_full, 0, &msg));
        return tail_call(w, f, d, 2, RS_S_MESBOX, (RsRef){0}, msg, out);
    }
    *go = 1;
    return 0;
}

/* attempt_fish_shrimp, [opnpc1,_saltfish] (2187-2205). */
static int s_attempt_fish_shrimp(World *w, Frame *f, int d, RsSusp *out) {
    int go;
    RS_TRY(fish_checks(w, f, d, out, &go));
    if (!go) return 0;
    Player *p = &w->p;
    RsRef npc = f->ref[d];
    if (p->varp[RS_VARP_action_delay] < w->tick) {
        p->varp[RS_VARP_action_delay] = w->tick + 4;
        RS_TRY(rs_p_opnpc(w, npc, 1));
        return done(out);
    }
    if (p->varp[RS_VARP_action_delay] == w->tick + 3) {
        RS_TRY(mes_text(w, RS_TX_fish_cast, 0));
        RS_TRY(rs_p_opnpc(w, npc, 4));
    } else if (p->varp[RS_VARP_action_delay] == w->tick) {
        RS_TRY(fish_roll(w, p->levels[RS_SKILL_FISHING] >= 15));
    }
    if (p->afk_ready) {
        p->afk_ready = 0;
        return RS_FAIL0(&w->err, RS_E_AFK_FISHING);
    }
    return done(out);
}

/* fish_shrimp, [opnpc4,_saltfish] (2208-2219). */
static int s_fish_shrimp(World *w, Frame *f, int d, RsSusp *out) {
    int go;
    RS_TRY(fish_checks(w, f, d, out, &go));
    if (!go) return 0;
    Player *p = &w->p;
    if (p->varp[RS_VARP_action_delay] < w->tick) p->varp[RS_VARP_action_delay] = w->tick + 5;
    else if (p->varp[RS_VARP_action_delay] == w->tick) RS_TRY(fish_roll(w, p->levels[RS_SKILL_FISHING] >= 15));
    RS_TRY(rs_p_opnpc(w, f->ref[d], 4));
    return done(out);
}

/* attempt_fish_sardine, [opnpc3,_saltfish] (2236-2244): site 1 the level mesbox, site 2 the rod check; with a rod
   and bait it is not supported. */
static int s_attempt_fish_sardine(World *w, Frame *f, int d, RsSusp *out) {
    int fin;
    switch (f->pc[d]) {
    case 0:
        if (w->p.levels[RS_SKILL_FISHING] < 5) {
            int32_t msg;
            RS_TRY(rs_text_msg(w, RS_TX_fish_level5, 0, &msg));
            return tail_call(w, f, d, 1, RS_S_MESBOX, (RsRef){0}, msg, out);
        }
        RS_TRY(push(w, f, d, RS_S_CHECK_FISH_EQUIPMENT, (RsRef){0}, EQUIP_FISHING_ROD));
        break;
    case 1: return resume_tail(w, f, d, 1, out);
    case 2: break;
    default: return bad_pc(w);
    }
    RS_TRY(sub(w, f, d, 2, out, &fin));
    if (!fin) return 0;
    if (!f->ret) return done(out);
    return RS_FAIL0(&w->err, RS_E_BAIT_FISHING);
}

/* use_on_saltfish, [opnpcu,_saltfish] (2247-2256): a net fishes (site 1); other fishing equipment is not supported. */
static int s_use_on_saltfish(World *w, Frame *f, int d, RsSusp *out) {
    if (f->pc[d] == 1) return resume_tail(w, f, d, 1, out);
    if (f->pc[d]) return bad_pc(w);
    const RsObjType *used;
    RS_TRY(obj_of(w, w->p.last_useitem, &used));
    if (used->flags & RS_OF_NET) return tail_call(w, f, d, 1, RS_S_ATTEMPT_FISH_SHRIMP, f->ref[d], 0, out);
    if (used->flags & RS_OF_FISH_EQUIPMENT) return RS_FAIL(&w->err, RS_E_FISHING_EQUIPMENT, used->debugname);
    RS_TRY(mes_text(w, RS_TX_dm_default, 0));
    return done(out);
}

/* ---- held items (engine.py) ---------------------------------------------------------------------------- */
/* wearpos (2947-2949): the obj's wear positions that are set. */
static int wearpos(const RsObjType *t, int16_t out[3]) {
    int n = 0;
    for (int i = 0; i < 3; i++)
        if (t->wearpos[i] >= 0) out[n++] = t->wearpos[i];
    return n;
}

static int wear_overlap(const RsObjType *a, const RsObjType *b) {
    int16_t pa[3], pb[3];
    int na = wearpos(a, pa), nb = wearpos(b, pb);
    for (int i = 0; i < na; i++)
        for (int j = 0; j < nb; j++)
            if (pa[i] == pb[j]) return 1;
    return 0;
}

/* try_equip (2952-2989): *ok its bool. */
static int try_equip(World *w, int32_t slot, int *ok) {
    const RsPack *P = w->P;
    Player *p = &w->p;
    Slot *item;
    RS_TRY(inv_slot(w, slot, &item));
    *ok = 0;
    if (item->id < 0) return 0;
    const RsObjType *ot;
    RS_TRY(obj_of(w, item->id, &ot));
    int16_t pos_list[3];
    if (!wearpos(ot, pos_list)) return 0;
    int32_t obj = item->id, count = item->count, pos = ot->wearpos[0];
    if (pos < 0 || pos >= RS_WORN_CAP)
        return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "try_equip: an obj without its first wearpos (a TypeError)");
    if (ot->stackable && rs_inv_total(p->worn, RS_WORN_CAP, obj) > 0) {
        int64_t overflow = rs_inv_itemspace2(P, p->worn, RS_WORN_CAP, 0, obj, count);
        if (overflow >= count) return 0;
        RS_TRY(rs_move_item(w, RS_LOCN_INV, RS_LOCN_WORN, obj, count - overflow));
        *ok = 1;
        return 0;
    }
    int32_t conflicts = 0;
    for (int i = 0; i < RS_WORN_CAP; i++) {
        if (p->worn[i].id < 0) continue;
        int32_t other = p->worn[i].id, number = p->worn[i].count;
        const RsObjType *t;
        RS_TRY(obj_of(w, other, &t));
        if (!wear_overlap(ot, t)) continue;
        if (t->stackable && rs_inv_total(p->inv, RS_INV_CAP, other) > 0) {
            if (rs_inv_itemspace2(P, p->inv, RS_INV_CAP, 0, other, number)) return 0;
        } else {
            conflicts += 1;
        }
    }
    if (conflicts > rs_inv_freespace(p->inv, RS_INV_CAP) + 1) return 0;
    Slot back = p->worn[pos];
    p->worn[pos] = (Slot){obj, count};
    *item = back.id >= 0 ? back : (Slot){-1, 0};
    RS_TRY(transfer_event(w, RS_LOCN_INV, RS_LOCN_WORN, obj, count));
    if (back.id >= 0) {
        RS_TRY(transfer_event(w, RS_LOCN_WORN, RS_LOCN_INV, back.id, back.count));
        *item = (Slot){-1, 0};
        if (rs_inv_add(P, p->inv, RS_INV_CAP, 0, back.id, back.count) < back.count)
            return RS_FAIL0(&w->err, RS_E_UNEQUIP_DROP);
    }
    Slot listed[RS_WORN_CAP];   /* p.worn.listed() once, before the moves */
    int n = 0;
    for (int i = 0; i < RS_WORN_CAP; i++)
        if (p->worn[i].id >= 0) listed[n++] = p->worn[i];
    for (int i = 0; i < n; i++) {
        const RsObjType *t;
        RS_TRY(obj_of(w, listed[i].id, &t));
        if (listed[i].id != obj && wear_overlap(ot, t))
            RS_TRY(rs_move_item(w, RS_LOCN_WORN, RS_LOCN_INV, listed[i].id, listed[i].count));
    }
    *ok = 1;
    return 0;
}

/* update_all (2992-2997): of its updates only ~player_combat_stat changes play. */
static int update_all(World *w) {
    for (int i = 0; i < RS_WORN_CAP; i++) {
        if (w->p.worn[i].id < 0) continue;
        const RsObjType *t;
        RS_TRY(obj_of(w, w->p.worn[i].id, &t));
        if (t->flags & RS_OF_WEIGHT) return RS_FAIL0(&w->err, RS_E_WEIGHT);
    }
    return rs_player_combat_stat(w);
}

/* equip (2935-2944) */
static int equip(World *w, int32_t slot) {
    Slot *item;
    RS_TRY(inv_slot(w, slot, &item));
    if (item->id >= 0) {
        const RsObjType *t;
        RS_TRY(obj_of(w, item->id, &t));
        if (t->flags & RS_OF_EQUIP_REFUSED) return RS_FAIL(&w->err, RS_E_EQUIPPING, t->debugname);
    }
    int ok;
    RS_TRY(try_equip(w, slot, &ok));
    if (!ok) return mes_text(w, RS_TX_equip_space, 0);
    return update_all(w);
}

/* wield, [opheld2,X] (2916-2932): arg[0] the WIELD_RULES row; the tutorial rule is @levelrequire_attack(1) past the
   tutorial. */
static int s_wield(World *w, Frame *f, int d, RsSusp *out) {
    if (f->pc[d]) return bad_pc(w);
    int32_t r = f->arg[d][0];
    if (r < 0 || r >= w->P->n_wield) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "wield of no WIELD_RULES row");
    const RsWieldRule *rule = &w->P->wield[r];
    Player *p = &w->p;
    int32_t kind = rule->kind;
    if (kind == RS_WIELD_TUTORIAL) {
        if (p->varp[RS_VARP_tutorial] <= K(w)->tutorial_equip) return RS_FAIL0(&w->err, RS_E_TUTORIAL_EQUIP);
        kind = RS_WIELD_LEVELREQUIRE;
    }
    if (kind == RS_WIELD_LEVELREQUIRE) {
        if (p->base_levels[rule->stat] < rule->level) return RS_FAIL(&w->err, RS_E_LEVELREQUIRE, rule->label);
    } else if (kind != RS_WIELD_EQUIP) {
        return RS_FAIL(&w->err, RS_E_WIELD_RULE, rule->label);
    }
    RS_TRY(equip(w, p->last_slot));
    return done(out);
}

/* consume_effect_apply (3020-3045) for food that only heals: the heals (cause eat), then the messages by the '0'
   sentinel and '' rules. */
static int consume_effect_apply(World *w, int32_t consumable) {
    const RsPack *P = w->P;
    Player *p = &w->p;
    const RsObjType *t;
    RS_TRY(obj_of(w, consumable, &t));
    int32_t hitpoints = p->levels[RS_SKILL_HITPOINTS];
    if (t->eat_row < 0) return 0;
    const RsEatRow *row = &P->eat_rows[t->eat_row];
    if (row->changes) return RS_FAIL0(&w->err, RS_E_FOOD_STATS);
    for (int h = 0; h < row->n_heal; h++)
        RS_TRY(rs_stat_heal(w, row->heal_stat[h], row->heal_constant[h], row->heal_percent[h], RS_HP_EAT));
    if (t->flags & RS_OF_POTION) return RS_FAIL0(&w->err, RS_E_POTION);
    if (row->message_delay) return RS_FAIL0(&w->err, RS_E_FOOD_MESSAGE_DELAY);
    int32_t c1 = row->c1, c2 = row->c2, r1 = row->r1, r2 = row->r2;
    int healed = hitpoints < p->levels[RS_SKILL_HITPOINTS];
    if (r1 != RS_EAT_ZERO && healed && r1 != RS_EAT_EMPTY) {
        RS_TRY(rs_mes(w, r1));
    } else if (c1 != RS_EAT_ZERO) {
        if (c1 != RS_EAT_EMPTY) RS_TRY(rs_mes(w, c1));
        else RS_TRY(mes_text(w, RS_TX_eat, consumable));
    }
    if (r2 != RS_EAT_ZERO && healed) {
        if (r2 != RS_EAT_EMPTY) RS_TRY(rs_mes(w, r2));
        else RS_TRY(mes_text(w, RS_TX_eat_heal, 0));
    }
    if (c2 != RS_EAT_ZERO && c2 != RS_EAT_EMPTY && (!healed || r2 == RS_EAT_ZERO)) RS_TRY(rs_mes(w, c2));
    return 0;
}

/* eat, [opheld1,...] (3000-3017): arg[0] the EAT_TRIGGERS row. No food before %eat_delay passes; skill_delay is added
   to action_delay even when stale; the slot is deleted before the effect. */
static int s_eat(World *w, Frame *f, int d, RsSusp *out) {
    if (f->pc[d]) return bad_pc(w);
    int32_t r = f->arg[d][0];
    if (r < 0 || r >= w->P->n_eat_triggers) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "eat of no EAT_TRIGGERS row");
    const RsEatTrigger *tr = &w->P->eat_triggers[r];
    Player *p = &w->p;
    if (tr->eat_delay != RS_NONE) {
        if (p->varp[RS_VARP_eat_delay] >= w->tick) return done(out);
        p->varp[RS_VARP_eat_delay] = w->tick + tr->eat_delay;
    }
    if (tr->skill_delay != RS_NONE) p->varp[RS_VARP_action_delay] += tr->skill_delay;
    if (!tr->effect_stat) return RS_FAIL(&w->err, RS_E_FOOD_EFFECT, tr->effect);
    int32_t consumable = p->last_item, obj, count;
    RS_TRY(rs_inv_delslot(w, p->last_slot, RS_WHY_EAT, &obj, &count));
    const RsObjType *t;
    RS_TRY(obj_of(w, consumable, &t));
    if (t->next_obj_stage != -1) return RS_FAIL0(&w->err, RS_E_FOOD_NEXT_STAGE);
    RS_TRY(consume_effect_apply(w, consumable));
    return done(out);
}

/* bury_bones, [opheld1,_bones] (3048-3059): dig, stopAction, p_delay(0) (site 1), then the slot noted at the start is
   deleted, its bone_exp added. arg[3] the slot. */
static int s_bury_bones(World *w, Frame *f, int d, RsSusp *out) {
    Player *p = &w->p;
    if (f->pc[d] == 0) {
        f->arg[d][3] = p->last_slot;
        RS_TRY(mes_text(w, RS_TX_bury_dig, 0));
        rs_stop_action(w);
        return delay_at(f, d, 1, 0, out);
    }
    if (f->pc[d] != 1) return bad_pc(w);
    int32_t obj, count;
    RS_TRY(rs_inv_delslot(w, f->arg[d][3], RS_WHY_BURY, &obj, &count));
    const RsObjType *t;
    RS_TRY(obj_of(w, obj, &t));
    if (t->bone_exp == RS_NONE) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "bury_bones: no bone_exp param (a KeyError)");
    RS_TRY(rs_add_xp(w, RS_SKILL_PRAYER, t->bone_exp, RS_XPC_BURY, obj, 0));
    RS_TRY(mes_text(w, RS_TX_bury_done, 0));
    if (p->afk_ready) {
        p->afk_ready = 0;
        return RS_FAIL0(&w->err, RS_E_AFK_BURY);
    }
    return done(out);
}

/* drop_slot, [opheld5,_] (3062-3067): inv_dropslot for 1.5 lootdrop durations. */
static int s_drop_slot(World *w, Frame *f, int d, RsSusp *out) {
    if (f->pc[d]) return bad_pc(w);
    Player *p = &w->p;
    int32_t obj, count;
    RS_TRY(rs_inv_delslot(w, p->last_slot, RS_WHY_DROP, &obj, &count));
    RS_TRY(rs_obj_add(w, p->level, p->x, p->z, obj, count, (int32_t)rs_floordiv((int64_t)K(w)->lootdrop_duration * 3, 2)));
    return done(out);
}

/* use_default(cases, categories, oil).script (3070-3082): a case of the used item (or, for [opheldu,_category_22], an
   oil as the target) is not supported; otherwise the default message. arg[1] the trigger name. */
static int s_use_default(World *w, Frame *f, int d, RsSusp *out, uint32_t case_flag, int oil) {
    if (f->pc[d]) return bad_pc(w);
    const RsObjType *used;
    RS_TRY(obj_of(w, w->p.last_useitem, &used));
    int refused = (used->flags & case_flag) != 0;
    if (!refused && oil) {
        const RsObjType *target;
        RS_TRY(obj_of(w, w->p.last_item, &target));
        refused = (target->flags & RS_OF_OIL) != 0;
    }
    if (refused) return RS_FAIL(&w->err, RS_E_USE_DEFAULT_CASE, f->arg[d][1], used->debugname);
    RS_TRY(mes_text(w, RS_TX_dm_default, 0));
    return done(out);
}

/* ---- ground items (engine.py) -------------------------------------------------------------------------- */
/* pickup_obj, [opobj3,_]: refused when the inventory cannot hold it all; off the obj's tile a p_delay(0) (site 1) first;
   take_obj re-checks that the obj is still there. */
static int s_pickup_obj(World *w, Frame *f, int d, RsSusp *out) {
    RsRef ref = f->ref[d];
    if (f->pc[d] == 1) {
        RS_TRY(rs_take_obj(w, ref));
        return done(out);
    }
    if (f->pc[d]) return bad_pc(w);
    const Obj *o = rs_get_obj(w, ref);
    if (!o) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "pickup_obj of an obj no longer listed");
    Player *p = &w->p;
    if (rs_inv_itemspace2(w->P, p->inv, RS_INV_CAP, 0, o->type, o->count)) {   /* ~pickup_obj_check_for_space */
        RS_TRY(mes_text(w, RS_TX_pickup_no_space, 0));
        return done(out);
    }
    if (p->x != o->x || p->z != o->z) return delay_at(f, d, 1, 0, out);
    RS_TRY(rs_take_obj(w, ref));
    return done(out);
}

/* ---- timers (agent.py) and [changestat,_] (engine.py) ---------------------------------------- */
/* stat_regen: stat_add(stat, 1, 0) below the base, except prayer and hitpoints. */
static int s_stat_regen(World *w, RsSusp *out) {
    for (int i = 0; i < RS_N_SKILLS; i++)
        if (w->p.base_levels[i] > w->p.levels[i] && i != RS_SKILL_PRAYER && i != RS_SKILL_HITPOINTS)
            RS_TRY(rs_stat_add(w, i, 1, 0));
    return done(out);
}

/* stat_boost_restore: stat_sub(stat, 1, 0) above the base, except prayer and hitpoints. */
static int s_stat_boost_restore(World *w, RsSusp *out) {
    for (int i = 0; i < RS_N_SKILLS; i++)
        if (w->p.base_levels[i] < w->p.levels[i] && i != RS_SKILL_PRAYER && i != RS_SKILL_HITPOINTS)
            RS_TRY(rs_stat_sub(w, i, 1, 0));
    return done(out);
}

/* ---- the dispatcher ---------------------------------------------------------------------------------------------- */
static int step(World *w, Frame *f, int d, RsSusp *out) {
    int16_t s = f->script[d];
    switch (s) {
    case RS_S_ATTEMPT_CUT_TREE: return s_attempt_cut_tree(w, f, d, out);
    case RS_S_CUT_TREE: return s_cut_tree(w, f, d, out);
    case RS_S_USE_ON_TREE: return s_use_on_tree(w, f, d, out);
    case RS_S_MINE_FIRST_NORMAL: return s_mining(w, f, d, out, 1, 0);
    case RS_S_MINE_CONT_NORMAL: return s_mining(w, f, d, out, 0, 0);
    case RS_S_MINE_FIRST_FAST: return s_mining(w, f, d, out, 1, 1);
    case RS_S_MINE_CONT_FAST: return s_mining(w, f, d, out, 0, 1);
    case RS_S_GET_ORE_NORMAL: return s_get_ore(w, f, d, out, 0);
    case RS_S_GET_ORE_FAST: return s_get_ore(w, f, d, out, 1);
    case RS_S_PROSPECT: return s_prospect(w, f, d, out);
    case RS_S_GATE_OPEN_MAIN: return s_gate(w, f, d, out, 1, 0);
    case RS_S_GATE_OPEN_OUTER: return s_gate(w, f, d, out, 1, 1);
    case RS_S_GATE_CLOSE_MAIN: return s_gate(w, f, d, out, 0, 0);
    case RS_S_GATE_CLOSE_OUTER: return s_gate(w, f, d, out, 0, 1);
    case RS_S_DOOR_OPEN: return s_door(w, f, d, out, 1);
    case RS_S_DOOR_CLOSE: return s_door(w, f, d, out, 0);
    case RS_S_PICK: return s_pick(w, f, d, out);
    case RS_S_ATTEMPT_FISH_SHRIMP: return s_attempt_fish_shrimp(w, f, d, out);
    case RS_S_FISH_SHRIMP: return s_fish_shrimp(w, f, d, out);
    case RS_S_ATTEMPT_FISH_SARDINE: return s_attempt_fish_sardine(w, f, d, out);
    case RS_S_USE_ON_SALTFISH: return s_use_on_saltfish(w, f, d, out);
    case RS_S_CHECK_FISH_EQUIPMENT: return s_check_fish_equipment(w, f, d, out);
    case RS_S_PICKUP_OBJ: return s_pickup_obj(w, f, d, out);
    case RS_S_BURY_BONES: return s_bury_bones(w, f, d, out);
    case RS_S_DROP_SLOT: return s_drop_slot(w, f, d, out);
    case RS_S_USE_DEFAULT_WEAPON_STAB: return s_use_default(w, f, d, out, RS_OF_CASE_WEAPON_STAB, 0);
    case RS_S_USE_DEFAULT_FEATHER: return s_use_default(w, f, d, out, RS_OF_CASE_FEATHER, 0);
    case RS_S_USE_DEFAULT_CATEGORY_22: return s_use_default(w, f, d, out, RS_OF_CASE_CATEGORY_22, 1);
    case RS_S_USE_DEFAULT_CABBAGE: return s_use_default(w, f, d, out, RS_OF_CASE_CABBAGE, 0);
    case RS_S_USE_DEFAULT_POTATO: return s_use_default(w, f, d, out, RS_OF_CASE_POTATO, 0);
    case RS_S_USE_DEFAULT_ONION: return s_use_default(w, f, d, out, RS_OF_CASE_ONION, 0);
    case RS_S_USE_DEFAULT_BLACK_CAPE: return s_use_default(w, f, d, out, RS_OF_CASE_BLACK_CAPE, 0);
    case RS_S_WIELD: return s_wield(w, f, d, out);
    case RS_S_EAT: return s_eat(w, f, d, out);
    case RS_S_MESBOX: return s_box(w, f, d, out, K(w)->s_message1, 0);
    case RS_S_OBJBOX: return s_box(w, f, d, out, K(w)->s_objbox1, 1);
    case RS_S_DOUBLEOBJBOX: return s_doubleobjbox(w, f, d, out);
    case RS_S_LEVELUP: return s_levelup(w, f, d, out);
    case RS_S_CHANGESTAT:                      /* _gen(player_combat_stat): no suspension */
        if (f->pc[d]) return bad_pc(w);
        RS_TRY(rs_player_combat_stat(w));
        return done(out);
    case RS_S_STAT_REGEN: return f->pc[d] ? bad_pc(w) : s_stat_regen(w, out);
    case RS_S_STAT_BOOST_RESTORE: return f->pc[d] ? bad_pc(w) : s_stat_boost_restore(w, out);
    case RS_S_HEALTH_REGEN:                    /* stat_heal(hitpoints, 1, 0, cause regen) */
        if (f->pc[d]) return bad_pc(w);
        RS_TRY(rs_stat_heal(w, RS_SKILL_HITPOINTS, 1, 0, RS_HP_REGEN));
        return done(out);
    default:
        if (s >= 0 && s < RS_N_SCRIPTS && RS_SCRIPT_INFO[s].status == RS_NP)
            return RS_FAIL(&w->err, RS_E_NOT_PORTED_SCRIPT, f->arg[0][1]);   /* the trigger's name */
        return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a player frame of a script scripts.c does not run");
    }
}

int rs_script_step(World *w, Frame *f, RsSusp *out) {
    if (f->depth < 1 || f->depth > RS_FRAME_DEPTH)
        return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a player frame of no level");
    switch (f->script[0]) {
    case RS_S_COMBAT_START_AP: case RS_S_COMBAT_START: case RS_S_MELEE_ATTACK: case RS_S_COMBAT_DAMAGE_PLAYER:
    case RS_S_PLAYER_DEATH: case RS_S_PLAYERHIT_N_RETALIATE:
        return rs_combat_script_step(w, f, out);
    default:
        return step(w, f, 0, out);
    }
}
