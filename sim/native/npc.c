/* sim/native/npc.c: the npcs of sim/engine.py (Npc.turn and the World methods it calls, engine.py,
 * 1015-1401, 2259-2298): hunting (the body of World.npc_hunt_all; world.c's process_world applies its filter), the
 * lifecycle and respawn, the turn (delay release and script resume, hunt clock, hunt target, regen, AI timer, the npc
 * queue with _links' visit quirk, movement), wander, escape, the ai modes and their interaction, naive chase steps,
 * mode changes and resets, teleports, removal, NPC_FINDUID, and the npc script runner (Npc.executeScript and
 * resume_npc) with move_fishing_spot and npc_poison_timer. The combat scripts an npc runs (attack, retaliate, damage,
 * the death script and its drop program) are combat.c's, reached through rs_npc_combat_step. Each function names the
 * Python it mirrors (file and function, as rs.h).
 *
 * Exactness notes the code relies on:
 *   - Npc.target can only be the world's one player (engine.py rejects any other), so it is a flag; so are
 *     hunt_target, vars['aggressive_player'] and vars['attacking'] (None or the player's name).
 *   - npc.type == npc.base always (respawn restores it), so the type a turn reads once is the type of every later
 *     lookup in that turn, as in Python.
 *   - Draws go through rs_wm / rs_wrng with the oracle's key tuples: the wander go-draw is rng(2, tick, nid) (World's
 *     _wander prefix cache folds the same keys), the spot's picks rng(6, tick, nid, i) and its interval rng(6, tick,
 *     nid, 99).
 *   - World.is_multi is false in every declared world (rs.h World), so the single-way branches always apply. */
#include <string.h>

#include "rs.h"

static inline int32_t imax(int32_t a, int32_t b) { return a > b ? a : b; }
static inline int32_t iabs(int32_t a) { return a < 0 ? -a : a; }

/* self.t.npc_types[npc.type]: world init refuses a spawn of a type the pack lacks and npcs keep their base type, so
   the lookup cannot fail on a running world. */
static const RsNpcType *ntype(const World *w, const Npc *n) {
    const RsNpcType *nt = rs_npc_type(w->P, n->type);
    if (!nt) rs_trap("npc.c: an npc of a type the pack lacks (KeyError)");
    return nt;
}

/* _distance_sw (engine.py): Chebyshev distance from a tile to a 1x1 target. */
static int32_t dist_sw(int32_t x, int32_t z, int32_t tx, int32_t tz) { return imax(iabs(tx - x), iabs(tz - z)); }

static int op_mode(int32_t mode) { return RS_MODE_OPPLAYER1 <= mode && mode <= RS_MODE_OPPLAYER1 + 4; }
static int ap_mode(int32_t mode) { return RS_MODE_APPLAYER1 <= mode && mode <= RS_MODE_APPLAYER1 + 4; }

/* {0: 0, 1: 1, 3: 3, 4: 4}.get(moverestrict) (engine.py, 1368): -1 for None. */
static int32_t collision_of(int32_t moverestrict) {
    return moverestrict == 0 || moverestrict == 1 || moverestrict == 3 || moverestrict == 4 ? moverestrict : -1;
}

/* ---- hunting (engine.py) ------------------------------------------------------------------------------ */
int rs_npc_hunt_all(World *w, Npc *n) {
    const RsNpcType *nt = ntype(w, n);
    const Player *p = &w->p;
    n->hunt_target_player = 0;
    if (n->hunt_clock < nt->hunt_rate - 1 || n->huntrange < 1) return 0;
    if (p->level != n->level || dist_sw(n->x, n->z, p->x, p->z) > n->huntrange) return 0;
    if (nt->hunt_check_vis == 1) {
        int visible;
        RS_TRY(rs_line_of_sight(w, p->x, p->z, n->x, n->z, &visible));
        if (!visible) return 0;
    }
    if (nt->hunt_check_vis == 2) return RS_FAIL0(&w->err, RS_E_LINE_OF_WALK);
    if (nt->hunt_check_not_busy && rs_player_busy(w)) return 0;
    /* checkAfk is never true for the clientless agent (engine.py) */
    /* vislevel * 2 (a None vislevel would be a TypeError; RsNpcType's int16 vislevel cannot hold RS_NONE, and every
       hunting type of the declared world has one) */
    if (nt->hunt_check_not_too_strong == 1 && rs_combat_level(w) > nt->vislevel * 2) return 0;
    if (!n->target_player) {           /* npc.target is not p (and the world is single-way) */
        if (nt->hunt_check_not_combat != -1 && p->varp[RS_VARP_lastcombat] + 8 > w->tick) return 0;
        if (nt->hunt_check_not_combat_self != -1 && n->v_lastcombat + 8 > w->tick) return 0;
    }
    if (nt->hunt_check_vars || nt->hunt_check_inv != -1
        || (nt->hunt_check_not_combat != -1 && nt->hunt_check_not_combat != 38)
        || (nt->hunt_check_not_combat_self != -1 && nt->hunt_check_not_combat_self != 6))
        return RS_FAIL0(&w->err, RS_E_HUNT_CHECKS);
    n->hunt_target_player = 1;
    return 0;
}

/* ---- mode, target, movement primitives (engine.py) ---------------------------------------------------- */
/* Npc.resetDefaults (engine.py). */
void rs_npc_reset_defaults(World *w, Npc *n) {
    const RsNpcType *nt = ntype(w, n);
    n->target_player = 0;
    n->mode = nt->defaultmode;
    n->hunt_mode = nt->huntmode, n->huntrange = nt->huntrange, n->hunt_clock = 0, n->hunt_target_player = 0;
    n->timer_interval = nt->timer;
}

/* NPC_SETMODE (engine.py). target_player: the script's player (always valid, Player.is_valid). */
void rs_npc_set_mode(World *w, Npc *n, int32_t mode, int target_player) {
    if (mode == RS_MODE_NONE || mode == RS_MODE_WANDER || mode == RS_MODE_PATROL) {
        n->target_player = 0;
        n->mode = (int16_t)mode;
    } else if (mode == RS_MODE_NULL || !target_player) {
        rs_npc_reset_defaults(w, n);
    } else {
        n->target_player = 1;
        n->mode = (int16_t)mode;
    }
}

/* Npc.validateTarget + targetWithinMaxRange for a player target (engine.py): ranges from the spawn. */
int rs_npc_validate_target(const World *w, const Npc *n) {
    const Player *p = &w->p;
    if (!n->target_player || !w->has_player || p->level != n->level) return 0;
    const RsNpcType *nt = ntype(w, n);
    if (op_mode(n->mode)) {
        int32_t dx = iabs(p->x - n->start_x), dz = iabs(p->z - n->start_z);
        if (imax(dx, dz) > nt->maxrange + 1 || (dx == nt->maxrange + 1 && dz == nt->maxrange + 1)) return 0;
    } else if (ap_mode(n->mode)) {
        if (dist_sw(n->start_x, n->start_z, p->x, p->z) > nt->maxrange + nt->attackrange) return 0;
    } else if (n->mode == RS_MODE_PLAYERESCAPE) {
        if (dist_sw(n->start_x, n->start_z, p->x, p->z) > nt->maxrange
            && imax(iabs(n->x - n->start_x), iabs(n->z - n->start_z)) > nt->maxrange)
            return 0;
    } else if (dist_sw(n->start_x, n->start_z, p->x, p->z) > nt->maxrange + 1) {
        return 0;
    }
    return 1;
}

/* World.npc_teleport (engine.py). */
void rs_npc_teleport(World *w, Npc *n, int32_t x, int32_t z) {
    rs_occupy(w, n->level, n->x, n->z, n->size, 0);
    n->x = (int16_t)x, n->z = (int16_t)z;
    rs_occupy(w, n->level, n->x, n->z, n->size, 1);
    n->has_path = 0;
    n->tele = 1;
}

/* World.remove_npc (engine.py): respawnrate scaled for one player. */
void rs_remove_npc(World *w, Npc *n) {
    rs_occupy(w, n->level, n->x, n->z, n->size, 0);
    n->active = 0;
    n->lifecycle = (int32_t)rs_floordiv((int64_t)(4000 - 1) * ntype(w, n)->respawnrate, 4000);
}

/* NPC_FINDUID (engine.py): the npc in that nid's slot if it still has that type (respawning ones too). */
Npc *rs_npc_found(World *w, int32_t uid) {
    int32_t nid = (int32_t)((uint32_t)uid & 0xffffu), type = (int32_t)(((uint32_t)uid >> 16) & 0xffffu);
    for (int i = 0; i < w->n_npcs; i++)
        if (w->npcs[i].nid == nid) return w->npcs[i].type == type ? &w->npcs[i] : NULL;
    return NULL;
}

/* World.npc_update_movement (engine.py): one step along the path; moved compares the position with the
   end of the previous cycle (so a respawn or teleport counts as moved on its next step). */
static int update_movement(World *w, Npc *n, const RsNpcType *nt, int *moved) {
    *moved = 0;
    if (nt->moverestrict != RS_NOMOVE && n->has_path) {
        int32_t collision = collision_of(nt->moverestrict);
        if (collision < 0) return RS_FAIL(&w->err, RS_E_MOVE_RESTRICT, nt->moverestrict);
        RS_TRY(rs_process_movement_npc(w, n, collision, RS_FLAG_NPC));
        *moved = n->x != n->last_tick_x || n->z != n->last_tick_z;
    }
    if (*moved) {
        n->last_movement = w->tick + 1;
        n->stuck = 0;
    }
    return 0;
}

/* ---- the npc script runner (engine.py, 1381-1401) ------------------------------------------------------ */
/* resume_npc: one next(npc.script). The step runs on a copy, written back at a suspension only when npc.script is
   still that generator (by serial), as Python keeps whatever npc.script names at the yield; its end clears npc.script
   whatever it names then (`npc.script = None`). */
static int resume_npc(World *w, Npc *n) {
    Frame f = n->script;
    RsSusp s = {0};
    RS_TRY(rs_npc_script_step(w, n, &f, &s));
    if (s.kind == RS_DONE) {
        memset(&n->script, 0, sizeof n->script);
        return 0;
    }
    if (s.kind == RS_NPC_DELAY) {
        n->delayed = 1;
        n->delayed_until = w->tick + 1 + s.n;
    } else if (s.kind == RS_NPC_ARRIVEDELAY) {
        n->delayed = 1;
        n->delayed_until = s.n;
    } else {
        return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "npc script suspension (an Unsupported in Python)");
    }
    if (n->script.serial == f.serial) n->script = f;
    return 0;
}

int rs_npc_run(World *w, Npc *n, int16_t script, int32_t a0, int32_t a1, int32_t a2) {
    rs_frame_new(w, &n->script, script, (RsRef){0}, a0, a1, a2);
    n->script.mode = RS_SCRIPT_SUSPENDED;     /* an npc's script is none or suspended (rs.h) */
    return resume_npc(w, n);
}

/* A trigger's script of an npc (npc_script, npc_ai_script, npc_queue_script, engine.py, 1382-1386): the
   oracle's Unsupported for a name it has no port of, RS_NOT_PORTED for one C lacks, else Npc.executeScript. */
static int run_trigger(World *w, Npc *n, const RsNpcType *nt, int slot, int32_t a0, int32_t a1) {
    int16_t script = nt->trig_script[slot];
    if (script == RS_S_UNKNOWN) return RS_FAIL(&w->err, RS_E_NPC_SCRIPT, nt->trig_name[slot]);
    if (script <= RS_S_UNKNOWN || script >= RS_N_SCRIPTS || RS_SCRIPT_INFO[script].status == RS_NP)
        return RS_FAIL(&w->err, RS_E_NOT_PORTED_SCRIPT, nt->trig_name[slot]);
    if (slot == RS_NTRIG_QUEUE + 2) a0 = nt->trig_arg[slot], a1 = 0;   /* a death script: its drop root */
    return rs_npc_run(w, n, script, a0, a1, 0);
}

/* move_fishing_spot (engine.py), [ai_timer,_saltfish]: up to 50 keyed picks of the layout's fishing
   coordinates (LayoutTables' override of the spot's enum, world.py: level 0) for one that no other npc of the
   same category stands on, npc_delay(2), then the teleport and the next interval. local[0], local[1]: the dest. */
static int move_fishing_spot(World *w, Npc *n, Frame *f, RsSusp *out) {
    const RsNpcType *nt = ntype(w, n);
    const RsLayout *L = w->L;
    switch (f->pc[0]) {
    case 0: {
        if (nt->kind != RS_NPCK_SPOT)
            return RS_FAIL_TEXT(&w->err, RS_E_NOT_PORTED, "move_fishing_spot of an npc whose enum is not the layout's");
        if (L->n_fish < 1) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a fishing enum with no values (IndexError)");
        int32_t dx = n->x, dz = n->z;
        for (int i = 0; i < 50; i++) {
            int64_t k = rs_pick(rs_wm(w, RS_KEYS(RS_R_NPC_SCRIPT, w->tick, n->nid, i)), L->n_fish);
            int32_t x = L->x0 + L->fish[2 * k], z = L->z0 + L->fish[2 * k + 1], level = 0;
            int taken = 0;
            for (int j = 0; j < w->n_npcs && !taken; j++) {
                const Npc *o = &w->npcs[j];
                taken = o != n && o->level == level && o->x == x && o->z == z && ntype(w, o)->category == nt->category;
            }
            if (taken) continue;
            if (level != n->level) return RS_FAIL0(&w->err, RS_E_SPOT_LEVEL);
            dx = x, dz = z;
            break;
        }
        f->local[0] = dx, f->local[1] = dz;
        f->pc[0] = 1;
        out->kind = RS_NPC_DELAY, out->n = 2;
        return 0;
    }
    case 1:
        rs_npc_teleport(w, n, f->local[0], f->local[1]);
        n->timer_interval = 280 + (int32_t)rs_pick(rs_wm(w, RS_KEYS(RS_R_NPC_SCRIPT, w->tick, n->nid, 99)), 250);
        out->kind = RS_DONE;
        return 0;
    default:
        return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "move_fishing_spot resumed at an unknown pc");
    }
}

int rs_npc_script_step(World *w, Npc *n, Frame *f, RsSusp *out) {
    switch (f->script[0]) {
    case RS_S_MOVE_FISHING_SPOT: return move_fishing_spot(w, n, f, out);
    case RS_S_NPC_POISON_TIMER: out->kind = RS_DONE; return 0;   /* engine.py: an npc that is not poisoned */
    default: return rs_npc_combat_step(w, n, f, out);
    }
}

/* ---- lifecycle (engine.py) --------------------------------------------------------------------------- */
/* World.npc_respawn: Npc.resetEntity(true) at the spawn; its [ai_spawn] runs in the next cycle's npc event phase. */
static int npc_respawn(World *w, Npc *n) {
    n->type = n->base;
    const RsNpcType *nt = ntype(w, n);
    n->active = 1;
    n->x = n->start_x, n->z = n->start_z;
    for (int k = 0; k < 6; k++) n->levels[k] = nt->stats[k];
    rs_req_clear(w, &n->queue);
    n->has_path = 0;
    n->has_hero = 0, n->hero_damage = 0;
    n->life += 1;
    n->v_action_delay = 0, n->v_lastcombat = 0, n->v_aggressive_player = 0, n->v_attacking = 0, n->v_xp_mult = 0;
    n->v_stunned = 0;
    rs_npc_reset_defaults(w, n);
    n->tele = 1;
    rs_occupy(w, n->level, n->x, n->z, n->size, 1);
    if (w->n_npc_events >= RS_MAX_NPCS) return RS_OVERFLOW(&w->err, "RS_MAX_NPCS npc events");
    w->npc_events[w->n_npc_events++] = (uint8_t)(n - w->npcs);
    return 0;
}

int rs_npc_lifecycle(World *w, Npc *n) {
    if (!n->delayed && n->lifecycle > 0) {
        n->lifecycle -= 1;
        if (n->lifecycle == 0 && !n->active) RS_TRY(npc_respawn(w, n));
    }
    return 0;
}

/* ---- the queue (engine.py) ------------------------------------------------------------------------- */
/* Npc.processQueue: a pass reads each successor at its visit; a delay counts down while the npc is not delayed, and
   a due request is unlinked, then its AI_QUEUE<id> script runs (the death script for id 3). */
static int process_queue(World *w, Npc *n) {
    if (!n->active) return 0;
    RsLinks it;
    rs_links_begin(w, &n->queue, &it);
    for (int16_t node; (node = rs_links_next(w, &it)) >= 0;) {
        Req *r = &w->req[node];
        if (!n->delayed) r->delay -= 1;
        if (n->delayed || r->delay > 0) continue;
        rs_req_unlink(w, &n->queue, node);
        int32_t id = r->script, arg = r->arg[0], source = r->arg[1];
        if (id < 1 || id > 3)                              /* AI_QUEUE<id> beyond 3: no ported script queues one */
            return RS_FAIL_TEXT(&w->err, RS_E_NOT_PORTED, "an npc queue request beyond AI_QUEUE3");
        const RsNpcType *nt = ntype(w, n);
        if (nt->trig_script[RS_NTRIG_QUEUE + id - 1] == RS_S_NONE) continue;
        RS_TRY(run_trigger(w, n, nt, RS_NTRIG_QUEUE + id - 1, arg, source));
    }
    rs_links_end(w);
    return 0;
}

/* ---- the ai modes (engine.py) --------------------------------------------------------------------------- */
/* npc_ai_script (engine.py): the trigger's script against npc.target, when the type has one. */
static int ai_script(World *w, Npc *n, const RsNpcType *nt, int slot) {
    if (nt->trig_script[slot] == RS_S_NONE) return 0;
    return run_trigger(w, n, nt, slot, 0, 0);
}

/* npc_try_interact (engine.py): op modes from a tile beside the player (entity_reaching of its tile, before
   the player moves this tick); ap modes within attackrange and sight, not from under the player. */
static int try_interact(World *w, Npc *n, const RsNpcType *nt, int *done) {
    const Player *p = &w->p;
    *done = 0;
    if (op_mode(n->mode)) {
        if (n->size != 1) return RS_FAIL0(&w->err, RS_E_LARGE_MELEE);
        uint16_t mask;
        RS_TRY(rs_entity_reaching(w, p->x, p->z, &mask));
        int beside = 0;
        for (int b = 0; b < 9; b++)
            beside |= (mask >> b & 1) && p->x + RS_NEIGHBOUR_DX[b] == n->x && p->z + RS_NEIGHBOUR_DZ[b] == n->z;
        if (beside) {
            RS_TRY(ai_script(w, n, nt, RS_NTRIG_OPPLAYER + n->mode - RS_MODE_OPPLAYER1));
            *done = 1;
        }
    } else if (ap_mode(n->mode)) {
        if (n->size != 1) return RS_FAIL0(&w->err, RS_E_LARGE_APPROACH);
        if (dist_sw(n->x, n->z, p->x, p->z) <= nt->attackrange && (n->x != p->x || n->z != p->z)) {
            int visible;
            RS_TRY(rs_line_of_sight(w, p->x, p->z, n->x, n->z, &visible));
            if (visible) {
                RS_TRY(ai_script(w, n, nt, RS_NTRIG_APPLAYER + n->mode - RS_MODE_APPLAYER1));
                *done = 1;
            }
        }
    }
    return 0;
}

/* Npc.aiMode (engine.py): interact before moving, else a naive step toward the target, then again. */
static int ai_mode(World *w, Npc *n, const RsNpcType *nt) {
    n->stuck = 0;
    int done, moved;
    RS_TRY(try_interact(w, n, nt, &done));
    if (done) return 0;
    if (nt->moverestrict != RS_NOMOVE) {
        int16_t step[2];
        RS_TRY(rs_naive_path(w, n->level, n->x, n->z, w->p.x, w->p.z, 1, 1, n->size, step));
        if (step[0] != n->x || step[1] != n->z) n->has_path = 1, n->path_x = step[0], n->path_z = step[1];
    }
    RS_TRY(update_movement(w, n, nt, &moved));
    if (moved && !nt->givechase) {
        rs_npc_reset_defaults(w, n);
        return 0;
    }
    if (n->target_player) RS_TRY(try_interact(w, n, nt, &done));
    return 0;
}

/* Npc.playerEscapeMode (engine.py): a step away from the player, diagonal, then x, then z, within maxrange
   of the spawn; reset after 5 stuck ticks unless at maxrange on both axes. */
static int escape(World *w, Npc *n, const RsNpcType *nt) {
    const Player *p = &w->p;
    if (dist_sw(n->x, n->z, p->x, p->z) > 25) {
        rs_npc_reset_defaults(w, n);
        return 0;
    }
    int32_t dx = p->x >= n->x ? -1 : 1, dz = p->z >= n->z ? -1 : 1;
    int32_t collision = collision_of(nt->moverestrict);
    if (collision < 0 || n->size != 1) return RS_FAIL0(&w->err, RS_E_RETREAT_KIND);
    const int32_t steps[3][2] = {{dx, dz}, {dx, 0}, {0, dz}};
    for (int k = 0; k < 3; k++) {
        int ok;
        int32_t x = n->x + steps[k][0], z = n->z + steps[k][1];
        RS_TRY(rs_can_travel(w, n->level, n->x, n->z, steps[k][0], steps[k][1], 1, RS_FLAG_NPC, collision, &ok));
        if (ok && imax(iabs(x - n->start_x), iabs(z - n->start_z)) <= nt->maxrange) {
            n->has_path = 1, n->path_x = (int16_t)x, n->path_z = (int16_t)z;
            break;
        }
    }
    int moved;
    RS_TRY(update_movement(w, n, nt, &moved));
    if (!moved) n->stuck += 1;
    int at_max = iabs(n->x - n->start_x) >= nt->maxrange && iabs(n->z - n->start_z) >= nt->maxrange;
    if (n->stuck >= 5 && !at_max) {
        rs_npc_reset_defaults(w, n);
        n->stuck = 0;
    }
    return 0;
}

/* Npc.processMovementInteraction (engine.py). */
static int npc_movement(World *w, Npc *n, const RsNpcType *nt) {
    int moved;
    if (n->delayed || !n->active) return 0;
    if (n->mode == RS_MODE_NULL) n->mode = nt->defaultmode;
    if (n->mode == RS_MODE_NONE) return update_movement(w, n, nt, &moved);
    if (n->mode == RS_MODE_WANDER) {
        if (nt->moverestrict != RS_NOMOVE && rs_wrng(w, RS_KEYS(RS_R_WANDER_GO, w->tick, n->nid)) < 0.125) {
            double r = nt->wanderrange;   /* Npc.wander: Math.round(Math.random() * 2r - r) on each axis */
            int64_t dx = rs_js_round(rs_wrng(w, RS_KEYS(RS_R_WANDER_DX, w->tick, n->nid)) * (double)(nt->wanderrange * 2) - r);
            int64_t dz = rs_js_round(rs_wrng(w, RS_KEYS(RS_R_WANDER_DZ, w->tick, n->nid)) * (double)(nt->wanderrange * 2) - r);
            int64_t x = n->start_x + dx, z = n->start_z + dz;
            if (x != n->x || z != n->z) n->has_path = 1, n->path_x = (int16_t)x, n->path_z = (int16_t)z;
        }
        RS_TRY(update_movement(w, n, nt, &moved));
        int on_spawn = n->x == n->start_x && n->z == n->start_z;
        n->stuck += 1;
        if (n->stuck - 1 > 500) {
            if (!on_spawn) rs_npc_teleport(w, n, n->start_x, n->start_z);
            n->stuck = 0;
        }
        return 0;
    }
    if (n->mode == RS_MODE_PATROL) return RS_FAIL(&w->err, RS_E_PATROL, nt->debugname);
    if (!n->target_player || !rs_npc_validate_target(w, n)) {
        rs_npc_reset_defaults(w, n);
        return 0;
    }
    if (n->mode == RS_MODE_PLAYERESCAPE) return escape(w, n, nt);
    if (n->mode == RS_MODE_PLAYERFOLLOW) return RS_FAIL0(&w->err, RS_E_NPC_FOLLOW);
    if (n->mode == RS_MODE_PLAYERFACE) return 0;
    if (n->mode == RS_MODE_PLAYERFACECLOSE) {
        if (dist_sw(n->x, n->z, w->p.x, w->p.z) > 1) rs_npc_reset_defaults(w, n);
        return 0;
    }
    return ai_mode(w, n, nt);
}

/* ---- the turn (engine.py) ------------------------------------------------------------------------------- */
/* npc_consume_hunt_target (engine.py). */
static int consume_hunt_target(World *w, Npc *n, const RsNpcType *nt) {
    if (!n->hunt_target_player || !nt->has_hunt || nt->hunt_type == 0) return 0;
    if (22 <= nt->hunt_find_new_mode && nt->hunt_find_new_mode <= 41) return RS_FAIL0(&w->err, RS_E_HUNT_QUEUE);
    n->target_player = 1, n->mode = (int16_t)nt->hunt_find_new_mode;
    n->hunt_target_player = 0, n->hunt_clock = 0;
    if (!nt->hunt_find_keep) n->hunt_mode = -1;
    return 0;
}

/* npc_regen (engine.py): each level one point toward its base. */
static void npc_regen(Npc *n, const RsNpcType *nt) {
    n->regen_clock += 1;
    if (n->regen_clock < nt->regenrate) return;
    n->regen_clock = 0;
    for (int k = 0; k < 6; k++) n->levels[k] = (int16_t)(n->levels[k] + (n->levels[k] < nt->stats[k])
                                                          - (n->levels[k] > nt->stats[k]));
}

int rs_npc_turn(World *w, Npc *n) {
    if (n->active) {
        if (n->delayed && w->tick >= n->delayed_until) n->delayed = 0;
        if (!n->delayed && n->script.serial) RS_TRY(resume_npc(w, n));
    }
    RS_TRY(rs_npc_lifecycle(w, n));
    if (!n->active || n->delayed) return 0;
    const RsNpcType *nt = ntype(w, n);
    if (n->hunt_mode != -1 && nt->has_hunt && (nt->hunt_nobody_near != 0 || n->observers > 0 || nt->hunt_type == 1)) {
        if (nt->hunt_type != 1) return RS_FAIL(&w->err, RS_E_HUNT_TYPE, nt->hunt_type);
        n->hunt_clock += 1;
    }
    RS_TRY(consume_hunt_target(w, n, nt));
    npc_regen(n, nt);
    if (n->timer_interval > 0) {
        n->timer_clock += 1;
        /* decide('npc_timer', True, npc) is True in training (world.control is None) */
        if (n->timer_clock >= n->timer_interval && nt->trig_script[RS_NTRIG_TIMER] != RS_S_NONE) {
            RS_TRY(run_trigger(w, n, nt, RS_NTRIG_TIMER, 0, 0));
            n->timer_clock = 0;
        }
    }
    RS_TRY(process_queue(w, n));
    return npc_movement(w, n, nt);
}
