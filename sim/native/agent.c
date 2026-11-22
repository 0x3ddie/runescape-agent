/* sim/native/agent.c: the environment agent of sim/agent.py over a World: Agent.start with _login and
 * _arrive, the [login,_] timers, Agent.step (_begin, the paced chunks of decision points, _finish), wait_ticks with the
 * per-tick bookkeeping (keep_alive: observe_npcs, visit, death_tick), option tracking (option_busy, begin, end_option,
 * act_done), act-v3's routing (a click's tryMove with its fallback, send_route, the loc, npc and obj ops by the
 * client's rule over paths.c's rs_client_route), dispatch and every action handler (pages, drop, WAIT, USE, CLICK, act,
 * held ops, use on held, dialogs, STYLE, CLOSE_IF). Each function names the Python it mirrors (sim/agent.py).
 *
 * The observation side (scan, build_obs, project_items, the mask, snapshot, describe_action, feedback with
 * publish_chat and dialog_text, and capture_text) is obs.c's and text.c's, called through rs.h only. This file
 * reads what dispatch reads from the last observation (s->a.obs) and the agent fields project_items fills (act
 * targets, the scanned objs, using). The route-v1 record (Agent.decision_route) is evidence for checks, never a policy
 * input or state, and is not kept.
 *
 * Exactness notes the code relies on:
 *   - Option identity (`self.option is running`) is RsOption.serial; a new option takes the next
 *     Agent.next_option_serial, so no two options of an episode share one.
 *   - item_state() equality (agent.py) is slot-by-slot equality of the inventory's listed (slot, id, count): slots
 *     are compared with empty slots (id < 0) normalized, so no stale count of an empty slot can differ.
 *   - The agent keeps no loc or obj RsRef across a cycle (rs.h IDENTITY): act() re-finds a loc by tile and type, an obj
 *     in the last scan, and an npc by its slot (npcs never leave their slot).
 *   - Python's strings become ids: options compare by their lowercase string id (RsPack.str_lower), labels by
 *     (RS_LABEL_*, k), reasons by RS_END_*. The diagnostic notes (actionNote, lastOptionNote) are
 *     not projected and not kept. */
#include <string.h>

#include "rs.h"

/* ---- small helpers ----------------------------------------------------------------------------------------------- */
static inline int32_t imax(int32_t a, int32_t b) { return a > b ? a : b; }
static inline int32_t imin(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t iabs(int32_t a) { return a < 0 ? -a : a; }
/* chebyshev (agent.py) */
static inline int32_t cheb(int32_t ax, int32_t az, int32_t bx, int32_t bz) {
    return imax(iabs(ax - bx), iabs(az - bz));
}
static inline int32_t hp_of(const World *w) { return w->p.levels[RS_SKILL_HITPOINTS]; }   /* Player.hp */
static inline int32_t lower_of(const RsPack *P, int32_t str) {
    return str >= 0 && str < P->n_strings ? P->str_lower[str] : 0; }

/* sum(p.stats) (int64: it can exceed 2^31). */
static int64_t xp_sum(const Player *p) {
    int64_t t = 0;
    for (int i = 0; i < RS_N_SKILLS; i++) t += p->stats[i];
    return t;
}

/* Agent.item_state (agent.py) as the inventory's slots, empty ones normalized. */
static void item_state(const Player *p, Slot out[RS_INV_CAP]) {
    for (int i = 0; i < RS_INV_CAP; i++) out[i] = p->inv[i].id < 0 ? (Slot){-1, 0} : p->inv[i];
}
static int same_items(const Slot *a, const Slot *b) {
    for (int i = 0; i < RS_INV_CAP; i++)
        if (a[i].id != b[i].id || a[i].count != b[i].count) return 0;
    return 1;
}

static int same_key(const RsAffKey *a, const RsAffKey *b) {
    return a->kind == b->kind && a->level == b->level && a->x == b->x && a->z == b->z && a->id == b->id
        && a->option_lc == b->option_lc;
}

/* act_contract.click_slot: the click slot of an action row, or -1. */
static int click_slot(int32_t action) {
    if (action >= RS_A_CLICK_LOW && action < RS_A_CLICK_LOW + RS_N_CLICK_LOW) return action - RS_A_CLICK_LOW;
    if (action >= RS_A_CLICK_HIGH && action < RS_A_CLICK_HIGH + RS_N_CLICK - RS_N_CLICK_LOW)
        return action - RS_A_CLICK_HIGH + RS_N_CLICK_LOW;
    return -1;
}

/* option_label (agent.py): (RS_LABEL_*, k) of act_k and nav_click_k, or 0 for an action that names no option. */
static int label_of(int32_t action, uint8_t *kind, uint8_t *k) {
    if (action >= RS_A_ACT0 && action < RS_A_ACT0 + RS_N_ACT) {
        *kind = RS_LABEL_ACT, *k = (uint8_t)(action - RS_A_ACT0);
        return 1;
    }
    int c = click_slot(action);
    if (c >= 0) {
        *kind = RS_LABEL_NAV_CLICK, *k = (uint8_t)c;
        return 1;
    }
    return 0;
}

/* `p.script is not None and p.script[1] == 'pausebutton'` */
static int suspended(const Player *p) { return p->script.serial && p->script.mode == RS_SCRIPT_PAUSE; }

/* ---- the per-tick bookkeeping (agent.py) ----------------------------------------------------------------- */
/* Agent.visit (agent.py): the player's (level, x // 8, z // 8) cell. A cell of the layout is a bit of
   Agent.visited; one off it (LOGIN_TILE's, before _arrive clears the set) goes to visited_off. */
static int visit(RsSlot *s) {
    Agent *a = &s->a;
    const Player *p = &s->w.p;
    const RsLayout *L = s->w.L;
    int32_t cx = (p->x >> 3) - (L->x0 >> 3), cz = (p->z >> 3) - (L->z0 >> 3);   /* x0, z0: multiples of 8 */
    if (p->level == 0 && p->x >= L->x0 && p->z >= L->z0 && cx < 8 && cz < 8) {
        a->visited |= 1ull << (cx * 8 + cz);
        return 0;
    }
    RsTile cell = {p->level, (int16_t)(p->x >> 3), (int16_t)(p->z >> 3)};   /* positive coordinates: // is >> */
    for (int i = 0; i < a->n_visited_off; i++)
        if (a->visited_off[i].level == cell.level && a->visited_off[i].x == cell.x && a->visited_off[i].z == cell.z)
            return 0;
    if (a->n_visited_off >= RS_VISIT_OFF_CAP)
        return RS_OVERFLOW(&s->w.err, "RS_VISIT_OFF_CAP visited cells off the layout");
    a->visited_off[a->n_visited_off++] = cell;
    return 0;
}

static void clear_visited(Agent *a) {
    a->visited = 0;
    a->n_visited_off = 0;
}

/* Agent.keep_alive (agent.py) with observe_npcs (254-260): npc.observers feeds the next cycle's hunting. The
   agent's player always exists here (it is made in _login before any tick). */
static int keep_alive(RsSlot *s) {
    World *w = &s->w;
    const Player *p = &w->p;
    for (int i = 0; i < w->n_npcs; i++) {
        Npc *n = &w->npcs[i];
        int near = n->active && !n->tele && n->level == p->level && cheb(p->x, p->z, n->x, n->z) <= 15;
        n->observers = (uint8_t)(near ? 1 : 0);
    }
    RS_TRY(visit(s));
    if ((hp_of(w) <= 0 || p->dead) && s->a.death_tick < 0) s->a.death_tick = w->tick;
    return 0;
}

/* Agent.wait_ticks (agent.py): per tick, the watched cycle (World.cycle, then GoalEpisode._watch once armed),
   the tick hash point, capture_text, keep_alive; nothing more once the episode's terminal cycle has passed. */
static int wait_ticks(RsSlot *s, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        if (rs_ended(s)) return 0;
        s->cycles++;
        RS_TRY(rs_world_cycle(&s->w));
        if (s->ep.armed) rs_task_watch(s);   /* task.py: _arm wraps world.cycle */
        rs_trace_tick(s);
        RS_TRY(rs_capture_text(s));
        RS_TRY(keep_alive(s));
    }
    return 0;
}

/* Agent.pace_ticks (agent.py): 0..pace_jitter extra ticks, keyed (R_PACE, pace_seed, pace, decision). */
static int32_t pace_ticks(RsSlot *s) {
    Agent *a = &s->a;
    a->pace += 1;
    return (int32_t)rs_pick(rs_wm(&s->w, RS_KEYS(RS_R_PACE, a->pace_seed, a->pace, a->decision)), a->pace_jitter + 1);
}

/* Agent.truncated (agent.py). */
static int truncated(const Agent *a) {
    return a->charged >= a->max_decisions || a->episode_ticks >= a->max_episode_ticks;
}

int rs_agent_truncated(const Agent *a) { return truncated(a); }

/* ---- [login,_] (agent.py) ------------------------------------------------------------------------------ */
int rs_agent_login(World *w) {
    Player *p = &w->p;
    static const int16_t scripts[3] = {RS_S_STAT_REGEN, RS_S_STAT_BOOST_RESTORE, RS_S_HEALTH_REGEN};
    for (int i = 0; i < 3; i++) {   /* Player.set_timer (engine.py): a new name appends to the dict */
        int k = 0;
        while (k < p->n_timers && p->timer[k].script != scripts[i]) k++;
        if (k == p->n_timers) {
            if (p->n_timers >= 3) return RS_OVERFLOW(&w->err, "three player timers");
            p->n_timers++;
        }
        p->timer[k] = (Timer){.interval = 100, .clock = w->tick, .script = scripts[i], .kind = 0};
    }
    return rs_player_combat_stat(w);
}

/* ---- option tracking (agent.py) -------------------------------------------------------------------------- */
/* Agent.end_option (agent.py): one slot, overwritten by a later end. */
static void end_option(Agent *a, int reason, int refused) {
    if (!a->option.has) return;
    a->option_end = (RsOptionEnd){.has = 1, .reason = (uint8_t)reason, .refused = (uint8_t)(refused != 0),
                                  .reported = 0, .action = a->option.has_action ? a->option.action : -1};
}

/* agent.refused[key] = until: an existing key is overwritten; a full table first drops the entries no tick can make
   active again (rs.h Agent). */
static int refuse(RsSlot *s, const RsAffKey *key, int32_t until) {
    Agent *a = &s->a;
    for (int i = 0; i < a->n_refused; i++)
        if (same_key(&a->refused[i].key, key)) { a->refused[i].until = until; return 0; }
    if (a->n_refused >= RS_REFUSED_CAP) {
        int k = 0;
        for (int i = 0; i < a->n_refused; i++)
            if (a->refused[i].until > s->w.tick) a->refused[k++] = a->refused[i];
        a->n_refused = k;
        if (k >= RS_REFUSED_CAP) return RS_OVERFLOW(&s->w.err, "RS_REFUSED_CAP refused affordances");
    }
    a->refused[a->n_refused++] = (RsRefused){*key, until};
    return 0;
}

/* Agent.loc_present (agent.py): a loc of that type still stands on the tile. */
static int loc_present(World *w, int32_t type, int32_t x, int32_t z, int32_t level) {
    uint8_t here[RS_MAX_LOCS];
    int n = rs_active_locs_at(w, level, x, z, here);
    for (int i = 0; i < n; i++)
        if (w->locs[here[i]].type == type) return 1;
    return 0;
}

/* Agent.begin (agent.py): a new option (no act or action yet). */
static RsOption *begin(RsSlot *s, uint8_t label, uint8_t k, uint8_t done) {
    Agent *a = &s->a;
    const World *w = &s->w;
    RsOption *o = &a->option;
    o->has = 1;
    o->label = label, o->label_k = k, o->done = done;
    o->has_act = o->has_action = 0;
    o->serial = ++a->next_option_serial;
    o->start = o->last_progress = w->tick;
    o->done_start = 0;
    o->last_x = w->p.x, o->last_z = w->p.z;
    o->action = -1;
    return o;
}

/* act_done(start) (agent.py): idle (no interaction, no waypoints) and more than IDLE_GRACE_TICKS since. */
static int act_done(const World *w, int32_t start) {
    return w->p.target.kind == RS_REF_NONE && w->p.path_n == 0 && w->tick - start > RS_IDLE_GRACE_TICKS;
}

/* ---- act-v3's routing (agent.py route, send_route, click, interact_loc, interact_npc, pick_up) ------------------- */
/* Agent.send_route: MoveClickHandler: clearPendingAction, then the turn points as the walk; a route that is only the
   player's own tile clears the walk and leaves no repath (AllowRepath.NONE). */
static void send_route(World *w, const int16_t route[][2], int n) {
    Player *p = &w->p;
    rs_clear_pending_action(w);
    if (n == 1 && route[0][0] == p->x && route[0][1] == p->z) {
        rs_queue_waypoints(w, route, 0);
        p->allow_repath = 0;
    } else {
        rs_queue_waypoints(w, route, n);
    }
}

/* Agent.click: tryMove to the slot's offset with its 3 x 3 fallback, then MoveClickHandler. Refused while the player is
   delayed, and with no route nothing is sent: an earlier walk continues and the click ends failed (refused) at once.
   The option ends when the walk ends, or stalls after NAV_STALL_TICKS. */
static int click(RsSlot *s, int32_t action) {
    World *w = &s->w;
    Player *p = &w->p;
    int k = click_slot(action), found = 0, n = 0;
    int16_t route[RS_PATH_MAX][2];
    uint8_t via;
    if (!p->delayed)
        RS_TRY(rs_client_route(w, p->x + RS_CLICK_DX[k], p->z + RS_CLICK_DZ[k], NULL, 1, route, &n, &via, &found));
    if (!found) {
        s->a.option_end = (RsOptionEnd){.has = 1, .reason = RS_END_FAILED, .refused = 1, .reported = 0, .action = action};
        return 0;   /* f'click {dx},{dz}: refused while delayed' or 'no route' */
    }
    send_route(w, (const int16_t (*)[2])route, n);
    begin(s, RS_LABEL_NAV_CLICK, (uint8_t)k, RS_DONE_WALK);
    return 0;
}

/* The index of the first op of a type's op list that is visible and equals `option` ignoring case (interact_loc and
   interact_npc), or -1. */
static int matching_op(const RsPack *P, const int32_t op[5], const uint8_t state[5], int32_t option) {
    int32_t want = lower_of(P, option);
    for (int i = 0; i < 5; i++)
        if (state[i] == RS_OPS_VISIBLE && lower_of(P, op[i]) == want) return i;
    return -1;
}

/* Agent.handled: the option an op handler's interaction waits for (it begins even when set_interaction refuses). */
static void handled(RsSlot *s, uint8_t k) {
    begin(s, RS_LABEL_ACT, k, RS_DONE_ACT)->done_start = s->w.tick;
}

/* Agent.interact_loc: tryMove to the loc by the client's rule (no fallback), a route sent as MoveClickHandler takes it,
   then OpLocHandler or OpLocUHandler (using): refused while the player is delayed; clearPendingAction; the
   interaction. With no route only the op is sent. */
static int interact_loc(RsSlot *s, int slot, int32_t option, uint8_t k, int has_using, int32_t using_id) {
    World *w = &s->w;
    Player *p = &w->p;
    const Loc *loc = &w->locs[slot];
    const RsLocType *lt = rs_loc_type(w->P, loc->type);
    if (!lt) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a loc of a type the pack lacks");
    int op = has_using ? -1 : matching_op(w->P, lt->op, lt->op_state, option);
    if (op < 0 && !has_using) return 0;   /* f'{label}: no matching op' */
    if (p->delayed) return 0;             /* f'{label}: refused while delayed' */
    RsArrival arrival = rs_loc_arrival(w->P, loc);
    int16_t route[RS_PATH_MAX][2];
    int n = 0, found = 0;
    uint8_t via;
    RS_TRY(rs_client_route(w, loc->x, loc->z, &arrival, 0, route, &n, &via, &found));
    if (found) send_route(w, (const int16_t (*)[2])route, n);
    rs_clear_pending_action(w);
    if (has_using) p->last_useitem = using_id;
    (void)rs_set_interaction(w, rs_ref_loc(w, slot), has_using ? RS_APLOCU : RS_APLOC1 + op);
    handled(s, k);
    return 0;
}

/* Agent.interact_npc: tryMove beside its tile (1 x 1, no fallback), then OpNpcHandler or OpNpcUHandler: refused while
   the player or the npc is delayed or while the npc is not in the player's npc list (observers); clearPendingAction;
   the interaction. The server's naive chase follows a moving npc. */
static int interact_npc(RsSlot *s, int slot, int32_t option, uint8_t k, int has_using, int32_t using_id) {
    World *w = &s->w;
    Player *p = &w->p;
    const Npc *npc = &w->npcs[slot];
    const RsNpcType *nt = rs_npc_type(w->P, npc->type);
    if (!nt) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an npc of a type the pack lacks");
    int op = has_using ? -1 : matching_op(w->P, nt->op, nt->op_state, option);
    if (op < 0 && !has_using) return 0;                    /* f'{label}: no matching op' */
    if (p->delayed || npc->delayed || !npc->observers) return 0;   /* f'{label}: refused ({refused})' */
    int16_t route[RS_PATH_MAX][2];
    int n = 0, found = 0;
    uint8_t via;
    RS_TRY(rs_client_route(w, npc->x, npc->z, &RS_ENTITY_ARRIVAL, 0, route, &n, &via, &found));
    if (found) send_route(w, (const int16_t (*)[2])route, n);
    rs_clear_pending_action(w);
    if (has_using) p->last_useitem = using_id;
    (void)rs_set_interaction(w, rs_ref_npc(w, slot), has_using ? RS_APNPCU : RS_APNPC1 + op);
    handled(s, k);
    return 0;
}

/* Agent.pick_up: tryMove onto the obj's tile, else beside it, then OpObjHandler on its Take op (RsObjType.take_op):
   refused while the player is delayed; clearPendingAction; the interaction. */
static int pick_up(RsSlot *s, int slot) {
    World *w = &s->w;
    Player *p = &w->p;
    const Obj *obj = &w->objs[slot];
    const RsObjType *ot = rs_obj_type(w->P, obj->type);
    if (!ot) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an obj of a type the pack lacks");
    if (p->delayed) return 0;   /* f'pick {name}: refused while delayed' */
    int16_t route[RS_PATH_MAX][2];
    int n = 0, found = 0;
    uint8_t via;
    RS_TRY(rs_client_route(w, obj->x, obj->z, NULL, 0, route, &n, &via, &found));
    if (!found) RS_TRY(rs_client_route(w, obj->x, obj->z, &RS_ENTITY_ARRIVAL, 0, route, &n, &via, &found));
    if (found) send_route(w, (const int16_t (*)[2])route, n);
    rs_clear_pending_action(w);
    (void)rs_set_interaction(w, rs_ref_obj(w, slot), RS_APOBJ1 + ot->take_op);
    begin(s, RS_LABEL_PICK, 0, RS_DONE_PICK);
    return 0;
}

/* ---- option_busy (agent.py) ------------------------------------------------------------------------------ */
int rs_option_busy(RsSlot *s, int *busy) {
    Agent *a = &s->a;
    World *w = &s->w;
    Player *p = &w->p;
    RsOption *o = &a->option;
    *busy = 0;
    if (!o->has) return 0;
    int32_t now = w->tick;
    if (p->x != o->last_x || p->z != o->last_z) o->last_x = p->x, o->last_z = p->z, o->last_progress = now;
    RsAct *act = o->has_act ? &o->act : NULL;
    Slot items[RS_INV_CAP];
    if (act) {
        int64_t xp = xp_sum(p);
        item_state(p, items);
        if (xp != act->last_xp || !same_items(items, act->last_items) || p->modal_main != -1) {
            act->last_xp = xp;
            memcpy(act->last_items, items, sizeof items);
            o->last_progress = now;
        }
    }
    int done = o->done == RS_DONE_ACT ? act_done(w, o->done_start)
             : o->done == RS_DONE_PICK ? p->target.kind == RS_REF_NONE : p->path_n == 0;   /* not has_waypoints() */
    if (done) {
        int refused = 0;
        if (act) {
            item_state(p, items);
            refused = xp_sum(p) == act->xp && same_items(items, act->items) && p->modal_main == -1
                && (!act->has_loc || loc_present(w, act->loc_type, act->loc_x, act->loc_z, act->loc_level));
        }
        if (refused) RS_TRY(refuse(s, &act->key, now + RS_REFUSED_TICKS));
        end_option(a, RS_END_DONE, refused);
        o->has = 0;
        return 0;
    }
    int nav_label = o->label == RS_LABEL_NAV_CLICK;   /* label.startswith('nav') */
    int stalled = nav_label ? now - o->last_progress > RS_NAV_STALL_TICKS
                            : act && now - o->last_progress > RS_ACT_STALL_TICKS;
    if (stalled && act) RS_TRY(refuse(s, &act->key, now + RS_REFUSED_TICKS));
    int expired = now - o->start > RS_OPTION_HARD_TIMEOUT_TICKS;
    if (stalled || expired) {
        end_option(a, stalled ? RS_END_STALLED : RS_END_EXPIRED, stalled && act);
        o->has = 0;
        return 0;
    }
    *busy = 1;
    return 0;
}

/* ---- dispatch (agent.py) --------------------------------------------------------------------------------- */
/* match_item (agent.py) of the inventory slot an invSel/heldSel entry names. */
static int matches_inventory(const Player *p, int32_t slot, int32_t id, int32_t count) {
    if (slot < 0 || slot >= RS_INV_CAP || p->inv[slot].id < 0) return 0;   /* no current item */
    return count > 0 && p->inv[slot].id == id && p->inv[slot].count == count;
}

/* Agent.act (agent.py). */
static int act(RsSlot *s, int32_t action) {
    Agent *a = &s->a;
    World *w = &s->w;
    Player *p = &w->p;
    int k = action - RS_A_ACT0;
    if (k >= a->obs.n_act || k >= a->n_act_targets) return 0;   /* 'act: slot empty' */
    const RsAff sel = a->obs.act_sel[k];
    int has_using = a->has_using;
    int32_t using_id = a->using_id;
    a->has_using = 0;   /* using, self.using = self.using, None: consumed even when the target is gone */
    a->using_slot = a->using_id = a->using_name = 0;
    if (sel.kind == RS_KIND_LOC) {
        uint8_t here[RS_MAX_LOCS];
        int n = rs_active_locs_at(w, p->level, sel.x, sel.z, here), slot = -1;
        for (int i = 0; i < n && slot < 0; i++)
            if (w->locs[here[i]].type == sel.id) slot = here[i];
        if (slot < 0 || cheb(w->locs[slot].x, w->locs[slot].z, p->x, p->z) > RS_LOC_SCAN_TILES) return 0;   /* gone */
        RS_TRY(interact_loc(s, slot, sel.option, (uint8_t)k, has_using, using_id));
    } else if (sel.kind == RS_KIND_NPC) {
        RsRef src = a->act_targets[k];
        if (src.kind != RS_REF_NPC || src.slot >= w->n_npcs)
            return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an npc affordance whose target is not an npc");
        const Npc *npc = &w->npcs[src.slot];
        if (!(npc->active && !npc->delayed) || npc->level != p->level
            || cheb(npc->x, npc->z, p->x, p->z) > RS_NPC_SCAN_TILES)
            return 0;   /* gone */
        RS_TRY(interact_npc(s, src.slot, sel.option, (uint8_t)k, has_using, using_id));
    } else {
        int slot = -1;   /* the first scanned obj of that type on that tile (agent.py) */
        for (int i = 0; i < a->n_scan_objs && slot < 0; i++) {
            const Obj *o = &w->objs[a->scan_objs[i]];
            if (o->type == sel.id && o->x == sel.x && o->z == sel.z) slot = a->scan_objs[i];
        }
        if (slot < 0) return 0;   /* gone */
        RS_TRY(pick_up(s, slot));
    }
    if (a->option.has) {
        RsAct *ac = &a->option.act;
        a->option.has_act = 1;
        ac->key = (RsAffKey){.kind = sel.kind, .level = sel.level, .x = sel.x, .z = sel.z, .id = sel.id,
                             .option_lc = lower_of(w->P, sel.option)};
        ac->xp = ac->last_xp = xp_sum(p);
        item_state(p, ac->items);
        memcpy(ac->last_items, ac->items, sizeof ac->items);
        ac->has_loc = sel.kind == RS_KIND_LOC;
        ac->loc_type = sel.id, ac->loc_x = sel.x, ac->loc_z = sel.z, ac->loc_level = sel.level;
        if (!ac->has_loc) ac->loc_type = 0, ac->loc_x = ac->loc_z = 0, ac->loc_level = 0;
    }
    return 0;
}

/* Agent.resume_dialog (agent.py). Choices are dead here (resume_buttons are set only by the unported
   p_choice2) but kept as written. */
static int resume_dialog(World *w, int choice) {
    Player *p = &w->p;
    if (!suspended(p)) return 0;   /* 'dialog: nothing suspended' */
    if (choice >= 0 && p->n_resume) {
        int k = choice < p->n_resume - 1 ? choice : p->n_resume - 1;
        p->last_com_iface = p->resume_iface[k], p->last_com_com = p->resume_com[k];
        p->n_resume = 0;
        return rs_resume(w);
    }
    if (p->n_resume) return 0;   /* 'dialog: choice required' */
    return rs_resume(w);
}

/* Agent.op_held (held-v2): refused while delayed; the iop matching the option (case-insensitively, by original index);
   clearPendingAction; the OPHELD trigger, run protected; no matching iop or no trigger is a note only. */
static int op_held(World *w, int32_t slot, int32_t option) {
    Player *p = &w->p;
    if (p->inv[slot].id < 0) return 0;   /* 'opheld: empty slot' */
    if (p->delayed) return 0;            /* 'opheld: refused while delayed' */
    int32_t obj = p->inv[slot].id;
    const RsObjType *t = rs_obj_type(w->P, obj);
    if (!t) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an obj of a type the pack lacks");
    int32_t want = lower_of(w->P, option), op = -1;
    for (int k = 0; k < 5 && op < 0; k++)
        if (t->iop[k] != 0 && lower_of(w->P, t->iop[k]) == want) op = k;
    if (op < 0) return 0;   /* 'opheld: no matching iop' */
    p->last_item = obj, p->last_slot = slot;
    rs_clear_pending_action(w);
    int j = RS_OTRIG_HELD + op;
    if (t->trig_script[j] == RS_S_NONE) return 0;   /* 'opheld: no script for op' */
    return rs_run_held(w, t->trig_script[j], t->trig_arg[j], t->trig_name[j]);
}

/* Agent.drop_slot (held-v2): refused while delayed; clearPendingAction; the OPHELD5 trigger. */
static int drop_slot(World *w, int32_t slot) {
    Player *p = &w->p;
    if (p->inv[slot].id < 0) return 0;   /* 'drop: empty slot' */
    if (p->delayed) return 0;            /* 'drop: refused while delayed' */
    int32_t obj = p->inv[slot].id;
    const RsObjType *t = rs_obj_type(w->P, obj);
    if (!t) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an obj of a type the pack lacks");
    p->last_item = obj, p->last_slot = slot;
    rs_clear_pending_action(w);
    int j = RS_OTRIG_HELD + 4;
    if (t->trig_script[j] == RS_S_NONE) return RS_FAIL0(&w->err, RS_E_DROP_NO_SCRIPT);
    return rs_run_held(w, t->trig_script[j], t->trig_arg[j], t->trig_name[j]);
}

/* use_on_held's script(obj, category) (agent.py): the OPHELDU trigger when it is the obj's own (category
   false) or its category's (true); '[opheldu,_]' and no trigger are RS_S_NONE in the pack. */
static int heldu_script(const RsPack *P, int32_t obj, int category, int16_t *script, int16_t *arg, int32_t *name) {
    const RsObjType *t = rs_obj_type(P, obj);
    if (!t || t->trig_script[RS_OTRIG_HELDU] == RS_S_NONE || (t->heldu_own != 0) == (category != 0)) return 0;
    *script = t->trig_script[RS_OTRIG_HELDU], *arg = t->trig_arg[RS_OTRIG_HELDU], *name = t->trig_name[RS_OTRIG_HELDU];
    return 1;
}

/* Agent.use_on_held: OpHeldUHandler, refused while delayed (held-v2), then its lookup; the swap after the second lookup
   is unconditional. */
static int use_on_held(World *w, int32_t slot, int32_t using_slot, int32_t using_id) {
    Player *p = &w->p;
    if (p->inv[slot].id < 0) return 0;   /* 'use: empty slot' */
    if (p->delayed) return 0;            /* 'use: refused while delayed' */
    int32_t item = p->inv[slot].id;
    const RsObjType *ti = rs_obj_type(w->P, item), *tu = rs_obj_type(w->P, using_id);
    if (!ti || !tu) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an obj of a type the pack lacks");
    p->last_item = item, p->last_slot = slot, p->last_useitem = using_id, p->last_useslot = using_slot;
    int16_t script = RS_S_NONE, arg = -1;
    int32_t name = 0, t;
    int found = heldu_script(w->P, item, 0, &script, &arg, &name);
    if (!found) {
        found = heldu_script(w->P, using_id, 0, &script, &arg, &name);
        t = p->last_item, p->last_item = p->last_useitem, p->last_useitem = t;
        t = p->last_slot, p->last_slot = p->last_useslot, p->last_useslot = t;
    }
    if (!found && ti->category != -1) found = heldu_script(w->P, item, 1, &script, &arg, &name);
    if (!found && tu->category != -1) {
        found = heldu_script(w->P, using_id, 1, &script, &arg, &name);
        t = p->last_item, p->last_item = p->last_useitem, p->last_useitem = t;
        t = p->last_slot, p->last_slot = p->last_useslot, p->last_useslot = t;
    }
    rs_clear_pending_action(w);
    if (!found) {   /* 'use: nothing interesting happens': OpHeldUHandler's debug line, then its literal */
        int32_t msg;
        RS_TRY(rs_msg(w, RS_MK_NO_TRIGGER_OPHELDU, item, &msg));
        RS_TRY(rs_mes(w, msg));
        RS_TRY(rs_msg(w, RS_MK_NOTHING_INTERESTING, 0, &msg));
        return rs_mes(w, msg);
    }
    return rs_run_held(w, script, arg, name);
}

/* Agent.dispatch (agent.py), in its order of precedence. Notes are not kept. */
static int dispatch(RsSlot *s, int32_t action) {
    Agent *a = &s->a;
    World *w = &s->w;
    Player *p = &w->p;
    const RsRawObs *obs = &a->obs;
    if (!obs->mask[action]) return 0;   /* 'action not legal in published observation' */
    if (action == RS_A_INV_PREV) { a->inv_page = imax(0, a->inv_page - 1); return 0; }
    if (action == RS_A_INV_NEXT) { a->inv_page = imin(3, a->inv_page + 1); return 0; }
    if (action == RS_A_BANK_PREV) { a->bank_page = imax(0, a->bank_page - 1); return 0; }
    if (action == RS_A_BANK_NEXT) { a->bank_page = imin(a->bank_pages - 1, a->bank_page + 1); return 0; }
    if ((action >= RS_A_WITHDRAW0 && action < RS_A_WITHDRAW0 + RS_N_INV_SEL)
        || (action >= RS_A_DEPOSIT0 && action < RS_A_DEPOSIT0 + RS_N_INV_SEL)) {
        /* Agent.item_transfer (agent.py): no bank in the declared worlds */
        if (p->modal_main == -1 || p->modal_main != w->P->consts->s_bank_main) return 0;   /* 'bank: closed' */
        return RS_FAIL_TEXT(&w->err, RS_E_NOT_PORTED, "agent item_transfer (the bank)");
    }
    if (action >= RS_A_DROP0 && action < RS_A_DROP0 + RS_N_INV_SEL) {
        const RsItem *sel = &obs->inv_sel[action - RS_A_DROP0];
        return matches_inventory(p, sel->slot, sel->id, sel->count) ? drop_slot(w, sel->slot) : 0;
    }
    if (action == RS_A_WAIT) return 0;
    if (action >= RS_A_USE0 && action < RS_A_USE0 + RS_N_INV_SEL) {
        const RsItem *sel = &obs->inv_sel[action - RS_A_USE0];
        if (!matches_inventory(p, sel->slot, sel->id, sel->count)) return 0;   /* 'use: stale selection' */
        a->has_using = 1, a->using_slot = sel->slot, a->using_id = sel->id, a->using_name = sel->name;
        return 0;
    }
    if (a->has_using && action == RS_A_CLOSE_IF) {
        a->has_using = 0, a->using_slot = a->using_id = a->using_name = 0;
        return 0;
    }
    if (a->has_using && action >= RS_A_HELD0 && action < RS_A_HELD0 + RS_N_HELD) {
        int k = action - RS_A_HELD0;
        int32_t using_slot = a->using_slot, using_id = a->using_id;
        a->has_using = 0, a->using_slot = a->using_id = a->using_name = 0;
        if (k >= obs->n_held) return 0;   /* 'use: slot empty' */
        return use_on_held(w, obs->held_sel[k].slot, using_slot, using_id);
    }
    if (click_slot(action) >= 0) return click(s, action);
    if (action >= RS_A_ACT0 && action < RS_A_ACT0 + RS_N_ACT) return act(s, action);
    if (action >= RS_A_HELD0 && action < RS_A_HELD0 + RS_N_HELD) {
        int k = action - RS_A_HELD0;
        if (k >= obs->n_held) return 0;   /* 'held: stale selection' */
        const RsHeld *h = &obs->held_sel[k];
        return matches_inventory(p, h->slot, h->id, h->count) ? op_held(w, h->slot, h->option) : 0;
    }
    if (action == RS_A_DIALOG_CONTINUE) return resume_dialog(w, -1);
    if (action >= RS_A_DIALOG_CHOICE0 && action < RS_A_DIALOG_CHOICE0 + RS_N_DIALOG_CHOICE)
        return resume_dialog(w, action - RS_A_DIALOG_CHOICE0);
    if (action >= RS_A_STYLE0 && action < RS_A_STYLE0 + 4) {
        /* [proc,set_attackstyle], executed protected: not at all while delayed or protected */
        if (!(p->protect || p->delayed)) {
            rs_close_modal(w, 1);
            rs_clear_pending_action(w);
            p->varp[RS_VARP_com_mode] = action - RS_A_STYLE0;
            RS_TRY(rs_player_combat_stat(w));
            a->style_stale = 0;
        }
        return 0;
    }
    if (action == RS_A_CLOSE_IF) {
        if (suspended(p)) return resume_dialog(w, p->n_resume ? 0 : -1);
        rs_close_modal(w, 1);
        return 0;
    }
    return 0;   /* f'unimplemented action {action}' */
}

/* ---- step (agent.py) ------------------------------------------------------------------------------------- */
/* Agent.keep_waiting (agent.py): the dialog is read first, then option_busy runs, always (its side effects, an
   option's end and a refusal, happen even after the terminal), then the hitpoints, in the oracle's order. */
static int keep_waiting(RsSlot *s, int32_t hp_at, int32_t waited, int *keep) {
    const Player *p = &s->w.p;
    int dialog = p->modal != 0 || p->modal_chat != -1 || suspended(p);
    int busy;
    RS_TRY(rs_option_busy(s, &busy));
    *keep = busy && hp_of(&s->w) > 0 && !dialog && hp_of(&s->w) >= hp_at && waited < RS_DECISION_WAIT_TICKS;
    return 0;
}

/* The end of an observation: the mask, the feedback, last_mask_sum (agent.py, 827-830). */
static int publish(RsSlot *s, const RsMeaning *meaning, const RsSnap *before, int running) {
    Agent *a = &s->a;
    RS_TRY(rs_build_mask(s));
    RsSnap after;
    rs_snapshot(s, &after);
    RS_TRY(rs_feedback(s, meaning, before, &after, running));
    int32_t sum = 0;
    for (int i = 0; i < RS_N_ACTIONS; i++) sum += a->obs.mask[i];
    a->last_mask_sum = sum;
    a->has_obs = 1;
    return 0;
}

int rs_agent_step(RsSlot *s, int32_t action) {
    Agent *a = &s->a;
    World *w = &s->w;
    Player *p = &w->p;
    /* _begin (agent.py) */
    a->decision += 1;
    if (a->last_mask_sum > 1) a->charged += 1;
    if (rs_action_family(action) < 0)   /* describe_action's family() raises */
        return RS_FAIL_TEXT(&w->err, RS_E_CONTRACT, "no family for the action");
    RsMeaning meaning;
    rs_describe_action(s, action, &meaning);
    RsSnap before;
    rs_snapshot(s, &before);
    int dead_at_start = hp_of(w) <= 0 || p->dead, busy = 0;
    if (!dead_at_start) RS_TRY(rs_option_busy(s, &busy));
    uint32_t running = a->option.has ? a->option.serial : 0;
    uint8_t kind, k;
    if (dead_at_start) {
        a->option.has = 0;   /* 'terminal: action not dispatched' */
    } else if (busy && (action == RS_A_WAIT
                        || (label_of(action, &kind, &k) && kind == a->option.label && k == a->option.label_k))) {
        /* continue the running option */
    } else {
        if (busy) {
            end_option(a, RS_END_SUPERSEDED, 0);
            a->option.has = 0;
        }
        RS_TRY(dispatch(s, action));
    }
    if (a->option.has && a->option.serial != running) a->option.has_action = 1, a->option.action = action;
    /* the paced chunks (agent.py) */
    int32_t ticks = RS_K_TICKS + pace_ticks(s);
    a->episode_ticks += ticks;
    if (!dead_at_start) RS_TRY(wait_ticks(s, ticks));
    if (a->decision_points && !dead_at_start) {
        int32_t hp_at = hp_of(w), waited = 0;
        for (;;) {
            int keep;
            RS_TRY(keep_waiting(s, hp_at, waited, &keep));
            if (!keep || truncated(a) || rs_ended(s)) break;
            int32_t chunk = RS_K_TICKS + pace_ticks(s);
            a->episode_ticks += chunk;
            waited += chunk;
            RS_TRY(wait_ticks(s, chunk));
        }
    }
    /* _finish (agent.py) */
    int dead = hp_of(w) <= 0 || p->dead;
    if (dead) a->option.has = 0;
    int trunc = !dead && truncated(a);
    RS_TRY(rs_build_obs(s, a->decision, dead || trunc, trunc, dead));
    int busy_after = 0;
    if (!dead) RS_TRY(rs_option_busy(s, &busy_after));
    RS_TRY(rs_project_items(s));
    a->obs.option_busy = (uint8_t)busy_after;
    a->obs.option_action = a->option.has && a->option.has_action ? a->option.action : -1;
    return publish(s, &meaning, &before, busy_after);
}

/* ---- start (agent.py, 296-359) --------------------------------------------------------------------------- */
/* Agent.set_level (agent.py): Player.setLevel. */
static void set_level(const RsPack *P, Player *p, int stat, int32_t level) {
    level = level < 1 ? 1 : level > 99 ? 99 : level;
    p->base_levels[stat] = (int16_t)level;
    p->stats[stat] = level >= 2 ? P->consts->level_xp[level - 2] : 0;   /* xp_for (engine.py) */
    p->levels[stat] = (int16_t)level;
}

/* Agent._login (agent.py): the agent's reset of its cfg and counters, a new Player at LOGIN_TILE admitted by
   the next cycle's process_logins (which runs [login,_], rs_agent_login), then two ticks. */
static int login(RsSlot *s, int64_t seed) {
    Agent *a = &s->a;
    World *w = &s->w;
    const RsConfig *cfg = s->cfg;
    a->decision_points = (uint8_t)(cfg->decision_points != 0);
    a->max_decisions = cfg->max_decisions, a->max_episode_ticks = cfg->max_episode_ticks;
    a->pace_jitter = cfg->pace_jitter;
    a->pace_seed = seed, a->pace = 0;
    a->inv_page = a->bank_page = 0;
    a->bank_pages = 1;
    a->has_using = 0, a->using_slot = a->using_id = a->using_name = 0;
    a->n_refused = 0;
    a->has_obs = 0;
    a->decision = a->charged = a->episode_ticks = a->last_mask_sum = 0;
    a->option.has = 0;
    a->option_end.has = 0;
    a->death_tick = -1;
    if (w->has_player) rs_occupy(w, w->p.level, w->p.x, w->p.z, 1, 0);
    rs_player_init(w);
    w->p.x = RS_LOGIN_X, w->p.z = RS_LOGIN_Z;
    w->p.origin_x = RS_LOGIN_X, w->p.origin_z = RS_LOGIN_Z;
    w->has_player = 0;
    a->n_chat = 0, a->n_texts = 0;
    a->chat_seen = w->chat.captured = w->chat.n;
    w->pending_login = 1;
    return wait_ticks(s, 2);
}

/* Agent._arrive (agent.py): the jump to the spawn, two ticks, then the episode's first observation. */
static int arrive(RsSlot *s, int32_t x, int32_t z) {
    Agent *a = &s->a;
    World *w = &s->w;
    Player *p = &w->p;
    rs_occupy(w, p->level, p->x, p->z, 1, 0);
    p->x = (int16_t)x, p->z = (int16_t)z, p->level = 0;
    rs_occupy(w, p->level, p->x, p->z, 1, 1);
    p->path_n = 0;
    RS_TRY(wait_ticks(s, 2));
    if (rs_source(w->L, p->level, p->x, p->z) < 0) return RS_FAIL0(&w->err, RS_E_SPAWN_OUTSIDE);
    a->n_chat = 0, a->n_texts = 0;
    a->chat_seen = w->chat.captured = w->chat.n;
    clear_visited(a);
    RS_TRY(visit(s));
    RS_TRY(rs_build_obs(s, 0, 0, 0, 0));
    RS_TRY(rs_project_items(s));
    return publish(s, NULL, NULL, 0);
}

int rs_agent_start(RsSlot *s, int64_t seed, int32_t tile_x, int32_t tile_z, const int32_t *kit_obj,
                   const int32_t *kit_count, int n_kit, const int16_t levels[7]) {
    Agent *a = &s->a;
    World *w = &s->w;
    /* Agent.__init__ (agent.py): a new Agent every episode */
    memset(a, 0, sizeof *a);
    a->bank_pages = 1;
    a->death_tick = -1;
    RS_TRY(login(s, seed));
    /* Agent.start (agent.py) */
    Player *p = &w->p;
    a->style_stale = 1;   /* bool(levels): the seven START_LEVELS */
    for (int i = 0; i < 7; i++) set_level(w->P, p, RS_START_LEVEL_SKILL[i], levels[i]);
    for (int i = 0; i < n_kit; i++) (void)rs_inv_add(w->P, p->inv, RS_INV_CAP, 0, kit_obj[i], kit_count[i]);
    (void)rs_inv_add(w->P, p->bank, RS_BANK_CAP, 1, w->P->consts->obj_coins, 25);
    return arrive(s, tile_x, tile_z);
}
