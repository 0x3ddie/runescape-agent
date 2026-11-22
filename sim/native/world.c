/* sim/native/world.c: the world core of sim/engine.py, World and Player (engine.py, 1404-1830):
 * World init and the nine cycle phases, collision flags and occupancy, can_travel / take_step / process_movement,
 * locs, zones and loc_events with the engine's reorder rules and the loc lifecycle, the objs list and its despawn,
 * Inventory, the three player queues and the npc queues with _links' visit quirk, the script runner (execute, _run,
 * resume), timers, the interaction machinery (validate_target, in_operable, in_approach, try_interact, default_op,
 * run_trigger, path_to_pathing_target, naive_path, process_interaction, update_movement), the stat ops, add_xp,
 * mes and settext, and the canonical record of the world's state. Each function names the Python it
 * mirrors (file and function, as rs.h). Content scripts (scripts.c), npc turns and combat (npc.c,
 * combat.c), the agent's login (agent.c) and path queries (paths.c) are called through rs.h only.
 *
 * Exactness notes the code relies on:
 *   - Every Python object identity is a slot plus a serial (rs.h IDENTITY); loc_at, zones and loc_events hold slots.
 *   - A queue pass (rs_links_*) reads a node's successor when it visits the node; a node no longer in its list at its
 *     visit ends the pass after its body; unlinked requests are freed only when a pass ends or a cycle starts, so a
 *     captured successor never names a reused slot. Passes never nest (no script iterates a queue).
 *   - The runner steps a copy of the frame and stores it into p.script only at a suspension, so a frame that a nested
 *     execute replaced is compared by serial, as Python compares generators by identity (engine.py). */
#include <string.h>

#include "rs.h"

/* ---- small helpers ----------------------------------------------------------------------------------------------- */
static inline int32_t imax(int32_t a, int32_t b) { return a > b ? a : b; }
static inline int32_t imin(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t iabs(int32_t a) { return a < 0 ? -a : a; }
static inline int32_t sgn(int32_t v) { return (v > 0) - (v < 0); }

/* level_for (engine.py) over RsConsts.level_xp (never pow). */
static int32_t level_for(const RsPack *P, int64_t xp) {
    const int32_t *t = P->consts->level_xp;
    for (int i = 98; i >= 0; i--)
        if (xp >= t[i]) return imin(i + 2, 99);
    return 1;
}

/* paths.shape_layer (paths.py). */
static int32_t shape_layer(int32_t shape) { return shape <= 3 ? 0 : shape <= 8 ? 1 : shape == 22 ? 3 : 2; }

/* ---- the targets an interaction names (npc, loc, obj) ------------------------------------------------------------- */
typedef struct {
    int kind;                  /* RS_REF_NPC, _LOC, _OBJ */
    int32_t x, z, level, type, width, length, angle, valid;
    Npc *npc; Loc *loc; Obj *obj;
} Target;

/* The entity an RsRef names, as Python holds it; 0 when there is none (no ref, or its slot no longer holds it, which
   only happens to objs that left world.objs: is_valid() is False there, and the caller reads it so). */
static int target_of(World *w, RsRef r, Target *t) {
    memset(t, 0, sizeof *t);
    t->kind = r.kind;
    if (r.kind == RS_REF_NPC) {
        if (r.slot >= w->n_npcs) return 0;
        Npc *n = t->npc = &w->npcs[r.slot];
        t->x = n->x, t->z = n->z, t->level = n->level, t->type = n->type, t->width = t->length = n->size;
        t->valid = n->active && !n->delayed;                       /* Npc.is_valid (engine.py) */
        return 1;
    }
    if (r.kind == RS_REF_LOC) {
        Loc *l = t->loc = rs_get_loc(w, r);
        if (!l) return 0;
        t->x = l->x, t->z = l->z, t->level = l->level, t->type = l->type, t->width = l->width, t->length = l->length;
        t->angle = l->angle, t->valid = l->active;                 /* Loc.is_valid (engine.py) */
        return 1;
    }
    if (r.kind == RS_REF_OBJ) {
        Obj *o = t->obj = rs_get_obj(w, r);
        if (!o) return 0;
        t->x = o->x, t->z = o->z, t->level = o->level, t->type = o->type, t->width = t->length = 1;
        t->valid = o->active && o->count >= 1;                     /* Obj.is_valid (engine.py) */
        return 1;
    }
    return 0;
}

/* _distance_to (engine.py): Chebyshev distance from the player's tile to the target's rectangle, a loc's
   rotated on odd angles. */
static int32_t distance_to(const Player *p, const Target *t) {
    int32_t w = t->width, l = t->length;
    if (t->kind == RS_REF_LOC && (t->angle & 1)) { int32_t s = w; w = l; l = s; }
    int32_t dx = imax(imax(t->x - p->x, p->x - (t->x + w - 1)), 0);
    int32_t dz = imax(imax(t->z - p->z, p->z - (t->z + l - 1)), 0);
    return imax(dx, dz);
}

/* _nearest_tile (engine.py). */
static void nearest_tile(const Player *p, const Target *t, int32_t *x, int32_t *z) {
    int32_t w = t->width, l = t->length;
    if (t->kind == RS_REF_LOC && (t->angle & 1)) { int32_t s = w; w = l; l = s; }
    *x = imin(imax(p->x, t->x), t->x + w - 1);
    *z = imin(imax(p->z, t->z), t->z + l - 1);
}

/* World.trigger (engine.py): the script of trigger `op` (an AP or OP trigger id of the target's kind), from
   the type's trigger slots (rs.h: AP k at RS_TRIG_AP + k, OP k at RS_TRIG_OP + k). */
static int trigger_of(World *w, const Target *t, int32_t op, int16_t *script, int16_t *arg, int32_t *name) {
    const int16_t *s = NULL, *a = NULL;
    const int32_t *nm = NULL;
    int32_t base;
    if (t->kind == RS_REF_NPC) {
        const RsNpcType *nt = rs_npc_type(w->P, t->type);
        if (!nt) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an npc of a type the pack lacks");
        s = nt->trig_script, a = nt->trig_arg, nm = nt->trig_name, base = RS_APNPC1;
    } else if (t->kind == RS_REF_LOC) {
        const RsLocType *lt = rs_loc_type(w->P, t->type);
        if (!lt) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a loc of a type the pack lacks");
        s = lt->trig_script, a = lt->trig_arg, nm = lt->trig_name, base = RS_APLOC1;
    } else {
        const RsObjType *ot = rs_obj_type(w->P, t->type);
        if (!ot) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an obj of a type the pack lacks");
        s = ot->trig_script, a = ot->trig_arg, nm = ot->trig_name, base = RS_APOBJ1;
    }
    int32_t k = op - base, i;
    if (k >= 0 && k < 6) i = RS_TRIG_AP + k;
    else if (k - RS_AP_TO_OP >= 0 && k - RS_AP_TO_OP < 6) i = RS_TRIG_OP + k - RS_AP_TO_OP;
    else return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an interaction op outside its target's triggers");
    *script = s[i], *arg = a[i], *name = nm[i];
    return 0;
}

/* ---- World.__init__ and Player.__init__ ---------------------------------------------------------------------------- */
static void reqs_init(World *w) {
    for (int i = 0; i < RS_MAX_REQS; i++) {
        w->req[i] = (Req){.prev = -1, .next = (int16_t)(i + 1 < RS_MAX_REQS ? i + 1 : -1), .dead_next = -1};
    }
    w->req_free = 0;
    w->req_dead = -1;
}

static int zone_index(const World *w, int32_t level, int32_t x, int32_t z) {
    const RsLayout *L = w->L;
    if (level != 0 || x < L->x0 || x >= L->x0 + L->W || z < L->z0 || z >= L->z0 + L->H) return -1;
    return ((x - L->x0) >> 3) * ((L->H + 7) >> 3) + ((z - L->z0) >> 3);
}

static int zone_append(World *w, int zi, int slot) {
    if (zi < 0) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a loc outside every zone");
    if (w->zone_n[zi] >= RS_ZONE_CAP) return RS_OVERFLOW(&w->err, "RS_ZONE_CAP locs in one zone");
    w->zone[zi][w->zone_n[zi]++] = (uint8_t)slot;
    return 0;
}

/* list.remove by identity: the first entry (a loc is listed at most once); 0 when absent. */
static int zone_remove(World *w, int zi, int slot) {
    if (zi < 0) return 0;
    uint8_t *z = w->zone[zi];
    int n = w->zone_n[zi];
    for (int i = 0; i < n; i++) {
        if (z[i] != slot) continue;
        memmove(z + i, z + i + 1, (size_t)(n - i - 1));
        w->zone_n[zi]--;
        return 1;
    }
    return 0;
}

void rs_player_init(World *w) {
    Player *p = &w->p;
    memset(p, 0, sizeof *p);
    p->target = p->next_target = (RsRef){0};
    p->target_op = p->target_type = -1;
    p->ap_range = 10;
    p->allow_repath = 1;
    p->delayed_until = -1;
    p->modal_main = p->modal_chat = -1;
    p->queue = p->weak = p->engine = (ReqList){-1, -1};
    p->stats[RS_SKILL_HITPOINTS] = w->P->consts->level_xp[8];   /* xp_for(10) */
    for (int i = 0; i < RS_N_SKILLS; i++) p->levels[i] = p->base_levels[i] = 1;
    p->levels[RS_SKILL_HITPOINTS] = p->base_levels[RS_SKILL_HITPOINTS] = 10;
    for (int i = 0; i < RS_INV_CAP; i++) p->inv[i] = (Slot){-1, 0};
    for (int i = 0; i < RS_WORN_CAP; i++) p->worn[i] = (Slot){-1, 0};
    for (int i = 0; i < RS_BANK_CAP; i++) p->bank[i] = (Slot){-1, 0};
    p->varp[RS_VARP_tutorial] = 1000;
    p->varp[RS_VARP_aggressive_npc] = -1;
    p->last_useitem = p->last_com_iface = p->last_com_com = -1;
    p->last_item = p->last_slot = p->last_useslot = -1;
}

int rs_world_init(World *w, const RsPack *P, const RsLayout *L, int64_t seed, int32_t start_tick,
                  const RsNpcSpawn *spawns, int n_spawns, uint8_t exclusions, int32_t xp_rate) {
    memset(w, 0, sizeof *w);
    w->P = P;
    w->L = L;
    if (xp_rate < 1) return RS_FAIL_TEXT(&w->err, RS_E_CONTRACT, "xp_rate: a positive whole multiplier");
    if (exclusions & ~(RS_EXCLUDE_RANDOM_EVENTS | RS_EXCLUDE_CLUE_DROPS | RS_EXCLUDE_LOOT_DROPS))
        return RS_FAIL_TEXT(&w->err, RS_E_CONTRACT, "unknown exclusions");
    w->xp_rate = xp_rate;
    w->excl = exclusions;
    w->loot = RS_OF_LOOT_OUTSIDE;  /* World(loot_outside=None): LOOT_OUTSIDE; a village life of its own list sets OWN */
    w->seed = rs_seed(seed);
    w->tick = start_tick;
    w->members = (uint8_t)(P->consts->members != 0);
    reqs_init(w);
    /* the map locs in table order, each in its tile's and its zone's list (engine.py) */
    if (L->n_locs > RS_MAX_LOCS) return RS_OVERFLOW(&w->err, "RS_MAX_LOCS map locs");
    for (int i = 0; i < L->n_locs; i++) {
        const int32_t *r = &L->locs[i * 5];
        const RsLocType *lt = rs_loc_type(P, r[0]);
        if (!lt) return RS_FAIL_TEXT(&w->err, RS_E_PACK, "a map loc of a type the pack lacks");
        Loc *l = &w->locs[i];
        *l = (Loc){.index = i, .serial = ++w->next_serial, .base = (int16_t)r[0], .type = (int16_t)r[0],
                   .x = (int16_t)(L->x0 + r[1]), .z = (int16_t)(L->z0 + r[2]), .shape = (uint8_t)r[3],
                   .angle = (uint8_t)r[4], .width = (uint8_t)lt->width, .length = (uint8_t)lt->length,
                   .base_shape = (uint8_t)r[3], .base_angle = (uint8_t)r[4], .layer = (uint8_t)shape_layer(r[3]),
                   .level = 0, .is_static = 1, .active = 1, .lifecycle = -1};
        RS_TRY(zone_append(w, zone_index(w, l->level, l->x, l->z), i));
    }
    w->n_loc_slots = w->n_stored_locs = w->next_loc = L->n_locs;
    /* npcs sorted by nid (engine.py), [ai_spawn] at boot, then their occupancy (515-516) */
    RsNpcSpawn own[RS_MAX_NPCS];
    if (!spawns) {
        if (L->n_npcs > RS_MAX_NPCS) return RS_OVERFLOW(&w->err, "RS_MAX_NPCS npcs");
        for (int i = 0; i < L->n_npcs; i++) {
            const int32_t *r = &L->npcs[i * 4];
            const RsNpcType *nt = rs_npc_type(P, r[1]);
            if (!nt) return RS_FAIL_TEXT(&w->err, RS_E_PACK, "a layout npc of a type the pack lacks");
            own[i] = (RsNpcSpawn){r[0], r[1], L->x0 + r[2], L->z0 + r[3], 0, nt->size};
        }
        spawns = own;
        n_spawns = L->n_npcs;
    }
    if (n_spawns > RS_MAX_NPCS) return RS_OVERFLOW(&w->err, "RS_MAX_NPCS npcs");
    int order[RS_MAX_NPCS];
    for (int i = 0; i < n_spawns; i++) {      /* sorted() by nid: stable insertion */
        int j = i;
        while (j > 0 && spawns[order[j - 1]].nid > spawns[i].nid) { order[j] = order[j - 1]; j--; }
        order[j] = i;
    }
    for (int i = 0; i < n_spawns; i++) {
        const RsNpcSpawn *s = &spawns[order[i]];
        const RsNpcType *nt = rs_npc_type(P, s->type);
        if (!nt) return RS_FAIL_TEXT(&w->err, RS_E_PACK, "an npc spawn of a type the pack lacks");
        Npc *n = &w->npcs[i];
        *n = (Npc){.nid = (int16_t)s->nid, .base = (int16_t)s->type, .type = (int16_t)s->type, .x = (int16_t)s->x,
                   .z = (int16_t)s->z, .start_x = (int16_t)s->x, .start_z = (int16_t)s->z, .size = (int16_t)s->size,
                   .level = (uint8_t)s->level, .active = 1, .delayed_until = -1, .mode = nt->defaultmode,
                   .timer_interval = nt->timer, .hunt_mode = nt->huntmode, .huntrange = nt->huntrange,
                   .lifecycle = -1, .last_movement = -1, .queue = {-1, -1}};
        for (int k = 0; k < 6; k++) n->levels[k] = nt->stats[k];
        rs_npc_spawned(w, n);
    }
    w->n_npcs = (uint8_t)n_spawns;
    for (int i = 0; i < w->n_npcs; i++)
        rs_occupy(w, w->npcs[i].level, w->npcs[i].x, w->npcs[i].z, w->npcs[i].size, 1);
    memcpy(w->flags, L->flags_boot, (size_t)L->T * sizeof *w->flags);
    rs_player_init(w);   /* no player yet (has_player 0): Agent._login makes the one the episode uses */
    rs_paths_reset(w);
    return 0;
}

/* ---- collision and movement (engine.py) ------------------------------------------------------------------ */
int32_t rs_flags(const World *w, int32_t level, int32_t x, int32_t z, int npc_bits) {
    int32_t t = rs_tile(w->L, level, x, z);
    if (t < 0) return RS_FLAG_NULL;
    int32_t f = w->flags[t];
    if (npc_bits && w->occ[t]) f |= RS_FLAG_NPC;
    return f;
}

void rs_occupy(World *w, int32_t level, int32_t x, int32_t z, int32_t size, int add) {
    for (int32_t dx = 0; dx < size; dx++)
        for (int32_t dz = 0; dz < size; dz++) {
            int32_t t = rs_tile(w->L, level, x + dx, z + dz);
            if (t >= 0) w->occ[t] = (int8_t)(w->occ[t] + (add ? 1 : -1));
        }
}

/* World.can_move (engine.py). */
static int can_move(World *w, int32_t collision, int32_t flag, int32_t block, int *ok) {
    uint32_t f = (uint32_t)flag, b = (uint32_t)block;
    switch (collision) {
    case 0: *ok = (f & b) == 0; return 0;
    case 1: *ok = (f & (b & ~(uint32_t)0x200000)) == 0 && (f & 0x200000u) != 0; return 0;
    case 3: *ok = (f & b) == 0 && (f & 0x80000000u) != 0; return 0;
    case 4: *ok = (f & (b | 0x80000000u)) == 0; return 0;
    default: return RS_FAIL(&w->err, RS_E_COLLISION_TYPE, collision);
    }
}

/* One `ok(a, b, mask)` of can_travel; *ok is the conjunction so far (Python's `and` stops at the first False). */
static int step_ok(World *w, int32_t level, int32_t x, int32_t z, int32_t mask, int32_t extra, int32_t collision,
                   int *ok) {
    if (!*ok) return 0;
    return can_move(w, collision, rs_flags(w, level, x, z, 1), mask | extra, ok);
}

int rs_can_travel(World *w, int32_t level, int32_t x, int32_t z, int32_t dx, int32_t dz, int32_t size, int32_t extra,
                  int32_t collision, int *ok) {
#define OK(a, b, m) RS_TRY(step_ok(w, level, (a), (b), (m), extra, collision, ok))
    *ok = 1;
    if (size == 1) {
        if (dx == 0 && dz == -1) { OK(x, z - 1, RS_BLOCK_S); return 0; }
        if (dx == 0 && dz == 1) { OK(x, z + 1, RS_BLOCK_N); return 0; }
        if (dx == -1 && dz == 0) { OK(x - 1, z, RS_BLOCK_W); return 0; }
        if (dx == 1 && dz == 0) { OK(x + 1, z, RS_BLOCK_E); return 0; }
        if (dx == -1 && dz == -1) { OK(x - 1, z - 1, RS_BLOCK_SW); OK(x - 1, z, RS_BLOCK_W); OK(x, z - 1, RS_BLOCK_S);
                                    return 0; }
        if (dx == -1 && dz == 1) { OK(x - 1, z + 1, RS_BLOCK_NW); OK(x - 1, z, RS_BLOCK_W); OK(x, z + 1, RS_BLOCK_N);
                                   return 0; }
        if (dx == 1 && dz == -1) { OK(x + 1, z - 1, RS_BLOCK_SE); OK(x + 1, z, RS_BLOCK_E); OK(x, z - 1, RS_BLOCK_S);
                                   return 0; }
        if (dx == 1 && dz == 1) { OK(x + 1, z + 1, RS_BLOCK_NE); OK(x + 1, z, RS_BLOCK_E); OK(x, z + 1, RS_BLOCK_N);
                                  return 0; }
        *ok = 0;
        return 0;
    }
    if (size == 2) {
        if (dx == 0 && dz == -1) { OK(x, z - 1, RS_BLOCK_SW); OK(x + 1, z - 1, RS_BLOCK_SE); return 0; }
        if (dx == 0 && dz == 1) { OK(x, z + 2, RS_BLOCK_NW); OK(x + 1, z + 2, RS_BLOCK_NE); return 0; }
        if (dx == -1 && dz == 0) { OK(x - 1, z, RS_BLOCK_SW); OK(x - 1, z + 1, RS_BLOCK_NW); return 0; }
        if (dx == 1 && dz == 0) { OK(x + 2, z, RS_BLOCK_SE); OK(x + 2, z + 1, RS_BLOCK_NE); return 0; }
        if (dx == -1 && dz == -1) { OK(x - 1, z, RS_BLOCK_N_AND_SW); OK(x - 1, z - 1, RS_BLOCK_SW);
                                    OK(x, z - 1, RS_BLOCK_NE_AND_W); return 0; }
        if (dx == -1 && dz == 1) { OK(x - 1, z + 1, RS_BLOCK_N_AND_SW); OK(x - 1, z + 2, RS_BLOCK_NW);
                                   OK(x, z + 2, RS_BLOCK_SE_AND_W); return 0; }
        if (dx == 1 && dz == -1) { OK(x + 1, z - 1, RS_BLOCK_NE_AND_W); OK(x + 2, z - 1, RS_BLOCK_SE);
                                   OK(x + 2, z, RS_BLOCK_N_AND_SE); return 0; }
        if (dx == 1 && dz == 1) { OK(x + 1, z + 2, RS_BLOCK_SE_AND_W); OK(x + 2, z + 2, RS_BLOCK_NE);
                                  OK(x + 2, z + 1, RS_BLOCK_N_AND_SE); return 0; }
        *ok = 0;
        return 0;
    }
#undef OK
    *ok = 0;
    return RS_FAIL(&w->err, RS_E_NPC_SIZE, size);
}

/* World.take_step (engine.py): toward the waypoint, the diagonal (size 1 only), then x, then z. */
static int take_step(World *w, int32_t level, int32_t x, int32_t z, int32_t tx, int32_t tz, int32_t size,
                     int32_t extra, int32_t collision, int32_t *sdx, int32_t *sdz) {
    int32_t dx = sgn(tx - x), dz = sgn(tz - z);
    int ok;
    *sdx = *sdz = 0;
    if (size == 1) {
        RS_TRY(rs_can_travel(w, level, x, z, dx, dz, size, extra, collision, &ok));
        if (ok) { *sdx = dx, *sdz = dz; return 0; }
    }
    if (dx != 0) {
        RS_TRY(rs_can_travel(w, level, x, z, dx, 0, size, extra, collision, &ok));
        if (ok) { *sdx = dx; return 0; }
    }
    if (dz != 0) {
        RS_TRY(rs_can_travel(w, level, x, z, 0, dz, size, extra, collision, &ok));
        if (ok) { *sdz = dz; return 0; }
    }
    return 0;
}

/* World.teleport (engine.py): PathingEntity.teleport on the player's level, refused off the supported tiles;
   the occupancy follows the player, the waypoints stay; no moveSpeed INSTANT (that is a teleJump's), no step taken and
   no last_movement. */
int rs_teleport(World *w, int32_t x, int32_t z) {
    Player *p = &w->p;
    if (rs_source(w->L, p->level, x, z) < 0) return RS_FAIL(&w->err, RS_E_TELEPORT_OUTSIDE, p->level, x, z);
    rs_occupy(w, p->level, p->x, p->z, 1, 0);
    p->x = (int16_t)x, p->z = (int16_t)z;
    rs_occupy(w, p->level, p->x, p->z, 1, 1);
    return 0;
}

/* World.process_movement (engine.py) for the player: one step toward path[0], which is popped once reached. */
int rs_process_movement_player(World *w) {
    Player *p = &w->p;
    if (!p->path_n) return 0;
    int32_t dx, dz, sx = p->x, sz = p->z;
    RS_TRY(take_step(w, p->level, p->x, p->z, p->path[0][0], p->path[0][1], 1, 0, 0, &dx, &dz));
    if (dx || dz) {
        rs_occupy(w, p->level, p->x, p->z, 1, 0);
        p->x = (int16_t)(p->x + dx), p->z = (int16_t)(p->z + dz);
        rs_occupy(w, p->level, p->x, p->z, 1, 1);
    }
    if (p->path_n && p->path[0][0] == p->x && p->path[0][1] == p->z) {
        p->path_n--;
        memmove(p->path[0], p->path[1], (size_t)p->path_n * sizeof p->path[0]);
    }
    if (p->x != sx || p->z != sz) p->steps_taken++;
    return 0;
}

/* The same for an npc, whose path is at most one waypoint. */
int rs_process_movement_npc(World *w, Npc *n, int32_t collision, int32_t extra) {
    if (!n->has_path) return 0;
    int32_t dx, dz;
    RS_TRY(take_step(w, n->level, n->x, n->z, n->path_x, n->path_z, n->size, extra, collision, &dx, &dz));
    if (dx || dz) {
        rs_occupy(w, n->level, n->x, n->z, n->size, 0);
        n->x = (int16_t)(n->x + dx), n->z = (int16_t)(n->z + dz);
        rs_occupy(w, n->level, n->x, n->z, n->size, 1);
    }
    if (n->has_path && n->path_x == n->x && n->path_z == n->z) n->has_path = 0;
    return 0;
}

/* ---- locs (engine.py) ---------------------------------------------------------------------------------- */
static int loc_zone(const World *w, const Loc *l) { return zone_index(w, l->level, l->x, l->z); }
static int loc_changed(const Loc *l) {                               /* Loc.is_changed (engine.py) */
    return l->type != l->base || l->shape != l->base_shape || l->angle != l->base_angle;
}
static void loc_revert(Loc *l) { l->type = l->base, l->shape = l->base_shape, l->angle = l->base_angle; }

/* loc_events.pop(loc, None) */
static void loc_events_pop(World *w, int slot) {
    for (int i = 0; i < w->n_loc_events; i++) {
        if (w->loc_events[i] != slot) continue;
        memmove(w->loc_events + i, w->loc_events + i + 1, (size_t)(w->n_loc_events - i - 1));
        w->n_loc_events--;
        return;
    }
}

/* NonPathingEntity.setLifeCycle (engine.py): re-tracking moves the loc to the tail. */
void rs_set_lifecycle(World *w, int slot, int32_t duration) {
    loc_events_pop(w, slot);
    if (duration > 0) {
        w->loc_events[w->n_loc_events++] = (uint8_t)slot;   /* at most one entry per slot: never past RS_MAX_LOCS */
        w->locs[slot].lifecycle = duration;
    } else {
        w->locs[slot].lifecycle = -1;
    }
}

/* Zone.changeLoc (engine.py): active again, moved to its zone's end. */
static int zone_changed(World *w, int slot) {
    Loc *l = &w->locs[slot];
    l->active = 1;
    int zi = loc_zone(w, l);
    zone_remove(w, zi, slot);
    RS_TRY(zone_append(w, zi, slot));
    w->loc_version++;
    return 0;
}

/* World.add_loc (engine.py): the collision of the loc as it is, then the revert. */
int rs_add_loc(World *w, int slot, int32_t duration) {
    Loc *l = &w->locs[slot];
    RS_TRY(rs_loc_collision(w, l, 1));
    if (!l->is_static) RS_TRY(zone_append(w, loc_zone(w, l), slot));
    loc_revert(l);
    l->active = 1;
    w->loc_version++;
    rs_set_lifecycle(w, slot, duration);
    return 0;
}

/* World.remove_loc (engine.py): a map loc goes to its zone's end; an added one leaves its zone and tile. */
int rs_remove_loc(World *w, int slot, int32_t duration) {
    Loc *l = &w->locs[slot];
    if (!l->active) return 0;
    RS_TRY(rs_loc_collision(w, l, 0));
    int zi = loc_zone(w, l);
    zone_remove(w, zi, slot);
    if (l->is_static) RS_TRY(zone_append(w, zi, slot));   /* else it leaves loc_at: loc_at lists added locs while active */
    l->active = 0;
    w->loc_version++;
    rs_set_lifecycle(w, slot, l->is_static ? duration : -1);
    return 0;
}

/* World.change_loc (engine.py). */
int rs_change_loc(World *w, int slot, int32_t type, int32_t shape, int32_t angle, int32_t duration) {
    Loc *l = &w->locs[slot];
    if (!l->is_static && !l->active) return 0;
    if (l->active) RS_TRY(rs_loc_collision(w, l, 0));
    l->type = (int16_t)type, l->shape = (uint8_t)shape, l->angle = (uint8_t)angle;
    RS_TRY(rs_loc_collision(w, l, 1));
    RS_TRY(zone_changed(w, slot));
    rs_set_lifecycle(w, slot, loc_changed(l) || !l->is_static ? duration : -1);
    return 0;
}

/* World.revert_loc (engine.py). */
int rs_revert_loc(World *w, int slot) {
    Loc *l = &w->locs[slot];
    RS_TRY(rs_loc_collision(w, l, 0));
    loc_revert(l);
    RS_TRY(rs_loc_collision(w, l, 1));
    RS_TRY(zone_changed(w, slot));
    rs_set_lifecycle(w, slot, -1);
    return 0;
}

/* A slot for a new added loc: a free one past the map locs (rs_world_cycle reclaims them), else a new one. */
static int loc_slot(World *w, int *slot) {
    for (int s = w->n_stored_locs; s < w->n_loc_slots; s++)
        if (!w->locs[s].serial) { *slot = s; return 0; }
    if (w->n_loc_slots >= RS_MAX_LOCS) return RS_OVERFLOW(&w->err, "RS_MAX_LOCS locs (map, live and referenced)");
    *slot = w->n_loc_slots++;
    return 0;
}

/* LOC_ADD (engine.py): the first loc of the zone on that tile and layer (gone or not) is changed, else a new
   one (index _next_loc) is added. */
int rs_loc_add(World *w, int32_t level, int32_t x, int32_t z, int32_t type, int32_t angle, int32_t shape,
               int32_t duration, int *slot) {
    const RsLayout *L = w->L;
    if (!(L->x0 <= x && x <= L->x0 + L->W - 1 && L->z0 <= z && z <= L->z0 + L->H - 1))
        return RS_FAIL(&w->err, RS_E_LOC_ADD_OUTSIDE, x, z);
    int32_t layer = shape_layer(shape);
    int zi = zone_index(w, level, x, z);
    if (zi >= 0) {
        uint8_t snap[RS_ZONE_CAP];
        int n = w->zone_n[zi];
        memcpy(snap, w->zone[zi], (size_t)n);
        for (int i = 0; i < n; i++) {
            Loc *l = &w->locs[snap[i]];
            if (l->x == x && l->z == z && l->layer == layer) {
                *slot = snap[i];
                return rs_change_loc(w, snap[i], type, shape, angle, duration);
            }
        }
    }
    const RsLocType *lt = rs_loc_type(w->P, type);
    if (!lt) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "LOC_ADD of a type the pack lacks");
    int s;
    RS_TRY(loc_slot(w, &s));
    w->locs[s] = (Loc){.index = w->next_loc++, .serial = ++w->next_serial, .base = (int16_t)type, .type = (int16_t)type,
                       .x = (int16_t)x, .z = (int16_t)z, .shape = (uint8_t)shape, .angle = (uint8_t)angle,
                       .width = (uint8_t)lt->width, .length = (uint8_t)lt->length, .base_shape = (uint8_t)shape,
                       .base_angle = (uint8_t)angle, .layer = (uint8_t)layer, .level = (uint8_t)level,
                       .is_static = 0, .active = 1, .lifecycle = -1};
    *slot = s;
    return rs_add_loc(w, s, duration);
}

int rs_locs_in_zone(const World *w, int32_t level, int32_t x, int32_t z, uint8_t out[RS_ZONE_CAP]) {
    int zi = zone_index(w, level, x, z), n = 0;
    if (zi < 0) return 0;
    for (int i = w->zone_n[zi] - 1; i >= 0; i--)
        if (w->locs[w->zone[zi][i]].active) out[n++] = w->zone[zi][i];
    return n;
}

/* Loc slots matching a filter, in Loc.index order (index is unique). */
static int sort_by_index(const World *w, uint8_t *s, int n) {
    for (int i = 1; i < n; i++) {
        uint8_t v = s[i];
        int j = i;
        while (j > 0 && w->locs[s[j - 1]].index > w->locs[v].index) { s[j] = s[j - 1]; j--; }
        s[j] = v;
    }
    return n;
}

/* World.locs_near (engine.py). Every active loc is in its zone, and the zones scanned hold every loc within
   the radius, so filtering all slots then sorting by index gives the same list. */
int rs_locs_near(const World *w, int32_t level, int32_t x, int32_t z, int32_t radius, uint8_t out[RS_MAX_LOCS]) {
    int n = 0;
    for (int s = 0; s < w->n_loc_slots; s++) {
        const Loc *l = &w->locs[s];
        if (l->serial && l->active && l->level == level && imax(iabs(l->x - x), iabs(l->z - z)) <= radius)
            out[n++] = (uint8_t)s;
    }
    return sort_by_index(w, out, n);
}

/* loc_at (engine.py, 879, 923): map locs always, added locs until removed (while active), in index order. */
int rs_loc_at(const World *w, int32_t level, int32_t x, int32_t z, uint8_t out[RS_MAX_LOCS]) {
    int n = 0;
    for (int s = 0; s < w->n_loc_slots; s++) {
        const Loc *l = &w->locs[s];
        if (l->serial && (l->is_static || l->active) && l->level == level && l->x == x && l->z == z)
            out[n++] = (uint8_t)s;
    }
    return sort_by_index(w, out, n);
}

int rs_active_locs_at(const World *w, int32_t level, int32_t x, int32_t z, uint8_t out[RS_MAX_LOCS]) {
    int n = 0;
    for (int s = 0; s < w->n_loc_slots; s++) {
        const Loc *l = &w->locs[s];
        if (l->serial && l->active && l->level == level && l->x == x && l->z == z) out[n++] = (uint8_t)s;
    }
    return sort_by_index(w, out, n);
}

/* World.loc_param(loc, 'next_loc_stage') (engine.py). */
int rs_loc_next_stage(World *w, const Loc *loc, int32_t *stage) {
    const RsLocType *lt = rs_loc_type(w->P, loc->type);
    if (!lt) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a loc of a type the pack lacks");
    if (lt->next_loc_stage == RS_NONE) {
        RS_FAIL(&w->err, RS_E_LOC_PARAM, lt->debugname);
        w->err.text = "next_loc_stage";
        return 1;
    }
    *stage = lt->next_loc_stage;
    return 0;
}

/* Loc slots that no holder names (rs.h IDENTITY), reclaimed at the start of a cycle. */
static int ref_names(RsRef r, int kind, int slot, uint32_t serial) {
    return r.kind == kind && r.slot == slot && r.serial == serial;
}
static int frame_names(const Frame *f, int kind, int slot, uint32_t serial) {
    if (!f->serial) return 0;
    for (int d = 0; d < f->depth && d < RS_FRAME_DEPTH; d++)
        if (ref_names(f->ref[d], kind, slot, serial)) return 1;
    return 0;
}
static void reclaim_locs(World *w) {
    for (int s = w->n_stored_locs; s < w->n_loc_slots; s++) {
        Loc *l = &w->locs[s];
        if (!l->serial || l->is_static || l->active) continue;
        int held = ref_names(w->p.target, RS_REF_LOC, s, l->serial) || ref_names(w->p.next_target, RS_REF_LOC, s,
            l->serial) || frame_names(&w->p.script, RS_REF_LOC, s, l->serial);
        for (int i = 0; !held && i < w->n_npcs; i++) held = frame_names(&w->npcs[i].script, RS_REF_LOC, s, l->serial);
        for (int i = 0; !held && i < w->n_loc_events; i++) held = w->loc_events[i] == s;
        int zi = loc_zone(w, l);
        for (int i = 0; !held && zi >= 0 && i < w->zone_n[zi]; i++) held = w->zone[zi][i] == s;
        if (!held) l->serial = 0;
    }
    while (w->n_loc_slots > w->n_stored_locs && !w->locs[w->n_loc_slots - 1].serial) w->n_loc_slots--;
}

/* ---- references and objs ------------------------------------------------------------------------------------------ */
RsRef rs_ref_loc(const World *w, int slot) { return (RsRef){RS_REF_LOC, (uint8_t)slot, 0, w->locs[slot].serial}; }
RsRef rs_ref_obj(const World *w, int slot) { return (RsRef){RS_REF_OBJ, (uint8_t)slot, 0, w->objs[slot].serial}; }
RsRef rs_ref_npc(const World *w, int slot) { (void)w; return (RsRef){RS_REF_NPC, (uint8_t)slot, 0, 0}; }

Loc *rs_get_loc(World *w, RsRef r) {
    if (r.kind != RS_REF_LOC || r.slot >= RS_MAX_LOCS || !r.serial || w->locs[r.slot].serial != r.serial) return NULL;
    return &w->locs[r.slot];
}

Obj *rs_get_obj(World *w, RsRef r) {
    if (r.kind != RS_REF_OBJ || r.slot >= RS_MAX_OBJS || !r.serial || w->objs[r.slot].serial != r.serial) return NULL;
    return &w->objs[r.slot];
}

/* world.objs.append(Obj(...)) (engine.py, 1152). */
int rs_obj_append(World *w, int32_t type, int32_t count, int32_t x, int32_t z, int32_t level, int is_static,
                  int32_t ticks, int receiver_player, int *slot) {
    if (w->n_objs >= RS_MAX_OBJS) return RS_OVERFLOW(&w->err, "RS_MAX_OBJS ground objs");
    int s = 0;
    while (w->objs[s].serial) s++;   /* n_objs < RS_MAX_OBJS slots are taken, so one is free */
    w->objs[s] = (Obj){.serial = ++w->next_serial, .type = (int16_t)type, .x = (int16_t)x, .z = (int16_t)z,
                       .level = (uint8_t)level, .is_static = (uint8_t)(is_static != 0), .active = 1,
                       .receiver_player = (uint8_t)(receiver_player != 0), .count = count, .ticks = ticks};
    w->obj_order[w->n_objs++] = (uint8_t)s;
    *slot = s;
    return 0;
}

/* world.objs.remove(obj) (engine.py, 1164): the slot is freed, so a held RsRef reads NULL (rs.h IDENTITY). */
void rs_obj_remove(World *w, int slot) {
    for (int i = 0; i < w->n_objs; i++) {
        if (w->obj_order[i] != slot) continue;
        memmove(w->obj_order + i, w->obj_order + i + 1, (size_t)(w->n_objs - i - 1));
        w->n_objs--;
        w->objs[slot].serial = 0;
        w->objs[slot].active = 0;
        return;
    }
}

/* ---- Inventory (engine.py) ---------------------------------------------------------------------------- */
/* Counts are int32 in the slots (STACK_LIMIT caps every stack) with int64 arithmetic; callers pass counts >= 0, as
   every oracle call site does. A slot with id >= 0 is occupied, [id, 0] included (a truthy list in Python). */
static int stacks(const RsPack *P, int stackall, int32_t obj) {
    if (stackall) return 1;
    const RsObjType *t = rs_obj_type(P, obj);
    if (!t) rs_trap("rs_inv: an obj type the pack lacks (KeyError)");
    return t->stackable != 0;
}

int64_t rs_inv_add(const RsPack *P, Slot *s, int cap, int stackall, int32_t obj, int64_t count) {
    int64_t completed = 0;
    if (!stacks(P, stackall, obj)) {
        for (int i = 0; i < cap; i++) {
            if (s[i].id >= 0) continue;
            s[i] = (Slot){obj, 1};
            completed += 1;
            if (completed >= count) break;
        }
        return completed;
    }
    int index = -1;
    for (int i = 0; i < cap && index < 0; i++)
        if (s[i].id >= 0 && s[i].id == obj) index = i;
    if (index < 0) {
        for (int i = 0; i < cap && index < 0; i++)
            if (s[i].id < 0) index = i;
        if (index < 0) return 0;
    }
    int64_t have = s[index].id >= 0 ? s[index].count : 0;
    int64_t total = have + count < RS_STACK_LIMIT ? have + count : RS_STACK_LIMIT;
    s[index] = (Slot){obj, (int32_t)total};
    return total - have;
}

int64_t rs_inv_remove(Slot *s, int cap, int32_t obj, int64_t count) {
    int64_t removed = 0;
    for (int i = 0; i < cap; i++) {
        if (s[i].id < 0 || s[i].id != obj) continue;
        int64_t take = s[i].count < count - removed ? s[i].count : count - removed;
        removed += take;
        s[i].count = (int32_t)(s[i].count - take);
        if (s[i].count == 0) s[i] = (Slot){-1, 0};
        if (removed >= count) break;
    }
    return removed;
}

int64_t rs_inv_total(const Slot *s, int cap, int32_t obj) {
    int64_t total = 0;
    for (int i = 0; i < cap; i++)
        if (s[i].id >= 0 && s[i].id == obj) total += s[i].count;
    return total;
}

int32_t rs_inv_freespace(const Slot *s, int cap) {
    int32_t n = 0;
    for (int i = 0; i < cap; i++) n += s[i].id < 0;
    return n;
}

int64_t rs_inv_itemspace2(const RsPack *P, const Slot *s, int cap, int stackall, int32_t obj, int64_t count) {
    if (stacks(P, stackall, obj)) {
        for (int i = 0; i < cap; i++) {
            if (s[i].id < 0 || s[i].id != obj) continue;
            int64_t over = (int64_t)s[i].count + count - RS_STACK_LIMIT;
            return over > 0 ? over : 0;
        }
        return rs_inv_freespace(s, cap) ? 0 : count;
    }
    int64_t over = count - rs_inv_freespace(s, cap);
    return over > 0 ? over : 0;
}

static Slot *inventory(World *w, int which, int *cap, int *stackall) {
    *stackall = which == RS_LOCN_BANK;
    *cap = which == RS_LOCN_INV ? RS_INV_CAP : which == RS_LOCN_WORN ? RS_WORN_CAP : RS_BANK_CAP;
    return which == RS_LOCN_INV ? w->p.inv : which == RS_LOCN_WORN ? w->p.worn : w->p.bank;
}

/* World.uncert (engine.py). */
static int32_t uncert(const RsPack *P, int32_t obj) {
    const RsObjType *t = rs_obj_type(P, obj);
    if (!t) rs_trap("uncert: an obj type the pack lacks (KeyError)");
    return t->certtemplate >= 0 && t->certlink >= 0 ? t->certlink : obj;
}

/* World.move_item, INV_MOVEITEM_UNCERT (engine.py): dst.add's result is ignored, as the oracle ignores it. */
int rs_move_item(World *w, int src, int dst, int32_t obj, int64_t count) {
    int cap, stackall;
    Slot *s = inventory(w, src, &cap, &stackall);
    int64_t done = rs_inv_remove(s, cap, obj, count);
    if (!done) return 0;
    int32_t target = uncert(w->P, obj);
    Slot *d = inventory(w, dst, &cap, &stackall);
    rs_inv_add(w->P, d, cap, stackall, target, done);
    RsEvent e = {.tick = w->tick, .kind = RS_EV_TRANSFER, .a8 = (uint8_t)src, .b8 = (uint8_t)dst, .obj = obj,
                 .count = (int32_t)done, .v0 = target};
    return rs_event(w, &e);
}

/* ---- Player modal and interaction ops (engine.py) ------------------------------------------------------ */
/* Player.close_modal (engine.py). modal_main is only ever bank_main, which the unported bank sets, so the
   close_bank branch never runs here. */
void rs_close_modal(World *w, int clear_weak) {
    Player *p = &w->p;
    if (clear_weak) rs_req_clear(w, &p->weak);
    if (!p->delayed) p->protect = 0;
    if (!p->modal) return;
    p->modal = 0;
    if (p->script.serial && p->script.mode == RS_SCRIPT_PAUSE) {
        memset(&p->script, 0, sizeof p->script);
        p->n_resume = 0;
    }
    p->modal_main = -1;
    p->modal_chat = -1;
}

/* Player.open_chat (engine.py): clears the main and side bits, and modal_chat, not modal_main. */
void rs_open_chat(World *w, int32_t iface) {
    Player *p = &w->p;
    if (p->modal & RS_MODAL_MAIN) { p->modal &= (uint8_t)~RS_MODAL_MAIN; p->modal_chat = -1; }
    if (p->modal & RS_MODAL_SIDE) { p->modal &= (uint8_t)~RS_MODAL_SIDE; p->modal_chat = -1; }
    p->modal |= RS_MODAL_CHAT;
    p->modal_chat = iface;
}

void rs_clear_interaction(World *w) {
    Player *p = &w->p;
    p->target = (RsRef){0};
    p->target_op = p->target_type = -1;
    p->ap_range = 10;
    p->ap_range_called = 0;
}

void rs_clear_pending_action(World *w) {
    rs_clear_interaction(w);
    rs_close_modal(w, 1);
}

void rs_stop_action(World *w) {
    rs_clear_pending_action(w);
    rs_clear_waypoints(w);
}

int rs_set_interaction(World *w, RsRef target, int32_t op) {
    Target t;
    if (!target_of(w, target, &t) || !t.valid) return 0;
    Player *p = &w->p;
    p->target = target;
    p->target_op = op;
    p->ap_range = 10;
    p->ap_range_called = 0;
    p->target_type = t.type;
    return 1;
}

void rs_clear_waypoints(World *w) { w->p.path_n = 0; }

void rs_queue_waypoint(World *w, int32_t x, int32_t z) {
    Player *p = &w->p;
    p->path[0][0] = (int16_t)x, p->path[0][1] = (int16_t)z;
    p->path_n = 1;
    p->allow_repath = 1;
}

void rs_queue_waypoints(World *w, const int16_t path[][2], int n) {
    Player *p = &w->p;
    if (n < 0 || n > RS_PATH_MAX) rs_trap("rs_queue_waypoints: more than RS_PATH_MAX waypoints");
    memmove(p->path, path, (size_t)n * sizeof p->path[0]);
    p->path_n = (uint8_t)n;
    p->allow_repath = 1;
}

int rs_player_busy(const World *w) { return w->p.delayed || (w->p.modal & (RS_MODAL_MAIN | RS_MODAL_CHAT)) != 0; }
int rs_player_can_access(const World *w) { return !w->p.protect && !rs_player_busy(w); }
int rs_player_busy2(const World *w) { return w->p.target.kind != RS_REF_NONE || w->p.path_n > 0; }

/* Player.combat_level (engine.py): doubles in Python's order (no FMA: -ffp-contract=off). */
int32_t rs_combat_level(const World *w) {
    const int16_t *b = w->p.base_levels;
    double base = 0.25 * (double)(b[RS_SKILL_DEFENCE] + b[RS_SKILL_HITPOINTS] + b[RS_SKILL_PRAYER] / 2);
    double melee = 0.325 * (double)(b[RS_SKILL_ATTACK] + b[RS_SKILL_STRENGTH]);
    double ranged = 0.325 * (double)(b[RS_SKILL_RANGED] / 2 + b[RS_SKILL_RANGED]);
    double magic = 0.325 * (double)(b[RS_SKILL_MAGIC] / 2 + b[RS_SKILL_MAGIC]);
    double best = melee;
    if (ranged > best) best = ranged;
    if (magic > best) best = magic;
    return (int32_t)(base + best);
}

/* ---- requests and queues (engine.py, 1083-1085, 1798-1816) ----------------------------------------- */
static int req_take(World *w, int16_t *node) {
    if (w->req_free < 0) return RS_OVERFLOW(&w->err, "RS_MAX_REQS queued requests");
    *node = w->req_free;
    w->req_free = w->req[*node].next;
    return 0;
}

static void req_append(World *w, ReqList *list, int16_t node) {
    Req *r = &w->req[node];
    r->prev = list->tail, r->next = -1, r->in_list = 1, r->dead_next = -1;
    if (list->tail >= 0) w->req[list->tail].next = node;
    else list->head = node;
    list->tail = node;
}

int rs_enqueue(World *w, int kind, int16_t script, int32_t a0, int32_t a1, int32_t a2, int32_t delay) {
    int16_t node;
    RS_TRY(req_take(w, &node));
    Req *r = &w->req[node];
    *r = (Req){.delay = delay, .arg = {a0, a1, a2}, .script = script, .kind = (uint8_t)kind};
    Player *p = &w->p;
    req_append(w, kind == RS_Q_ENGINE ? &p->engine : kind == RS_Q_WEAK ? &p->weak : &p->queue, node);
    return 0;
}

int rs_npc_enqueue(World *w, Npc *n, int16_t queue_id, int32_t arg, int32_t delay, int source_player) {
    int16_t node;
    RS_TRY(req_take(w, &node));
    w->req[node] = (Req){.delay = delay, .arg = {arg, source_player != 0, 0}, .script = queue_id};
    req_append(w, &n->queue, node);
    return 0;
}

/* _unlink: a no-op for a node already out; the slot waits on the dead list until a pass ends. */
void rs_req_unlink(World *w, ReqList *list, int16_t node) {
    Req *r = &w->req[node];
    if (!r->in_list) return;
    if (r->prev >= 0) w->req[r->prev].next = r->next;
    else list->head = r->next;
    if (r->next >= 0) w->req[r->next].prev = r->prev;
    else list->tail = r->prev;
    r->in_list = 0;
    r->dead_next = w->req_dead;
    w->req_dead = node;
}

void rs_req_clear(World *w, ReqList *list) {
    while (list->head >= 0) rs_req_unlink(w, list, list->head);
}

void rs_links_begin(const World *w, const ReqList *list, RsLinks *it) {
    (void)w;
    it->node = list->head;
    it->following = -1;
    it->ended = 0;
}

/* One visit of _links: the node to run, its successor read now (items[k + 1]); a node out of its list (k < 0) is
   still run, then the pass ends. */
int16_t rs_links_next(const World *w, RsLinks *it) {
    if (it->ended || it->node < 0) return -1;
    int16_t node = it->node;
    const Req *r = &w->req[node];
    if (r->in_list) {
        it->following = r->next;
        it->node = r->next;
    } else {
        it->ended = 1;
    }
    return node;
}

void rs_links_end(World *w) {
    while (w->req_dead >= 0) {
        int16_t node = w->req_dead;
        Req *r = &w->req[node];
        w->req_dead = r->dead_next;
        *r = (Req){.prev = -1, .next = w->req_free, .dead_next = -1};
        w->req_free = node;
    }
}

/* ---- the script runner (engine.py) ----------------------------------------------------------------- */
void rs_frame_new(World *w, Frame *f, int16_t script, RsRef ref, int32_t a0, int32_t a1, int32_t a2) {
    memset(f, 0, sizeof *f);
    f->serial = ++w->next_serial;
    f->depth = 1;
    f->script[0] = script;
    f->arg[0][0] = a0, f->arg[0][1] = a1, f->arg[0][2] = a2;
    f->ref[0] = ref;
}

/* World._run: one next(gen) of frame f (a copy the caller owns). */
static int run(World *w, Frame *f, int protect) {
    RsSusp s = {0};
    RS_TRY(rs_script_step(w, f, &s));
    Player *p = &w->p;
    if (s.kind == RS_DONE) {
        if (protect) p->protect = 0;
        if (p->script.serial && p->script.serial == f->serial) {
            memset(&p->script, 0, sizeof p->script);
            p->n_resume = 0;
            if (!(p->modal & RS_MODAL_MAIN)) rs_close_modal(w, 0);   /* close chat dialogues, leave a main modal */
        }
        return 0;
    }
    if (protect) p->protect = 0;
    if (s.kind == RS_PAUSE) {
        p->script = *f;
        p->script.mode = RS_SCRIPT_PAUSE;
    } else if (s.kind == RS_DELAY) {
        p->delayed = 1;
        p->delayed_until = w->tick + 1 + s.n;
        p->script = *f;
        p->script.mode = RS_SCRIPT_SUSPENDED;
        p->protect = (uint8_t)protect;
    } else {
        return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a player script returned an npc suspension");
    }
    return 0;
}

/* Player.executeScript (World.execute). */
int rs_execute(World *w, int16_t script, RsRef ref, int32_t a0, int32_t a1, int32_t a2, int protect, int force) {
    if (script == RS_S_NONE) return 0;   /* gen is None */
    Player *p = &w->p;
    if (!force && protect && (p->protect || p->delayed)) return 0;
    if (protect) p->protect = 1;
    Frame f;
    rs_frame_new(w, &f, script, ref, a0, a1, a2);
    return run(w, &f, protect);
}

/* World.resume: p.script = (gen, 'running'), then _run with protect semantics (p.protect not set first). */
int rs_resume(World *w) {
    Player *p = &w->p;
    if (!p->script.serial) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "resume with no script (a TypeError)");
    p->script.mode = RS_SCRIPT_RUNNING;
    Frame f = p->script;
    return run(w, &f, 1);
}

/* World.run_held (engine.py): held_script names the port, or the trigger names none. */
int rs_run_held(World *w, int16_t script, int16_t arg, int32_t name) {
    if (script == RS_S_UNKNOWN || script == RS_S_NONE) return RS_FAIL(&w->err, RS_E_HELD_SCRIPT, name);
    return rs_execute(w, script, (RsRef){0}, arg, name, 0, 1, 0);
}

/* World.run_trigger (engine.py). */
int rs_run_trigger(World *w, int16_t script, int16_t arg, int32_t name, RsRef target) {
    if (script == RS_S_UNKNOWN || script == RS_S_NONE) return RS_FAIL(&w->err, RS_E_ENGINE_SCRIPT, name);
    return rs_execute(w, script, target, arg, name, 0, 1, 0);
}

/* ---- RuneScript randomness (engine.py, 1726-1728) ------------------------------------------------- */
static uint32_t script_m(const World *w, int nkeys, const int64_t *keys) {
    int64_t k[RS_DRAW_KEYS];
    if (nkeys < 0 || nkeys > RS_DRAW_KEYS - 2) rs_trap("a script draw with more than six keys");
    k[0] = RS_R_SCRIPT, k[1] = w->tick;
    for (int i = 0; i < nkeys; i++) k[2 + i] = keys[i];
    return rs_wm(w, 2 + nkeys, k);
}

int64_t rs_random(const World *w, int64_t n, int nkeys, const int64_t *keys) {
    return rs_pick(script_m(w, nkeys, keys), n);
}

int64_t rs_randominc(const World *w, int64_t n, int nkeys, const int64_t *keys) {
    return rs_pick(script_m(w, nkeys, keys), n + 1);
}

int rs_stat_random(const World *w, int32_t stat, int32_t low, int32_t high, int nkeys, const int64_t *keys) {
    int64_t level = w->p.levels[stat];
    int64_t value = rs_floordiv(low * (99 - level), 98) + rs_floordiv(high * (level - 1), 98) + 1;
    return value > rs_pick(script_m(w, nkeys, keys), 256);
}

/* ---- stats (engine.py, 1459-1475, 2361-2363) ------------------------------------------------------ */
int rs_changestat(World *w) { return rs_enqueue(w, RS_Q_ENGINE, RS_S_CHANGESTAT, 0, 0, 0, 0); }

/* int(constant + base * percent / 100): Python's true division, then truncation toward zero. */
static int32_t stat_delta(int32_t base, int32_t constant, int32_t percent) {
    return (int32_t)((double)constant + (double)((int64_t)base * percent) / 100.0);
}

static int hp_event(World *w, int cause, int32_t planned, int32_t before, int32_t after) {
    RsEvent e = {.tick = w->tick, .kind = RS_EV_AGENT_HP, .a8 = (uint8_t)cause, .count = planned, .v0 = before,
                 .v1 = after};
    return rs_event(w, &e);
}

int rs_stat_sub(World *w, int32_t stat, int32_t constant, int32_t percent) {
    Player *p = &w->p;
    int32_t current = p->levels[stat];
    int32_t subbed = current - stat_delta(p->base_levels[stat], constant, percent);
    p->levels[stat] = (int16_t)imax(subbed, 0);
    p->dead |= p->levels[RS_SKILL_HITPOINTS] <= 0;
    if (subbed != current) RS_TRY(rs_changestat(w));
    return 0;
}

int rs_stat_add(World *w, int32_t stat, int32_t constant, int32_t percent) {
    Player *p = &w->p;
    int32_t current = p->levels[stat];
    int32_t added = current + stat_delta(p->base_levels[stat], constant, percent);
    p->levels[stat] = (int16_t)imin(added, 255);
    if (added != current) RS_TRY(rs_changestat(w));
    return 0;
}

int rs_stat_heal(World *w, int32_t stat, int32_t constant, int32_t percent, int cause) {
    Player *p = &w->p;
    int32_t current = p->levels[stat];
    int32_t healed = current + stat_delta(p->base_levels[stat], constant, percent);
    p->levels[stat] = (int16_t)imax(imin(healed, p->base_levels[stat]), current);
    if (stat == RS_SKILL_HITPOINTS && p->levels[stat] != current)
        RS_TRY(hp_event(w, cause, healed - current, current, p->levels[stat]));
    if (healed != current) RS_TRY(rs_changestat(w));
    return 0;
}

/* Player.addXp with the world XP rate (World.add_xp): the 'xp' event follows the stat, the HP event precedes
   [changestat], which precedes [advancestat,<skill>] on the engine queue; levelup reads base_levels when it runs. */
int rs_add_xp(World *w, int32_t stat, int32_t xp, int cause, int32_t obj, int32_t src) {
    if (xp == 0) return 0;
    Player *p = &w->p;
    int32_t was = p->stats[stat];
    int64_t total = (int64_t)p->stats[stat] + (int64_t)xp * w->xp_rate;
    p->stats[stat] = (int32_t)(total < 2000000000 ? total : 2000000000);
    RsEvent e = {.tick = w->tick, .kind = RS_EV_XP, .a8 = (uint8_t)stat, .b8 = (uint8_t)cause, .obj = obj, .count = xp,
                 .v0 = was, .v1 = p->stats[stat], .v2 = src};
    RS_TRY(rs_event(w, &e));
    int32_t before = p->base_levels[stat], current = p->levels[stat];
    if (p->levels[stat] == p->base_levels[stat]) p->levels[stat] = (int16_t)level_for(w->P, p->stats[stat]);
    p->base_levels[stat] = (int16_t)level_for(w->P, p->stats[stat]);
    if (p->base_levels[stat] > before) {
        if (p->levels[stat] < before) p->levels[stat] = (int16_t)(p->levels[stat] + p->base_levels[stat] - before);
        if (stat == RS_SKILL_HITPOINTS) RS_TRY(hp_event(w, RS_HP_LEVELUP, p->levels[stat] - current, current,
                                                        p->levels[stat]));
        RS_TRY(rs_changestat(w));
        RS_TRY(rs_enqueue(w, RS_Q_ENGINE, RS_S_LEVELUP, stat, 0, 0, 0));
    }
    return 0;
}

/* World.npc_spawned, [ai_spawn,_] (engine.py). */
void rs_npc_spawned(World *w, Npc *n) {
    const RsNpcType *nt = rs_npc_type(w->P, n->type);
    if (!nt) rs_trap("rs_npc_spawned: an npc type the pack lacks (KeyError)");
    n->v_xp_mult = nt->combat_xp_multiplier;
}

/* World.produce (engine.py). */
int rs_produce(World *w, int32_t obj, int32_t script_name, int64_t *added) {
    *added = rs_inv_add(w->P, w->p.inv, RS_INV_CAP, 0, obj, 1);
    RsEvent e = {.tick = w->tick, .kind = RS_EV_PRODUCE, .obj = obj, .count = (int32_t)*added, .v0 = script_name};
    return rs_event(w, &e);
}

/* ---- text and evidence (engine.py, 532) ------------------------------------------------------------- */
int rs_mes(World *w, int32_t msg) {
    ChatLog *c = &w->chat;
    if (c->n - c->captured >= RS_CHAT_RING) return RS_OVERFLOW(&w->err, "RS_CHAT_RING chat lines between captures");
    c->ring[c->n % RS_CHAT_RING] = (RsChatLine){w->tick, msg};
    c->n++;
    return 0;
}

int rs_settext(World *w, int32_t iface, int32_t com, int32_t msg) {
    TextLog *t = &w->text;
    if (t->n >= RS_TEXT_LOG) return RS_OVERFLOW(&w->err, "RS_TEXT_LOG settext lines between captures");
    t->line[t->n++] = (RsTextLine){w->tick, iface, com, msg};
    t->total++;
    return 0;
}

int rs_event(World *w, const RsEvent *e) {
    if (w->n_events >= RS_MAX_EVENTS) return RS_OVERFLOW(&w->err, "RS_MAX_EVENTS world events");
    w->ev[w->n_events++] = *e;
    return 0;
}

/* ---- interaction (engine.py) ------------------------------------------------------------------------ */
static int in_operable(World *w, const Target *t, int *ok) {
    const Player *p = &w->p;
    *ok = 0;
    if (t->level != p->level) return 0;
    if (t->kind == RS_REF_NPC || t->kind == RS_REF_OBJ) {
        /* an obj: beside it as for an npc, or on it (Player.inOperableDistance) */
        if (t->kind == RS_REF_OBJ && p->x == t->x && p->z == t->z) { *ok = 1; return 0; }
        uint16_t mask;
        RS_TRY(rs_entity_reaching(w, t->x, t->z, &mask));
        for (int b = 0; b < 9; b++)
            if ((mask >> b & 1) && t->x + RS_NEIGHBOUR_DX[b] == p->x && t->z + RS_NEIGHBOUR_DZ[b] == p->z) *ok = 1;
        return 0;
    }
    int16_t tiles[128][2];
    int n = 0;
    RS_TRY(rs_loc_reaching(w, t->loc, tiles, &n, 128));
    for (int i = 0; i < n; i++)
        if (tiles[i][0] == p->x && tiles[i][1] == p->z) *ok = 1;
    return 0;
}

int rs_in_operable(World *w, RsRef target, int *ok) {
    Target t;
    *ok = 0;
    if (!target_of(w, target, &t)) return 0;
    return in_operable(w, &t, ok);
}

/* PathingEntity.inApproachDistance for a 1x1 player (engine.py). */
static int in_approach(World *w, int32_t range, const Target *t, int *ok) {
    const Player *p = &w->p;
    *ok = 0;
    if (t->kind == RS_REF_NPC && t->x <= p->x && p->x < t->x + t->width && t->z <= p->z && p->z < t->z + t->length)
        return 0;
    if (distance_to(p, t) > range) return 0;
    int32_t nx, nz;
    nearest_tile(p, t, &nx, &nz);
    return rs_line_of_sight(w, p->x, p->z, nx, nz, ok);
}

int rs_in_approach(World *w, int32_t range, RsRef target, int *ok) {
    Target t;
    *ok = 0;
    if (!target_of(w, target, &t)) return 0;
    return in_approach(w, range, &t, ok);
}

/* World.validate_target (engine.py). */
static int validate_target(World *w) {
    Target t;
    if (!target_of(w, w->p.target, &t)) return 0;   /* None, or an obj that left world.objs (inactive in Python) */
    if (t.level != w->p.level || t.type != w->p.target_type) return 0;
    return t.valid;
}

/* Player.defaultOp outside production (engine.py). */
static int default_op(World *w, const Target *t) {
    Player *p = &w->p;
    int32_t key, debugname;
    if (t->kind == RS_REF_NPC) debugname = rs_npc_type(w->P, t->type)->debugname;
    else if (t->kind == RS_REF_LOC) debugname = rs_loc_type(w->P, t->type)->debugname;
    else debugname = rs_obj_type(w->P, t->type)->debugname;
    if (p->target_op == RS_APLOCU) key = RS_MK_NO_TRIGGER_OPLOCU;
    else if (p->target_op == RS_APNPCU) key = RS_MK_NO_TRIGGER_OPNPCU;
    else return RS_FAIL(&w->err, RS_E_NO_SCRIPT_FOR_OP, p->target_op, debugname);
    int32_t msg;
    RS_TRY(rs_msg(w, key, t->type, &msg));
    RS_TRY(rs_mes(w, msg));
    RS_TRY(rs_text_msg(w, RS_TX_dm_default, 0, &msg));
    RS_TRY(rs_mes(w, msg));
    rs_clear_waypoints(w);
    return 0;
}

/* World.try_interact (engine.py). */
static int try_interact(World *w, int allow_op_scenery, int *interacted) {
    Player *p = &w->p;
    *interacted = 0;
    if (p->target.kind == RS_REF_NONE || !rs_player_can_access(w)) return 0;
    RsRef target = p->target;
    Target t;
    if (!target_of(w, target, &t)) return 0;   /* validated just before by both callers: unreachable */
    int16_t op_script, op_arg, ap_script, ap_arg;
    int32_t op_name, ap_name;
    RS_TRY(trigger_of(w, &t, p->target_op + RS_AP_TO_OP, &op_script, &op_arg, &op_name));
    RS_TRY(trigger_of(w, &t, p->target_op, &ap_script, &ap_arg, &ap_name));
    int npc = t.kind == RS_REF_NPC, ok = 0;
    if (op_script != RS_S_NONE && (npc || allow_op_scenery)) {
        RS_TRY(in_operable(w, &t, &ok));
        if (ok) {
            p->target = (RsRef){0};
            rs_clear_waypoints(w);
            RS_TRY(rs_run_trigger(w, op_script, op_arg, op_name, target));
            p->next_target = p->target;
            p->target = target;
            *interacted = 1;
            return 0;
        }
    }
    if (ap_script != RS_S_NONE) {
        RS_TRY(in_approach(w, p->ap_range, &t, &ok));
        if (ok) {
            p->ap_range_called = 0;
            int16_t saved[RS_PATH_MAX][2];
            uint8_t saved_n = p->path_n;
            memcpy(saved, p->path, sizeof saved);
            p->target = (RsRef){0};
            rs_clear_waypoints(w);
            RS_TRY(rs_run_trigger(w, ap_script, ap_arg, ap_name, target));
            p->next_target = p->target;
            p->target = target;
            if (p->next_target.kind != RS_REF_NONE) {
                rs_clear_waypoints(w);
            } else if (p->ap_range_called) {
                memcpy(p->path, saved, sizeof saved);
                p->path_n = saved_n;
                p->target = target;
                return 0;
            }
            *interacted = 1;
            return 0;
        }
    }
    RS_TRY(in_approach(w, p->ap_range, &t, &ok));
    if (ok) {
        p->ap_range = -1;
        return 0;
    }
    if (npc || allow_op_scenery) {
        RS_TRY(in_operable(w, &t, &ok));
        if (ok) {
            RS_TRY(default_op(w, &t));
            *interacted = 1;
        }
    }
    return 0;
}

/* NaivePathFinder.naiveDestination for a 1x1 source and destination (engine.py). */
static int naive_destination(World *w, int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t *x, int32_t *z) {
    int32_t diagonal = sx - dx + (sz - dz), anti = sx - dx - (sz - dz), off;
    int sw = anti < 0, nw = diagonal >= 0, ne = anti > 0, se = diagonal <= 0;
    if (sw && !nw) {
        off = diagonal >= -1 ? imin(diagonal + 1, 0) : anti > -1 ? -(1 + anti) : 0;
        *x = dx - 1, *z = dz + off;
        return 0;
    }
    if (nw && !ne) {
        off = anti >= -1 ? imin(anti + 1, 0) : diagonal < 1 ? imax(diagonal - 1, 0) : 0;
        *x = dx + off, *z = dz + 1;
        return 0;
    }
    if (ne && !se) {
        off = anti <= 1 ? 1 - anti : diagonal < 1 ? imax(diagonal - 1, 0) : 0;
        *x = dx + 1, *z = dz + off;
        return 0;
    }
    if (!(se && !sw)) return RS_FAIL0(&w->err, RS_E_NAIVE_NO_SIDE);
    off = diagonal > -1 ? imin(diagonal + 1, 0) : anti < 1 ? imax(anti - 1, 0) : 0;
    *x = dx + off, *z = dz - 1;
    return 0;
}

static int is_diagonal(int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t dw, int32_t dl) {
    return (sx + 1 == dx && sz + 1 == dz) || (sx - 1 == dx + dw - 1 && sz - 1 == dz + dl - 1)
        || (sx + 1 == dx && sz - 1 == dz + dl - 1) || (sx - 1 == dx + dw - 1 && sz + 1 == dz);
}

/* Whether (x, z) lies outside the dw x dl rectangle at (dx, dz) (the naive finder's test, engine.py, 1595). */
static int outside_rect(int32_t x, int32_t z, int32_t dx, int32_t dz, int32_t dw, int32_t dl) {
    return dx >= x + 1 || dx + dw <= x || dz >= z + 1 || dz + dl <= z;
}

/* World.naive_path (engine.py). */
int rs_naive_path(World *w, int32_t level, int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t dw, int32_t dl,
                  int32_t size, int16_t out[2]) {
    static const int8_t DIRS[4][2] = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
    if (size != 1) return RS_FAIL0(&w->err, RS_E_NAIVE_LARGE);
    if (!outside_rect(sx, sz, dx, dz, dw, dl)) {
        int64_t d = rs_pick(rs_wm(w, RS_KEYS(RS_R_SCRIPT, w->tick, 902)), 4);
        out[0] = (int16_t)(sx + DIRS[d][0]), out[1] = (int16_t)(sz + DIRS[d][1]);
        return 0;
    }
    int32_t x, z;
    RS_TRY(naive_destination(w, sx, sz, dx, dz, &x, &z));
    if (is_diagonal(x, z, dx, dz, dw, dl) || !outside_rect(x, z, dx, dz, dw, dl)) {
        out[0] = (int16_t)x, out[1] = (int16_t)z;
        return 0;
    }
    while (x != dx && z != dz) {
        int32_t step_x = sgn(dx - x), step_z = sgn(dz - z);
        int ok;
        RS_TRY(rs_can_travel(w, level, x, z, step_x, step_z, 1, 0, 0, &ok));
        if (ok) { x += step_x, z += step_z; continue; }
        if (step_x) {
            RS_TRY(rs_can_travel(w, level, x, z, step_x, 0, 1, 0, 0, &ok));
            if (ok) { x += step_x; continue; }
        }
        if (step_z) {
            RS_TRY(rs_can_travel(w, level, x, z, 0, step_z, 1, 0, 0, &ok));
            if (ok) { z += step_z; continue; }
        }
        break;
    }
    out[0] = (int16_t)x, out[1] = (int16_t)z;
    return 0;
}

/* Player.pathToPathingTarget, NAIVE (engine.py). */
static int path_to_pathing_target(World *w) {
    Player *p = &w->p;
    if (p->target.kind != RS_REF_NPC || !rs_player_can_access(w)) return 0;
    const Npc *t = &w->npcs[p->target.slot];
    if (t->x <= p->x && p->x < t->x + t->size && t->z <= p->z && p->z < t->z + t->size) {
        int32_t sx = 0, sz = 0;   /* draws in Python's order (the trace logs them): 900, then 901 */
        if (rs_wrng(w, RS_KEYS(RS_R_SCRIPT, w->tick, 900)) < 0.5)
            sx = rs_wrng(w, RS_KEYS(RS_R_SCRIPT, w->tick, 901)) < 0.5 ? -1 : 1;
        else
            sz = rs_wrng(w, RS_KEYS(RS_R_SCRIPT, w->tick, 901)) < 0.5 ? -1 : 1;
        rs_queue_waypoint(w, p->x + sx, p->z + sz);
        return 0;
    }
    if (p->path_n <= 1 && p->allow_repath) {
        if (!p->path_n || iabs(t->x - p->path[p->path_n - 1][0]) > 1 || iabs(t->z - p->path[p->path_n - 1][1]) > 1) {
            int16_t step[1][2];
            RS_TRY(rs_naive_path(w, p->level, p->x, p->z, t->x, t->z, t->size, t->size, 1, step[0]));
            rs_queue_waypoints(w, step, 1);
        }
    }
    return 0;
}

/* Player.updateMovement (engine.py). */
static int update_movement(World *w) {
    Player *p = &w->p;
    if (p->path_n && !p->instant) {
        RS_TRY(rs_process_movement_player(w));
        if (rs_source(w->L, p->level, p->x, p->z) < 0) return RS_FAIL(&w->err, RS_E_PLAYER_LEFT, p->level, p->x, p->z);
    }
    if (p->steps_taken > 0) p->last_movement = w->tick + 1;
    return 0;
}

/* Player.processInteraction (engine.py). */
static int process_interaction(World *w) {
    Player *p = &w->p;
    p->next_target = (RsRef){0};
    int interacted = 0;
    if (p->target.kind != RS_REF_NONE && rs_player_can_access(w)) {
        if (!validate_target(w)) {
            rs_clear_interaction(w);
            return 0;
        }
        RS_TRY(try_interact(w, 0, &interacted));
    }
    if (!interacted) {
        RS_TRY(path_to_pathing_target(w));
        RS_TRY(update_movement(w));
        if (p->target.kind != RS_REF_NONE && rs_player_can_access(w)) {
            RS_TRY(try_interact(w, p->steps_taken == 0, &interacted));
            if (!interacted && !p->ap_range_called && !p->path_n && p->steps_taken == 0) {
                int32_t msg;
                RS_TRY(rs_msg(w, RS_MK_CANT_REACH, 0, &msg));
                RS_TRY(rs_mes(w, msg));
                rs_clear_interaction(w);
            }
        }
    }
    if (p->next_target.kind != RS_REF_NONE) p->target = p->next_target;
    else if (interacted && !p->ap_range_called) rs_clear_interaction(w);
    return 0;
}

/* ---- the cycle (engine.py) ---------------------------------------------------------------------------- */
/* process_world (engine.py): the hunt filter; the hunt itself is npc.c's. */
static int process_world(World *w) {
    if (!w->has_player) return 0;
    for (int i = 0; i < w->n_npcs; i++) {
        Npc *n = &w->npcs[i];
        if (!n->active || n->hunt_mode == -1 || n->observers <= 0) continue;
        const RsNpcType *nt = rs_npc_type(w->P, n->type);
        if (nt->has_hunt && nt->hunt_id == n->hunt_mode && nt->hunt_type == 1) RS_TRY(rs_npc_hunt_all(w, n));
    }
    return 0;
}

/* process_clients_in (engine.py): the AFK readiness roll (dead within the current horizons). */
static void process_clients_in(World *w) {
    if (w->has_player && rs_mod(w->tick, RS_AFK_EVENTRATE) == 0 && !(w->excl & RS_EXCLUDE_RANDOM_EVENTS))
        w->p.afk_ready = rs_wrng(w, RS_KEYS(RS_R_AFK, w->tick)) < RS_AFK_CHANCE1;
}

/* process_npc_events (engine.py). */
static void process_npc_events(World *w) {
    uint8_t events[RS_MAX_NPCS];
    int n = w->n_npc_events;
    memcpy(events, w->npc_events, (size_t)n);
    w->n_npc_events = 0;
    for (int i = 0; i < n; i++) rs_npc_spawned(w, &w->npcs[events[i]]);
}

/* process_npcs (engine.py): npcs outside the active chunks around the player turn dormant. */
static int process_npcs(World *w) {
    const Player *p = &w->p;
    int64_t cx0 = 0, cx1 = -1, cz0 = 0, cz1 = -1;
    if (w->has_player) {
        cx0 = rs_asr(p->x - RS_NPC_ACTIVE_RADIUS, 6), cx1 = rs_asr(p->x + RS_NPC_ACTIVE_RADIUS, 6);
        cz0 = rs_asr(p->z - RS_NPC_ACTIVE_RADIUS, 6), cz1 = rs_asr(p->z + RS_NPC_ACTIVE_RADIUS, 6);
    }
    for (int i = 0; i < w->n_npcs; i++) {
        Npc *n = &w->npcs[i];
        int64_t cx = rs_asr(n->x, 6), cz = rs_asr(n->z, 6);
        int active = w->has_player && n->level == p->level && cx0 <= cx && cx <= cx1 && cz0 <= cz && cz <= cz1;
        RS_TRY(active ? rs_npc_turn(w, n) : rs_npc_lifecycle(w, n));
    }
    return 0;
}

/* Player.processQueues (engine.py): the queue then the weak queue; each visit counts down, and a due request is
   unlinked, then executed. No ported script queues a 'strong' request, so the strong close never runs. */
static int process_queues(World *w) {
    Player *p = &w->p;
    ReqList *lists[2] = {&p->queue, &p->weak};
    for (int q = 0; q < 2; q++) {
        RsLinks it;
        rs_links_begin(w, lists[q], &it);
        for (int16_t node; (node = rs_links_next(w, &it)) >= 0;) {
            Req *r = &w->req[node];
            int32_t delay = r->delay;
            r->delay -= 1;
            if (rs_player_can_access(w) && delay <= 0) {
                rs_req_unlink(w, lists[q], node);
                RS_TRY(rs_execute(w, r->script, (RsRef){0}, r->arg[0], r->arg[1], r->arg[2], 1, 0));
            }
        }
        rs_links_end(w);
    }
    return 0;
}

/* process_players (engine.py). */
static int process_players(World *w) {
    if (!w->has_player) return 0;
    Player *p = &w->p;
    if (p->delayed && w->tick >= p->delayed_until) p->delayed = 0;
    if (!p->delayed && p->script.serial && p->script.mode == RS_SCRIPT_SUSPENDED) RS_TRY(rs_resume(w));
    RS_TRY(process_queues(w));
    for (int kind = 0; kind < 2; kind++)         /* ('normal', 'soft'), timers in login order */
        for (int i = 0; i < p->n_timers; i++) {
            Timer *t = &p->timer[i];
            if (t->kind != kind) continue;
            if (w->tick >= t->clock + t->interval && (kind == 1 || rs_player_can_access(w))) {
                t->clock = w->tick;
                RS_TRY(rs_execute(w, t->script, (RsRef){0}, 0, 0, 0, kind == 0, 0));
            }
        }
    /* the engine queue executes, then unlinks (engine.py) */
    RsLinks it;
    rs_links_begin(w, &p->engine, &it);
    for (int16_t node; (node = rs_links_next(w, &it)) >= 0;) {
        Req *r = &w->req[node];
        int32_t delay = r->delay;
        r->delay -= 1;
        if (rs_player_can_access(w) && delay <= 0) {
            RS_TRY(rs_execute(w, r->script, (RsRef){0}, r->arg[0], r->arg[1], r->arg[2], 1, 0));
            rs_req_unlink(w, &p->engine, node);
        }
    }
    rs_links_end(w);
    return process_interaction(w);
}

/* World.processLogins (engine.py). */
static int process_logins(World *w) {
    if (!w->pending_login) return 0;
    w->pending_login = 0;
    w->has_player = 1;
    rs_occupy(w, w->p.level, w->p.x, w->p.z, 1, 1);
    return rs_agent_login(w);
}

/* process_zones (engine.py): Loc.turn over a snapshot of loc_events, then Obj.turn over one of objs. */
static int process_zones(World *w) {
    uint8_t snap[RS_MAX_LOCS];
    int n = w->n_loc_events;
    memcpy(snap, w->loc_events, (size_t)n);
    for (int i = 0; i < n; i++) {
        int s = snap[i];
        Loc *l = &w->locs[s];
        l->lifecycle -= 1;
        if (l->lifecycle == 0) {
            if (!l->is_static && l->active) RS_TRY(rs_remove_loc(w, s, 0));
            else if (l->is_static && loc_changed(l) && l->active) RS_TRY(rs_revert_loc(w, s));
            else if (l->is_static && !l->active) RS_TRY(rs_add_loc(w, s, 0));
            else rs_set_lifecycle(w, s, -1);
        } else if (l->lifecycle < 0) {
            rs_set_lifecycle(w, s, -1);
        }
    }
    uint8_t objs[RS_MAX_OBJS];
    n = w->n_objs;
    memcpy(objs, w->obj_order, (size_t)n);
    for (int i = 0; i < n; i++) {
        Obj *o = &w->objs[objs[i]];
        if (o->ticks <= 0) continue;
        o->ticks -= 1;
        if (o->ticks == 0 && o->is_static && !o->active) {
            o->active = 1, o->ticks = -1;
        } else if (o->ticks == 0 && !o->is_static) {
            o->active = 0;   /* gone: a pickup suspended on it finds nothing */
            rs_obj_remove(w, objs[i]);
        }
    }
    return 0;
}

/* BuildArea.rebuildNormal (engine.py). */
static void process_info(World *w) {
    if (!w->has_player) return;
    Player *p = &w->p;
    int64_t ox = rs_asr(p->origin_x, 3), oz = rs_asr(p->origin_z, 3);
    if (p->x < (ox - 4) * 8 || p->z < (oz - 4) * 8 || p->x > (ox + 5) * 8 - 1 || p->z > (oz + 5) * 8 - 1)
        p->origin_x = p->x, p->origin_z = p->z;
}

/* resetEntity (engine.py). */
static void process_cleanup(World *w) {
    if (w->has_player) {
        w->p.steps_taken = 0;
        w->p.instant = 0;
    }
    for (int i = 0; i < w->n_npcs; i++) {
        Npc *n = &w->npcs[i];
        n->tele = 0;
        n->last_tick_x = n->x, n->last_tick_z = n->z;
    }
}

int rs_world_cycle(World *w) {
    reclaim_locs(w);
    rs_links_end(w);   /* requests unlinked between cycles (a close_modal from the agent's dispatch) */
    RS_TRY(process_world(w));
    process_clients_in(w);
    process_npc_events(w);
    RS_TRY(process_npcs(w));
    RS_TRY(process_players(w));
    RS_TRY(process_logins(w));
    RS_TRY(process_zones(w));
    process_info(w);
    process_cleanup(w);
    w->tick += 1;
    return 0;
}

/* ---- the canonical record of the world's state (rs.h RS_CANON_FIELDS, RS_CF_TICK..RS_CF_EVENT_LAST) ----------- */
typedef struct { int64_t *buf; int cap, n, at; } Emit;
static void em_v(Emit *e, int64_t v) {
    if (e->n < e->cap) e->buf[e->n] = v;
    e->n++;
}
static void em_begin(Emit *e, int id) { em_v(e, id); e->at = e->n; em_v(e, 0); }
static void em_end(Emit *e) { if (e->at < e->cap) e->buf[e->at] = e->n - e->at - 1; }

/* An entity reference as the oracle identity: npc nid, Loc.index, position in world.objs (-1 when gone), 0 for the
   player, -1 for none (rs.h canonical state, as sim/native_harness.py Canon). */
static void em_ref(Emit *e, const World *w, RsRef r) {
    em_v(e, r.kind);
    if (r.kind == RS_REF_NPC) { em_v(e, w->npcs[r.slot].nid); return; }
    if (r.kind == RS_REF_LOC) {
        Loc *l = rs_get_loc((World *)w, r);
        em_v(e, l ? l->index : -1);
        return;
    }
    if (r.kind == RS_REF_OBJ) {
        int64_t at = -1;
        if (rs_get_obj((World *)w, r))
            for (int i = 0; i < w->n_objs; i++)
                if (w->obj_order[i] == r.slot) at = i;
        em_v(e, at);
        return;
    }
    em_v(e, r.kind == RS_REF_PLAYER ? 0 : -1);
}

static void em_frame(Emit *e, const Frame *f) {
    em_v(e, f->serial ? f->mode : 0);
    em_v(e, f->serial ? f->depth : 0);
    for (int d = 0; f->serial && d < f->depth && d < RS_FRAME_DEPTH; d++) {
        em_v(e, f->script[d]);
        em_v(e, f->pc[d]);
    }
}

static void em_reqs(Emit *e, const World *w, const ReqList *list, int npc) {
    for (int16_t node = list->head; node >= 0; node = w->req[node].next) {
        const Req *r = &w->req[node];
        if (npc) {                       /* [id, arg, delay, source] */
            em_v(e, r->script), em_v(e, r->arg[0]), em_v(e, r->delay), em_v(e, r->arg[1]);
        } else {                         /* script, delay, args[3] */
            em_v(e, r->script), em_v(e, r->delay), em_v(e, r->arg[0]), em_v(e, r->arg[1]), em_v(e, r->arg[2]);
        }
    }
}

static void em_slots(Emit *e, const Slot *s, int cap) {
    for (int i = 0; i < cap; i++)
        if (s[i].id >= 0) em_v(e, i), em_v(e, s[i].id), em_v(e, s[i].count);
}

static int32_t msg_text(const World *w, int32_t msg) {
    return msg >= 0 && msg < w->P->n_msgs ? w->P->msgs[msg].text : -1;
}

/* rs_world_canon, the World's part of rs_canon (canon.c appends the agent's and the
   episode's fields). Python's counterpart is sim/tests/test_native_world.py world_canon. */
int rs_world_canon(const World *w, int64_t *buf, int cap, int *n, RsErr *err) {
    Emit e = {buf, cap, 0, 0};
    const Player *p = &w->p;
    int player = w->has_player || w->pending_login;
    em_begin(&e, RS_CF_TICK);
    em_v(&e, w->tick), em_v(&e, w->n_stored_locs), em_v(&e, w->next_loc), em_v(&e, w->loc_version);
    em_v(&e, w->pending_login), em_v(&e, w->has_player);
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER);
    if (player) {
        int64_t v[] = {p->x, p->z, p->level, p->instant, p->steps_taken, p->last_movement, p->delayed, p->delayed_until,
                       p->protect, p->dead, p->afk_ready, p->origin_x, p->origin_z, p->allow_repath};
        for (size_t i = 0; i < sizeof v / sizeof *v; i++) em_v(&e, v[i]);
    }
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_PATH);
    for (int i = 0; player && i < p->path_n; i++) em_v(&e, p->path[i][0]), em_v(&e, p->path[i][1]);
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_TARGET);
    if (player) {
        em_ref(&e, w, p->target);
        em_v(&e, p->target_op), em_v(&e, p->target_type), em_v(&e, p->ap_range), em_v(&e, p->ap_range_called);
        em_ref(&e, w, p->next_target);
    }
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_STATS);
    for (int i = 0; player && i < RS_N_SKILLS; i++)
        em_v(&e, p->levels[i]), em_v(&e, p->base_levels[i]), em_v(&e, p->stats[i]);
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_VARPS);
    for (int i = 0; player && i < RS_N_VARPS; i++) em_v(&e, p->varp[i]);
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_LAST);
    if (player) {
        em_v(&e, p->last_useitem), em_v(&e, p->last_com_iface), em_v(&e, p->last_com_com);
        em_v(&e, p->last_item), em_v(&e, p->last_slot), em_v(&e, p->last_useslot);
    }
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_INV);
    if (player) em_slots(&e, p->inv, RS_INV_CAP);
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_WORN);
    if (player) em_slots(&e, p->worn, RS_WORN_CAP);
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_BANK);
    if (player) em_slots(&e, p->bank, RS_BANK_CAP);
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_MODAL);
    if (player) {
        em_v(&e, p->modal), em_v(&e, p->modal_main), em_v(&e, p->modal_chat);
        for (int i = 0; i < p->n_resume; i++) em_v(&e, p->resume_iface[i]), em_v(&e, p->resume_com[i]);
    }
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_SCRIPT);
    if (player) em_frame(&e, &p->script);
    em_end(&e);
    em_begin(&e, RS_CF_PLAYER_TIMERS);
    for (int i = 0; player && i < p->n_timers; i++)
        em_v(&e, p->timer[i].script), em_v(&e, p->timer[i].kind), em_v(&e, p->timer[i].interval),
            em_v(&e, p->timer[i].clock);
    em_end(&e);
    const ReqList *queues[3] = {&p->queue, &p->weak, &p->engine};
    for (int q = 0; q < 3; q++) {
        em_begin(&e, RS_CF_PLAYER_QUEUE + q);
        if (player) em_reqs(&e, w, queues[q], 0);
        em_end(&e);
    }
    em_begin(&e, RS_CF_NPC);
    for (int i = 0; i < w->n_npcs; i++) {
        const Npc *m = &w->npcs[i];
        int64_t v[] = {m->nid, m->type, m->base, m->x, m->z, m->level, m->active, m->delayed, m->delayed_until,
                       m->mode, m->target_player, m->timer_interval, m->timer_clock, m->hunt_clock, m->regen_clock,
                       m->stuck, m->tele, m->observers, m->last_tick_x, m->last_tick_z, m->hunt_mode, m->huntrange,
                       m->hunt_target_player, m->lifecycle, m->last_movement, m->life, m->has_path,
                       m->has_path ? m->path_x : 0, m->has_path ? m->path_z : 0, m->has_hero,
                       m->has_hero ? m->hero_damage : 0, m->v_action_delay, m->v_lastcombat, m->v_aggressive_player,
                       m->v_attacking, m->v_xp_mult, m->v_stunned};
        for (size_t k = 0; k < sizeof v / sizeof *v; k++) em_v(&e, v[k]);
        for (int k = 0; k < 6; k++) em_v(&e, m->levels[k]);
    }
    em_end(&e);
    em_begin(&e, RS_CF_NPC_SCRIPT);
    for (int i = 0; i < w->n_npcs; i++) em_v(&e, w->npcs[i].nid), em_frame(&e, &w->npcs[i].script);
    em_end(&e);
    em_begin(&e, RS_CF_NPC_QUEUE);
    for (int i = 0; i < w->n_npcs; i++) {
        int64_t k = 0;
        for (int16_t node = w->npcs[i].queue.head; node >= 0; node = w->req[node].next) k++;
        em_v(&e, w->npcs[i].nid), em_v(&e, k);
        em_reqs(&e, w, &w->npcs[i].queue, 1);
    }
    em_end(&e);
    em_begin(&e, RS_CF_NPC_EVENTS);
    for (int i = 0; i < w->n_npc_events; i++) em_v(&e, w->npcs[w->npc_events[i]].nid);
    em_end(&e);
    em_begin(&e, RS_CF_LOCS);
    {
        uint8_t s[RS_MAX_LOCS];
        int k = 0;
        for (int i = 0; i < w->n_loc_slots; i++)
            if (w->locs[i].serial && (w->locs[i].is_static || w->locs[i].active)) s[k++] = (uint8_t)i;
        sort_by_index(w, s, k);
        for (int i = 0; i < k; i++) {
            const Loc *l = &w->locs[s[i]];
            int64_t v[] = {l->index, l->base, l->type, l->x, l->z, l->shape, l->angle, l->width, l->length,
                           l->is_static, l->active, l->lifecycle, l->layer};
            for (size_t j = 0; j < sizeof v / sizeof *v; j++) em_v(&e, v[j]);
        }
    }
    em_end(&e);
    em_begin(&e, RS_CF_ZONES);
    {
        const RsLayout *L = w->L;
        int zh = (L->H + 7) >> 3, zw = (L->W + 7) >> 3;
        for (int zx = 0; zx < zw; zx++)
            for (int zz = 0; zz < zh; zz++) {
                int zi = zx * zh + zz;
                if (!w->zone_n[zi]) continue;
                em_v(&e, (L->x0 >> 3) + zx), em_v(&e, (L->z0 >> 3) + zz), em_v(&e, w->zone_n[zi]);
                for (int i = 0; i < w->zone_n[zi]; i++) em_v(&e, w->locs[w->zone[zi][i]].index);
            }
    }
    em_end(&e);
    em_begin(&e, RS_CF_LOC_EVENTS);
    for (int i = 0; i < w->n_loc_events; i++) em_v(&e, w->locs[w->loc_events[i]].index);
    em_end(&e);
    em_begin(&e, RS_CF_OBJS);
    for (int i = 0; i < w->n_objs; i++) {
        const Obj *o = &w->objs[w->obj_order[i]];
        int64_t v[] = {o->type, o->count, o->x, o->z, o->level, o->is_static, o->ticks, o->receiver_player, o->active};
        for (size_t j = 0; j < sizeof v / sizeof *v; j++) em_v(&e, v[j]);
    }
    em_end(&e);
    em_begin(&e, RS_CF_OCCUPIED);   /* tiles in x-major order are (x, z) sorted */
    for (int t = 0; t < w->L->T; t++)
        if (w->occ[t]) em_v(&e, w->L->x0 + t / w->L->H), em_v(&e, w->L->z0 + t % w->L->H), em_v(&e, w->occ[t]);
    em_end(&e);
    em_begin(&e, RS_CF_CHANGED);
    for (int t = 0; t < w->L->T; t++)
        if (w->changed[t >> 6] >> (t & 63) & 1) em_v(&e, w->L->x0 + t / w->L->H), em_v(&e, w->L->z0 + t % w->L->H);
    em_end(&e);
    em_begin(&e, RS_CF_LOGS);
    {
        const ChatLog *c = &w->chat;
        const RsChatLine *last = c->n ? &c->ring[(c->n - 1) % RS_CHAT_RING] : NULL;
        em_v(&e, c->n), em_v(&e, last ? msg_text(w, last->msg) : -1), em_v(&e, last ? last->tick : -1);
        const RsTextLine *t = w->text.n ? &w->text.line[w->text.n - 1] : NULL;
        em_v(&e, w->text.n), em_v(&e, t ? t->iface : -1), em_v(&e, t ? t->com : -1);
        em_v(&e, t ? msg_text(w, t->msg) : -1);
        em_v(&e, w->ev_base + w->n_events);   /* len(world.events): a free-play life keeps the last event */
    }
    em_end(&e);
    em_begin(&e, RS_CF_EVENT_LAST);
    if (w->n_events) {
        const RsEvent *v = &w->ev[w->n_events - 1];
        em_v(&e, v->kind), em_v(&e, v->tick), em_v(&e, v->a8), em_v(&e, v->b8), em_v(&e, v->has_source);
        em_v(&e, v->obj), em_v(&e, v->count), em_v(&e, v->v0), em_v(&e, v->v1), em_v(&e, v->v2), em_v(&e, v->v3);
    }
    em_end(&e);
    *n = e.n;
    if (e.n > cap) return RS_OVERFLOW(err, "the canonical record's buffer");
    return 0;
}
