/* sim/native/obs.c: the observation of an agent (sim/agent.py scan, build_obs, project_items, build_mask,
 * snapshot, describe_action, feedback; rl/feedback_contract.py episode_start) and its projection (rl/obs.py project
 * with the act, item and feedback contracts): obs-v3 arrays, the mask-v3 rule and feedback-v2, bit for bit.
 *
 * Every function mirrors the Python one its comment names (file and function). Python's pure memo
 * caches (Agent._scan_locs, Agent._loc_cache) are dropped: their keys cover every input, so recomputing gives the
 * same lists. Agent.steps(c) for a loc is walk_approach: the four side tiles operable_steps reads lie inside the 3x3
 * walk_approach scans, so min(side, beside) is beside; only its missing-tile refusal is kept.
 * Floats follow rs.h FLOATS: every feature is the Python double expression cast to float once; log1p features read
 * the batch's tables (RsSlot.log1p); text vectors are text.c's. Contract validators that cannot fire while the agent's
 * invariants hold are not repeated, except those an engine state can reach (an occupied slot of count 0, a busy
 * option without its action), which raise their ValueError as RS_E_CONTRACT. */
#include <string.h>

#include "rs.h"

static inline int32_t imax(int32_t a, int32_t b) { return a > b ? a : b; }
static inline int32_t imin(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t iabs(int32_t a) { return a < 0 ? -a : a; }
static inline int32_t cheb(int32_t ax, int32_t az, int32_t bx, int32_t bz) { return imax(iabs(ax - bx), iabs(az - bz)); }
/* rl/obs.py _clip: max(lo, min(hi, v)) */
static inline double clip(double v, double lo, double hi) { double m = hi <= v ? hi : v; return lo >= m ? lo : m; }
/* Python min(v, cap) with a one-sided cap */
static inline double cap(double v, double c) { return c < v ? c : v; }

static inline int suspended(const Player *p) { return p->script.serial && p->script.mode == RS_SCRIPT_PAUSE; }

/* ---- type names and option lists ----------------------------------------------------------------------------- */
/* name_of (agent.py) and the visible ops ([o for o in op if o and o !=
   'hidden'], in op order, not deduped) of a loc, npc or obj type (obj: its iop list). */
static int32_t name_of(const RsPack *P, int kind, int32_t type) {
    return kind == RS_KIND_LOC ? P->loc_types[type].name_of : kind == RS_KIND_NPC ? P->npc_types[type].name_of
        : P->obj_types[type].name_of;
}

static int visible_ops(const RsPack *P, int kind, int32_t type, int32_t out[5]) {
    const int32_t *op;
    const uint8_t *index;
    int n;
    if (kind == RS_KIND_LOC) {
        const RsLocType *t = &P->loc_types[type];
        op = t->op, index = t->opt_index, n = t->n_opt;
    } else if (kind == RS_KIND_NPC) {
        const RsNpcType *t = &P->npc_types[type];
        op = t->op, index = t->opt_index, n = t->n_opt;
    } else {
        const RsObjType *t = &P->obj_types[type];
        op = t->iop, index = t->iopt_index, n = t->n_iopt;
    }
    for (int k = 0; k < n; k++) out[k] = op[index[k]];
    return n;
}

/* list(dict.fromkeys(options)): first occurrences of each string, in order (a string id is its string) */
static int dedupe(const int32_t *in, int n, int32_t out[5]) {
    int m = 0;
    for (int k = 0; k < n; k++) {
        int seen = 0;
        for (int j = 0; j < m; j++) seen |= out[j] == in[k];
        if (!seen && in[k]) out[m++] = in[k];
    }
    return m;
}

/* ---- the scan (agent.py) ------------------------------------------------------------------------------ */
/* Locs come from locs_near in Loc.index order, then a stable sort on (distance, x, z, type); npcs (nid order) on
   (distance, x, z, nid); objs, in world.objs order, a stable sort on (distance, x, z, type). */
static void scan(RsSlot *s) {
    World *w = &s->w;
    Agent *a = &s->a;
    const Player *p = &w->p;
    uint8_t *l = a->scan_locs;
    int n = rs_locs_near(w, p->level, p->x, p->z, RS_LOC_SCAN_TILES, l);
    for (int i = 1; i < n; i++) {
        uint8_t v = l[i];
        const Loc *c = &w->locs[v];
        int32_t d = cheb(p->x, p->z, c->x, c->z), j = i;
        for (; j > 0; j--) {
            const Loc *b = &w->locs[l[j - 1]];
            int32_t e = cheb(p->x, p->z, b->x, b->z);
            if (e < d || (e == d && (b->x < c->x || (b->x == c->x && (b->z < c->z || (b->z == c->z && b->type <= c->type))))))
                break;
            l[j] = l[j - 1];
        }
        l[j] = v;
    }
    a->n_scan_locs = n;
    uint8_t *q = a->scan_npcs;
    n = 0;
    for (int i = 0; i < w->n_npcs; i++) {
        const Npc *c = &w->npcs[i];
        if (!c->active || c->level != p->level || cheb(p->x, p->z, c->x, c->z) > RS_NPC_SCAN_TILES) continue;
        int32_t d = cheb(p->x, p->z, c->x, c->z), j = n++;
        for (; j > 0; j--) {
            const Npc *b = &w->npcs[q[j - 1]];
            int32_t e = cheb(p->x, p->z, b->x, b->z);
            if (e < d || (e == d && (b->x < c->x || (b->x == c->x && (b->z < c->z || (b->z == c->z && b->nid < c->nid))))))
                break;
            q[j] = q[j - 1];
        }
        q[j] = (uint8_t)i;
    }
    a->n_scan_npcs = n;
    uint8_t *o = a->scan_objs;
    n = 0;
    for (int k = 0; k < w->n_objs; k++) {
        const Obj *c = &w->objs[w->obj_order[k]];
        if (!(c->active && c->count >= 1) || c->level != p->level || cheb(p->x, p->z, c->x, c->z) > RS_GROUND_SCAN_TILES)
            continue;
        int32_t d = cheb(p->x, p->z, c->x, c->z), j = n++;
        for (; j > 0; j--) {
            const Obj *b = &w->objs[o[j - 1]];
            int32_t e = cheb(p->x, p->z, b->x, b->z);
            if (e < d || (e == d && (b->x < c->x || (b->x == c->x && (b->z < c->z || (b->z == c->z && b->type <= c->type))))))
                break;
            o[j] = o[j - 1];
        }
        o[j] = w->obj_order[k];
    }
    a->n_scan_objs = n;
}

static void list_item(const RsPack *P, const Slot *slot, int k, int options, RsItem *out) {
    out->slot = (int16_t)k, out->id = slot->id, out->count = slot->count;
    out->name = P->obj_types[slot->id].name_of;
    out->n_opt = options ? (uint8_t)visible_ops(P, RS_KIND_OBJ, slot->id, out->opt) : 0;
}

/* Agent.build_obs (agent.py) after scan: what project_items and the mask read of the dict. Fields
   project_items writes (pages, item pages, using, affordances, clicks, style, bankOpen) and the
   feedback start empty; optionBusy and optionAction (agent.py, set by the agent after project_items) start
   as an observation without them (False, -1). */
int rs_build_obs(RsSlot *s, int32_t decision, int done, int truncated, int is_dead) {
    const RsPack *P = s->P;
    World *w = &s->w;
    const Player *p = &w->p;
    Agent *a = &s->a;
    RsRawObs *o = &a->obs;
    scan(s);
    memset(o, 0, sizeof *o);
    o->tick = w->tick, o->decision = decision;
    o->done = (uint8_t)(done != 0), o->truncated = (uint8_t)(truncated != 0), o->is_dead = (uint8_t)(is_dead != 0);
    o->option_action = -1;
    for (int i = 0; i < RS_N_SKILLS; i++)
        o->levels[i] = p->levels[i], o->base_levels[i] = p->base_levels[i], o->xp[i] = p->stats[i] / 10;
    for (int k = 0; k < RS_INV_CAP; k++)
        if (p->inv[k].id >= 0) list_item(P, &p->inv[k], k, 1, &o->inventory[o->n_inv++]);
    for (int k = 0; k < RS_WORN_CAP; k++)
        if (p->worn[k].id >= 0) list_item(P, &p->worn[k], k, 0, &o->equipment[o->n_worn++]);
    o->target_index = -1;
    for (int i = 0; i < a->n_scan_npcs; i++) {
        const Npc *n = &w->npcs[a->scan_npcs[i]];
        if (n->target_player && !o->in_combat) o->in_combat = 1, o->target_index = n->nid;
        if (i >= RS_N_NPC_OBS) continue;
        const RsNpcType *t = &P->npc_types[n->type];
        o->npcs[o->n_npcs++] = (RsNpcObs){.nid = n->nid, .name = t->name_of, .combat_level = t->vislevel0,
            .dx = (int16_t)(n->x - p->x), .dz = (int16_t)(n->z - p->z), .distance = (int16_t)cheb(p->x, p->z, n->x, n->z),
            .in_combat = n->target_player, .attack_op = t->attack_op};
    }
    /* act_contract.banks: the banks among the listed locs (the first N_LOC_OBS scanned) and npcs */
    o->n_banks = 0;
    for (int i = 0; i < a->n_scan_locs && i < RS_N_LOC_OBS; i++) {
        const Loc *l = &w->locs[a->scan_locs[i]];
        if (P->loc_types[l->type].bank_listed) o->bank_x[o->n_banks] = l->x, o->bank_z[o->n_banks++] = l->z;
    }
    for (int i = 0; i < a->n_scan_npcs && i < RS_N_NPC_OBS; i++) {
        const Npc *n = &w->npcs[a->scan_npcs[i]];
        if (P->npc_types[n->type].bank_listed) o->bank_x[o->n_banks] = n->x, o->bank_z[o->n_banks++] = n->z;
    }
    o->n_dialog_options = suspended(p) ? p->n_resume : 0;
    o->px = p->x, o->pz = p->z, o->plevel = p->level;
    o->hp = p->levels[RS_SKILL_HITPOINTS], o->max_hp = p->base_levels[RS_SKILL_HITPOINTS];
    o->combat_level = rs_combat_level(w);
    o->dialog_open = (uint8_t)(suspended(p) || p->modal_chat != -1);
    o->modal_open = (uint8_t)(p->modal != 0);
    return 0;
}

/* ---- project_items (agent.py) ------------------------------------------------------------------------- */
static const RsItem EMPTY_ITEM = {.slot = -1, .id = -1};

/* items.ts itemPage (agent.py) over items listed in slot order */
static void item_page(const RsItem *items, int n, int page, int capacity, RsItem out[8]) {
    for (int k = 0; k < RS_N_INV_SEL; k++) {
        int slot = page * 8 + k;
        out[k] = EMPTY_ITEM;
        if (slot >= capacity) continue;
        for (int i = 0; i < n; i++)
            if (items[i].slot == slot) { out[k] = items[i]; break; }
    }
}

typedef struct {               /* act_candidates' dict (agent.py) */
    uint8_t kind, level;
    int16_t x, z;
    int32_t id, name, combat;
    RsRef ref;
    int n_opt;
    int32_t opt[5];
} Cand;

typedef struct {               /* an affordance entry (agent.py) and its candidate's entity */
    RsAff a;
    RsRef ref;
    int32_t option_lc, name_lc;
    int16_t group, next;       /* _group index; next entry of the group (-1) */
    uint8_t chosen;
} Entry;

static int refused(const Agent *a, const RsAffKey *k, int32_t tick) {
    /* self.refused.get(affordance_key(a), -1) > tick (agent.py) */
    for (int i = 0; i < a->n_refused; i++) {
        const RsAffKey *r = &a->refused[i].key;
        if (r->kind == k->kind && r->id == k->id && r->x == k->x && r->z == k->z && r->level == k->level
            && r->option_lc == k->option_lc)
            return a->refused[i].until > tick;
    }
    return 0;
}

/* affordance_entries (agent.py) up to the counts: one entry per new option of each candidate in reach. */
static int affordance_entries(RsSlot *s, const Cand *c, int nc, int32_t src, Entry *e, int *n) {
    const RsPack *P = s->P;
    World *w = &s->w;
    const Player *p = &w->p;
    for (int i = 0; i < nc; i++) {
        if (c[i].level != p->level || !c[i].name || cheb(c[i].x, c[i].z, p->x, p->z) > RS_ACT_SCAN_TILES) continue;
        int32_t options[5];
        int no = dedupe(c[i].opt, c[i].n_opt, options);
        if (!no) continue;
        int32_t walk;
        if (c[i].kind == RS_KIND_LOC && rs_tile(w->L, p->level, c[i].x, c[i].z) < 0)
            return RS_FAIL(&w->err, RS_E_NO_WALL_DATA, c[i].x, c[i].z);   /* operable_steps (agent.py) */
        if (rs_walk_approach(w, src, p->x, p->z, c[i].x, c[i].z, &walk)) return rs_wrap_outside(&w->err);
        if (walk < 0) continue;
        for (int k = 0; k < no; k++) {
            int32_t lc = P->str_lower[options[k]];
            RsAffKey key = {c[i].kind, c[i].level, c[i].x, c[i].z, c[i].id, lc};
            int seen = 0;
            for (int j = 0; j < *n && !seen; j++) {
                const RsAff *b = &e[j].a;
                seen = b->kind == key.kind && b->id == key.id && b->x == key.x && b->z == key.z && b->level == key.level
                    && e[j].option_lc == lc;
            }
            if (seen || refused(&s->a, &key, w->tick)) continue;
            if (*n >= RS_AFF_CAP) return RS_OVERFLOW(&w->err, "RS_AFF_CAP affordance entries");
            Entry *x = &e[(*n)++];
            x->a = (RsAff){.kind = c[i].kind, .level = c[i].level, .x = c[i].x, .z = c[i].z,
                           .dx = (int16_t)(c[i].x - p->x), .dz = (int16_t)(c[i].z - p->z),
                           .distance = (int16_t)cheb(c[i].x, c[i].z, p->x, p->z), .id = c[i].id, .name = c[i].name,
                           .option = options[k], .steps = walk, .count = 1, .combat_level = c[i].combat};
            x->ref = c[i].ref, x->option_lc = lc, x->name_lc = P->str_lower[c[i].name], x->chosen = 0;
        }
    }
    return 0;
}

/* The counts (agent.py): entries of the same _group (kind, name lower, option lower) within
   ACT_CLUSTER_TILES, the entry itself included, bucketed by group. Python counts the loc list and the npc and obj
   list apart; the group names the kind, so one pass over both is the same. */
static void count_groups(Entry *e, int n) {
    int16_t head[RS_AFF_CAP];
    int groups = 0;
    for (int i = 0; i < n; i++) {
        int g = 0;
        for (; g < groups; g++) {
            const Entry *h = &e[head[g]];
            if (h->a.kind == e[i].a.kind && h->name_lc == e[i].name_lc && h->option_lc == e[i].option_lc) break;
        }
        if (g == groups) head[groups++] = (int16_t)i, e[i].next = -1;
        else e[i].next = head[g], head[g] = (int16_t)i;
        e[i].group = (int16_t)g;
    }
    for (int g = 0; g < groups; g++)
        for (int16_t i = head[g]; i >= 0; i = e[i].next) {
            int32_t count = 0;
            for (int16_t j = head[g]; j >= 0; j = e[j].next)
                count += cheb(e[i].a.x, e[i].a.z, e[j].a.x, e[j].a.z) <= RS_ACT_CLUSTER_TILES;
            e[i].a.count = count;
        }
}

/* choose_affordances' order (agent.py): (steps, x, z, kind, type id, option), the option by code points. */
static int aff_before(const RsPack *P, const RsAff *x, const RsAff *y) {
    if (x->steps != y->steps) return x->steps < y->steps;
    if (x->x != y->x) return x->x < y->x;
    if (x->z != y->z) return x->z < y->z;
    if (x->kind != y->kind) return x->kind < y->kind;
    if (x->id != y->id) return x->id < y->id;
    return P->str_rank[x->option] < P->str_rank[y->option];
}

/* choose_affordances (agent.py): the first of each group by the order, then the nearest left, up to N_ACT,
   in the order (keys are unique after the dedupe, so a mark per entry and one sorted pass give Python's lists). */
static int choose_affordances(const RsPack *P, Entry *e, int n, int16_t out[RS_N_ACT]) {
    int16_t order[RS_AFF_CAP];
    for (int i = 0; i < n; i++) {
        int j = i;
        for (; j > 0 && aff_before(P, &e[i].a, &e[order[j - 1]].a); j--) order[j] = order[j - 1];
        order[j] = (int16_t)i;
    }
    uint8_t covered[RS_AFF_CAP] = {0};
    int chosen = 0;
    for (int k = 0; k < n && chosen < RS_N_ACT; k++) {
        Entry *x = &e[order[k]];
        if (!covered[x->group]) covered[x->group] = 1, x->chosen = 1, chosen++;
    }
    for (int k = 0; k < n && chosen < RS_N_ACT; k++)
        if (!e[order[k]].chosen) e[order[k]].chosen = 1, chosen++;
    int m = 0;
    for (int k = 0; k < n; k++)
        if (e[order[k]].chosen) out[m++] = order[k];
    return m;
}

/* select_held (agent.py) over the inventory page. */
static int select_held(const RsPack *P, const RsRawObs *o, RsHeld out[RS_N_HELD]) {
    int n = 0;
    for (int k = 0; k < RS_N_INV_SEL; k++) {
        const RsItem *it = &o->inv_sel[k];
        if (!(it->count > 0) || (o->has_using && it->slot == o->using_.slot)) continue;
        int32_t options[5];
        int no = o->has_using ? (options[0] = P->consts->s_use, 1) : dedupe(it->opt, it->n_opt, options);
        for (int j = 0; j < no; j++) {
            if (n == RS_N_HELD) return n;
            if (!options[j] || P->str_lower[options[j]] == P->str_lower[P->consts->s_drop]) continue;
            out[n++] = (RsHeld){.slot = (int8_t)it->slot, .id = it->id, .count = it->count, .name = it->name,
                                .option = options[j]};
        }
    }
    return n;
}

int rs_project_items(RsSlot *s) {
    const RsPack *P = s->P;
    World *w = &s->w;
    const Player *p = &w->p;
    Agent *a = &s->a;
    RsRawObs *o = &a->obs;
    o->bank_open = (uint8_t)(p->modal_main >= 0 && p->modal_main == P->consts->s_bank_main);
    RsItem bank[RS_BANK_CAP];
    int nb = 0, top = -1;
    if (o->bank_open)
        for (int k = 0; k < RS_BANK_CAP; k++)
            if (p->bank[k].id >= 0) list_item(P, &p->bank[k], k, 0, &bank[nb++]), top = k;
    a->bank_pages = imax(1, (top + 1 + RS_N_INV_SEL - 1) / RS_N_INV_SEL);
    a->bank_page = o->bank_open ? imin(a->bank_page, a->bank_pages - 1) : 0;
    o->inv_page = a->inv_page, o->bank_page = a->bank_page, o->bank_pages = a->bank_pages;
    item_page(o->inventory, o->n_inv, a->inv_page, RS_INV_CAP, o->inv_sel);
    item_page(bank, nb, a->bank_page, RS_BANK_CAP, o->bank_sel);
    if (a->has_using) {
        const Slot *picked = &p->inv[a->using_slot];
        if (picked->id < 0 || picked->id != a->using_id || o->bank_open) a->has_using = 0;
    }
    o->has_using = a->has_using;
    if (a->has_using) o->using_.slot = a->using_slot, o->using_.id = a->using_id, o->using_.name = a->using_name;
    int32_t src = rs_source(w->L, p->level, p->x, p->z);
    if (src < 0) return RS_FAIL0(&w->err, RS_E_PLAYER_OUTSIDE);

    /* act_candidates (agent.py); while an item is picked, the npcs and locs with the one option Use */
    Cand cand[RS_SCAN_LOCS + RS_SCAN_NPCS + RS_SCAN_OBJS];
    int nl = 0, nc;
    for (int i = 0; i < a->n_scan_locs; i++) {
        const Loc *l = &w->locs[a->scan_locs[i]];
        Cand *c = &cand[nl++];
        *c = (Cand){RS_KIND_LOC, l->level, l->x, l->z, l->type, name_of(P, RS_KIND_LOC, l->type), 0,
                    rs_ref_loc(w, a->scan_locs[i]), 0, {0}};
        c->n_opt = o->has_using ? (c->opt[0] = P->consts->s_use, 1) : visible_ops(P, RS_KIND_LOC, l->type, c->opt);
    }
    nc = nl;
    for (int i = 0; i < a->n_scan_npcs; i++) {
        const Npc *v = &w->npcs[a->scan_npcs[i]];
        if (cheb(p->x, p->z, v->x, v->z) > RS_ACT_SCAN_TILES) continue;
        Cand *c = &cand[nc++];
        *c = (Cand){RS_KIND_NPC, v->level, v->x, v->z, v->type, name_of(P, RS_KIND_NPC, v->type),
                    P->npc_types[v->type].vislevel0, rs_ref_npc(w, a->scan_npcs[i]), 0, {0}};
        c->n_opt = o->has_using ? (c->opt[0] = P->consts->s_use, 1) : visible_ops(P, RS_KIND_NPC, v->type, c->opt);
    }
    if (!o->has_using)
        for (int i = 0; i < a->n_scan_objs; i++) {
            const Obj *v = &w->objs[a->scan_objs[i]];
            cand[nc++] = (Cand){RS_KIND_OBJ, v->level, v->x, v->z, v->type, name_of(P, RS_KIND_OBJ, v->type), 0,
                                rs_ref_obj(w, a->scan_objs[i]), 1, {P->consts->s_take}};
        }
    /* in Python's order: loc entries, then the other entries */
    Entry entries[RS_AFF_CAP];
    int ne = 0;
    RS_TRY(affordance_entries(s, cand, nl, src, entries, &ne));
    RS_TRY(affordance_entries(s, cand + nl, nc - nl, src, entries, &ne));
    count_groups(entries, ne);
    int16_t chosen[RS_N_ACT];
    o->n_act = a->n_act_targets = choose_affordances(P, entries, ne, chosen);
    for (int k = 0; k < o->n_act; k++) o->act_sel[k] = entries[chosen[k]].a, a->act_targets[k] = entries[chosen[k]].ref;
    o->n_held = select_held(P, o, o->held_sel);
    /* Agent.click_slots (clicks.ts clickSlots): a slot's target in the declared world (the layout) and whether its live
       collision has none of WALK_BLOCKED (walls and npcs aside; FLAG_NULL off the layout) */
    for (int k = 0; k < RS_N_CLICK; k++) {
        int32_t x = p->x + RS_CLICK_DX[k], z = p->z + RS_CLICK_DZ[k];
        o->clicks[k].inside = (uint8_t)(rs_tile(w->L, 0, x, z) >= 0);
        o->clicks[k].open = (uint8_t)!((uint32_t)rs_flags(w, p->level, x, z, 0) & (uint32_t)RS_WALK_BLOCKED);
    }
    /* the attacker: the first scanned npc whose target is the player */
    const Npc *attacker = NULL;
    for (int i = 0; i < a->n_scan_npcs && !attacker; i++)
        if (w->npcs[a->scan_npcs[i]].target_player) attacker = &w->npcs[a->scan_npcs[i]];
    int32_t rhand = p->worn[RS_WEARPOS_RHAND].id;
    o->style.mode = p->varp[RS_VARP_com_mode], o->style.count = rs_weapon_styles(w, rhand >= 0 ? rhand : -1)->n;
    o->style_stale = a->style_stale;
    o->has_attacker = attacker != NULL;
    if (attacker)
        o->attacker_name = P->npc_types[attacker->type].name_of, o->attacker_dx = (int16_t)(attacker->x - p->x),
        o->attacker_dz = (int16_t)(attacker->z - p->z);
    a->has_obs = 1;
    return 0;
}

/* ---- the mask (agent.py build_mask, mask-v5) ------------------------------------------------------------------- */
int rs_build_mask(RsSlot *s) {
    const RsPack *P = s->P;
    RsRawObs *o = &s->a.obs;
    int8_t *m = o->mask;
    if (!(o->style.mode >= 0 && o->style.count >= 1)) return RS_FAIL0(&s->w.err, RS_E_INVALID_STYLE);
    memset(m, 0, RS_N_ACTIONS);
    m[RS_A_WAIT] = 1;
    if (o->bank_open) {
        m[RS_A_CLOSE_IF] = 1;
        if (o->inv_page > 0) m[RS_A_INV_PREV] = 1;
        if (o->inv_page < 3) m[RS_A_INV_NEXT] = 1;
        if (o->bank_page > 0) m[RS_A_BANK_PREV] = 1;
        if (o->bank_page + 1 < o->bank_pages) m[RS_A_BANK_NEXT] = 1;
        for (int k = 0; k < RS_N_INV_SEL; k++) {
            if (o->inv_sel[k].count > 0) m[RS_A_DEPOSIT0 + k] = 1;
            if (o->bank_sel[k].count > 0) m[RS_A_WITHDRAW0 + k] = 1;
        }
        return 0;
    }
    if (o->has_using) {
        if (o->inv_page > 0) m[RS_A_INV_PREV] = 1;   /* mask-v5: the page turns, the selection kept */
        if (o->inv_page < 3) m[RS_A_INV_NEXT] = 1;
        for (int k = 0; k < o->n_act; k++) m[RS_A_ACT0 + k] = 1;
        for (int k = 0; k < o->n_held; k++) m[RS_A_HELD0 + k] = 1;
        m[RS_A_CLOSE_IF] = 1;
        return 0;
    }
    if (o->dialog_open || o->modal_open) {
        /* dialogOptions' texts are '' in the simulator (agent.py), so none reads 'click here to continue' */
        if (o->n_dialog_options)
            for (int k = 0; k < imin(o->n_dialog_options, RS_N_DIALOG_CHOICE); k++) m[RS_A_DIALOG_CHOICE0 + k] = 1;
        else if (o->dialog_open)
            m[RS_A_DIALOG_CONTINUE] = 1;
        m[RS_A_CLOSE_IF] = 1;
        return 0;
    }
    if (o->inv_page > 0) m[RS_A_INV_PREV] = 1;
    if (o->inv_page < 3) m[RS_A_INV_NEXT] = 1;
    for (int k = 0; k < o->n_act; k++) m[RS_A_ACT0 + k] = 1;
    for (int k = 0; k < RS_N_CLICK; k++)   /* a click into the declared world, routed or not */
        if (o->clicks[k].inside) m[RS_CLICK_ROWS[k]] = 1;
    for (int k = 0; k < o->n_held; k++) m[RS_A_HELD0 + k] = 1;
    int pressure = o->n_inv >= RS_DROP_MIN_STACKS;
    for (int k = 0; k < RS_N_INV_SEL; k++) {
        const RsItem *it = &o->inv_sel[k];
        if (pressure && it->count > 0 && !(P->obj_types[it->id].flags & RS_OF_TOOL_PROTECT)) m[RS_A_DROP0 + k] = 1;
        if (it->count > 0) m[RS_A_USE0 + k] = 1;
    }
    int32_t current = imin(o->style.mode, o->style.count - 1);
    for (int k = 0; k < 4; k++)
        if (o->style_stale || imin(k, o->style.count - 1) != current) m[RS_A_STYLE0 + k] = 1;
    return 0;
}

/* ---- snapshot, describe, feedback (agent.py, 1039-1105; feedback_contract.py) ------------------ */
static void snap_add(RsSnap *out, int32_t obj, int64_t count) {
    for (int i = 0; i < out->n_items; i++)
        if (out->item_obj[i] == obj) { out->item_count[i] += count; return; }
    out->item_obj[out->n_items] = obj, out->item_count[out->n_items++] = count;
}

void rs_snapshot(const RsSlot *s, RsSnap *out) {
    const World *w = &s->w;
    const Player *p = &w->p;
    memset(out, 0, sizeof *out);
    out->tick = w->tick, out->x = p->x, out->z = p->z, out->level = p->level, out->hp = p->levels[RS_SKILL_HITPOINTS];
    for (int i = 0; i < RS_N_SKILLS; i++) out->xp += p->stats[i];
    for (int k = 0; k < RS_INV_CAP; k++)
        if (p->inv[k].id >= 0) snap_add(out, p->inv[k].id, p->inv[k].count);
    for (int k = 0; k < RS_WORN_CAP; k++)
        if (p->worn[k].id >= 0) snap_add(out, p->worn[k].id, p->worn[k].count);
    out->bank_open = (uint8_t)(p->modal_main >= 0 && p->modal_main == s->P->consts->s_bank_main);
    out->dialog_open = (uint8_t)(p->modal_chat != -1 || suspended(p));
}

/* feedback.ts familyOf (agent.py family); -1 for RETREAT (71), which has none (ValueError there, and the task refuses
   it as masked before the agent sees one) */
int rs_action_family(int32_t a) {
    if (a == RS_A_WAIT) return RS_ACTFAM_WAIT;
    if ((a >= RS_A_CLICK_LOW && a < RS_A_CLICK_LOW + RS_N_CLICK_LOW)
        || (a >= RS_A_CLICK_HIGH && a < RS_A_CLICK_HIGH + RS_N_CLICK - RS_N_CLICK_LOW)) return RS_ACTFAM_CLICK;
    if (a >= RS_A_ACT0 && a < RS_A_ACT0 + RS_N_ACT) return RS_ACTFAM_ACT;
    if (a >= RS_A_HELD0 && a < RS_A_HELD0 + RS_N_HELD) return RS_ACTFAM_HELD;
    if (a >= RS_A_DROP0 && a < RS_A_DROP0 + RS_N_INV_SEL) return RS_ACTFAM_DROP;
    if (a >= RS_A_DEPOSIT0 && a < RS_A_DEPOSIT0 + RS_N_INV_SEL) return RS_ACTFAM_DEPOSIT;
    if (a >= RS_A_WITHDRAW0 && a < RS_A_WITHDRAW0 + RS_N_INV_SEL) return RS_ACTFAM_WITHDRAW;
    if (a == RS_A_CLOSE_IF) return RS_ACTFAM_CLOSE;
    if (a == RS_A_DIALOG_CONTINUE || (a >= RS_A_DIALOG_CHOICE0 && a < RS_A_DIALOG_CHOICE0 + 4)) return RS_ACTFAM_DIALOG;
    if (a >= RS_A_INV_PREV && a < RS_A_INV_PREV + 4) return RS_ACTFAM_PAGE;
    if (a >= RS_A_STYLE0 && a < RS_A_STYLE0 + 4) return RS_ACTFAM_STYLE;
    if (a >= RS_A_USE0 && a < RS_A_USE0 + RS_N_INV_SEL) return RS_ACTFAM_USE;
    return -1;
}

/* describe_action (agent.py) on last_observation (s->a.obs; none before the first) and the agent's using. */
void rs_describe_action(const RsSlot *s, int32_t action, RsMeaning *out) {
    const RsPack *P = s->P;
    const Agent *a = &s->a;
    const RsRawObs *o = a->has_obs ? &a->obs : NULL;
    int f = rs_action_family(action);
    *out = (RsMeaning){.action = action, .family = (uint8_t)(f < 0 ? 255 : f), .kind = RS_KIND_NONE};
    out->item = o && a->has_using ? a->using_name : 0;
    if (!o) return;
    if (o->has_using) out->item = o->using_.name;
    int32_t k;
    switch (f) {
    case RS_ACTFAM_ACT:
        k = action - RS_A_ACT0;
        if (k < o->n_act) out->kind = o->act_sel[k].kind, out->name = o->act_sel[k].name, out->option = o->act_sel[k].option;
        break;
    case RS_ACTFAM_CLICK:   /* the slot's offset as "dx,dz" */
        k = action >= RS_A_CLICK_HIGH ? action - RS_A_CLICK_HIGH + RS_N_CLICK_LOW : action - RS_A_CLICK_LOW;
        out->option = P->consts->s_click[k];
        break;
    case RS_ACTFAM_HELD:
        k = action - RS_A_HELD0;
        if (k < o->n_held)
            out->kind = RS_KIND_OBJ, out->name = o->held_sel[k].name,
            out->option = o->has_using ? P->consts->s_use : o->held_sel[k].option;
        break;
    case RS_ACTFAM_DROP: case RS_ACTFAM_DEPOSIT: case RS_ACTFAM_USE: {
        const RsItem *it = &o->inv_sel[action - (f == RS_ACTFAM_DROP ? RS_A_DROP0 : f == RS_ACTFAM_DEPOSIT ? RS_A_DEPOSIT0
                                                 : RS_A_USE0)];
        if (it->slot >= 0)
            out->kind = RS_KIND_OBJ, out->name = it->name, out->item = 0,
            out->option = f == RS_ACTFAM_DROP ? P->consts->s_drop : f == RS_ACTFAM_DEPOSIT ? P->consts->s_deposit
                : P->consts->s_use;
        break;
    }
    case RS_ACTFAM_WITHDRAW: {
        const RsItem *it = &o->bank_sel[action - RS_A_WITHDRAW0];
        if (it->slot >= 0) out->kind = RS_KIND_OBJ, out->name = it->name, out->option = P->consts->s_withdraw;
        break;
    }
    default:
        break;
    }
}

/* Agent.feedback (agent.py) with publish_chat (833-838) and dialog_text (1030-1037). */
int rs_feedback(RsSlot *s, const RsMeaning *meaning, const RsSnap *before, const RsSnap *after, int running) {
    const RsPack *P = s->P;
    World *w = &s->w;
    const Player *p = &w->p;
    Agent *a = &s->a;
    RsRawObs *o = &a->obs;
    o->fb_has_action = meaning != NULL;
    if (meaning) o->fb_meaning = *meaning;
    o->fb_running = (uint8_t)(running != 0);
    o->fb_has_ended = 0;
    if (a->option_end.has && !a->option_end.reported) {
        o->fb_has_ended = 1, o->fb_ended_action = a->option_end.action;
        o->fb_ended_reason = a->option_end.reason, o->fb_ended_refused = a->option_end.refused;
        a->option_end.reported = 1;
    }
    o->fb_ticks = o->fb_dx = o->fb_dz = o->fb_hp = o->fb_gained = o->fb_lost = 0;
    o->fb_xp = 0, o->fb_level_changed = o->fb_bank_opened = o->fb_dialog_opened = 0;
    if (before) {   /* _changes (agent.py): gains and losses per obj over inv + worn */
        int64_t gained = 0, lost = 0;
        for (int i = 0; i < after->n_items; i++) {
            int64_t was = 0;
            for (int j = 0; j < before->n_items; j++)
                if (before->item_obj[j] == after->item_obj[i]) was = before->item_count[j];
            int64_t delta = after->item_count[i] - was;
            if (delta > 0) gained += delta;
            else lost -= delta;
        }
        for (int j = 0; j < before->n_items; j++) {
            int in = 0;
            for (int i = 0; i < after->n_items; i++) in |= after->item_obj[i] == before->item_obj[j];
            if (!in) lost += before->item_count[j];
        }
        o->fb_ticks = after->tick - before->tick, o->fb_dx = after->x - before->x, o->fb_dz = after->z - before->z;
        o->fb_level_changed = after->level != before->level, o->fb_xp = after->xp - before->xp;
        o->fb_hp = after->hp - before->hp, o->fb_gained = gained, o->fb_lost = lost;
        o->fb_bank_opened = after->bank_open && !before->bank_open;
        o->fb_dialog_opened = after->dialog_open && !before->dialog_open;
    }
    /* publish_chat: every kept line above the last covered sequence, aged from its arrival */
    o->n_chat = 0;
    for (int i = 0; i < a->n_chat; i++)
        if (a->chat[i].seq > a->chat_covered)
            o->chat_msg[o->n_chat] = a->chat[i].msg, o->chat_age[o->n_chat++] = w->tick - a->chat[i].tick;
    a->chat_covered = a->chat_seq;
    /* dialog_text: the nonblank texts of the open chat interface's components by com */
    o->n_dialog_parts = 0;
    if (p->modal_chat != -1)
        for (int i = 0; i < a->n_texts; i++) {
            const RsTextEntry *t = &a->texts[i];
            if (t->iface != p->modal_chat || !P->msgs[t->msg].nonblank) continue;
            int j = o->n_dialog_parts++;
            for (; j > 0 && a->texts[o->dialog_part[j - 1]].com > t->com; j--) o->dialog_part[j] = o->dialog_part[j - 1];
            o->dialog_part[j] = i;
        }
    for (int k = 0; k < o->n_dialog_parts; k++) o->dialog_part[k] = a->texts[o->dialog_part[k]].msg;
    /* the choices: the texts of the resume buttons while suspended. Only p_choice2 (talk_to_banker, RS_NP) sets them,
       and RsRawObs does not carry their texts, so a choice is not ported. */
    o->n_choices = suspended(p) ? p->n_resume : 0;
    if (o->n_choices) return RS_FAIL_TEXT(&w->err, RS_E_NOT_PORTED, "dialog choices (resume buttons)");
    return 0;
}

void rs_episode_start_feedback(RsRawObs *obs) {
    /* quiet(running, dialog) in place of the feedback (feedback_contract.py) */
    obs->fb_has_action = obs->fb_has_ended = 0;
    obs->fb_ticks = obs->fb_dx = obs->fb_dz = obs->fb_hp = obs->fb_gained = obs->fb_lost = 0;
    obs->fb_xp = 0, obs->fb_level_changed = obs->fb_bank_opened = obs->fb_dialog_opened = 0;
    obs->n_chat = 0;
}

/* ---- rl/obs.project (rl/obs.py project with the act and feedback contracts) ------------------------------------ */
#define F32(key) ((float *)(row + RS_OBS_##key##_OFF))
#define I32(key) ((int32_t *)(row + RS_OBS_##key##_OFF))

/* log1p(count) / K[k] from the batch's table (rs.h FLOATS); past it, a late feature of this row (RsLate). The
   oracle's counts are never negative (the contract refuses a negative feedback change and an empty page slot). */
static int log1p_k(RsSlot *s, const uint8_t *row, int k, int64_t count, float *out) {
    if (count < 0) return RS_FAIL_TEXT(&s->w.err, RS_E_INTERNAL, "a negative log1p feature count");
    if (count < RS_LOG1P_COUNTS) {
        *out = s->log1p[(size_t)k * RS_LOG1P_COUNTS + (size_t)count];
        return 0;
    }
    if (s->n_late_next >= RS_LATE_CAP) return RS_OVERFLOW(&s->w.err, "RS_LATE_CAP late log1p features");
    s->late_next[s->n_late_next++] = (RsLate){(int32_t)((const uint8_t *)out - row), k, count};
    *out = 0.0f;
    return 0;
}


/* act_contract.bank_distance over the banks of the 15-tile loc and npc lists (RsRawObs bank tiles) */
static double bank_distance(const RsRawObs *o, int32_t x, int32_t z) {
    int32_t best = -1;
    for (int k = 0; k < o->n_banks; k++) {
        int32_t d = cheb(x, z, o->bank_x[k], o->bank_z[k]);
        if (best < 0 || d < best) best = d;
    }
    return best < 0 ? RS_NO_BANK : cap((double)best / RS_BANK_SCALE, RS_NO_BANK);
}

int rs_project(RsSlot *s, uint8_t *row) {
    const RsPack *P = s->P;
    const RsRawObs *o = &s->a.obs;
    RsErr *err = &s->w.err;
    s->n_late_next = 0;
    memset(row, 0, RS_OBS_BUDGET_VEC_OFF);  /* the 28 arrays project() makes precede the task's three */
    /* observation_contract (item_contract.py): an occupied page slot of count 0 (an [id, 0] stack) is refused */
    for (int k = 0; k < RS_N_INV_SEL; k++)
        if ((o->inv_sel[k].slot >= 0 && o->inv_sel[k].count <= 0) || (o->bank_sel[k].slot >= 0 && o->bank_sel[k].count <= 0))
            return RS_FAIL_TEXT(err, RS_E_CONTRACT, "Item slot does not match physical page");
    const double hp = clip((double)o->hp / (double)imax(1, o->max_hp), 0.0, 1.0);
    const double cl = clip((double)o->combat_level / 126.0, 0.0, 1.0);
    float *pv = F32(PLAYER_VEC);
    pv[0] = (float)o->plevel, pv[1] = (float)hp, pv[2] = (float)cl, pv[3] = 0.0f;  /* animId is -1 */
    pv[4] = o->is_dead, pv[5] = o->in_combat, pv[6] = o->target_index != -1;

    int32_t *ids = I32(NPC_IDS);
    float *nf = F32(NPC_FEATS);
    for (int i = 0; i < imin(o->n_npcs, RS_OBSV_N_NPC); i++) {
        const RsNpcObs *n = &o->npcs[i];
        ids[i] = rs_name_row(P, RS_KIND_NPC, n->name);
        float *f = nf + 6 * i;
        f[0] = (float)clip(n->dx / 64.0, -1.0, 1.0), f[1] = (float)clip(n->dz / 64.0, -1.0, 1.0);
        f[2] = (float)clip(n->distance / 32.0, 0.0, 1.0), f[3] = (float)clip(n->combat_level / 40.0, 0.0, 4.0);
        f[4] = n->in_combat, f[5] = n->attack_op;
    }
    ids = I32(INV_IDS);
    float *inv = F32(INV_FEATS);
    int64_t gp = 0;
    for (int i = 0; i < o->n_inv; i++) {
        const RsItem *it = &o->inventory[i];
        if (P->obj_types[it->id].flags & RS_OF_COINS) gp += it->count;
        if (it->slot < 0 || it->slot >= RS_OBSV_N_INV) continue;
        ids[it->slot] = rs_name_row(P, RS_KIND_OBJ, it->name);
        RS_TRY(log1p_k(s, row, RS_LOG1P_5, it->count, &inv[it->slot]));
    }
    ids = I32(WORN_IDS);
    for (int i = 0; i < o->n_worn; i++)
        if (o->equipment[i].slot >= 0 && o->equipment[i].slot < RS_OBSV_N_WORN)
            ids[o->equipment[i].slot] = rs_name_row(P, RS_KIND_OBJ, o->equipment[i].name);
    ids = I32(SEL_IDS);
    float *sel = F32(SEL_FEATS);
    for (int i = 0; i < RS_N_INV_SEL; i++) {
        if (o->inv_sel[i].slot == -1) continue;
        ids[i] = rs_name_row(P, RS_KIND_OBJ, o->inv_sel[i].name);
        RS_TRY(log1p_k(s, row, RS_LOG1P_5, o->inv_sel[i].count, &sel[i]));
    }

    float *pg = F32(PROG_VEC);
    int32_t total = 0;
    for (int i = 0; i < RS_N_SKILLS; i++) {
        int32_t level = o->base_levels[i] ? o->base_levels[i] : o->levels[i];   /* baseLevel or level */
        total += level;
        pg[i] = (float)clip(level / 99.0, 0.0, 1.0);
    }
    pg[RS_N_SKILLS] = (float)clip(total / 2079.0, 0.0, 1.0);
    RS_TRY(log1p_k(s, row, RS_LOG1P_12, gp, &pg[RS_N_SKILLS + 1]));
    pg[RS_N_SKILLS + 2] = (float)hp, pg[RS_N_SKILLS + 3] = (float)cl;

    float *tv = F32(TASK_VEC);
    tv[0] = (float)clip((double)o->decision / RS_OBS_MAX_DECISIONS, 0.0, 1.0);
    tv[1] = o->bank_open, tv[2] = o->dialog_open, tv[3] = o->modal_open;
    tv[4] = (float)clip(o->n_inv / 28.0, 0.0, 1.0), tv[5] = (float)clip(o->n_dialog_options / 5.0, 0.0, 1.0);
    tv[6] = o->has_using;
    I32(USE_IDS)[0] = o->has_using ? rs_name_row(P, RS_KIND_OBJ, o->using_.name) : 0;
    if (o->option_busy) {
        if (!(o->option_action > 0 && o->option_action < RS_N_ACTIONS))
            return RS_FAIL_TEXT(err, RS_E_CONTRACT, "busy observation without a running optionAction");
        F32(OPTION_VEC)[o->option_action] = 1.0f;
    }
    memcpy(row + RS_OBS_MASK_OFF, o->mask, RS_N_ACTIONS);

    ids = I32(BANK_IDS);
    float *bf = F32(BANK_FEATS);
    for (int i = 0; i < RS_N_BANK_SEL; i++) {
        if (o->bank_sel[i].slot == -1) continue;
        ids[i] = rs_name_row(P, RS_KIND_OBJ, o->bank_sel[i].name);
        RS_TRY(log1p_k(s, row, RS_LOG1P_5, o->bank_sel[i].count, &bf[2 * i]));
        bf[2 * i + 1] = (float)((double)o->bank_sel[i].slot / (RS_BANK_CAPACITY - 1));
    }
    float *iv = F32(ITEM_VEC);
    iv[0] = (float)(o->inv_page / 3.0), iv[1] = (float)((double)o->bank_page / imax(1, o->bank_pages - 1));
    iv[2] = o->bank_open, iv[3] = (float)((double)o->bank_pages / RS_MAX_BANK_PAGES);

    /* project_acts (act_contract.py) and the bank column (obs.py) */
    int32_t combat = o->combat_level, *ops = I32(ACT_OP_IDS);
    ids = I32(ACT_IDS);
    float *af = F32(ACT_FEATS);
    for (int i = 0; i < o->n_act; i++) {
        const RsAff *x = &o->act_sel[i];
        int32_t distance = cheb(x->x, x->z, o->px, o->pz);
        ids[i] = rs_name_row(P, x->kind, x->name), ops[i] = P->str_name_id[x->option];
        float *f = af + 11 * i;
        f[0] = (float)(x->dx / 15.0), f[1] = (float)(x->dz / 15.0), f[2] = (float)(distance / 15.0);
        f[3] = (float)cap(x->steps / 32.0, 2.0), f[4] = (float)(imin(x->count, 10) / 10.0);
        f[5] = x->kind == RS_KIND_LOC, f[6] = x->kind == RS_KIND_NPC, f[7] = x->kind == RS_KIND_OBJ;
        f[8] = (float)cap(x->combat_level / 40.0, 4.0), f[9] = x->combat_level > combat;
        f[10] = (float)bank_distance(o, x->x, x->z);
    }
    ids = I32(HELD_IDS), ops = I32(HELD_OP_IDS);
    float *hf = F32(HELD_FEATS);
    for (int i = 0; i < o->n_held; i++) {
        const RsHeld *h = &o->held_sel[i];
        const RsItem *it = NULL;
        for (int k = 0; k < RS_N_INV_SEL; k++)
            if (o->inv_sel[k].slot >= 0 && o->inv_sel[k].slot == h->slot) it = &o->inv_sel[k];
        if (!it) return RS_FAIL_TEXT(err, RS_E_CONTRACT, "Held affordance does not match the visible page");
        ids[i] = rs_name_row(P, RS_KIND_OBJ, it->name), ops[i] = P->str_name_id[h->option];
        RS_TRY(log1p_k(s, row, RS_LOG1P_5, it->count, &hf[2 * i]));
        hf[2 * i + 1] = 1.0f;
    }
    /* project_clicks (act_contract.py): each slot's offset over CLICK_SCALE and its open bit */
    float *cf = F32(CLICK_FEATS);
    for (int k = 0; k < RS_N_CLICK; k++) {
        float *f = cf + RS_CLICK_FEATS * k;
        f[0] = (float)(RS_CLICK_DX[k] / (double)RS_CLICK_SCALE), f[1] = (float)(RS_CLICK_DZ[k] / (double)RS_CLICK_SCALE);
        f[2] = o->clicks[k].open;
    }
    /* project_feedback (feedback_contract.py) */
    if (o->fb_ticks < 0 || o->fb_xp < 0 || o->fb_gained < 0 || o->fb_lost < 0)
        return RS_FAIL_TEXT(err, RS_E_CONTRACT, "Negative feedback change");
    float *fb = F32(FB_VEC);
    int32_t *fid = I32(FB_IDS);
    if (!o->fb_has_action) {
        fb[RS_N_ACTFAMS] = 1.0f;
    } else {
        const RsMeaning *m = &o->fb_meaning;
        if (m->family >= RS_N_ACTFAMS) return RS_FAIL_TEXT(err, RS_E_CONTRACT, "Invalid feedback action");
        fb[m->family] = 1.0f;
        if (m->kind != RS_KIND_NONE && m->name) fid[0] = rs_name_row(P, m->kind, m->name);
        if (m->item) fid[1] = rs_name_row(P, RS_KIND_OBJ, m->item);
        I32(FB_OP)[0] = m->option ? P->str_name_id[m->option] : 0;
    }
    int k = RS_N_ACTFAMS + 1;
    fb[k] = o->fb_running;
    if (o->fb_has_ended) fb[k + 1 + o->fb_ended_reason] = 1.0f, fb[k + 1 + 5] = o->fb_ended_refused;
    k += 2 + 5;
    fb[k] = (float)cap(o->fb_ticks / 50.0, 4.0);
    fb[k + 1] = (float)clip(o->fb_dx / 16.0, -2.0, 2.0), fb[k + 2] = (float)clip(o->fb_dz / 16.0, -2.0, 2.0);
    fb[k + 3] = (float)cap(imax(iabs(o->fb_dx), iabs(o->fb_dz)) / 16.0, 2.0);
    fb[k + 4] = o->fb_level_changed;
    RS_TRY(log1p_k(s, row, RS_LOG1P_10, o->fb_xp, &fb[k + 5]));
    fb[k + 6] = (float)clip(o->fb_hp / 10.0, -1.0, 1.0);
    RS_TRY(log1p_k(s, row, RS_LOG1P_3, o->fb_gained, &fb[k + 7]));
    RS_TRY(log1p_k(s, row, RS_LOG1P_3, o->fb_lost, &fb[k + 8]));
    fb[k + 9] = o->fb_bank_opened, fb[k + 10] = o->fb_dialog_opened;
    float *text = F32(TEXT_VEC);
    rs_text_vector_chat(P, o->chat_msg, o->n_chat, RS_TEXT_DIM, text);
    rs_text_vector_parts(P, o->dialog_part, o->n_dialog_parts, RS_TEXT_DIM, text + RS_TEXT_DIM);
    int32_t age = 0;
    for (int i = 0; i < o->n_chat; i++) age = i == 0 || o->chat_age[i] < age ? o->chat_age[i] : age;
    text[2 * RS_TEXT_DIM] = o->n_chat > 0;
    text[2 * RS_TEXT_DIM + 1] = (float)cap(age / 50.0, 4.0);
    text[2 * RS_TEXT_DIM + 2] = (float)(imin(o->n_chat, 4) / 4.0);
    text[2 * RS_TEXT_DIM + 3] = o->n_dialog_parts > 0;   /* the parts are nonblank: dialog.strip() is nonempty */
    text[2 * RS_TEXT_DIM + 4] = (float)(imin(o->n_choices, 5) / 5.0);
    if (o->n_choices) return RS_FAIL_TEXT(err, RS_E_NOT_PORTED, "choice_vec of dialog choices");
    return 0;
}

/* project_v6's progress (rl/obs.py level_progress) of each SKILL_ORDER skill, at its observed whole XP (stats // 10)
   and level (baseLevel or level), into the row's progress block: 0 at 99; else (10 * xp - low) / (next - low),
   clipped to [0, 1], with the client's table RS_OBS_LEVEL_XP (entry k: level k + 2; a level below 1 reads Python's
   negative index, as LEVEL_XP[level - 1] does). Ints are exact in double, so the one division is Python's. */
_Static_assert(RS_OBS_PROGRESS_N == RS_N_SKILLS, "obs-v6 progress per SKILL_ORDER skill is per engine skill");
void rs_project_progress(const RsSlot *s, uint8_t *row) {
    const RsRawObs *o = &s->a.obs;
    float *pr = (float *)(row + RS_OBS_PROGRESS_OFF);
    for (int i = 0; i < RS_N_SKILLS; i++) {
        int32_t level = o->base_levels[i] ? o->base_levels[i] : o->levels[i];
        if (level >= 99) {
            pr[i] = 0.0f;
            continue;
        }
        int32_t next = level - 1 < 0 ? level - 1 + 99 : level - 1;
        int64_t low = level >= 2 ? RS_OBS_LEVEL_XP[level - 2] : 0;
        double v = (double)(10 * (int64_t)o->xp[i] - low) / (double)(RS_OBS_LEVEL_XP[next] - low);
        pr[i] = (float)clip(v, 0.0, 1.0);
    }
}
