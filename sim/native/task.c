/* sim/native/task.c: tasks: start draws, episodes, the terminal hook, ledgers, audit, fights, supply,
 * facts. First, the keyed start draws of CpuBatch._episode (task.draw_start, combat.draw_combat and its
 * npcs and kit, search.draw_supply with the variant record supply_episode plays), the step ledgers of CpuBatch.step
 * (info made and kills), and the terminal accounts (audit.audit, combat.fights' summary, CpuBatch._facts). Then
 * the episode (setup, arm and step as GoalEpisode and CpuBatch.step run them), the post-cycle watch, the task's
 * projection inputs and the SupplyTracker.
 *
 * Draws are pure functions of (key, slot, episode, purpose[, index]) (rs.h RNG): every draw here is
 * int(Rng.draw(h, *keys) * n) = rs_pick(m, n) on h = Rng(key).prefix(slot, episode), so call order does not matter
 * and only the key tuples must be the oracle's. Pool tables (start tiles, camp, outside regions, groves, sites) are
 * the exporter's, x-major as np.argwhere lists them, and are indexed exactly as the Python indexes its lists.
 *
 * Ledgers are accumulators fed one typed event at a time (RsEvent, rs.h). The kill ledger lives in the Episode and is
 * fed every step from its cursor (rs_ledger_update), as CpuBatch.step recounts kills() over events[event_start:]
 * every step; its one error, 'an npc defeated without a hit', fires at the step whose events hold the defeat, as
 * Python's does. The audit and the fights summary are only ever taken at the terminal step, over the window
 * (events from arm with tick < the terminal tick), and their errors are the terminal's in Python too, so rs_audit and
 * rs_fights feed fresh accumulators over the window then, each in its own pass and with its own kill ledger, as
 * audit() and fights() each call kill_lives on the window (the audit stays an account independent of the payment). */
#include <math.h>
#include <string.h>

#include "pysum.h"
#include "rs.h"

/* ---- start draws ------------------------------------------------------------------------------------------------ */
/* Rng(key).prefix(slot, episode): the chain every start draw of an episode shares (task.py, combat.py,
   search.py). */
static uint32_t start_prefix(int64_t key, int32_t slot, int32_t episode) {
    return rs_prefix(rs_seed(key), RS_KEYS(slot, episode));
}
/* int(Rng.draw(h, *keys) * n) */
#define PICK(h, n, ...) rs_pick(rs_m((h), RS_KEYS(__VA_ARGS__)), (n))

#define N_START_LEVELS ((int)(sizeof RS_START_LEVEL_N / sizeof *RS_START_LEVEL_N))
_Static_assert(N_START_LEVELS == (int)(sizeof ((RsStart *)0)->levels / sizeof (int16_t)),
               "RsStart.levels holds START_LEVELS");
#define N_VARIANTS (RS_VARIANT_SCARCE + 1)  /* SUPPLY['variants'] = ('ample', 'scarce') */

static int32_t imax(int32_t a, int32_t b) { return a > b ? a : b; }
static int32_t imin(int32_t a, int32_t b) { return a < b ? a : b; }
static int32_t iabs(int32_t a) { return a < 0 ? -a : a; }

/* A layout array the draws index is present whenever its count says it holds anything (pack.c resolves an absent
   array to NULL whatever the count). */
static int arrays_ok(const RsLayout *L) {
    return (L->start_tile || !L->n_start) && (L->locs || !L->n_locs) && (L->npcs || !L->n_npcs)
        && (L->camp || !L->n_camp) && (L->outside || !L->n_outside) && L->n_outside <= RS_MAX_T
        && ((L->grove_off && L->grove_row && L->grove_normal) || !L->n_groves)
        && ((L->site_grove && L->site_off && L->site_tile) || !L->n_sites);
}

/* The pool layout a start names (RsStart.layout indexes the compact or the supply pool). */
static const RsLayout *pool_layout(const RsPack *P, int pool, int32_t layout, RsErr *err) {
    int32_t first = pool == RS_LAYOUT_SUPPLY ? P->supply0 : P->compact0;
    int32_t n = pool == RS_LAYOUT_SUPPLY ? P->n_supply : P->n_compact;
    if (layout < 0 || layout >= n || !arrays_ok(&P->layouts[first + layout])) {
        rs_fail_text(err, RS_E_PACK, "a start names a layout outside its pool, or one lacking an array");
        return NULL;
    }
    return &P->layouts[first + layout];
}

/* draw_start (task.py): layout (R_LAYOUT), start tile (R_TILE) among np.argwhere(layout.start), weapon
   (R_WEAPON), each START_LEVELS stat (R_LEVEL, i) in the dict's order, world seed (R_WORLD; int(d * 2^24) is m).
   tile = ORIGIN + local, ORIGIN being every pool layout's (x0, z0). rec: the layout record played (the pool's). */
int rs_draw_start(const RsPack *P, int pool, int64_t key, int32_t slot, int32_t episode, RsStart *out, RsErr *err) {
    if (pool != RS_LAYOUT_COMPACT && pool != RS_LAYOUT_SUPPLY)
        return RS_FAIL_TEXT(err, RS_E_INTERNAL, "draw_start from a pool that is neither compact nor supply");
    int32_t first = pool == RS_LAYOUT_SUPPLY ? P->supply0 : P->compact0;
    int32_t n = pool == RS_LAYOUT_SUPPLY ? P->n_supply : P->n_compact;
    if (n < 1) return RS_FAIL_TEXT(err, RS_E_PACK, "the pack lacks the pool a start is drawn from");
    uint32_t h = start_prefix(key, slot, episode);
    memset(out, 0, sizeof *out);
    out->kind = RS_START_SHORT;
    out->layout = (int32_t)PICK(h, n, RS_R_LAYOUT);
    const RsLayout *L = pool_layout(P, pool, out->layout, err);
    if (!L) return 1;
    if (L->n_start < 1) return RS_FAIL_TEXT(err, RS_E_PACK, "a pool layout has no start tile");
    int32_t t = L->start_tile[PICK(h, L->n_start, RS_R_TILE)];
    out->tile_x = (int16_t)(L->x0 + t / L->H);
    out->tile_z = (int16_t)(L->z0 + t % L->H);
    out->weapon = (int32_t)PICK(h, RS_N_START_WEAPONS, RS_R_WEAPON);
    for (int i = 0; i < N_START_LEVELS; i++)
        out->levels[i] = RS_START_LEVEL_CHOICES[i * RS_START_LEVEL_WIDTH + PICK(h, RS_START_LEVEL_N[i], RS_R_LEVEL, i)];
    out->world_seed = (int32_t)PICK(h, 1 << 24, RS_R_WORLD);
    out->grove = out->variant = -1;
    out->rec = first + out->layout;
    return 0;
}

/* START_LEVELS' hitpoints entry (draw_combat's start['levels']['hitpoints']). */
static int hitpoints_level(void) {
    for (int i = 0; i < N_START_LEVELS; i++)
        if (RS_START_LEVEL_SKILL[i] == RS_SKILL_HITPOINTS) return i;
    return -1;
}

/* draw_combat (combat.py) after draw_start's draws on the compact pool, which it leaves alone: group (R_GROUP),
   hitpoints at arm (R_HP; absolute or below the base level, clamped to 1..base), food count (R_FOOD_N) and each item
   (R_FOOD, i), muggers (R_MUGGER, i) and worn (R_WORN). Mugger options: outside(layout)'s tiles in x-major order whose
   Chebyshev distance from the camp centre (np.mean of the camp tiles: an exact integer sum, one correctly rounded
   division, in double) lies in mugger_camp, at least mugger_start from the start tile, and not a camp tile. */
static int draw_combat_in(const RsPack *P, int64_t key, int32_t slot, int32_t episode, RsStart *s, int32_t group,
                          int place_muggers, RsErr *err) {
    const RsLayout *L = pool_layout(P, RS_LAYOUT_COMPACT, s->layout, err);
    if (!L) return 1;
    int hpk = hitpoints_level();
    if (hpk < 0) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "START_LEVELS has no hitpoints");
    uint32_t h = start_prefix(key, slot, episode);
    int32_t g = group >= 0 ? group : (int32_t)PICK(h, RS_N_GROUPS, RS_R_GROUP);  /* `group` replaces its draw */
    int32_t base = s->levels[hpk], lo = RS_GROUP_HP_LO[g], hi = RS_GROUP_HP_HI[g];
    int32_t d = lo + (int32_t)PICK(h, hi - lo + 1, RS_R_HP);
    int32_t hp = RS_GROUP_HP_ABS[g] ? d : base - d;
    s->hp = imax(1, imin(base, hp));
    s->group = g;
    lo = RS_GROUP_FOOD_LO[g], hi = RS_GROUP_FOOD_HI[g];
    int32_t n_food = lo + (int32_t)PICK(h, hi - lo + 1, RS_R_FOOD_N);
    if (n_food > (int32_t)(sizeof s->food / sizeof *s->food)) return RS_OVERFLOW(err, "combat start food");
    s->n_food = n_food;
    for (int32_t i = 0; i < n_food; i++) s->food[i] = (int8_t)PICK(h, RS_COMBAT_N_FOOD, RS_R_FOOD, i);
    int32_t want = RS_GROUP_MUGGERS[g];
    s->n_muggers = 0;
    if (want > 0 && place_muggers) {   /* `if spec['muggers'] and place_muggers`: no camp read, no tiles, no draw */
        if (want > (int32_t)(sizeof s->mugger_x / sizeof *s->mugger_x)) return RS_OVERFLOW(err, "combat start muggers");
        if (L->n_camp < 1) return RS_FAIL_TEXT(err, RS_E_PACK, "a combat layout has no camp");
        int64_t sum_x = 0, sum_z = 0;
        for (int32_t c = 0; c < L->n_camp; c++) sum_x += L->camp[c] / L->H, sum_z += L->camp[c] % L->H;
        double cx = (double)sum_x / (double)L->n_camp, cz = (double)sum_z / (double)L->n_camp;
        int32_t sx = s->tile_x - L->x0, sz = s->tile_z - L->z0;
        int16_t options[RS_MAX_T];
        int32_t n = 0;
        for (int32_t k = 0; k < L->n_outside; k++) {
            int32_t t = L->outside[k], x = t / L->H, z = t % L->H;
            double ax = (double)x - cx, az = (double)z - cz;
            ax = ax < 0 ? -ax : ax, az = az < 0 ? -az : az;
            double far = az > ax ? az : ax;  /* Python's max(a, b): a unless b > a */
            if (!(RS_COMBAT_CAMP_LO <= far && far <= RS_COMBAT_CAMP_HI)) continue;
            if (imax(iabs(x - sx), iabs(z - sz)) < RS_COMBAT_MUGGER_START) continue;
            int camp = 0;
            for (int32_t c = 0; c < L->n_camp && !camp; c++) camp = L->camp[c] == t;
            if (!camp) options[n++] = (int16_t)t;
        }
        if (n < want) return RS_FAIL(err, RS_E_MUGGER_TILES, (int32_t)L->rec->seed, n, want);
        for (int32_t i = 0; i < want; i++) {
            int32_t k = (int32_t)PICK(h, n, RS_R_MUGGER, i), t = options[k];
            memmove(&options[k], &options[k + 1], (size_t)(n - k - 1) * sizeof *options);  /* options.remove(tile) */
            n--;
            s->mugger_x[i] = (int16_t)(t / L->H);
            s->mugger_z[i] = (int16_t)(t % L->H);
        }
        s->n_muggers = want;
    }
    s->worn = rs_draw(h, RS_KEYS(RS_R_WORN)) < RS_COMBAT_WORN;
    s->kind = RS_START_COMBAT;
    return 0;
}

int rs_draw_combat(const RsPack *P, int64_t key, int32_t slot, int32_t episode, RsStart *s, RsErr *err) {
    return draw_combat_in(P, key, slot, episode, s, -1, 1, err);
}

/* scarce_rows' kept row (search.py:346-351): the grove's normal tree nearest the start tile (local; Chebyshev from
   its origin), ties to the lower row; -1 when the grove has no normal tree (min() of nothing: never, for a site). */
static int32_t scarce_keep(const RsLayout *L, int32_t grove, int32_t tx, int32_t tz) {
    int32_t keep = -1, best = 0;
    for (int32_t r = L->grove_off[grove]; r < L->grove_off[grove + 1]; r++) {
        if (!L->grove_normal[r]) continue;
        int32_t k = L->grove_row[r];
        int32_t d = imax(iabs(L->locs[k * 5 + 1] - tx), iabs(L->locs[k * 5 + 2] - tz));
        if (keep < 0 || d < best || (d == best && k < keep)) keep = k, best = d;
    }
    return keep;
}

/* draw_supply (search.py): draw_start on the supply pool (its tile replaced), then the site (R_SUPPLY_GROVE)
   among supply_sites, the tile (R_SUPPLY_TILE) among that site's tiles, and the variant (R_SUPPLY_VARIANT). rec is the
   layout record supply_episode plays (CpuBatch._supply_tables, vec.py): the pool layout's for ample, the
   precomputed variant of (layout, grove, scarce_rows' kept row) for scarce, whose failing check is variant_layout's
   RuntimeError (search.py), raised where CpuBatch._episode raises it, before the episode is built. */
int rs_draw_supply(const RsPack *P, int64_t key, int32_t slot, int32_t episode, RsStart *out, RsErr *err) {
    RS_TRY(rs_draw_start(P, RS_LAYOUT_SUPPLY, key, slot, episode, out, err));
    const RsLayout *L = &P->layouts[P->supply0 + out->layout];
    uint32_t h = start_prefix(key, slot, episode);
    if (L->n_sites < 1) return RS_FAIL_TEXT(err, RS_E_PACK, "a supply layout has no site");
    int32_t site = (int32_t)PICK(h, L->n_sites, RS_R_SUPPLY_GROVE);
    int32_t grove = L->site_grove[site], a = L->site_off[site], b = L->site_off[site + 1];
    if (b <= a || grove < 0 || grove >= L->n_groves)
        return RS_FAIL_TEXT(err, RS_E_PACK, "a supply site has no tile or names no grove");
    int32_t t = L->site_tile[a + PICK(h, b - a, RS_R_SUPPLY_TILE)];
    out->kind = RS_START_SUPPLY;
    out->tile_x = (int16_t)(L->x0 + t / L->H);
    out->tile_z = (int16_t)(L->z0 + t % L->H);
    out->grove = grove;
    out->variant = (int32_t)PICK(h, N_VARIANTS, RS_R_SUPPLY_VARIANT);
    if (out->variant == RS_VARIANT_AMPLE) {
        out->rec = P->supply0 + out->layout;
        return 0;
    }
    int32_t keep = scarce_keep(L, grove, t / L->H, t % L->H);
    if (keep < 0) return RS_FAIL_TEXT(err, RS_E_PACK, "a supply grove has no normal tree");
    int32_t rec = rs_variant_find(P, out->layout, grove, keep);
    if (rec == RS_VARIANT_FAILS) return RS_FAIL(err, RS_E_SCARCE_CHECK, (int32_t)L->rec->seed, grove);
    if (rec < 0) return RS_FAIL_TEXT(err, RS_E_PACK, "the pack lacks the scarce variant of a drawn start");
    out->rec = rec;
    return 0;
}

/* The draw half of CpuBatch._episode (vec.py): a v5 wood slot draws a supply start, every other slot
   draw_start on the compact pool, and a melee slot of a combat task (v3, v5; combat.task_spec) adds draw_combat. */
int rs_episode_draw(const RsSlot *s, int64_t key, RsStart *out, RsErr *err) {
    int task = s->cfg->task;
    if (task == RS_TASK_FREE) return rs_draw_village(s->P, s->cfg, key, s->slot, s->episode, out, err);
    if (task != RS_TASK_V2 && task != RS_TASK_V3 && task != RS_TASK_V5)
        return RS_FAIL_TEXT(err, RS_E_CONTRACT, "the native engine runs goal-activity-v2, v3 and v5");
    if (task == RS_TASK_V5 && s->goal == RS_GOAL_WOOD) return rs_draw_supply(s->P, key, s->slot, s->episode, out, err);
    RS_TRY(rs_draw_start(s->P, RS_LAYOUT_COMPACT, key, s->slot, s->episode, out, err));
    if (task != RS_TASK_V2 && s->goal == RS_GOAL_MELEE) RS_TRY(rs_draw_combat(s->P, key, s->slot, s->episode, out, err));
    return 0;
}

/* combat_npcs (combat.py): the layout's npc rows that are not chickens (by the consts' chicken ids, as Python's
   last-wins debugname map), a goblin at every camp tile, then the muggers; nid = position, level 0, the type's size. */
int rs_combat_npcs(const RsPack *P, const RsStart *s, RsNpcSpawn out[RS_MAX_NPCS], int *n, RsErr *err) {
    const RsConsts *C = P->consts;
    const RsLayout *L = pool_layout(P, RS_LAYOUT_COMPACT, s->layout, err);
    if (!L) return 1;
    if (!C) return RS_FAIL_TEXT(err, RS_E_PACK, "the pack has no constants");
    int k = 0;
    for (int pass = 0; pass < 3; pass++) {
        int32_t rows = pass == 0 ? L->n_npcs : pass == 1 ? L->n_camp : s->n_muggers;
        for (int32_t i = 0; i < rows; i++) {
            int32_t type, x, z;
            if (pass == 0) {
                type = L->npcs[i * 4 + 1], x = L->npcs[i * 4 + 2], z = L->npcs[i * 4 + 3];
                if (type == C->npc_chicken || type == C->npc_chicken_brown) continue;
            } else if (pass == 1) {
                type = C->npc_goblin, x = L->camp[i] / L->H, z = L->camp[i] % L->H;
            } else {
                type = C->npc_mugger, x = s->mugger_x[i], z = s->mugger_z[i];
            }
            const RsNpcType *t = rs_npc_type(P, type);
            if (!t) return RS_FAIL_TEXT(err, RS_E_PACK, "a combat npc's type is not in the pack");
            if (k >= RS_MAX_NPCS) return RS_OVERFLOW(err, "combat npcs");
            out[k] = (RsNpcSpawn){.nid = k, .type = type, .x = L->x0 + x, .z = L->z0 + z, .level = 0, .size = t->size};
            k++;
        }
    }
    *n = k;
    return 0;
}

/* The kit GoalEpisode.start hands agent.start (task.py): combat.kit(weapon, food) for a combat start (the tools,
   the drawn food in order, then the weapon; combat.py), else START_KIT plus the weapon. Obj ids are the
   consts' (obj_id: the lowest id of a debugname; rs_batch_new requires the pack's consts). Returns the number of
   entries (at most RS_START_KIT_MAX). */
int rs_start_kit(const RsPack *P, const RsStart *s, int32_t obj[RS_START_KIT_MAX], int32_t count[RS_START_KIT_MAX]) {
    const RsConsts *C = P->consts;
    int n = 0;
    if (s->kind == RS_START_COMBAT || s->kind == RS_START_VILLAGE) {   /* combat.kit (village_life's too) */
        for (int i = 0; i < (int)(sizeof C->combat_tools / sizeof *C->combat_tools); i++)
            obj[n] = C->combat_tools[i], count[n++] = C->combat_tool_count[i];
        for (int i = 0; i < s->n_food; i++) obj[n] = C->combat_food[s->food[i]], count[n++] = 1;
    } else {
        for (int i = 0; i < RS_N_START_KIT; i++) obj[n] = C->kit[i], count[n++] = C->kit_count[i];
    }
    obj[n] = C->weapons[s->weapon], count[n++] = 1;
    return n;
}

/* ---- the kill ledger (audit.kill_lives, audit.py) ------------------------------------------------------- */
/* A view of one ledger: the Episode's (fed every step) or a local one (the audit's and fights' window passes). */
typedef struct { RsLife *lives; int32_t *n, *kills; const int32_t *enemies; int32_t n_enemies; } Kills;

static int certified(const Kills *k, const RsLife *l) {
    if (!l->fresh || !l->mine || l->defeated < 0) return 0;
    for (int32_t i = 0; i < k->n_enemies; i++)
        if (k->enemies[i] == l->type) return 1;
    return 0;
}

static RsLife *find_life(const Kills *k, int32_t npc, int32_t life) {
    for (int32_t i = *k->n - 1; i >= 0; i--)
        if (k->lives[i].npc == npc && k->lives[i].life == life) return &k->lives[i];
    return NULL;
}

/* One event of kill_lives: a hit's first sight of a life sets its type and fresh (this agent's hit at full
   hitpoints) and every hit keeps mine only while all are the agent's; a defeat needs a prior hit and sets defeated
   to its tick when the agent's, else None (-1). kills, the count of certified lives, follows every change, since a
   later event can withdraw a credit (a defeat or hit by someone else). */
static int kill_feed(const Kills *k, const RsEvent *e, RsErr *err) {
    if (e->kind != RS_EV_NPC_DAMAGE && e->kind != RS_EV_NPC_DEFEATED) return 0;
    RsLife *l = find_life(k, e->obj, e->v0);
    if (e->kind == RS_EV_NPC_DAMAGE && !l) {
        if (*k->n >= RS_KILL_LIVES) return RS_OVERFLOW(err, "kill ledger lives");
        l = &k->lives[(*k->n)++];
        *l = (RsLife){.npc = e->obj, .life = e->v0, .type = e->count, .defeated = -1, .fresh = e->a8 && e->v2 == e->v3,
                      .mine = 1};
    }
    if (!l) return RS_FAIL0(err, RS_E_AUDIT_DEFEAT);
    int before = certified(k, l);
    if (e->kind == RS_EV_NPC_DAMAGE) l->mine = l->mine && e->a8;
    else l->defeated = e->a8 ? e->tick : -1;
    *k->kills += certified(k, l) - before;
    return 0;
}

/* CpuBatch.step's info made and kills (vec.py) over events[event_start:], fed from the cursor ev_seen (at arm:
   event_start, made, lives and kills zero; the episode code sets them). made: produce events of the FAMILIES objs
   (task.produced; pickups do not count); kills: audit.kills with the episode's enemies (MELEE_ENEMIES for every
   non-combat slot). The melee payment reads ep.kills after this call, as GoalEpisode.held recounts kills(); at the
   terminal step rs_audit runs first, as task.step's audit precedes vec.step's kills(). */
int rs_ledger_update(RsSlot *s) {
    World *w = &s->w;
    Episode *ep = &s->ep;
    Kills k = {ep->lives, &ep->n_lives, &ep->kills, ep->enemies, ep->n_enemies};
    for (; ep->ev_seen < w->n_events; ep->ev_seen++) {
        const RsEvent *e = &w->ev[ep->ev_seen];
        if (e->kind == RS_EV_PRODUCE) {
            const RsObjType *t = rs_obj_type(s->P, e->obj);
            if (t && t->made_family >= 0) ep->made[t->made_family] += e->count;
        } else {
            RS_TRY(kill_feed(&k, e, &w->err));
        }
    }
    return 0;
}

/* ---- the audit (audit.audit, audit.py) ----------------------------------------------------------------- */
/* Counter tables by (table, obj): the three holdings in _totals' order, then production and pickups. Counter reads of
   an absent key are 0 and do not insert; writes insert. */
enum { T_INVENTORY = 0, T_BANK, T_WORN, T_PRODUCED, T_EXTERNAL };
static const char *const HOLDING_NAMES[3] = {"inventory", "bank", "worn"};
#define AUDIT_CELLS 512
typedef struct { int32_t obj; uint8_t table; int64_t count; } Cell;
typedef struct { int32_t n; Cell cell[AUDIT_CELLS]; } Counters;

static int64_t counter_get(const Counters *c, int table, int32_t obj) {
    for (int32_t i = 0; i < c->n; i++)
        if (c->cell[i].table == table && c->cell[i].obj == obj) return c->cell[i].count;
    return 0;
}

static int counter_add(Counters *c, int table, int32_t obj, int64_t delta, RsErr *err) {
    for (int32_t i = 0; i < c->n; i++)
        if (c->cell[i].table == table && c->cell[i].obj == obj) {
            c->cell[i].count += delta;
            return 0;
        }
    if (c->n >= AUDIT_CELLS) return RS_OVERFLOW(err, "audit counters");
    c->cell[c->n++] = (Cell){.obj = obj, .table = (uint8_t)table, .count = delta};
    return 0;
}

/* _totals of a holdings dict: every listed slot (occupied, count 0 included) of inventory, bank and worn. */
static int counter_holdings(Counters *c, const RsHoldings *h, RsErr *err) {
    const Slot *slots[3] = {h->inv, h->bank, h->worn};
    const int caps[3] = {RS_INV_CAP, RS_BANK_CAP, RS_WORN_CAP};
    for (int table = 0; table < 3; table++)
        for (int i = 0; i < caps[table]; i++)
            if (slots[table][i].id >= 0) RS_TRY(counter_add(c, table, slots[table][i].id, slots[table][i].count, err));
    return 0;
}

/* +a[table] == +b[table]: the same positive counts. */
static int same_positive(const Counters *a, const Counters *b, int table) {
    for (int pass = 0; pass < 2; pass++) {
        const Counters *x = pass ? b : a, *y = pass ? a : b;
        for (int32_t i = 0; i < x->n; i++) {
            const Cell *c = &x->cell[i];
            if (c->table == table && c->count > 0 && counter_get(y, table, c->obj) != c->count) return 0;
        }
    }
    return 1;
}

static const int LOCATION_TABLE[3] = {[RS_LOCN_INV] = T_INVENTORY, [RS_LOCN_WORN] = T_WORN, [RS_LOCN_BANK] = T_BANK};

/* One event of the audit's loop: produce (by a GATHERING_SCRIPTS script; the pack names the four produce sites'
   scripts, all members) and pickup add to the inventory, transfers move between holdings (to may be the noted or
   unnoted id) and removals leave the inventory, each refused when the source holds too few. npc and agent_hp events
   and exclusions are evidence only. */
static int audit_feed(Counters *held, const RsConsts *C, const RsEvent *e, RsErr *err) {
    switch (e->kind) {
    case RS_EV_PRODUCE:
        if (e->v0 != C->s_produce_get_logs && e->v0 != C->s_produce_ore_normal && e->v0 != C->s_produce_ore_fast
            && e->v0 != C->s_produce_fish_roll)
            return RS_FAIL(err, RS_E_AUDIT_PRODUCTION, e->v0);
        RS_TRY(counter_add(held, T_INVENTORY, e->obj, e->count, err));
        return counter_add(held, T_PRODUCED, e->obj, e->count, err);
    case RS_EV_PICKUP:
        RS_TRY(counter_add(held, T_INVENTORY, e->obj, e->count, err));
        return counter_add(held, T_EXTERNAL, e->obj, e->count, err);
    case RS_EV_TRANSFER: {
        if (e->a8 > RS_LOCN_BANK || e->b8 > RS_LOCN_BANK) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "transfer location");
        int src = LOCATION_TABLE[e->a8], dst = LOCATION_TABLE[e->b8];
        if (counter_get(held, src, e->obj) < e->count) return RS_FAIL0(err, RS_E_AUDIT_TRANSFER);
        RS_TRY(counter_add(held, src, e->obj, -(int64_t)e->count, err));
        return counter_add(held, dst, e->v0, e->count, err);
    }
    case RS_EV_REMOVE:
        if (counter_get(held, T_INVENTORY, e->obj) < e->count) return RS_FAIL0(err, RS_E_AUDIT_REMOVAL);
        return counter_add(held, T_INVENTORY, e->obj, -(int64_t)e->count, err);
    default:
        return 0;
    }
}

static int in_family(const RsPack *P, int32_t obj, int goal) {
    const RsObjType *t = rs_obj_type(P, obj);
    return t && goal >= 0 && goal < 8 && (t->family_mask >> goal & 1);
}

/* audit(events[event_start:], initial, terminal, goal, tables, agent, enemies) of the closed episode: the loop over
   the window (it stops at the first event at or past the terminal tick), each holding's positive totals against the
   terminal holdings (inventory, bank, worn), the goal family's kept, produced and picked-up counts and their bounds,
   then the window's kills. */
int rs_audit(const RsSlot *s, RsAudit *out, RsErr *err) {
    const RsPack *P = s->P;
    const World *w = &s->w;
    const Episode *ep = &s->ep;
    if (!ep->terminal) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "audit of an episode without a terminal");
    if (!P->consts) return RS_FAIL_TEXT(err, RS_E_PACK, "the pack has no constants");
    Counters held = {0}, final = {0};
    RS_TRY(counter_holdings(&held, &ep->initial, err));
    int32_t end = ep->event_start;
    for (; end < w->n_events && w->ev[end].tick < ep->term_tick; end++) RS_TRY(audit_feed(&held, P->consts, &w->ev[end], err));
    RS_TRY(counter_holdings(&final, &ep->term_holdings, err));
    for (int table = T_INVENTORY; table <= T_WORN; table++)
        if (!same_positive(&held, &final, table)) return RS_FAIL_TEXT(err, RS_E_AUDIT_HOLDINGS, HOLDING_NAMES[table]);
    int64_t kept = 0, made = 0, picked = 0;
    for (int32_t i = 0; i < final.n; i++)
        if ((final.cell[i].table == T_INVENTORY || final.cell[i].table == T_BANK) && in_family(P, final.cell[i].obj, ep->goal))
            kept += final.cell[i].count;
    for (int32_t i = 0; i < held.n; i++) {
        if (!in_family(P, held.cell[i].obj, ep->goal)) continue;
        if (held.cell[i].table == T_PRODUCED) made += held.cell[i].count;
        else if (held.cell[i].table == T_EXTERNAL) picked += held.cell[i].count;
    }
    int64_t lower = kept - picked > 0 ? kept - picked : 0, upper = kept < made ? kept : made;
    if (lower > upper) return RS_FAIL0(err, RS_E_AUDIT_GOAL_ITEMS);
    RsLife lives[RS_KILL_LIVES];
    int32_t n_lives = 0, kills = 0;
    Kills k = {lives, &n_lives, &kills, ep->enemies, ep->n_enemies};
    for (int32_t i = ep->event_start; i < end; i++) RS_TRY(kill_feed(&k, &w->ev[i], err));
    *out = (RsAudit){.survived = ep->term_reason != RS_TERM_DEATH, .reason = ep->term_reason, .tick = ep->term_tick,
                     .produced = (int32_t)made, .retained = (int32_t)lower, .retained_upper = (int32_t)upper,
                     .picked = (int32_t)picked, .kills = kills};
    return 0;
}

/* ---- fights (combat.fights' summary, combat.py) ------------------------------------------------------ */
#define FIGHT_NPCS 64
/* fights(task)['summary'] as vec.COMBAT_FACTS reads it, over the window (events from arm with tick < the terminal
   tick): kill_lives of the window first (its AuditError), then the rows keyed (npc, life), each created by the
   agent's hit (that life) or an npc's hit on the agent (the life fights last saw for that npc: its hit's, one past
   a defeat, else 0) and an aggressor by whichever came first; heals by cause; eaten, the eat removals. */
int rs_fights(const RsSlot *s, int32_t out[6], RsErr *err) {
    const World *w = &s->w;
    const Episode *ep = &s->ep;
    if (!ep->terminal) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "fights of an episode without a terminal");
    RsLife lives[RS_KILL_LIVES];
    int32_t n_lives = 0, kills = 0;
    Kills k = {lives, &n_lives, &kills, ep->enemies, ep->n_enemies};
    for (int32_t i = ep->event_start; i < w->n_events; i++)
        if (w->ev[i].tick < ep->term_tick) RS_TRY(kill_feed(&k, &w->ev[i], err));
    struct { int32_t npc, life; uint8_t npc_first; } rows[RS_KILL_LIVES];
    struct { int32_t npc, life; } seen[FIGHT_NPCS];
    int32_t n_rows = 0, n_seen = 0, eats = 0, eaten = 0, eat_gain = 0, regen_gain = 0, levelup_gain = 0;
    for (int32_t i = ep->event_start; i < w->n_events; i++) {
        const RsEvent *e = &w->ev[i];
        if (e->tick >= ep->term_tick) continue;
        int32_t npc = -1, life = 0, npc_first = 0;
        if (e->kind == RS_EV_NPC_DAMAGE && e->a8) {
            npc = e->obj, life = e->v0;
        } else if (e->kind == RS_EV_NPC_DEFEATED) {
            int32_t j = 0;
            while (j < n_seen && seen[j].npc != e->obj) j++;
            if (j == n_seen) {
                if (n_seen >= FIGHT_NPCS) return RS_OVERFLOW(err, "fight npcs");
                seen[n_seen++].npc = e->obj;
            }
            seen[j].life = e->v0 + 1;
        } else if (e->kind == RS_EV_AGENT_HP && e->a8 == RS_HP_DAMAGE) {
            if (e->has_source) {
                npc = e->v2, npc_first = 1;
                for (int32_t j = 0; j < n_seen; j++)
                    if (seen[j].npc == npc) life = seen[j].life;
            }
        } else if (e->kind == RS_EV_AGENT_HP) {
            int32_t gained = e->v1 - e->v0;
            if (e->a8 == RS_HP_EAT) eats++, eat_gain += gained;
            else if (e->a8 == RS_HP_REGEN) regen_gain += gained;
            else if (e->a8 == RS_HP_LEVELUP) levelup_gain += gained;
        } else if (e->kind == RS_EV_REMOVE && e->a8 == RS_WHY_EAT) {
            eaten++;
        }
        if (npc < 0) continue;
        if (e->kind == RS_EV_NPC_DAMAGE) {  /* life[npc] = data['life'] */
            int32_t j = 0;
            while (j < n_seen && seen[j].npc != npc) j++;
            if (j == n_seen) {
                if (n_seen >= FIGHT_NPCS) return RS_OVERFLOW(err, "fight npcs");
                seen[n_seen++].npc = npc;
            }
            seen[j].life = life;
        }
        int32_t r = 0;
        while (r < n_rows && !(rows[r].npc == npc && rows[r].life == life)) r++;
        if (r == n_rows) {
            if (n_rows >= RS_KILL_LIVES) return RS_OVERFLOW(err, "fight rows");
            rows[n_rows].npc = npc, rows[n_rows].life = life, rows[n_rows++].npc_first = (uint8_t)npc_first;
        }
    }
    int32_t aggressed = 0;
    for (int32_t r = 0; r < n_rows; r++) aggressed += rows[r].npc_first;
    /* COMBAT_FACTS: eaten (eats + eats_without_gain), eat_gain, regen_gain, levelup_gain, aggressed, eats_without_gain */
    out[0] = eaten, out[1] = eat_gain, out[2] = regen_gain, out[3] = levelup_gain, out[4] = aggressed;
    out[5] = eaten - eats;
    return 0;
}

/* ---- facts (CpuBatch._facts, vec.py) ------------------------------------------------------------------ */
/* SupplyTracker.gap (search.py): ticks from the grove's exhaustion to the first log produced elsewhere at or
   after it, -1 for None. */
static int32_t supply_gap(const RsSupply *t) {
    if (t->exhausted < 0) return -1;
    for (int32_t i = 0; i < t->n_logs; i++)
        if (!t->log_local[i] && t->log_tick[i] >= t->exhausted) return t->log_tick[i] - t->exhausted;
    return -1;
}

static int popcount64(uint64_t v) {
    int n = 0;
    for (; v; v &= v - 1) n++;
    return n;
}

/* info['facts'] of the step that closes an episode, from its audit: group (1 + GROUPS index; 0 unless a combat
   start), success (the return is 1.0; a supply start: survival and at least the quota of certified logs), the combat
   start's fights summary, the search facts (no native task searches: 0, 0, -1), the visited cells (len(agent.visited))
   and the supply facts (1 + variant, the exhaustion tick and the gap, -1 for None; a supply start only). The outcome
   is the audit's kills for melee, else its retained items, against the episode's target (task.py). */
int rs_facts(const RsSlot *s, const RsAudit *report, int32_t facts[RS_N_FACTS], RsErr *err) {
    const Episode *ep = &s->ep;
    int32_t outcomes = ep->goal == RS_GOAL_MELEE ? report->kills : report->retained;
    if (ep->target < 1) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "an episode without a target");
    double total = report->survived ? (double)imin(outcomes, ep->target) / (double)ep->target : 0.0;
    memset(facts, 0, sizeof(int32_t) * RS_N_FACTS);
    facts[RS_FACT_SUCCESS] = total == 1.0;
    facts[RS_FACT_DISCOVERY_TICKS] = -1;
    facts[RS_FACT_CELLS] = popcount64(s->a.visited) + s->a.n_visited_off;  /* len(agent.visited) */
    facts[RS_FACT_EXHAUSTED_TICK] = facts[RS_FACT_ELSEWHERE_GAP] = -1;
    if (s->start.kind == RS_START_SUPPLY) {
        facts[RS_FACT_SUCCESS] = report->survived && outcomes >= RS_SUPPLY_QUOTA;
        facts[RS_FACT_SUPPLY] = 1 + s->start.variant;
        facts[RS_FACT_EXHAUSTED_TICK] = s->supply.exhausted < 0 ? -1 : s->supply.exhausted;
        facts[RS_FACT_ELSEWHERE_GAP] = supply_gap(&s->supply);
    }
    if (s->start.kind == RS_START_COMBAT) {
        int32_t f[6];
        RS_TRY(rs_fights(s, f, err));
        facts[RS_FACT_GROUP] = 1 + s->start.group;
        facts[RS_FACT_EATEN] = f[0], facts[RS_FACT_EAT_GAIN] = f[1], facts[RS_FACT_REGEN_GAIN] = f[2];
        facts[RS_FACT_LEVELUP_GAIN] = f[3], facts[RS_FACT_AGGRESSED] = f[4], facts[RS_FACT_EATS_WITHOUT_GAIN] = f[5];
    }
    return 0;
}

/* ---- the episode (GoalEpisode, task.py, with CpuBatch.step's info, vec.py) ---------- */
/* The SupplyTracker's and the gathering payment's goal family: GoalEpisode.family / run.family_ids, the goal's items
   and their notes (RsObjType.family_mask). */
static int ep_family(const RsSlot *s, int32_t obj) { return in_family(s->P, obj, s->ep.goal); }

/* GoalEpisode.carried (task.py): the goal family's items in the inventory (listed(): occupied slots). */
static int32_t carried(const RsSlot *s) {
    int64_t n = 0;
    for (int i = 0; i < RS_INV_CAP; i++)
        if (s->w.p.inv[i].id >= 0 && ep_family(s, s->w.p.inv[i].id)) n += s->w.p.inv[i].count;
    return (int32_t)n;
}

/* GoalEpisode.holdings (task.py): the three inventories as they stand. */
static void holdings(const RsSlot *s, RsHoldings *h) {
    memcpy(h->inv, s->w.p.inv, sizeof h->inv);
    memcpy(h->worn, s->w.p.worn, sizeof h->worn);
    memcpy(h->bank, s->w.p.bank, sizeof h->bank);
}

/* _watch's watched() after the cycle (task.py), once armed and until the terminal: death first (hitpoints
   at 0 or the dead flag; the terminal tick is world.tick, already incremented), else the horizon at tick >= end (the
   terminal tick is end); the holdings are taken here, before capture_text and keep_alive. */
void rs_task_watch(RsSlot *s) {
    Episode *ep = &s->ep;
    const Player *p = &s->w.p;
    if (ep->terminal || !ep->armed) return;
    if (p->levels[RS_SKILL_HITPOINTS] <= 0 || p->dead) {
        ep->terminal = 1, ep->term_reason = RS_TERM_DEATH, ep->term_tick = s->w.tick;
    } else if (s->w.tick >= ep->end) {
        ep->terminal = 1, ep->term_reason = RS_TERM_HORIZON, ep->term_tick = ep->end;
    } else {
        return;
    }
    holdings(s, &ep->term_holdings);
}

/* GoalEpisode.project's task inputs (task.py): budget_vec [(end - tick) / horizon, horizon / 10000] with tick
   the terminal's, else min(world.tick, end); critic_vec [banked / scale]; goal_vec one-hot in GOAL_VEC order. Each a
   Python float (double) stored into float32 once. */
void rs_task_project(const RsSlot *s, uint8_t *row) {
    const Episode *ep = &s->ep;
    int32_t tick = ep->terminal ? ep->term_tick : imin(s->w.tick, ep->end);
    float *budget = (float *)(row + RS_OBS_BUDGET_VEC_OFF), *critic = (float *)(row + RS_OBS_CRITIC_VEC_OFF);
    float *goal = (float *)(row + RS_OBS_GOAL_VEC_OFF);
    budget[0] = (float)((double)(ep->end - tick) / (double)ep->horizon);
    budget[1] = (float)((double)ep->horizon / 10000.0);
    critic[0] = (float)((double)ep->banked / ep->scale);
    for (int g = 0; g < RS_GOAL_VEC; g++) goal[g] = g == ep->goal ? 1.0f : 0.0f;
}

/* SupplyTracker.observe (search.py), at arm and after every step that was not unsupported: each log the
   world produced since the cursor, one entry per item, as (its tick from arm, local), local when the player stands
   (now, at the decision's end) within 3 tiles (Chebyshev) of a grove position; then the grove's standing trees
   (positions where an active loc's current type is a tree or an oak), and the first observation with none sets
   exhausted (ticks from arm). Tracker.observe's cells and sees_tree feed nothing CpuBatch reads. */
static int supply_local(const RsSupply *t, int32_t x, int32_t z) {
    for (int32_t i = 0; i < t->n_pos; i++)
        if (imax(iabs(x - t->pos_x[i]), iabs(z - t->pos_z[i])) <= 3) return 1;
    return 0;
}

int rs_supply_observe(RsSlot *s) {
    RsSupply *t = &s->supply;
    World *w = &s->w;
    if (!t->active) return 0;
    for (; t->seen < w->n_events; t->seen++) {
        const RsEvent *e = &w->ev[t->seen];
        if (e->kind != RS_EV_PRODUCE || !ep_family(s, e->obj)) continue;
        uint8_t local = (uint8_t)supply_local(t, w->p.x, w->p.z);
        for (int32_t k = 0; k < e->count; k++) {
            if (t->n_logs >= RS_SUPPLY_LOGS) return RS_OVERFLOW(&w->err, "RS_SUPPLY_LOGS supply logs");
            t->log_tick[t->n_logs] = e->tick - s->ep.start_tick;
            t->log_local[t->n_logs++] = local;
        }
    }
    int32_t standing = 0;
    for (int32_t i = 0; i < t->n_pos; i++) {
        uint8_t at[RS_MAX_LOCS];
        int n = rs_active_locs_at(w, 0, t->pos_x[i], t->pos_z[i], at);
        int found = 0;
        for (int k = 0; k < n && !found; k++) {
            const RsLocType *lt = rs_loc_type(s->P, w->locs[at[k]].type);
            found = lt && lt->supply_tree;
        }
        standing += found;
    }
    if (standing == 0 && t->exhausted < 0) t->exhausted = w->tick - s->ep.start_tick;
    return 0;
}

/* The start kind a slot's episodes draw (CpuBatch._episode, vec.py). */
static int slot_kind(const RsSlot *s) {
    if (s->cfg->task == RS_TASK_FREE) return RS_START_VILLAGE;
    if (s->cfg->task == RS_TASK_V5 && s->goal == RS_GOAL_WOOD) return RS_START_SUPPLY;
    if (s->cfg->task != RS_TASK_V2 && s->goal == RS_GOAL_MELEE) return RS_START_COMBAT;
    return RS_START_SHORT;
}

int rs_start_rec(const RsPack *P, RsStart *st, RsErr *err) {
    if (st->kind != RS_START_SUPPLY) {
        if (!pool_layout(P, RS_LAYOUT_COMPACT, st->layout, err)) return 1;
        st->rec = P->compact0 + st->layout;
        return 0;
    }
    const RsLayout *L = pool_layout(P, RS_LAYOUT_SUPPLY, st->layout, err);
    if (!L) return 1;
    if (st->variant == RS_VARIANT_AMPLE) {
        st->rec = P->supply0 + st->layout;
        return 0;
    }
    if (st->variant != RS_VARIANT_SCARCE || st->grove < 0 || st->grove >= L->n_groves)
        return RS_FAIL_TEXT(err, RS_E_CONTRACT, "a supply start names no variant or grove of its layout");
    int32_t keep = scarce_keep(L, st->grove, st->tile_x - L->x0, st->tile_z - L->z0);
    if (keep < 0) return RS_FAIL_TEXT(err, RS_E_PACK, "a supply grove has no normal tree");
    int32_t rec = rs_variant_find(P, st->layout, st->grove, keep);
    if (rec == RS_VARIANT_FAILS) return RS_FAIL(err, RS_E_SCARCE_CHECK, (int32_t)L->rec->seed, st->grove);
    if (rec < 0) return RS_FAIL_TEXT(err, RS_E_PACK, "the pack lacks the scarce variant of a drawn start");
    st->rec = rec;
    return 0;
}

/* GoalEpisode.__init__ (task.py) for the slot's goal: sustained combat (its target, horizon and enemies) for a
   combat start, the supply cap and horizon for a supply start, else the goal's target with the batch's horizon;
   scale 1.0 (CpuBatch's); never armed or active before _arm. */
static void episode_init(RsSlot *s, int kind) {
    Episode *ep = &s->ep;
    memset(ep, 0, sizeof *ep);
    ep->goal = (int8_t)s->goal;
    ep->combat = kind == RS_START_COMBAT;
    ep->scale = 1.0;
    if (kind == RS_START_VILLAGE) {   /* FreePlayEpisode.__init__: no goal (info goal -1), the life's own length */
        ep->goal = -1;
        ep->horizon = s->start.length;
        return;
    }
    if (ep->combat) {
        ep->horizon = RS_COMBAT_HORIZON, ep->target = RS_COMBAT_TARGET;
        ep->n_enemies = 2, ep->enemies[0] = s->P->consts->npc_goblin, ep->enemies[1] = s->P->consts->npc_mugger;
    } else {
        ep->horizon = kind == RS_START_SUPPLY ? RS_SUPPLY_HORIZON : s->cfg->horizon;
        ep->target = kind == RS_START_SUPPLY ? RS_SUPPLY_CAP : RS_GOAL_TARGETS[s->goal];
        ep->n_enemies = RS_N_MELEE_ENEMIES;
        for (int i = 0; i < RS_N_MELEE_ENEMIES; i++) ep->enemies[i] = RS_MELEE_ENEMIES[i];
    }
}

void rs_commit_obs(RsSlot *s, const uint8_t *row) {
    memcpy(s->obs, row, RS_OBS_ROW_BYTES);   /* obs-v6's progress block included */
    s->n_late = s->n_late_next;
    memcpy(s->late, s->late_next, (size_t)s->n_late * sizeof *s->late);
}

/* ---- free-play-v1 (sim/task.py FreePlayEpisode, sim/combat.py VILLAGE, sim/audit.py XpCheck) ------------------ */
/* draw_village then draw_life (combat.py, task.py): draw_start on the compact pool; the village world that plays the
   drawn layout (village_of: its record's world; none is a ValueError), whose muggers the batch may not ask for where
   it holds none; draw_combat in that world's start group (its R_GROUP draw skipped), whose mugger tiles are drawn
   (and their shortage raised) in both forms where the world places muggers and kept only when the batch places them;
   then the life's length, fixed or lo + int(draw(R_LIFE) * (hi - lo + 1)). */
int rs_draw_village(const RsPack *P, const RsConfig *cfg, int64_t key, int32_t slot, int32_t episode, RsStart *out,
                    RsErr *err) {
    if (cfg->life_lo < 1 || cfg->life_lo > RS_MAX_HORIZON
        || (cfg->life_kind == RS_LIFE_UNIFORM && (cfg->life_hi < cfg->life_lo || cfg->life_hi > RS_MAX_HORIZON))
        || (cfg->life_kind != RS_LIFE_UNIFORM && cfg->life_kind != RS_LIFE_FIXED))
        return RS_FAIL_TEXT(err, RS_E_CONTRACT, "a life length is ('fixed', ticks) or ('uniform', low, high) within 1..MAX_HORIZON");
    RS_TRY(rs_draw_start(P, RS_LAYOUT_COMPACT, key, slot, episode, out, err));
    int32_t v = P->layouts[out->rec].village;
    if (v < 0 || v >= RS_N_VILLAGES) return RS_FAIL_TEXT(err, RS_E_CONTRACT, RS_VILLAGE_NONE);
    if (cfg->muggers && !RS_VILLAGE_PLACE_MUGGERS[v]) return RS_FAIL_TEXT(err, RS_E_CONTRACT, RS_VILLAGE_NO_MUGGERS[v]);
    RS_TRY(draw_combat_in(P, key, slot, episode, out, RS_VILLAGE_GROUP[v], RS_VILLAGE_PLACE_MUGGERS[v], err));
    out->kind = RS_START_VILLAGE;
    out->placed = cfg->muggers != 0;
    if (!out->placed) out->n_muggers = 0;
    uint32_t h = start_prefix(key, slot, episode);
    out->length = cfg->life_kind == RS_LIFE_FIXED ? cfg->life_lo
        : cfg->life_lo + (int32_t)PICK(h, cfg->life_hi - cfg->life_lo + 1, RS_R_LIFE);
    return 0;
}

/* village_npcs (combat.py): combat_npcs of the start (the placed muggers only), then the layout's chickens at their
   spawns in layout order, with nids following and the type's size. */
int rs_village_npcs(const RsPack *P, const RsStart *s, RsNpcSpawn out[RS_MAX_NPCS], int *n, RsErr *err) {
    RS_TRY(rs_combat_npcs(P, s, out, n, err));
    const RsConsts *C = P->consts;
    const RsLayout *L = pool_layout(P, RS_LAYOUT_COMPACT, s->layout, err);
    if (!L) return 1;
    int k = *n;
    for (int32_t i = 0; i < L->n_npcs; i++) {
        int32_t type = L->npcs[i * 4 + 1], x = L->npcs[i * 4 + 2], z = L->npcs[i * 4 + 3];
        if (type != C->npc_chicken && type != C->npc_chicken_brown) continue;
        const RsNpcType *t = rs_npc_type(P, type);
        if (!t) return RS_FAIL_TEXT(err, RS_E_PACK, "a village chicken's type is not in the pack");
        if (k >= RS_MAX_NPCS) return RS_OVERFLOW(err, "RS_MAX_NPCS village npcs");
        out[k] = (RsNpcSpawn){.nid = k, .type = type, .x = L->x0 + x, .z = L->z0 + z, .level = 0, .size = t->size};
        k++;
    }
    *n = k;
    return 0;
}

/* audit.fractional_level: L + (x - X(L)) / (X(L+1) - X(L)) at XP x (tenths), L = level_for(x), 99.0 from level 99;
   ints exact in double, one division, then the int level added, as Python computes it. */
double rs_fractional_level(const RsPack *P, int32_t xp) {
    const int32_t *t = P->consts->level_xp;
    int32_t level = 1;
    for (int i = 98; i >= 0; i--)
        if (xp >= t[i]) {
            level = i + 2 < 99 ? i + 2 : 99;
            break;
        }
    if (level >= 99) return 99.0;
    int64_t low = level >= 2 ? t[level - 2] : 0, high = t[level - 1];
    return (double)level + (double)((int64_t)xp - low) / (double)(high - low);
}

/* The obs-v6 projection of a free-play life (FreePlayEpisode.project: project_v6 of the raw observation), built
   aside and kept only once whole. */
static int free_project(RsSlot *s) {
    _Alignas(8) uint8_t row[RS_OBS_ROW_BYTES];
    memset(row, 0, sizeof row);
    RS_TRY(rs_project(s, row));
    rs_project_progress(s, row);
    rs_commit_obs(s, row);
    return 0;
}

/* FreePlayEpisode._arm: the setup observation's feedback quieted, the clock (start_tick, end = tick + length), the
   event cursor, the arm hitpoints, stats and levels, XpCheck over the stats as they stand, the fractional levels over
   S, nothing paid, armed and active, then the first projection. No goal: nothing carried, banked or targeted. */
static int free_arm(RsSlot *s) {
    Episode *ep = &s->ep;
    World *w = &s->w;
    RsFree *f = &s->fp;
    const Player *p = &w->p;
    rs_episode_start_feedback(&s->a.obs);
    memset(f, 0, sizeof *f);
    f->length = s->start.length;
    ep->start_tick = w->tick, ep->end = w->tick + f->length;
    ep->event_start = (int32_t)(w->ev_base + w->n_events);
    f->checked = w->ev_base + w->n_events;
    ep->arm_hp = p->levels[RS_SKILL_HITPOINTS];
    memcpy(f->arm_stats, p->stats, sizeof f->arm_stats);
    memcpy(f->arm_levels, p->base_levels, sizeof f->arm_levels);
    memcpy(f->xp, p->stats, sizeof f->xp);
    for (int k = 0; k < RS_FREE_N_SKILLS; k++)
        f->level[k] = f->start_level[k] = rs_fractional_level(s->P, p->stats[RS_FREE_SKILLS[k]]);
    for (int i = 0; i < RS_MAX_NPCS; i++) f->row_life[i] = f->hit_life[i] = -1;
    ep->paid = 0.0, ep->decisions = 0;
    ep->armed = ep->active = 1;
    return free_project(s);
}

/* ---- XpCheck.observe, one tick's events at a time (sim/audit.py) */
#define XP_CAUSES 64              /* productions, hits and burials in one tick */
typedef struct {
    int32_t n_made, n_hit, n_bury, n_resp;
    const RsEvent *made[XP_CAUSES], *hit[XP_CAUSES], *bury[XP_CAUSES];
    const RsEvent *resp[RS_MAX_NPCS];  /* the tick's npc_respawn events (one an npc: a second of a life raises) */
    uint8_t made_used[XP_CAUSES], bury_used[XP_CAUSES];
    uint32_t hit_used[XP_CAUSES];  /* stats of the hit already given their XP */
} XpCauses;

/* XpCheck._respawn_cycle: the hit's npc life respawned in the hit's own tick (respawned[(npc, life)] == tick). A life's
   one respawn event is its key's only tick (a second raises before any XP of its tick is checked), so the tick's own
   npc_respawn events decide it. */
static int respawn_cycle(const XpCauses *c, const RsEvent *hit) {
    for (int32_t k = 0; k < c->n_resp; k++)
        if (c->resp[k]->obj == hit->obj && c->resp[k]->v0 == hit->v0) return 1;
    return 0;
}

/* XpCheck._percent: a hit's percent per stat (the style's HIT_PERCENT row, hitpoints 133 for every style); 0: none. */
static int32_t hit_percent(int32_t style, int32_t stat) {
    if (stat == RS_SKILL_HITPOINTS) return 133;
    if (style == 0) return stat == RS_SKILL_ATTACK ? 400 : 0;
    if (style == 1) return stat == RS_SKILL_STRENGTH ? 400 : 0;
    if (style == 2) return stat == RS_SKILL_DEFENCE ? 400 : 0;
    if (style == 3) return stat == RS_SKILL_ATTACK || stat == RS_SKILL_STRENGTH || stat == RS_SKILL_DEFENCE ? 133 : 0;
    return 0;
}

/* XpCheck._hit_amount: (min(damage, hp) * 10 * percent // 100) * the npc type's combat_xp_multiplier // 1000. */
static int hit_amount(const RsPack *P, const RsEvent *hit, int32_t stat, int64_t *out, RsErr *err) {
    const RsNpcType *nt = rs_npc_type(P, hit->count);
    if (!nt) return RS_FAIL_TEXT(err, RS_E_PACK, "a hit's npc type is not in the pack");
    int64_t dealt = hit->v1 < hit->v2 ? hit->v1 : hit->v2;
    *out = rs_floordiv(rs_floordiv(dealt * 10 * hit_percent(hit->v3, stat), 100) * nt->combat_xp_multiplier, 1000);
    return 0;
}

/* The stat a production script trains (PRODUCE_SKILL), -1 for a script outside it. */
static int produce_skill(const RsConsts *C, int32_t script) {
    if (script == C->s_produce_get_logs) return RS_SKILL_WOODCUTTING;
    if (script == C->s_produce_ore_normal || script == C->s_produce_ore_fast) return RS_SKILL_MINING;
    if (script == C->s_produce_fish_roll) return RS_SKILL_FISHING;
    return -1;
}

/* XpCheck.owed[script][obj] > 0: every table row of the script's skill whose product is obj gives XP (the least of
   them above 0); a product no row lists owes none. */
static int produce_owed(const RsPack *P, int32_t script, int32_t obj) {
    const RsConsts *C = P->consts;
    int found = 0;
    int32_t least = 0;
    if (script == C->s_produce_get_logs) {
        for (int32_t r = 0; r < P->n_trees; r++)
            if (P->trees[r].product == obj && (!found || P->trees[r].productexp < least))
                least = P->trees[r].productexp, found = 1;
    } else if (script == C->s_produce_ore_normal || script == C->s_produce_ore_fast) {
        for (int32_t r = 0; r < P->n_rocks; r++)
            if (P->rocks[r].output == obj && (!found || P->rocks[r].exp < least)) least = P->rocks[r].exp, found = 1;
    } else if (script == C->s_produce_fish_roll) {
        for (int32_t r = 0; r < P->n_fish; r++)
            if (P->fish[r].obj == obj) least = P->fish[r].productexp, found = 1;
    }
    return found && least > 0;
}

static int xp_fail(RsErr *err, const RsEvent *e, const char *why) {
    rs_fail(err, RS_E_AUDIT_XP, 2, (const int32_t[]){e->tick, e->a8});
    err->text = why;
    return 1;
}

/* XpCheck._product: the amount the table row gives the production: the tree's (by the xp's src, its loc type) or the
   compact rock's whose product is the obj, the fish's. Cooking has no C port. */
static int produce_amount(const RsPack *P, const RsEvent *x, int32_t script, int64_t *out, RsErr *err) {
    const RsConsts *C = P->consts;
    if (script == C->s_produce_get_logs || script == C->s_produce_ore_normal || script == C->s_produce_ore_fast) {
        const RsLocType *lt = rs_loc_type(P, x->v2);
        int wood = script == C->s_produce_get_logs;
        int32_t r = !lt ? -1 : wood ? lt->tree_row : lt->rock_row;
        if (r < 0 || (wood ? P->trees[r].product : P->rocks[r].output) != x->obj)
            return xp_fail(err, x, wood ? "no tree of that type gives that product" : "no rock of that type gives that ore");
        *out = wood ? P->trees[r].productexp : P->rocks[r].exp;
        return 0;
    }
    for (int32_t r = 0; r < P->n_fish; r++)
        if (P->fish[r].obj == x->obj) {
            *out = P->fish[r].productexp;
            return 0;
        }
    return xp_fail(err, x, "no fish row for that item");
}

/* XpCheck._amount: the cause in its tick (each production and burial used once, each stat of a hit once) and the
   amount the tables give it. */
static int xp_amount(const RsSlot *s, XpCauses *c, const RsEvent *x, int64_t *out, RsErr *err) {
    const RsPack *P = s->P;
    int32_t stat = x->a8;
    switch (x->b8) {
    case RS_XPC_PRODUCE: {
        int32_t k = 0;
        while (k < c->n_made && (c->made_used[k] || c->made[k]->obj != x->obj)) k++;
        if (k == c->n_made) return xp_fail(err, x, "no production of that item in its tick");
        c->made_used[k] = 1;
        int32_t script = c->made[k]->v0;
        if (produce_skill(P->consts, script) != stat || c->made[k]->count < 1)
            return xp_fail(err, x, "production by a script that trains no such skill");
        return produce_amount(P, x, script, out, err);
    }
    case RS_XPC_HIT: {
        int32_t k = 0;
        while (k < c->n_hit && !(c->hit[k]->obj == x->obj && c->hit[k]->v0 == x->v2)) k++;
        if (k == c->n_hit || !c->hit[k]->a8) return xp_fail(err, x, "no hit by this agent on that npc life in its tick");
        if (respawn_cycle(c, c->hit[k])) return xp_fail(err, x, "a hit in its npc life's respawn cycle gives no XP");
        if (!hit_percent(c->hit[k]->v3, stat) || (c->hit_used[k] >> stat & 1u))
            return xp_fail(err, x, "a hit in its style gives that stat no more XP");
        c->hit_used[k] |= 1u << stat;
        return hit_amount(P, c->hit[k], stat, out, err);
    }
    case RS_XPC_BURY: {
        int32_t k = 0;
        while (k < c->n_bury && (c->bury_used[k] || c->bury[k]->obj != x->obj)) k++;
        if (k == c->n_bury || stat != RS_SKILL_PRAYER) return xp_fail(err, x, "no bones of that type buried in its tick");
        c->bury_used[k] = 1;
        const RsObjType *t = rs_obj_type(P, x->obj);
        if (!t || t->bone_exp == RS_NONE) return xp_fail(err, x, "bones without a bone_exp param");
        *out = t->bone_exp;
        return 0;
    }
    case RS_XPC_LIGHT:   /* no light reaches the C engine (no firemaking port): a light's XP is never checked here */
        return RS_FAIL_TEXT(err, RS_E_NOT_PORTED, "XpCheck of a light's XP (firemaking)");
    default:
        return xp_fail(err, x, "no cause");
    }
}

/* One tick's events ev[a..b): the causes, every xp event checked in order (its stat's running XP before, the cause's
   amount at the declared rate after, capped), then XpCheck._owed: each cause that gives XP has had it. */
static int xp_tick(RsSlot *s, int32_t a, int32_t b) {
    World *w = &s->w;
    RsFree *f = &s->fp;
    RsErr *err = &w->err;
    XpCauses c;
    c.n_made = c.n_hit = c.n_bury = c.n_resp = 0;
    for (int32_t i = a; i < b; i++) {
        const RsEvent *e = &w->ev[i];
        if (e->kind == RS_EV_PRODUCE) {
            if (c.n_made >= XP_CAUSES) return RS_OVERFLOW(err, "XP_CAUSES productions in one tick");
            c.made_used[c.n_made] = 0, c.made[c.n_made++] = e;
        } else if (e->kind == RS_EV_HIT) {
            if (c.n_hit >= XP_CAUSES) return RS_OVERFLOW(err, "XP_CAUSES hits in one tick");
            c.hit_used[c.n_hit] = 0, c.hit[c.n_hit++] = e;
        } else if (e->kind == RS_EV_REMOVE && e->a8 == RS_WHY_BURY) {
            if (c.n_bury >= XP_CAUSES) return RS_OVERFLOW(err, "XP_CAUSES burials in one tick");
            c.bury_used[c.n_bury] = 0, c.bury[c.n_bury++] = e;
        } else if (e->kind == RS_EV_NPC_RESPAWN) {  /* XpCheck.respawned: each npc life respawns once */
            if (e->obj < 0 || e->obj >= RS_MAX_NPCS) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "a respawn outside the world");
            if (e->v0 < 0 || e->v0 >= RS_RESPAWN_LIVES) return RS_OVERFLOW(err, "RS_RESPAWN_LIVES lives of one npc");
            uint8_t *seen = &f->respawned[e->obj][e->v0 >> 3], bit = (uint8_t)(1u << (e->v0 & 7));
            if (*seen & bit) return RS_FAIL(err, RS_E_AUDIT_RESPAWN, e->obj, e->v0, e->tick);
            *seen |= bit;
            if (c.n_resp >= RS_MAX_NPCS) return RS_OVERFLOW(err, "RS_MAX_NPCS respawns in one tick");
            c.resp[c.n_resp++] = e;
        }
    }
    for (int32_t i = a; i < b; i++) {
        const RsEvent *x = &w->ev[i];
        if (x->kind != RS_EV_XP) continue;
        int32_t stat = x->a8;
        if (stat >= 21) return xp_fail(err, x, "a stat outside the player's");
        if (x->v0 != f->xp[stat]) return xp_fail(err, x, "the stat's XP before is not its checked XP");
        int64_t amount;
        RS_TRY(xp_amount(s, &c, x, &amount, err));
        int64_t after = (int64_t)f->xp[stat] + amount * s->cfg->xp_rate;
        if (after > 2000000000) after = 2000000000;   /* XP_CAP */
        if (x->count != amount || x->v1 != after) return xp_fail(err, x, "the tables give another amount at its rate");
        f->gained[stat] += (int64_t)x->v1 - x->v0;
        f->xp[stat] = x->v1;
    }
    for (int32_t k = 0; k < c.n_made; k++)
        if (!c.made_used[k] && c.made[k]->count >= 1 && produce_owed(s->P, c.made[k]->v0, c.made[k]->obj))
            return RS_FAIL(err, RS_E_AUDIT_XP_OWED, c.made[k]->tick), err->text = "production", 1;
    for (int32_t k = 0; k < c.n_hit; k++) {
        const RsEvent *hit = c.hit[k];
        if (!hit->a8 || hit->v3 < 0 || hit->v3 > 3 || respawn_cycle(&c, hit)) continue;  /* a respawn cycle owes none */
        for (int32_t stat = 0; stat < 21; stat++) {
            int64_t amount;
            if (!hit_percent(hit->v3, stat) || (c.hit_used[k] >> stat & 1u)) continue;
            RS_TRY(hit_amount(s->P, hit, stat, &amount, err));
            if (amount > 0) return RS_FAIL(err, RS_E_AUDIT_XP_OWED, hit->tick), err->text = "hit", 1;
        }
    }
    for (int32_t k = 0; k < c.n_bury; k++) {
        if (c.bury_used[k]) continue;
        const RsObjType *t = rs_obj_type(s->P, c.bury[k]->obj);
        if (!t || t->bone_exp == RS_NONE) return xp_fail(err, c.bury[k], "bones without a bone_exp param");
        if (t->bone_exp > 0) return RS_FAIL(err, RS_E_AUDIT_XP_OWED, c.bury[k]->tick), err->text = "bones buried", 1;
    }
    return 0;
}

/* XpCheck.observe over ev[first..n): tick by tick (every event of a tick lies in it). */
int rs_free_xp_check(RsSlot *s, int32_t first, int32_t n) {
    const World *w = &s->w;
    for (int32_t i = first; i < n;) {
        int32_t j = i;
        while (j < n && w->ev[j].tick == w->ev[i].tick) j++;
        RS_TRY(xp_tick(s, i, j));
        i = j;
    }
    return 0;
}

/* ---- fights' summary of a life (combat.fights), event by event over the window */
/* fights' row of (npc, life): new when that life is past the npc's last row (aggressor: whoever came first). An
   npc's lives only grow, so its rows come in life order; an event of an earlier row is a path C does not keep. */
static int fight_row(RsFree *f, int32_t npc, int32_t life, int npc_first, RsErr *err) {
    if (npc < 0 || npc >= RS_MAX_NPCS) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "a fight with an npc outside the world");
    if (f->row_life[npc] == life) return 0;
    if (f->row_life[npc] > life) return RS_FAIL_TEXT(err, RS_E_NOT_PORTED, "a fight row of an npc's earlier life");
    f->row_life[npc] = life;
    f->aggressed += npc_first;
    return 0;
}

static int fights_feed(RsFree *f, const RsEvent *e, RsErr *err) {
    switch (e->kind) {
    case RS_EV_NPC_DAMAGE:
        if (e->obj < 0 || e->obj >= RS_MAX_NPCS) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "damage to an npc outside the world");
        /* kill_lives' keys: every hit's (npc, life), whoever dealt it */
        if (e->v0 < f->hit_life[e->obj]) return RS_FAIL_TEXT(err, RS_E_NOT_PORTED, "a hit on an npc's earlier life");
        f->hit_life[e->obj] = e->v0;
        if (!e->a8) return 0;
        f->life_seen[e->obj] = e->v0, f->has_life[e->obj] = 1;   /* life[npc] = data['life'] */
        return fight_row(f, e->obj, e->v0, 0, err);
    case RS_EV_NPC_DEFEATED:
        if (e->obj < 0 || e->obj >= RS_MAX_NPCS) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "a defeat outside the world");
        if (e->v0 < f->hit_life[e->obj]) return RS_FAIL_TEXT(err, RS_E_NOT_PORTED, "a defeat of an npc's earlier life");
        if (e->v0 != f->hit_life[e->obj]) f->defeat_unhit = 1;   /* kill_lives: an npc defeated without a hit */
        f->life_seen[e->obj] = e->v0 + 1, f->has_life[e->obj] = 1;
        return 0;
    case RS_EV_AGENT_HP: {
        int32_t gained = e->v1 - e->v0;
        if (e->a8 == RS_HP_DAMAGE) {
            if (!e->has_source) return 0;
            int32_t npc = e->v2;
            if (npc < 0 || npc >= RS_MAX_NPCS) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "damage from an npc outside the world");
            return fight_row(f, npc, f->has_life[npc] ? f->life_seen[npc] : 0, 1, err);
        }
        if (e->a8 == RS_HP_EAT) f->eats++, f->eat_gain += gained;
        else if (e->a8 == RS_HP_REGEN) f->regen_gain += gained;
        else if (e->a8 == RS_HP_LEVELUP) f->levelup_gain += gained;
        return 0;
    }
    case RS_EV_REMOVE:
        f->eaten += e->a8 == RS_WHY_EAT;
        return 0;
    default:
        return 0;
    }
}

/* FreePlayEpisode._account, settled in steps: the events since the cursor (every event of a tick lies in one step,
   since a step ends at a cycle's end) through XpCheck, tick by tick; each stat's XP against its checked XP; the
   production (made by family), the agent's defeats (kills) and fights' window (tick < the terminal tick); the cursor
   moves past them and the World keeps only the last (rs_free_settle's compaction: world.events' length and last event
   stay the canonical record's). *pay is this step's change of fractional level over S: its terms in S's order, summed
   as the oracle's sum() sums them (rs_py_sum). */
static int free_settle(RsSlot *s, double *pay) {
    World *w = &s->w;
    RsFree *f = &s->fp;
    Episode *ep = &s->ep;
    RsErr *err = &w->err;
    if (f->checked < w->ev_base || f->checked > w->ev_base + w->n_events)
        return RS_FAIL_TEXT(err, RS_E_INTERNAL, "a free-play cursor outside the world's events");
    int32_t first = (int32_t)(f->checked - w->ev_base), n = w->n_events;
    RS_TRY(rs_free_xp_check(s, first, n));
    for (int stat = 0; stat < 21; stat++)
        if (w->p.stats[stat] != f->xp[stat]) return RS_FAIL(err, RS_E_AUDIT_XP_HELD, stat, w->p.stats[stat], f->xp[stat]);
    for (int32_t i = first; i < n; i++) {
        const RsEvent *e = &w->ev[i];
        if (e->kind == RS_EV_PRODUCE) {
            const RsObjType *t = rs_obj_type(s->P, e->obj);
            if (t && t->made_family >= 0) ep->made[t->made_family] += e->count;
        } else if (e->kind == RS_EV_NPC_DEFEATED) {
            f->kills += e->a8;
        }
        if (!ep->terminal || e->tick < ep->term_tick) RS_TRY(fights_feed(f, e, err));
    }
    f->checked = w->ev_base + n;
    double terms[RS_FREE_N_SKILLS];
    for (int k = 0; k < RS_FREE_N_SKILLS; k++) {
        double level = rs_fractional_level(s->P, w->p.stats[RS_FREE_SKILLS[k]]);
        terms[k] = level - f->level[k];
        f->level[k] = level;
    }
    *pay = rs_py_sum(terms, RS_FREE_N_SKILLS);
    /* the settled events leave the buffer, but the last, which the canonical record shows */
    if (n > 1) {
        int32_t drop = n - 1;
        w->ev[0] = w->ev[n - 1];
        w->ev_base += drop;
        w->n_events = 1;
        if (w->trace) w->trace->ev_seen = w->trace->ev_seen > drop ? w->trace->ev_seen - drop : 0;
    }
    return 0;
}

/* FreePlayEpisode.report's checks at the life's end: check_stats (each stat's XP from arm to end is its events'
   gain), the paid sum against the fractional gain over S (within 1e-6 of max(1, |gain|)); returns the integer level
   gain over S (info outcomes). */
static int free_report(RsSlot *s, int32_t *gain) {
    World *w = &s->w;
    RsFree *f = &s->fp;
    for (int stat = 0; stat < 21; stat++) {
        int64_t moved = (int64_t)w->p.stats[stat] - f->arm_stats[stat];
        if (moved != f->gained[stat])
            return RS_FAIL(&w->err, RS_E_AUDIT_XP_STATS, stat, (int32_t)moved, (int32_t)f->gained[stat]);
    }
    double terms[RS_FREE_N_SKILLS];
    for (int k = 0; k < RS_FREE_N_SKILLS; k++) terms[k] = f->level[k] - f->start_level[k];
    double fractional = rs_py_sum(terms, RS_FREE_N_SKILLS);  /* report's sum(b - a for ...) */
    double bound = fabs(fractional) > 1.0 ? fabs(fractional) : 1.0;
    if (fabs(s->ep.paid - fractional) > 1e-6 * bound) return RS_FAIL0(&w->err, RS_E_AUDIT_PAID);
    int32_t total = 0;
    for (int k = 0; k < RS_FREE_N_SKILLS; k++)
        total += w->p.base_levels[RS_FREE_SKILLS[k]] - f->arm_levels[RS_FREE_SKILLS[k]];
    *gain = total;
    return 0;
}

/* FreePlayEpisode.step with CpuBatch._step_free's info: the action against the last mask, the decision counted, the
   agent's step until the terminal (an Unsupported closes the life and fails the call; the agent's limits must not
   cut it), the settlement and its pay, the projection; made and kills so far; at the life's end its report, then
   fights' window (kill_lives' error) and the facts: no group, success, search or supply, the fights' food and
   healing, the visited cells. goal -1: no goal. */
static int free_step(RsSlot *s, int32_t action, RsStepInfo *out) {
    World *w = &s->w;
    Episode *ep = &s->ep;
    RsFree *f = &s->fp;
    memset(out, 0, sizeof *out);
    out->goal = -1;
    int32_t before = w->tick;
    if (!ep->active) return RS_FAIL0(&w->err, RS_E_CLOSED_LIFE);
    const int8_t *mask = (const int8_t *)(s->obs + RS_OBS_MASK_OFF);
    if (action < 0 || action >= RS_N_ACTIONS || !mask[action]) return RS_FAIL(&w->err, RS_E_MASKED_ACTION, action);
    ep->decisions++;
    if (!ep->terminal) {
        if (rs_agent_step(s, action)) {
            if (w->err.code == RS_UNSUPPORTED) ep->active = 0;
            return 1;
        }
        if (rs_agent_truncated(&s->a)) return RS_FAIL0(&w->err, RS_E_FREE_CUT);
    }
    double pay;
    RS_TRY(free_settle(s, &pay));
    ep->paid += pay;
    RS_TRY(free_project(s));
    out->reward = (float)pay;
    out->decision_ticks = w->tick - before;
    for (int k = 0; k < 3; k++) out->made[k] = ep->made[k];
    out->kills = f->kills;
    if (!ep->terminal) return 0;
    ep->active = 0;
    RS_TRY(free_report(s, &out->outcomes));
    if (f->defeat_unhit) return RS_FAIL0(&w->err, RS_E_AUDIT_DEFEAT);
    out->done = 1;
    out->terminal = ep->term_reason == RS_TERM_DEATH ? RS_TERMINAL_DEATH : RS_TERMINAL_HORIZON;
    out->episode_return = (float)ep->paid;
    out->episode_decisions = ep->decisions;
    out->episode_ticks = ep->term_tick - ep->start_tick;
    int32_t *facts = out->facts;
    memset(facts, 0, sizeof out->facts);
    facts[RS_FACT_DISCOVERY_TICKS] = -1;
    facts[RS_FACT_CELLS] = popcount64(s->a.visited) + s->a.n_visited_off;   /* len(agent.visited) */
    facts[RS_FACT_EXHAUSTED_TICK] = facts[RS_FACT_ELSEWHERE_GAP] = -1;
    facts[RS_FACT_EATEN] = f->eaten, facts[RS_FACT_EAT_GAIN] = f->eat_gain, facts[RS_FACT_REGEN_GAIN] = f->regen_gain;
    facts[RS_FACT_LEVELUP_GAIN] = f->levelup_gain, facts[RS_FACT_AGGRESSED] = f->aggressed;
    facts[RS_FACT_EATS_WITHOUT_GAIN] = f->eaten - f->eats;
    return 0;
}

/* GoalEpisode._arm (task.py:216-229) with the projection it returns: the setup observation's feedback quieted
   (feedback_contract.episode_start), the clock (start_tick, end), the event cursor, the arm holdings and hitpoints,
   the gathering count carried (goal items held at arm are a ValueError), the payment at 0, armed and active; then the
   ledgers start at the cursor (made, lives, kills 0) and project() writes the first observation. */
static int episode_arm(RsSlot *s) {
    Episode *ep = &s->ep;
    World *w = &s->w;
    rs_episode_start_feedback(&s->a.obs);
    ep->start_tick = w->tick, ep->end = w->tick + ep->horizon;
    ep->event_start = w->n_events;
    holdings(s, &ep->initial);
    ep->arm_hp = w->p.levels[RS_SKILL_HITPOINTS];
    ep->banked = 0, ep->carried_before = carried(s), ep->bank_was_open = s->a.obs.bank_open != 0;
    if (ep->carried_before) return RS_FAIL0(&w->err, RS_E_GOAL_ITEMS_AT_ARM);
    ep->paid = 0.0, ep->decisions = 0;
    ep->armed = ep->active = 1;
    ep->ev_seen = ep->event_start;
    _Alignas(8) uint8_t row[RS_OBS_ROW_BYTES];
    memset(row, 0, sizeof row);
    RS_TRY(rs_project(s, row));
    rs_task_project(s, row);
    rs_commit_obs(s, row);
    return 0;
}

/* CpuBatch._episode from its start (vec.py; search.py supply_episode): a fresh World on the start's layout
   record (rs_world_init) with combat_npcs for a combat start and the task's exclusions, traced when the slot is;
   the GoalEpisode; GoalEpisode.start (task.py): agent.start (seed 0, the kit, the levels, the tile: login and
   arrive), the hitpoints of a combat start written then (ValueError outside 1..base), the STYLE_0 setup step, for a
   combat start with its weapon worn the Wield step (the first heldSel entry with the weapon's name and option Wield;
   none is a ValueError), then _arm; finally a supply start's SupplyTracker over the pool layout's grove rows, which
   observes at once (Tracker.__init__). An Unsupported in any of it fails the call. */
int rs_episode_begin(RsSlot *s, const RsStart *start) {
    const RsPack *P = s->P;
    World *w = &s->w;
    const RsConsts *C = P->consts;
    int kind = slot_kind(s);
    if (!C) return RS_FAIL_TEXT(&w->err, RS_E_PACK, "the pack has no constants");
    if (start->kind != kind) return RS_FAIL_TEXT(&w->err, RS_E_CONTRACT, "a start of another kind than its slot draws");
    if (start->rec < 0 || start->rec >= P->n_layouts)
        return RS_FAIL_TEXT(&w->err, RS_E_CONTRACT, "a start names no layout record (rs_start_rec)");
    RsNpcSpawn spawns[RS_MAX_NPCS];
    int n_spawns = 0;
    int village = kind == RS_START_VILLAGE, fights = kind == RS_START_COMBAT || village;
    if (kind == RS_START_COMBAT) RS_TRY(rs_combat_npcs(P, start, spawns, &n_spawns, &w->err));
    if (village) {
        if (start->length < 1 || start->length > RS_MAX_HORIZON || !start->placed != !s->cfg->muggers
            || (!start->placed && start->n_muggers))
            return RS_FAIL_TEXT(&w->err, RS_E_CONTRACT, "a village start of another life length or muggers setting");
        RS_TRY(rs_village_npcs(P, start, spawns, &n_spawns, &w->err));
    }
    s->start = *start;
    s->has_start = 1;
    int32_t v = village ? P->layouts[start->rec].village : -1;   /* village_life: village_of(layout) */
    if (village && (v < 0 || v >= RS_N_VILLAGES)) return RS_FAIL_TEXT(&w->err, RS_E_CONTRACT, RS_VILLAGE_NONE);
    uint8_t excl = kind == RS_START_COMBAT ? RS_COMBAT_EXCLUSIONS : kind == RS_START_SUPPLY ? RS_SUPPLY_EXCLUSIONS
        : village ? RS_VILLAGE_EXCLUSIONS[v] : 0;
    RS_TRY(rs_world_init(w, P, &P->layouts[start->rec], start->world_seed, RS_WORLD_START_TICK, fights ? spawns : NULL,
                         n_spawns, excl, s->cfg->xp_rate));
    if (village && RS_VILLAGE_OWN_LOOT[v]) w->loot = RS_OF_LOOT_OUTSIDE_OWN;   /* World(loot_outside=its list) */
    w->trace = s->trace;
    if (s->trace) rs_trace_begin(s->trace, s->episode);
    episode_init(s, kind);
    memset(&s->supply, 0, sizeof s->supply);
    s->supply.exhausted = -1;
    int32_t kit[RS_START_KIT_MAX], count[RS_START_KIT_MAX];
    int n_kit = rs_start_kit(P, start, kit, count);
    RS_TRY(rs_agent_start(s, 0, start->tile_x, start->tile_z, kit, count, n_kit, start->levels));
    if (fights) {
        int32_t base = w->p.base_levels[RS_SKILL_HITPOINTS];
        if (start->hp < 1 || start->hp > base) return RS_FAIL(&w->err, RS_E_HP_RANGE, start->hp, base);
        w->p.levels[RS_SKILL_HITPOINTS] = (int16_t)start->hp;
    }
    RS_TRY(rs_agent_step(s, RS_A_STYLE0));
    if (fights && start->worn) {
        const RsObjType *t = rs_obj_type(P, C->weapons[start->weapon]);
        if (!t) return RS_FAIL_TEXT(&w->err, RS_E_PACK, "a start weapon is not in the pack");
        int32_t slot = -1;
        for (int32_t k = 0; k < s->a.obs.n_held && slot < 0; k++)
            if (s->a.obs.held_sel[k].name == t->name && s->a.obs.held_sel[k].option == C->s_wield) slot = k;
        if (slot < 0) return RS_FAIL(&w->err, RS_E_NO_WIELD, t->name);
        RS_TRY(rs_agent_step(s, RS_A_HELD0 + slot));
    }
    RS_TRY(village ? free_arm(s) : episode_arm(s));
    if (kind == RS_START_SUPPLY) {
        const RsLayout *L = &P->layouts[P->supply0 + start->layout];
        RsSupply *t = &s->supply;
        int32_t a = L->grove_off[start->grove], b = L->grove_off[start->grove + 1];
        if (b - a > RS_GROVE_MAX) return RS_OVERFLOW(&w->err, "RS_GROVE_MAX grove rows");
        for (int32_t r = a; r < b; r++) {
            int32_t k = L->grove_row[r];
            t->pos_x[t->n_pos] = (int16_t)(L->x0 + L->locs[k * 5 + 1]);
            t->pos_z[t->n_pos++] = (int16_t)(L->z0 + L->locs[k * 5 + 2]);
        }
        t->active = 1;
        t->seen = s->ep.event_start;
        RS_TRY(rs_supply_observe(s));
    }
    return 0;
}

int rs_episode_new(RsSlot *s, int64_t key) {
    RsStart start;
    s->has_start = 0;  /* CpuBatch._start: starts[b] = None before the draw */
    RS_TRY(rs_episode_draw(s, key, &start, &s->w.err));
    return rs_episode_begin(s, &start);
}

/* GoalEpisode.track_bank (task.py): banked follows what an open bank took (no declared world has a bank). */
static void track_bank(RsSlot *s) {
    Episode *ep = &s->ep;
    int32_t now = carried(s);
    if (s->a.obs.bank_open || ep->bank_was_open) ep->banked += ep->carried_before - now;
    ep->carried_before = now;
    ep->bank_was_open = s->a.obs.bank_open != 0;
}

/* GoalEpisode.step (task.py) and the info CpuBatch.step derives from it (vec.py): the action checked
   against the last projected mask (ValueError), the decision counted, the agent's step (an Unsupported closes the
   episode and fails the call: the batch flags the world), the bank tracked, the projection; then a running step pays
   held - paid, held = min(count, target) / target * scale (count: certified kills since arm for melee, read after
   the ledgers take this step's events, else banked + carried), and the terminal step settles to the audit's return
   (min(outcomes, target) / target if the player survived, else 0) less what was paid, the audit taken before the
   step ledgers (audit() precedes vec.step's kills()). Then the SupplyTracker observes, made and kills are the
   ledgers', and a terminal step adds the ended episode's fields and facts. The projection is built aside and kept
   only once it is whole, so a failed step leaves the last observation (CpuBatch's stale task.projected). */
int rs_episode_step(RsSlot *s, int32_t action, RsStepInfo *out) {
    if (s->start.kind == RS_START_VILLAGE) return free_step(s, action, out);
    World *w = &s->w;
    Episode *ep = &s->ep;
    memset(out, 0, sizeof *out);
    out->goal = ep->goal;
    int32_t before = w->tick;
    if (!ep->active) return RS_FAIL0(&w->err, RS_E_CLOSED_EPISODE);
    const int8_t *mask = (const int8_t *)(s->obs + RS_OBS_MASK_OFF);
    if (action < 0 || action >= RS_N_ACTIONS || !mask[action]) return RS_FAIL(&w->err, RS_E_MASKED_ACTION, action);
    ep->decisions++;
    if (!ep->terminal) {
        if (rs_agent_step(s, action)) {
            if (w->err.code == RS_UNSUPPORTED) ep->active = 0;
            return 1;
        }
        track_bank(s);
    }
    _Alignas(8) uint8_t row[RS_OBS_ROW_BYTES];
    memset(row, 0, sizeof row);
    RS_TRY(rs_project(s, row));
    rs_task_project(s, row);
    rs_commit_obs(s, row);
    RsAudit report = {0};
    double total = 0.0;
    if (!ep->terminal) {
        RS_TRY(rs_ledger_update(s));
        int32_t count = ep->goal == RS_GOAL_MELEE ? ep->kills : ep->banked + ep->carried_before;
        double held = (double)imin(count, ep->target) / (double)ep->target * ep->scale;
        out->reward = (float)((held - ep->paid) / ep->scale);
        ep->paid = held;
    } else {
        RS_TRY(rs_audit(s, &report, &w->err));
        out->outcomes = ep->goal == RS_GOAL_MELEE ? report.kills : report.retained;
        total = report.survived ? (double)imin(out->outcomes, ep->target) / (double)ep->target : 0.0;
        out->reward = (float)(total - ep->paid / ep->scale);
        ep->active = 0;
        out->done = 1;
        RS_TRY(rs_ledger_update(s));
    }
    out->decision_ticks = w->tick - before;
    RS_TRY(rs_supply_observe(s));
    for (int f = 0; f < 3; f++) out->made[f] = ep->made[f];
    out->kills = ep->kills;
    if (out->done) {
        out->terminal = ep->term_reason == RS_TERM_DEATH ? RS_TERMINAL_DEATH : RS_TERMINAL_HORIZON;
        out->episode_return = (float)total;
        out->episode_decisions = ep->decisions;
        out->episode_ticks = ep->term_tick - ep->start_tick;
        RS_TRY(rs_facts(s, &report, out->facts, &w->err));
    }
    return 0;
}
