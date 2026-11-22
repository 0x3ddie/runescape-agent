/* sim/native/combat.c: the melee loop of sim/engine.py between the player and npcs (engine.py):
 * equip bonuses, weapon styles and player_combat_stat; the player's combat scripts (player_combat_start[_ap],
 * player_melee_attack with its hit and defence rolls and give_combat_experience, combat_damage_player, player_death,
 * playerhit_n_retaliate); the npc's (npc_default_attack and playerhit_n_melee, npc_default_retaliate,
 * npc_default_damage, npc_death); the death scripts as the exporter's drop programs run with DeathContext's
 * semantics (engine.py, 2775-2901; sim/rs2.py run/_block/_expr); Player.applyDamage; and the ground-item
 * ops OBJ_ADD and OBJ_TAKEITEM (engine.py) on world.c's objs list. Each function names the Python it
 * mirrors (file and function, as rs.h).
 *
 * Exactness notes the code relies on:
 *   - The world has one player, the agent's: a hero, an aggressor, a receiver or an event source is that player or
 *     None, so each is a flag (rs.h Npc); World.is_multi is false in every declared world (rs.h World).
 *   - None of these scripts suspends except player_death (delay 1, then delay 3) and the death script (npc_death's
 *     arrivedelay and npc_delay), so the others run to their end in one step, as their `yield from ()` does.
 *   - Draw keys are the oracle's: (R_SCRIPT, tick, nid, 1|2|3) for the player's rolls, (R_SCRIPT, tick, nid, 11|12|13)
 *     for the npc's, (R_NPC_SCRIPT, tick, nid, 1000 + k) for a death's k-th draw (DeathContext.random).
 *   - rs2.py evaluates every entry of its arithmetic dict, so DOP_ARITH raises ZeroDivisionError for every operator
 *     when its right side is 0 (rs.h DOP_ARITH); Python's ints are unbounded, so a register value past int32 is
 *     RS_E_OVERFLOW here, never a wrapped value. */
#include <string.h>

#include "rs.h"

static inline int32_t imax(int32_t a, int32_t b) { return a > b ? a : b; }
static inline int32_t imin(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t iabs(int32_t a) { return a < 0 ? -a : a; }

static const RsNpcType *ntype(const World *w, const Npc *n) {
    const RsNpcType *nt = rs_npc_type(w->P, n->type);
    if (!nt) rs_trap("combat.c: an npc of a type the pack lacks (KeyError)");
    return nt;
}

static const RsObjType *otype(const World *w, int32_t obj) {
    const RsObjType *ot = rs_obj_type(w->P, obj);
    if (!ot) rs_trap("combat.c: an obj type the pack lacks (KeyError)");
    return ot;
}

/* npc_param / obj_param with the param defaults the pack resolved; RS_NONE is a param with no value, a KeyError in
   Python (no reachable type has one). */
static int param(World *w, int32_t value, int32_t *out) {
    if (value == RS_NONE) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a param without a value (a KeyError)");
    *out = value;
    return 0;
}

/* Npc.uid (engine.py). */
static int32_t uid_of(const Npc *n) { return (int32_t)(((uint32_t)n->type << 16) | (uint16_t)n->nid); }

/* The wielded weapon: p.worn.items[WEARPOS_RHAND][0], -1 for None (an occupied [id, 0] slot counts). */
static int32_t rhand(const World *w) { return w->p.worn[RS_WEARPOS_RHAND].id; }

/* ---- equipment and combat stats (engine.py) ------------------------------------------------------------ */
typedef struct { int64_t bonus[12], range; } Bonuses;   /* engine.BONUSES order, then rangebonus */

/* [proc,equip_get_bonuses] (engine.py): every worn obj's bonuses; a quiver's rangebonus only beside a
   matching bow or crossbow. */
static int equip_bonuses(World *w, Bonuses *b) {
    memset(b, 0, sizeof *b);
    int32_t weapon = rhand(w);
    const RsObjType *wt = weapon >= 0 ? otype(w, weapon) : NULL;
    for (int slot = 0; slot < RS_WORN_CAP; slot++) {
        int32_t obj = w->p.worn[slot].id;
        if (obj < 0) continue;
        const RsObjType *ot = otype(w, obj);
        int32_t v;
        if (slot != RS_WEARPOS_QUIVER || !wt || ((ot->flags & RS_OF_ARROWS) && (wt->flags & RS_OF_BOW))
            || ((ot->flags & RS_OF_BOLTS) && (wt->flags & RS_OF_CROSSBOW))) {
            RS_TRY(param(w, ot->rangebonus, &v));
            b->range += v;
        }
        for (int k = 0; k < 12; k++) {
            RS_TRY(param(w, ot->bonus[k], &v));
            b->bonus[k] += v;
        }
    }
    return 0;
}
enum { B_STABATTACK, B_SLASHATTACK, B_CRUSHATTACK, B_MAGICATTACK, B_RANGEATTACK, B_STABDEFENCE, B_SLASHDEFENCE,
       B_CRUSHDEFENCE, B_MAGICDEFENCE, B_RANGEDEFENCE, B_STRENGTH, B_PRAYER };

/* [proc,combat_get_weapon_style_data] (engine.py): the exporter resolved each obj's table for the members
   world; a members weapon in a free world is unarmed. */
const RsStyleTable *rs_weapon_styles(const World *w, int32_t obj) {
    const RsPack *P = w->P;
    int32_t row = P->consts->style_unarmed;
    if (obj >= 0) {
        const RsObjType *ot = otype(w, obj);
        if (w->members || !ot->members) row = ot->style_table;
    }
    if (row < 0 || row >= P->n_styles) rs_trap("rs_weapon_styles: a style row outside the pack's STYLES");
    return &P->styles[row];
}

/* [proc,combat_stat]: effective * (bonus + 64). */
static int64_t combat_stat(int64_t effective, int64_t bonus) { return effective * (bonus + 64); }

static int set_varp(World *w, int varp, int64_t value) {
    if (value < INT32_MIN || value > INT32_MAX) return RS_OVERFLOW(&w->err, "a combat varp past int32");
    w->p.varp[varp] = (int32_t)value;
    return 0;
}

/* [proc,player_combat_stat] (engine.py): rolls and max hit from the current levels, the worn bonuses and
   the wielded weapon's style (no prayer: combat_effective_stat is the level); com_mode is clamped and stored back. */
int rs_player_combat_stat(World *w) {
    Player *p = &w->p;
    Bonuses b;
    RS_TRY(equip_bonuses(w, &b));
    const RsStyleTable *st = rs_weapon_styles(w, rhand(w));
    p->varp[RS_VARP_com_mode] = imin(p->varp[RS_VARP_com_mode], st->n - 1);
    int32_t mode = p->varp[RS_VARP_com_mode], at = mode < 0 ? mode + st->n : mode;   /* Python's negative index */
    if (at < 0 || at >= st->n) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a style index past its table (IndexError)");
    int32_t damagetype = st->damagetype[at], damagestyle = st->damagestyle[at];
    int32_t attack = 0, strength = 0, defence = 0, ranged = 0;   /* STYLE_BONUSES.get(damagestyle, (0, 0, 0, 0)) */
    for (int k = 0; k < (int)sizeof RS_STYLE_BONUS_STYLES; k++)
        if (RS_STYLE_BONUS_STYLES[k] == damagestyle) {
            attack = RS_STYLE_BONUSES[4 * k], strength = RS_STYLE_BONUSES[4 * k + 1];
            defence = RS_STYLE_BONUSES[4 * k + 2], ranged = RS_STYLE_BONUSES[4 * k + 3];
        }
    const int16_t *lv = p->levels;
    int64_t magic_defence = rs_floordiv(7 * (int64_t)lv[RS_SKILL_MAGIC] + 3 * (int64_t)lv[RS_SKILL_DEFENCE], 10) + 8;
    int64_t eff_attack = lv[RS_SKILL_ATTACK] + 8 + attack, eff_defence = lv[RS_SKILL_DEFENCE] + 8 + defence;
    int64_t eff_ranged = lv[RS_SKILL_RANGED] + 8 + ranged;
    p->varp[RS_VARP_damagetype] = damagetype;
    p->varp[RS_VARP_damagestyle] = damagestyle;
    RS_TRY(set_varp(w, RS_VARP_com_stabattack, combat_stat(eff_attack, b.bonus[B_STABATTACK])));
    RS_TRY(set_varp(w, RS_VARP_com_slashattack, combat_stat(eff_attack, b.bonus[B_SLASHATTACK])));
    RS_TRY(set_varp(w, RS_VARP_com_crushattack, combat_stat(eff_attack, b.bonus[B_CRUSHATTACK])));
    RS_TRY(set_varp(w, RS_VARP_com_magicattack, combat_stat(lv[RS_SKILL_MAGIC] + 9, b.bonus[B_MAGICATTACK])));
    RS_TRY(set_varp(w, RS_VARP_com_rangeattack, combat_stat(eff_ranged, b.bonus[B_RANGEATTACK])));
    RS_TRY(set_varp(w, RS_VARP_com_stabdef, combat_stat(eff_defence, b.bonus[B_STABDEFENCE])));
    RS_TRY(set_varp(w, RS_VARP_com_slashdef, combat_stat(eff_defence, b.bonus[B_SLASHDEFENCE])));
    RS_TRY(set_varp(w, RS_VARP_com_crushdef, combat_stat(eff_defence, b.bonus[B_CRUSHDEFENCE])));
    RS_TRY(set_varp(w, RS_VARP_com_magicdef, combat_stat(magic_defence, b.bonus[B_MAGICDEFENCE])));
    RS_TRY(set_varp(w, RS_VARP_com_rangedef, combat_stat(eff_defence, b.bonus[B_RANGEDEFENCE])));
    if (damagetype == RS_STAB || damagetype == RS_SLASH || damagetype == RS_CRUSH)   /* [proc,combat_maxhit] */
        RS_TRY(set_varp(w, RS_VARP_com_maxhit, rs_floordiv(combat_stat(lv[RS_SKILL_STRENGTH] + 8 + strength,
                                                                        b.bonus[B_STRENGTH]) + 320, 640)));
    else if (damagetype == RS_RANGED_STYLE)
        RS_TRY(set_varp(w, RS_VARP_com_maxhit, rs_floordiv(combat_stat(eff_ranged, b.range) + 320, 640)));
    return 0;
}

/* ---- the player's side (engine.py) --------------------------------------------------------------------- */
/* Player.applyDamage with its agent_hp evidence (engine.py); every hit, a zero included. */
int rs_player_damage(World *w, int32_t damage, int32_t source_nid) {
    Player *p = &w->p;
    int32_t before = p->levels[RS_SKILL_HITPOINTS];
    p->levels[RS_SKILL_HITPOINTS] = (int16_t)imax(0, before - damage);
    p->dead |= p->levels[RS_SKILL_HITPOINTS] <= 0;
    RsEvent e = {.tick = w->tick, .kind = RS_EV_AGENT_HP, .a8 = RS_HP_DAMAGE, .has_source = source_nid >= 0,
                 .count = damage, .v0 = before, .v1 = p->levels[RS_SKILL_HITPOINTS],
                 .v2 = source_nid >= 0 ? source_nid : 0};
    return rs_event(w, &e);
}

/* [proc,player_in_combat_check] (engine.py): single-way combat allows one opponent each. */
static int in_combat_check(World *w, const Npc *n, int *ok) {
    const Player *p = &w->p;
    int32_t msg;
    *ok = 0;
    if (p->varp[RS_VARP_lastcombat_pvp] + 8 > w->tick
        || (p->varp[RS_VARP_lastcombat] + 8 > w->tick && p->varp[RS_VARP_aggressive_npc] != uid_of(n)
            && p->varp[RS_VARP_aggressive_npc] != -1)) {
        RS_TRY(rs_text_msg(w, RS_TX_under_attack, 0, &msg));
        return rs_mes(w, msg);
    }
    if (n->v_lastcombat + 8 > w->tick && !n->v_aggressive_player) {   /* aggressive_player != p.name */
        RS_TRY(rs_text_msg(w, RS_TX_fighting_other, 0, &msg));
        return rs_mes(w, msg);
    }
    *ok = 1;   /* macro_target is always None (rs.h Npc) */
    return 0;
}

/* [proc,player_attackrange] (engine.py): the weapon's attackrange (melee 0), longrange + 2. */
static int attackrange(World *w, int32_t *out) {
    int32_t weapon = rhand(w), v;
    *out = 0;
    if (weapon < 0) return 0;
    RS_TRY(param(w, otype(w, weapon)->attackrange, &v));
    v = imin(v, 10);
    *out = w->p.varp[RS_VARP_damagestyle] == 6 ? imin(v + 2, 10) : v;
    return 0;
}

/* npc_is_attackable (engine.py): its quest npcs raise. */
static int attackable(World *w, const Npc *n, int *ok) {
    const RsNpcType *nt = ntype(w, n);
    if (nt->refuse_attack) return RS_FAIL(&w->err, RS_E_ATTACKING, nt->debugname);
    *ok = nt->attackable;
    return 0;
}

/* ROLL_VARPS: the attack and defence varps of a damage type (KeyError past MAGIC_STYLE). */
static int roll_varps(World *w, int32_t damagetype, int *attack, int *defence) {
    static const int ATT[5] = {RS_VARP_com_stabattack, RS_VARP_com_slashattack, RS_VARP_com_crushattack,
                               RS_VARP_com_rangeattack, RS_VARP_com_magicattack};
    static const int DEF[5] = {RS_VARP_com_stabdef, RS_VARP_com_slashdef, RS_VARP_com_crushdef, RS_VARP_com_rangedef,
                               RS_VARP_com_magicdef};
    if (damagetype < 0 || damagetype > 4) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a damage type past ROLL_VARPS");
    *attack = ATT[damagetype], *defence = DEF[damagetype];
    return 0;
}

/* npc_defence_roll (engine.py): (stat + 9) * (the npc's defence bonus + 64). */
static int npc_defence_roll(World *w, const Npc *n, int32_t damagetype, int64_t *out) {
    const RsNpcType *nt = ntype(w, n);
    const int32_t bonus[5] = {nt->stabdefence, nt->slashdefence, nt->crushdefence, nt->rangedefence, nt->magicdefence};
    int32_t b;
    RS_TRY(param(w, bonus[damagetype], &b));
    int64_t stat = n->levels[damagetype == RS_MAGIC_STYLE ? RS_NPC_MAGIC : RS_NPC_DEFENCE];
    *out = (stat + 9) * ((int64_t)b + 64);
    return 0;
}

/* give_combat_experience (engine.py:2469-2483) for the melee styles. */
static int give_combat_xp(World *w, int32_t style, int32_t damage, int32_t mult, const Npc *n) {
    int64_t base = (int64_t)damage * 10;
#define XP(pct) ((int32_t)rs_floordiv(rs_floordiv(base * (pct), 100) * mult, 1000))
#define ADD(stat, pct) rs_add_xp(w, stat, XP(pct), RS_XPC_HIT, n->nid, n->life)   /* the hit is the cause */
    if (style == 0) RS_TRY(ADD(RS_SKILL_ATTACK, 400));
    else if (style == 1) RS_TRY(ADD(RS_SKILL_STRENGTH, 400));
    else if (style == 2) RS_TRY(ADD(RS_SKILL_DEFENCE, 400));
    else if (style == 3) {
        RS_TRY(ADD(RS_SKILL_ATTACK, 133));
        RS_TRY(ADD(RS_SKILL_STRENGTH, 133));
        RS_TRY(ADD(RS_SKILL_DEFENCE, 133));
    } else {
        return RS_FAIL(&w->err, RS_E_COMBAT_XP_STYLE, style);
    }
    return ADD(RS_SKILL_HITPOINTS, 133);
#undef ADD
#undef XP
}

/* npc_retaliate (engine.py). */
static int npc_retaliate(World *w, Npc *n, int32_t delay) {
    RS_TRY(rs_npc_enqueue(w, n, 1, 0, delay, 0));
    n->v_aggressive_player = 1;
    n->v_lastcombat = imax(n->v_lastcombat, w->tick);
    return 0;
}

/* [label,player_melee_attack] (engine.py). */
static int melee_attack(World *w, Npc *n) {
    Player *p = &w->p;
    RsRef ref = rs_ref_npc(w, (int)(n - w->npcs));
    int ok;
    rs_stop_action(w);
    RS_TRY(attackable(w, n, &ok));
    if (!ok) return 0;
    if (p->varp[RS_VARP_action_delay] > w->tick) return rs_p_opnpc(w, ref, 2);
    if (n->levels[RS_NPC_HP] == 0) {
        rs_stop_action(w);
        return 0;
    }
    int32_t damage = 0;
    int att, def;
    RS_TRY(roll_varps(w, p->varp[RS_VARP_damagetype], &att, &def));
    int64_t defence;
    RS_TRY(npc_defence_roll(w, n, p->varp[RS_VARP_damagetype], &defence));
    int64_t a = rs_randominc(w, p->varp[att], RS_KEYS(n->nid, 1)), d = rs_randominc(w, defence, RS_KEYS(n->nid, 2));
    if (a > d) {                                    /* player_npc_hit_roll */
        int32_t max_dealt;
        RS_TRY(param(w, ntype(w, n)->max_dealt, &max_dealt));
        damage = (int32_t)rs_randominc(w, imin(p->varp[RS_VARP_com_maxhit], max_dealt), RS_KEYS(n->nid, 3));
        int32_t capped = imin(damage, n->levels[RS_NPC_HP]);
        /* the hit as its xp's cause (sim/audit.py XpCheck): the damage queued below lands later */
        RsEvent hit = {.tick = w->tick, .kind = RS_EV_HIT, .a8 = 1, .obj = n->nid, .count = n->type, .v0 = n->life,
                       .v1 = damage, .v2 = n->levels[RS_NPC_HP], .v3 = p->varp[RS_VARP_damagestyle]};
        RS_TRY(rs_event(w, &hit));
        RS_TRY(give_combat_xp(w, p->varp[RS_VARP_damagestyle], capped, n->v_xp_mult, n));
        if (capped >= 1) n->has_hero = 1, n->hero_damage += capped;
    }
    int32_t weapon = rhand(w), v;
    if (weapon >= 0 && damage > 0) {
        RS_TRY(param(w, otype(w, weapon)->poison_severity, &v));
        if (v > 0) return RS_FAIL0(&w->err, RS_E_POISONED_WEAPON);
    }
    RS_TRY(npc_retaliate(w, n, 0));
    RS_TRY(rs_npc_enqueue(w, n, 2, damage, 0, 1));
    if (weapon >= 0) RS_TRY(param(w, otype(w, weapon)->attackrate, &v));
    p->varp[RS_VARP_action_delay] = w->tick + (weapon < 0 ? 4 : v);
    return rs_p_opnpc(w, ref, 2);
}

/* [label,player_combat_start] (engine.py). */
static int combat_start(World *w, Npc *n) {
    int ok;
    RS_TRY(in_combat_check(w, n, &ok));
    if (!ok) return 0;
    if (w->p.varp[RS_VARP_damagetype] == RS_RANGED_STYLE) return RS_FAIL0(&w->err, RS_E_RANGED_ATTACK);
    return melee_attack(w, n);
}

/* [label,player_combat_start_ap] (engine.py): in melee, wait for melee distance (p_aprange). */
static int combat_start_ap(World *w, Npc *n) {
    Player *p = &w->p;
    int32_t range;
    int ok = 1;
    RS_TRY(attackrange(w, &range));
    if (range <= 1) RS_TRY(in_combat_check(w, n, &ok));
    if (!ok || imax(iabs(p->x - n->x), iabs(p->z - n->z)) > range) {   /* _distance_to(npc, p) */
        p->ap_range = range, p->ap_range_called = 1;
        return 0;
    }
    return combat_start(w, n);
}

/* [queue,combat_damage_player] (engine.py). */
static int combat_damage_player(World *w, int32_t damage, int32_t source) {
    Player *p = &w->p;
    if (p->levels[RS_SKILL_HITPOINTS] == 0) return 0;
    rs_close_modal(w, 1);
    RS_TRY(rs_player_damage(w, damage, source));
    if (p->levels[RS_SKILL_HITPOINTS] == 0) RS_TRY(rs_enqueue(w, RS_Q_NORMAL, RS_S_PLAYER_DEATH, 0, 0, 0, 0));
    return 0;
}

/* [queue,player_death] (engine.py): %death, a tick, stop and close, three ticks, then the unported rest. */
static int player_death(World *w, Frame *f, RsSusp *out) {
    switch (f->pc[0]) {
    case 0:
        w->p.varp[RS_VARP_death] = 1;
        f->pc[0] = 1;
        out->kind = RS_DELAY, out->n = 1;
        return 0;
    case 1:
        rs_stop_action(w);
        rs_close_modal(w, 1);
        f->pc[0] = 2;
        out->kind = RS_DELAY, out->n = 3;
        return 0;
    default:
        w->p.varp[RS_VARP_death] = 0;
        return RS_FAIL0(&w->err, RS_E_PLAYER_DEATH);
    }
}

/* [queue,playerhit_n_retaliate] (engine.py): auto-retaliate when otherwise idle; the flinch is half the worn
   weapon's attackrate, 4 unarmed. */
static int playerhit_n_retaliate(World *w, int32_t uid) {
    Player *p = &w->p;
    Npc *n = rs_npc_found(w, uid);
    if (p->varp[RS_VARP_option_nodef] != 0 || !n || rs_player_busy2(w)) return 0;
    if (p->varp[RS_VARP_action_delay] < w->tick) {
        int32_t weapon = rhand(w), rate = 4;
        if (weapon >= 0) RS_TRY(param(w, otype(w, weapon)->attackrate, &rate));
        p->varp[RS_VARP_action_delay] = w->tick + rate / 2;
    }
    return rs_p_opnpc(w, rs_ref_npc(w, (int)(n - w->npcs)), 2);
}

/* The npc an op script names (its ref[0], engine.py run_trigger's target). */
static int target_npc(World *w, const Frame *f, Npc **out) {
    if (f->ref[0].kind != RS_REF_NPC || f->ref[0].slot >= w->n_npcs)
        return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a combat script without its npc");
    *out = &w->npcs[f->ref[0].slot];
    return 0;
}

int rs_combat_script_step(World *w, Frame *f, RsSusp *out) {
    Npc *n;
    out->kind = RS_DONE;
    switch (f->script[0]) {
    case RS_S_COMBAT_START_AP: RS_TRY(target_npc(w, f, &n)); return combat_start_ap(w, n);
    case RS_S_COMBAT_START: RS_TRY(target_npc(w, f, &n)); return combat_start(w, n);
    case RS_S_MELEE_ATTACK: RS_TRY(target_npc(w, f, &n)); return melee_attack(w, n);
    case RS_S_COMBAT_DAMAGE_PLAYER: return combat_damage_player(w, f->arg[0][0], f->arg[0][1]);
    case RS_S_PLAYER_DEATH: return player_death(w, f, out);
    case RS_S_PLAYERHIT_N_RETALIATE: return playerhit_n_retaliate(w, f->arg[0][0]);
    default: return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "rs_combat_script_step: not a player combat script");
    }
}

/* ---- the npc's side (engine.py) ------------------------------------------------------------------------ */
/* npc_check_notcombat (engine.py). npc_check_notcombat_self (2544-2548) always holds here: its refusal
   needs an aggressor other than None and the player. */
static int check_notcombat(const World *w, const Npc *n) {
    const Player *p = &w->p;
    if (p->varp[RS_VARP_lastcombat_pvp] + 8 > w->tick) return 0;
    if (p->varp[RS_VARP_lastcombat] + 8 > w->tick && p->varp[RS_VARP_aggressive_npc] != -1
        && p->varp[RS_VARP_aggressive_npc] != uid_of(n))
        return 0;
    return 1;
}

/* [proc,playerhit_n_melee] (engine.py): close the player's interface, npc_set_attack_vars, the attack delay,
   then (the npc alive) the damage and the auto-retaliate queued. */
static int playerhit_n_melee(World *w, Npc *n, int32_t damage, int32_t delay) {
    Player *p = &w->p;
    rs_close_modal(w, 1);
    p->varp[RS_VARP_lastcombat] = w->tick;          /* npc_set_attack_vars (2551-2554) */
    n->v_attacking = 1;
    p->varp[RS_VARP_aggressive_npc] = uid_of(n);
    n->v_action_delay = w->tick + delay;
    if (n->levels[RS_NPC_HP] == 0) return 0;
    RS_TRY(rs_enqueue(w, RS_Q_NORMAL, RS_S_COMBAT_DAMAGE_PLAYER, damage, n->nid, 0, 0));
    return rs_enqueue(w, RS_Q_NORMAL, RS_S_PLAYERHIT_N_RETALIATE, uid_of(n), 0, 0, 0);
}

/* [ai_opplayer2,_] -> npc_default_attack (engine.py) against the player. */
static int npc_default_attack(World *w, Npc *n) {
    const Player *p = &w->p;
    const RsNpcType *nt = ntype(w, n);
    if (p->varp[RS_VARP_death] == 1) {
        rs_npc_set_mode(w, n, RS_MODE_NULL, 0);
        return 0;
    }
    if (n->v_action_delay > w->tick) return 0;
    if (!check_notcombat(w, n)) {
        rs_npc_set_mode(w, n, RS_MODE_NULL, 0);
        return 0;
    }
    int32_t attackbonus, damagetype, strengthbonus, attackrate, poison;
    RS_TRY(param(w, nt->attackbonus, &attackbonus));
    int64_t attack_roll = ((int64_t)n->levels[RS_NPC_ATTACK] + 9) * ((int64_t)attackbonus + 64);
    RS_TRY(param(w, nt->damagetype, &damagetype));
    int att, def;
    RS_TRY(roll_varps(w, damagetype, &att, &def));
    RS_TRY(param(w, nt->strengthbonus, &strengthbonus));
    int64_t maxhit = rs_floordiv(((int64_t)n->levels[RS_NPC_STRENGTH] + 9) * ((int64_t)strengthbonus + 64) + 320, 640);
    int64_t a = rs_randominc(w, attack_roll, RS_KEYS(n->nid, 11));
    int64_t d = rs_randominc(w, p->varp[def], RS_KEYS(n->nid, 12));
    if (a > d) {
        int32_t damage = (int32_t)rs_randominc(w, maxhit, RS_KEYS(n->nid, 13));
        RS_TRY(param(w, nt->attackrate, &attackrate));
        RS_TRY(playerhit_n_melee(w, n, damage, attackrate));
        RS_TRY(param(w, nt->poison_severity, &poison));   /* npc_poison_player */
        if (poison > 0) return RS_FAIL0(&w->err, RS_E_POISON);
        return 0;
    }
    RS_TRY(param(w, nt->attackrate, &attackrate));
    return playerhit_n_melee(w, n, 0, attackrate);
}

/* [ai_queue1,_] -> npc_default_retaliate (engine.py): opplayer2, or applayer2 for the _ap variant; escape
   at or under the retreat hitpoints, or out of range. */
static int npc_default_retaliate(World *w, Npc *n, int ap) {
    const RsNpcType *nt = ntype(w, n);
    if (!w->has_player || !n->v_aggressive_player) return 0;   /* finduid */
    int32_t retreat, attackrate;
    RS_TRY(param(w, nt->retreat, &retreat));
    if (n->levels[RS_NPC_HP] <= retreat) rs_npc_set_mode(w, n, RS_MODE_PLAYERESCAPE, 1);
    int ready = n->v_action_delay + 8 < w->tick || (n->mode != RS_MODE_APPLAYER2 && n->mode != RS_MODE_OPPLAYER2);
    if (!ready) return 0;
    RS_TRY(param(w, nt->attackrate, &attackrate));
    n->v_action_delay = w->tick + (int32_t)rs_floordiv(attackrate, 2);
    n->v_attacking = n->v_aggressive_player;
    if (n->levels[RS_NPC_HP] > retreat) {
        rs_npc_set_mode(w, n, ap ? RS_MODE_APPLAYER2 : RS_MODE_OPPLAYER2, 1);
        if (!rs_npc_validate_target(w, n)) rs_npc_set_mode(w, n, RS_MODE_PLAYERESCAPE, 1);   /* npc_inrange */
    }
    return 0;
}

/* [ai_queue2,_] -> npc_default_damage (engine.py): a hit (or block) with its evidence, the death queue at 0. */
static int npc_default_damage(World *w, Npc *n, int32_t damage, int32_t source) {
    if (n->levels[RS_NPC_HP] == 0) return 0;
    int32_t hp = n->levels[RS_NPC_HP];
    RsEvent e = {.tick = w->tick, .kind = RS_EV_NPC_DAMAGE, .a8 = (uint8_t)(source != 0), .obj = n->nid,
                 .count = n->type, .v0 = n->life, .v1 = damage, .v2 = hp, .v3 = ntype(w, n)->stats[RS_NPC_HP]};
    RS_TRY(rs_event(w, &e));
    n->levels[RS_NPC_HP] = (int16_t)imax(0, hp - damage);
    if (n->levels[RS_NPC_HP] > 0) return 0;
    RsEvent d = {.tick = w->tick, .kind = RS_EV_NPC_DEFEATED, .a8 = (uint8_t)(source != 0), .obj = n->nid,
                 .count = n->type, .v0 = n->life};
    RS_TRY(rs_event(w, &d));
    return rs_npc_enqueue(w, n, 3, 0, 0, 0);
}

/* [proc,npc_death] (engine.py), the death program's sub-frame (Frame level 1, pc[1]): stop, wait for the
   arrival, clear the player's %lastcombat, a tick, then npc_del. */
static int npc_death(World *w, Npc *n, Frame *f, RsSusp *out) {
    switch (f->pc[1]) {
    case 0:
        n->has_path = 1, n->path_x = n->x, n->path_z = n->z;   /* npc_walk(npc_coord) */
        rs_npc_set_mode(w, n, RS_MODE_NONE, 0);
        if (n->last_movement >= w->tick - 1) {                  /* npc_arrivedelay */
            f->pc[1] = 1;
            out->kind = RS_NPC_ARRIVEDELAY, out->n = n->last_movement == w->tick - 1 ? w->tick + 1 : w->tick + 2;
            return 0;
        }
        /* fall through */
    case 1:
        if (w->has_player && n->v_aggressive_player) w->p.varp[RS_VARP_lastcombat] = -1;
        f->pc[1] = 2;
        out->kind = RS_NPC_DELAY, out->n = 1;
        return 0;
    case 2:
        rs_remove_npc(w, n);
        out->kind = RS_DONE;
        return 0;
    default:
        return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "npc_death resumed at an unknown pc");
    }
}

/* ---- the death drop programs (rs.h DOP_*; DeathContext, engine.py) ------------------------------------ */
enum { DROP_PC = 0, DROP_DRAWS = 1, DROP_HERO = 2, DROP_REG0 = 4 };   /* Frame.local of a death program (rs.h) */

/* The operands of a program's ops are in range (registers, varps, inventories, jump targets), checked once at the
   program's start: a bad pack is RS_E_PACK, never an out-of-bounds read. */
static int drop_check(World *w, const RsDropRoot *root) {
    const RsPack *P = w->P;
    if (root->first < 0 || root->n_ops < 1 || root->first > P->n_drop_ops - root->n_ops)
        return RS_FAIL_TEXT(&w->err, RS_E_PACK, "a drop root outside DROP_OPS");
#define REG(v) ((uint32_t)(v) < RS_DROP_REGS)
    for (int32_t k = 0; k < root->n_ops; k++) {
        const RsDropOp *o = &P->drop_ops[root->first + k];
        int ok;
        switch (o->op) {
        case DOP_END: case DOP_UNSUPPORTED: case DOP_NPC_DEATH: case DOP_IF_CLOSE: ok = 1; break;
        case DOP_CONST: case DOP_FINDHERO: case DOP_MEMBERS: case DOP_P_FINDUID: case DOP_ENUM_COUNT: ok = REG(o->a); break;
        case DOP_MOVE: case DOP_RANDOM: case DOP_RANDOMINC: ok = REG(o->a) && REG(o->b); break;
        case DOP_ARITH: ok = REG(o->a) && REG(o->b) && REG(o->d) && (uint32_t)o->c < 4; break;
        case DOP_AND_BITS: case DOP_OR_BITS: case DOP_OBJ_ADD: ok = REG(o->a) && REG(o->b) && REG(o->c); break;
        case DOP_JUMP: ok = (uint32_t)o->a < (uint32_t)root->n_ops; break;
        case DOP_JUMP_IF:
            ok = (uint32_t)o->a < (uint32_t)root->n_ops && REG(o->b) && REG(o->d) && (uint32_t)o->c < 6;
            break;
        case DOP_JUMP_ZERO: ok = (uint32_t)o->a < (uint32_t)root->n_ops && REG(o->b); break;
        case DOP_INV_TOTAL: ok = REG(o->a) && (uint32_t)o->b < 3 && REG(o->c); break;
        case DOP_INV_TOTALCAT: ok = REG(o->a) && (uint32_t)o->b < 3; break;
        case DOP_ENUM: ok = REG(o->a) && REG(o->c); break;
        case DOP_VARP: ok = REG(o->a) && (uint32_t)o->b < RS_N_VARPS; break;
        case DOP_SET_VARP: ok = (uint32_t)o->a < RS_N_VARPS && REG(o->b); break;
        case DOP_CLEARBIT_RANGE: ok = REG(o->a) && REG(o->b) && REG(o->c) && REG(o->d); break;
        default: ok = 0;
        }
        if (!ok) return RS_FAIL_TEXT(&w->err, RS_E_PACK, "a drop op with an operand out of range");
    }
#undef REG
    return 0;
}

static int put_reg(World *w, int32_t *r, int32_t at, int64_t value) {
    if (value < INT32_MIN || value > INT32_MAX) return RS_OVERFLOW(&w->err, "a drop register past int32");
    r[at] = (int32_t)value;
    return 0;
}

static Slot *inventory(World *w, int32_t which, int *cap) {
    *cap = which == 0 ? RS_INV_CAP : which == 1 ? RS_WORN_CAP : RS_BANK_CAP;
    return which == 0 ? w->p.inv : which == 1 ? w->p.worn : w->p.bank;
}

/* npc_death_script (engine.py): rs2.run of the type's [ai_queue3] root, compiled. Frame level 0 is
   (RS_S_NPC_DEATH_SCRIPT, pc 1 while suspended in npc_death); its program position, draw count, hero flag and
   registers live in local[] (rs.h SCRIPT FRAMES). */
static int death_script(World *w, Npc *n, Frame *f, RsSusp *out) {
    const RsPack *P = w->P;
    const RsNpcType *nt = ntype(w, n);
    int32_t row = f->arg[0][0];
    if (row < 0 || row >= P->n_drop_roots) return RS_FAIL_TEXT(&w->err, RS_E_PACK, "a death script of no drop root");
    const RsDropRoot *root = &P->drop_roots[row];
    int32_t *r = &f->local[DROP_REG0];
#define HERO() do { if (!f->local[DROP_HERO]) \
        return RS_FAIL(&w->err, RS_E_DEATH_SCRIPT, nt->debugname, P->consts->s_death_no_hero); } while (0)
    if (f->pc[0] == 1) {                       /* resumed inside npc_death */
        RS_TRY(npc_death(w, n, f, out));
        if (out->kind != RS_DONE) return 0;
        f->depth = 1, f->script[1] = 0, f->pc[1] = 0, f->pc[0] = 0;
    } else if (f->local[DROP_PC] == 0) {
        RS_TRY(drop_check(w, root));
    }
    for (;;) {
        const RsDropOp *o = &P->drop_ops[root->first + f->local[DROP_PC]];
        f->local[DROP_PC] += 1;
        switch (o->op) {
        case DOP_END:
            out->kind = RS_DONE;
            return 0;
        case DOP_UNSUPPORTED:
            return RS_FAIL(&w->err, RS_E_DEATH_SCRIPT, nt->debugname, o->a);
        case DOP_NPC_DEATH:
            f->depth = 2, f->script[1] = RS_S_NPC_DEATH, f->pc[1] = 0;
            memset(f->arg[1], 0, sizeof f->arg[1]);
            f->ref[1] = (RsRef){0};
            RS_TRY(npc_death(w, n, f, out));
            if (out->kind != RS_DONE) {
                f->pc[0] = 1;
                return 0;
            }
            f->depth = 1, f->script[1] = 0, f->pc[1] = 0;
            break;
        case DOP_CONST: r[o->a] = o->b; break;
        case DOP_MOVE: r[o->a] = r[o->b]; break;
        case DOP_ARITH: {
            int64_t x = r[o->b], y = r[o->d];
            if (y == 0) return RS_FAIL0(&w->err, RS_E_ZERO_DIVISION);   /* every operator divides (rs2.py:355) */
            int64_t v = o->c == 0 ? x + y : o->c == 1 ? x - y : o->c == 2 ? x * y : (int64_t)((double)x / (double)y);
            RS_TRY(put_reg(w, r, o->a, v));
            break;
        }
        case DOP_AND_BITS: r[o->a] = r[o->b] & r[o->c]; break;
        case DOP_OR_BITS: r[o->a] = r[o->b] | r[o->c]; break;
        case DOP_JUMP: f->local[DROP_PC] = o->a; break;
        case DOP_JUMP_IF: {
            int32_t x = r[o->b], y = r[o->d];
            int holds = o->c == 0 ? x == y : o->c == 1 ? x != y : o->c == 2 ? x < y : o->c == 3 ? x > y
                      : o->c == 4 ? x <= y : x >= y;
            if (!holds) f->local[DROP_PC] = o->a;
            break;
        }
        case DOP_JUMP_ZERO: if (r[o->b] == 0) f->local[DROP_PC] = o->a; break;
        case DOP_FINDHERO:   /* max(heroes) by value: the player's key or none (rs.h Npc) */
            f->local[DROP_HERO] = w->has_player && n->has_hero;
            r[o->a] = f->local[DROP_HERO];
            break;
        case DOP_MEMBERS: r[o->a] = w->members; break;
        case DOP_RANDOM: case DOP_RANDOMINC: {
            f->local[DROP_DRAWS] += 1;
            int64_t bound = (int64_t)r[o->b] + (o->op == DOP_RANDOMINC);
            uint32_t m = rs_wm(w, RS_KEYS(RS_R_NPC_SCRIPT, w->tick, n->nid, 1000 + f->local[DROP_DRAWS]));
            RS_TRY(put_reg(w, r, o->a, rs_pick(m, bound)));
            break;
        }
        case DOP_OBJ_ADD: {
            int32_t obj = r[o->a];
            const RsObjType *ot = NULL;
            if (w->excl & (RS_EXCLUDE_CLUE_DROPS | RS_EXCLUDE_LOOT_DROPS)) {
                ot = rs_obj_type(P, obj);
                if (!ot) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "an excluded drop of no obj type (KeyError)");
            }
            if ((w->excl & RS_EXCLUDE_CLUE_DROPS) && (ot->flags & RS_OF_TRAIL)) {
                RsEvent e = {.tick = w->tick, .kind = RS_EV_EXCLUDED, .a8 = RS_EXCL_CLUE, .obj = obj, .v0 = n->nid};
                RS_TRY(rs_event(w, &e));
                break;
            }
            if ((w->excl & RS_EXCLUDE_LOOT_DROPS) && (ot->flags & w->loot)) {  /* loot_outside(w.t, obj, w.loot_outside) */
                RsEvent e = {.tick = w->tick, .kind = RS_EV_EXCLUDED, .a8 = RS_EXCL_LOOT, .obj = obj, .v0 = n->nid};
                RS_TRY(rs_event(w, &e));
                break;
            }
            HERO();
            RS_TRY(rs_obj_add(w, n->level, n->x, n->z, obj, r[o->b], r[o->c]));
            break;
        }
        case DOP_INV_TOTAL: {
            HERO();
            int cap;
            Slot *s = inventory(w, o->b, &cap);
            RS_TRY(put_reg(w, r, o->a, rs_inv_total(s, cap, r[o->c])));
            break;
        }
        case DOP_INV_TOTALCAT: {
            HERO();   /* player() is called whether or not the category exists */
            int cap;
            Slot *s = inventory(w, o->b, &cap);
            int64_t total = 0;
            for (int i = 0; o->c != RS_NONE && i < cap; i++)
                if (s[i].id >= 0 && otype(w, s[i].id)->category == o->c) total += s[i].count;
            RS_TRY(put_reg(w, r, o->a, total));
            break;
        }
        case DOP_IF_CLOSE: HERO(); rs_close_modal(w, 1); break;
        case DOP_P_FINDUID: HERO(); r[o->a] = rs_player_can_access(w); break;
        case DOP_ENUM: {
            int32_t value = o->d;
            for (int i = 0; i < P->n_enums; i++)
                if (P->enums[i].table == o->b && P->enums[i].key == r[o->c]) { value = P->enums[i].value; break; }
            r[o->a] = value;
            break;
        }
        case DOP_ENUM_COUNT: {
            int32_t count = 0;
            for (int i = 0; i < P->n_enums; i++) count += P->enums[i].table == o->b;
            r[o->a] = count;
            break;
        }
        case DOP_VARP: HERO(); r[o->a] = w->p.varp[o->b]; break;
        case DOP_SET_VARP: HERO(); w->p.varp[o->a] = r[o->b]; break;
        case DOP_CLEARBIT_RANGE: {   /* value & ~(((1 << (high - low + 1)) - 1) << low) */
            int64_t value = r[o->b], low = r[o->c], width = (int64_t)r[o->d] - r[o->c] + 1;
            if (low < 0 || width < 0)
                return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "clearbit_range with a negative shift (ValueError)");
            if (low + width > 62) return RS_OVERFLOW(&w->err, "clearbit_range past 62 bits");
            RS_TRY(put_reg(w, r, o->a, value & ~((((int64_t)1 << width) - 1) << low)));
            break;
        }
        default:
            return RS_FAIL_TEXT(&w->err, RS_E_PACK, "an unknown drop op");
        }
    }
#undef HERO
}

int rs_npc_combat_step(World *w, Npc *n, Frame *f, RsSusp *out) {
    out->kind = RS_DONE;
    switch (f->script[0]) {
    case RS_S_NPC_DEFAULT_ATTACK: return npc_default_attack(w, n);
    case RS_S_NPC_DEFAULT_RETALIATE: return npc_default_retaliate(w, n, 0);
    case RS_S_NPC_DEFAULT_RETALIATE_AP: return npc_default_retaliate(w, n, 1);
    case RS_S_NPC_DEFAULT_DAMAGE: return npc_default_damage(w, n, f->arg[0][0], f->arg[0][1]);
    case RS_S_NPC_DEATH_SCRIPT: return death_script(w, n, f, out);
    case RS_S_HIGHWAYMAN_APPROACH:
        /* [ai_applayer2,highwayman] (engine.py highwayman_approach; highwayman.rs2): npc_say, which nothing the slice or
           the interface reads and the oracle leaves out, then npc_setmode(opplayer2) with the npc's target as the
           script's player (fn(world, npc, npc.target)); it ends without a suspension */
        rs_npc_set_mode(w, n, RS_MODE_OPPLAYER2, n->target_player);
        return 0;
    default: return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "rs_npc_combat_step: not an npc script");
    }
}

/* ---- ground items (engine.py) ---------------------------------------------------------------------- */
/* OBJ_ADD for the player: a members obj needs a members world; one obj per unstackable item; a stack joins the
   player's stack of that obj on the tile while it stays within STACK_LIMIT (its despawn timer restarts). */
int rs_obj_add(World *w, int32_t level, int32_t x, int32_t z, int32_t obj, int32_t count, int32_t duration) {
    if (obj == -1 || count == -1) return 0;
    const RsObjType *ot = rs_obj_type(w->P, obj);
    if (!ot) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "obj_add of no obj type (KeyError)");
    if (ot->dummyitem) return RS_FAIL(&w->err, RS_E_DUMMY_OBJ, ot->debugname);
    if (ot->members && !w->members) return 0;
    int stackable = ot->stackable != 0;
    int32_t n = !stackable ? count : 1, amount = !stackable || count == 1 ? 1 : count;   /* [1] * count or [count] */
    for (int32_t k = 0; k < n; k++) {
        if (stackable) {
            Obj *mine = NULL;
            for (int i = 0; i < w->n_objs && !mine; i++) {
                Obj *o = &w->objs[w->obj_order[i]];
                if (!o->is_static && o->active && o->type == obj && o->level == level && o->x == x && o->z == z
                    && o->receiver_player)
                    mine = o;
            }
            if (mine && (int64_t)mine->count + amount <= RS_STACK_LIMIT) {
                mine->count += amount;
                mine->ticks = duration;
                continue;
            }
        }
        int slot;
        RS_TRY(rs_obj_append(w, obj, amount, x, z, level, 0, duration, 1, &slot));
    }
    return 0;
}

/* OBJ_TAKEITEM into the inventory: a spawn respawns after its type's respawnrate, a drop leaves the list. */
int rs_take_obj(World *w, RsRef ref) {
    Obj *o = rs_get_obj(w, ref);
    if (!o || !o->active || o->count < 1) return 0;
    int64_t added = rs_inv_add(w->P, w->p.inv, RS_INV_CAP, 0, o->type, o->count);
    RsEvent e = {.tick = w->tick, .kind = RS_EV_PICKUP, .obj = o->type, .count = (int32_t)added};
    RS_TRY(rs_event(w, &e));
    o->active = 0;
    if (o->is_static) o->ticks = (int32_t)rs_floordiv((int64_t)(4000 - 1) * otype(w, o->type)->respawnrate, 4000);
    else rs_obj_remove(w, ref.slot);
    return 0;
}
