"""An independent account of a simulated episode's goal outcomes, from the world's events.

The payment in sim/task.py follows held items or kills as they happen; this replays the evidence
instead: the holdings at arm, every production, pickup, transfer and removal (eating, burying,
dropping) event, and the holdings at close. Each location's totals must be explained exactly by
the events, transfers must conserve items, and only the gathering scripts may produce
(baseline.native_value.GATHERING_SCRIPTS). The return counts goal items held at close that
production accounts for, and nothing after a death, as baseline.resource_task.goal_return does
for native reports.

Kills (goal-activity-v2 melee, sustained-combat-v1) are counted from the hits npcs take: one credit per enemy life of
an eligible type (baseline.resource_task.MELEE_ENEMIES, or the task's `enemies`), when this agent's hit was the first that
life took in the episode and found it at full hitpoints, every hit on it was this agent's, and this
agent's hit defeated it (hitpoints 0) before the close.

XP (free-play-v1): `XpCheck` is the independent account of each XP gain,
from its cause in the event log and the native tables at the world's XP rate; `check_stats` is the bookkeeping check from
start to end; `fractional_level` is the paid quantity's level, from the engine's level table.
"""
from collections import Counter

from baseline.native_value import GATHERING_SCRIPTS
from baseline.resource_task import GOAL_FAMILIES, MELEE_ENEMIES
from sim.engine import (ATTACK, COOKING, DEFENCE, FIREMAKING, FISHING, HITPOINTS, MINING, PRAYER, STRENGTH, WOODCUTTING,
                        level_for, xp_for)


class AuditError(Exception):
    pass


def _totals(holdings):
    totals = {loc: Counter() for loc in ('inventory', 'bank', 'worn')}
    for loc, rows in holdings.items():
        for _, obj, count in rows:
            totals[loc][obj] += count
    return totals


def kill_ticks(events, agent, enemies=MELEE_ENEMIES):
    """The ticks of the certified kills by `agent` among (tick, kind, data) events (module docstring): each
    credited life's defeat, in order."""
    return sorted(kill_lives(events, agent, enemies).values())


def kill_lives(events, agent, enemies=MELEE_ENEMIES):
    """{(npc, life): defeat tick} of the certified kills by `agent` (module docstring)."""
    lives = {}
    for tick, kind, data in events:
        if kind not in ('npc_damage', 'npc_defeated'):
            continue
        key = (data['npc'], data['life'])
        if kind == 'npc_damage':
            life = lives.setdefault(key, dict(type=data['type'], fresh=data['source'] == agent and data['hp'] == data['full'],
                                              mine=True, defeated=None))
            life['mine'] &= data['source'] == agent
        else:
            if key not in lives:
                raise AuditError('an npc defeated without a hit')
            lives[key]['defeated'] = tick if data['source'] == agent else None
    return {key: life['defeated'] for key, life in lives.items()
            if life['type'] in enemies and life['fresh'] and life['mine'] and life['defeated'] is not None}


def kills(events, agent, enemies=MELEE_ENEMIES):
    """Certified kills by `agent` among (tick, kind, data) events (module docstring)."""
    return len(kill_ticks(events, agent, enemies))


def audit(events, initial, terminal, goal, tables, agent=None, enemies=MELEE_ENEMIES):
    family = set(GOAL_FAMILIES.get(goal, ()))
    for base in list(family):
        note = tables.obj_types.get(base, {}).get('certlink', -1)
        if note is not None and note >= 0:
            family.add(note)
    held, produced, external = _totals(initial), Counter(), Counter()
    names = {'inv': 'inventory', 'bank': 'bank', 'worn': 'worn'}
    window = []
    for tick, kind, data in events:
        if tick >= terminal['tick']:
            break
        window.append((tick, kind, data))
        if kind == 'produce':
            if data['script'] not in GATHERING_SCRIPTS:
                raise AuditError(f"production by {data['script']}")
            held['inventory'][data['obj']] += data['count']
            produced[data['obj']] += data['count']
        elif kind == 'pickup':  # a ground item: held, but not produced
            held['inventory'][data['obj']] += data['count']
            external[data['obj']] += data['count']
        elif kind == 'transfer':
            src, dst = names[data['src']], names[data['dst']]
            if held[src][data['obj']] < data['count']:
                raise AuditError('transfer of items not held')
            held[src][data['obj']] -= data['count']
            held[dst][data['to']] += data['count']
        elif kind == 'remove':  # eaten, buried or dropped from the inventory
            if held['inventory'][data['obj']] < data['count']:
                raise AuditError('removal of items not held')
            held['inventory'][data['obj']] -= data['count']
        elif kind not in ('npc_damage', 'npc_defeated', 'agent_hp', 'excluded', 'hit', 'xp', 'npc_respawn'):
            raise AuditError(f'unaccounted event {kind}')
    final = _totals(terminal['holdings'])
    for loc in held:
        if +held[loc] != +final[loc]:
            raise AuditError(f'{loc} holdings not explained by the events: {dict(+held[loc])} != {dict(+final[loc])}')
    kept = sum(final[loc][obj] for loc in ('inventory', 'bank') for obj in family)
    made = sum(produced[obj] for obj in family)
    picked = sum(external[obj] for obj in family)
    # as native_trace's bounds: picked-up goal items may be among those kept
    lower, upper = max(0, kept - picked), min(kept, made)
    if lower > upper:
        raise AuditError('goal items held that production does not account for')
    killed = kills(window, agent, enemies)
    return dict(survived=terminal['reason'] != 'death', reason=terminal['reason'], produced=made, retained=lower,
                retained_upper=upper, picked=picked, produced_items=dict(produced), tick=terminal['tick'],
                kills=dict(lower=killed, upper=killed))


# ---- XP and levels (free-play-v1) -----------------------------------------------------------------------------------
# The skill a production script's XP trains, and the XP a hit gives each stat by damage style, in percent of ten times
# the damage ([proc,give_combat_experience]): the style's own stat 400 (attack 0, strength 1, defence 2), each of the
# three 133 for the shared style 3, hitpoints 133 for every style.
PRODUCE_SKILL = {'[label,get_logs]': WOODCUTTING, '[label,get_ore_normal]': MINING, '[label,get_ore_fast]': MINING,
                 '[proc,fish_roll]': FISHING, '[label,cook_item]': COOKING}
HIT_PERCENT = {0: {ATTACK: 400}, 1: {STRENGTH: 400}, 2: {DEFENCE: 400}, 3: {ATTACK: 133, STRENGTH: 133, DEFENCE: 133}}
XP_CAP = 2_000_000_000  # Player.addXp's ceiling, in tenths


def fractional_level(xp):
    """l(x) = L + (x - X(L)) / (X(L+1) - X(L)) at XP x (tenths, as Player.stats), L its level by the engine's table
    (sim/engine.py LEVEL_XP), capped at 99."""
    level = level_for(xp)
    if level >= 99:
        return 99.0
    low = xp_for(level)
    return level + (xp - low) / (xp_for(level + 1) - low)


def _param(tables, kind, type_, name):
    rows = tables.obj_types if kind == 'obj' else tables.npc_types
    params = rows[type_]['params']
    return params[name] if name in params else tables.meta['param_defaults'][name]


class XpCheck:
    """The independent account of a life's XP: every 'xp' event (sim/engine.py World.add_xp)
    must have its cause in the event log and the amount the native tables give that cause at the world's XP rate.

    - 'produce' (woodcutting, mining, fishing, cooking): a 'produce' event of the same tick, obj and script, each used
      once, whose script trains the stat (PRODUCE_SKILL); the amount is the table row's: the tree's (content trees, by
      `src`, the loc type) or the rock's (compact rocks) whose product is the obj, the fish's, the raw item's cooking row
      (by `src`, the raw obj) whose cooked item it is.
    - 'hit' (attack, strength, defence, hitpoints): a 'hit' event of the same tick, npc and life by this agent, each stat
      of its style used once; the amount is (min(damage, hp) x 10 x percent // 100) x multiplier // 1000, with the npc
      type's combat_xp_multiplier. A hit on an npc life in the tick of that life's 'npc_respawn' event (the cycle in
      which it respawned, before its [ai_spawn,_] set the multiplier) gives none: it owes no XP, and an XP event for it
      raises. The check takes that cycle from the event and its tick, never from the engine's multiplier or the
      XP given; a second 'npc_respawn' of the same npc life raises.
    - 'bury' (prayer): a 'remove' of the obj 'bury' in the same tick, used once; the obj's bone_exp.
    - 'light' (firemaking): an earlier or same-tick 'remove' of the obj 'light' not yet used; the obj's productexp.

    Each event's `before` must be the stat's running XP (from `stats`, the XP at arm) and `after` must be
    min(before + amount x rate, XP_CAP). The other side too: a cause that gives XP must have its XP in its tick (a
    production whose product's rows all give XP, OWED; each stat of a hit by this agent whose amount is above 0; bones
    buried), so XP an engine drops is caught as well; gems and burnt food give none, and a light's XP comes when the fire
    catches, so a light needs none. Any other XP event or a missing one (no cause, another amount, another stat, a
    broken chain, a cause without its XP) raises AuditError, which stops the run as incomplete; no life is dropped.
    `observe` takes events in order, a step at a time; every event of a tick must come in one call (a step ends at a
    cycle's end)."""

    def __init__(self, tables, rate, agent, stats):
        self.t, self.rate, self.agent = tables, rate, agent
        self.xp, self.gained = list(stats), [0] * len(stats)
        self.lit = []  # 'light' removals not yet used: (tick, obj)
        self.respawned = {}  # (npc, life) -> the tick of its 'npc_respawn' event
        content = tables.meta['content']

        def least(rows, product, exp):
            out = {}
            for row in rows:
                obj = tables.obj_id(row[product])
                out[obj] = min(out.get(obj, row[exp]), row[exp])
            return out
        ores = least(content.get('compact', {}).get('rocks', {}).values(), 'output', 'exp')
        # OWED: per production script, each product's least XP over its table rows (a product not listed gives none)
        self.owed = {'[label,get_logs]': least(content.get('trees', {}).values(), 'product', 'productexp'),
                     '[label,get_ore_normal]': ores, '[label,get_ore_fast]': ores,
                     '[proc,fish_roll]': {tables.obj_id(name): row['productexp'] for name, row in content.get('fish', {}).items()},
                     '[label,cook_item]': least(content.get('cooking', {}).get('rows', {}).values(), 'cooked', 'exp')}

    def _fail(self, tick, data, why):
        raise AuditError(f'xp event at tick {tick} ({data}): {why}')

    def _amount(self, tick, data, causes):
        cause, stat = data.get('cause'), data['stat']
        if cause == 'produce':
            match = next((k for k, e in enumerate(causes['produce']) if e['obj'] == data['obj']), None)
            if match is None:
                self._fail(tick, data, 'no production of that item in its tick')
            made = causes['produce'].pop(match)
            if PRODUCE_SKILL.get(made['script']) != stat or made['count'] < 1:
                self._fail(tick, data, f"production by {made['script']} trains no such skill")
            return self._product(tick, data, made['script'])
        if cause == 'hit':
            hit = next((e for e in causes['hit'] if (e['npc'], e['life']) == (data['npc'], data['life'])), None)
            if hit is None or hit['source'] != self.agent:
                self._fail(tick, data, 'no hit by this agent on that npc life in its tick')
            if self._respawn_cycle(tick, hit):
                self._fail(tick, data, "a hit in its npc life's respawn cycle gives no XP")
            if stat not in self._percent(hit) or stat in hit['used']:
                self._fail(tick, data, f"a hit in style {hit['style']} gives that stat no more XP")
            hit['used'].add(stat)
            return self._hit_amount(hit, stat)
        if cause == 'bury':
            match = next((k for k, e in enumerate(causes['bury']) if e['obj'] == data['obj']), None)
            if match is None or stat != PRAYER:
                self._fail(tick, data, 'no bones of that type buried in its tick')
            causes['bury'].pop(match)
            return _param(self.t, 'obj', data['obj'], 'bone_exp')
        if cause == 'light':
            match = next((k for k, (_, obj) in enumerate(self.lit) if obj == data['obj']), None)
            if match is None or stat != FIREMAKING:
                self._fail(tick, data, 'no logs of that type put down to light')
            self.lit.pop(match)
            return _param(self.t, 'obj', data['obj'], 'productexp')
        self._fail(tick, data, 'no cause')

    @staticmethod
    def _percent(hit):
        return {**HIT_PERCENT.get(hit['style'], {}), HITPOINTS: 133}

    def _respawn_cycle(self, tick, hit):
        return self.respawned.get((hit['npc'], hit['life'])) == tick

    def _hit_amount(self, hit, stat):
        mult = _param(self.t, 'npc', hit['type'], 'combat_xp_multiplier')
        return (min(hit['damage'], hit['hp']) * 10 * self._percent(hit)[stat] // 100) * mult // 1000

    def _owed(self, tick, causes):
        """The other side: each cause of this tick that gives XP has had it."""
        for made in causes['produce']:
            if made['count'] >= 1 and self.owed.get(made['script'], {}).get(made['obj'], 0) > 0:
                raise AuditError(f'production at tick {tick} ({made}) has no xp event')
        for hit in causes['hit']:
            if hit['source'] == self.agent and hit['style'] in HIT_PERCENT and not self._respawn_cycle(tick, hit):
                missing = [stat for stat in self._percent(hit) if stat not in hit['used'] and self._hit_amount(hit, stat) > 0]
                if missing:
                    raise AuditError(f'hit at tick {tick} ({hit}) has no xp event for stats {missing}')
        for bones in causes['bury']:
            if _param(self.t, 'obj', bones['obj'], 'bone_exp') > 0:
                raise AuditError(f'bones buried at tick {tick} ({bones}) have no xp event')

    def _product(self, tick, data, script):
        t, content, obj = self.t, self.t.meta['content'], data['obj']
        name = lambda n: t.obj_id(n)
        if script == '[label,get_logs]':
            row = content['trees'].get(t.loc_types[data['src']]['debugname'])
            if row is None or name(row['product']) != obj:
                self._fail(tick, data, 'no tree of that type gives that product')
            return row['productexp']
        if script in ('[label,get_ore_normal]', '[label,get_ore_fast]'):
            row = content['compact']['rocks'].get(t.loc_types[data['src']]['debugname'])
            if row is None or name(row['output']) != obj:
                self._fail(tick, data, 'no rock of that type gives that ore')
            return row['exp']
        if script == '[proc,fish_roll]':
            row = content['fish'].get(t.obj_types[obj]['debugname'])
            if row is None:
                self._fail(tick, data, 'no fish row for that item')
            return row['productexp']
        row = content['cooking']['rows'].get(t.obj_types[data['src']]['debugname'])
        if row is None or name(row['cooked']) != obj:
            self._fail(tick, data, 'no cooking row gives that item')
        return row['exp']

    def observe(self, events):
        """Check `events` (tick, kind, data), in order; returns the XP each stat gained in them (tenths)."""
        gained, k = [0] * len(self.xp), 0
        while k < len(events):
            tick = events[k][0]
            end = k
            while end < len(events) and events[end][0] == tick:
                end += 1
            causes = dict(produce=[], hit=[], bury=[])
            for _, kind, data in events[k:end]:
                if kind == 'produce':
                    causes['produce'].append(data)
                elif kind == 'hit':
                    causes['hit'].append(dict(data, used=set()))
                elif kind == 'remove' and data['why'] == 'bury':
                    causes['bury'].append(data)
                elif kind == 'remove' and data['why'] == 'light':
                    self.lit.append((tick, data['obj']))
                elif kind == 'npc_respawn':
                    if (data['npc'], data['life']) in self.respawned:
                        raise AuditError(f'a second npc_respawn of npc {data["npc"]} life {data["life"]} at tick {tick}')
                    self.respawned[(data['npc'], data['life'])] = tick
            for _, kind, data in events[k:end]:
                if kind != 'xp':
                    continue
                stat = data['stat']
                if data['before'] != self.xp[stat]:
                    self._fail(tick, data, f'the stat held {self.xp[stat]}, not {data["before"]}')
                amount = self._amount(tick, data, causes)
                if data['xp'] != amount or data['after'] != min(self.xp[stat] + amount * self.rate, XP_CAP):
                    self._fail(tick, data, f'the tables give {amount} at rate {self.rate}')
                gained[stat] += data['after'] - data['before']
                self.xp[stat] = data['after']
            self._owed(tick, causes)
            k = end
        for stat, g in enumerate(gained):
            self.gained[stat] += g
        return gained


def check_stats(start, end, gained):
    """The bookkeeping check beside XpCheck: each stat's XP from `start` to `end` is the XP its events gained."""
    for stat, (a, b, g) in enumerate(zip(start, end, gained)):
        if b - a != g:
            raise AuditError(f'stat {stat}: {b - a} XP from start to end, {g} in its events')
