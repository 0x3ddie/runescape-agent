"""Live checks that the agent's controls can do what a task needs, run by a reference player.

usage (inside scripts/stack.py): python -m baseline.control_check --output DIR [--checks NAME ...] [--trials N]

The reference player is a test instrument. It knows each check's goal and chooses only legal
actions of the policy's action table, so a pass shows that the controls can do the thing, not
that a learned policy will; nothing it knows reaches the policy. Each check states the correct
behaviour and is judged from the recorded effect rows (baseline/telemetry.py); a failure names
a control defect. Writes results.json (with the manifest) and trajectory.jsonl.gz.
"""
import argparse
import json
import re
from pathlib import Path
import time

import numpy as np

from baseline import config
from baseline.evaluate import run_trial
from baseline.telemetry import EpisodeRecorder, manifest
from baseline.world import JsonlEvidence, connect, make_episode
from act_contract import CLICK_ROWS

A_WAIT, A_ACT0, N_ACT, A_DIALOG_CONTINUE = 0, 1, 16, 62
A_HELD0, N_HELD, A_STYLE0 = 29, 8, 72
DOORS = ('Door', 'Gate', 'Large door')
# A click goes 4, 9 or 16 tiles (act-v3), so it reaches a point only to within about half the shortest.
ARRIVED = 5
AWAY_GAIN = 4  # the bar the click away must clear: act-v2's retreat gain (its env-ts/core/moves.ts, since removed)

CHECKS = dict(
    open_route=dict(spawn=(3268, 3256), goal=(3268, 3288), horizon=600, doors=False,
                    doc='walk 32 tiles north along a door-free open corridor east of Lumbridge (every tile within 3 '
                        'of the line walkable) by clicks: arrive within 5 tiles, and no offered click fails'),
    closed_door=dict(spawn=(3101, 3258), goal=(3112, 3258), horizon=400, doors=True, close_first=True,
                     door=(3101, 3258), outside=lambda x, z: x >= 3102,
                     doc='leave a Draynor house by its ordinary door (east edge of 3101,3258; closed first if an '
                         'earlier episode left it open): the door is closed, opening it is not refused and leaves it '
                         'open, the player crosses it and arrives within 5 tiles of a point 11 tiles out, and no '
                         'offered click fails'),
    locked_door=dict(spawn=(2609, 3316), goal=(2614, 3317), horizon=300, doors=True, open_first=True,
                     door=(2610, 3316), room=lambda x, z: 2611 <= x <= 2616 and 3315 <= z <= 3318,
                     doc='beside a locked East Ardougne door (east edge of 2610,3316; "Pick Lock" in the game '
                         'data), try to enter the room behind it: every try on that door is refused with "This door '
                         'is locked.", the door stays closed, the player never enters, and no offered click fails'),
    retreat=dict(spawn=(3250, 3238), target='Goblin', horizon=300,
                 doc='attack a goblin in the open field east of Lumbridge, then click away (act-v3 has no RETREAT: '
                     'the legal click farthest from it). Control (the verdict): the click ends done on its route\'s '
                     'end, which is 4+ tiles farther than the player was from '
                     "where the attacker stood. Effect (reported): distance to the actual attacker, combat and "
                     'hitpoints before and after'),
)



def levels(**given):
    """setStats for a declared start: every START_LEVELS stat (sim/spec.py), 1 (hitpoints 10) unless given."""
    return ','.join(f'{name}:{given.get(name, 10 if name == "hitpoints" else 1)}'
                    for name in ('woodcutting', 'fishing', 'mining', 'attack', 'strength', 'defence', 'hitpoints'))


# Mechanics and refusals of the four-goal task, each from a declared start: exactly these
# items (TaskCfg kit 'none', seedItems) and base levels (setStats). Development sites only, never a final
# start set's (baseline/config.py START_SETS): the Varrock south-east mine, an oak north-east of Lumbridge
# and the Lumbridge farm chickens, none with a hunting npc within 20 tiles. (A bear killed the reference
# at the oak beside the Varrock mine.) The reference requests ordinary actions only; each verdict reads
# the effect rows, and the native capture (with combat) is kept in the trajectory.
MINE, OAKS, PEN = (3285, 3366), (3229, 3265), (3230, 3297)
GOBLINS, MUGGER_START = (3252, 3240), (3175, 3361)  # the Lumbridge goblin field; a mugger's spawn west of Varrock
IRON, TIN, COPPER, OAK = (3285, 3368), (3283, 3365), (3286, 3364), (3228, 3266)
MECHANICS = dict(
    iron_14=dict(spawn=MINE, horizon=60, goal='mine', items='bronze_pickaxe:1', stats=levels(mining=14),
                 plan=dict(do='work', target='Rocks', option='Mine', at=IRON, tries=3),
                 expect=dict(refused=r'You need a Mining level of 15 to mine this rock'),
                 doc='Mining 14 with a bronze pickaxe: every Mine on an iron rock is refused with the level message'),
    iron_15=dict(spawn=MINE, horizon=240, goal='mine', items='bronze_pickaxe:1', stats=levels(mining=15),
                 plan=dict(do='work', target='Rocks', option='Mine', at=IRON, tries=30, want=('Iron ore', 2)),
                 expect=dict(produced='Iron ore'), doc='Mining 15: the same iron rock yields iron ore'),
    no_pickaxe=dict(spawn=MINE, horizon=60, goal='mine', items='bread:1', stats=levels(mining=15),
                    plan=dict(do='work', target='Rocks', option='Mine', at=TIN, tries=3),
                    expect=dict(refused=r'You need a Pickaxe to mine this rock'),
                    doc='no pickaxe: every Mine on a tin rock is refused with the pickaxe message'),
    full_inventory=dict(spawn=MINE, horizon=60, goal='mine', items='bronze_pickaxe:1,bread:27', stats=levels(mining=15),
                        plan=dict(do='work', target='Rocks', option='Mine', at=TIN, tries=3),
                        expect=dict(refused=r'Your inventory is too full to hold any more tin'),
                        doc='28 items carried: every Mine is refused with the full-inventory message'),
    prospect=dict(spawn=MINE, horizon=120, goal='mine', items='bronze_pickaxe:1', stats=levels(),
                  plan=dict(do='visit', target='Rocks', option='Prospect', at=(COPPER, TIN, IRON)),
                  expect=dict(said=(r'This rock contains copper', r'This rock contains tin', r'This rock contains iron')),
                  doc='Prospect names the ore of a copper, a tin and an iron rock, and produces nothing'),
    mine_rolls=dict(spawn=MINE, horizon=400, goal='mine', items='bronze_pickaxe:1', stats=levels(),
                    plan=dict(do='work', target='Rocks', option='Mine', at=(TIN, COPPER), tries=60, want=('Tin ore', 4)),
                    expect=dict(produced='Tin ore'),
                    doc='Mining 1: repeated tin mining, its rolls, depletion and respawn (copper while the tin is empty)'),
    oak_14=dict(spawn=OAKS, horizon=60, goal='wood', items='bronze_axe:1', stats=levels(woodcutting=14),
                plan=dict(do='work', target='Oak', option='Chop down', at=OAK, tries=3),
                expect=dict(refused=r'You need a Woodcutting level of 15 to chop down this tree'),
                doc='Woodcutting 14: every Chop down on an oak is refused with the level message'),
    oak_15=dict(spawn=OAKS, horizon=240, goal='wood', items='bronze_axe:1', stats=levels(woodcutting=15),
                plan=dict(do='work', target='Oak', option='Chop down', at=OAK, tries=20, want=('Oak logs', 1)),
                expect=dict(produced='Oak logs'), doc='Woodcutting 15: the oak yields oak logs'),
    wield=dict(spawn=PEN, horizon=40, goal='melee', items='bronze_dagger:1,bronze_sword:1', stats=levels(),
               plan=dict(do='held', uses=(('Bronze dagger', 'Wield'), ('Bronze sword', 'Wield'), ('Bronze dagger', 'Wield'))),
               expect=dict(worn=('Bronze dagger', 'Bronze sword', 'Bronze dagger')),
               doc='Wield each weapon in turn: each is worn after its request and the other returns to the pack'),
    eat=dict(spawn=PEN, horizon=60, goal='melee', items='shrimp:2,bread:1', stats=levels(hitpoints=12),
             plan=dict(do='held', damage=True, gap=3,
                       uses=(('Bread', 'Eat'), ('Shrimps', 'Eat', 0), ('Shrimps', 'Eat'), ('Shrimps', 'Eat'))),
             expect=dict(healed=True),
             doc='after 9 damage (test hook damage_three, three times; mechanics only), Eat bread, shrimps at once '
                 '(inside the eating delay), then shrimps twice 3 decisions apart: each heal and its tick'),
    # The Lumbridge farm pen's double gate is the type of the Entrana pen's (loc_1553 outer and loc_1551 main,
    # wall shape, angle 2; general_use/scripts/gates.rs2 opens both for 500 ticks): how long does it stay open?
    gate_timer=dict(spawn=(3234, 3296), horizon=700, goal='melee', items='shrimp:1', stats=levels(),
                    plan=dict(do='gate', gate=((3236, 3295), (3236, 3296))), expect=dict(reopened=True),
                    doc='open the Lumbridge pen gate once, then wait: the ticks from opening until it offers Open again'),
    # setStats writes levels without [changestat], so the combat varps of [proc,player_combat_stat] keep a new
    # player's values until something recomputes them. Re-selecting a new player's style (row 0) does
    # ([proc,set_attackstyle]); the declared-start setup takes that step (scripted.prepare_declared). Under mask-v2
    # row 0 is offered only until then, so the probe's first step is a WAIT (until 20 October it was row 0).
    combat_stats=dict(spawn=PEN, horizon=40, goal='melee', items='bronze_sword:1',
                      stats=levels(attack=10, strength=10, defence=10, hitpoints=12),
                      plan=dict(do='probe', steps=(A_WAIT, A_WAIT, ('Bronze sword', 'Wield'), A_WAIT)),
                      expect=dict(computed_at_arm=True),
                      doc='read the combat varps (test hook combat_varps, read only) at the first decision, after WAIT '
                          'and after Wield: computed from the set levels at arm, and the wielded sword changes them'),
    # The one check allowed at a final start set (config.START_SETS): does anything on Entrana remove or block a
    # weapon? The only such script is the Port Sarim boat monk's boarding check, which a reset never meets.
    final_weapon_kept=dict(spawn=config.START_SETS['entrana']['start'], horizon=240, goal='melee',
                           items=f"{config.START_KIT_ITEMS},bronze_dagger:1,bronze_sword:1", stats=levels(),
                           plan=dict(do='held', uses=(('Bronze dagger', 'Wield'),)), expect=dict(kept='Bronze dagger'),
                           doc='at the Entrana start (either engine build), wield the dagger and carry the sword for the '
                               'horizon: the dagger stays worn and no carried item is removed'),
    **{f'melee_{weapon or "unarmed"}_{style}': dict(
        spawn=PEN, horizon=400, goal='melee', items=f'{weapon}:1,shrimp:2' if weapon else 'shrimp:2', stats=levels(),
        plan=dict(do='melee', weapon={'bronze_dagger': 'Bronze dagger', 'bronze_sword': 'Bronze sword'}.get(weapon),
                  style=style),
        expect=dict(killed=True),
        doc=f'level 1 melee {"unarmed" if not weapon else "with a " + weapon}, style row {style}: attack chickens for '
            'the horizon, eating at 4 hitpoints; hits, damage, retaliation, deaths and respawns are captured')
        for weapon in (None, 'bronze_dagger', 'bronze_sword') for style in range(4)},
    # sustained-combat-v1 (sim/combat.py): the enemies that damage the player, at development sites outside every final
    # start set. The capture records each enemy attack on the agent with its planned damage (agent-hit), so the npc-side
    # rolls, the attack cadence, aggression and the healing causes can be compared with the simulator (sim.check combat).
    **{f'combat_goblins_d{d}': dict(
        spawn=GOBLINS, horizon=600, goal='melee', items='bronze_sword:1,shrimp:2,bread:2',
        stats=levels(attack=5, strength=5, defence=d, hitpoints=12),
        plan=dict(do='fight', enemies=('Goblin',), weapon='Bronze sword', eat_at=4, wait_for_attack=False),
        expect=dict(fought='Goblin'),
        doc=f'attack 5, strength 5, defence {d}, hitpoints 12, sword: fight the Lumbridge goblins one after another for the '
            'horizon, eating at 4 hitpoints; every goblin attack on the player (planned damage, zero included), the '
            'player\'s hits, eats, regeneration and level-ups are captured')
        for d in (1, 10)},
    combat_mugger=dict(spawn=MUGGER_START, horizon=500, goal='melee', items='bronze_sword:1,shrimp:2,bread:2',
                       stats=levels(attack=5, strength=5, defence=5, hitpoints=12),
                       plan=dict(do='fight', enemies=('Mugger',), weapon='Bronze sword', eat_at=4, wait_for_attack=True),
                       expect=dict(aggressed='Mugger'),
                       doc='attack, strength and defence 5, hitpoints 12, sword: wait at the spawn of the mugger west of Varrock until it '
                           'attacks (its hunt), then fight it and each respawn, eating at 4 hitpoints'),
    # sim.check combat, rule C6: does the auto-retaliate answer an attack with no request, and leave a walking player alone?
    combat_idle=dict(spawn=MUGGER_START, horizon=500, goal='melee', items='bronze_sword:1,shrimp:2,bread:2',
                     stats=levels(attack=5, strength=5, defence=5, hitpoints=12),
                     plan=dict(do='idle'), expect=dict(aggressed='Mugger'),
                     doc='the mugger check with WAIT only (dialogs continued, the sword stays in the pack): the mugger attacks '
                         'after its hunt, and the capture shows whether the agent hits it with no request'),
    combat_walk=dict(spawn=MUGGER_START, horizon=500, goal='melee', items='bronze_sword:1,shrimp:2,bread:2',
                     stats=levels(attack=5, strength=5, defence=5, hitpoints=12),
                     plan=dict(do='walk'), expect=dict(aggressed='Mugger'),
                     doc='the mugger check: WAIT until the mugger attacks, then minimap clicks one after another for the '
                         'horizon: the capture shows whether the agent hits anything while it walks under attack'),
    combat_regen=dict(spawn=PEN, horizon=480, goal='melee', items='shrimp:1', stats=levels(hitpoints=12),
                      plan=dict(do='regen'), expect=dict(regenerated=True),
                      doc='after 9 damage (test hook damage_three, three times; mechanics only), wait for the horizon: '
                          'natural healing, 1 hitpoint at a time, on the health_regen timer'),
)


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class Reference:
    """Walks to within ARRIVED tiles of `goal` (into `room`, when the check has one) by the legal compass move that best closes the
    distance, skipping a move that failed from the same tile, and, in door checks, opens a door
    within 3 tiles (at most 3 tries a door) when no move helps; or, with `target`, attacks the
    nearest such NPC and retreats once in combat. It continues any open dialog first, and does not
    try again a door the game said is locked."""
    def __init__(self, check):
        self.check = check

    def reset(self, seed):
        self.failed, self.opened, self.retreated, self.last, self.setup = set(), {}, False, None, True
        self.tried, self.locked = None, set()  # the door just tried, and doors the game called locked
        self.facts = dict(arrived_tick=None, door_opens=0, entered=False, setup=None, retreat_decision=None)
        self.trace = []  # one entry per decision: what the reference saw before acting

    def door_state(self, raw):
        """The check's door: closed while it offers Open on its own tile, open while its open form
        (offering Close) stands within a tile of it, else unseen."""
        door = self.check.get('door')
        if not door:
            return None
        acts = [a for a in raw.get('actSel', []) if a and a['name'] in DOORS]
        if any(a['option'] == 'Open' and (a['x'], a['z']) == door for a in acts):
            return 'closed'
        return 'open' if any(a['option'] == 'Close' and cheb((a['x'], a['z']), door) <= 1 for a in acts) else None

    def see(self, raw):
        p, a = raw['player'], raw.get('attacker')
        self.trace.append(dict(tick=raw.get('tick'), at=(p['x'], p['z']), hp=p.get('hp'), in_combat=bool(p.get('inCombat')),
                               attacker=(p['x'] + a['dx'], p['z'] + a['dz']) if a else None, door=self.door_state(raw)))

    def click_probability(self, observation):
        return None

    def _remember(self, raw):
        """What the last decision did: a click that failed from its tile is not tried there again."""
        fb = raw.get('feedback') or {}
        ended = fb.get('ended') or {}
        if self.last and ended.get('reason') == 'failed' and ended.get('action', -1) in CLICK_ROWS:
            self.failed.add((*self.last, ended['action']))

    def act(self, observation, task):
        raw, mask = task.raw, np.asarray(observation['mask']).reshape(-1)
        p = raw['player']
        here = (p['x'], p['z'])
        self.see(raw)
        self._remember(raw)
        if self.tried and any('locked' in line['text'].lower() for line in (raw.get('feedback') or {}).get('chat', [])):
            self.locked.add(self.tried)  # told it is locked: a player stops trying it
        self.last = here
        if raw.get('dialogOpen') and mask[A_DIALOG_CONTINUE]:
            return A_DIALOG_CONTINUE  # a level-up message blocks every other action
        if 'target' in self.check:
            return self._retreat(raw, mask, here)
        if self.setup:
            # The world keeps doors as earlier episodes left them: close this check's door first,
            # or try the locked door before walking.
            self.setup = False
            option = 'Close' if self.check.get('close_first') else 'Open' if self.check.get('open_first') else None
            near = [A_ACT0 + i for i, a in enumerate(raw.get('actSel', [])[:N_ACT])
                    if option and a and mask[A_ACT0 + i] and a['name'] in DOORS and a['option'] == option and a['distance'] <= 2]
            near = near or [A_ACT0 + i for i, a in enumerate(raw.get('actSel', [])[:N_ACT])
                            if option and a and mask[A_ACT0 + i] and a['name'] in DOORS and a['option'] == option]
            if near:
                self.tried = self.check.get('door') if option == 'Open' else None
                self.facts['door_opens'] += option == 'Open'
                self.facts['setup'] = option
                return near[0]
        goal, room = self.check['goal'], self.check.get('room')
        self.facts['entered'] |= bool(room and room(*here))
        if (room(*here) if room else cheb(here, goal) <= ARRIVED):
            self.facts['arrived_tick'] = self.facts['arrived_tick'] or raw.get('tick')
            return A_WAIT
        moves = []
        for k, c in enumerate(raw.get('clicks', [])):
            if mask[CLICK_ROWS[k]] and (*here, CLICK_ROWS[k]) not in self.failed:
                after = (here[0] + c['dx'], here[1] + c['dz'])
                moves.append((cheb(here, goal) - cheb(after, goal), c['open'], CLICK_ROWS[k]))
        best = max(moves)[0::2] if moves else (0, None)
        if best[0] > 0:
            return best[1]
        doors = [(a['distance'], A_ACT0 + i, (a['x'], a['z'])) for i, a in enumerate(raw.get('actSel', [])[:N_ACT])
                 if a and mask[A_ACT0 + i] and a['name'] in DOORS and a['option'] == 'Open' and a['distance'] <= 3
                 and self.opened.get((a['x'], a['z']), 0) < 3 and (a['x'], a['z']) not in self.locked]
        if doors and self.check['doors']:
            _, action, door = min(doors)
            self.tried = door
            self.opened[door] = self.opened.get(door, 0) + 1
            self.facts['door_opens'] += 1
            self.failed.clear()  # an opened door may make a failed click possible
            return action
        return A_WAIT if mask[A_WAIT] else int(np.flatnonzero(mask)[0])

    def _retreat(self, raw, mask, here):
        name = self.check['target']
        foes = [n for n in raw.get('npcs', []) if n.get('name') == name]
        if self.retreated:
            return A_WAIT
        if raw['player'].get('inCombat') and foes:
            away = [(min(cheb((c['dx'], c['dz']), (n['dx'], n['dz'])) for n in foes), CLICK_ROWS[k])
                    for k, c in enumerate(raw.get('clicks', [])) if mask[CLICK_ROWS[k]]]
            if away:
                self.retreated = True
                self.facts['retreat_decision'] = len(self.trace) - 1
                self.facts['retreat_action'] = max(away)[1]
                return max(away)[1]
        attacks = [(a['distance'], A_ACT0 + i) for i, a in enumerate(raw.get('actSel', [])[:N_ACT])
                   if a and mask[A_ACT0 + i] and a['name'] == name and a['option'] == 'Attack']
        if attacks:
            return min(attacks)[1]
        return A_WAIT if mask[A_WAIT] else int(np.flatnonzero(mask)[0])


class Mechanic:
    """Requests one mechanics check's ordinary actions from its declared start. `work` repeats an
    option on the target at a tile (the next listed tile while one is not offered) until its tries or
    wanted items are reached; `visit` uses the option once on each tile; `held` uses inventory options
    in order (after the damage hook, for eating); `melee` wields, sets the style row, then attacks the
    nearest chicken, eating at 4 hitpoints or less. It continues dialogs and waits while an option runs."""
    def __init__(self, check):
        self.check = check

    def reset(self, seed):
        self.requests, self.uses, self.waited, self.damaged, self.styled, self.attacked = 0, 0, 0, False, False, False
        self.home, self.walking = None, None
        self.facts = dict(requests=[], worn=[], hp=[])

    def click_probability(self, observation):
        return None

    def offered(self, raw, mask, option, name=None, at=None):
        return [(a['distance'], A_ACT0 + i) for i, a in enumerate(raw.get('actSel', [])[:N_ACT])
                if a and mask[A_ACT0 + i] and a['option'] == option and (name is None or a['name'] == name)
                and (at is None or (a['x'], a['z']) == at)]

    def held(self, raw, mask, name, option):
        return next((A_HELD0 + k for k, h in enumerate(raw.get('heldSel', [])[:N_HELD])
                     if h and h['name'] == name and h['option'] == option and mask[A_HELD0 + k]), None)

    def act(self, observation, task):
        raw, mask = task.raw, np.asarray(observation['mask']).reshape(-1)
        plan, p = self.check['plan'], raw['player']
        self.facts['hp'].append((raw.get('tick'), p.get('hp')))
        self.facts['worn'].append([e['name'] for e in raw.get('equipment', [])])
        if (raw.get('dialogOpen') or raw.get('dialogOptions')) and mask[A_DIALOG_CONTINUE]:
            return A_DIALOG_CONTINUE  # a refusal box or a level-up holds every other action
        if raw.get('optionBusy') and plan['do'] not in ('melee', 'fight', 'walk'):
            return A_WAIT
        if plan['do'] in ('fight', 'idle', 'walk'):
            self.facts.setdefault('fight', []).append(dict(tick=raw.get('tick'), hp=p.get('hp'), in_combat=bool(p.get('inCombat')),
                                                           attacker=(raw.get('attacker') or {}).get('name')))
        do = dict(work=self._work, visit=self._visit, held=self._held, melee=self._melee, gate=self._gate,
                  probe=self._probe, fight=self._fight, regen=self._regen, idle=lambda *_: None, walk=self._walk)[plan['do']]
        action = do(raw, mask, plan, task)
        if action:
            self.facts['requests'].append(raw.get('tick'))
        return action or A_WAIT

    def _work(self, raw, mask, plan, task):
        tiles = plan['at'] if isinstance(plan['at'][0], tuple) else (plan['at'],)
        want = plan.get('want')
        if self.requests >= plan['tries'] or (want and sum(i['count'] for i in raw['inventory'] if i['name'] == want[0]) >= want[1]):
            return None
        for tile in tiles:
            found = self.offered(raw, mask, plan['option'], plan['target'], tile)
            if found:
                self.requests += 1
                return found[0][1]
        return None

    def _visit(self, raw, mask, plan, task):
        while self.uses < len(plan['at']):
            found = self.offered(raw, mask, plan['option'], plan['target'], plan['at'][self.uses])
            if not found:
                return None  # not offered yet: wait for it
            self.uses += 1
            return found[0][1]
        return None

    def _held(self, raw, mask, plan, task):
        if plan.get('damage') and not self.damaged:
            # Mechanics fault injection (engine-env test_hook), never a policy action: 3 damage, three times.
            for _ in range(3):
                reply = task.transport._rpc(dict(cmd='test_hook', operation='damage_three'), timeout_s=10.0)
                task.record('fault_injected', reply=reply)
            self.damaged = True
            return None
        if self.uses < len(plan['uses']):
            name, option, *gap = plan['uses'][self.uses]
            if self.uses and self.waited < (gap[0] if gap else plan.get('gap', 0)):
                self.waited += 1
                return None
            action = self.held(raw, mask, name, option)
            if action is not None:
                self.uses, self.waited = self.uses + 1, 0
            return action
        return None

    def _probe(self, raw, mask, plan, task):
        reply = task.transport._rpc(dict(cmd='test_hook', operation='combat_varps'), timeout_s=10.0)
        task.record('combat_varps', reply=reply)
        self.facts.setdefault('varps', []).append(dict(tick=reply['tick'], varps=reply['varps'], levels=reply['levels']))
        if self.uses >= len(plan['steps']):
            return None
        step = plan['steps'][self.uses]
        self.uses += 1
        return self.held(raw, mask, *step) if isinstance(step, tuple) else step

    def _gate(self, raw, mask, plan, task):
        near = lambda a: any(max(abs(a['x'] - x), abs(a['z'] - z)) <= 2 for x, z in plan['gate'])
        offered = sorted({a['option'] for a in raw.get('actSel', []) if a and a['name'] == 'Gate' and near(a)})
        self.facts.setdefault('gate', []).append((raw.get('tick'), offered))
        if self.uses:
            return None
        found = [(a['distance'], A_ACT0 + i) for i, a in enumerate(raw.get('actSel', [])[:N_ACT])
                 if a and mask[A_ACT0 + i] and a['name'] == 'Gate' and a['option'] == 'Open' and (a['x'], a['z']) in plan['gate']]
        if found:
            self.uses = 1
            return min(found)[1]
        return None

    def _melee(self, raw, mask, plan, task):
        p, worn = raw['player'], [e['name'] for e in raw.get('equipment', [])]
        if plan['weapon'] and plan['weapon'] not in worn:
            return self.held(raw, mask, plan['weapon'], 'Wield')
        if not self.styled and mask[A_STYLE0 + plan['style']]:
            self.styled = True
            return A_STYLE0 + plan['style']
        if p.get('hp') is not None and p['hp'] <= 4:
            food = self.held(raw, mask, 'Shrimps', 'Eat')
            if food is not None:
                return food
        if raw.get('optionBusy'):
            return None
        found = self.offered(raw, mask, 'Attack', 'Chicken')
        return min(found)[1] if found else None


    def _fight(self, raw, mask, plan, task):
        """Wield, eat at eat_at, fight back an attacker, else (unless waiting to be attacked first) attack the nearest
        listed enemy; continue a running attack."""
        p, worn = raw['player'], [e['name'] for e in raw.get('equipment', [])]
        if plan['weapon'] not in worn:
            return self.held(raw, mask, plan['weapon'], 'Wield')
        if p.get('hp') is not None and p['hp'] <= plan['eat_at']:
            for food in ('Shrimps', 'Bread'):
                eat = self.held(raw, mask, food, 'Eat')
                if eat is not None:
                    return eat
        attacker = (raw.get('attacker') or {}).get('name')
        if attacker in plan['enemies']:
            self.attacked = True
        if raw.get('optionBusy'):
            return None
        if plan['wait_for_attack'] and not self.attacked:
            return None
        found = [f for name in plan['enemies'] for f in self.offered(raw, mask, 'Attack', name)]
        return min(found)[1] if found else None

    def finish(self, raw):
        """The last observation closes a running walk."""
        self._walk_ended(raw)

    def _walk_ended(self, raw):
        """Note a running walk's arrival, the first observation that says no option runs; facts['walks'] holds [request tick,
        arrival tick or None]. True while no walk runs."""
        if self.walking and not (raw.get('feedback') or {}).get('running'):
            self.facts['walks'][-1][1], self.walking = raw.get('tick'), False
        return not self.walking

    def _walk(self, raw, mask, plan, task):
        """WAIT until an attacker shows, then one minimap click after another: the farthest legal open click, or once 8 or
        more tiles from where the walking began, the one that aims nearest that tile, so the attacker stays in reach."""
        here = (raw['player']['x'], raw['player']['z'])
        self.attacked = self.attacked or bool(raw.get('attacker'))
        if not self.attacked or not self._walk_ended(raw):
            return None
        self.home = self.home or here
        clicks = [(c, CLICK_ROWS[k]) for k, c in enumerate(raw.get('clicks', [])) if mask[CLICK_ROWS[k]]]
        if not clicks:
            return None
        row = (min(clicks, key=lambda cr: cheb((here[0] + cr[0]['dx'], here[1] + cr[0]['dz']), self.home))
               if cheb(here, self.home) >= 8 else max(clicks, key=lambda cr: (cr[0]['open'], max(abs(cr[0]['dx']), abs(cr[0]['dz'])))))[1]
        self.walking = True
        self.facts.setdefault('walks', []).append([raw.get('tick'), None])
        return row

    def _regen(self, raw, mask, plan, task):
        if not self.damaged:
            for _ in range(3):  # the eat check's fault injection: 3 damage, three times (never a policy action)
                reply = task.transport._rpc(dict(cmd='test_hook', operation='damage_three'), timeout_s=10.0)
                task.record('fault_injected', reply=reply)
            self.damaged = True
        return None


def mechanic_verdict(check, facts, effects, report):
    """Judge a mechanics check from its effect rows, the reference's view and the certified report."""
    plan, expect = check['plan'], check['expect']
    said = lambda e: ' '.join(e.get('chat', [])) + ' ' + e.get('dialog', '')
    tried = [e for e in effects if 'option' in plan and (e.get('meaning') or {}).get('option') == plan['option']
             and (e.get('meaning') or {}).get('name') == plan['target']]
    gained = lambda name: sum(v for e in effects for k, v in (e.get('inventory') or {}).items() if k.startswith(name + ' (') and v > 0)
    found = dict(requests=len(tried), refused=[(e.get('ended') or {}).get('refused') for e in tried],
                 produced=report['produced'])
    if 'refused' in expect:
        texts = [said(e) for e in effects]
        found.update(message=any(re.search(expect['refused'], t) for t in texts),
                     gained=sum(v for e in effects for v in (e.get('inventory') or {}).values() if v > 0))
        found['passed'] = bool(tried) and found['message'] and found['gained'] == 0 and report['produced'] == 0
    elif 'produced' in expect:
        found.update(gained=gained(expect['produced']))
        found['passed'] = found['gained'] >= 1 and report['survived']
    elif 'said' in expect:
        texts = ' '.join(said(e) for e in effects)
        found.update(messages=[bool(re.search(m, texts)) for m in expect['said']])
        found['passed'] = all(found['messages']) and report['produced'] == 0
    elif 'worn' in expect:
        changes = [w for i, w in enumerate(facts['worn']) if i == 0 or w != facts['worn'][i - 1]]
        found.update(worn=changes)
        found['passed'] = [w for c in changes for w in c][-len(expect['worn']):] == list(expect['worn']) and \
            all(len(c) <= 1 for c in changes)
    elif 'kept' in expect:
        losses = {k: v for e in effects if (e.get('meaning') or {}).get('option') != 'Wield'
                  for k, v in (e.get('inventory') or {}).items() if v < 0}
        found.update(worn_at_end=facts['worn'][-1] if facts['worn'] else None, removed=losses)
        found['passed'] = found['worn_at_end'] == [expect['kept']] and not losses and report['survived']
    elif 'computed_at_arm' in expect:
        seen = [v['varps'] for v in facts.get('varps', [])]
        found.update(varps=seen[:4])
        found['passed'] = len(seen) >= 4 and seen[0] == seen[1] and seen[3] != seen[1]
    elif 'reopened' in expect:
        seen = facts.get('gate', [])
        opened = next((t for t, options in seen if options == ['Close']), None)
        closed = next((t for t, options in seen if opened is not None and t > opened and 'Open' in options), None)
        found.update(opened_tick=opened, closed_tick=closed, open_for=None if closed is None else closed - opened)
        found['passed'] = closed is not None
    elif 'fought' in expect or 'aggressed' in expect:
        seen = facts.get('fight', [])
        attacked = [f['tick'] for f in seen if f['attacker'] == (expect.get('fought') or expect.get('aggressed'))]
        attacks = [e for e in effects if (e.get('meaning') or {}).get('option') == 'Attack']
        found.update(attacked_from=attacked[0] if attacked else None, attack_requests=len(attacks),
                     first_attack_request=attacks[0]['tick'] if attacks else None,
                     hp=[f['hp'] for f in seen][::25])
        if 'aggressed' in expect:  # it attacked first: before this player asked to attack anything
            found['passed'] = bool(attacked) and (not attacks or attacked[0] <= attacks[0]['tick'])
        else:
            found['passed'] = bool(attacks) and bool(attacked)
    elif 'regenerated' in expect:
        hp = [h for _, h in facts['hp'] if h is not None]
        found.update(hp_low=min(hp) if hp else None, hp_end=hp[-1] if hp else None)
        found['passed'] = bool(hp) and hp[-1] > min(hp)
    elif 'healed' in expect:
        heals = [(e['tick'], e['hp'], (e.get('meaning') or {}).get('name')) for e in effects if e.get('hp', 0) > 0]
        eats = [(e['tick'], (e.get('meaning') or {}).get('name'), e.get('hp', 0)) for e in effects
                if (e.get('meaning') or {}).get('option') == 'Eat']
        found.update(heals=heals, eats=eats)
        found['passed'] = len(heals) >= 2
    else:
        kills = report.get('kills') or {}
        found.update(kills=kills)
        found['passed'] = kills.get('lower', 0) >= 1 and report['survived']
    return found


def verdict(name, check, facts, effects, trace):
    """Judge a check from what was recorded: the effect rows (indexed by decision) and the
    reference's view before each decision (`trace`, ending with the final observation)."""
    moves_failed = sum(1 for e in effects if 'ended' in e and e['ended']['reason'] == 'failed'
                       and (e['ended']['action'] or -1) in CLICK_ROWS)
    found = dict(clicks_failed=moves_failed, arrived_tick=facts['arrived_tick'], setup=facts['setup'])
    door = check.get('door')
    if door:
        # Tries on this door, by the engine's own note of what it dispatched.
        tries = [e for e in effects if (e.get('meaning') or {}).get('option') == 'Open'
                 and f'@{door[0]},{door[1]},' in e.get('action_note', '')]
        states = [t['door'] for t in trace]
        after = lambda e: trace[e['decision'] + 1]['door'] if e['decision'] + 1 < len(trace) else None
        found.update(door_tries=len(tries), tries_refused=[(e.get('ended') or {}).get('refused') for e in tries],
                     door_states=[x for i, x in enumerate(states) if i == 0 or x != states[i - 1]])
    if name == 'open_route':
        found['passed'] = facts['arrived_tick'] is not None and moves_failed == 0
    elif name == 'closed_door':
        opened = [e for e in tries if trace[e['decision']]['door'] == 'closed' and after(e) == 'open'
                  and (e.get('ended') or {}).get('refused') is False]
        found.update(opened=len(opened), crossed=any(check['outside'](*t['at']) for t in trace))
        found['passed'] = bool(opened) and found['crossed'] and facts['arrived_tick'] is not None and moves_failed == 0
    elif name == 'locked_door':
        # The door's own text, in the try's row or the next one.
        said = lambda e: any('this door is locked' in line.lower() for r in effects
                             if r['decision'] in (e['decision'], e['decision'] + 1) for line in r.get('chat', []))
        found.update(said_locked=[said(e) for e in tries], entered=any(check['room'](*t['at']) for t in trace))
        found['passed'] = (bool(tries) and all(r is True for r in found['tries_refused']) and all(found['said_locked'])
                           and 'open' not in states and not found['entered'] and moves_failed == 0)
    else:
        found['passed'] = False
        r = facts['retreat_decision']
        ends = [] if r is None else [e for e in effects if (e.get('ended') or {}).get('action') == facts.get('retreat_action')
                                     and e['decision'] >= r]
        issued = next((e for e in effects if e['decision'] == r), None)
        destination = re.search(r'to (\d+),(\d+)', (issued or {}).get('action_note', ''))
        if r is None or not ends or not destination:
            return found
        destination = (int(destination[1]), int(destination[2]))
        start, end = trace[r], trace[min(ends[0]['decision'] + 1, len(trace) - 1)]
        attacker = start['attacker']
        found['control'] = dict(retreat_end=ends[0]['ended']['reason'], destination=destination,
                                reached=cheb(end['at'], destination) <= 1,
                                gain_from_attacker_start=cheb(destination, attacker) - cheb(start['at'], attacker) if attacker else None)
        found['effect'] = dict(attacker_distance=[cheb(start['at'], attacker) if attacker else None,
                                                  cheb(end['at'], end['attacker']) if end['attacker'] else None],
                               in_combat=[start['in_combat'], end['in_combat']], hp=[start['hp'], end['hp']])
        control = found['control']
        found['passed'] = (control['retreat_end'] == 'done' and control['reached']
                           and control['gain_from_attacker_start'] is not None and control['gain_from_attacker_start'] >= AWAY_GAIN)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checks', nargs='+', choices=[*CHECKS, *MECHANICS], default=list(CHECKS),
                        help='default: the control checks; mechanics checks run when named')
    parser.add_argument('--trials', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = JsonlEvidence(args.output / 'trajectory.jsonl.gz')
    effects = []
    def write(kind, **data):
        evidence(kind, **data)
        if kind == 'effect':
            effects.append(data)
    record = EpisodeRecorder(write)
    results = dict(status='started', manifest=manifest(args.output), checks={}, trials=[])
    started = time.monotonic()
    task = transport = None
    try:
        (transport, service), = connect(record)
        previous = None
        for name in args.checks:
            check = CHECKS.get(name) or MECHANICS[name]
            results['checks'][name] = check['doc']
            x, z = check['spawn']
            fixture = dict(id=f'check-{name}', kind='drop', horizon=check['horizon'], capacity='respawn', reset_seed=0,
                           spawnTele=f'0,{x >> 6},{z >> 6},{x & 63},{z & 63}')
            if name in MECHANICS:
                fixture.update(capacity='declared', seed_items=check['items'], set_stats=check['stats'], goal=check['goal'])
            for trial in range(args.trials):
                record('scheduled_episode', index=len(results['trials']), fixture=fixture, check=name, trial=trial)
                task = make_episode(transport, service, fixture, record)
                task.previous_identity = previous
                effects.clear()
                player = Mechanic(check) if name in MECHANICS else Reference(check)
                outcome = run_trial(task, fixture, player, trial, record, config.MAX_EVAL_DECISIONS)
                previous = task.previous_identity
                if name in MECHANICS:
                    found = mechanic_verdict(check, player.facts, effects, outcome['report'])
                else:
                    player.see(task.raw)  # the final observation closes the trace
                    found = verdict(name, check, player.facts, effects, player.trace)
                start = next((e for e in effects), None)
                results['trials'].append(dict(check=name, trial=trial, decisions=len(effects), **found,
                                              survived=record.last['survived'], option_ends=record.last['option_ends'],
                                              first_effect_tick=start and start.get('tick')))
                print(f"[{name} {trial}] {'PASS' if found['passed'] else 'FAIL'} " +
                      ' '.join(f'{k}={v}' for k, v in found.items() if k != 'passed'), flush=True)
        results['status'] = 'completed'
    except BaseException as error:
        results.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if task is not None: task.close()
            elif transport is not None: transport.close()
        finally:
            evidence.close()
            results['seconds'] = time.monotonic() - started
            (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
