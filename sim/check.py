"""Checks of the simulator against the saved native evidence (logs/wt/act/logs/goal).

    .venv-sim/bin/python -m sim.check replay --evaluations logs/native-click/fidelity --out logs/native-click/fidelity/replay.json
    .venv-sim/bin/python -m sim.check chase --evaluations logs/native-click/chase --out logs/native-click/chase/chase.json
    .venv-sim/bin/python -m sim.check policy --runs logs/sim/runs/eval
    .venv-sim/bin/python -m sim.check coverage

`replay` re-plays each native evaluation episode (baseline.evaluate's trajectories keep every
observation; each ran alone on a fresh world) in the simulator: the same setup, then each native
decision translated by its meaning (the same target, option or slot) into the simulator's
action. What the native record fixes and the simulator cannot reproduce is controlled:
each decision ends at the native tick (native pace jitter is unseeded), a gathering roll
succeeds only while the native decision's item count is not reached, a depletion roll
follows the native target's state, and before each decision the visible npcs and locs are
moved to their native places. Everything else is compared after every decision, before any
resynchronisation, and a mismatch is counted against the field it is in. Decisions that
touch an npc or combat are reported apart, since npc movement inside a decision and combat
draws are not controlled. Only act-v3 recordings replay: an earlier recording's rows mean other
actions (the act-v1 and act-v2 evaluations under logs/wt/act/logs/goal-eval stop at their start).

`chase` is the native-click check of an ACT on a wandering npc, on a recording made with RL_TICK_TRACE set and
`baseline.evaluate --policy chase`: the replay above with every traced npc forced onto its native tile at every tick, so the
wander draws are the native ones. The player's tile must equal the native one at every tick from the ACT to the first hit or
experience on either side (combat draws are not controlled), and natively the player must reach the npc and attack it.

`policy` compares simulator evaluations (sim.run, same policies, fixtures, trial seeds and
horizon) with the native scores, per policy and goal. The rule was fixed before any
simulator result was read: the 90% bootstrap interval of the difference in mean return
(simulator minus native) must lie inside +-MARGIN for "equivalent"; an interval that
excludes 0 is "different"; anything else is "inconclusive", which is not evidence of
equivalence. A coverage failure has no return, so it is scored both 0 and 1 and both
bounds must pass.

`coverage` reads native goal-v1 training (episodes.jsonl.gz) and counts, per quarter of each
run, the episodes whose every decision the simulator covers by meaning (`uncovered`).

`mechanics` compares the compact world's mechanics with native captures of the development sites
(the native captures' manifest; only its listed captures, their sha256 checked). Native draws are
not reproducible, so each rule is either exact, on deterministic structure, or a bounded
distribution check. The rules were fixed before any comparison was computed:

- E1 texts: every refusal and notice text natively shown for a check (dialog or chat) is the
  simulator's text for that case.
- E2 Prospect: "This rock contains <ore>." arrives exactly 3 ticks after "You examine the rock for
  ores..." (when that first line is captured) and nothing is produced.
- E3 mining cadence: production tick minus the last "You swing your pick at the rock." arrival is
  6 mod 8 for [label,get_ore_fast] (iron) and 7 mod 8 for [label,get_ore_normal] (copper, tin).
- E4 woodcutting cadence: production tick minus the last "You swing your axe at the tree." arrival
  is 2 mod 4.
- E5 Wield: after each Wield, the inventory and worn slots equal the simulator's from the same state.
- E6 Eat: each eat's heal is min(base, hp + the food's heal) - hp with the simulator's messages; an
  Eat dispatched within 2 ticks of the previous one changes nothing.
- E7 hits: each damage event comes 1 tick after its hit, for min(planned damage, hitpoints).
- E8 attack cadence: per life, agent hits are at least the weapon's attackrate apart (4 unarmed,
  dagger and sword), and that is the most frequent gap.
- E9 lifecycle: defeat-to-removal and removal-to-respawn intervals lie in the simulator's support
  for the same npc type, measured from simulated fights. Added 16 October, after the first result (kept as
  it was): the defeat-to-removal mix differed (native mostly 3 ticks, simulated mostly 5) because it
  depends on the fights, not only the rules. npc_death runs npc_arrivedelay, which waits by the npc's
  last step (Npc.lastMovement, the tick after it), so per death the interval from the lethal damage at T
  is 3 + [lastMovement == T] + 2 [lastMovement == T + 1]; `rule` checks that per death on both sides
  (newer native captures record lastMovement at removal; older ones lack it).
- E10 retaliation: chickens never lower the agent's hitpoints (their max hit is 0).
- E11 depletion: a normal rock depletes on every ore; observations bracketing a rock's return
  contain its respawn (copper and tin 9 ticks, iron 19, oak 29).
- E12 gate: the observations bracketing the gate's return to "Open" contain 500 ticks after it opened.
- E13 combat varps: the native varps at each step of the combat-stats capture equal the simulator's.
- D1 mining and D2 oak rolls: successes per inferred roll against the simulator's probability at
  that roll's level (no gem roll on a gem draw), |z| <= 3 per group.
- D3 melee: per attack, P(damage >= 1) = P(hit) * maxhit / (maxhit + 1) from the simulator's rolls
  at the levels the capture's own damage and xp imply; |z| <= 3 per weapon, and no damage above
  that max hit.

`combat` compares sustained-combat-v1's mechanics (sim/combat.py) with native captures of baseline.control_check's
combat_* checks (development sites: the Lumbridge goblins, a mugger west of Varrock, the Lumbridge farm). The native
capture records every enemy attack on the agent with its planned damage, a zero included ('agent-hit'), the agent's hits
on each enemy life and the agent's hitpoints each tick. Rules fixed on 20 October before any combat capture was read:

- C1 enemy hit chance: per enemy type and declared player levels, the attacks with damage >= 1 against the simulator's
  probability P(hit) * maxhit / (maxhit + 1), from its own rolls (npc_default_attack: attack (level + 9) x (attackbonus
  + 64) against the player's defence varp for the npc's damage type, [proc,player_combat_stat] from the declared levels
  and the worn sword) and max hit; |z| <= 3 each, and no planned damage above that max hit.
- C2 enemy attack cadence: per enemy life, consecutive attacks on the agent at least its attackrate apart (4 for both),
  and that is the most frequent gap.
- C3 aggression: in the mugger check, a mugger's first attack on the agent comes before the agent asks to attack
  anything, in every trial.
- C4 natural healing: hitpoint gains outside an Eat decision and outside a Hitpoints level-up are +1 each, at least 100
  ticks apart, and hitpoints never exceed the base level.
- C5 eating in a fight: two consecutive agent hits on one enemy life with an eat strictly between them are at least the
  weapon's attackrate (the table's, 4 for the sword by the param default) + 3 apart (consume.rs2's skill delay of 3 on
  %action_delay).

C4 and C5 were corrected the same day after their first application to the simulator alone, before any combat capture
was read: their first versions demanded gains exact multiples of 100 apart, but a normal timer waits while the player
cannot act and restarts from the tick it fires (Player.processTimers; the simulator's World.process_players), and they
took the sword's attackrate as 5, which the tables do not give it.

C6 was fixed on 6 November before any capture of its two checks was made. combat_idle is
the mugger check with WAIT only: the sword stays in the pack, so the hits are unarmed, whose rate of 4 is the sword's.
combat_walk waits for the mugger's first attack, then walks, one minimap click after another. The native capture records
a hit at its roll and the simulator one tick later (env-ts/engine/combat-capture.ts), so the ticks below are native ticks
after that shift.

- C6 idle: with no Attack request in the trial, the agent still hits the mugger. In each trial's first fight, its first
  hit and the mugger's attacks differ by 2 ticks modulo 4 (a capture can begin after the first attack). Gaps between its
  hits on one life are at least 4 and most often 4, C2's form. An attack too late in a trial to be answered is no test.
- C6 walking: between a walk's request and its arrival, the agent makes no hit (busy2). At least one mugger attack lands
  inside such a walk, or the check has tested nothing.

The simulator runs the same captures' declared starts through sim/combat.py's episode code with the same reference player
on its compact layouts (combat_sim), and the same rules are applied to its world events, so a rule is shown to hold on
both sides. Fight summaries (hitpoints lost and ticks per kill) are reported, not judged: the native site is not a
compact layout.

Two refined rules were added after the first comparison and are reported beside the declared
counts, never in place of them: E4 measures from the last anchor of any kind (a dialog continue's
dispatch fires the op again; after a log that leaves the oak standing, rolls go on from that roll),
and E7 takes the capture's amount as the planned damage of [proc,npc_default_damage]'s npc_damage
with hitpoints floored at 0 (engine Npc.applyDamage), as the simulator applies it, so a hit above
the remaining hitpoints is no exception.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from act_contract import ACT_V3
from baseline import config
from sim import agent as A
from sim.data import Tables
from baseline.resource_task import GOAL_FAMILIES, GOAL_ITEMS
from sim.audit import audit
from sim.engine import Unsupported, World
from sim.task import GoalEpisode

NATIVE = Path('logs/wt/act/logs/goal')
MARGIN = 0.15
LEVEL = 0.90
RESAMPLES = 10_000


def bootstrap_difference(sim, native, rng):
    sim, native = np.asarray(sim, float), np.asarray(native, float)
    a = rng.choice(sim, (RESAMPLES, len(sim))).mean(1)
    b = rng.choice(native, (RESAMPLES, len(native))).mean(1)
    tail = (1 - LEVEL) / 2
    low, high = np.quantile(a - b, [tail, 1 - tail])
    return float(sim.mean() - native.mean()), float(low), float(high)


def verdict(low, high):
    if -MARGIN <= low and high <= MARGIN:
        return 'equivalent'
    if low > 0 or high < 0:
        return 'different'
    return 'inconclusive'


def native_trials(policy, seed, goal, endpoint='k65'):
    score = json.loads((NATIVE / f'score-s{seed}-{endpoint}.json').read_text())
    name = f'{policy}-s{seed}' if policy != 'wait' and policy != 'random' else policy
    block = score['goals'][f'goal_{goal}']
    return dict(returns=block['per_trial'][name], successes=block['successes'][name], deaths=block['deaths'][name])


def sim_trials(path):
    result = json.loads((path / 'results.json').read_text())
    rows = result['trials']
    returns = [r.get('episode_return') for r in rows]
    return dict(returns=returns, coverage=sum(r is None for r in returns), successes=result['successes'],
                deaths=result['deaths'], completed=result['completed'])


def policy_check(runs, out=None):
    rng = np.random.default_rng(0)
    rows = []
    for path in sorted(p for p in runs.iterdir() if p.is_dir() and (p / 'results.json').exists()):
        policy, seed, goal = path.name.split('-')
        endpoint = policy if policy in ('k65', 'k262') else 'k65'
        native = native_trials(policy, seed[1:], goal, endpoint)
        sim = sim_trials(path)
        bounds = []
        for fill in (0.0, 1.0):
            filled = [fill if r is None else r for r in sim['returns']]
            bounds.append(bootstrap_difference(filled, native['returns'], rng))
        verdicts = {verdict(low, high) for _, low, high in bounds}
        overall = verdicts.pop() if len(verdicts) == 1 else 'inconclusive'
        rows.append(dict(run=path.name, native_mean=float(np.mean(native['returns'])), native_successes=native['successes'],
                         native_deaths=native['deaths'], sim_completed=sim['completed'], sim_coverage_failures=sim['coverage'],
                         sim_successes=sim['successes'], sim_deaths=sim['deaths'],
                         difference_if_failures_0=bounds[0], difference_if_failures_1=bounds[1], verdict=overall))
    for r in rows:
        d0, d1 = r['difference_if_failures_0'], r['difference_if_failures_1']
        print(f"{r['run']:18s} native {r['native_mean']:.3f} ({r['native_successes']:2d} ok, {r['native_deaths']} dead) | "
              f"sim {r['sim_completed']:2d} done {r['sim_coverage_failures']:2d} uncovered {r['sim_successes']:2d} ok "
              f"{r['sim_deaths']} dead | diff [{d0[1]:+.2f},{d0[2]:+.2f}]..[{d1[1]:+.2f},{d1[2]:+.2f}] {r['verdict']}")
    if out:
        Path(out).write_text(json.dumps(dict(margin=MARGIN, level=LEVEL, resamples=RESAMPLES, rows=rows), indent=1) + '\n')
    return rows


# ---- replay ---------------------------------------------------------------------------------
GATHERED = ('logs', 'oak_logs', 'willow_logs', 'raw_shrimp', 'raw_anchovies')


def native_episodes(path):
    """Each native episode of a trajectory file (evaluate's trajectory.jsonl.gz, train's
    traces.jsonl.gz): goal, reset cfg and observation, the setup decisions, the observation
    at arm, the decisions after it as (action, observation after), and the terminal record."""
    episode = None
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            kind = r['kind']
            if kind == 'scheduled_episode':
                fixture = r['fixture']
                goal = fixture.get('goal') if isinstance(fixture, dict) else str(fixture).replace('goal_', '')
                episode = dict(index=r['episode'], goal=goal, setup=[], steps=[], start=None, file=str(path))
            elif episode is None:
                continue
            elif kind == 'reset':
                episode.update(cfg=r['cfg'], seed=r['seed'], reset=r['observation'])
            elif kind == 'resource_control' and 'name' not in episode:
                identity = (r.get('response') or {}).get('identity') or {}
                if identity.get('accountId'):
                    episode['name'] = identity['accountId']  # dialog text shows the account name
            elif kind == 'episode_start':
                episode['start'] = r['observation']
            elif kind == 'observed':
                (episode['steps'] if episode['start'] is not None else episode['setup']).append((r['action'], r['observation']))
            elif kind == 'task_terminal':
                episode['terminal'] = r
            elif kind == 'episode_measures':
                episode['measures'] = r
                yield episode
                episode = None


class ForcedTransport:
    """The setup's transport with each decision ending at its native tick; records whether
    the setup chose the native actions."""

    def __init__(self, agent, setup):
        self.agent, self.setup, self.k, self.mismatched = agent, setup, 0, []

    def step(self, action):
        if self.k >= len(self.setup):
            raise Unsupported('setup longer than the native one')
        native_action, native_obs = self.setup[self.k]
        if action != native_action:
            self.mismatched.append((self.k, native_action, action))
        self.k += 1
        obs, _ = self.agent.step_to(int(action), native_obs['tick'])
        return obs


def translate(action, native_obs, sim_obs):
    """The simulator action with the native action's meaning, or None."""
    fam = A.family(action)
    def find(key, k, fields):
        rows = native_obs.get(key) or []
        if k >= len(rows):
            return None
        want = tuple(rows[k].get(f) for f in fields)
        mine = [tuple(r.get(f) for f in fields) for r in (sim_obs.get(key) or [])]
        return mine.index(want) if want in mine else None
    if fam in ('wait', 'close', 'dialog', 'page', 'style', 'click'):  # a click's meaning is its row's offset
        return action
    if fam == 'act':
        j = find('actSel', action - A.A_ACT0, ('kind', 'id', 'option', 'x', 'z', 'level'))
        return None if j is None else A.A_ACT0 + j
    if fam == 'held':
        j = find('heldSel', action - A.A_HELD0, ('slot', 'id', 'option'))
        return None if j is None else A.A_HELD0 + j
    if fam in ('drop', 'deposit', 'use'):
        start = {'drop': A.A_DROP0, 'deposit': A.A_DEPOSIT0, 'use': A.A_USE0}[fam]
        j = find('invSel', action - start, ('slot', 'id', 'count'))
        return None if j is None else start + j
    if fam == 'withdraw':
        j = find('bankSel', action - A.A_WITHDRAW0, ('slot', 'id', 'count'))
        return None if j is None else A.A_WITHDRAW0 + j
    return None


def gathered(t, before, after):
    ids = {t.obj_ids[name] for name in GATHERED}
    def count(obs):
        total = Counter()
        for i in obs['inventory']:
            if i['id'] in ids:
                total[i['id']] += i['count']
        return total
    return count(after) - count(before)


class Controller:
    """Draws for one decision: the native decision's gathering successes, each on the latest
    roll that leaves room for the rest before the native end (a success ends a normal tree's
    option, and a level-up's dialog ends the decision, so native successes come late), and a
    depletion exactly when the native target loc changed."""

    def __init__(self, t, native_before, native_after, target):
        self.spots = {k for k, v in t.npc_types.items() if v['params'].get('fishing_movement_enum') is not None}
        self.left = gathered(t, native_before, native_after)
        self.deadline = native_after['tick'] - 1
        self.last_roll, self.interval = None, 4
        self.depleted = target is not None and not any(
            l['id'] == target['id'] and native_after['player']['x'] + l['dx'] == target['x']
            and native_after['player']['z'] + l['dz'] == target['z'] for l in native_after['locs'])

    def __call__(self, kind, default, obj, tick):
        if kind == 'deplete':
            return self.depleted
        if kind == 'respawn':  # a stump stays until a native observation shows the tree again
            return 1_000_000
        if kind == 'npc_timer':  # fishing spots move only when the native ones did
            return obj.type not in self.spots
        if kind != 'gather':
            return default
        if self.last_roll is not None and tick > self.last_roll:
            self.interval = tick - self.last_roll
        self.last_roll = tick
        if self.left[obj] <= 0:
            return False
        later = max(0, (self.deadline - tick) // self.interval)
        if sum(self.left.values()) > later:
            self.left[obj] -= 1
            return True
        return False


def resync(world, obs):
    """Move the npcs and locs the native observation shows to their native places: npcs by
    id from the npc list, then by type and place from the act targets. Everything shown is on
    the player's level; npcs on other levels are not observed and stay."""
    px, pz, level = obs['player']['x'], obs['player']['z'], obs['player'].get('level', 0)
    placed = set()
    def move(npc, x, z):
        # A fishing spot that moved was delayed two ticks first, which ends an interaction with it.
        spot = world.t.npc_types[npc.type]['params'].get('fishing_movement_enum') is not None
        if spot and world.player is not None and world.player.target is npc:
            world.player.clear_interaction()
            world.player.clear_waypoints()
        world.npc_teleport(npc, x, z)
    for n in obs['npcs']:
        npc = world.npc_by_nid.get(n['id'])
        if npc is not None and npc.active and npc.level == level:
            placed.add(npc.nid)
            x, z = px + n['dx'], pz + n['dz']
            if (npc.x, npc.z) != (x, z):
                move(npc, x, z)
    names = lambda npc: world.t.npc_types[npc.type]['name']
    wanted = [(a.get('id'), a.get('name'), a['x'], a['z']) for a in obs.get('actSel', []) if a['kind'] == 'npc']
    for type_id, name, x, z in dict.fromkeys(wanted):
        there = [npc for npc in world.npcs if npc.active and (npc.level, npc.x, npc.z) == (level, x, z)
                 and (npc.type == type_id if type_id is not None else names(npc) == name)]
        if there:  # one listed by id stands for the target; any other there is moved out below
            if not any(npc.nid in placed for npc in there):
                placed.add(there[0].nid)
            continue
        free = [npc for npc in world.npcs if npc.active and npc.level == level and npc.nid not in placed
                and (npc.type == type_id if type_id is not None else names(npc) == name)]
        if free:
            npc = min(free, key=lambda n: max(abs(n.x - x), abs(n.z - z)))
            placed.add(npc.nid)
            move(npc, x, z)
    # An npc no native list shows was farther than every list reaches (the npc list's 8th, or
    # its whole radius when it has fewer, and the act radius): one nearer here goes out to that distance.
    listed = [n['distance'] for n in obs['npcs']]
    far = A.NPC_SCAN_TILES + 1 if len(listed) < A.N_NPC_OBS else max(A.ACT_SCAN_TILES + 2, max(listed) + 1)
    for npc in world.npcs:
        if npc.active and npc.level == level and npc.nid not in placed and max(abs(npc.x - px), abs(npc.z - pz)) < far:
            spot = far_tile(world, level, px, pz, far)
            if spot is not None:
                move(npc, *spot)
    shown = [(px + l['dx'], pz + l['dz'], l['id']) for l in obs['locs']]
    shown += [(a['x'], a['z'], a['id']) for a in obs.get('actSel', []) if a['kind'] == 'loc']
    for x, z, type_id in shown:
        set_stage(world, level, x, z, lambda loc, stage: type_id if type_id in (loc.base, stage) else None)


def observe_resynced(agent):
    """The player's npc list as the native observation had it (Agent.observe_npcs, without its teleport rule): every
    npc within 15 tiles where the resync left it, since its moves are no teleports of the game."""
    p = agent.player
    for npc in agent.world.npcs:
        npc.observers = int(npc.active and npc.level == p.level and A.chebyshev(p.x, p.z, npc.x, npc.z) <= 15)


def set_stage(world, level, x, z, choose):
    for loc in world.active_locs_at(level, x, z):
        stage = world.t.loc_types[loc.base]['params'].get('next_loc_stage')
        want = choose(loc, stage)
        if want is not None and loc.type != want:
            world.change_loc(loc, want, loc.shape, loc.angle, 1_000_000)  # a stump stays until native shows the tree


def far_tile(world, level, px, pz, radius):
    """A free walkable data-area tile on the level exactly `radius` (Chebyshev) from the player."""
    ring = [(px + d, pz + e) for d in range(-radius, radius + 1) for e in (-radius, radius)]
    ring += [(px + e, pz + d) for d in range(-radius + 1, radius) for e in (-radius, radius)]
    for x, z in ring:
        a = world.t.area
        if a['x0'] <= x <= a['x1'] and a['z0'] <= z <= a['z1'] and not world.occupied.get((level, x, z)) \
                and not (world.flags(level, x, z, npc_bits=False) & 0x100):
            return x, z
    return None


def holdings(obs, key):
    return sorted((i['slot'], i['id'], i['count']) for i in obs.get(key) or [])


STATE = ('tick', 'position', 'hp', 'inventory', 'skills', 'bankOpen', 'dialogOpen', 'modalOpen', 'optionBusy')
# What replay_episode compares right after each decision, before any resync.
DECISION_FIELDS = STATE + ('feedback', 'route')
ROUTE_FIELDS = ('kind', 'action', 'source', 'offset', 'thing', 'target', 'via', 'refused', 'waypoints')


def route_key(record):
    """The route-v1 fields replay compares (ROUTE_FIELDS), or None for a decision that sent no click or ACT."""
    return None if not record else tuple(record.get(f) for f in ROUTE_FIELDS)


def route_outcome(record):
    """A native route-v1 record counted by kind and ending: 'click target', 'click fallback', 'click no route',
    'act reach', 'act delayed', ...; None for a decision that sent no click or ACT."""
    return None if not record else f"{record['kind']} {record['refused'] or record['via']}"


def compare(native, sim, fields=None):
    """Field-by-field differences of two observations (empty when they agree)."""
    out = {}
    def same(name, a, b):
        if a != b and (fields is None or name in fields or any(name.startswith(f) for f in fields)):
            out[name] = (a, b)
    np_, sp = native['player'], sim['player']
    same('tick', native['tick'], sim['tick'])
    same('position', (np_['x'], np_['z'], np_.get('level', 0)), (sp['x'], sp['z'], sp['level']))
    same('hp', np_['hp'], sp['hp'])
    same('inventory', holdings(native, 'inventory'), holdings(sim, 'inventory'))
    same('skills', [(k['level'], k['xp']) for k in native['skills']], [(k['level'], k['xp']) for k in sim['skills']])
    for key in ('bankOpen', 'dialogOpen', 'modalOpen'):
        same(key, bool(native.get(key)), bool(sim.get(key)))
    same('mask', list(native['mask']), list(sim['mask']))
    static = lambda rows: [(r['kind'], r['id'], r['option'], r['x'], r['z'], r['steps']) for r in rows if r['kind'] != 'npc']
    same('actSel static', static(native['actSel']), static(sim['actSel']))
    same('actSel npc', [(r['id'], r['option']) for r in native['actSel'] if r['kind'] == 'npc'],
         [(r['id'], r['option']) for r in sim['actSel'] if r['kind'] == 'npc'])
    same('clicks', [(c['dx'], c['dz'], c['open']) for c in native['clicks']], [(c['dx'], c['dz'], c['open']) for c in sim['clicks']])
    # the route each decision's click or ACT sent (route-v1): where from, the click's offset or the thing acted on, its
    # target, how tryMove ended or why it was refused, and the turn points
    same('route', route_key(native.get('route')), route_key(sim.get('route')))
    same('locs', [(r['id'], r['dx'], r['dz']) for r in native['locs']], [(r['id'], r['dx'], r['dz']) for r in sim['locs']])
    same('ground', [(r['id'], r['count'], r['dx'], r['dz']) for r in native['ground']],
         [(r['id'], r['count'], r['dx'], r['dz']) for r in sim['ground']])
    nf, sf = native['feedback'], sim['feedback']
    for key in ('ticks', 'dx', 'dz', 'xp', 'hp', 'gained', 'lost', 'bankOpened', 'dialogOpened', 'running', 'levelChanged'):
        same(f'feedback {key}', nf.get(key), sf.get(key))
    same('feedback ended', (nf.get('ended') or {}).get('reason'), (sf.get('ended') or {}).get('reason'))
    same('feedback chat', [c['text'] for c in nf.get('chat', [])], [c['text'] for c in sf.get('chat', [])])
    same('feedback dialog', nf.get('dialog'), sf.get('dialog'))
    same('optionBusy', native.get('optionBusy'), sim.get('optionBusy'))
    return out


FATAL = ('position', 'inventory', 'skills', 'bankOpen', 'dialogOpen', 'modalOpen', 'hp')


def npc_touched(native_before, native_after, sim, meaning):
    # combat in either world: a hit only the simulated player took is an uncontrolled draw too
    combat = (native_after['player']['inCombat'] or sim['player']['inCombat'] or native_before['player']['inCombat']
              or native_after['player']['hp'] != native_before['player']['hp']
              or sim['player']['hp'] != native_before['player']['hp'])
    return combat or meaning.get('kind') == 'npc'


def replay_episode(tables, episode, seed, prepare=None):
    cfg = dict(config.TASK_CFG)
    cfg.update({k: episode['cfg'][k] for k in ('setStats', 'paceJitterTicks', 'spawnTele') if k in episode['cfg']})
    # A fresh native world shows the reset observation at tick 10; the simulated reset takes 4.
    world = World(tables, seed=seed, start_tick=episode['reset']['tick'] - 4)
    task = GoalEpisode(world, cfg, episode['goal'], name=episode.get('name', 'wd1a0'))
    rows = dict(file=episode['file'], index=episode['index'], goal=episode['goal'], decisions=len(episode['steps']),
                replayed=0, mismatches=Counter(), npc_mismatches=Counter(), routes=Counter(), stop=None, examples={})
    version = episode['start'].get('actTargetVersion')
    if version != ACT_V3:
        rows['stop'] = f'not an {ACT_V3} recording: {version}'
        return rows
    transport = ForcedTransport(task.agent, episode['setup'])
    try:
        task.reset(0, transport)
    except Unsupported as error:
        rows['stop'] = f'setup coverage failure: {error}'
        return rows
    if transport.mismatched or transport.k != len(episode['setup']):
        rows['mismatches']['setup actions'] += 1
        rows['examples']['setup actions'] = dict(mismatched=transport.mismatched[:5], sim=transport.k,
                                                 native=len(episode['setup']))
    agent = task.agent
    if prepare is not None:
        prepare(world, agent)
    resync(world, episode['start'])
    observe_resynced(agent)
    agent.last_observation = sim = agent_obs(agent, world, task.raw)
    start = compare(episode['start'], sim)
    for name in ('tick', 'position', 'inventory', 'skills', 'mask', 'actSel static', 'actSel npc', 'clicks', 'locs', 'ground'):
        if name in start:
            rows['mismatches'][f'start {name}'] += 1
            rows['examples'].setdefault(f'start {name}', start[name])
    native = episode['start']
    for action, after in episode['steps']:
        meaning = A.describe_action(native, action, native.get('using'))
        mine = translate(action, native, sim)
        if mine is None or not sim['mask'][mine]:
            rows['stop'] = f"no simulator action with the native meaning {meaning['family']} {meaning['kind']} " \
                           f"{meaning['name']} {meaning['option']}".strip()
            break
        target = None
        if meaning['family'] == 'act' and meaning['kind'] == 'loc':
            sel = native['actSel'][action - A.A_ACT0]
            target = dict(id=sel['id'], x=sel['x'], z=sel['z'])
        world.control = Controller(tables, native, after, target)
        try:
            sim, would_stop = agent.step_to(mine, after['tick'])
        except Unsupported as error:
            rows['stop'] = f'coverage failure: {error}'
            break
        finally:
            world.control = None
        rows['replayed'] += 1
        outcome = route_outcome(after.get('route'))
        if outcome is not None:
            rows['routes'][outcome] += 1
        # the outcome now; what the agent projects once the npcs and locs are where native had them
        diff = compare(after, sim, DECISION_FIELDS)
        resync(world, after)
        observe_resynced(agent)
        if not after.get('optionBusy') and sim.get('optionBusy'):  # e.g. a native spot moved mid-decision
            agent.end_option('done', False)
            agent.option = None
        projected = agent_obs(agent, world, sim)
        diff.update(compare(after, projected, ('mask', 'actSel', 'clicks', 'locs', 'ground')))
        bucket = rows['npc_mismatches'] if npc_touched(native, after, sim, meaning) else rows['mismatches']
        for name, values in diff.items():
            bucket[name] += 1
            rows['examples'].setdefault(name, dict(decision=rows['replayed'], meaning=meaning, native=values[0], sim=values[1]))
        if would_stop is not None and would_stop < after['tick'] - 3:  # beyond any native chunking
            bucket['decision end'] += 1
            rows['examples'].setdefault('decision end', dict(decision=rows['replayed'], meaning=meaning,
                                                             native=after['tick'], sim=would_stop))
        fatal = [name for name in FATAL if name in diff]
        if fatal:
            rows['stop'] = f"state diverged ({', '.join(fatal)}) at decision {rows['replayed']}"
            break
        agent.last_observation = sim = projected
        native = after
    if rows['stop'] is None and task.terminal is not None and 'measures' in episode:
        # the return from the simulator's own production and ownership events, against the
        # native capture's: an independent account of the same native decisions
        report = audit(world.events[task.event_start:], task.initial, task.terminal, episode['goal'], tables)
        sim_return = min(report['retained'], GOAL_ITEMS) / GOAL_ITEMS if report['survived'] else 0.
        families = episode['terminal']['report']['families']
        native_goal = sum(f['produced'] for base, f in families.items() if int(base) in GOAL_FAMILIES[episode['goal']])
        rows['returns'] = dict(native=episode['measures']['episode_return'], sim=sim_return,
                               native_produced=episode['measures']['items_produced'],
                               sim_produced=sum(report['produced_items'].values()),
                               native_goal_produced=native_goal, sim_goal_produced=report['produced'])
    return rows


def agent_obs(agent, world, previous=None):
    """The agent's current observation, rebuilt after a resync moved npcs or locs."""
    obs = agent.build_obs(dict(decision=agent.decision, done=False, truncated=False, isDead=agent.player.hp <= 0))
    agent.project_items(obs)
    obs['mask'] = agent.mask(obs)
    if previous is not None:
        obs['feedback'] = previous['feedback']
    agent.last_mask_sum = sum(obs['mask'])
    return obs


# Held out until the final check: the 262k endpoints' evaluations.
HELD_OUT = 'k262-'


def replay(evaluations, pattern, out=None, limit=None, held_out=False):
    tables = Tables.load()
    results = []
    files = sorted(Path(evaluations).glob(f'{pattern}/trajectory.jsonl.gz'))
    files = [f for f in files if f.parent.name.startswith(HELD_OUT) == held_out][:limit]
    for k, path in enumerate(files):
        for episode in native_episodes(path):
            if episode['start'] is not None:
                results.append(replay_episode(tables, episode, 7919 * (k + 1)))
    total = Counter(); npc = Counter(); routes = Counter()
    decisions = sum(r['decisions'] for r in results)
    replayed = sum(r['replayed'] for r in results)
    for r in results:
        total.update(r['mismatches'])
        npc.update(r['npc_mismatches'])
        routes.update(r.get('routes', {}))
    stops = Counter((r['stop'] or 'completed').split(' at decision')[0].split(':')[0] for r in results)
    print(f'{len(results)} episodes, {replayed} of {decisions} native decisions replayed')
    print('stops:', dict(stops.most_common()))
    print('mismatches without npc or combat:', dict(total.most_common()))
    print('mismatches with npc or combat:', dict(npc.most_common()))
    print(f"native route records in replayed decisions: {sum(routes.values())}, route mismatches "
          f"{total['route']} without and {npc['route']} with npc or combat:", dict(sorted(routes.items())))
    returns = [r['returns'] for r in results if 'returns' in r]
    same = lambda key: sum(r[f'native{key}'] == r[f'sim{key}'] for r in returns)
    print(f'fully replayed episodes: return equal in {same("")} of {len(returns)}, goal production in '
          f'{same("_goal_produced")}, all production in {same("_produced")}')
    if out:
        for r in results:
            r['mismatches'], r['npc_mismatches'] = dict(r['mismatches']), dict(r['npc_mismatches'])
            r['routes'] = dict(r.get('routes', {}))
        Path(out).write_text(json.dumps(dict(episodes=results, decisions=decisions, replayed=replayed,
                                             routes=dict(sorted(routes.items()))), indent=1, default=str) + '\n')
    return results


# ---- an ACT on a wandering npc, tick by tick ----------------------------------------------------------
def chase_episode(tables, episode, seed):
    """One chase recording replayed with the native npc tiles forced tick by tick; its verdict."""
    trace = {row[0]: dict(tile=(row[1], row[2]), hp=row[3], xp=row[4], npcs={n: (x, z) for n, x, z in row[5]})
             for _, after in episode['steps'] for row in after.get('tickTrace', [])}
    # the first decision that attacks an npc, from the raw observation before it
    native, act = episode['start'], None
    for k, (action, after) in enumerate(episode['steps']):
        meaning = A.describe_action(native, action, native.get('using'))
        if meaning['family'] == 'act' and meaning['kind'] == 'npc' and meaning['option'] == 'Attack':
            sel = native['actSel'][action - A.A_ACT0]
            act = dict(decision=k, tick=native['tick'], distance=sel['distance'], route=after.get('route'),
                       ticks=[row[0] for row in after.get('tickTrace', [])], target=sel['x'], target_z=sel['z'], sel=sel)
            break
        native = after
    if act is None:
        return dict(file=episode['file'], index=episode['index'], status='no attack')
    source = next(n['id'] for n in native['npcs'] if (native['player']['x'] + n['dx'], native['player']['z'] + n['dz'])
                  == (act['target'], act['target_z']) and n['name'] == act['sel']['name'])
    seen = {}

    def prepare(world, agent):
        """A forced tile for every traced npc in the npc phase, and the player's state after each cycle."""
        moved, cycle = world.npc_movement, world.cycle

        def movement(npc, nt):
            tile = trace.get(world.tick + 1, dict(npcs={}))['npcs'].get(npc.nid)
            if tile is None:
                return moved(npc, nt)
            if tile != (npc.x, npc.z):
                world._occupy(npc, False)
                npc.x, npc.z = tile
                world._occupy(npc, True)
            npc.path = []

        def watched():
            cycle()
            p = agent.player
            seen[world.tick] = dict(tile=(p.x, p.z), hp=p.hp, xp=sum(p.stats))
        world.npc_movement, world.cycle = movement, watched

    rows = replay_episode(tables, episode, seed, prepare)
    ticks = sorted(t for t in trace if t > act['tick'])
    start = trace[ticks[0]]
    # to the first hit or experience on either side
    last, compared, mismatch = None, 0, None
    for t in ticks:
        n, s = trace[t], seen.get(t)
        if s is None or (n['hp'], n['xp']) != (start['hp'], start['xp']) or (s['hp'], s['xp']) != (seen[ticks[0]]['hp'], seen[ticks[0]]['xp']):
            last = t
            break
        compared += 1
        if n['tile'] != s['tile']:
            mismatch = dict(tick=t, native=n['tile'], sim=s['tile'])
            break
    near = next((t for t in ticks if source in trace[t]['npcs'] and
                 max(abs(trace[t]['tile'][0] - trace[t]['npcs'][source][0]),
                     abs(trace[t]['tile'][1] - trace[t]['npcs'][source][1])) <= 1), None)
    hit = next((t for t in ticks if (trace[t]['hp'], trace[t]['xp']) != (start['hp'], start['xp'])), None)
    steps = [trace[t]['npcs'].get(source) for t in ticks if near is None or t <= near]
    return dict(file=episode['file'], index=episode['index'], status='mismatch' if mismatch else 'equal', distance=act['distance'],
                dispatch=act['tick'], reached=near, first_hit=hit, compared=compared, compared_to=last, mismatch=mismatch,
                target_moves=sum(a != b for a, b in zip(steps, steps[1:]) if a and b), via=(act['route'] or {}).get('via'),
                replay_stop=rows['stop'], route_mismatches=rows['mismatches'].get('route', 0) + rows['npc_mismatches'].get('route', 0))


def chase(evaluations, out=None):
    tables = Tables.load()
    rows = [chase_episode(tables, episode, 7919 * (k + 1))
            for k, path in enumerate(sorted(Path(evaluations).glob('*/trajectory.jsonl.gz')))
            for episode in native_episodes(path) if episode['start'] is not None]
    for r in rows:
        print({k: r.get(k) for k in ('index', 'status', 'distance', 'dispatch', 'reached', 'first_hit', 'compared', 'target_moves',
                                     'via', 'mismatch', 'route_mismatches')})
    good = [r for r in rows if r['status'] == 'equal' and r.get('reached') is not None and r.get('first_hit') is not None
            and not r.get('route_mismatches')]
    print(f'{len(good)} of {len(rows)} trials reached and hit the npc with the player on the native tile at every compared tick; '
          f"{sum(r.get('compared', 0) for r in rows)} ticks compared, the target moved {sum(r.get('target_moves', 0) for r in rows)} times")
    if out:
        Path(out).write_text(json.dumps(dict(trials=rows, passed=len(good)), indent=1, default=str) + '\n')
    return rows


# ---- coverage of native training ------------------------------------------------------------------
ACT_NOTE = re.compile(r' (loc|npc|obj)=(\d+)@\d+,\d+,(\d+)')
USE_NOTE = re.compile(r'use (\d+) on (\d+)')
COVERED_FAMILIES = ('wait', 'close', 'dialog', 'click', 'style', 'withdraw', 'deposit', 'page', 'use')


def uncovered(t, meaning, note, supported):
    """What a native training decision did that the simulator does not run, judged by its
    meaning and the engine scripts its target's triggers name; None when covered (a continued
    option counts where it began). Walks are not judged: leaving the area or a long route the
    simulator cannot follow shows only when it runs."""
    family = meaning['family']
    if family in COVERED_FAMILIES:
        return None
    what = f"{meaning.get('kind', '')}:{meaning.get('name', '')}:{meaning.get('option', '')}"
    if family == 'held':
        hit = USE_NOTE.search(note or '')
        if meaning.get('option') == 'Use' and (hit is None or not any(
                t.obj_types[int(obj)]['triggers'].get('OPHELDU') for obj in hit.groups())):
            return None
        return f"held:{meaning.get('option', '')}"
    if family != 'act':
        return family
    hit = ACT_NOTE.search(note or '')
    if hit is None:
        return None
    kind, type_id, level = hit.group(1), int(hit.group(2)), int(hit.group(3))
    if level not in t.levels:
        return f'{what} (upper floor)'
    ty = {'loc': t.loc_types, 'npc': t.npc_types, 'obj': t.obj_types}[kind].get(type_id)
    if ty is None:
        return what
    ops = [o for o in (ty['op'] or [])]
    option = meaning.get('option')
    if option == 'Use' and meaning.get('item'):
        k = 'U'
    elif kind == 'obj':  # EngineAgent.pickUp: the first op that takes, else op 3
        k = str(next((i for i, o in enumerate(ops) if o and 'take' in o.lower()), 2) + 1)
    elif option in ops:
        k = str(ops.index(option) + 1)
    else:
        return what
    base = kind.upper()
    op, ap = ty['triggers'].get(f'OP{base}{k}'), ty['triggers'].get(f'AP{base}{k}')
    if (op or ap or k == 'U') and (op is None or op in supported) and (ap is None or ap in supported):
        return None
    return what


def coverage(seeds, supported=None, out=None):
    """Native goal-v1 training episodes that use only simulated content, by quarter of training."""
    from sim.engine import SCRIPTS
    t = Tables.load()
    supported = set(SCRIPTS) if supported is None else set(supported)
    rows = {}
    for seed in seeds:
        episodes, current = [], Counter()
        with gzip.open(NATIVE / seed / 'episodes.jsonl.gz', 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                if r['kind'] == 'effect':
                    missing = uncovered(t, r['meaning'], r.get('action_note'), supported)
                    if missing:
                        current[missing] += 1
                elif r['kind'] == 'episode_measures':
                    episodes.append(current)
                    current = Counter()
        n = len(episodes)
        quarters = [episodes[q * n // 4:(q + 1) * n // 4] for q in range(4)]
        shares = [sum(not e for e in part) / len(part) for part in quarters]
        touching = Counter(k for e in episodes for k in e)
        early = Counter(k for e in quarters[0] for k in e)
        rows[seed] = dict(episodes=n, covered=[sum(not e for e in part) for part in quarters],
                          per_quarter=[len(part) for part in quarters], shares=shares,
                          first_quarter_touching=dict(early.most_common(20)), touching=dict(touching.most_common(30)))
        print(f"{seed}: {n} episodes; covered by quarter " + ', '.join(f'{s:.0%}' for s in shares))
        print('  first quarter, episodes touching: ' + ', '.join(f'{k} {v}' for k, v in early.most_common(10)))
    if out:
        Path(out).write_text(json.dumps(rows, indent=1) + '\n')
    return rows


# ---- mechanics: native captures of the compact world's activities --------------------------------------
def capture_episodes(manifest):
    """Each episode of the manifest's listed captures: its check, fixture, start and decision
    observations, production ops and combat events. A trace whose sha256 differs stops the check."""
    import hashlib
    spec = json.loads(Path(manifest).read_text())
    out = []
    for capture in spec['captures']:
        path = Path(capture['trace'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != capture['sha256']:
            raise ValueError(f'{path} differs from its manifest entry')
        episodes = {}
        for line in gzip.open(path, 'rt'):
            row = json.loads(line)
            e = episodes.setdefault(row.get('episode'), dict(group=capture['group'], decisions=[], ops=[], combat=[]))
            obs = row.get('observation')
            obs = json.loads(obs) if isinstance(obs, str) else obs
            if row['kind'] == 'scheduled_episode':
                e.update(check=row['check'], trial=row['trial'], fixture=row['fixture'])
            elif row['kind'] == 'declared_start':
                e['start'] = obs
            elif row['kind'] == 'observed':
                e['decisions'].append((row['action'], obs))
            elif row['kind'] == 'resource_control' and row['operation'] == 'release':
                audit_ = row['response']['audit']
                e['ops'] = [ev for ev in audit_['events'] if ev['kind'] == 'native-op']
                e['combat'] = (audit_.get('combat') or {}).get('events', [])
                e['arm'], e['end'] = audit_['startTick'], audit_['endTick']
        out += [e for k, e in sorted(episodes.items(), key=lambda kv: kv[0]) if 'check' in e]
    return out


def arrivals(episode):
    """(tick, text) of every chat line the episode's observations published, in order."""
    lines = []
    for _, obs in episode['decisions']:
        lines += [(obs['tick'] - c['age'], c['text']) for c in obs['feedback']['chat']]
    return lines


def texts(episode):
    """The dialog texts and chat lines an episode showed."""
    shown = {obs['feedback']['dialog']['text'] for _, obs in episode['decisions'] if obs['feedback']['dialog']['text']}
    return shown | {text for _, text in arrivals(episode)}


def sim_texts(t):
    """Every text the compact world's activities can show, digits as '#' (the E1 candidates)."""
    c, x = t.meta['content'], t.meta['content']['texts']
    fill = dict(level=15, ore=('copper', 'tin', 'iron'), product=('logs', 'oak logs'), food=('shrimps', 'bread'))
    out = set()
    for key, text in x.items():
        options = [text]
        for name, values in fill.items():
            if '{' + name + '}' in text:
                options = [o.replace('{' + name + '}', str(v)) for o in options for v in (values if isinstance(values, tuple) else (values,))]
        out |= set(options)
    out |= {m for row in c['compact']['eat_messages'].values() for m in row.values() if isinstance(m, str)}
    for row in c['levelup'].values():
        out |= {row['message'], f"@dbl@{row['message']} {row['level_prefix']} 0."}
        out |= {text for _, text in row['unlock_texts'].values()}
    out |= {"I can't reach that!", 'You examine the rock for ores...'}
    return {re.sub(r'\d+', '#', o) for o in out}


def roll_starts(episode, swing):
    """Op-fire ticks that restart a gathering cadence: a swing message arrives the tick after its op
    fired; a dialog closed by a continue lets the op fire at the continue's dispatch tick."""
    anchors, prev = [], episode['start']['tick']
    for action, obs in episode['decisions']:
        if action == A.A_DIALOG_CONTINUE:
            anchors.append((prev, 'continue'))
        prev = obs['tick']
    anchors += [(tick - 1, 'swing') for tick, text in arrivals(episode) if text == swing]
    return sorted(anchors)


def cadence(episode, swing, rate, script_delay):
    """E3/E4: (production tick, residues) against the last swing (declared) and the last anchor of any
    kind (refined); rolls come at anchor + rate + k (rate + 1), production script_delay later."""
    anchors = roll_starts(episode, swing)
    rows = []
    for op in episode['ops']:
        tick = op['tick']
        swings = [a for a, kind in anchors if kind == 'swing' and a < tick]
        any_ = [a for a, _ in anchors if a < tick]
        prior = [o['tick'] for o in episode['ops'] if o['tick'] < tick]
        last = max(any_ + prior) if any_ or prior else None
        declared = (tick - script_delay[op['scriptName']] - swings[-1] - rate) % (rate + 1) if swings else None
        if last in prior:  # after an ore that did not deplete (an oak), rolls go on from that roll
            refined = (tick - script_delay[op['scriptName']] - last) % (rate + 1)
        else:
            refined = (tick - script_delay[op['scriptName']] - last - rate) % (rate + 1) if last is not None else None
        rows.append(dict(tick=tick, script=op['scriptName'], declared=declared, refined=refined))
    return rows


def hit_probability(attack, defence):
    """P(randinc(attack) > randinc(defence)), randinc(n) uniform on 0..n."""
    return sum(max(0, attack - d) for d in range(defence + 1)) / ((attack + 1) * (defence + 1))


def mechanics(manifest, out=None):
    """The rules of the module docstring against the manifest's captures."""
    from sim import world as W
    from sim.agent import A_HELD0, A_STYLE0
    from sim.engine import NPC_DEFENCE, Player, give_combat_experience, player_combat_stat, stat_random_value
    from sim.run import reference_action
    from sim.task import WORLD_START_TICK
    tables = Tables.load()
    episodes = capture_episodes(manifest)
    report = {}
    by = lambda *checks: [e for e in episodes if e['check'] in checks]
    # E1 texts
    candidates = sim_texts(tables)
    shown = {re.sub(r'\d+', '#', text) for e in episodes for text in texts(e)}
    report['E1 texts'] = dict(native=len(shown), missing=sorted(shown - candidates))
    # E2 Prospect
    pairs = []
    for e in by('prospect'):
        lines = arrivals(e)
        for k, (tick, text) in enumerate(lines):
            if text == 'You examine the rock for ores...':
                after = [(t2, x) for t2, x in lines[k + 1:] if x.startswith('This rock contains')]
                if after:
                    pairs.append(after[0][0] - tick)
        assert not e['ops']
    report['E2 prospect'] = dict(pairs=len(pairs), delays=sorted(set(pairs)), expected=3)
    # E3/E4 cadence
    rows = [r for e in by('iron_15', 'mine_rolls') for r in cadence(e, 'You swing your pick at the rock.', 7,
                                                                       {'[label,get_ore_fast]': 0, '[label,get_ore_normal]': 1})]
    report['E3 mining cadence'] = dict(productions=len(rows), declared_zero=sum(r['declared'] == 0 for r in rows),
                                       refined_zero=sum(r['refined'] == 0 for r in rows))
    rows = [r for e in by('oak_15') for r in cadence(e, 'You swing your axe at the tree.', 3, {'[label,get_logs]': 0})]
    report['E4 woodcutting cadence'] = dict(productions=len(rows), declared_zero=sum(r['declared'] == 0 for r in rows),
                                            refined_zero=sum(r['refined'] == 0 for r in rows),
                                            failures=[r for r in rows if r['refined'] != 0])
    # a scratch compact world for the item, varp and lifecycle scenarios
    layout = W.generate(W.split_seeds('development', 1)[0], tables)
    lt = W.LayoutTables(tables, layout)
    tile = tuple(int(v) + o for v, o in zip(np.argwhere(layout.start)[0], W.ORIGIN))

    def agent_at(kit, levels):
        world = World(lt, seed=1, start_tick=WORLD_START_TICK)
        agent = A.Agent(world)
        agent.start(0, config.TASK_CFG, tile, kit, levels)
        return world, agent
    # E5 Wield: the native sequence of Wield slots, from the same kit
    wield = []
    for e in by('wield'):
        world, agent = agent_at([('bronze_dagger', 1), ('bronze_sword', 1)], {})
        for action, obs in e['decisions']:
            if action == 0:
                continue
            slot = int(obs['actionNote'].split()[-1])
            agent.op_held(slot, 'Wield')
            mine = ([(s, tables.obj_types[o]['name']) for s, o, _ in agent.player.inv.listed()],
                    [(s, tables.obj_types[o]['name']) for s, o, _ in agent.player.worn.listed()])
            theirs = ([(i['slot'], i['name']) for i in obs['inventory']], [(i['slot'], i['name']) for i in obs['equipment']])
            wield.append(mine == theirs)
    report['E5 wield'] = dict(requests=len(wield), equal=sum(wield))
    # E6 Eat: each native eat from the native hitpoints, on the same food
    eats = []
    for e in by('eat'):
        prev = e['start']
        for action, obs in e['decisions']:
            note = obs.get('actionNote') or ''
            if 'Eat' in note:
                food = next(i['name'] for i in prev['inventory'] if i['slot'] == int(note.split()[-1]))
                name = dict(Shrimps='shrimp', Bread='bread')[food]
                world, agent = agent_at([(name, 1)], dict(hitpoints=prev['player']['maxHp']))
                agent.player.levels[3] = prev['player']['hp']
                start = world.tick
                agent.op_held(0, 'Eat')
                said = [x for tick, x in world.chat_log if tick >= start]
                eats.append(dict(native=(obs['player']['hp'], [c['text'] for c in obs['feedback']['chat']]),
                                 sim=(agent.player.hp, said)))
            prev = obs
    gaps_ = []  # ticks between the dispatches of consecutive Eats
    for e in by('eat'):
        dispatched = [prev['tick'] for prev, (_, obs) in zip([e['start']] + [o for _, o in e['decisions']], e['decisions'])
                      if 'Eat' in (obs.get('actionNote') or '')]
        gaps_ += np.diff(dispatched).tolist()
    report['E6 eat'] = dict(eats=len(eats), equal=sum(x['native'] == x['sim'] for x in eats),
                            eat_to_eat_ticks=sorted(set(gaps_)), within_delay_cases=sum(g <= 2 for g in gaps_))
    # E7, E8, E10 and the melee data
    hits = {e['check']: e for e in episodes if e['group'] == 'melee'}
    e7, e7_refined, gaps, lowered = 0, 0, Counter(), 0
    for e in hits.values():
        seq = {ev['sequence']: ev for ev in e['combat']}
        for ev in e['combat']:
            if ev['kind'] == 'damage':
                hit = seq[ev['hit']]
                e7 += ev['tick'] - hit['tick'] == 1 and ev['amount'] == min(hit['damage'], ev['hpBefore'])
                e7_refined += ev['tick'] - hit['tick'] == 1 and ev['amount'] == hit['damage'] and \
                    ev['hpAfter'] == max(0, ev['hpBefore'] - ev['amount'])
            if ev['kind'] == 'agent-hp' and ev['hpAfter'] < ev['hpBefore']:
                lowered += 1
        lives = {}
        for ev in e['combat']:
            if ev['kind'] == 'hit' and ev['hitter'] == 'agent':
                lives.setdefault(ev['life'], []).append(ev['tick'])
        for ticks in lives.values():
            gaps.update(np.diff(ticks).tolist())
    report['E7 hit to damage'] = dict(damage_events=sum(ev['kind'] == 'damage' for e in hits.values() for ev in e['combat']),
                                      equal=e7, refined_equal=e7_refined)
    report['E8 attack cadence'] = dict(gaps=dict(sorted(gaps.items())[:12]), min=min(gaps), mode=gaps.most_common(1)[0][0],
                                       expected=4)
    report['E10 retaliation'] = dict(hp_lowered_in_melee=lowered)
    # E9 lifecycle, native against simulated fights
    native_life = dict(removal=Counter(), respawn=Counter())
    rule = dict(native=Counter(), sim=Counter(), native_without_last_movement=0)

    def arrival(lethal, last_movement):
        """npc_death's interval from the lethal damage tick: npc_arrivedelay, npc_delay(1), npc_del."""
        return 3 + (last_movement == lethal) + 2 * (last_movement == lethal + 1)
    for e in hits.values():
        defeated, removed, lives = {}, {}, []
        for ev in e['combat']:
            if ev['kind'] == 'damage' and ev['hpAfter'] == 0:
                defeated[ev['life']] = ev['tick']
            elif ev['kind'] == 'removed':
                removed[ev['life']] = ev['tick']
                if ev['life'] in defeated:
                    native_life['removal'][ev['tick'] - defeated[ev['life']]] += 1
                    if 'lastMovement' in ev:
                        rule['native'][ev['tick'] - defeated[ev['life']] == arrival(defeated[ev['life']], ev['lastMovement'])] += 1
                    else:
                        rule['native_without_last_movement'] += 1
            elif ev['kind'] == 'life' and ev['previous'] in removed:
                native_life['respawn'][(ev['type'], ev['tick'] - removed[ev['previous']])] += 1
    sim_life = dict(removal=Counter(), respawn=Counter())
    for k in range(30):
        task = GoalEpisode(World(lt, seed=100 + k, start_tick=WORLD_START_TICK), config.TASK_CFG, 'melee', horizon=400,
                           scale=1.0)
        w, removed = task.world, {}
        remove, respawn = w.remove_npc, w.npc_respawn

        def removing(npc, remove=remove, w=w, removed=removed):
            removed[npc.nid] = (w.tick, npc.type)
            defeat = max(t for t, kind, d in w.events if kind == 'npc_defeated' and d['npc'] == npc.nid)
            sim_life['removal'][w.tick - defeat] += 1
            rule['sim'][w.tick - defeat == arrival(defeat, npc.last_movement)] += 1
            remove(npc)

        def respawning(npc, respawn=respawn, w=w, removed=removed):
            sim_life['respawn'][(removed[npc.nid][1], w.tick - removed[npc.nid][0])] += 1
            respawn(npc)
        w.remove_npc, w.npc_respawn = removing, respawning
        task.start(tile, 'bronze_sword', dict(attack=10, strength=10))
        done = False
        while not done:
            _, _, done, _ = task.step(reference_action(task))
    keyed = lambda counts: {k: {(f'type {i[0]}: {i[1]}' if isinstance(i, tuple) else i): n for i, n in v.items()}
                            for k, v in counts.items()}
    report['E9 lifecycle'] = dict(native=keyed(native_life), sim=keyed(sim_life),
                                  removal_inside=set(native_life['removal']) <= set(sim_life['removal']),
                                  respawn_inside=set(native_life['respawn']) <= set(sim_life['respawn']),
                                  rule=dict(native_holds=rule['native'][True], native_deaths=sum(rule['native'].values()),
                                            sim_holds=rule['sim'][True], sim_deaths=sum(rule['sim'].values()),
                                            native_without_last_movement=rule['native_without_last_movement']))
    # E11 depletion and respawn of mined rocks and the oak
    respawns = dict(copper=9, tin=9, iron=19, oak=29)
    brackets, emptied = [], set()
    for e in by('iron_15', 'mine_rolls', 'oak_15'):
        targets, prev = [], e['start']['tick']
        for action, obs in e['decisions']:
            m = re.search(r'loc=(\d+)@(\d+),(\d+)', obs.get('actionNote') or '')
            if m:
                targets.append((prev, (int(m.group(2)), int(m.group(3)))))
            prev = obs['tick']
        for op in e['ops']:
            kind = {436: 'copper', 438: 'tin', 440: 'iron', 1521: 'oak'}[op['stackTail'][1]]
            target = [xy for tick, xy in targets if tick < op['tick']][-1]
            later = [o['tick'] for o in e['ops'] if o['tick'] > op['tick']
                     and [xy for tick, xy in targets if tick < o['tick']][-1] == target]
            seen = []  # the target until its next production: a stump or empty rock seen there is this op's
            for _, obs in e['decisions']:
                if obs['tick'] < op['tick'] or (later and obs['tick'] >= later[0]):
                    continue
                px, pz = obs['player']['x'], obs['player']['z']
                here = [l['id'] for l in obs['locs'] for tx, tz in [target] if (px + l['dx'], pz + l['dz']) == (tx, tz)]
                if here:
                    seen.append((obs['tick'], here[0]))
            gone = {tables.loc_types[ident]['debugname'] for ident in (450, 452, 1355)}
            empty = [tick for tick, ident in seen if tables.loc_types[ident]['debugname'] in gone]
            full = [tick for tick, ident in seen if tables.loc_types[ident]['debugname'] not in gone and empty and tick > empty[0]]
            if empty:
                emptied.add((id(e), op['tick']))
            if empty and full:
                low, high = max(t for t in empty if t < full[0]) - op['tick'], full[0] - op['tick']
                brackets.append((kind, low, high, low < respawns[kind] <= high, op['tick']))
    report['E11 depletion'] = dict(brackets=len(brackets), contain=sum(b[3] for b in brackets),
                                   rows=brackets)
    # E12 gate
    gates = []
    for e in by('gate_timer'):
        opened = closed_again = last_close = None
        prev = e['start']['tick']
        for action, obs in e['decisions']:
            options = {s['option'] for s in obs['actSel'] if s['name'] == 'Gate'}
            if opened is None and 'Close' in options:
                opened = (prev, obs['tick'])  # the request's dispatch, the first observation of it open
            if opened is not None and closed_again is None and 'Close' not in options:
                closed_again = (last_close, obs['tick'])
            if 'Close' in options:
                last_close = obs['tick']
            prev = obs['tick']
        if closed_again:  # the return bracket and the open bracket shifted by 500 ticks overlap
            gates.append(max(closed_again[0] + 1, opened[0] + 500) <= min(closed_again[1], opened[1] - 1 + 500))
    report['E12 gate'] = dict(episodes=len(gates), contain=sum(gates))
    # E13 combat varps: the native steps (start, STYLE_0, Wield) on the simulator
    names = ('com_maxhit', 'com_stabattack', 'com_slashattack', 'com_crushattack', 'com_stabdef', 'damagestyle', 'damagetype')
    capture = {c['group']: c for c in json.loads(Path(manifest).read_text())['captures']}
    native_states = []
    for group in ('combat-stats-before', 'combat-stats'):
        results = json.loads(Path(capture[group]['results']).read_text())
        states = [tuple(v[k] for k in names) for v in results['trials'][0]['varps']]
        native_states.append([s for k, s in enumerate(states) if k == 0 or s != states[k - 1]])
    world, agent = agent_at([('bronze_sword', 1)], dict(attack=10, strength=10, defence=10, hitpoints=12))
    sim_states = [tuple(agent.player.varps[k] for k in names)]
    agent.step(A_STYLE0)
    sim_states.append(tuple(agent.player.varps[k] for k in names))
    agent.step(A_HELD0)
    sim_states.append(tuple(agent.player.varps[k] for k in names))
    report['E13 combat varps'] = dict(native=native_states, sim=sim_states,
                                      equal=native_states[0] == sim_states and native_states[1] == sim_states[1:])
    # D1, D2 gathering rolls inferred from the bouts: a swing starts one (its op fired the tick before);
    # an ore always ends it (the rock depletes), a log only when the oak turns to a stump; a level-up
    # dialog after a log pauses it, and every continue's dispatch in a live bout re-anchors it (the
    # unlock dialog follows the level-up's, so two continues can come in a row).
    def rolls(e, swing, rate, delay, chance, stat, gem, gone):
        anchors = roll_starts(e, swing)
        produced = {op['tick'] - delay[op['scriptName']]: op for op in e['ops']}
        levelups = {tick for tick, text in arrivals(e) if text.startswith('Congratulations, you just advanced')}
        levels = [(e['start']['tick'], next(x['level'] for x in e['start']['skills'] if x['name'] == stat))]
        levels += [(obs['tick'], next(x['level'] for x in obs['skills'] if x['name'] == stat)) for _, obs in e['decisions']]
        out, live = [], False
        for k, (a, kind) in enumerate(anchors):
            if kind == 'continue' and not live:
                continue
            live = True
            until = anchors[k + 1][0] if k + 1 < len(anchors) else e['end']
            tick = a + rate
            while tick < until:
                level = [lv for t_, lv in levels if t_ <= tick][-1]
                success = tick in produced
                out.append((stat_random_value(level, *chance) / 256 * (255 / 256 if gem else 1), success))
                if success and gone(e, produced[tick]):
                    live = False
                    break
                if success and (tick + 1 in levelups or tick + 2 in levelups):
                    break
                tick += rate + 1
        return out

    def depleted(e, op):
        """An ore (a normal rock depletes on every ore, E11), or a log after which the episode shows
        its oak as a stump before the next log from it."""
        return op['scriptName'] != '[label,get_logs]' or (id(e), op['tick']) in emptied
    d = []
    for e in by('iron_15', 'mine_rolls'):
        rock = 'iron' if e['check'] == 'iron_15' else 'tin'
        d += rolls(e, 'You swing your pick at the rock.', 7, {'[label,get_ore_fast]': 0, '[label,get_ore_normal]': 1},
                   tables.meta['content']['compact']['rocks'][rock + 'rock1']['chance'], 'Mining', True, depleted)
    report['D1 mining rolls'] = z_report(d)
    d = []
    for e in by('oak_15'):
        d += rolls(e, 'You swing your axe at the tree.', 3, {'[label,get_logs]': 0},
                   tables.meta['content']['trees']['oaktree']['successchance']['bronze_axe'], 'Woodcutting', False, depleted)
    report['D2 oak rolls'] = z_report(d)
    # D3 melee: per attack, from levels the capture's own damage implies
    rows = {}
    for check, e in sorted(hits.items()):
        weapon = None if 'unarmed' in check else check.split('_', 1)[1].rsplit('_', 1)[0]
        style = int(check[-1])
        world = World(lt, seed=1)
        p = Player(world, 'x')
        if weapon:
            p.worn.items[3] = [tables.obj_id(weapon), 1]
        p.varps['com_mode'] = style
        player_combat_stat(world, p)
        chicken_defence = (tables.npc_types[41]['stats'][NPC_DEFENCE] + 9) * (tables.npc_types[41]['params']['stabdefence'] + 64)
        above = 0
        for ev in e['combat']:
            if ev['kind'] != 'hit' or ev['hitter'] != 'agent':
                continue
            maxhit = p.varps['com_maxhit']
            attack = p.varps[f"com_{('stab', 'slash', 'crush')[p.varps['damagetype']]}attack"]
            rows.setdefault(weapon or 'unarmed', []).append((hit_probability(attack, chicken_defence) * maxhit / (maxhit + 1),
                                                             ev['damage'] >= 1))
            above += ev['damage'] > maxhit
            give_combat_experience(world, p, p.varps['damagestyle'], min(ev['damage'], ev['hp']), 1000)
            p.engine_queue.clear()
            player_combat_stat(world, p)
        rows.setdefault('above_maxhit', []).append(above)
    report['D3 melee'] = {k: (z_report(v) if k != 'above_maxhit' else sum(v)) for k, v in rows.items()}
    if out:
        Path(out).write_text(json.dumps(report, indent=1, default=str) + '\n')
    return report


COMBAT_CHECKS = dict(combat_goblins_d1=('goblin', dict(attack=5, strength=5, defence=1, hitpoints=12)),
                     combat_goblins_d10=('goblin', dict(attack=5, strength=5, defence=10, hitpoints=12)),
                     combat_mugger=('mugger', dict(attack=5, strength=5, defence=5, hitpoints=12)),
                     combat_regen=(None, dict(hitpoints=12)))
RETALIATE_CHECKS = dict(combat_idle=COMBAT_CHECKS['combat_mugger'], combat_walk=COMBAT_CHECKS['combat_mugger'])
RETALIATE_TRIALS = 16
SKILL_DELAY, REGEN_TICKS = 3, 100


def npc_roll(tables, name, levels, weapon='bronze_sword'):
    """C1's simulator probability for an enemy type against a player with these levels and a worn weapon (style row 0):
    (P(damage >= 1), max hit), from the simulator's own formulas."""
    from sim.engine import (ATTACK, DEFENCE, HITPOINTS, NPC_ATTACK, NPC_STRENGTH, ROLL_VARPS, SKILLS, WEARPOS_RHAND, Player,
                            World, player_combat_stat)
    nt = next(v for v in tables.npc_types.values() if v['debugname'] == name)
    param = lambda k: (nt['params'] or {}).get(k, tables.meta['param_defaults'].get(k, 0))
    world = World(tables)
    p = Player(world, 'x')
    for stat, level in levels.items():
        i = SKILLS.index(stat)
        p.levels[i] = p.base_levels[i] = level
    p.worn.items[WEARPOS_RHAND] = (tables.obj_id(weapon), 1)
    player_combat_stat(world, p)
    attack = (nt['stats'][NPC_ATTACK] + 9) * (param('attackbonus') + 64)
    defence = p.varps[f"com_{ROLL_VARPS[param('damagetype')]}def"]
    maxhit = ((nt['stats'][NPC_STRENGTH] + 9) * (param('strengthbonus') + 64) + 320) // 640
    assert ATTACK != DEFENCE != HITPOINTS
    return hit_probability(attack, defence) * maxhit / (maxhit + 1), maxhit


def combat_rules(trials, tables):
    """C1-C5 over trials: dicts of check, attacks [(tick, nid, type name, planned damage)], agent_hits {life: [ticks]},
    hp [(tick, before, after)], eat_ticks, levelup_ticks, base_hp, aggressed_before_request (mugger) and fights."""
    report = {}
    for check, (enemy, levels) in COMBAT_CHECKS.items():
        mine = [t for t in trials if t['check'] == check]
        if not mine or enemy is None:
            continue
        p, maxhit = npc_roll(tables, enemy, levels)
        attacks = [a for t in mine for a in t['attacks'] if a[2] == enemy]
        z = z_report([(p, a[3] >= 1) for a in attacks])
        report[f'C1 {check}'] = dict(z, probability=round(p, 4), maxhit=maxhit, above_max=sum(a[3] > maxhit for a in attacks),
                                    passed=bool(attacks) and abs(z['z']) <= 3 and all(a[3] <= maxhit for a in attacks))
        gaps = Counter()
        for t in mine:
            by = {}
            for tick, nid, *_ in t['attacks']:
                by.setdefault(nid, []).append(tick)
            for ticks in by.values():  # an enemy's attacks within one life: a gap over 60 ticks is a later life
                gaps.update(b - a for a, b in zip(ticks, ticks[1:]) if b - a <= 60)
        report[f'C2 {check}'] = dict(gaps=dict(sorted(gaps.items())), passed=bool(gaps) and min(gaps) >= 4 and gaps.most_common(1)[0][0] == 4)
    mugger = [t for t in trials if t['check'] == 'combat_mugger']
    if mugger:
        report['C3 aggression'] = dict(trials=len(mugger), first=[t['aggressed_before_request'] for t in mugger],
                                       passed=all(t['aggressed_before_request'] for t in mugger))
    gains, bad = [], []
    for t in trials:
        free = [(tick, after - before) for tick, before, after in t['hp'] if after > before and tick not in t['eat_ticks']
                and tick not in t['levelup_ticks']]
        gains += [g for _, g in free]
        ticks = [tick for tick, _ in free]
        bad += [b - a for a, b in zip(ticks, ticks[1:]) if b - a < REGEN_TICKS]
        bad += ['above base'] * t['over_base']
    report['C4 natural healing'] = dict(gains=dict(Counter(gains)), off_timer=bad[:10], off_timer_count=len(bad),
                                        passed=bool(gains) and set(gains) == {1} and not bad)
    spans = []
    for t in trials:
        for ticks in t['agent_hits'].values():
            spans += [b - a for a, b in zip(ticks, ticks[1:]) if any(a < e < b for e in t['eat_ticks'])]
    sword = tables.obj_types[tables.obj_id('bronze_sword')]
    rate = (sword.get('params') or {}).get('attackrate', tables.meta['param_defaults']['attackrate'])
    report['C5 eating in a fight'] = dict(gaps=dict(Counter(spans)), attackrate=rate,
                                          passed=bool(spans) and min(spans) >= rate + SKILL_DELAY)
    return report


def retaliate_rule(trials):
    """C6 (module docstring) over trials: dicts of check, trial, attacks [(tick, nid, type name, planned damage)],
    agent_hits {life: [roll ticks]}, attack_requests, walks [(request tick, arrival tick or None)] and end (the last tick)."""
    if not trials:
        return {}
    broken, offsets, gaps, late = [], Counter(), Counter(), 0
    idle = [t for t in trials if t['check'] == 'combat_idle']
    for t in idle:
        if t['attack_requests']:
            broken.append((t['trial'], 'an Attack request'))
        for ticks in t['agent_hits'].values():
            gaps.update(b - a for a, b in zip(ticks, ticks[1:]))
        if not any(a[0] <= t['end'] - 3 for a in t['attacks']):
            late += 1
            continue
        if not any(t['agent_hits'].values()):
            broken.append((t['trial'], 'attacked, no hit'))
            continue
        first, life = min((ticks[0], life) for life, ticks in t['agent_hits'].items() if ticks)
        nid, last = int(life.split(':')[0]), t['agent_hits'][life][-1]
        fight = [a[0] for a in t['attacks'] if a[1] == nid and a[0] <= last]
        offsets.update((first - a) % 4 for a in fight)
        if not fight or any((first - a) % 4 != 2 for a in fight):
            broken.append((t['trial'], f'first hit {first} against attacks {fight[:4]}'))
    report = {'C6 idle': dict(trials=len(idle), too_late=late, offsets=dict(sorted(offsets.items())),
                              gaps=dict(sorted(gaps.items())), broken=broken[:10],
                              passed=len(idle) > late and not broken and bool(gaps) and min(gaps) >= 4
                              and gaps.most_common(1)[0][0] == 4)}
    walked = [t for t in trials if t['check'] == 'combat_walk']
    inside, attacked, walks = [], 0, 0
    for t in walked:
        spans = [(r, t['end'] + 1 if a is None else a) for r, a in t['walks']]
        walks += len(spans)
        attacked += sum(any(r < a[0] < e for r, e in spans) for a in t['attacks'])
        inside += [(t['trial'], h) for ticks in t['agent_hits'].values() for h in ticks if any(r < h < e for r, e in spans)]
    report['C6 walking'] = dict(trials=len(walked), walks=walks, attacks_inside_walks=attacked, hits_inside_walks=inside[:10],
                                passed=walks > 0 and attacked > 0 and not inside)
    return report


def combat_captures(trajectory):
    """A control_check trajectory's episodes: check, trial, effect rows, observations and the capture's combat events."""
    episodes = {}
    for line in gzip.open(trajectory, 'rt'):
        row = json.loads(line)
        e = episodes.setdefault(row.get('episode'), dict(effects=[], observed=[]))
        if row['kind'] == 'scheduled_episode':
            e.update(check=row['check'], trial=row['trial'])
        elif row['kind'] == 'effect':
            e['effects'].append(row)
        elif row['kind'] == 'observed':
            obs = row['observation']
            e['observed'].append(json.loads(obs) if isinstance(obs, str) else obs)
        elif row['kind'] == 'resource_control' and row['operation'] == 'release':
            e['combat'] = (row['response']['audit'].get('combat') or {}).get('events', [])
    return episodes.values()


def capture_fight(combat):
    """From a capture's combat events: attacks on the agent [(tick, nid, type name, planned damage)], and the agent's hits
    {life: [ticks]}."""
    name = lambda t: None if t is None else 'goblin' if t == 100 else 'mugger' if t == 175 else str(t)
    attacks = [(ev['tick'], ev['nid'], name(ev['type']), ev['damage']) for ev in combat if ev['kind'] == 'agent-hit']
    hits = {}
    for ev in combat:
        if ev['kind'] == 'hit' and ev['hitter'] == 'agent':
            hits.setdefault(ev['life'], []).append(ev['tick'])
    return attacks, hits


def native_retaliate_trials(trajectory):
    """C6's trial dicts from a control_check combat_idle or combat_walk trajectory. A walk runs from the observation its
    request answered to the first observation that reports its end."""
    out = []
    for e in combat_captures(trajectory):
        if e.get('check') not in RETALIATE_CHECKS or 'combat' not in e:
            continue
        attacks, hits = capture_fight(e['combat'])
        ticks, walks, walking = [o['tick'] for o in e['observed']], [], None
        for i, eff in enumerate(e['effects']):
            if A.click_slot(eff['action']) is not None:
                walks.append([ticks[i], None])
                walking = eff['action']
            if walking is not None and (eff.get('ended') or {}).get('action') == walking and i + 1 < len(ticks):
                walks[-1][1], walking = ticks[i + 1], None
        out.append(dict(check=e['check'], trial=e['trial'], attacks=attacks, agent_hits=hits, walks=walks, end=ticks[-1],
                        attack_requests=sum((eff.get('meaning') or {}).get('option') == 'Attack' for eff in e['effects'])))
    return out


def native_combat_trials(trajectory):
    """C1-C5's trial dicts from a control_check combat_* trajectory (its capture, effect rows and observations)."""
    out = []
    for e in combat_captures(trajectory):
        if e.get('check') not in COMBAT_CHECKS or 'combat' not in e:
            continue
        attacks, hits = capture_fight(e['combat'])
        hp = [(ev['tick'], ev['hpBefore'], ev['hpAfter']) for ev in e['combat'] if ev['kind'] == 'agent-hp']
        # each decision spans (the tick of the observation it answered, the tick of the next]: an Eat request, or a
        # Hitpoints base level rise, explains a gain in it. observed[0] is the reset's observation, before any decision.
        windows, base = [], lambda o: next((sk['baseLevel'] for sk in o.get('skills', []) if sk['name'] == 'Hitpoints'), None)
        for eff, before, after in zip(e['effects'], e['observed'], e['observed'][1:]):
            windows.append((before['tick'], after['tick'], (eff.get('meaning') or {}).get('option') == 'Eat',
                            base(after) is not None and base(before) is not None and base(after) > base(before)))
        inside = lambda t, k: any(a < t <= b and w[k] for a, b, *w in windows)
        first_request = next((eff['tick'] for eff in e['effects'] if (eff.get('meaning') or {}).get('option') == 'Attack'), None)
        first_attack = next((t for t, _, n, _ in attacks if n == 'mugger'), None)
        out.append(dict(check=e['check'], trial=e['trial'], attacks=attacks, agent_hits=hits, hp=hp,
                        eat_ticks={t for t, b, a in hp if a > b and inside(t, 0)},
                        levelup_ticks={t for t, b, a in hp if a > b and inside(t, 1)},
                        over_base=sum(o['player']['hp'] > o['player']['maxHp'] for o in e['observed']),
                        aggressed_before_request=first_attack is not None and (first_request is None or first_attack <= first_request)))
    return out


def sim_combat_trials(tables, checks=tuple(COMBAT_CHECKS), trials=3, key=17):
    """The same checks in the simulator: each check's declared levels and items from a development start on its own
    compact layout (the goblin camp of sim/combat.py; for the mugger check one mugger 6 tiles from the start), played by
    baseline.control_check's own Mechanic for the horizon, and read from the world's events."""
    from baseline.control_check import MECHANICS, Mechanic
    from sim import combat
    from sim.engine import HITPOINTS, World
    from sim.task import WORLD_START_TICK, GoalEpisode, draw_start
    from sim.world import LayoutTables, ORIGIN, generate, split_seeds
    pool = [generate(seed, tables) for seed in split_seeds('development', 6, 500)]
    out = []
    mugger_only = ('combat_mugger', *RETALIATE_CHECKS)
    for check in checks:
        spec = MECHANICS[check]
        levels = dict(woodcutting=1, fishing=1, mining=1, attack=1, strength=1, defence=1, hitpoints=10)
        levels.update({**COMBAT_CHECKS, **RETALIATE_CHECKS}[check][1])
        items = [(name, int(n)) for name, n in (pair.split(':') for pair in spec['items'].split(','))]
        for k in range(trials):
            start = draw_start(key, 3, k, pool)
            layout = pool[start['layout']]
            sx, sz = start['tile'][0] - ORIGIN[0], start['tile'][1] - ORIGIN[1]
            muggers = []
            if check in mugger_only:
                region = combat.outside(layout)
                near = [(int(x), int(z)) for x, z in np.argwhere(region) if max(abs(x - sx), abs(z - sz)) == 6]
                muggers = [near[k % len(near)]]
            npcs = combat.combat_npcs(layout, tables, muggers)
            if check in (*mugger_only, 'combat_regen'):  # no goblins: the mugger alone, or nothing to fight
                npcs = [n for n in npcs if tables.npc_types[n['type']]['debugname'] != 'goblin']
            world = World(LayoutTables(tables, layout), seed=start['world_seed'], start_tick=WORLD_START_TICK, npcs=npcs,
                          exclusions=combat.COMBAT['exclusions'])
            task = GoalEpisode(world, config.TASK_CFG, 'melee', horizon=combat.COMBAT['horizon'], scale=1.0, combat=True)
            task.start(start['tile'], 'bronze_sword', levels, kit=items)
            player, p = Mechanic(spec), task.agent.player
            player.reset(k)

            class Transport:  # the regen check's fault injection, as the native test hook applies it
                def _rpc(self, request, timeout_s):
                    world.player_damage(p, 3)
                    return dict(hp=p.hp)
            task.transport = Transport()
            for _ in range(spec['horizon'] // 2):
                if task.world.tick - task.start_tick >= spec['horizon']:
                    break
                action = player.act(task.projected, task)
                if task.step(int(action or 0))[2]:
                    break
            player.finish(task.raw)
            events = [ev for ev in world.events[task.event_start:]]
            names = {n.nid: tables.npc_types[n.base]['debugname'] for n in world.npcs}
            attacks = [(t, d['source'], names.get(d['source']), d['planned']) for t, kind, d in events
                       if kind == 'agent_hp' and d['cause'] == 'damage']
            hits = {}
            for t, kind, d in events:
                if kind == 'npc_damage' and d['source'] == task.agent.name:
                    hits.setdefault(f"{d['npc']}:{d['life']}", []).append(t)
            hp = [(t, d['before'], d['after']) for t, kind, d in events if kind == 'agent_hp']
            first_request = next((t for t, kind, d in events if kind == 'npc_damage' and d['source'] == task.agent.name), None)
            first_attack = next((t for t, _, n, _ in attacks if n == 'mugger'), None)
            out.append(dict(check=check, trial=k, attacks=attacks, agent_hits=hits, hp=hp,
                            eat_ticks={t for t, kind, d in events if kind == 'agent_hp' and d['cause'] == 'eat'},
                            levelup_ticks={t for t, kind, d in events if kind == 'agent_hp' and d['cause'] == 'levelup'},
                            over_base=int(p.levels[HITPOINTS] > p.base_levels[HITPOINTS]),
                            aggressed_before_request=first_attack is not None and (first_request is None or first_attack <= first_request),
                            attack_requests=0, walks=player.facts.get('walks', []), end=world.tick))
    return out


def combat_check(manifest, out=None):
    """C1-C6 (module docstring) on the manifest's native combat captures and on the same checks in the simulator. A
    manifest without a combat_idle or combat_walk capture reports C6 for the simulator alone."""
    tables = Tables.load()
    spec = json.loads(Path(manifest).read_text())
    native, retaliate = [], []
    for capture in spec['captures']:
        path = Path(capture['trajectory'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != capture['sha256']:
            raise ValueError(f'{path} differs from its manifest entry')
        native += native_combat_trials(path)
        retaliate += native_retaliate_trials(path)
    simulated = sim_combat_trials(tables)
    # a simulator hit's event comes a tick after its roll, the tick the native capture records
    idle_walk = [dict(t, agent_hits={life: [h - 1 for h in ticks] for life, ticks in t['agent_hits'].items()})
                 for t in sim_combat_trials(tables, checks=tuple(RETALIATE_CHECKS), trials=RETALIATE_TRIALS)]
    report = dict(native={**combat_rules(native, tables), **retaliate_rule(retaliate)},
                  simulator={**combat_rules(simulated, tables), **retaliate_rule(idle_walk)},
                  trials=dict(native=Counter(t['check'] for t in native + retaliate),
                              simulator=Counter(t['check'] for t in simulated + idle_walk)))
    report['passed'] = all(r['passed'] for side in ('native', 'simulator') for r in report[side].values())
    if out:
        Path(out).write_text(json.dumps(report, indent=1, default=str) + '\n')
    return report


def z_report(pairs):
    """Observed against expected successes of independent trials with probabilities p: z."""
    p = np.array([q for q, _ in pairs], float)
    hits = sum(h for _, h in pairs)
    sd = float(np.sqrt((p * (1 - p)).sum())) or 1.0
    return dict(trials=len(pairs), observed=int(hits), expected=round(float(p.sum()), 1),
                z=round((hits - float(p.sum())) / sd, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='check', required=True)
    p = sub.add_parser('policy', help='simulator evaluations against the native scores')
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path)
    r = sub.add_parser('replay', help='native evaluation episodes replayed by meaning')
    r.add_argument('--evaluations', type=Path, required=True, help='the directory of act-v3 evaluations')
    r.add_argument('--pattern', default='*', help='evaluation directories under --evaluations')
    r.add_argument('--limit', type=int, help='at most this many episodes')
    r.add_argument('--held-out', action='store_true', help='the held-out 262k evaluations (final check only)')
    r.add_argument('--out', type=Path)
    ch = sub.add_parser('chase', help='an ACT on a wandering npc, tick by tick against a native recording')
    ch.add_argument('--evaluations', type=Path, required=True, help='the directory of chase recordings')
    ch.add_argument('--out', type=Path)
    m = sub.add_parser('mechanics', help='the compact activities against native captures (module docstring rules)')
    m.add_argument('--manifest', type=Path, required=True)
    m.add_argument('--out', type=Path)
    cb = sub.add_parser('combat', help='sustained-combat-v1 mechanics against native captures (rules C1-C6)')
    cb.add_argument('--manifest', type=Path, required=True)
    cb.add_argument('--out', type=Path)
    c = sub.add_parser('coverage', help='native training episodes that use only simulated content, by quarter')
    c.add_argument('--seeds', nargs='+', default=['s101', 's202', 's303'], help='runs under logs/wt/act/logs/goal')
    c.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.check == 'policy':
        policy_check(args.runs, args.out)
    elif args.check == 'replay':
        replay(args.evaluations, args.pattern, args.out, args.limit, args.held_out)
    elif args.check == 'chase':
        chase(args.evaluations, args.out)
    elif args.check == 'coverage':
        coverage(args.seeds, out=args.out)
    elif args.check == 'mechanics':
        print(json.dumps(mechanics(args.manifest, args.out), indent=1, default=str))
    elif args.check == 'combat':
        print(json.dumps(combat_check(args.manifest, args.out), indent=1, default=str))


if __name__ == '__main__':
    main()
