"""Goal tasks over the simulator, with ResourceEpisode's interface and rules.

GoalEpisode is one goal episode of one agent in a world, armed after its setup:

- `reset(seed)`: the goal-v1 Draynor task. Setup is baseline.scripted.prepare_capacity itself,
  run through the simulated agent, so the start state is made by the same steps as the native one.
- `start(tile, weapon, levels)`: the four-goal task (goal-activity-v2) in a compact world
  (sim/world.py): START_KIT plus the weapon carried, not worn, the levels (setStats) and the start
  tile, all drawn by `draw_start` before and independently of the goal. As native's declared start
  (baseline/scripted.py prepare_declared), setup then takes one ordinary step, STYLE_0: setStats
  writes levels without [changestat], and re-selecting a new player's style row 0 runs
  [proc,set_attackstyle] -> ~player_combat_stat, so the combat varps follow the set levels at arm.

With `combat`, the melee goal is sustained-combat-v1 (sim/combat.py, goal-activity-v3): its target, eligible enemies
and horizon, and `start`'s `kit`, `hp` (hitpoints at arm, written as the reset writes levels) and `wield` (a Wield of the
carried weapon after STYLE_0, a second ordinary setup step). wood, fish and mine are unchanged.

The episode lasts `horizon` native ticks from arm; its holdings are taken at the end of the cycle
that reaches the end tick, as the native capture closes (env-ts/engine/resource-capture.ts), or at
death; nothing after that terminal cycle is simulated for the episode: its last decision
stops there. Rewards are paid as the goal's certified progress changes (goal items held, or the audit's
kills so far) and settled to the audited return: the rewards of an episode sum to it. The return
is decided by sim.audit from the world's events, not by the running payment, so the payment is
checked against an independent account.

A step the simulator cannot model (sim.engine.Unsupported) closes the episode and re-raises
: it is never a done, a zero reward, a truncation or a stale observation.
"""
from copy import deepcopy

import numpy as np

from baseline.resource_task import GOAL_FAMILIES, GOAL_TARGETS, GOAL_VEC, GOALS
from baseline.scripted import AXE, prepare_capacity
from feedback_contract import episode_start
from obs import project
from sim.agent import A_HELD0, A_STYLE0, Agent
from sim.audit import audit, kills
from sim.combat import COMBAT, enemy_types
from baseline.resource_task import MELEE_ENEMIES
from sim.engine import HITPOINTS, Rng, Unsupported, World
from sim.spec import START_KIT, START_LEVELS, START_WEAPONS
from sim.world import ORIGIN


class SimTransport:
    def __init__(self, agent):
        self.agent = agent

    def step(self, action):
        return self.agent.step(int(action))


# Purposes of the keyed draws of an episode's start (draw_start).
R_LAYOUT, R_TILE, R_WEAPON, R_LEVEL, R_WORLD = range(1, 6)
# A compact episode's fresh world starts here, so the four ticks of its reset end at tick 10, where a
# fresh native world shows its reset observation (sim/check.py); the STYLE_0 setup step (K_TICKS plus
# 0-1 pace jitter) arms it at 12 or 13, as natively. At map_clock < 8 a new account's %lastcombat_pvp
# of 0 still refuses attacks (player_in_combat_check); the random-event check at tick 500 lies beyond
# the episode.
WORLD_START_TICK = 6


FAMILIES = ('wood', 'fish', 'mine')  # goal families whose production and pickups are counted (info `made`)


def object_families(tables):
    """Per obj id (0 to the largest), the FAMILIES index of the goal family it or its note belongs to, else -1."""
    M = max(tables.obj_types) + 1
    family = np.full(M, -1, np.int32)
    for k, goal in enumerate(FAMILIES):
        for obj in GOAL_FAMILIES[goal]:
            for o in (obj, tables.obj_types.get(obj, {}).get('certlink', -1)):
                if o is not None and 0 <= o < M:
                    family[o] = k
    return family


def produced(events, family):
    """Goal items produced per family in `events`, as the step info's `made` counts them (sim/vec.py)."""
    made = np.zeros(len(FAMILIES), np.int32)
    for _, kind, data in events:
        k = family[data.get('obj', -1)] if kind == 'produce' else -1
        if k >= 0:
            made[k] += data['count']
    return made


def draw_start(key, slot, episode, layouts, layout=None):
    """The physical start of episode `episode` of world slot `slot`: a layout of the pool, one of
    its start tiles (in x-major order), a START_WEAPONS weapon, START_LEVELS levels (each stat
    drawn uniformly from its list, in the dict's order) and the world's seed. Keyed draws
    (engine.Rng on (key, slot, episode, purpose)), so a batched port reproduces them; the goal
    takes no part. `layout` (an index of `layouts`) replaces the layout draw; each other draw has its
    own purpose, so it is the same either way."""
    h = Rng(key).prefix(slot, episode)
    draw = lambda *k: Rng.draw(h, *k)
    layout = int(draw(R_LAYOUT) * len(layouts)) if layout is None else layout
    tiles = np.argwhere(layouts[layout].start)
    x, z = tiles[int(draw(R_TILE) * len(tiles))]
    return dict(layout=layout, tile=(ORIGIN[0] + int(x), ORIGIN[1] + int(z)),
                weapon=START_WEAPONS[int(draw(R_WEAPON) * len(START_WEAPONS))],
                levels={stat: choices[int(draw(R_LEVEL, i) * len(choices))] for i, (stat, choices) in enumerate(START_LEVELS.items())},
                world_seed=int(draw(R_WORLD) * (1 << 24)))


class GoalEpisode:
    """One agent in one simulated world; reset() or start() begins an episode there."""

    def __init__(self, world: World, cfg, goal, horizon=240, scale=40, record=None, name='wd1a0', combat=False, target=None):
        if goal not in GOALS:
            raise ValueError('invalid goal')
        self.combat = combat and goal == 'melee'
        if self.combat and horizon != COMBAT['horizon']:
            raise ValueError(f"{COMBAT['version']} has a {COMBAT['horizon']}-tick horizon")
        self.enemies = enemy_types(world.t) if self.combat else MELEE_ENEMIES
        self.world, self.cfg, self.goal, self.horizon, self.scale = world, dict(cfg), goal, horizon, scale
        self.agent = Agent(world, name)
        self.record = record or (lambda kind, **data: None)
        self.family = set(GOAL_FAMILIES.get(goal, ()))
        for base in list(self.family):
            note = world.t.obj_types.get(base, {}).get('certlink', -1)
            if note is not None and note >= 0:
                self.family.add(note)
        # target: a gathering task's own return cap (the supply study's production cap, sim/search.py SUPPLY)
        self.target = COMBAT['target'] if self.combat else target or GOAL_TARGETS[goal]
        self.raw = self.projected = None
        self.active = False

    # ---- held measure (ResourceEpisode._carried/_track_bank/_held)
    def carried(self):
        return sum(c for _, o, c in self.agent.player.inv.listed() if o in self.family)

    def track_bank(self):
        carried = self.carried()
        if self.raw.get('bankOpen') or self.bank_was_open:
            self.banked += self.carried_before - carried
        self.carried_before = carried
        self.bank_was_open = bool(self.raw.get('bankOpen'))

    def held(self):
        if self.goal == 'melee':
            count = kills(self.world.events[self.event_start:], self.agent.name, self.enemies)
        else:
            count = self.banked + self.carried_before
        return min(count, self.target) / self.target * self.scale

    def budget(self):
        tick = min(self.world.tick, self.end) if self.terminal is None else self.terminal['tick']
        status = 'active' if self.terminal is None else self.terminal['reason']
        return tick, status

    def project(self):
        result = project(self.raw)
        tick, _ = self.budget()
        result['budget_vec'] = np.asarray([(self.end - tick) / self.horizon, self.horizon / 10000], np.float32)
        result['critic_vec'] = np.asarray([self.banked / self.scale], np.float32)
        result['goal_vec'] = np.asarray([g == self.goal for g in GOAL_VEC], np.float32)
        self.projected = result
        return result

    # ---- the native clock: holdings at the end of the cycle that reaches the end tick
    def _watch(self):
        world, episode = self.world, self
        cycle = world.cycle

        def watched():
            cycle()
            if episode.terminal is None and episode.armed:
                p = episode.agent.player
                if p.hp <= 0 or p.dead:
                    episode.terminal = dict(reason='death', tick=world.tick, holdings=episode.holdings())
                elif world.tick >= episode.end:
                    episode.terminal = dict(reason='horizon', tick=episode.end, holdings=episode.holdings())
        world.cycle = watched
        self._unwatch = lambda: setattr(world, 'cycle', cycle)

    def holdings(self):
        p = self.agent.player
        return dict(inventory=[(s, o, c) for s, o, c in p.inv.listed()], bank=[(s, o, c) for s, o, c in p.bank.listed()],
                    worn=[(s, o, c) for s, o, c in p.worn.listed()])

    def reset(self, seed=0, transport=None):
        """The goal-v1 Draynor start; transport replaces SimTransport for the setup steps (sim.check
        forces native ticks)."""
        self.terminal, self.armed = None, False
        raw = self.agent.reset(seed, deepcopy(self.cfg))
        self.record('reset', seed=seed, observation=raw)
        raw = prepare_capacity(transport or SimTransport(self.agent), raw, self.record, 'both-tools', AXE)
        return self._arm(raw)

    def start(self, tile, weapon, levels, seed=0, kit=None, hp=None, wield=False):
        """The four-goal compact-world start (module docstring); kit, hp and wield are the combat start's."""
        self.terminal, self.armed = None, False
        if (kit is not None or hp is not None or wield) and not self.combat:
            raise ValueError('only a combat start sets its kit, hitpoints or worn weapon')
        kit = list(START_KIT) + [(weapon, 1)] if kit is None else list(kit)
        raw = self.agent.start(seed, deepcopy(self.cfg), tile, kit, levels)
        if hp is not None:
            p = self.agent.player
            if not 1 <= hp <= p.base_levels[HITPOINTS]:
                raise ValueError(f'hitpoints {hp} outside 1..{p.base_levels[HITPOINTS]}')
            p.levels[HITPOINTS] = hp
        self.record('start', tile=tile, weapon=weapon, levels=levels, kit=kit, hp=hp, observation=raw)
        raw = self.agent.step(A_STYLE0)  # the declared start's one setup step (module docstring)
        self.record('setup', action=A_STYLE0, observation=raw)
        if wield:
            name = self.world.t.obj_types[self.world.t.obj_id(weapon)]['name']
            slot = next((k for k, h in enumerate(raw['heldSel']) if h['name'] == name and h['option'] == 'Wield'), None)
            if slot is None:
                raise ValueError(f'no Wield offered for {name} at setup')
            raw = self.agent.step(A_HELD0 + slot)
            self.record('setup', action=A_HELD0 + slot, observation=raw)
        return self._arm(raw)

    def _arm(self, raw):
        self.raw = episode_start(raw)
        self.start_tick, self.end = self.world.tick, self.world.tick + self.horizon
        self.event_start = len(self.world.events)
        self.initial = self.holdings()
        self.arm_hp = self.agent.player.hp
        self.banked, self.carried_before, self.bank_was_open = 0, self.carried(), bool(self.raw.get('bankOpen'))
        if self.carried_before:
            raise ValueError('goal items held at arm')
        self.paid, self.decisions = 0.0, 0
        self.armed, self.active = True, True
        self._watch()
        self.agent.ended = lambda: self.terminal is not None  # the episode ends at its terminal cycle
        return self.project()

    def step(self, action):
        if not self.active:
            raise RuntimeError('step on a closed episode')
        if not (type(action) is int and 0 <= action < 96 and self.projected['mask'][action]):
            raise ValueError(f'invalid/masked action {action}')
        self.decisions += 1
        if self.terminal is None:
            try:
                self.raw = self.agent.step(action)
            except Unsupported:
                self.active = False
                self._unwatch()
                raise
            self.track_bank()
        result = self.project()
        if self.terminal is None:
            held = self.held()
            reward = (held - self.paid) / self.scale
            self.paid = held
            return result, reward, False, {}
        report = audit(self.world.events[self.event_start:], self.initial, self.terminal, self.goal, self.world.t,
                       self.agent.name, self.enemies)
        outcomes = report['kills']['lower'] if self.goal == 'melee' else report['retained']
        total = min(outcomes, self.target) / self.target if report['survived'] else 0.0
        reward = total - self.paid / self.scale
        self.active = False
        self._unwatch()
        info = dict(report=report, goal=self.goal, outcomes=outcomes,
                    episode=dict(r=total, l=self.decisions, native_ticks=self.terminal['tick'] - self.start_tick))
        return result, reward, True, info
