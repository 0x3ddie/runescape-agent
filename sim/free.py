"""free-play-v1's controls and its reachability soak.

The controls are yardsticks on a free-play life (sim/task.py FreePlayEpisode): WAIT, uniform random legal play, each
single-activity reference, the exploit scripts and a fixed-timer rotation. Each reads the world to choose (privileged)
and acts only through the published legal actions. Their actions are never labels. The study's references are the forms
chosen on 11 November; sim/run.py reference_action, which they build on, stays as it is for every other task and
caller:

- `wood`, `fish`, `mine`: reference_action for that goal, which drops its own product at a full inventory, turning
  inventory pages to find one (no turn back to page 0 afterwards);
- `melee`: sim/combat.py Reference('full') on the goblins, its style set to train the lowest of attack, strength and
  defence between fights, and its first rule (fight back an attacker) kept to attackers it can reach (ReachReference);
- the exploit scripts: `chickens` attacks chickens only (reference_action's melee targets), each new fight in the style of
  the lowest of attack, strength and defence, and fights back an npc that attacks it when it can reach it and that npc's
  Attack is offered (single-way combat refuses a chicken while one does); `once` gains XP once in each skill of S
  (woodcutting, mining, fishing, then a fight in the attack, strength and defence styles, hitpoints coming with them; a
  phase not done in ONCE_TICKS is passed), then waits; `eat` eats every food at once, then waits;
- `rotation` (Claim R's bar): ROTATION_ORDER's references, each for `period` ticks from arm in turn. Its gatherers drop
  any gatherer's product, fight back an attacker they can reach (a handover can come in the middle of a fight) and open
  a closed pen gate when the player stands in a pen (a melee phase can end there, in a chase that runs on). The period is a setting with no
  default: the study freezes it on development records. This definition of 11 November was a proposal until the
  freeze.

"Reach" is a walk over the live collision with the gates as they stand (Reach): an offered, legal Attack does not show
that a hit can land, since an npc can keep the player as its target from across a closed pen gate's fence. A melee
control whose own target is out of reach (a chase begun with the gate open) opens the closed gate. The agent goes on with
a running option when the ACT slot that started it is chosen again, whatever that slot holds now; so an ACT on such a
slot with another target is preceded by an action that ends the option (issue).

`soak` is the reachability soak: random legal play and the controls on development records of
village-world-v1 (sim/combat.py VILLAGE), one fresh world and one life a record, writing one JSONL row a life: the
record, the actor, ticks, decisions, death, the integer level gain by skill and each unsupported site with its evidence.
An unsupported site ends that life there; it is reported, never a death, a horizon or a score. It runs on
a local machine and it trains nothing.

    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.free soak --world village-world-v1 --muggers 0 1 --xp-rate 1 --ticks 36000 \
        --first 4000 --records 64 --key 223 --seed 4253 --periods 1500 4500 9000 --random-decisions 1000000 \
        --processes 6 --output logs/free-play-sim/soak

village-world-v2 (sim/combat.py VILLAGE_V2) plays over draynor-nc2.npz (world_tables), without muggers, on its own
development layouts; `measure` reads what a saved pool of layouts holds (measure_pool):

    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.free soak --world village-world-v2 --muggers 0 --xp-rate 1 --ticks 36000 \
        --first 7000 --records 64 --key 239 --seed 4253 --periods 4500 --random-decisions 1000000 --processes 6 \
        --output logs/world-v2-sim/soak
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.free measure --pool logs/sim/world/train-128-20000-v2.npz \
        --output logs/world-v2-sim/pool-measure.json
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np

from act_contract import CLICK_ROWS
from sim import paths
from sim.agent import (A_ACT0, A_CLOSE_IF, A_DIALOG_CONTINUE, A_DROP0, A_HELD0, A_INV_NEXT, A_INV_PREV, A_STYLE0, A_WAIT,
                       N_INV_SEL, affordance_key)
from sim.combat import Reference, outside, village_of
from sim.engine import ATTACK, DEFENCE, HITPOINTS, SKILLS, STRENGTH, WEARPOS_RHAND, Npc, Unsupported, weapon_styles
from sim.run import click_toward, family_ids, reference_action, reference_targets
from sim.world import ORIGIN, PEN_BARRIERS, PEN_DOOR, flood

FOOD = ('Shrimps', 'Bread')
ONCE_TICKS = 2000                                       # a phase of `once` not done in this many ticks is passed
ROTATION_ORDER = ('melee', 'wood', 'fish', 'mine')   # the four-way timer of 11 November (9,000 ticks a member there)
STYLE_SKILLS = ('attack', 'strength', 'defence')        # damagestyle 0, 1, 2 (sim/engine.py give_combat_experience)
FIRST_WORLD = 'village-world-v1'  # the world of a Life whose caller names none


def view(task, goal):
    """The life as sim/run.py reference_action reads a goal task: its agent, its world and the reference's goal."""
    return SimpleNamespace(agent=task.agent, world=task.world, goal=goal)


def attackers(task):
    p = task.agent.player
    return [n for n in task.world.npcs if n.is_valid() and n.target is p]


def style_row(task, skill, between=True):
    """The style action (A_STYLE0 + row) whose damage style trains `skill`, when another row is selected and that row is
    legal; with `between`, only between fights (no target, no attacker, no option running). None otherwise, and with no
    weapon worn (a reference wields first)."""
    p, obs = task.agent.player, task.agent.last_observation
    if p.worn.items[WEARPOS_RHAND] is None or (between and (p.target is not None or attackers(task) or obs.get('optionBusy'))):
        return None
    rows = [k for k, d in enumerate(weapon_styles(task.world, p.worn.items[WEARPOS_RHAND][0])['damagestyle'])
            if d == STYLE_SKILLS.index(skill)]
    if rows and p.varps['com_mode'] != rows[0] and obs['mask'][A_STYLE0 + rows[0]]:
        return A_STYLE0 + rows[0]
    return None


def lowest_skill(task):
    """The lowest of attack, strength and defence by base level (ties in that order)."""
    p = task.agent.player
    levels = dict(attack=p.base_levels[ATTACK], strength=p.base_levels[STRENGTH], defence=p.base_levels[DEFENCE])
    return min(STYLE_SKILLS, key=lambda s: (levels[s], STYLE_SKILLS.index(s)))


def drop_own(task, family):
    """At a full inventory, DROP_k of the first item of `family` on the inventory page, else the page turn toward the
    nearest page that holds one. None when there is nothing to do: no turn back to page 0 after a drop (the
    study's form asks for a drop and page turns; a control that needs page 0 turns back itself)."""
    p, obs = task.agent.player, task.agent.last_observation
    mask, page = obs['mask'], obs['invPage']
    if p.inv.freespace() == 0:
        for k, item in enumerate(obs['invSel'][:N_INV_SEL]):
            if item['slot'] != -1 and item['id'] in family and mask[A_DROP0 + k]:
                return A_DROP0 + k
        pages = {slot // N_INV_SEL for slot, obj, _ in p.inv.listed() if obj in family}
        if any(q > page for q in pages) and mask[A_INV_NEXT]:
            return A_INV_NEXT
        if any(q < page for q in pages) and mask[A_INV_PREV]:
            return A_INV_PREV
    return None


class Reach:
    """Steps from the player's tile over the layout's live collision (gates as they stand, npcs left out), -1 where
    unreached; the step masks are kept until a loc changes, the steps until the player moves."""
    def __init__(self):
        self.world = self.version = self.at = None

    def steps(self, task):
        w, p = task.world, task.agent.player
        W, H = w.t.W, w.t.H
        if self.world is not w or self.version != w.loc_version:
            flags = np.array([[w.flags(p.level, ORIGIN[0] + x, ORIGIN[1] + z, npc_bits=False) for z in range(H)]
                              for x in range(W)], np.int64)
            self.world, self.version, self.masks, self.at = w, w.loc_version, paths.step_masks(flags), None
        if self.at != (p.x, p.z):
            seed = np.zeros((W, H), bool)
            seed[p.x - ORIGIN[0], p.z - ORIGIN[1]] = True
            self.at, self.dist = (p.x, p.z), flood(self.masks, seed)
        return self.dist

    def reachable(self, task, npcs):
        """The npcs of `npcs` that the player can walk to a hit of: a tile from which a hit lands on the npc
        (World.entity_reaching) is reachable. An offered Attack is not enough: across a closed pen gate's fence an npc can
        keep the player as its target from a diagonal tile, offered and legal, where no hit lands."""
        if not npcs:
            return []
        dist, w = self.steps(task), task.world
        inside = lambda x, z: 0 <= x < dist.shape[0] and 0 <= z < dist.shape[1] and dist[x, z] >= 0
        return [n for n in npcs if any(inside(x - ORIGIN[0], z - ORIGIN[1]) for x, z in w.entity_reaching(n.level, n.x, n.z))]

    def attackers(self, task):
        """The npcs attacking the player that it can reach (reachable)."""
        return self.reachable(task, attackers(task))

    def chasing(self, task):
        """Whether the player's target is an npc it cannot reach: a chase begun with a pen gate open, and the gate closed
        since, runs on (the option keeps moving) and holds a reference that waits while an option runs."""
        t = task.agent.player.target
        return isinstance(t, Npc) and t.is_valid() and not self.reachable(task, [t])


def fight_back(task, reach):
    """ACT Attack on an attacker the player can reach (Reach.attackers) whose Attack is offered, matched by the slot's
    npc (a chicken can share a goblin's tile); WAIT while the player already fights one; None when there is none."""
    a, p, obs = task.agent, task.agent.player, task.agent.last_observation
    hit = reach.attackers(task)
    if not hit:
        return None
    if obs.get('optionBusy') and p.target in hit:
        return A_WAIT
    for k, sel in enumerate(obs['actSel']):
        if obs['mask'][A_ACT0 + k] and sel['option'] == 'Attack' and k < len(a.act_targets) and a.act_targets[k] in hit:
            return A_ACT0 + k
    return None


def swallowed(task, action):
    """Whether sim/agent.py would go on with the running option in place of `action`: an ACT on the slot that started
    the running option (the agent goes on with an option whose ACT slot is chosen again) that now holds another target:
    another npc than the one the player has as its target, or another loc or ground item than the option's."""
    a, obs = task.agent, task.agent.last_observation
    k = action - A_ACT0
    if not (0 <= k < len(obs['actSel']) and obs.get('optionBusy') and obs.get('optionAction') == action and a.option):
        return False
    sel = obs['actSel'][k]
    if sel['kind'] == 'npc':
        return k >= len(a.act_targets) or a.act_targets[k] is not a.player.target
    return (a.option.get('act') or {}).get('key') != affordance_key(sel)


def issue(control, task, action):
    """`action`, unless it would be swallowed: then a legal action that ends the running option first, the click toward
    the ACT's target, else an inventory page turn (no game effect), counted in control.enders; the ACT comes at the next
    decision. (A gate's Open in the slot of a running chase was lost this way in 4 of 12 soak lives of the melee
    reference.)"""
    if not swallowed(task, action):
        return action
    obs, control.enders = task.agent.last_observation, control.enders + 1
    sel = obs['actSel'][action - A_ACT0]
    end = click_toward(task, {(sel['x'], sel['z'])})
    if end != A_WAIT:
        return end
    return next((turn for turn in (A_INV_NEXT, A_INV_PREV) if obs['mask'][turn]), action)


def in_pen(task, memo):
    """Whether the player stands outside its layout's start component (sim/combat.py outside, every gate closed): inside
    a chicken pen. `memo` keeps the component of the last layout."""
    w, p = task.world, task.agent.player
    if memo.get('layout') is not w.t.layout:
        memo.update(layout=w.t.layout, start=outside(w.t.layout))
    return not memo['start'][p.x - ORIGIN[0], p.z - ORIGIN[1]]


def closed(task, names):
    """The Open targets (kind, x, z, option) of the active map locs of `names`: a pen's closed barriers."""
    w = task.world
    return {('loc', loc.x, loc.z, 'Open') for loc in w.locs if loc.active and w.t.loc_types[loc.type]['debugname'] in names}


FROZEN = 'village-world-v1'  # its controls are frozen as first built: a later fix leaves them as they are


def dialog_step(task):
    """reference_action's first step: the dialog's continue when it is offered, else the close of an open dialog or modal
    (a level-up's chat leaves only WAIT, the close and the continue legal); None with none open."""
    obs = task.agent.last_observation
    if obs['mask'][A_DIALOG_CONTINUE]:
        return A_DIALOG_CONTINUE
    return A_CLOSE_IF if obs['dialogOpen'] or obs['modalOpen'] else None


def open_gate(task, names=PEN_BARRIERS):
    """A closed pen barrier's Open (a gate leaf, or village-world-v2's pen door): an offered ACT, else the click toward it
    (either replaces a running option, and the same ACT again goes on with it); None with none closed. In a world other
    than FROZEN an open dialog comes first (dialog_step): it masks the Open and every click, so the click toward the
    barrier was a WAIT that left the dialog open for good (the chickens control's door rule, which runs before
    reference_action's own dialog step, waited from a door's revert to the end of the life on training layout 20014)."""
    obs = task.agent.last_observation
    gate = closed(task, names)
    if not gate:
        return None
    if village_of(task.world.t.layout)['version'] != FROZEN:
        step = dialog_step(task)
        if step is not None:
            return step
    for k, sel in enumerate(obs['actSel']):
        if obs['mask'][A_ACT0 + k] and (sel['kind'], sel['x'], sel['z'], sel['option']) in gate:
            return A_ACT0 + k
    return click_toward(task, {(t[1], t[2]) for t in gate})


def in_scene(task, x, z):
    """Whether a tile lies in the client's scene of the player's build area (sim/paths.py scene_base, SCENE tiles a side)."""
    p = task.agent.player
    bx, bz = paths.scene_base(p.origin_x, p.origin_z)
    return bx <= x < bx + paths.SCENE and bz <= z < bz + paths.SCENE


def toward_scene(task, action, targets):
    """`action`, unless a reference waits (A_WAIT, no option running) with targets (kind, x, z, option) of which the
    client search reaches none and some lie past the client's scene (in_scene): in a 64-tile layout (village-world-v2) a
    target can lie past the scene of the player's build area, which recentres only when the player nears its edge, and
    click_toward reads the client search, which reaches no tile outside the scene, so the reference waited for the rest
    of the phase (13 rotation phases of the first village-world-v2 soak). Then the legal click whose route ends nearest
    (Chebyshev) the nearest target past the scene, if nearer than the player; else A_WAIT. Every tile of a 48-tile layout
    lies in the scene, so this never acts in compact-world-v1."""
    obs = task.agent.last_observation
    if action != A_WAIT or obs.get('optionBusy') or not targets:
        return action
    past = [(x, z) for _, x, z, _ in targets if not in_scene(task, x, z)]
    if not past:
        return action
    p = task.agent.player
    tree = task.world.player_scene(p)
    if any(tree.at(tree.dist, x + dx, z + dz, paths.SCENE_UNREACHED) < paths.SCENE_UNREACHED
           for _, x, z, _ in targets for dx in (-1, 0, 1) for dz in (-1, 0, 1)):
        return action
    far = lambda x, z: min(max(abs(x - a), abs(z - b)) for a, b in past)
    best, choice = far(p.x, p.z), A_WAIT
    for k, c in enumerate(obs['clicks']):
        if obs['mask'][CLICK_ROWS[k]]:
            route = paths.client_route(tree, p.x, p.z, (p.x + c['dx'], p.z + c['dz']), nearest=True)
            if route and far(*route[-1]) < best:
                best, choice = far(*route[-1]), CLICK_ROWS[k]
    return choice


def reference(task, goal):
    """reference_action for `goal`, walking toward its targets past the client's scene (toward_scene)."""
    v = view(task, goal)
    action = reference_action(v)
    return toward_scene(task, action, reference_targets(v)) if action == A_WAIT else action


def leave_pen(task, memo):
    """With the player inside a pen (in_pen) and a pen barrier closed, open_gate, whatever option runs (a chase the
    fighter began can run on); None otherwise."""
    return open_gate(task) if in_pen(task, memo) else None


# ---- the controls ------------------------------------------------------------------------------------------------
class Wait:
    def reset(self, seed):
        pass

    def act(self, observation, task):
        return A_WAIT


class Random:
    """Uniform random legal play, keyed by the life's seed."""
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, observation, task):
        return int(self.rng.choice(np.flatnonzero(observation['mask'])))


class Gatherer:
    """reference_action for `goal` (wood, fish or mine), after drop_own of its own product, or of any product of `drops`'
    goals (a rotation's gatherers). In a rotation (`rotation`), first: fight back an attacker it can reach (fight_back;
    a handover can fall in the middle of the fighter's fight), then leave a closed pen the fighter's phase ended in
    (leave_pen), since reference_action opens a pen gate only for its melee goal."""
    def __init__(self, goal, drops=None, rotation=False):
        self.goal, self.drops, self.rotation, self.family = goal, tuple(drops or (goal,)), rotation, None
        self.reach, self.memo, self.enders = Reach(), {}, 0

    def reset(self, seed):
        self.reach, self.enders = Reach(), 0

    def act(self, observation, task):
        if self.rotation:
            action = fight_back(task, self.reach)
            if action is None:
                action = leave_pen(task, self.memo)
            if action is not None:
                return issue(self, task, action)
        if self.family is None:
            self.family = set().union(*(family_ids(task.world.t, g) for g in self.drops))
        action = drop_own(task, self.family)
        return issue(self, task, reference(task, self.goal) if action is None else action)


class ReachReference(Reference):
    """sim/combat.py Reference('full') whose first rule, fight back an attacker, takes only the attackers the player can
    reach (Reach.attackers); an attacker across a closed pen fence would hold it for the rest of the life, while the
    rule after it opens the gate. Reference itself is unchanged for the goal tasks."""
    def __init__(self):
        super().__init__('full')
        self.reach = Reach()

    def reset(self, seed):
        super().reset(seed)
        self.reach = Reach()

    def targets(self, task, attackers):
        """Reference.targets of the reachable attackers; with none, a closed pen door first (village-world-v2), as its rule
        takes a closed pen gate first."""
        reach = [n for n in attackers if n in self.reach.attackers(task)]
        return (not reach and closed(task, (PEN_DOOR,))) or Reference.targets(task, reach)


class Fighter:
    """ReachReference on the goblins, its style set to the lowest of attack, strength and defence between fights. Its
    food and weapon are on inventory page 0, so it turns back there first (after a rotation's gatherer). A chase of a
    goblin it cannot reach (Reach.chasing: a closed pen gate between) gives way to the gate's Open."""
    def __init__(self):
        self.reference, self.enders = ReachReference(), 0

    def reset(self, seed):
        self.reference.reset(seed)
        self.enders = 0

    def act(self, observation, task):
        obs = task.agent.last_observation
        if obs['invPage'] > 0 and obs['mask'][A_INV_PREV]:
            return A_INV_PREV
        action = open_gate(task) if self.reference.reach.chasing(task) else None
        if action is None:
            action = style_row(task, lowest_skill(task))
        if action is None:
            p, near = task.agent.player, attackers(task)
            action = self.reference.action(task)
            if action == A_WAIT and not task.agent.last_observation.get('optionBusy') and \
                    (p.hp >= p.base_levels[HITPOINTS] or p.target is not None or near):  # not resting to full hitpoints
                action = toward_scene(task, action, self.reference.targets(task, near))
        return issue(self, task, action)


class Chickens:
    """Chickens only (sim/run.py reference_action's melee targets), each new fight in the lowest style; an attacker it can
    reach whose Attack is offered is fought back first (fight_back). (An npc can keep the player as its target from where
    no hit lands: a hen hit once outside a closed pen with the player inside stopped this control in 2 of 64 soak lives
    when it chased every attacker.) A chase of an npc it cannot reach gives way to a closed pen gate's Open. `skill`
    forces a style at once in place of the lowest between fights (`once`'s fights)."""
    def __init__(self, skill=None):
        self.skill, self.reach, self.enders = skill, Reach(), 0

    def reset(self, seed):
        self.reach, self.enders = Reach(), 0

    def act(self, observation, task):
        action = open_gate(task) if self.reach.chasing(task) else None
        if action is None:
            action = style_row(task, self.skill, between=False) if self.skill else style_row(task, lowest_skill(task))
        if action is None:
            action = fight_back(task, self.reach)
        if action is None:  # reference_action's melee targets take a closed pen gate first; a pen door likewise, through
            # toward_scene as the fighter's door target (a door past the client scene: open_gate's click toward it waits)
            door = (PEN_DOOR,)
            action = open_gate(task, door)
            if action == A_WAIT:
                action = toward_scene(task, action, closed(task, door))
        return issue(self, task, reference(task, 'melee') if action is None else action)


class Once:
    """E2: one XP gain in each skill of S, then WAIT (module docstring)."""
    PHASES = ('woodcutting', 'mining', 'fishing', 'attack', 'strength', 'defence')
    GOALS = dict(woodcutting='wood', mining='mine', fishing='fish')

    def reset(self, seed):
        self.phase, self.xp, self.since, self.fights = 0, None, None, {}

    def act(self, observation, task):
        p, w = task.agent.player, task.world
        while self.phase < len(self.PHASES):
            skill = self.PHASES[self.phase]
            stat = SKILLS.index(skill)
            if self.xp is None:
                self.xp, self.since = p.stats[stat], w.tick
            if p.stats[stat] > self.xp or w.tick - self.since > ONCE_TICKS:
                self.phase, self.xp = self.phase + 1, None
                continue
            if skill in self.GOALS:
                return reference(task, self.GOALS[skill])
            return self.fights.setdefault(skill, Chickens(skill)).act(observation, task)
        return A_WAIT


class EatAll:
    """E3: eat every food at once (turning to the page that holds it), then WAIT."""
    def reset(self, seed):
        pass

    def act(self, observation, task):
        p, obs = task.agent.player, task.agent.last_observation
        mask, page = obs['mask'], obs['invPage']
        for k, held in enumerate(obs['heldSel']):
            if held['option'] == 'Eat' and held['name'] in FOOD and mask[A_HELD0 + k]:
                return A_HELD0 + k
        food = {task.world.t.obj_id(name) for name in ('shrimp', 'bread')}
        pages = {slot // N_INV_SEL for slot, obj, _ in p.inv.listed() if obj in food}
        if any(q > page for q in pages) and mask[A_INV_NEXT]:
            return A_INV_NEXT
        if any(q < page for q in pages) and mask[A_INV_PREV]:
            return A_INV_PREV
        return A_WAIT


class Rotation:
    """A fixed timer over controls: order[k] acts in ticks [k x period, (k + 1) x period) from arm, cyclically. Its
    gatherers drop the product of every gatherer of the order (a full inventory of another member's product would stop
    them otherwise), as the four-way timer of 11 November did."""
    def __init__(self, order, period):
        if type(period) is not int or period < 1:
            raise ValueError(f'a rotation period of {period!r} ticks (the study freezes it on development records)')
        self.order, self.period = tuple(order), period
        goals = [name for name in self.order if name in GATHERERS]
        self.members = [Gatherer(name, goals, rotation=True) if name in GATHERERS else make(name) for name in self.order]

    def reset(self, seed):
        for member in self.members:
            member.reset(seed)

    def act(self, observation, task):
        k = (task.world.tick - task.start_tick) // self.period % len(self.members)
        return self.members[k].act(observation, task)


class PhaseMarks:
    """The soak's evidence of a rotation life: at each phase's first decision, the member, the tick, the XP in S, whether
    the player is in a pen, the attackers, hitpoints and free slots."""
    def __init__(self, rotation):
        self.rotation, self.phase, self.marks, self.memo = rotation, None, [], {}

    def __call__(self, task):
        phase = (task.world.tick - task.start_tick) // self.rotation.period
        if phase != self.phase:
            p, self.phase = task.agent.player, phase
            self.marks.append(dict(phase=phase, member=self.rotation.order[phase % len(self.rotation.order)],
                                   tick=task.world.tick - task.start_tick, xp=sum(p.stats[k] for k in task.skills),
                                   in_pen=in_pen(task, self.memo), attackers=len(attackers(task)), hp=p.hp,
                                   free=p.inv.freespace()))


GATHERERS = ('wood', 'fish', 'mine')
CONTROLS = ('wait', 'random', *GATHERERS, 'melee', 'chickens', 'once', 'eat', 'rotation')


def make(name, period=None):
    """A control by name (CONTROLS); `period`: the rotation's ticks a member, which it needs (no default)."""
    if name == 'wait':
        return Wait()
    if name == 'random':
        return Random()
    if name in GATHERERS:
        return Gatherer(name)
    if name == 'melee':
        return Fighter()
    if name == 'chickens':
        return Chickens()
    if name == 'once':
        return Once()
    if name == 'eat':
        return EatAll()
    if name == 'rotation':
        return Rotation(ROTATION_ORDER, period)
    raise ValueError(f'unknown control {name!r}; the controls are {CONTROLS}')


# ---- the reachability soak ---------------------------------------------------------------------------------------
_TABLES = None  # village-world-v1's archive, loaded once a process (a test may set it)
_WORLD_TABLES = {}  # another world's, by name


def world_tables(world):
    """The archive a free-play world plays over: sim/build.py DATA for village-world-v1, the archive its spec names
    beside it otherwise (village-world-v2, and the real engine's village-world-v3 of sim/workers.py REAL_WORLDS, the
    world it is based on: draynor-nc2.npz, with the pick content)."""
    from sim.build import DATA
    from sim.combat import VILLAGES
    from sim.workers import REAL_WORLDS
    name = VILLAGES[REAL_WORLDS[world]['based_on'] if world in REAL_WORLDS else world].get('tables')
    return DATA if name is None else DATA.with_name(name)


def _tables(world=FIRST_WORLD):
    global _TABLES
    from sim.data import Tables
    if world == FIRST_WORLD:
        if _TABLES is None:
            _TABLES = Tables.load(world_tables(world))
        return _TABLES
    if world not in _WORLD_TABLES:
        _WORLD_TABLES[world] = Tables.load(world_tables(world))
    return _WORLD_TABLES[world]


ROOT = Path(__file__).resolve().parents[1]


def site(error):
    """Where an Unsupported came from: the innermost frame in sim/ outside the death interpreter's forwarding helper
    (sim/engine.py DeathContext.unsupported), with the message and the last three frames in sim/, by path from the root."""
    where = lambda f: str(Path(f.filename).resolve().relative_to(ROOT))
    frames = [f for f in traceback.extract_tb(error.__traceback__) if Path(f.filename).resolve().is_relative_to(ROOT / 'sim')]
    k = len(frames) - 1
    while k > 0 and frames[k].name == 'unsupported':
        k -= 1
    f = frames[k] if frames else None
    return dict(file=f and where(f), line=f and f.lineno, function=f and f.name, message=str(error),
                frames=[[where(x), x.lineno, x.name] for x in frames[-3:]])


def play(task, actor, seed, history=20, watch=None):
    """One life of `task` (an armed FreePlayEpisode) with `actor`: (the final info or None, decisions, the Unsupported
    raised or None, the last `history` actions, the actions chosen that the agent swallowed). `watch(task)` sees each
    decision's state first."""
    actor.reset(seed)
    obs, decisions, actions, lost = task.projected, 0, [], 0
    while True:
        if watch is not None:
            watch(task)
        action = int(actor.act(obs, task))
        actions, lost = (actions + [action])[-history:], lost + swallowed(task, action)
        try:
            obs, _, done, info = task.step(action)
        except Unsupported as error:
            return None, decisions + 1, error, actions, lost
        decisions += 1
        if done:
            return info, decisions, None, actions, lost


SPLIT = 'development'  # the soak's records: development layouts only
PICK_SCRIPT = '[proc,pickup_loc_floor]'  # the script of a pick's 'produce' event (sim/engine.py pick)


class Life:
    """Record `record` of a record set, armed: the `split`'s layout first + record in `world`'s declared layouts
    (sim/world.py generate with the world's layout spec), the draw_village start of (key, world slot 0, episode record)
    on it, and its free-play life of `ticks` native ticks at `xp_rate` in a fresh world (village_life). The soak and the
    free-play evaluation (sim/hour.py) play the same record this way; `row` is the soak's row of the life once played.
    `world`: a name of sim/combat.py VILLAGES; FIRST_WORLD when it is None (a caller that names no world).
    village-world-v2 plays over an archive with the pick content (`tables`)."""
    def __init__(self, first, key, record, muggers, xp_rate, ticks, tables, split=None, world=None):
        from sim.combat import VILLAGES, draw_village, village_life
        from sim.search import SEARCH
        from sim.world import WORLDS, LayoutTables, generate, split_seeds
        self.key, self.record, self.muggers, self.xp_rate, self.ticks = key, record, muggers, xp_rate, ticks
        self.world = FIRST_WORLD if world is None else world
        spec = WORLDS[VILLAGES[self.world]['layouts']]
        self.layout = generate(split_seeds(split or SPLIT, 1, first + record)[0], tables, spec)
        lt = LayoutTables(tables, self.layout)
        lt.CACHE = SEARCH['cache']  # CpuBatch's bound for a layout without derived tables (a memo: no output changes)
        self.start = draw_village(key, 0, record, [self.layout], tables, muggers, layout=0)
        self.task = village_life(self.start, self.layout, lt, tables, xp_rate, ticks)
        p = self.task.agent.player
        self.base, self.xp = list(p.base_levels), list(p.stats)

    def hunter(self, events):
        """A life's evidence of the layout's highwayman (village-world-v2), None in a layout without one: whether the start
        tile lies within its reach (sim/world.py hunter_reach), the damage events it dealt (its hits of 0 included), the
        hitpoints it took, and the ticks from arm of its first hit and of its first death by the agent (None if never)."""
        from sim.world import HUNTER, hunter_reach
        w = self.task.world
        hunter = next((n for n in w.npcs if w.t.npc_types[n.base]['debugname'] == HUNTER), None)
        if hunter is None:
            return None
        tx, tz = self.start['tile']
        hits = [(tick, data) for tick, kind, data in events
                if kind == 'agent_hp' and data['cause'] == 'damage' and data['source'] == hunter.nid]
        deaths = [tick for tick, kind, data in events if kind == 'npc_defeated' and data['npc'] == hunter.nid]
        arm = self.task.start_tick
        return dict(near=max(abs(tx - hunter.start_x), abs(tz - hunter.start_z)) <= hunter_reach(w.t), hits=len(hits),
                    damage=sum(d['before'] - d['after'] for _, d in hits), first_hit=hits[0][0] - arm if hits else None,
                    first_kill=deaths[0] - arm if deaths else None, kills=len(deaths))

    def row(self, name, period, actor, played, marks, t0):
        """The soak row (module docstring) of the life that `actor` (control `name`, `period` for the rotation) played:
        `played` is play's result, `marks` the rotation's PhaseMarks or None, `t0` when the record's setup began."""
        task, start, base, xp = self.task, self.start, self.base, self.xp
        info, decisions, error, actions, lost = played
        p = task.agent.player
        events = task.world.events[task.event_start:]
        gain = {s: p.base_levels[k] - base[k] for k, s in enumerate(SKILLS) if p.base_levels[k] != base[k]}
        report = info['report'] if info else None
        damage, killer = {}, None
        names = {n.nid: task.world.t.npc_types[n.base]['debugname'] for n in task.world.npcs}
        for _, kind, data in events:
            if kind == 'agent_hp' and data['cause'] == 'damage' and data['before'] > data['after']:
                killer = names.get(data['source'], str(data['source']))
                damage[killer] = damage.get(killer, 0) + data['before'] - data['after']
        row = dict(world=self.world, form='muggers' if self.muggers else 'none', muggers=bool(self.muggers), xp_rate=self.xp_rate,
                   length=self.ticks, record=self.record, layout=int(self.layout.seed), key=self.key, actor=name,
                   period=period if name == 'rotation' else None, reason='unsupported' if error else report['reason'],
                   ticks=(report['tick'] if report else task.world.tick) - task.start_tick, decisions=decisions,
                   death=dict(tick=report['tick'] - task.start_tick, killer=killer) if report and report['reason'] == 'death'
                   else None, gain=gain, gain_s=sum(gain.get(s, 0) for s in task.skill_names),
                   paid=report['paid'] if report else None, xp={s: p.stats[k] - xp[k] for k, s in enumerate(SKILLS)
                                                               if p.stats[k] != xp[k]},
                   levels0={s: base[SKILLS.index(s)] for s in task.skill_names}, hp0=task.arm_hp,
                   food=len(start['village']['food']), muggers_placed=len(start['village']['muggers']), damage=damage,
                   eats=sum(kind == 'remove' and data['why'] == 'eat' for _, kind, data in events),
                   excluded=sum(kind == 'excluded' for _, kind, _ in events),
                   picks=sum(kind == 'produce' and data['script'] == PICK_SCRIPT for _, kind, data in events),
                   hunter=self.hunter(events),
                   unsupported=None if error is None else dict(site(error), decision=decisions,
                                                              tick=task.world.tick - task.start_tick, actions=actions,
                                                              start=start, x=p.x, z=p.z, hp=p.hp, inventory=p.inv.listed(),
                                                              observation=task.raw),
                   swallowed=lost, enders=sum(getattr(c, 'enders', 0) for c in [actor, *getattr(actor, 'members', ())]),
                   seconds=round(time.perf_counter() - t0, 3))
        if marks is not None:  # each phase's state at its start and the XP in S it gained
            ends = [m['xp'] for m in marks.marks[1:]] + [sum(p.stats[k] for k in task.skills)]
            row['phases'] = [dict(m, gained=end - m['xp']) for m, end in zip(marks.marks, ends)]
        return row


def soak_life(job):
    """One soak row (module docstring) for a job (world, muggers, xp_rate, ticks, first, key, record, actor, period,
    seed); a job of the nine fields after `world` is village-world-v1's, the form a job had before the second world."""
    world, muggers, xp_rate, ticks, first, key, record, name, period, seed = job if len(job) == 10 else (FIRST_WORLD, *job)
    tables = _tables(world)
    t0 = time.perf_counter()
    life = Life(first, key, record, muggers, xp_rate, ticks, tables, world=world)
    actor = make(name, period)
    marks = PhaseMarks(actor) if isinstance(actor, Rotation) else None
    return life.row(name, period, actor, play(life.task, actor, [seed, record], watch=marks), marks, t0)


RANDOM_CHUNK = 48  # random play's records go in chunks of this many, so the records played do not depend on processes


def soak(output, *, world, forms=(False, True), xp_rate=1, ticks=36000, first=0, records=64, key=0, controls=CONTROLS,
         random_decisions=10 ** 6, periods=(), seed=0, processes=6, chunk=RANDOM_CHUNK):
    """The reachability soak (module docstring) of `world` (sim/combat.py VILLAGES; no default): in each form (muggers
    off and on; off alone in a world that holds none), random legal play on records 0, 1, ... in chunks of `chunk` until
    its decisions reach `random_decisions` and it has played `records` records, then every other control on records 0 to
    records - 1, the rotation once at each of `periods`. Record r is the world's development layout first + r. Writes
    lives.jsonl (one row a life, in completion order) and manifest.json (the settings, the counts and the times)."""
    from sim.combat import VILLAGES
    if world not in VILLAGES:
        raise ValueError(f'a soak names its world, one of {sorted(VILLAGES)}: {world!r}')
    if any(forms) and VILLAGES[world].get('muggers') is False:
        raise ValueError(f'{world} holds no muggers: its soak plays the form without them')
    if 'rotation' in controls and not periods:
        raise ValueError('the rotation needs its periods: the study declares none')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    t0, counts, phases = time.time(), {}, {}
    job = lambda muggers, record, name, period=None: (world, bool(muggers), xp_rate, ticks, first, key, record, name, period, seed)
    with (output / 'lives.jsonl').open('w') as stream, mp.get_context('spawn').Pool(processes) as pool:
        def write(row):
            stream.write(json.dumps(row, default=str) + '\n')
            stream.flush()
            name = row['actor'] if row['period'] is None else f"{row['actor']}@{row['period']}"
            c = counts.setdefault(f"{row['form']}/{name}", dict(lives=0, decisions=0, seconds=0.0, deaths=0, unsupported=0))
            c['lives'] += 1
            c['decisions'] += row['decisions']
            c['seconds'] += row['seconds']
            c['deaths'] += row['death'] is not None
            c['unsupported'] += row['unsupported'] is not None
        for muggers in forms if 'random' in controls else ():
            started, total, r = time.time(), 0, 0
            while total < random_decisions or r < records:
                for row in pool.imap(soak_life, [job(muggers, k, 'random') for k in range(r, r + chunk)]):
                    write(row)
                    total += row['decisions']
                r += chunk
            phases[f"{'muggers' if muggers else 'none'}/random"] = dict(seconds=time.time() - started, records=r,
                                                                        decisions=total)
        others = [job(muggers, r, name, period) for muggers in forms for name in controls if name != 'random'
                  for period in (periods if name == 'rotation' else (None,)) for r in range(records)]
        started = time.time()
        for row in pool.imap_unordered(soak_life, others, chunksize=1):
            write(row)
        phases['controls'] = dict(seconds=time.time() - started, lives=len(others))
    from baseline.telemetry import sha256, source
    archive = world_tables(world)
    manifest = dict(version='soak', world=world, source=source(output), tables=dict(path=str(archive), sha256=sha256(archive)),
                    started=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t0)), cpus=mp.cpu_count(),
                    forms=['muggers' if m else 'none' for m in forms], xp_rate=xp_rate, ticks=ticks,
                    first=first, records=records, key=key, controls=list(controls), random_decisions=random_decisions,
                    random_chunk=chunk, periods=list(periods), seed=seed, processes=processes,
                    wall_seconds=time.time() - t0, phases=phases, counts=counts)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


# ---- what a pool of layouts holds (village-world-v2's proposal, measured before its freeze) -------------------------------
LIMITS = dict(locs=64, zone=32, npcs=16)  # sim/native/rs.h RS_MAX_LOCS, RS_ZONE_CAP, RS_MAX_NPCS today: reported, not enforced


def _spread(values):
    values = sorted(values)
    return dict(median=float(np.median(values)), low=values[0], high=values[-1]) if values else None


def measure_layout(layout, tables):
    """One layout's facts: its seed and attempt; pen (none, gate, door), field kind and plant count, willow, yew and
    highwayman; start tiles; over the start tiles each goal's steps to its nearest target (GOALS order, sim/world.py
    goal_dist), the farthest of the four and the steps to the field's nearest plant (walked over the closed state); the
    share of start tiles within the highwayman's reach; map locs, the open stages a barrier adds, the most locs an 8x8 zone
    holds and the npcs a life's world spawns (sim/combat.py village_npcs), each against LIMITS."""
    from baseline.resource_task import GOALS
    from sim import world as W
    from sim.combat import camp
    names = [tables.loc_types[int(r[0])]['debugname'] for r in layout.locs]
    npcs = [tables.npc_types[int(r[1])]['debugname'] for r in layout.npcs]
    plants = [k for k, n in enumerate(names) if n in dict(W.VILLAGE_WORLD['field_kinds'])]
    start = layout.start
    starts = np.argwhere(start)
    dist = {g: layout.goal_dist[k][start] for k, g in enumerate(GOALS)}
    farthest = np.max(np.stack([dist[g] for g in GOALS]), axis=0)
    region = outside(layout)
    field = None
    if plants:
        things = [('loc', *(int(v) for v in layout.locs[k])) for k in plants]
        steps = W.flood(paths.step_masks(layout.flags), W._seeds(W._reach(layout.flags, things, tables), region))
        field = steps[start]
    hunter = next((r for r, n in zip(layout.npcs, npcs) if n == W.HUNTER), None)
    share = None if hunter is None else \
        float(np.mean(np.max(np.abs(starts - (int(hunter[2]), int(hunter[3]))), axis=1) <= W.hunter_reach(tables)))
    zones = {}
    for _, x, z, _, _ in layout.locs:
        key = ((W.ORIGIN[0] + int(x)) >> 3, (W.ORIGIN[1] + int(z)) >> 3)
        zones[key] = zones.get(key, 0) + 1
    gate = [int(k) for k in layout.gate if k >= 0]
    village = len(layout.npcs) + len(camp(layout, tables))  # the layout's npcs and a goblin at every chicken spawn
    return dict(seed=layout.seed, attempt=layout.attempt, world=W.world_of(layout)['version'],
                pen='none' if not gate else 'door' if names[gate[0]] == W.PEN_DOOR else 'gate',
                field=names[plants[0]] if plants else None, plants=len(plants), willow=W.WILLOW in names, yew=W.YEW in names,
                hunter=hunter is not None, starts=int(start.sum()),
                steps={g: _spread(d.tolist()) for g, d in dist.items()}, farthest=_spread(farthest.tolist()),
                field_steps=None if field is None else _spread(field.tolist()), hunter_share=share,
                locs=len(layout.locs), open_locs=len(layout.open_locs), zone_most=max(zones.values()), npcs=village,
                over={k: v for k, v in dict(locs=len(layout.locs) + len(layout.open_locs), zone=max(zones.values()),
                                            npcs=village).items() if v > LIMITS[k]})


def measure_pool(path, tables):
    """measure_layout over a saved pool (sim/world.py save_pool), and the pool's summary: the share of drawn attempts that
    passed the checks (one passing attempt a seed, after `attempt` that failed), the farthest source's steps from a start
    (the median over layouts of each layout's median, and the range of every start's), the field's steps likewise, the
    highwayman's share of starts in the layouts that hold one, the counts of pens, field kinds, willows, yews and
    highwaymen, each count's range, and every layout over a LIMITS count."""
    from sim.world import load_pool
    layouts, _ = load_pool(path)
    rows = [measure_layout(layout, tables) for layout in layouts]
    count = lambda key, value: sum(r[key] == value for r in rows)
    summary = dict(
        layouts=len(rows), world=sorted({r['world'] for r in rows}), attempts=sum(r['attempt'] + 1 for r in rows),
        pass_share=len(rows) / sum(r['attempt'] + 1 for r in rows),
        farthest=dict(median_of_medians=float(np.median([r['farthest']['median'] for r in rows])),
                      low=min(r['farthest']['low'] for r in rows), high=max(r['farthest']['high'] for r in rows)),
        field_steps=dict(median_of_medians=float(np.median([r['field_steps']['median'] for r in rows if r['field_steps']])),
                         low=min(r['field_steps']['low'] for r in rows if r['field_steps']),
                         high=max(r['field_steps']['high'] for r in rows if r['field_steps'])) if any(r['field_steps'] for r in rows) else None,
        hunter_share=_spread([r['hunter_share'] for r in rows if r['hunter']]),
        pens=dict(none=count('pen', 'none'), gate=count('pen', 'gate'), door=count('pen', 'door')),
        fields={kind: count('field', kind) for kind in sorted({r['field'] for r in rows if r['field']})},
        willow=count('willow', True), yew=count('yew', True), hunter=count('hunter', True),
        plants=_spread([r['plants'] for r in rows]), starts=_spread([r['starts'] for r in rows]),
        locs=_spread([r['locs'] + r['open_locs'] for r in rows]), zone_most=_spread([r['zone_most'] for r in rows]),
        npcs=_spread([r['npcs'] for r in rows]), limits=LIMITS, over=[dict(seed=r['seed'], **r['over']) for r in rows if r['over']])
    return dict(summary=summary, layouts=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('measure', help="what a saved pool's layouts hold (measure_pool)")
    m.add_argument('--pool', type=Path, required=True)
    m.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('soak', help='random legal play and the controls on development records of a village world')
    s.add_argument('--world', required=True, choices=('village-world-v1', 'village-world-v2'),
                   help='the declared world (sim/combat.py VILLAGES); its records are its own layouts')
    s.add_argument('--muggers', type=int, nargs='+', choices=(0, 1), required=True,
                   help='the forms to play: 0 without muggers, 1 with them')
    s.add_argument('--xp-rate', type=int, default=1)
    s.add_argument('--ticks', type=int, default=36000, help='each life\'s length (fixed here)')
    s.add_argument('--first', type=int, required=True, help='the development split index of record 0\'s layout')
    s.add_argument('--records', type=int, default=64, help='records each control plays in each form')
    s.add_argument('--key', type=int, required=True, help='the start draws\' key (sim/combat.py draw_village)')
    s.add_argument('--seed', type=int, required=True, help='actor seeds: default_rng([seed, record])')
    s.add_argument('--controls', nargs='+', default=list(CONTROLS), choices=CONTROLS)
    s.add_argument('--random-decisions', type=int, default=10 ** 6)
    s.add_argument('--periods', type=int, nargs='+', default=(), help='the rotation\'s ticks a member, one run each '
                   '(needed with the rotation; the study declares none)')
    s.add_argument('--processes', type=int, default=6)
    s.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'measure':
        from sim.data import Tables
        result = measure_pool(args.pool, Tables.load())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + '\n')
        print(json.dumps(result['summary'], indent=1))
        return
    manifest = soak(args.output, world=args.world, forms=tuple(bool(m) for m in args.muggers), xp_rate=args.xp_rate, ticks=args.ticks,
                    first=args.first, records=args.records, key=args.key, controls=tuple(args.controls),
                    random_decisions=args.random_decisions, periods=tuple(args.periods), seed=args.seed,
                    processes=args.processes)
    print(json.dumps(manifest['counts'], indent=1))


if __name__ == '__main__':
    main()
