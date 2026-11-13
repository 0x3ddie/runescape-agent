"""Scripted play: episode setup and the gathering reference player.

prepare_capacity runs before the task clock is armed. Teacher is a reference player for
feasibility checks (baseline.evaluate --policy reference), Walker a walk by clicks and Chaser one attack on a wandering npc
for fidelity recordings (--policy edge, chase); their actions are never labels and they never act inside a training rollout. The demonstration and DAgger tools that once used
it were removed.
"""
import re
from collections import Counter

from act_contract import A_ACT0, BANK_NAME, BANK_OPTION, CLICK_ROWS, banks
from item_contract import observation_contract

LOGS = 1511
AXE = 'bronze axe'
NET = 'small fishing net'
# act-v3 rows (env-ts/core/actions.ts) the scripted code uses by number.
A_HELD0, A_DEPOSIT0, A_WITHDRAW0, A_STYLE0 = 29, 45, 53, 72
A_CLOSE_IF, A_DIALOG_CONTINUE, A_DIALOG_CHOICE0 = 61, 62, 63
A_INV_PREV, A_INV_NEXT, A_BANK_PREV, A_BANK_NEXT = 67, 68, 69, 70
# What the teacher gathers with each skill, by the game's own names: its tool, the targets
# it works and the option it uses on them. Which targets its level can work
# (plain trees, shrimp nets) is privileged, like its labels. Order is priority: with both
# tools it chops, so a fishing fixture where trees are labelled carries only the net.
SKILLS = dict(
    woodcutting=dict(tool=AXE, target=re.compile(r'^tree$', re.I), verb='chop down'),
    fishing=dict(tool=NET, target=re.compile(r'^fishing spot$', re.I), verb='net'),
)
# Banking keeps every gathering tool, not only the one in use. Banking "everything but my
# tool" made a chopping teacher bank the net while another teacher kept it, and imitation
# of both banked the net and then netted without one (0 of 16 at Draynor with both tools).
GATHER_TOOLS = frozenset((AXE, NET, 'bronze pickaxe'))


def counts(items):
    result = Counter()
    for item in items:
        if item['count']:
            result[item['id']] += item['count']
    return result


def assert_transfer(inv_before, bank_before, inv_after, bank_after, selected, quantity, deposit):
    """Exact identity/quantity and whole inventory+bank conservation."""
    i0, b0, i1, b1 = map(counts, (inv_before, bank_before, inv_after, bank_after))
    assert i0 + b0 == i1 + b1, 'inventory/bank conservation failed'
    delta = Counter({selected['id']: -quantity if deposit else quantity})
    actual = i1.copy(); actual.subtract(i0)
    assert {k: v for k, v in actual.items() if v} == dict(delta), 'wrong selected transfer quantity'
    actual = b1.copy(); actual.subtract(b0)
    assert {k: v for k, v in actual.items() if v} == {k: -v for k, v in delta.items()}, 'wrong bank delta'


class ItemGate:
    """Drive inventory/bank actions one step at a time and verify every transfer."""
    def __init__(self, env, record):
        self.env, self.record, self.obs = env, record, None

    def accept(self, obs):
        observation_contract(obs)
        assert not obs['done'] and not obs.get('truncated') and not obs['player']['isDead'] and obs['player']['hp'] > 0
        assert len(obs['mask']) == 96
        if not obs['bankOpen']:
            assert not any(obs['mask'][A_WITHDRAW0:A_WITHDRAW0 + 8]), 'withdraw exposed while bank closed'
        self.obs = obs
        return obs

    def step(self, action):
        if action != 0:
            self.settle()
        assert self.obs['mask'][action], f'illegal action {action}'
        self.record('requested', action=action)
        obs = self.env.step(action)
        self.record('observed', action=action, observation=obs)
        return self.accept(obs)

    def settle(self):
        for _ in range(80):
            if not self.obs.get('optionBusy', False):
                return
            self.step(0)
        raise AssertionError('asynchronous item option did not settle')

    def interface(self, opened):
        # After a long teleport the booth can appear a few ticks late, and an opener that
        # does nothing (a banker across a counter) is refused, so the next one is tried.
        for _ in range(3):
            for _ in range(20):
                action = bank_opener(self.obs) if opened else A_CLOSE_IF
                if action and self.obs['mask'][action]:
                    break
                self.step(0)
            self.step(action)
            for _ in range(40):
                if self.obs['bankOpen'] is opened and not self.obs.get('optionBusy', False):
                    return
                self.step(0)
        raise AssertionError('bank interface did not reach requested state')

    def bank(self):
        self.settle()
        assert self.obs['bankOpen']
        while self.obs['bankPage']:
            self.step(A_BANK_PREV)
        items = []
        pages = self.obs['bankPages']
        for page in range(pages):
            assert self.obs['bankPage'] == page and self.obs['bankPages'] == pages
            items.extend(dict(i) for i in self.obs['bankSel'] if i['count'])
            if page + 1 < pages:
                self.step(A_BANK_NEXT)
        return items

    def select(self, name, bank=False):
        key, prev, nxt = ('bankPage', A_BANK_PREV, A_BANK_NEXT) if bank else ('invPage', A_INV_PREV, A_INV_NEXT)
        while self.obs[key]:
            self.step(prev)
        for page in range(self.obs['bankPages'] if bank else 4):
            for k, item in enumerate(self.obs['bankSel' if bank else 'invSel']):
                if item['name'].lower() == name.lower() and item['count']:
                    return k, dict(item)
            if self.obs['mask'][nxt]:
                self.step(nxt)
        raise AssertionError(f'missing selected item {name}')

    def transfer(self, name, deposit):
        bank_before = self.bank()
        k, selected = self.select(name, bank=not deposit)
        inv_before = self.obs['inventory']
        quantity = selected['count'] if deposit else 1
        self.step((A_DEPOSIT0 if deposit else A_WITHDRAW0) + k)
        expected = counts(inv_before)[selected['id']] + (-quantity if deposit else quantity)
        for _ in range(80):
            if counts(self.obs['inventory'])[selected['id']] == expected:
                break
            self.step(0)
        bank_after = self.bank()
        assert_transfer(inv_before, bank_before, self.obs['inventory'], bank_after, selected, quantity, deposit)
        self.record('transfer_verified', selected=selected, quantity=quantity, deposit=deposit,
                    inventory_before=inv_before, bank_before=bank_before,
                    inventory_after=self.obs['inventory'], bank_after=bank_after)
        return bank_after


def has_tool(obs, tool):
    """Carried or wielded."""
    return any(i['count'] and i['name'].lower() == tool for i in obs['inventory']) or \
        any(i['name'].lower() == tool for i in obs.get('equipment', []))


def prepare_capacity(transport, obs, record, capacity, tool=AXE):
    """Leave the start state a fixture asks for (see baseline/config.py)."""
    gate = ItemGate(transport, record); gate.accept(obs)
    if gate.obs.get('modalOpen') and not gate.obs.get('bankOpen'): gate.step(A_CLOSE_IF)
    if capacity not in ('ample', 'starter', 'tool-banked', 'respawn', 'both-tools'): raise ValueError('unknown capacity')
    keep = dict(ample={tool}, starter=None, respawn=None, **{'tool-banked': set(), 'both-tools': {AXE, NET}})[capacity]
    bank = []
    if capacity != 'respawn':
        gate.interface(True)
        if keep is not None:
            names = [i['name'] for i in gate.obs['inventory'] if i['count'] and i['name'].lower() not in keep]
            for name in names: gate.transfer(name, True)
        bank = gate.bank(); gate.interface(False)
    inventory = gate.obs['inventory']
    assert len(inventory) == dict(ample=1, starter=18, respawn=18, **{'both-tools': 2}).get(capacity, 0)
    assert counts(inventory)[LOGS] == counts(bank)[LOGS] == 0
    assert has_tool(gate.obs, tool) == (capacity != 'tool-banked')
    if capacity == 'both-tools':
        assert has_tool(gate.obs, AXE) and has_tool(gate.obs, NET)
    if capacity == 'tool-banked':
        assert any(i['name'].lower() == tool for i in bank)
    record('capacity_prepared', capacity=capacity, occupied_slots=len(inventory), bank=bank, observation=gate.obs)
    return gate.obs


# The game's names of the items a declared start may hold, by their debug names (TaskCfg seedItems).
KIT_NAMES = dict(bronze_axe='Bronze axe', net='Small fishing net', bronze_pickaxe='Bronze pickaxe', shrimp='Shrimps',
                 bread='Bread', bronze_dagger='Bronze dagger', bronze_sword='Bronze sword')


def prepare_declared(transport, obs, record, items, stats):
    """A declared start (kit 'none', baseline/world.py): the inventory is exactly `items`, nothing is
    worn and the base levels are `stats`. Any difference stops the episode before arm. setStats writes
    levels without [changestat], so the combat varps ([proc,player_combat_stat]) keep a new player's
    values until something recomputes them; re-selecting the new player's style (row 0, com_mode 0)
    does, through [proc,set_attackstyle]. That ordinary action is the setup's one step."""
    gate = ItemGate(transport, record)
    gate.accept(obs)
    want, have = Counter(), Counter()
    for pair in items.split(','):
        name, count = pair.split(':')
        want[KIT_NAMES[name]] += int(count)
    for item in obs['inventory']:
        have[item['name']] += item['count']
    assert have == want and not obs.get('equipment'), f'declared inventory differs: {dict(have)} worn {obs.get("equipment")}'
    levels = {s['name'].lower(): s['baseLevel'] for s in obs['skills']}
    for pair in stats.split(','):
        name, level = pair.split(':')
        assert levels[name] == int(level), f'declared {name} level differs: {levels[name]}'
    obs = gate.step(A_STYLE0)
    record('declared_start', items=items, stats=stats, observation=obs)
    return obs


# Bank a full inventory only if more than this many ticks remain. Logs count the same
# carried or banked, so a late trip costs nothing; at 200 the teacher waited out the end
# of every long episode full, and 962 of 1,201 labels in full idle states were WAIT. The
# policy learned "full, so wait" and waited 40 steps from a booth at Varrock east with
# 78% of the time left. A larger value is a deliberately wrong banking rule.
TRIP_TICKS = 0
# Keep this far from an npc that could win a fight: it offers Attack and outranks the
# player's combat level. Draynor's jail guards (26 against 23) wander to the trees east
# of the bank, and killed a teacher in Chromium that kept chopping from 18 hitpoints to 7.
DANGER_RADIUS = 8
# Within this many tiles of its labelled point, the teacher works any target in reach.
AREA_RADIUS = 6
# Resting beside a bank means within this many walking steps of it.
REST_STEPS = 3


def threats(obs):
    level = obs['player'].get('combatLevel', 0)
    return [n for n in obs.get('npcs', []) if (n.get('combatLevel') or 0) > level
            and any('attack' in o.lower() for o in n.get('options', []))]


def works(affordance, skill):
    """An affordance the teacher uses with this skill: the target's name and the skill's option."""
    return bool(skill['target'].match(affordance['name'])) and affordance['option'].lower() == skill['verb']


def bank_opener(obs):
    """The ACT row that opens the bank: a booth's banking option, nearest first, else a
    banker's (at Falador the bankers stand behind the counter and cannot be reached)."""
    banking = [(a['kind'] != 'loc', A_ACT0 + k) for k, a in enumerate(obs.get('actSel', []))
               if BANK_NAME.search(a['name']) and BANK_OPTION.fullmatch(a['option'])]
    return min(banking)[1] if banking else None


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def click_to(obs, point, failed=()):
    """The legal click (act-v3) whose target closes most on `point` (Chebyshev; an open target first among equals, then
    the lower slot), skipping (x, z, row) in `failed`; None when none closes on it."""
    mask, p = obs['mask'], obs['player']
    here, best = (p['x'], p['z']), None
    for k, c in enumerate(obs.get('clicks', [])):
        row = CLICK_ROWS[k]
        if mask[row] and (*here, row) not in failed:
            key = (cheb(here, point) - cheb((here[0] + c['dx'], here[1] + c['dz']), point), c['open'])
            if key[0] > 0 and (best is None or key > best[0]):
                best = (key, row)
    return None if best is None else best[1]


def click_from(obs, foes):
    """The legal click whose target lies farthest from the nearest of `foes` (relative positions), or None."""
    mask, best = obs['mask'], None
    for k, c in enumerate(obs.get('clicks', [])):
        if mask[CLICK_ROWS[k]]:
            away = min(max(abs(c['dx'] - n['dx']), abs(c['dz'] - n['dz'])) for n in foes)
            if best is None or away > best[0]:
                best = (away, CLICK_ROWS[k])
    return None if best is None else best[1]


class Walker:
    """Walks to a fixed point by clicks (click_to), as baseline/control_check.py's walks do: it continues a dialog,
    closes the bank, never repeats a click that failed from its tile, and waits once no click closes on the point."""

    def __init__(self, point):
        self.point, self.failed, self.last = point, set(), None

    def action(self, obs, remaining=None):
        mask, p = obs['mask'], obs['player']
        ended = (obs.get('feedback') or {}).get('ended') or {}
        if self.last and ended.get('reason') == 'failed' and ended.get('action', -1) in CLICK_ROWS:
            self.failed.add((*self.last, ended['action']))
        self.last = (p['x'], p['z'])
        if obs.get('dialogOpen') and mask[A_DIALOG_CONTINUE]:
            return A_DIALOG_CONTINUE
        if (obs.get('modalOpen') or obs.get('bankOpen')) and mask[A_CLOSE_IF]:
            return A_CLOSE_IF
        action = click_to(obs, self.point, self.failed)
        return 0 if action is None else action


class Chaser:
    """Attacks the offered npc of this name nearest `distance` tiles away, once, then waits, continuing dialogs and closing
    the bank: an ACT that the server's naive chase must carry to a wandering npc."""

    def __init__(self, name, distance):
        self.name, self.distance, self.sent = name, distance, False

    def action(self, obs, remaining=None):
        mask = obs['mask']
        if obs.get('dialogOpen') and mask[A_DIALOG_CONTINUE]:
            return A_DIALOG_CONTINUE
        if (obs.get('modalOpen') or obs.get('bankOpen')) and mask[A_CLOSE_IF]:
            return A_CLOSE_IF
        near = [(abs(a['distance'] - self.distance), k) for k, a in enumerate(obs['actSel'])
                if a['name'] == self.name and a['option'] == 'Attack' and mask[A_ACT0 + k]]
        if self.sent or not near:
            return 0
        self.sent = True
        return A_ACT0 + min(near)[1]


def nearest_bank(obs):
    """(x, z) of the nearest bank in the 15-tile loc and npc lists, or None."""
    p = obs['player']
    return min(banks(obs), key=lambda b: (cheb((p['x'], p['z']), b), b), default=None)


class Teacher:
    """Scripted gatherer. The one fact it has that the policy lacks is where to gather.

    Each labelled village has a point per skill (config.GATHER_AREAS): privileged
    information, as in "Learning by Cheating" (Chen et al. 2019). The teacher chooses
    with it; the policy learns from what it sees and is deployed without it. The teacher
    acts on the label only through actions the policy has, working a target slot or
    clicking toward the point (act-v3), and never waits beside a workable target while away from
    its area: a wait for a reason the policy could not see taught imitation to wait
    beside choppable trees in new villages.

    It works the first skill (SKILLS order) whose tool it holds and whose point its
    village has; holding none, it fetches the first one's tool from the bank. What it
    holds decides, so the policy can see which skill. It wields its axe when nothing is
    worn, walks away when a stronger npc attacks, heads for the labelled point and works
    there (preferring targets away from such npcs), and when the inventory is full banks
    everything but its gathering tools (unless fewer than trip_ticks remain); at half hitpoints or
    less it rests by a bank. Because the choice depends on the observation (plus the
    labels and which bank pages it has looked at), it can label any state a learner
    reaches. It never picks up, drops or consumes items.
    """
    def __init__(self, areas, trip_ticks=TRIP_TICKS, skills=SKILLS):
        if not areas: raise ValueError('no labelled gathering point')  # test villages have none
        self.skills = skills  # what it gathers per skill
        self.areas = {skill: areas[skill] for skill in skills if skill in areas}
        self.skill = next(iter(self.areas))
        self.checked_pages = set()
        self.trip_ticks = trip_ticks

    def action(self, obs, remaining=None):
        mask = obs['mask']
        # A dialog can open mid-option (a level-up during a chop) and block it until
        # dismissed, so answer it before waiting on the option. Interface options
        # (banking) hold the mask at WAIT and are left to finish.
        if obs.get('dialogOptions'): return next((a for a in range(A_DIALOG_CHOICE0, A_DIALOG_CHOICE0 + 4) if mask[a]), 0)
        if obs.get('modalOpen') and not obs.get('bankOpen') and mask[A_CLOSE_IF]: return A_CLOSE_IF
        # Walk away at once when something stronger attacks, even mid-chop, to the place
        # farthest from it along a real path. RETREAT's straight line ran into a wall by
        # the jail; re-issued 35 times without moving, it let the guard kill the teacher.
        # At half hitpoints or less, walk away from any attacker: a gather option holds
        # the player, who then never hits back, and a scorpion (level 14) killed a teacher
        # at Al Kharid from 20 hitpoints while its net stayed busy for 230 ticks.
        player = obs['player']
        hurt = player.get('hp') is not None and player.get('maxHp') and 2 * player['hp'] <= player['maxHp']
        armed = [n for n in obs.get('npcs', []) if any('attack' in o.lower() for o in n.get('options', []))]
        attackers = [n for n in (armed if hurt else threats(obs)) if n['distance'] <= 2]
        fleeing = obs.get('optionBusy') and obs.get('optionAction', -1) in CLICK_ROWS
        if obs['player'].get('inCombat') and attackers and not fleeing:
            away = click_from(obs, attackers)
            if away: return away
        if obs.get('optionBusy'): return 0
        inventory = [i for i in obs['inventory'] if i['count']]
        held = [skill for skill in self.areas if has_tool(obs, self.skills[skill]['tool'])]
        self.skill = held[0] if held else next(iter(self.areas))
        if obs.get('bankOpen'):
            return self._at_bank(obs, inventory, bool(held))
        self.checked_pages.clear()
        if not held:
            return self._to_bank(obs)
        wield = next((A_HELD0 + k for k, h in enumerate(obs.get('heldSel', []))
                      if h['name'].lower() == AXE and h['option'].lower() == 'wield'), None)
        if wield is not None and not obs.get('equipment') and mask[wield]:
            return wield
        if len(inventory) == 28:
            return self._to_bank(obs) if remaining is None or remaining > self.trip_ticks else 0
        if hurt:
            return self._rest(obs)
        return self._gather(obs)

    def _rest(self, obs):
        """At half hitpoints or less, wait by a bank while they come back (about one a minute).
        Jail guards wore a teacher from 20 hitpoints to 3 over 800 decisions as it went back
        to their trees, and a policy killed at 6 hitpoints had been worn down the same way."""
        bank, p = nearest_bank(obs), obs['player']
        if bank is None: return self._to_bank(obs)  # back towards the labelled point, where one was
        return (click_to(obs, bank) or 0) if cheb((p['x'], p['z']), bank) > REST_STEPS else 0

    def _to_bank(self, obs):
        opener = bank_opener(obs)
        if opener and obs['mask'][opener]: return opener
        # A booth, or a banker when the booths are out of the list.
        bank = nearest_bank(obs)
        if bank: return click_to(obs, bank) or 0
        # No bank in sight: back towards the labelled point, where one was. At Ardougne
        # south a far tree filled the inventory 36 tiles from the bank, and the teacher
        # waited there for the last 935 decisions.
        return click_to(obs, self.areas[self.skill]) or 0

    def _from_area(self, target):
        """Tiles from the labelled point of the skill in use."""
        x, z = self.areas[self.skill]
        return max(abs(target['x'] - x), abs(target['z'] - z))

    def _gather(self, obs):
        danger, mask, to_area, skill = threats(obs), obs['mask'], self._from_area, self.skills[self.skill]
        def clearance(target):
            return min((max(abs(target['dx'] - n['dx']), abs(target['dz'] - n['dz'])) for n in danger), default=99)
        def safest(candidates):
            # Clear of stronger npcs, else the one farthest from them: a jail guard that
            # stood by every tree near Draynor's bank, without attacking, held a teacher
            # that only avoided it waiting for 1,424 decisions.
            clear = [c for c in candidates if clearance(c[1]) > DANGER_RADIUS]
            return clear or sorted(candidates, key=lambda c: -clearance(c[1]))
        # Affordances list only targets the player can walk up to.
        targets = [(A_ACT0 + k, a) for k, a in enumerate(obs['actSel']) if works(a, skill) and mask[A_ACT0 + k]]
        here = to_area(obs['player'])
        if here <= AREA_RADIUS:
            # At the labelled point: work the nearest target in reach; WAIT only while none is.
            return next(iter(safest(targets)), (0,))[0]
        # Away from it: a target in reach nearer the labelled point than the player is, else a click toward it.
        closer = [c for c in targets if to_area(c[1]) < here]
        if closer: return min(safest(closer), key=lambda c: to_area(c[1]))[0]
        click = click_to(obs, self.areas[self.skill])
        if click: return click
        # Nothing in sight leads closer: work what is in reach rather than wait. Seven
        # tiles out at Draynor, the teacher waited 1,061 decisions beside choppable trees.
        return next(iter(safest(targets)), (0,))[0]

    def _at_bank(self, obs, inventory, held):
        tool = self.skills[self.skill]['tool']
        if not held:
            for k, item in enumerate(obs['bankSel']):
                if item['count'] and item['name'].lower() == tool:
                    return A_WITHDRAW0 + k
            self.checked_pages.add(obs['bankPage'])
            unchecked = [p for p in range(obs['bankPages']) if p not in self.checked_pages]
            if not unchecked: return A_CLOSE_IF
            return A_BANK_NEXT if unchecked[0] > obs['bankPage'] else A_BANK_PREV
        deposit = [i for i in inventory if i['name'].lower() not in GATHER_TOOLS]
        if not deposit: return A_CLOSE_IF
        selected = deposit[0]; page = selected['slot'] // 8
        if obs['invPage'] < page: return A_INV_NEXT
        if obs['invPage'] > page: return A_INV_PREV
        return A_DEPOSIT0 + selected['slot'] % 8


# The four-goal validator. Its actions are never training labels.
A_DIALOG_CONTINUE = 62
GOAL_WORK = dict(wood=(re.compile(r'^tree$', re.I), 'Chop down'), fish=(re.compile(r'^fishing spot$', re.I), 'Net'),
                 mine=(re.compile(r'^rocks$', re.I), 'Mine'), melee=(re.compile(r'^chicken$', re.I), 'Attack'))
GOAL_PRODUCTS = dict(wood=('Logs',), fish=('Raw shrimps', 'Raw anchovies'), mine=('Copper ore', 'Tin ore', 'Iron ore'))
FOODS = ('Shrimps', 'Bread')
WEAPONS = ('Bronze dagger', 'Bronze sword')  # sim/spec.py START_WEAPONS


class GoalReference:
    """Completes one requested activity from a declared four-goal start, with privileged knowledge of its
    start set (baseline/config.py START_SETS): each activity's area, waypoints from the start and which rocks
    hold copper or tin. It continues dialogs; for melee it wields the carried weapon first, eats at half
    hitpoints and fights back anything else that engages it; it works the nearest offered target of the
    activity (a rock only if the set lists it), else follows the waypoints by clicks (act-v3); stopped short beside a closed gate or door
    (within 2 tiles), it opens it (three tries each). It waits once two items are carried. Every action is an ordinary legal
    action; a refused target is left to the game."""
    def __init__(self, goal, site):
        self.goal, self.site = goal, site
        self.route, self.failed, self.last = list(site['routes'].get(goal, ())), set(), None
        self.opened = Counter()

    def action(self, obs, remaining=None):
        mask, p = obs['mask'], obs['player']
        if (obs.get('dialogOpen') or obs.get('dialogOptions')) and mask[A_DIALOG_CONTINUE]:
            return A_DIALOG_CONTINUE
        here = (p['x'], p['z'])
        ended = (obs.get('feedback') or {}).get('ended') or {}
        if self.last and ended.get('reason') == 'failed' and ended.get('action', -1) in CLICK_ROWS:
            self.failed.add((*self.last, ended['action']))
        self.last = here
        held = lambda name, option: next((A_HELD0 + k for k, h in enumerate(obs.get('heldSel', []))
                                          if h['name'] == name and h['option'] == option and mask[A_HELD0 + k]), None)
        if self.goal == 'melee':
            weapon = next((h['name'] for h in obs.get('heldSel', []) if h['option'] == 'Wield' and h['name'] in WEAPONS), None)
            if weapon and not obs.get('equipment') and held(weapon, 'Wield') is not None:
                return held(weapon, 'Wield')
            if p.get('hp') is not None and 2 * p['hp'] <= p['maxHp']:
                food = next((a for a in (held(f, 'Eat') for f in FOODS) if a is not None), None)
                if food is not None:
                    return food
            # Engaged by something else (the game refuses another target meanwhile): fight it first.
            foe = obs.get('attacker')
            if p.get('inCombat') and foe and foe['name'] != 'Chicken' and not obs.get('optionBusy'):
                back = [A_ACT0 + k for k, a in enumerate(obs.get('actSel', [])) if mask[A_ACT0 + k]
                        and a['name'] == foe['name'] and a['option'] == 'Attack' and (a['dx'], a['dz']) == (foe['dx'], foe['dz'])]
                return back[0] if back else 0
        if obs.get('optionBusy'):
            return 0
        if self.goal in GOAL_PRODUCTS and sum(i['count'] for i in obs['inventory'] if i['name'] in GOAL_PRODUCTS[self.goal]) >= 2:
            return 0
        target, verb = GOAL_WORK[self.goal]
        # Stopped short ("I can't reach that!", a walk that failed or stalled) beside a closed gate or door: open it.
        if ended.get('refused') or ended.get('reason') in ('failed', 'stalled'):
            doors = [(a['distance'], A_ACT0 + k, (a['x'], a['z'])) for k, a in enumerate(obs.get('actSel', []))
                     if mask[A_ACT0 + k] and a['name'] in ('Gate', 'Door') and a['option'] == 'Open' and a['distance'] <= 2
                     and self.opened[(a['x'], a['z'])] < 3]
            if doors:
                _, action, tile = min(doors)
                self.opened[tile] += 1
                return action
        # A declared gate on the way (the Entrana pen): open it when it is offered closed; until it is seen
        # open, walk the waypoints toward it rather than at targets behind it.
        gates = self.site.get('gates', {}).get(self.goal, ())
        if gates:
            # An opened gate stands beside its closed tile, offering Close.
            near = [(a['distance'], a['option'], A_ACT0 + k) for k, a in enumerate(obs.get('actSel', []))
                    if mask[A_ACT0 + k] and a['option'] in ('Open', 'Close')
                    and any(max(abs(a['x'] - x), abs(a['z'] - z)) <= 2 for x, z in gates)]
            closed = [(d, action) for d, option, action in near
                      if option == 'Open' and (obs['actSel'][action - A_ACT0]['x'], obs['actSel'][action - A_ACT0]['z']) in gates]
            if closed:
                return min(closed)[1]
            if not near and self.route and (move := self._follow(obs, here, self.route)):
                return move
        rocks = self.site.get('rocks', ())
        work = [(a['steps'], A_ACT0 + k) for k, a in enumerate(obs.get('actSel', [])) if mask[A_ACT0 + k]
                and target.match(a['name']) and a['option'] == verb
                and (self.goal != 'mine' or (a['x'], a['z']) in rocks)]
        if work:
            return min(work)[1]
        return self._follow(obs, here, [*self.route, self.site['areas'][self.goal]]) or 0

    def _follow(self, obs, here, route):
        """The click that best closes on the first waypoint, from the nearest on, still 4+ tiles away."""
        nearest = min(range(len(route)), key=lambda k: cheb(here, route[k]))
        for point in route[nearest:]:
            click = click_to(obs, point, self.failed) if cheb(here, point) > 3 else None
            if click:
                return click
        return None
