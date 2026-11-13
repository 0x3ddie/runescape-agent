"""The environment agent over the simulated world: a port of env-ts/engine/agent.ts and the
pure env-ts/core modules it uses (actions, affordances, clicks, mask, feedback, items,
decision-points, budgets).

It returns raw observations in the native schema, so rl/obs.py projects them unchanged and
a native observation can be compared with a simulated one field by field. The walk map and
the engine's paths come from the extracted tables; the clicks' and ACT's routes are the
client's tryMove over the live collision (sim/paths.py).
"""
from __future__ import annotations

import math
import re

from act_contract import ACT_V3, CLICK_OFFSETS, CLICK_ROWS, click_slot
from sim import paths
from sim.engine import (APLOC1, APLOCU, APNPC1, APNPCU, APOBJ1, FISHING, HITPOINTS, INV, PRAYER, WALK_BLOCKED,
                        WEARPOS_RHAND, Npc, Player, R_PACE, SKILLS, Unsupported, World, level_for, p_oploc,
                        player_combat_stat, weapon_styles, xp_for)

# env-ts/core/actions.ts (act-v3: the clicks are rl/act_contract.py's CLICK_OFFSETS on CLICK_ROWS)
N_ACTIONS, A_WAIT, A_ACT0, N_ACT = 96, 0, 1, 16
A_HELD0, N_HELD, N_INV_SEL, A_DROP0, A_DEPOSIT0, A_WITHDRAW0 = 29, 8, 8, 37, 45, 53
A_CLOSE_IF, A_DIALOG_CONTINUE, A_DIALOG_CHOICE0, N_DIALOG_CHOICE = 61, 62, 63, 4
A_INV_PREV, A_INV_NEXT, A_BANK_PREV, A_BANK_NEXT, A_STYLE0, A_USE0 = 67, 68, 69, 70, 72, 76  # row 71: no action
DROP_MIN_STACKS = 24
# env-ts/core/mask.ts: the legality rule each observation names (maskVersion); mask-v5 adds the page turns while an
# item is selected for use
MASK_CONTRACT = 'mask-v5'
# an inventory option follows OpHeldHandler and OpHeldUHandler, refused while the
# player is delayed, then clearPendingAction, so an Eat, Wield or Drop ends a running chop, mine, net, attack or light.
# held-v1 (every result before it) ran the option with neither. Each observation (heldVersion) and simulator header
# (held) names it; one without the field is held-v1, and results of the two are never pooled.
HELD_CONTRACT = 'held-v2'
# OpHeldUHandler's two game messages for a use on a held item with no script (part of held-v2): the debug line, which
# names the target item's debugname (native worlds run with the engine's default node.debug), then the literal.
NO_HELDU_TRIGGER, NOTHING_INTERESTING = 'No trigger for [opheldu,{}]', 'Nothing interesting happens.'
TOOL_PROTECT_RE = re.compile(r'axe|pickaxe|fishing net|tinderbox|fishing rod|harpoon|lobster pot|hammer|knife|needle', re.I)
# env-ts/core/budgets.ts and decision-points.ts
K_TICKS, NAV_STALL_TICKS, ACT_STALL_TICKS, OPTION_HARD_TIMEOUT_TICKS = 2, 60, 100, 1500
DECISION_WAIT_TICKS = 100
MAX_DECISIONS, MAX_EPISODE_TICKS = 1500, 20000
# engine/agent.ts and core modules
REFUSED_TICKS, IDLE_GRACE_TICKS = 100, 4
ACT_SCAN_TILES, ACT_CLUSTER_TILES = 15, 5
LOC_SCAN_TILES, GROUND_SCAN_TILES, NPC_SCAN_TILES = 15, 15, 15  # npcs: the server's view distance (rsbuf/build.ts)
N_NPC_OBS, N_LOC_OBS, N_GROUND_OBS = 8, 12, 6
VISIT_CELL = 8
CHAT_KEEP = 16
TAKE, USE = 'Take', 'Use'
FAMILIES = ('wait', 'act', 'click', 'held', 'drop', 'deposit', 'withdraw', 'close', 'dialog', 'page', 'style', 'use')
SKILL_NAMES = ('Attack', 'Defence', 'Strength', 'Hitpoints', 'Ranged', 'Prayer', 'Magic', 'Cooking', 'Woodcutting',
               'Fletching', 'Fishing', 'Firemaking', 'Crafting', 'Smithing', 'Mining', 'Herblore', 'Agility', 'Thieving',
               'Slayer', 'Farming', 'Runecrafting')
STARTER_KIT = (('bronze_axe', 1), ('tinderbox', 1), ('net', 1), ('shrimp', 1), ('bucket_empty', 1), ('pot_empty', 1),
               ('bread', 1), ('bronze_pickaxe', 1), ('bronze_dagger', 1), ('bronze_sword', 1), ('wooden_shield', 1),
               ('shortbow', 1), ('bronze_arrow', 25), ('airrune', 25), ('mindrune', 15), ('waterrune', 6),
               ('earthrune', 4), ('bodyrune', 2))
BANK_NAME_RE = re.compile(r'bank|booth', re.I)
# A new account's position before reset teleports it (entity/Player.ts constructor: Tutorial Island).
LOGIN_TILE = (3094, 3106)


def name_of(t):
    """observe.ts nameOf: the name, else the debugname (only a missing one falls through)."""
    return str(t['name'] if t.get('name') is not None else t['debugname'] if t.get('debugname') is not None else '')


def chebyshev(ax, az, bx, bz):
    return max(abs(ax - bx), abs(az - bz))


def option_label(action):
    """actions.ts optionLabelFor: a click's walk is a 'nav' option, which the tracker clears after NAV_STALL_TICKS
    without a step."""
    if A_ACT0 <= action < A_ACT0 + N_ACT:
        return f'act_{action - A_ACT0}'
    k = click_slot(action)
    return None if k is None else f'nav_click_{k}'


def family(action):
    """feedback.ts familyOf."""
    def within(start, n):
        return start <= action < start + n
    if action == 0:
        return 'wait'
    if click_slot(action) is not None:
        return 'click'
    for name, start, n in (('act', A_ACT0, N_ACT), ('held', A_HELD0, N_HELD),
                           ('drop', A_DROP0, N_INV_SEL), ('deposit', A_DEPOSIT0, N_INV_SEL),
                           ('withdraw', A_WITHDRAW0, N_INV_SEL)):
        if within(start, n):
            return name
    if action == A_CLOSE_IF:
        return 'close'
    if action == A_DIALOG_CONTINUE or within(A_DIALOG_CHOICE0, 4):
        return 'dialog'
    if within(A_INV_PREV, 4):
        return 'page'
    if within(A_STYLE0, 4):
        return 'style'
    if within(A_USE0, N_INV_SEL):
        return 'use'
    raise ValueError(f'no family for action {action}')


def empty_item():
    return dict(slot=-1, id=-1, count=0, name='')


def item_page(items, page, capacity):
    """items.ts itemPage."""
    slots = {i['slot']: i for i in items}
    return [dict(slots[page * 8 + k]) if page * 8 + k in slots and page * 8 + k < capacity else empty_item()
            for k in range(N_INV_SEL)]


def match_item(current, selected):
    return (current is not None and selected is not None and selected['count'] > 0 and current['slot'] == selected['slot']
            and current['id'] == selected['id'] and current['count'] == selected['count'])


class Agent:
    """One clientless agent in a World, as env-ts/engine/agent.ts EngineAgent."""

    def __init__(self, world: World, name='wd1a0'):
        self.world, self.name = world, name
        self.player = None
        self.cfg = {}
        self.decision = self.charged = self.episode_ticks = self.last_mask_sum = 0
        self.option = None
        self.last_option_note = ''
        self.option_end = None
        self.visited = set()
        self.click_offsets = CLICK_OFFSETS  # a slot's offset, or None for a slot the declared table leaves empty
        self.decision_route = None  # route-v1 (env-ts/core/clicks.ts): this decision's click or ACT; evidence only
        self.acting = None  # the ACT row being dispatched, for its route record
        self.style_stale = False  # engine/agent.ts styleStale
        self.inv_page = self.bank_page = 0
        self.bank_pages = 1
        self.refused = {}
        self._loc_cache = None   # (key, loc act entries, loc places): see project_items
        self._scan_locs = None   # (tile, loc version, scanned locs)
        self.last_observation = None
        self.act_targets = []
        self.using = None
        self.pace = 0
        self.chat_seen = 0
        self.chat_seq = self.chat_covered = 0  # chat line sequence numbers: the last arrived, the last published
        self.texts = {}
        self.dead = False
        self.death_tick = None
        self.ended = lambda: False  # a task's terminal cycle has passed: nothing more is simulated

    # ---- ticks (waitTicks in lockstep: advance, then keepAlive)
    def wait_ticks(self, n):
        for _ in range(n):
            if self.ended():
                return
            self.world.cycle()
            self.capture_text()
            self.keep_alive()

    def keep_alive(self):
        if self.player is None:
            return
        self.observe_npcs()
        self.visit()
        if (self.player.hp <= 0 or self.player.dead) and self.death_tick is None:
            self.death_tick = self.world.tick

    def observe_npcs(self):
        """rsbuf npcInfo: npcs on the player's level within 15 tiles (Chebyshev) are observed
        once added; a teleported one drops out for that update."""
        p = self.player
        for npc in self.world.npcs:
            near = npc.active and not npc.tele and npc.level == p.level and chebyshev(p.x, p.z, npc.x, npc.z) <= 15
            npc.observers = 1 if near else 0

    def visit(self):
        p = self.player
        self.visited.add((p.level, p.x // VISIT_CELL, p.z // VISIT_CELL))

    def capture_text(self):
        """TextCapture: chat lines with their tick and an increasing sequence number (16 kept), the
        latest text of each component."""
        log = self.world.chat_log
        for tick, text in log[self.chat_seen:]:
            self.chat_seq += 1
            self.chat.append((text, tick, self.chat_seq))
        self.chat_seen = len(log)
        del self.chat[:-CHAT_KEEP]
        for tick, component, text in self.world.text_log:
            self.texts[component] = (text, tick)
        self.world.text_log.clear()

    # ---- reset
    def reset(self, seed, cfg):
        """EngineAgent.reset with a fresh Player in the persisting world."""
        p, world = self._login(seed, cfg), self.world
        self.style_stale = bool(cfg.get('setStats'))
        if cfg.get('setStats'):
            for pair in cfg['setStats'].split(','):
                name, level = pair.split(':')
                self.set_level(p, SKILL_NAMES.index(name.strip().capitalize()), int(level))
        for name, count in STARTER_KIT:
            p.inv.add(world.t, world.t.obj_id(name), count)
        p.bank.add(world.t, world.t.obj_id('coins'), 25)
        base = _from_tele(cfg.get('spawnTele', '0,50,50,22,22'))
        if seed != 0 or cfg.get('spawnSteps'):
            raise Unsupported('spawn jitter other than reset seed 0')
        return self._arrive(seed, base)

    def start(self, seed, cfg, tile, kit, levels):
        """EngineAgent.reset for a compact-world start (sim/task.py): the same writes, with `levels`
        (stat name -> level) for setStats, `kit` ((debugname, count) in order) for STARTER_KIT, and
        the start tile (x, z) for the spawn."""
        p, world = self._login(seed, cfg), self.world
        self.style_stale = bool(levels)
        for name, level in levels.items():
            self.set_level(p, SKILLS.index(name), level)
        for name, count in kit:
            p.inv.add(world.t, world.t.obj_id(name), count)
        p.bank.add(world.t, world.t.obj_id('coins'), 25)
        return self._arrive(seed, (*tile, 0))

    def _login(self, seed, cfg):
        """replacePlayerForReset: the old player leaves, a new account logs in at its own spawn."""
        self.cfg = dict(cfg)
        jitter = cfg.get('paceJitterTicks', 0)
        self.pace_seed, self.pace = seed, 0
        self.pace_jitter = jitter
        self.inv_page = self.bank_page = 0
        self.bank_pages = 1
        self.using = None
        self.refused.clear()
        self.last_observation = None
        self.decision = self.charged = self.episode_ticks = self.last_mask_sum = 0
        self.option = None
        self.last_option_note = ''
        self.option_end = None
        self.death_tick = None
        world = self.world
        if world.player is not None:
            world._occupy(world.player, False)
        p = Player(world, self.name)
        p.x, p.z = LOGIN_TILE
        p.origin_x, p.origin_z = p.x, p.z
        world.player = None
        self.player = p
        self.chat, self.texts, self.chat_seen = [], {}, len(world.chat_log)
        world.pending_login = (p, login)
        self.wait_ticks(2)
        return p

    def _arrive(self, seed, base):
        """The jump to the spawn, two ticks, then the episode's first observation."""
        p, world = self.player, self.world
        world._occupy(p, False)
        p.x, p.z, p.level = base
        world._occupy(p, True)
        p.path = []
        self.wait_ticks(2)
        if world.t.source(p.level, p.x, p.z) < 0:
            raise Unsupported('spawn outside the supported area')
        self.chat, self.texts = [], {}
        self.chat_seen = len(world.chat_log)
        self.visited.clear()
        self.visit()
        obs = self.build_obs(dict(decision=0, done=False, truncated=False, isDead=False))
        self.project_items(obs)
        obs['mask'] = self.mask(obs)
        obs['feedback'] = self.feedback(None, None, self.snapshot(), False, self.publish_chat(), self.dialog_text())
        obs['seed'] = seed
        self.last_mask_sum = sum(obs['mask'])
        return obs

    def set_level(self, p, stat, level):
        """Player.setLevel."""
        level = min(99, max(1, level))
        p.base_levels[stat] = level
        p.stats[stat] = xp_for(level)
        p.levels[stat] = level

    # ---- option tracking
    def snapshot(self):
        p = self.player
        items = {}
        for inv in (p.inv, p.worn):
            for _, obj, count in inv.listed():
                items[obj] = items.get(obj, 0) + count
        return dict(tick=self.world.tick, x=p.x, z=p.z, level=p.level, hp=p.hp, xp=sum(p.stats), items=items,
                    bankOpen=p.modal_main == 'bank_main',
                    dialogOpen=p.modal_chat is not None or (p.script is not None and p.script[1] == 'pausebutton'))

    def item_state(self):
        return ','.join(f'{s}:{o}:{c}' for s, o, c in self.player.inv.listed())

    def end_option(self, reason, refused):
        if self.option is not None:
            self.option_end = dict(action=self.option.get('action', -1), reason=reason, refused=refused, reported=False)

    def option_busy(self):
        """EngineAgent.optionBusy."""
        o = self.option
        if o is None:
            return False
        now, p = self.world.tick, self.player
        pos = (p.x, p.z)
        if pos != o['last_pos']:
            o['last_pos'], o['last_progress'] = pos, now
        act = o.get('act')
        if act:
            xp, items = sum(p.stats), self.item_state()
            if xp != act['last_xp'] or items != act['last_items'] or p.modal_main is not None:
                act['last_xp'], act['last_items'], o['last_progress'] = xp, items, now
        if o['done']():
            refused = (bool(act) and sum(p.stats) == act['xp'] and self.item_state() == act['items']
                       and p.modal_main is None and (act.get('loc') is None or self.loc_present(act['loc'])))
            if refused:
                self.refused[act['key']] = now + REFUSED_TICKS
            self.last_option_note = f"{o['label']} done"
            self.end_option('done', refused)
            self.option = None
            return False
        stalled = (now - o['last_progress'] > NAV_STALL_TICKS) if o['label'].startswith('nav') else \
            (bool(act) and now - o['last_progress'] > ACT_STALL_TICKS)
        if stalled and act:
            self.refused[act['key']] = now + REFUSED_TICKS
        expired = now - o['start'] > OPTION_HARD_TIMEOUT_TICKS
        if stalled or expired:
            self.last_option_note = f"{o['label']} {'stalled-cleared' if stalled else 'stale-cleared'}"
            self.end_option('stalled' if stalled else 'expired', stalled and bool(act))
            self.option = None
            return False
        return True

    def loc_present(self, loc):
        """Whether a loc of this type still stands on this tile, on its level: (type, x, z, level)."""
        return any(l.type == loc[0] for l in self.world.active_locs_at(loc[3], loc[1], loc[2]))

    def begin(self, label, done):
        now, p = self.world.tick, self.player
        self.option = dict(label=label, start=now, last_pos=(p.x, p.z), last_progress=now, done=done)

    def act_done(self, start):
        def done():
            p = self.player
            idle = p.target is None and not p.has_waypoints()
            return idle and self.world.tick - start > IDLE_GRACE_TICKS
        return done

    # ---- act-v3's routing: the client's tryMove, then the server's handlers, run off
    def route(self, dest, arrival=None, nearest=False):
        """paths.client_route from the player's server tile over its scene (a mid-walk click is measured from there)."""
        w, p = self.world, self.player
        return paths.client_route(w.player_scene(p), p.x, p.z, dest, arrival, nearest)

    def send_route(self, route):
        """MoveClickHandler: clearPendingAction, then the turn points as the walk; a route that is only the player's own
        tile clears the walk and leaves no repath (AllowRepath.NONE), so no chase follows it. processWalktrigger runs no
        script: none of the slice's sets a walk trigger."""
        p = self.player
        p.clear_pending_action()
        if len(route) == 1 and route[0] == (p.x, p.z):
            p.queue_waypoints([])
            p.allow_repath = False
        else:
            p.queue_waypoints(route)

    def record(self, kind, action, target, route, **extra):
        """route-v1 for this decision, as env-ts/engine/agent.ts records it: what the click or ACT asked for and what was
        sent. Evidence for checks and the spectator, never a policy input."""
        p = self.player
        self.decision_route = dict(
            dict(version='route-v1', kind=kind, action=action, tick=self.world.tick, source=dict(x=p.x, z=p.z, level=p.level),
                 offset=None, target=dict(x=target[0], z=target[1]), thing=None, via=route.via if route else 'none',
                 end=dict(x=route[-1][0], z=route[-1][1]) if route else None, refused=None, waypoints=[]), **extra)

    def click(self, action):
        """A minimap click at the slot's offset: tryMove with its 3 x 3 fallback, then MoveClickHandler. Refused while
        the player is delayed, and when no route is found nothing is sent; an earlier walk then continues and the
        decision records the refusal. The option ends when the walk ends, or stalls after NAV_STALL_TICKS."""
        p = self.player
        dx, dz = self.click_offsets[click_slot(action)]
        x, z = p.x + dx, p.z + dz
        route = None if p.delayed else self.route((x, z), nearest=True)
        if route is None:
            self.record('click', action, (x, z), None, offset=dict(dx=dx, dz=dz), refused='delayed' if p.delayed else 'no route')
            self.option_end = dict(action=action, reason='failed', refused=True, reported=False)
            return f"click {dx},{dz}: {'refused while delayed' if p.delayed else 'no route'}"
        self.send_route(route)
        self.record('click', action, (x, z), route, offset=dict(dx=dx, dz=dz), waypoints=[dict(x=a, z=b) for a, b in route])
        self.begin(option_label(action), lambda: not self.player.has_waypoints())
        return f'click {dx},{dz} to {route[-1][0]},{route[-1][1]}'

    def handled(self, target, ok, label, done=None):
        """The interaction an op handler sets after clearPendingAction, and the option that waits for it."""
        self.begin(label, done or self.act_done(self.world.tick))
        return f'{label} issued' if ok else f'{label}: setInteraction refused'

    def interact_loc(self, loc, option, label, using=None):
        """A loc op by the client's rule: tryMove to the loc (its rotated size and forceapproach, or its wall shape; no
        fallback), a route sent as MoveClickHandler takes it, then OpLocHandler or OpLocUHandler: refused while the
        player is delayed; clearPendingAction; the interaction. With no route only the op is sent: an earlier walk
        continues and the op resolves when it ends ("I can't reach that!" if out of reach)."""
        w, p = self.world, self.player
        t = w.t.loc_types[loc.type]
        op = -1 if using else next((i for i, o in enumerate(t['op']) if o and o != 'hidden' and o.lower() == option.lower()), -1)
        if op < 0 and not using:
            return f'{label}: no matching op'
        thing = dict(kind='loc', id=loc.type, name=name_of(t), option='Use' if using else option)
        if p.delayed:
            self.record('act', self.acting, (loc.x, loc.z), None, thing=thing, refused='delayed')
            return f'{label}: refused while delayed'
        route = self.route((loc.x, loc.z), paths.loc_arrival(loc.shape, loc.angle, t['width'], t['length'], t['forceapproach']))
        if route is not None:
            self.send_route(route)
        self.record('act', self.acting, (loc.x, loc.z), route, thing=thing, refused=None if route else 'no route',
                    waypoints=[dict(x=a, z=b) for a, b in route or ()])
        p.clear_pending_action()
        if using:
            p.last_useitem = using['id']
            return f"{self.handled(loc, p.set_interaction(loc, APLOCU), label)} use {using['id']}"
        return self.handled(loc, p.set_interaction(loc, APLOC1 + op), label)

    def interact_npc(self, npc, option, label, using=None):
        """An npc op by the client's rule: tryMove beside its tile (1 x 1, no fallback), then OpNpcHandler or
        OpNpcUHandler: refused while the player or the npc is delayed or while the npc is not in the player's npc list;
        clearPendingAction; the interaction. The server's naive chase follows a moving npc (World.path_to_pathing_target)."""
        p = self.player
        op = -1 if using else next((i for i, o in enumerate(self.world.t.npc_types[npc.type]['op'])
                                    if o and o != 'hidden' and o.lower() == option.lower()), -1)
        if op < 0 and not using:
            return f'{label}: no matching op'
        thing = dict(kind='npc', id=npc.type, name=name_of(self.world.t.npc_types[npc.type]), option='Use' if using else option)
        refused = 'delayed' if p.delayed else 'npc delayed' if npc.delayed else 'npc not in view' if not npc.observers else None
        if refused:
            self.record('act', self.acting, (npc.x, npc.z), None, thing=thing, refused=refused)
            return f'{label}: refused ({refused})'
        route = self.route((npc.x, npc.z), paths.ENTITY_ARRIVAL)
        if route is not None:
            self.send_route(route)
        self.record('act', self.acting, (npc.x, npc.z), route, thing=thing, refused=None if route else 'no route',
                    waypoints=[dict(x=a, z=b) for a, b in route or ()])
        p.clear_pending_action()
        if using:
            p.last_useitem = using['id']
            return f"{self.handled(npc, p.set_interaction(npc, APNPCU), label)} use {using['id']}"
        return self.handled(npc, p.set_interaction(npc, APNPC1 + op), label)

    def pick_up(self, obj):
        """A ground item's Take by the client's rule: tryMove onto its tile, else beside it, then OpObjHandler: refused
        while the player is delayed; clearPendingAction; the interaction on its Take op."""
        w, p = self.world, self.player
        ops = w.t.obj_types[obj.type]['op'] or []
        op = next((i for i, o in enumerate(ops) if o and 'take' in o.lower()), 2)
        name = name_of(w.t.obj_types[obj.type])
        thing = dict(kind='obj', id=obj.type, name=name, option='Take')
        if p.delayed:
            self.record('act', self.acting, (obj.x, obj.z), None, thing=thing, refused='delayed')
            return f'pick {name}: refused while delayed'
        onto = self.route((obj.x, obj.z))
        route = onto or self.route((obj.x, obj.z), paths.ENTITY_ARRIVAL)
        if route is not None:
            self.send_route(route)
        self.record('act', self.acting, (obj.x, obj.z), route, thing=thing, refused=None if route else 'no route',
                    waypoints=[dict(x=a, z=b) for a, b in route or ()], via=onto.via if onto else 'adjacent' if route else 'none')
        p.clear_pending_action()
        ok = p.set_interaction(obj, APOBJ1 + op)
        self.begin('pick_item', lambda: self.player.target is None)
        return f'pick {name}' if ok else f'pick {name}: setInteraction refused'

    # ---- dispatch
    def act(self, action):
        self.acting = action
        k = action - A_ACT0
        selected = self.last_observation['actSel'][k] if k < len(self.last_observation['actSel']) else None
        source = self.act_targets[k] if k < len(self.act_targets) else None
        if not selected or source is None:
            return 'act: slot empty'
        label = option_label(action)
        what = f"{selected['option']} {selected['kind']}={selected['id']}@{selected['x']},{selected['z']},{selected['level']}"
        using, self.using = self.using, None
        p = self.player
        if selected['kind'] == 'loc':
            here = self.world.active_locs_at(p.level, selected['x'], selected['z'])
            loc = next((l for l in here if l.type == selected['id']), None)
            if loc is None or chebyshev(loc.x, loc.z, p.x, p.z) > LOC_SCAN_TILES:
                return f'act: {what} gone'
            note = self.interact_loc(loc, selected['option'], label, using)
        elif selected['kind'] == 'npc':
            npc = source
            if not npc.is_valid() or npc.level != p.level or chebyshev(npc.x, npc.z, p.x, p.z) > NPC_SCAN_TILES:
                return f'act: {what} gone'
            note = self.interact_npc(npc, selected['option'], label, using)
        else:
            obj = next((o['obj'] for o in self._scan['objs'] if o['id'] == selected['id'] and o['x'] == selected['x']
                        and o['z'] == selected['z']), None)
            if obj is None:
                return f'act: {what} gone'
            note = self.pick_up(obj)
        if self.option is not None:
            xp, items = sum(self.player.stats), self.item_state()
            loc = (selected['id'], selected['x'], selected['z'], selected['level']) if selected['kind'] == 'loc' else None
            self.option['act'] = dict(key=affordance_key(selected), xp=xp, items=items, last_xp=xp, last_items=items, loc=loc)
        return f'{note} {what}'

    def item_transfer(self, index, withdraw):
        p, w = self.player, self.world
        if p.modal_main != 'bank_main':
            return 'bank: closed'
        selected = self.last_observation['bankSel' if withdraw else 'invSel'][index]
        items = self.bank_items() if withdraw else self.inventory_items()
        current = next((i for i in items if i['slot'] == selected['slot']), None)
        if not match_item(current, selected):
            return 'bank: stale or empty selection'
        label = 'bank_withdraw' if withdraw else 'bank_deposit'
        w.bank_transfer(p, selected['slot'], 1 if withdraw else selected['count'], withdraw)
        return f"{label}: slot {selected['slot']} id {selected['id']} requested {1 if withdraw else selected['count']}"

    def resume_dialog(self, choice):
        p, w = self.player, self.world
        if p.script is None or p.script[1] != 'pausebutton':
            return 'dialog: nothing suspended'
        buttons = list(p.resume_buttons)
        if choice is not None and buttons:
            com = buttons[min(choice, len(buttons) - 1)]
            p.last_com = com
            p.resume_buttons = []
            w.resume(p)
            return f'dialog choice {choice} -> com {com}'
        if buttons:
            return 'dialog: choice required'
        w.resume(p)
        return 'dialog continue'

    def dispatch(self, action):
        p, obs = self.player, self.last_observation
        if not obs['mask'][action]:
            return 'action not legal in published observation'
        if action in (A_INV_PREV, A_INV_NEXT, A_BANK_PREV, A_BANK_NEXT):
            if action == A_INV_PREV:
                self.inv_page = max(0, self.inv_page - 1)
            elif action == A_INV_NEXT:
                self.inv_page = min(3, self.inv_page + 1)
            elif action == A_BANK_PREV:
                self.bank_page = max(0, self.bank_page - 1)
            else:
                self.bank_page = min(self.bank_pages - 1, self.bank_page + 1)
            return 'item page changed'
        if A_WITHDRAW0 <= action < A_WITHDRAW0 + N_INV_SEL:
            return self.item_transfer(action - A_WITHDRAW0, True)
        if A_DEPOSIT0 <= action < A_DEPOSIT0 + N_INV_SEL:
            return self.item_transfer(action - A_DEPOSIT0, False)
        if A_DROP0 <= action < A_DROP0 + N_INV_SEL:
            selected = obs['invSel'][action - A_DROP0]
            current = next((i for i in self.inventory_items() if i['slot'] == selected['slot']), None)
            return self.drop_slot(selected['slot']) if match_item(current, selected) else 'drop: stale selection'
        if action == A_WAIT:
            return 'wait'
        if A_USE0 <= action < A_USE0 + N_INV_SEL:
            selected = obs['invSel'][action - A_USE0]
            current = next((i for i in self.inventory_items() if i['slot'] == selected['slot']), None)
            if not match_item(current, selected):
                return 'use: stale selection'
            self.using = dict(slot=selected['slot'], id=selected['id'], name=selected['name'])
            return f"use: picked {selected['name']}"
        if self.using and action == A_CLOSE_IF:
            self.using = None
            return 'use: put back'
        if self.using and A_HELD0 <= action < A_HELD0 + N_HELD:
            target = obs['heldSel'][action - A_HELD0] if action - A_HELD0 < len(obs['heldSel']) else None
            using, self.using = self.using, None
            if target is None:
                return 'use: slot empty'
            return self.use_on_held(target['slot'], using)
        if click_slot(action) is not None:
            return self.click(action)
        if A_ACT0 <= action < A_ACT0 + N_ACT:
            return self.act(action)
        if A_HELD0 <= action < A_HELD0 + N_HELD:
            selected = obs['heldSel'][action - A_HELD0] if action - A_HELD0 < len(obs['heldSel']) else None
            current = next((i for i in self.inventory_items() if i['slot'] == (selected or {}).get('slot')), None)
            return self.op_held(selected['slot'], selected['option']) if selected and match_item(current, selected) \
                else 'held: stale selection'
        if action == A_DIALOG_CONTINUE:
            return self.resume_dialog(None)
        if A_DIALOG_CHOICE0 <= action < A_DIALOG_CHOICE0 + 4:
            return self.resume_dialog(action - A_DIALOG_CHOICE0)
        if A_STYLE0 <= action < A_STYLE0 + 4:
            # [proc,set_attackstyle], executed protected: not at all while delayed or protected
            if not (p.protect or p.delayed):
                p.close_modal()
                p.clear_pending_action()
                p.varps['com_mode'] = action - A_STYLE0
                player_combat_stat(self.world, p)
                self.style_stale = False
            return f'style {action - A_STYLE0}'
        if action == A_CLOSE_IF:
            if p.script is not None and p.script[1] == 'pausebutton':
                return self.resume_dialog(0 if p.resume_buttons else None)
            p.close_modal()
            return 'close_if'
        return f'unimplemented action {action}'

    def op_held(self, slot, option):
        """EngineAgent.opHeld (OpHeldHandler, held-v2): refused while the player is delayed; the item's iop
        matching the option; lastItem; clearPendingAction (the backpack is never the main modal's layer), which
        ends a running activity; then the OPHELD trigger, run protected."""
        w, p = self.world, self.player
        item = p.inv.items[slot]
        if item is None:
            return 'opheld: empty slot'
        if p.delayed:
            return 'opheld: refused while delayed'
        t = w.t.obj_types[item[0]]
        op = next((k for k, o in enumerate(t['iop'] or []) if o and o.lower() == option.lower()), -1)
        if op < 0:
            return f"opheld: no matching iop on {t['debugname']}"
        p.last_item, p.last_slot = item[0], slot
        p.clear_pending_action()
        script = t['triggers'].get(f'OPHELD{op + 1}')
        if not script:
            return f'opheld: no script for op {op + 1}'
        w.run_held(p, script)
        return f"opheld {t['iop'][op]} slot {slot}"

    def drop_slot(self, slot):
        """EngineAgent.dropSlot (OpHeldHandler's op 5, held-v2): refused while delayed; lastItem;
        clearPendingAction; the OPHELD5 trigger."""
        w, p = self.world, self.player
        item = p.inv.items[slot]
        if item is None:
            return 'drop: empty slot'
        if p.delayed:
            return 'drop: refused while delayed'
        p.last_item, p.last_slot = item[0], slot
        p.clear_pending_action()
        script = w.t.obj_types[item[0]]['triggers'].get('OPHELD5')
        if not script:
            raise Unsupported('a drop with no OPHELD5 script')
        w.run_held(p, script)
        return f'drop slot {slot}'

    def use_on_held(self, slot, using):
        """EngineAgent.useOnHeld: OpHeldUHandler, refused while delayed (held-v2), then its lookup, swaps included
        (the target's own script, the used item's, the target's category's, the used item's category's); with none,
        the handler's two game messages: NO_HELDU_TRIGGER of the target's debugname, then NOTHING_INTERESTING."""
        w, p = self.world, self.player
        item = p.inv.items[slot]
        if item is None:
            return 'use: empty slot'
        if p.delayed:
            return 'use: refused while delayed'
        p.last_item, p.last_slot, p.last_useitem, p.last_useslot = item[0], slot, using['id'], using['slot']

        def swap():
            p.last_item, p.last_useitem = p.last_useitem, p.last_item
            p.last_slot, p.last_useslot = p.last_useslot, p.last_slot

        def script(obj, category):
            # the archive keeps getByTrigger's choice: an obj's own script, else its category's ('_name')
            name = w.t.obj_types[obj]['triggers'].get('OPHELDU')
            own = name is not None and not name.split(',')[1].startswith('_')
            return name if name and name != '[opheldu,_]' and own != category else None
        found = script(item[0], False)
        if not found:
            found = script(using['id'], False)
            swap()
        if not found and w.t.obj_types[item[0]]['category'] != -1:
            found = script(item[0], True)
        if not found and w.t.obj_types[using['id']]['category'] != -1:
            found = script(using['id'], True)
            swap()
        p.clear_pending_action()
        if not found:  # OpHeldUHandler.ts:117-126: the debug line (native worlds run with node.debug), then its literal
            w.mes(NO_HELDU_TRIGGER.format(w.t.obj_types[item[0]]['debugname']))
            w.mes(NOTHING_INTERESTING)
            return 'use: nothing interesting happens'
        w.run_held(p, found)
        return f"use {using['id']} on {item[0]}"

    # ---- step
    def step(self, action):
        """EngineAgent.step: dispatch, K_TICKS plus jitter, then at decision points keep
        waiting in paced chunks while the option runs and nothing happens."""
        start = self._begin(action)
        dead_at_start = start[2]
        ticks = K_TICKS + self.pace_ticks()
        self.episode_ticks += ticks
        if not dead_at_start:
            self.wait_ticks(ticks)
        if self.cfg.get('decisionPoints') and not dead_at_start:
            hp_at, waited = self.player.hp, 0
            while self.keep_waiting(hp_at, waited) and not self.truncated() and not self.ended():
                chunk = K_TICKS + self.pace_ticks()
                self.episode_ticks += chunk
                waited += chunk
                self.wait_ticks(chunk)
        return self._finish(action, *start)

    def step_to(self, action, end_tick):
        """step, but the decision ends at end_tick, as a replay of a native decision does
        (native jitter is not reproducible, its ticks are recorded). The option is checked only
        at chunk boundaries, as step checks it: chunks of K_TICKS, the last one longer if the
        span is odd. Returns the observation and the first boundary before end_tick at which
        this world would have stopped waiting, or None."""
        start = self._begin(action)
        began, would_stop = self.world.tick, None
        span = end_tick - began
        self.episode_ticks += span
        if start[2] or span <= 0:
            return self._finish(action, *start), None
        chunks = [K_TICKS] * max(1, span // K_TICKS)
        chunks[-1] += span - sum(chunks)
        hp_at, waited = None, 0
        for k, chunk in enumerate(chunks):
            self.wait_ticks(chunk)
            if k == 0:
                hp_at = self.player.hp  # taken after the first chunk, as step does
            else:
                waited += chunk
            last = k == len(chunks) - 1
            if not last and would_stop is None and (not self.cfg.get('decisionPoints')
                                                    or not self.keep_waiting(hp_at, waited)):
                would_stop = self.world.tick
        return self._finish(action, *start), would_stop

    def keep_waiting(self, hp_at, waited):
        """decision-points.ts keepWaiting."""
        p = self.player
        dialog = (bool(p.modal_state) or p.modal_chat is not None
                  or (p.script is not None and p.script[1] == 'pausebutton'))  # modalState, modalChat, PAUSEBUTTON
        return self.option_busy() and p.hp > 0 and not dialog and p.hp >= hp_at and waited < DECISION_WAIT_TICKS

    def _begin(self, action):
        self.decision += 1
        if self.last_mask_sum > 1:
            self.charged += 1
        meaning = describe_action(self.last_observation, action, self.using)
        before = self.snapshot()
        dead_at_start = self.player.hp <= 0 or self.player.dead
        busy = not dead_at_start and self.option_busy()
        running = self.option
        self.decision_route = None
        if dead_at_start:
            self.option = None
            note = 'terminal: action not dispatched'
        elif busy and (action == 0 or option_label(action) == (self.option or {}).get('label')):
            note = f"continue:{self.option['label']}"
        else:
            if busy:
                self.last_option_note = f"{self.option['label']} superseded by action {action}"
                self.end_option('superseded', False)
                self.option = None
            note = self.dispatch(action)
        if self.option is not None and self.option is not running:
            self.option['action'] = action
        return meaning, before, dead_at_start, note

    def _finish(self, action, meaning, before, dead_at_start, note):
        dead = self.player.hp <= 0 or self.player.dead
        if dead:
            self.option = None
        truncated = not dead and self.truncated()
        obs = self.build_obs(dict(decision=self.decision, done=dead or truncated, truncated=truncated, isDead=dead))
        busy_after = not dead and self.option_busy()
        self.project_items(obs)
        obs['deathTick'] = self.death_tick
        obs['action'] = action
        obs['actionNote'] = note
        obs['optionBusy'] = busy_after
        obs['optionAction'] = self.option.get('action', -1) if self.option else -1
        obs['lastOptionNote'] = self.last_option_note
        obs['route'] = self.decision_route  # evidence for checks, logs and the spectator, not a policy input
        obs['mask'] = self.mask(obs)
        obs['feedback'] = self.feedback(meaning, before, self.snapshot(), busy_after, self.publish_chat(), self.dialog_text())
        obs['charged'] = self.charged
        self.last_mask_sum = sum(obs['mask'])
        return obs

    def publish_chat(self):
        """The chat of an observation: every line whose sequence is above the last one an earlier
        observation covered, each once, aged from its arrival; no tick decides inclusion."""
        lines = [dict(text=t, age=self.world.tick - tick) for t, tick, seq in self.chat if seq > self.chat_covered]
        self.chat_covered = self.chat_seq
        return lines

    def pace_ticks(self):
        """0..paceJitterTicks extra ticks; native draws from Math.random, so only the
        distribution matches."""
        self.pace += 1
        return int(self.world.rng(R_PACE, self.pace_seed, self.pace, self.decision) * (self.pace_jitter + 1))

    def truncated(self):
        return (self.charged >= self.cfg.get('maxDecisions', MAX_DECISIONS)
                or self.episode_ticks >= self.cfg.get('maxEpisodeTicks', MAX_EPISODE_TICKS))

    # ---- observation
    def inventory_items(self):
        t = self.world.t
        return [dict(slot=s, id=o, count=c, name=name_of(t.obj_types[o])) for s, o, c in self.player.inv.listed()]

    def bank_items(self):
        t = self.world.t
        return [dict(slot=s, id=o, count=c, name=name_of(t.obj_types[o])) for s, o, c in self.player.bank.listed()]

    def scan(self):
        """observe.ts scanWorld: on the player's level, locs and ground items within 15, npcs
        within 52, nearest first with ties by position then id."""
        p, w = self.player, self.world
        if self._scan_locs is None or self._scan_locs[:2] != ((p.level, p.x, p.z), w.loc_version):
            locs = []
            for loc in w.locs_near(p.level, p.x, p.z, LOC_SCAN_TILES):
                t = w.t.loc_types[loc.type]
                raw = [o if o and o != 'hidden' else None for o in (t['op'] or [])]
                locs.append(dict(loc=loc, name=name_of(t), options=[o for o in raw if o], raw=raw,
                                 distance=chebyshev(p.x, p.z, loc.x, loc.z)))
            locs.sort(key=lambda l: (l['distance'], l['loc'].x, l['loc'].z, l['loc'].type))
            self._scan_locs = ((p.level, p.x, p.z), w.loc_version, locs)
        locs = self._scan_locs[2]
        npcs = []
        for npc in w.npcs:
            if not npc.active or npc.level != p.level or chebyshev(p.x, p.z, npc.x, npc.z) > NPC_SCAN_TILES:
                continue
            t = w.t.npc_types[npc.type]
            raw = [o if o and o != 'hidden' else None for o in (t['op'] or [])]
            npcs.append(dict(npc=npc, name=name_of(t), options=[o for o in raw if o], raw=raw,
                             distance=chebyshev(p.x, p.z, npc.x, npc.z), combatLevel=t['vislevel'] or 0))
        npcs.sort(key=lambda n: (n['distance'], n['npc'].x, n['npc'].z, n['npc'].nid))
        objs = [dict(obj=o, id=o.type, x=o.x, z=o.z, count=o.count, name=name_of(w.t.obj_types[o.type]),
                     distance=chebyshev(p.x, p.z, o.x, o.z))
                for o in w.objs if o.is_valid() and o.level == p.level and chebyshev(p.x, p.z, o.x, o.z) <= GROUND_SCAN_TILES]
        objs.sort(key=lambda o: (o['distance'], o['x'], o['z'], o['id']))
        return dict(locs=locs, npcs=npcs, objs=objs)

    def build_obs(self, meta):
        p, w = self.player, self.world
        scan = self.scan()
        self._scan = scan
        t = w.t
        skills = [dict(name=name, level=p.levels[i], baseLevel=p.base_levels[i], xp=p.stats[i] // 10)
                  for i, name in enumerate(SKILL_NAMES)]
        inventory = [dict(slot=s, id=o, name=name_of(t.obj_types[o]), count=c,
                          options=[x for x in (t.obj_types[o]['iop'] or []) if x and x != 'hidden'])
                     for s, o, c in p.inv.listed()]
        equipment = [dict(slot=s, id=o, name=name_of(t.obj_types[o])) for s, o, c in p.worn.listed()]
        npcs = [dict(id=n['npc'].nid, name=n['name'], combatLevel=n['combatLevel'], dx=n['npc'].x - p.x,
                     dz=n['npc'].z - p.z, distance=n['distance'], inCombat=n['npc'].target is not None,
                     options=n['options']) for n in scan['npcs'][:N_NPC_OBS]]
        locs = [dict(id=l['loc'].type, name=l['name'], dx=l['loc'].x - p.x, dz=l['loc'].z - p.z, distance=l['distance'],
                     options=l['options']) for l in scan['locs'][:N_LOC_OBS]]
        suspended = p.script is not None and p.script[1] == 'pausebutton'
        attackers = [n for n in scan['npcs'] if n['npc'].target is p]
        return dict(
            type='obs', tick=w.tick, decision=meta['decision'], done=meta['done'], truncated=meta['truncated'],
            ground=[dict(id=o['id'], name=o['name'], count=o['count'], dx=o['x'] - p.x, dz=o['z'] - p.z,
                         distance=o['distance'], x=o['x'], z=o['z']) for o in scan['objs'][:N_GROUND_OBS]],
            invSel=[dict(slot=i['slot'], id=i['id'], name=i['name'], count=i['count'], options=i['options'])
                               for i in inventory[:N_INV_SEL]],
            dialogOptions=[dict(text='', componentId=c) for c in p.resume_buttons] if suspended else [],
            player=dict(x=p.x, z=p.z, level=p.level, hp=p.hp, maxHp=p.base_levels[HITPOINTS], combatLevel=p.combat_level(),
                        animId=-1, isDead=meta.get('isDead', p.hp <= 0), inCombat=bool(attackers),
                        targetIndex=attackers[0]['npc'].nid if attackers else -1),
            skills=skills, inventory=inventory, equipment=equipment, npcs=npcs, locs=locs,
            bankOpen=p.modal_main is not None and any(BANK_NAME_RE.search(l['name']) and l['distance'] <= 3 for l in scan['locs']),
            dialogOpen=suspended or p.modal_chat is not None,
            modalOpen=bool(p.modal_state),
            gameMessages=[])

    def project_items(self, obs):
        """EngineAgent.projectItems."""
        p, w = self.player, self.world
        obs['bankOpen'] = p.modal_main == 'bank_main'
        visible = self.bank_items() if obs['bankOpen'] else []
        self.bank_pages = max(1, math.ceil((max([-1] + [i['slot'] for i in visible]) + 1) / N_INV_SEL))
        self.bank_page = min(self.bank_page, self.bank_pages - 1) if obs['bankOpen'] else 0
        obs['contractVersion'] = 'item-v2'
        obs['invPage'], obs['bankPage'], obs['bankPages'] = self.inv_page, self.bank_page, self.bank_pages
        obs['invSel'] = item_page(obs['inventory'], self.inv_page, 28)
        obs['bankSel'] = item_page(visible, self.bank_page, 240)
        if self.using is not None:
            picked = p.inv.items[self.using['slot']]
            if picked is None or picked[0] != self.using['id'] or obs['bankOpen']:
                self.using = None
        obs['using'] = dict(self.using) if self.using else None
        src = w.t.source(p.level, p.x, p.z)
        if src < 0:
            raise Unsupported('player outside the supported area')

        def steps(c):
            side = self.operable_steps(src, c) if c['kind'] == 'loc' else -1
            beside = paths.walk_approach(w.t, src, p.x, p.z, c['x'], c['z'])
            return beside if side < 0 else side if beside < 0 else min(side, beside)
        refused = lambda a: self.refused.get(affordance_key(a), -1) > w.tick
        candidates = act_candidates(self._scan, w.t)
        sources = [c for c in candidates if c['kind'] != 'obj'] if obs['using'] else candidates
        offered = [dict(c, options=[USE]) for c in sources] if obs['using'] else sources

        # Loc entries depend only on this tile, the loc states, the build area, the refused targets and the picked
        # item; npcs and ground items are added fresh. Groups are per kind, so the merged lists are the uncached ones.
        key = (p.level, p.x, p.z, w.loc_version, p.origin_x >> 3, p.origin_z >> 3, bool(obs['using']),
               frozenset(k for k, until in self.refused.items() if until > w.tick))
        if self._loc_cache is None or self._loc_cache[0] != key:
            locs = [c for c in offered if c['kind'] == 'loc']
            self._loc_cache = (key, affordance_entries(locs, obs['player'], lambda c: w.path(lambda: steps(c)), refused))
        _, loc_entries = self._loc_cache
        others = affordance_entries([c for c in offered if c['kind'] != 'loc'], obs['player'],
                                    lambda c: w.path(lambda: steps(c)), refused)
        chosen = choose_affordances(loc_entries + others)
        obs['actTargetVersion'] = ACT_V3
        obs['actSel'] = [dict(a) for a, _ in chosen]
        obs['heldSel'] = select_held(obs['invSel'], obs['using'])
        self.act_targets = [c['entry'] for _, c in chosen]
        obs['clicks'] = self.click_slots()
        attacker = next((n for n in self._scan['npcs'] if n['npc'].target is p), None)
        rhand = p.worn.items[WEARPOS_RHAND][0] if p.worn.items[WEARPOS_RHAND] else None
        obs['style'] = dict(mode=p.varps['com_mode'], count=len(weapon_styles(w, rhand)['damagestyle']), stale=self.style_stale)
        obs['attacker'] = (dict(name=attacker['name'], dx=attacker['npc'].x - p.x, dz=attacker['npc'].z - p.z)
                           if attacker else None)
        self.last_observation = obs

    def operable_steps(self, src, c):
        """reach.ts walk.operable: a side of the loc origin with no wall between (SDK snapshot)."""
        w, p = self.world, self.player
        tile = int(w.t.tile(p.level, c['x'], c['z']))
        if tile < 0:
            raise Unsupported(f"no snapshot wall data for the loc at {c['x']},{c['z']}")
        sides = int(w.t.sides[tile])
        best = -1
        for k, (dx, dz) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            if not sides >> k & 1:
                continue
            s = paths.walk_steps(w.t, src, p.x, p.z, c['x'] + dx, c['z'] + dz)
            if s >= 0 and (best < 0 or s < best):
                best = s
        return best

    def click_slots(self):
        """clicks.ts clickSlots: each slot's offset, whether its target lies in the declared world (the data area: the
        simulated layout) and whether the target tile's live collision has none of WALK_BLOCKED (walls and npcs aside)."""
        p, w = self.player, self.world
        out = []
        for offset in self.click_offsets:
            if offset is None:
                out.append(dict(dx=0, dz=0, inside=False, open=False))
                continue
            x, z = p.x + offset[0], p.z + offset[1]
            out.append(dict(dx=offset[0], dz=offset[1], inside=w.inside(x, z),
                            open=not w.flags(p.level, x, z, npc_bits=False) & WALK_BLOCKED))
        return out

    def mask(self, obs):
        obs['maskVersion'], obs['heldVersion'] = MASK_CONTRACT, HELD_CONTRACT
        return build_mask(obs)

    def dialog_text(self):
        p = self.player
        chat = p.modal_chat
        text = '' if chat is None else ' '.join(v[0].strip() for k, v in sorted(self.texts.items(), key=lambda kv: kv[0][1])
                                                if k[0] == chat and v[0].strip())
        suspended = p.script is not None and p.script[1] == 'pausebutton'
        choices = [self.texts.get(c, ('', 0))[0].strip() for c in p.resume_buttons] if suspended else []
        return dict(text=text, choices=choices)

    def feedback(self, action, before, after, running, chat, dialog):
        none = dict(ticks=0, dx=0, dz=0, levelChanged=False, xp=0, hp=0, gained=0, lost=0, bankOpened=False,
                    dialogOpened=False)
        ended = None
        end = self.option_end
        if end is not None and not end['reported']:
            ended = dict(action=end['action'], reason=end['reason'], refused=end['refused'])
            end['reported'] = True
        changes = none if before is None else _changes(before, after)
        return dict(version='feedback-v2', action=action, running=running, ended=ended, **changes, chat=chat, dialog=dialog)


# ---- pure helpers (env-ts/core) -----------------------------------------------------------
def _from_tele(tele):
    """A tele string (level, mx, mz, lx, lz) as (x, z, level)."""
    c = [int(v) for v in tele.split(',')]
    return (c[1] << 6) + c[3], (c[2] << 6) + c[4], c[0]


def _changes(before, after):
    gained = lost = 0
    for obj in set(before['items']) | set(after['items']):
        delta = after['items'].get(obj, 0) - before['items'].get(obj, 0)
        if delta > 0:
            gained += delta
        else:
            lost -= delta
    return dict(ticks=after['tick'] - before['tick'], dx=after['x'] - before['x'], dz=after['z'] - before['z'],
                levelChanged=after['level'] != before['level'], xp=after['xp'] - before['xp'],
                hp=after['hp'] - before['hp'], gained=gained, lost=lost,
                bankOpened=after['bankOpen'] and not before['bankOpen'],
                dialogOpened=after['dialogOpen'] and not before['dialogOpen'])


def describe_action(obs, action, using):
    """feedback.ts describeAction."""
    fam = family(action)
    meaning = dict(action=action, family=fam, kind='', name='', option='', item=(using or {}).get('name', '') if obs else '')
    if obs is not None and obs.get('using'):
        meaning['item'] = obs['using']['name']
    def slot(key, start):
        lst = (obs or {}).get(key) or []
        k = action - start
        return lst[k] if 0 <= k < len(lst) else None
    if fam == 'act':
        a = slot('actSel', A_ACT0)
        if a:
            meaning.update(kind=a['kind'], name=a['name'], option=a['option'])
    elif fam == 'click':
        c = ((obs or {}).get('clicks') or [None] * len(CLICK_ROWS))[click_slot(action)]
        if c:
            meaning['option'] = f"{c['dx']},{c['dz']}"
    elif fam == 'held':
        h = slot('heldSel', A_HELD0)
        if h:
            meaning.update(kind='obj', name=h['name'], option='Use' if (obs or {}).get('using') else h['option'])
    elif fam in ('drop', 'deposit', 'use'):
        i = slot('invSel', A_DROP0 if fam == 'drop' else A_DEPOSIT0 if fam == 'deposit' else A_USE0)
        if i and i['slot'] >= 0:
            meaning.update(kind='obj', name=i['name'], option={'drop': 'Drop', 'deposit': 'Deposit', 'use': 'Use'}[fam], item='')
    elif fam == 'withdraw':
        i = slot('bankSel', A_WITHDRAW0)
        if i and i['slot'] >= 0:
            meaning.update(kind='obj', name=i['name'], option='Withdraw')
    return meaning


def affordance_key(a):
    return f"{a['kind']}:{a['id']}:{a['x']}:{a['z']}:{a['level']}:{a['option'].lower()}"


def act_candidates(scan, t):
    """observe.ts actCandidates: every scanned loc, npcs within 15, ground items."""
    out = [dict(kind='loc', id=l['loc'].type, name=l['name'], options=l['options'], x=l['loc'].x, z=l['loc'].z,
                level=l['loc'].level, entry=l['loc']) for l in scan['locs']]
    out += [dict(kind='npc', id=n['npc'].type, name=n['name'], options=n['options'], x=n['npc'].x, z=n['npc'].z,
                 level=n['npc'].level, combatLevel=n['combatLevel'], entry=n['npc'])
            for n in scan['npcs'] if n['distance'] <= ACT_SCAN_TILES]
    out += [dict(kind='obj', id=o['id'], name=o['name'], options=[TAKE], x=o['x'], z=o['z'], level=o['obj'].level,
                 entry=o['obj']) for o in scan['objs']]
    return out


def affordance_entries(candidates, player, steps, refused):
    """affordances.ts selectAffordances, up to its ranking: one entry per new option of each
    candidate in reach, counted with same-group entries within ACT_CLUSTER_TILES."""
    all_, seen = [], set()
    for c in candidates:
        if c['level'] != player['level'] or not c['name'] \
                or chebyshev(c['x'], c['z'], player['x'], player['z']) > ACT_SCAN_TILES:
            continue
        options = list(dict.fromkeys(o for o in c['options'] if isinstance(o, str) and o))
        if not options:
            continue
        walk = steps(c)
        if walk < 0:
            continue
        for option in options:
            a = dict(kind=c['kind'], id=c['id'], name=c['name'], option=option, x=c['x'], z=c['z'], level=c['level'],
                     dx=c['x'] - player['x'], dz=c['z'] - player['z'], distance=chebyshev(c['x'], c['z'], player['x'], player['z']),
                     steps=walk, count=1, combatLevel=c.get('combatLevel', 0))
            key = affordance_key(a)
            if key in seen or refused(a):
                continue
            seen.add(key)
            all_.append((a, c))
    for a, _ in all_:
        a['count'] = sum(1 for b, _ in all_ if _group(b) == _group(a) and chebyshev(a['x'], a['z'], b['x'], b['z']) <= ACT_CLUSTER_TILES)
    return all_


def _group(a):
    return f"{a['kind']}:{a['name'].lower()}:{a['option'].lower()}"


def choose_affordances(all_):
    """The rest of selectAffordances: by steps, one of each group first, then the nearest."""
    order = lambda ac: (ac[0]['steps'], ac[0]['x'], ac[0]['z'], ac[0]['kind'], ac[0]['id'], ac[0]['option'])
    all_ = sorted(all_, key=order)
    covered, chosen = set(), []
    for ac in all_:
        if len(chosen) == N_ACT:
            break
        if _group(ac[0]) not in covered:
            covered.add(_group(ac[0]))
            chosen.append(ac)
    for ac in all_:
        if len(chosen) < N_ACT and all(ac is not x for x in chosen):
            chosen.append(ac)
    return sorted(chosen, key=order)


def select_held(page, using):
    held = []
    for item in page:
        if not item['count'] > 0 or (using and item['slot'] == using['slot']):
            continue
        for option in dict.fromkeys([USE] if using else item.get('options') or []):
            if len(held) == N_HELD:
                return held
            if not option or option.lower() == 'drop':
                continue
            held.append(dict(slot=item['slot'], id=item['id'], name=item['name'], count=item['count'], option=option))
    return held


def build_mask(obs):
    """mask.ts buildMask (no allow list)."""
    style = obs['style']
    if not (type(style['mode']) is int and style['mode'] >= 0 and type(style['count']) is int and style['count'] >= 1
            and type(style['stale']) is bool):
        raise ValueError('invalid style observation')
    mask = [0] * N_ACTIONS
    mask[0] = 1
    if obs['bankOpen']:
        mask[A_CLOSE_IF] = 1
        if obs['invPage'] > 0:
            mask[A_INV_PREV] = 1
        if obs['invPage'] < 3:
            mask[A_INV_NEXT] = 1
        if obs['bankPage'] > 0:
            mask[A_BANK_PREV] = 1
        if obs['bankPage'] + 1 < obs['bankPages']:
            mask[A_BANK_NEXT] = 1
        for k, it in enumerate(obs['invSel']):
            if it['count'] > 0:
                mask[A_DEPOSIT0 + k] = 1
        for k, it in enumerate(obs['bankSel']):
            if it['count'] > 0:
                mask[A_WITHDRAW0 + k] = 1
        return mask
    if obs['using']:
        if obs['invPage'] > 0:  # mask-v5: the page turns, the selection kept
            mask[A_INV_PREV] = 1
        if obs['invPage'] < 3:
            mask[A_INV_NEXT] = 1
        for k in range(len(obs['actSel'])):
            mask[A_ACT0 + k] = 1
        for k in range(len(obs['heldSel'])):
            mask[A_HELD0 + k] = 1
        mask[A_CLOSE_IF] = 1
        return mask
    if obs['dialogOpen'] or obs['modalOpen']:
        real = [o for o in obs['dialogOptions'] if not re.search(r'click here to continue', o.get('text') or '', re.I)]
        if real:
            for k in range(min(len(real), N_DIALOG_CHOICE)):
                mask[A_DIALOG_CHOICE0 + k] = 1
        elif obs['dialogOpen']:
            mask[A_DIALOG_CONTINUE] = 1
        mask[A_CLOSE_IF] = 1
        return mask
    if obs['invPage'] > 0:
        mask[A_INV_PREV] = 1
    if obs['invPage'] < 3:
        mask[A_INV_NEXT] = 1
    for k in range(len(obs['actSel'])):
        mask[A_ACT0 + k] = 1
    for k, c in enumerate(obs['clicks']):
        if c['inside']:
            mask[CLICK_ROWS[k]] = 1
    for k in range(len(obs['heldSel'])):
        mask[A_HELD0 + k] = 1
    pressure = len(obs['inventory']) >= DROP_MIN_STACKS
    for k, it in enumerate(obs['invSel']):
        if pressure and it['count'] > 0 and not TOOL_PROTECT_RE.search(it['name'] or ''):
            mask[A_DROP0 + k] = 1
        if it['count'] > 0:
            mask[A_USE0 + k] = 1
    # a style row only where it selects another style (the engine clamps row k to the weapon's last one), or
    # while the reset's levels wait for a style request's recompute
    current = min(style['mode'], style['count'] - 1)
    for k in range(4):
        if style['stale'] or min(k, style['count'] - 1) != current:
            mask[A_STYLE0 + k] = 1
    return mask


def login(world, p):
    """[login,_]: the three 100-tick stat timers (no random events: map_random_events is off),
    then ~update_all, whose ~player_combat_stat sets the combat varps from the new account's
    levels; they stay until a style change or a [changestat] recomputes them."""
    p.set_timer('stat_regen', 100, lambda: stat_regen(world, p))
    p.set_timer('stat_boost_restore', 100, lambda: stat_boost_restore(world, p))
    p.set_timer('health_regen', 100, lambda: health_regen(world, p))
    player_combat_stat(world, p)


def stat_regen(world, p):
    """[timer,stat_regen]: stat_add(stat, 1, 0) below the base, except prayer and hitpoints."""
    for i in range(len(SKILLS)):
        if p.base_levels[i] > p.levels[i] and i not in (PRAYER, HITPOINTS):
            world.stat_add(p, i, 1, 0)
    yield from ()


def stat_boost_restore(world, p):
    for i in range(len(SKILLS)):
        if p.base_levels[i] < p.levels[i] and i not in (PRAYER, HITPOINTS):
            world.stat_sub(p, i, 1, 0)
    yield from ()


def health_regen(world, p):
    """[timer,health_regen]: stat_heal(hitpoints, 1, 0)."""
    world.stat_heal(p, HITPOINTS, 1, 0, cause='regen')
    yield from ()


__all__ = ['Agent', 'build_mask', 'FISHING', 'level_for', 'p_oploc', 'Npc', 'INV']
