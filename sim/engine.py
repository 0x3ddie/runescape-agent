"""A tick-level model of the native engine for the Draynor goal slice (reference implementation).

This mirrors the vendor engine's own structure, so each rule can be checked against its
source: World.cycle's phases, Player.processInteraction and PathingEntity movement,
Npc.turn, and the content scripts the slice needs, written as Python generators that
suspend where RuneScript does (p_pausebutton, p_delay, npc_delay). File references are to
vendor/rs-sdk/server/engine/src and server/content/scripts.

What it does not implement raises Unsupported, and the caller records a coverage failure:
an engine script outside SUPPORTED, a tile or level without path tables, a random event.
The content scripts' text comes from the private tables (sim/build.py content); some engine
messages (for example "I can't reach that!") are quoted here.
Randomness is keyed (Rng): a draw depends on (world seed, tick, entity, purpose, index),
not on call order, so a batched implementation can reproduce it exactly. Native draws
(Math.random, JavaRandom) cannot be matched; only their distributions can.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np

from sim import paths, rs2
from sim.data import BLOCK, FLAG_NPC, Tables

# ---- constants from the engine source ---------------------------------------------------
# ServerTriggerType (script/ServerTriggerType.ts): an AP trigger plus 7 is its OP trigger.
APNPC1, APNPCU, APOBJ1, APLOC1, APLOCU = 3, 8, 31, 59, 64
AP_TO_OP = 7
# PlayerStat order (entity/PlayerStat.ts) as env-ts/core/skills.ts names them.
SKILLS = ('attack', 'defence', 'strength', 'hitpoints', 'ranged', 'prayer', 'magic', 'cooking', 'woodcutting',
          'fletching', 'fishing', 'firemaking', 'crafting', 'smithing', 'mining', 'herblore', 'agility', 'thieving',
          'stat18', 'stat19', 'runecraft')
ATTACK, DEFENCE, STRENGTH, HITPOINTS, RANGED, PRAYER, MAGIC = range(7)
COOKING, WOODCUTTING, FISHING, FIREMAKING, MINING = 7, 8, 10, 11, 14
XP_RATE = 25  # NODE_XPRATE of the training worlds (sim/build.py ENGINE_ENV); World's default
WALK_BLOCKED = 0x240100
# World.ts: random-event readiness every 500 ticks at 1/24 (1/12 once a zone has gone afk).
AFK_EVENTRATE, AFK_CHANCE1 = 500, 1 / 24
NPC_ACTIVE_RADIUS = 64  # RL_NPC_RADIUS, patch 0009
INV, WORN, BANK = 'inv', 'worn', 'bank'
WEARPOS_FRONT, WEARPOS_RHAND, WEARPOS_QUIVER = 2, 3, 13  # player/configs/equip.constant
# Npc modes (entity/NpcMode.ts) used here.
MODE_NULL, MODE_NONE, MODE_WANDER, MODE_PATROL, MODE_PLAYERESCAPE, MODE_PLAYERFOLLOW = -1, 0, 1, 2, 3, 4
MODE_PLAYERFACE, MODE_PLAYERFACECLOSE, MODE_OPPLAYER1, MODE_APPLAYER1 = 5, 6, 7, 12
MODE_OPPLAYER2, MODE_APPLAYER2 = 8, 13
# NpcStat order (entity/NpcStat.ts).
NPC_ATTACK, NPC_DEFENCE, NPC_STRENGTH, NPC_HP, _, NPC_MAGIC = range(6)
# Damage types (combat_damagetypes.constant) and the attack, strength, defence and ranged
# bonuses of each damage style ([proc,combat_get_damagestyle_bonuses]).
STAB, SLASH, CRUSH, RANGED_STYLE, MAGIC_STYLE = 0, 1, 2, 3, 4
STYLE_BONUSES = {0: (3, 0, 0, 0), 1: (0, 3, 0, 0), 2: (0, 0, 3, 0), 3: (1, 1, 1, 0), 4: (0, 0, 0, 3), 6: (0, 0, 3, 0)}
NOMOVE = 5  # MoveRestrict.NOMOVE
STACK_LIMIT = 2147483647


class Unsupported(Exception):
    """The native engine would do something this model does not implement."""


# ---- levels (entity/Player.ts levelExperience, xp stored x10) ----------------------------
def _level_table():
    table, acc = [], 0
    for level in range(1, 100):
        acc += int(level + 2.0 ** (level / 10.0) * 300.0)
        table.append(acc // 4 * 10)
    return table


LEVEL_XP = _level_table()


def level_for(xp):
    for i in range(98, -1, -1):
        if xp >= LEVEL_XP[i]:
            return min(i + 2, 99)
    return 1


def xp_for(level):
    return LEVEL_XP[level - 2] if level >= 2 else 0


# ---- keyed randomness ---------------------------------------------------------------------
def _mix(h):
    h ^= h >> 16
    h = (h * 0x7feb352d) & 0xffffffff
    h ^= h >> 15
    h = (h * 0x846ca68b) & 0xffffffff
    h ^= h >> 16
    return h


class Rng:
    """Uniform [0, 1) draws keyed by integers (lowbias32 chain); 24 bits, exact in float32."""
    def __init__(self, seed):
        self.seed = _mix(seed & 0xffffffff)

    def __call__(self, *keys):
        return self.draw(self.seed, *keys)

    def prefix(self, *keys):
        """The chain state after keys, so draws sharing them (one per npc) hash only the rest."""
        h = self.seed
        for k in keys:
            h = _mix(h ^ ((k * 0x9e3779b9) & 0xffffffff))
        return h

    @staticmethod
    def draw(h, *keys):
        for k in keys:
            h = _mix(h ^ ((k * 0x9e3779b9) & 0xffffffff))
        return (h >> 8) / 16777216.0


# Purposes for Rng keys.
R_AFK, R_WANDER_GO, R_WANDER_DX, R_WANDER_DZ, R_SCRIPT, R_NPC_SCRIPT, _, R_PACE = range(1, 9)


# ---- inventories (engine/Inventory.ts) ------------------------------------------------------
class Inventory:
    def __init__(self, capacity, stackall=False):
        self.capacity, self.stackall = capacity, stackall
        self.items = [None] * capacity  # [id, count] or None

    def stacks(self, t, obj):
        return self.stackall or bool(t.obj_types[obj]['stackable'])

    def add(self, t, obj, count=1):
        completed = 0
        if not self.stacks(t, obj):
            for i in range(self.capacity):
                if self.items[i] is None:
                    self.items[i] = [obj, 1]
                    completed += 1
                    if completed >= count:
                        break
            return completed
        index = next((i for i, it in enumerate(self.items) if it and it[0] == obj), -1)
        if index < 0:
            index = next((i for i, it in enumerate(self.items) if it is None), -1)
            if index < 0:
                return 0
        have = self.items[index][1] if self.items[index] else 0
        total = min(STACK_LIMIT, have + count)
        self.items[index] = [obj, total]
        return total - have

    def remove(self, obj, count=1):
        removed = 0
        for i, it in enumerate(self.items):
            if not it or it[0] != obj:
                continue
            take = min(it[1], count - removed)
            removed += take
            it[1] -= take
            if it[1] == 0:
                self.items[i] = None
            if removed >= count:
                break
        return removed

    def total(self, obj):
        return sum(it[1] for it in self.items if it and it[0] == obj)

    def freespace(self):
        return sum(it is None for it in self.items)

    def itemspace2(self, t, obj, count):
        """Inventory.itemspace2: how many of `count` would NOT fit."""
        if self.stacks(t, obj):
            have = next((it[1] for it in self.items if it and it[0] == obj), None)
            if have is None:
                return 0 if self.freespace() else count
            return max(0, have + count - STACK_LIMIT)
        return max(0, count - self.freespace())

    def listed(self):
        return [(slot, it[0], it[1]) for slot, it in enumerate(self.items) if it]


# ---- entities -------------------------------------------------------------------------------
@dataclass(eq=False)
class Loc:
    """entity/Loc.ts: a map loc (static, EntityLifeCycle.RESPAWN) or one a script added
    (DESPAWN). Its size is fixed when it is made; a change keeps the base to revert to."""
    index: int
    base: int
    type: int
    x: int
    z: int
    shape: int
    angle: int
    width: int
    length: int
    static: bool = True
    active: bool = True
    lifecycle_tick: int = -1  # counts down to the next lifecycle event while tracked
    level: int = 0

    def __post_init__(self):
        self.base_shape, self.base_angle = self.shape, self.angle
        self.layer = paths.shape_layer(self.shape)

    def is_valid(self):
        return self.active

    def is_changed(self):
        return (self.type, self.shape, self.angle) != (self.base, self.base_shape, self.base_angle)

    def revert(self):
        self.type, self.shape, self.angle = self.base, self.base_shape, self.base_angle


@dataclass(eq=False)
class Obj:
    """A ground item: a map spawn (static, respawns after it is taken) or a drop."""
    type: int
    count: int
    x: int
    z: int
    static: bool
    ticks: int = -1          # lifecycle countdown: despawn of a drop, respawn of a taken spawn
    receiver: object = None  # the player a drop belongs to
    active: bool = True
    level: int = 0

    width = length = 1

    def is_valid(self):
        return self.active and self.count >= 1


@dataclass(eq=False)
class Npc:
    nid: int
    base: int
    x: int
    z: int
    size: int
    type: int = -1
    start_x: int = 0
    start_z: int = 0
    level: int = 0
    active: bool = True
    delayed: bool = False
    delayed_until: int = -1
    path: list = field(default_factory=list)
    mode: int = MODE_WANDER
    target: object = None
    timer_interval: int = -1
    timer_clock: int = 0
    hunt_clock: int = 0
    regen_clock: int = 0
    stuck: int = 0
    tele: bool = False
    script: object = None       # NPC_SUSPENDED generator
    observers: int = 0
    last_tick_x: int = 0
    last_tick_z: int = 0
    width: int = 1
    length: int = 1
    levels: list = field(default_factory=list)
    queue: list = field(default_factory=list)     # [queue id, arg, delay]
    hunt_mode: int = -1
    huntrange: int = 0
    hunt_target: object = None
    lifecycle_tick: int = -1                        # respawn countdown after npc_del
    last_movement: int = -1
    heroes: dict = field(default_factory=dict)      # player name -> damage dealt (npc_heropoints)
    life: int = 0                                   # counts respawns: one enemy life, for the kill audit
    vars: dict = field(default_factory=dict)        # varns the slice's scripts use (NPC_VARS)

    def is_valid(self):
        return self.active and not self.delayed

    @property
    def uid(self):
        return (self.type << 16) | self.nid


# Npc varns the combat scripts read, as Npc.resetEntity leaves them: ints 0, uids null.
NPC_VARS = dict(action_delay=0, lastcombat=0, aggressive_player=None, attacking=None, xp_mult=0, stunned=0,
                macro_target=None)


@dataclass
class Request:
    script: object      # a factory: () -> generator
    name: str
    delay: int
    kind: str = 'normal'


COM_VARPS = ('stabattack', 'slashattack', 'crushattack', 'magicattack', 'rangeattack', 'stabdef', 'slashdef', 'crushdef',
             'magicdef', 'rangedef', 'maxhit')


class Player:
    width = length = 1

    def __init__(self, world, name):
        self.world, self.name = world, name
        self.display = name[:1].upper() + name[1:]
        self.x = self.z = self.level = 0
        self.instant = False            # moveSpeed INSTANT: set by a teleJump, reset in the cycle's cleanup
        self.path = []                  # remaining waypoints, next first
        self.steps_taken = 0
        self.last_movement = 0
        self.target = None
        self.target_op = -1
        self.target_type = -1
        self.ap_range, self.ap_range_called = 10, False
        self.next_target = None
        self.allow_repath = True        # AllowRepath.BEFOREDEST, which queueWaypoints sets
        self.delayed, self.delayed_until, self.protect = False, -1, False
        self.script = None              # (generator, state)
        self.resume_buttons = []
        # Player.modalState's bits and the modal ids, which the engine keeps apart: opening a
        # chat clears the main and side bits but leaves modalMain set (it clears modalChat
        # instead), so bank_main can stay the main modal id under a dialog.
        self.modal_state = set()
        self.modal_main = self.modal_chat = None
        self.queue, self.weak_queue, self.engine_queue = [], [], []
        self.timers = {}
        self.stats = [0] * len(SKILLS)
        self.levels = [1] * len(SKILLS)
        self.base_levels = [1] * len(SKILLS)
        self.stats[HITPOINTS] = xp_for(10)
        self.levels[HITPOINTS] = self.base_levels[HITPOINTS] = 10
        self.inv, self.worn, self.bank = Inventory(28), Inventory(14), Inventory(240, stackall=True)
        self.varps = dict(action_delay=0, com_mode=0, bankcert=0, bankinsert=0, tutorial=1000, lastcombat=0,
                          lastcombat_pvp=0, aggressive_npc=None, death=0, option_nodef=0, damagetype=0, damagestyle=0,
                          trail_status=0, eat_delay=0, **{f'com_{k}': 0 for k in COM_VARPS})
        self.last_useitem = self.last_com = -1
        self.last_item = self.last_slot = self.last_useslot = -1  # the held item an OPHELD/OPHELDU script reads
        self.origin_x = self.origin_z = 0
        self.afk_ready = False
        # Hitpoints reached 0 (the native adapter's death monitor, env-ts/engine/death.ts, samples them at applyDamage and
        # changeStat): a heal later in the same tick, such as [timer,health_regen], does not undo the death it queued.
        self.dead = False

    def is_valid(self):
        """Player.isValid: logged in and visible, which the slice's player always is."""
        return True

    # ---- state the agent reads
    @property
    def hp(self):
        return self.levels[HITPOINTS]

    def has_waypoints(self):
        return bool(self.path)

    def busy(self):
        """Player.busy: delayed, or a main or chat modal (containsModalInterface)."""
        return self.delayed or bool(self.modal_state & {'main', 'chat'})

    def can_access(self):
        return not self.protect and not self.busy()

    def combat_level(self):
        """Player.getCombatLevel (entity/Player.ts)."""
        b = self.base_levels
        base = 0.25 * (b[DEFENCE] + b[HITPOINTS] + b[PRAYER] // 2)
        melee = 0.325 * (b[ATTACK] + b[STRENGTH])
        ranged = 0.325 * (b[RANGED] // 2 + b[RANGED])
        magic = 0.325 * (b[MAGIC] // 2 + b[MAGIC])
        return int(base + max(melee, ranged, magic))

    def busy2(self):
        """BUSY2: an interaction (other than follow) or a walk."""
        return self.target is not None or bool(self.path)

    # ---- inventories and modals
    def inventory(self, name):
        return getattr(self, name)

    def close_modal(self, clear_weak=True):
        """Player.closeModal."""
        if clear_weak:
            self.weak_queue.clear()
        if not self.delayed:
            self.protect = False
        if not self.modal_state:
            return
        self.modal_state = set()
        if self.script is not None and self.script[1] == 'pausebutton':
            self.script = None
            self.resume_buttons = []
        if self.modal_main is not None:
            closed, self.modal_main = self.modal_main, None
            if closed == 'bank_main':
                self.world.close_bank(self)
        self.modal_chat = None

    def open_main_side(self, main):
        """Player.openMainSideModal: closes a chat modal and a suspended dialog script (no
        side modal of the slice has a close trigger, so its id is not kept)."""
        if 'chat' in self.modal_state:
            self.modal_state.discard('chat')
            self.modal_chat = None
        self.modal_state |= {'main', 'side'}
        self.modal_main = main
        if self.script is not None and self.script[1] == 'pausebutton':
            self.script = None
            self.resume_buttons = []

    def open_chat(self, com):
        """Player.openChatModal: clears the main and side bits (not their ids)."""
        if 'main' in self.modal_state:
            self.modal_state.discard('main')
            self.modal_chat = None
        if 'side' in self.modal_state:
            self.modal_state.discard('side')
            self.modal_chat = None
        self.modal_state.add('chat')
        self.modal_chat = com

    def clear_waypoints(self):
        self.path = []

    def queue_waypoints(self, waypoints):
        self.path = list(waypoints)
        self.allow_repath = True

    def queue_waypoint(self, x, z):
        self.path = [(x, z)]
        self.allow_repath = True

    def clear_interaction(self):
        self.target, self.target_op, self.target_type = None, -1, -1
        self.ap_range, self.ap_range_called = 10, False

    def clear_pending_action(self):
        self.clear_interaction()
        self.close_modal()

    def stop_action(self):
        """Player.stopAction: also unsetMapFlag, which clears the walk."""
        self.clear_pending_action()
        self.clear_waypoints()

    def set_interaction(self, target, op):
        if not target.is_valid():
            return False
        self.target, self.target_op = target, op
        self.ap_range, self.ap_range_called = 10, False
        self.target_type = target.type
        return True

    def enqueue(self, name, factory, kind='normal', delay=0):
        request = Request(factory, name, delay, kind)
        (self.engine_queue if kind == 'engine' else self.weak_queue if kind == 'weak' else self.queue).append(request)

    def set_timer(self, name, interval, fn, kind='normal'):
        self.timers[name] = dict(kind=kind, interval=interval, clock=self.world.tick, fn=fn)


# ---- the world --------------------------------------------------------------------------------
class World:
    """One native world: the persistent npcs, locs and ground items of the data area, and
    at most one player. `tick` is World.currentTick: scripts in a cycle see it as map_clock,
    and it advances after the cycle."""

    def __init__(self, tables: Tables, seed=0, start_tick=0, npcs=None, exclusions=frozenset(), xp_rate=XP_RATE):
        """npcs: the world's npc spawns (dicts as tables.meta['npcs']) in place of the tables' own, for a task that
        places its enemies per episode (sim/combat.py); every rule applies to them as to map spawns. exclusions: native
        behavior a task's declared world leaves out (EXCLUSIONS), in training and evaluation alike. xp_rate: the world's
        XP multiplier, as NODE_XPRATE multiplies Player.addXp natively (25 in every training world)."""
        if type(xp_rate) is not int or xp_rate < 1:
            raise ValueError(f'xp_rate {xp_rate!r}: a positive whole multiplier')
        self.xp_rate = xp_rate
        unknown = set(exclusions) - set(EXCLUSIONS)
        if unknown:
            raise ValueError(f'unknown exclusions {sorted(unknown)}')
        self.exclusions = frozenset(exclusions)
        self.t = tables
        self.rng = Rng(seed)
        self.tick = start_tick
        self.texts = tables.meta.get('content', {}).get('texts', {})
        # The map locs (in table order; a script's locs are numbered after them while they
        # exist), each zone's list in the engine's order (Zone.locs: a removed map loc and a
        # changed one move to the end), the locs on each tile, and the lifecycle events in the
        # order they were set (World.locObjTracker).
        self.locs = [Loc(i, l['type'], l['type'], l['x'], l['z'], l['shape'], l['angle'], l['width'], l['length'],
                         level=l['level']) for i, l in enumerate(tables.meta['locs'])]
        self.n_stored_locs = self._next_loc = len(self.locs)
        self.loc_at, self.loc_zones = {}, {}  # by (level, x, z) and (level, zone x, zone z)
        for loc in self.locs:
            self.loc_at.setdefault((loc.level, loc.x, loc.z), []).append(loc)
            self._zone(loc).append(loc)
        self.loc_events = {}
        self.npcs = []
        self.npc_by_nid = {}
        for n in sorted(tables.meta['npcs'] if npcs is None else npcs, key=lambda n: n['nid']):
            npc = Npc(n['nid'], n['type'], n['x'], n['z'], n['size'], type=n['type'], start_x=n['x'], start_z=n['z'],
                      level=n['level'], width=n['size'], length=n['size'])
            nt = tables.npc_types[npc.type]
            npc.mode, npc.timer_interval = nt['defaultmode'], nt['timer']
            npc.levels = list(nt['stats'])
            npc.hunt_mode, npc.huntrange = nt['huntmode'], nt['huntrange']
            npc.vars = dict(NPC_VARS)
            self.npc_spawned(npc)  # [ai_spawn,_] ran when the world booted
            self.npcs.append(npc)
            self.npc_by_nid[npc.nid] = npc
        self.objs = [Obj(o['type'], o['count'], o['x'], o['z'], static=True, level=o['level']) for o in tables.meta['objs']]
        self.members = bool(tables.meta['environment']['members'])
        self.multiway = {tuple(zone) for zone in tables.meta['multiway']}
        self.npc_events = []     # npcs whose [ai_spawn,_] runs in the next cycle's npc event phase
        self.queue = []          # [delay, generator]: scripts a world_delay suspended (World.queue)
        self.control = None      # (kind, default) -> outcome: a replay's controlled draws (sim.check)
        self.loc_version = 0     # counts loc changes, so the agent's per-tile caches know them
        self._wander = (None, 0)  # (tick, keyed-draw prefix of this tick's wander rolls)
        self.occupied = {}  # (level, x, z) -> npc/player count carrying the NPC collision flag
        for npc in self.npcs:
            self._occupy(npc, True)
        a = tables.area
        self._x0, self._x1, self._z0, self._z1, self._h = a['x0'], a['x1'], a['z0'], a['z1'], a['z1'] - a['z0'] + 1
        self._wh, self._levels = tables.size[0] * tables.size[1], len(tables.levels)
        # The live collision of the data area on each level (entity bits apart, in `occupied`),
        # the boot map's, and the tiles (level, x, z) where they differ: the stored trees, reach
        # sets and sight lines hold only for queries that read none of those tiles; the rest are
        # searched here (sim.paths).
        self._flags = tables.collision.tolist()
        self._boot_flags = tables.collision
        self.changed = set()
        self._steps, self._trees = {}, None
        self._client, self._scene_trees = {}, None  # the client's collision and searches (native-click)
        self.player = None
        self.pending_login = None  # (player, login script) admitted in the next cycle's processLogins
        self.chat_log = []       # (tick, text) of MessageGame lines to the player
        self.text_log = []       # (tick, component, text) of IfSetText to the player
        self.events = []         # (tick, kind, data): production and ownership evidence for sim.audit

    # ---- text
    def text(self, key, **fill):
        text = self.texts.get(key)
        if text is None:
            raise Unsupported(f'game text {key} not extracted')
        for name, value in fill.items():
            text = text.replace('{' + name + '}', str(value))
        return text

    def mes(self, text):
        self.chat_log.append((self.tick, text))

    def settext(self, component, text):
        self.text_log.append((self.tick, component, text))

    # ---- collision and movement (routefinder/StepValidator.ts, CollisionStrategy.ts)
    def flags(self, level, x, z, npc_bits=True):
        if not (self._x0 <= x <= self._x1 and self._z0 <= z <= self._z1 and 0 <= level < self._levels):
            return 0x7fffffff
        f = self._flags[level * self._wh + (x - self._x0) * self._h + (z - self._z0)]
        if npc_bits and self.occupied.get((level, x, z)):
            f |= FLAG_NPC
        return f

    def _occupy(self, entity, add):
        for dx in range(entity.width):
            for dz in range(entity.length):
                key = (entity.level, entity.x + dx, entity.z + dz)
                self.occupied[key] = self.occupied.get(key, 0) + (1 if add else -1)

    def can_move(self, collision, flag, block):
        if collision == 0:  # NORMAL
            return (flag & block) == 0
        if collision == 1:  # BLOCKED: only where there is floor
            b = block & ~0x200000
            return (flag & b) == 0 and (flag & 0x200000) != 0
        if collision == 3:  # INDOORS
            return (flag & block) == 0 and (flag & 0x80000000) != 0
        if collision == 4:  # OUTDOORS
            return (flag & (block | 0x80000000)) == 0
        raise Unsupported(f'collision type {collision}')

    def can_travel(self, level, x, z, dx, dz, size, extra, collision):
        f = lambda a, b: self.flags(level, a, b)
        ok = lambda a, b, mask: self.can_move(collision, f(a, b), mask | extra)
        if size == 1:
            if dx == 0 and dz == -1:
                return ok(x, z - 1, BLOCK['S'])
            if dx == 0 and dz == 1:
                return ok(x, z + 1, BLOCK['N'])
            if dx == -1 and dz == 0:
                return ok(x - 1, z, BLOCK['W'])
            if dx == 1 and dz == 0:
                return ok(x + 1, z, BLOCK['E'])
            if dx == -1 and dz == -1:
                return ok(x - 1, z - 1, BLOCK['SW']) and ok(x - 1, z, BLOCK['W']) and ok(x, z - 1, BLOCK['S'])
            if dx == -1 and dz == 1:
                return ok(x - 1, z + 1, BLOCK['NW']) and ok(x - 1, z, BLOCK['W']) and ok(x, z + 1, BLOCK['N'])
            if dx == 1 and dz == -1:
                return ok(x + 1, z - 1, BLOCK['SE']) and ok(x + 1, z, BLOCK['E']) and ok(x, z - 1, BLOCK['S'])
            if dx == 1 and dz == 1:
                return ok(x + 1, z + 1, BLOCK['NE']) and ok(x + 1, z, BLOCK['E']) and ok(x, z + 1, BLOCK['N'])
            return False
        if size == 2:
            if dx == 0 and dz == -1:
                return ok(x, z - 1, BLOCK['SW']) and ok(x + 1, z - 1, BLOCK['SE'])
            if dx == 0 and dz == 1:
                return ok(x, z + 2, BLOCK['NW']) and ok(x + 1, z + 2, BLOCK['NE'])
            if dx == -1 and dz == 0:
                return ok(x - 1, z, BLOCK['SW']) and ok(x - 1, z + 1, BLOCK['NW'])
            if dx == 1 and dz == 0:
                return ok(x + 2, z, BLOCK['SE']) and ok(x + 2, z + 1, BLOCK['NE'])
            if dx == -1 and dz == -1:
                return (ok(x - 1, z, BLOCK['N_AND_SW']) and ok(x - 1, z - 1, BLOCK['SW'])
                        and ok(x, z - 1, BLOCK['NE_AND_W']))
            if dx == -1 and dz == 1:
                return (ok(x - 1, z + 1, BLOCK['N_AND_SW']) and ok(x - 1, z + 2, BLOCK['NW'])
                        and ok(x, z + 2, BLOCK['SE_AND_W']))
            if dx == 1 and dz == -1:
                return (ok(x + 1, z - 1, BLOCK['NE_AND_W']) and ok(x + 2, z - 1, BLOCK['SE'])
                        and ok(x + 2, z, BLOCK['N_AND_SE']))
            if dx == 1 and dz == 1:
                return (ok(x + 1, z + 2, BLOCK['SE_AND_W']) and ok(x + 2, z + 2, BLOCK['NE'])
                        and ok(x + 2, z + 1, BLOCK['N_AND_SE']))
            return False
        raise Unsupported(f'npc size {size}')

    @staticmethod
    def face(sx, sz, dx, dz):
        return (dx > sx) - (dx < sx), (dz > sz) - (dz < sz)

    def take_step(self, e, collision, extra):
        """PathingEntity.takeStep for walk speed: toward the current waypoint, diagonal, then
        east/west, then north/south; no step if all three are blocked."""
        tx, tz = e.path[0]
        dx, dz = self.face(e.x, e.z, tx, tz)
        size = e.width
        if size == 1 and self.can_travel(e.level, e.x, e.z, dx, dz, size, extra, collision):
            return dx, dz
        if dx != 0 and self.can_travel(e.level, e.x, e.z, dx, 0, size, extra, collision):
            return dx, 0
        if dz != 0 and self.can_travel(e.level, e.x, e.z, 0, dz, size, extra, collision):
            return 0, dz
        return 0, 0

    def process_movement(self, e, collision, extra):
        """PathingEntity.processMovement + validateAndAdvanceStep, one step (walk)."""
        if not e.path:
            return False
        dx, dz = self.take_step(e, collision, extra)
        sx, sz = e.x, e.z
        if (dx or dz):
            self._occupy(e, False)
            e.x, e.z = e.x + dx, e.z + dz
            self._occupy(e, True)
        if e.path and e.path[0] == (e.x, e.z):
            e.path.pop(0)
        if (e.x, e.z) != (sx, sz):
            if isinstance(e, Player):
                e.steps_taken += 1
            return True
        return True

    # ---- the cycle (World.cycle)
    def cycle(self):
        self.process_world()
        self.process_clients_in()
        self.process_npc_events()
        self.process_npcs()
        self.process_players()
        self.process_logins()
        self.process_zones()
        self.process_info()
        self.process_cleanup()
        self.tick += 1

    def process_world(self):
        """World.processWorld: the world queue, then npcs that hunt players do so while observed, even when delayed."""
        for request in _links(self.queue):
            delay = request[0]
            request[0] -= 1
            if delay <= 0:
                _unlink(self.queue, request)
                self.world_run(request[1])
        if self.player is None:
            return
        for npc in self.npcs:
            if npc.active and npc.hunt_mode != -1 and npc.observers > 0:
                hunt = self.t.npc_types[npc.type]['hunt']
                if hunt is not None and hunt['id'] == npc.hunt_mode and hunt['type'] == 1:
                    self.npc_hunt_all(npc, hunt)

    def world_run(self, gen):
        """A script goes on from the world queue, to its end or to its next world_delay(n). World.enqueueScript queues it
        with n + 1; from the next cycle on each pass takes 1 off, and the pass that finds nothing left runs it: n + 2
        cycles after the world_delay."""
        step = next(gen, None)
        if step is None:
            return
        if step[0] != 'world_delay':
            raise Unsupported(f'world script suspension {step}')
        self.queue.append([step[1] + 1, gen])

    def npc_hunt_all(self, npc, hunt):
        """Npc.huntAll + huntPlayers for the one player (HuntIterator searches the npc's level)."""
        npc.hunt_target = None
        if npc.hunt_clock < hunt['rate'] - 1 or npc.huntrange < 1:
            return
        p, nt = self.player, self.t.npc_types[npc.type]
        if p.level != npc.level or _distance_sw(npc.x, npc.z, p) > npc.huntrange:
            return
        if hunt['checkVis'] == 1 and not self.line_of_sight(p.level, p.x, p.z, npc.x, npc.z):
            return
        if hunt['checkVis'] == 2:
            raise Unsupported('line-of-walk hunting')
        if hunt['checkNotBusy'] and p.busy():
            return
        # checkAfk: zonesAfk() needs updateAfkZones, which runs only for players with a
        # client, so it is never true for the clientless agent.
        if hunt['checkNotTooStrong'] == 1 and p.combat_level() > nt['vislevel'] * 2:  # outside the wilderness
            return
        if npc.target is not p and not self.is_multi(p.level, p.x, p.z):
            # the hunt checks name %lastcombat (varp 38) and %npc_lastcombat (varn 6)
            if hunt['checkNotCombat'] != -1 and p.varps['lastcombat'] + 8 > self.tick:
                return
            if hunt['checkNotCombatSelf'] != -1 and npc.vars['lastcombat'] + 8 > self.tick:
                return
        if hunt['checkVars'] or hunt['checkInv'] != -1 or hunt['checkNotCombat'] not in (-1, 38) \
                or hunt['checkNotCombatSelf'] not in (-1, 6):
            raise Unsupported('hunting that checks other variables or an inventory')
        npc.hunt_target = p

    def process_clients_in(self):
        p = self.player
        if p is not None and self.tick % AFK_EVENTRATE == 0 and 'random_events' not in self.exclusions:
            p.afk_ready = self.rng(R_AFK, self.tick) < AFK_CHANCE1

    def process_npc_events(self):
        """World.processNpcEventQueue: the spawn trigger of npcs added since the last cycle."""
        events, self.npc_events = self.npc_events, []
        for npc in events:
            self.npc_spawned(npc)

    def npc_spawned(self, npc):
        """[ai_spawn,_]: the combat xp multiplier (%npc_start_coord is not read by the slice)."""
        npc.vars['xp_mult'] = self.npc_param(npc, 'combat_xp_multiplier')

    def is_multi(self, level, x, z):
        """GameMap.isMulti: multi-combat zones are whole 8x8 zones."""
        return (x & ~7, z & ~7, level) in self.multiway

    def active_chunks(self):
        """World.activeNpcChunks (patch 0009): the 64-tile chunks within RL_NPC_RADIUS of the
        player, on its level; npcs elsewhere, other floors included, turn dormant."""
        p = self.player
        if p is None:
            return set()
        r = NPC_ACTIVE_RADIUS
        return {(p.level, cx, cz) for cx in range((p.x - r) >> 6, ((p.x + r) >> 6) + 1)
                for cz in range((p.z - r) >> 6, ((p.z + r) >> 6) + 1)}

    def process_npcs(self):
        active = self.active_chunks()
        for npc in self.npcs:
            if (npc.level, npc.x >> 6, npc.z >> 6) not in active:
                self.npc_lifecycle(npc)
                continue
            self.npc_turn(npc)

    def process_players(self):
        p = self.player
        if p is None:
            return
        if p.delayed and self.tick >= p.delayed_until:
            p.delayed = False
        if not p.delayed and p.script is not None and p.script[1] == 'suspended':
            self.resume(p)
        self.process_queues(p)
        for kind in ('normal', 'soft'):
            for name, timer in list(p.timers.items()):
                if timer['kind'] != kind:
                    continue
                if self.tick >= timer['clock'] + timer['interval'] and (kind == 'soft' or p.can_access()):
                    timer['clock'] = self.tick
                    self.execute(p, timer['fn'](), protect=kind == 'normal', name=name)
        for request in _links(p.engine_queue):
            delay = request.delay
            request.delay -= 1
            if p.can_access() and delay <= 0:
                self.execute(p, request.script(), protect=True, name=request.name)
                _unlink(p.engine_queue, request)
        self.process_interaction(p)

    def process_logins(self):
        """World.processLogins: a new player enters after the player phase and runs [login,_]."""
        if self.pending_login is None:
            return
        p, login = self.pending_login
        self.pending_login = None
        self.player = p
        self._occupy(p, True)
        login(self, p)

    def process_queues(self, p):
        """Player.processQueues: a strong script closes modals, then the queue, then weak."""
        if any(r.kind == 'strong' for r in p.queue):
            p.close_modal()
        for queue in (p.queue, p.weak_queue):
            for request in _links(queue):
                delay = request.delay
                request.delay -= 1
                if p.can_access() and delay <= 0:
                    _unlink(queue, request)
                    self.execute(p, request.script(), protect=True, name=request.name)

    def process_zones(self):
        for loc in list(self.loc_events):  # Loc.turn
            loc.lifecycle_tick -= 1
            if loc.lifecycle_tick == 0:
                if not loc.static and loc.active:
                    self.remove_loc(loc, 0)
                elif loc.static and loc.is_changed() and loc.active:
                    self.revert_loc(loc)
                elif loc.static and not loc.active:
                    self.add_loc(loc, 0)
                else:
                    self.set_lifecycle(loc, -1)
            elif loc.lifecycle_tick < 0:
                self.set_lifecycle(loc, -1)
        for obj in list(self.objs):  # Obj.turn
            if obj.ticks > 0:
                obj.ticks -= 1
                if obj.ticks == 0 and obj.static and not obj.active:
                    obj.active, obj.ticks = True, -1
                elif obj.ticks == 0 and not obj.static:
                    obj.active = False  # gone: a pickup suspended on it finds nothing
                    self.objs.remove(obj)

    def process_info(self):
        """BuildArea.rebuildNormal: the loaded 104-tile area recentres near its edge."""
        p = self.player
        if p is None:
            return
        ox, oz = p.origin_x >> 3, p.origin_z >> 3
        if p.x < (ox - 4) << 3 or p.z < (oz - 4) << 3 or p.x > ((ox + 5) << 3) - 1 or p.z > ((oz + 5) << 3) - 1:
            p.origin_x, p.origin_z = p.x, p.z

    def process_cleanup(self):
        """Player and Npc resetEntity: steps, and the moveSpeed a teleJump set (resetPathingEntity)."""
        if self.player is not None:
            self.player.steps_taken = 0
            self.player.instant = False
        for npc in self.npcs:
            npc.tele = False
            npc.last_tick_x, npc.last_tick_z = npc.x, npc.z

    # ---- locs (World.addLoc, removeLoc, changeLoc, revertLoc; Zone; LocOps)
    def set_lifecycle(self, loc, duration):
        """NonPathingEntity.setLifeCycle: a positive duration tracks the loc at the tail."""
        self.loc_events.pop(loc, None)
        if duration > 0:
            self.loc_events[loc] = True
            loc.lifecycle_tick = duration
        else:
            loc.lifecycle_tick = -1

    def loc_collision(self, loc, add):
        """GameMap.changeLocCollision for the loc's current type, shape and angle."""
        masks = paths.collision_masks(loc.x, loc.z, loc.shape, loc.angle, self.t.loc_types[loc.type])
        for x, z, mask in masks:
            if not (self._x0 <= x <= self._x1 and self._z0 <= z <= self._z1):
                raise Unsupported(f'collision changed outside the data area at {x},{z}')
            i = loc.level * self._wh + (x - self._x0) * self._h + (z - self._z0)
            self._flags[i] = (self._flags[i] | mask) if add else (self._flags[i] & ~mask)
            (self.changed.discard if self._flags[i] == self._boot_flags[i] else self.changed.add)((loc.level, x, z))
        if masks:
            self._steps.pop(loc.level, None)
            self._client.pop(loc.level, None)
            self._trees = self._scene_trees = None

    def _zone(self, loc):
        return self.loc_zones.setdefault((loc.level, loc.x >> 3, loc.z >> 3), [])

    def add_loc(self, loc, duration):
        """World.addLoc: the collision of the loc as it is, then Zone.addLoc reverts it."""
        self.loc_collision(loc, True)
        if not loc.static:
            self._zone(loc).append(loc)
        loc.revert()
        loc.active = True
        self.loc_version += 1
        self.set_lifecycle(loc, duration)

    def remove_loc(self, loc, duration):
        """World.removeLoc: a map loc goes to its zone's end and respawns after `duration`."""
        if not loc.active:
            return
        self.loc_collision(loc, False)
        zone = self._zone(loc)
        zone.remove(loc)
        if loc.static:
            zone.append(loc)
        else:
            self.loc_at[(loc.level, loc.x, loc.z)].remove(loc)
        loc.active = False
        self.loc_version += 1
        self.set_lifecycle(loc, duration if loc.static else -1)

    def change_loc(self, loc, type_id, shape, angle, duration):
        """World.changeLoc: an added loc that is gone stays gone; the old collision goes only
        if the loc is active, the new one comes in any case."""
        if not loc.static and not loc.active:
            return
        if loc.active:
            self.loc_collision(loc, False)
        loc.type, loc.shape, loc.angle = type_id, shape, angle
        self.loc_collision(loc, True)
        self._zone_changed(loc)
        self.set_lifecycle(loc, duration if loc.is_changed() or not loc.static else -1)

    def revert_loc(self, loc):
        self.loc_collision(loc, False)
        loc.revert()
        self.loc_collision(loc, True)
        self._zone_changed(loc)
        self.set_lifecycle(loc, -1)

    def _zone_changed(self, loc):
        """Zone.changeLoc: active again, moved to the end of its zone."""
        loc.active = True
        zone = self._zone(loc)
        zone.remove(loc)
        zone.append(loc)
        self.loc_version += 1

    def loc_add(self, level, x, z, type_id, angle, shape, duration):
        """LOC_ADD: a loc on that tile and layer (gone or not) is changed, else a new one is added."""
        if not (self._x0 <= x <= self._x1 and self._z0 <= z <= self._z1):
            raise Unsupported(f'a loc added outside the data area at {x},{z}')
        layer = paths.shape_layer(shape)
        for loc in list(self.loc_zones.get((level, x >> 3, z >> 3), ())):
            if (loc.x, loc.z) == (x, z) and loc.layer == layer:
                self.change_loc(loc, type_id, shape, angle, duration)
                return loc
        t = self.t.loc_types[type_id]
        loc = Loc(self._next_loc, type_id, type_id, x, z, shape, angle, t['width'], t['length'], static=False, level=level)
        self._next_loc += 1
        self.loc_at.setdefault((level, x, z), []).append(loc)
        self.add_loc(loc, duration)
        return loc

    def map_locaddunsafe(self, level, x, z):
        """MAP_LOCADDUNSAFE (script/handlers/ServerOps.ts:213-242): whether a loc of an active type covers the tile, among
        all locs (gone ones too, except walls) of its zone and the zones to the south and west, by each loc's turned size."""
        for zx in ((x - 8) >> 3, x >> 3):
            for zz in ((z - 8) >> 3, z >> 3):
                for loc in self.loc_zones.get((level, zx, zz), ()):
                    if self.t.loc_types[loc.type]['active'] != 1 or (not loc.active and loc.layer == 0):
                        continue
                    width, length = (loc.length, loc.width) if loc.angle & 1 else (loc.width, loc.length)
                    if loc.x <= x < loc.x + width and loc.z <= z < loc.z + length:
                        return True
        return False

    def locs_in_zone(self, level, x, z):
        """LOC_FINDALLZONE: the zone's active locs, last first (getAllLocsSafe(true))."""
        return [loc for loc in reversed(self.loc_zones.get((level, x >> 3, z >> 3), ())) if loc.active]

    def obj_param(self, obj, name):
        """oc_param: the obj's own value, else the param's default."""
        params = self.t.obj_types[obj]['params']
        return params[name] if name in params else self.t.meta['param_defaults'][name]

    def loc_param_or_default(self, loc, name):
        params = self.t.loc_types[loc.type]['params']
        return params[name] if name in params else self.t.meta['param_defaults'][name]

    def loc_param(self, loc, name):
        params = self.t.loc_types[loc.type]['params']
        if name not in params:
            raise Unsupported(f"param {name} of loc {self.t.loc_types[loc.type]['debugname']}")
        return params[name]

    def locs_near(self, level, x, z, radius):
        """Active locs on the level whose origin is within radius (Chebyshev), in world order."""
        near = [loc for zx in range((x - radius) >> 3, ((x + radius) >> 3) + 1)
                for zz in range((z - radius) >> 3, ((z + radius) >> 3) + 1)
                for loc in self.loc_zones.get((level, zx, zz), ())
                if loc.active and max(abs(loc.x - x), abs(loc.z - z)) <= radius]
        return sorted(near, key=lambda loc: loc.index)

    def active_locs_at(self, level, x, z):
        return [loc for loc in self.loc_at.get((level, x, z), ()) if loc.active]

    # ---- paths, reach and sight on the live collision of a level (sim.paths)
    def _changed_in(self, level, x0, x1, z0, z1):
        return any(l == level and x0 <= x <= x1 and z0 <= z <= z1 for l, x, z in self.changed)

    def live_flags(self, level):
        return lambda x, z: self.flags(level, x, z, npc_bits=False)

    def window_changed(self, level, sx, sz):
        """Whether a changed tile lies in the path finder's search window from (sx, sz)."""
        return self._changed_in(level, sx - paths.HALF, sx + paths.HALF - 1, sz - paths.HALF, sz + paths.HALF - 1)

    def tree(self, level, sx, sz):
        """The path finder's tree from (sx, sz): stored, unless a changed tile lies in its window."""
        if not self.window_changed(level, sx, sz):
            return paths.Tree.stored(self.t, paths.source_index(self.t, level, sx, sz))
        if self._trees is None or len(self._trees) > 64:
            self._trees = {}
        tree = self._trees.get((level, sx, sz))
        if tree is None:
            steps = self._steps.get(level)
            if steps is None:
                flags = self._flags[level * self._wh:(level + 1) * self._wh]
                steps = self._steps[level] = paths.step_masks(np.asarray(flags, np.int64).reshape(self.t.size))
            tree = self._trees[(level, sx, sz)] = paths.search(steps, self._x0, self._z0, sx, sz, self.t.bounded)
        return tree

    def path_to_tile(self, level, sx, sz, x, z):
        return paths.path_to_tile(self.tree(level, sx, sz), sx, sz, x, z)

    def entity_reaching(self, level, x, z):
        if self._changed_in(level, x - 1, x + 1, z - 1, z + 1):
            return paths.entity_reaching_live(self.live_flags(level), x, z)
        return paths.entity_reaching(self.t, level, x, z)

    def loc_reaching(self, loc):
        """The tiles from which a loc is operable (Player.inOperableDistance's reachedLoc)."""
        t = self.t.loc_types[loc.type]
        box = paths.REACH_BOX
        if loc.index < self.n_stored_locs and (loc.shape, loc.angle) == (loc.base_shape, loc.base_angle) \
                and t['forceapproach'] == self.t.loc_types[loc.base]['forceapproach'] \
                and not self._changed_in(loc.level, loc.x - box, loc.x + box + loc.width, loc.z - box,
                                         loc.z + box + loc.length):
            tiles = self.t.meta['loc_reach'][loc.index]
            return list(zip(tiles[0::2], tiles[1::2]))
        return paths.loc_reaching(self.live_flags(loc.level), loc.x, loc.z, loc.width, loc.length, loc.angle, loc.shape,
                                  t['forceapproach'])

    def inside(self, x, z):
        """Whether a tile lies in the data area: the declared world of a simulated layout."""
        return self._x0 <= x <= self._x1 and self._z0 <= z <= self._z1

    def client_grid(self, level):
        """(the client's collision, its tryMove steps) of a level's live flags, until a loc changes them."""
        grid = self._client.get(level)
        if grid is None:
            flags = np.asarray(self._flags[level * self._wh:(level + 1) * self._wh], np.int64).reshape(self.t.size)
            cflags = paths.client_flags(flags)
            grid = self._client[level] = (cflags, paths.client_steps(cflags))
        return grid

    def scene_tree(self, level, x, z, base):
        """Client.tryMove's search from (x, z) over the scene at `base` (paths.client_search), until a loc changes the
        collision. Answers outside the tables are Unsupported (World.path)."""
        if self._scene_trees is None or len(self._scene_trees) > 64:
            self._scene_trees = {}
        key = (level, x, z, base)
        tree = self._scene_trees.get(key)
        if tree is None:
            cflags, steps = self.client_grid(level)
            tree = self._scene_trees[key] = self.path(lambda: paths.client_search(steps, cflags, self._x0, self._z0, x, z,
                                                                                  base, self.t.bounded))
        return tree

    def player_scene(self, p):
        """The player's search: from its server tile over the scene of its build area."""
        return self.scene_tree(p.level, p.x, p.z, paths.scene_base(p.origin_x, p.origin_z))

    def line_of_sight(self, level, ax, az, bx, bz):
        if self._changed_in(level, min(ax, bx) - 1, max(ax, bx) + 1, min(az, bz) - 1, max(az, bz) + 1):
            return paths.line_of_sight(self.live_flags(level), ax, az, bx, bz)
        return self.t.line_of_sight(level, ax, az, bx, bz)

    # ---- npcs (entity/Npc.ts turn)
    def npc_lifecycle(self, npc):
        """Npc.processLifecycle: a deleted npc respawns at its spawn after respawnrate ticks."""
        if not npc.delayed and npc.lifecycle_tick > 0:
            npc.lifecycle_tick -= 1
            if npc.lifecycle_tick == 0 and not npc.active:
                self.npc_respawn(npc)

    def npc_respawn(self, npc):
        """World.addNpc after npc_del: Npc.resetEntity(true) at the spawn."""
        nt = self.t.npc_types[npc.base]
        npc.type, npc.active, npc.x, npc.z = npc.base, True, npc.start_x, npc.start_z
        npc.levels = list(nt['stats'])
        npc.queue, npc.path, npc.heroes = [], [], {}
        npc.life += 1
        npc.vars = dict(NPC_VARS)
        self.npc_reset_defaults(npc, nt)
        npc.tele = True
        self._occupy(npc, True)
        self.npc_events.append(npc)

    def npc_turn(self, npc):
        if npc.active:
            if npc.delayed and self.tick >= npc.delayed_until:
                npc.delayed = False
            if not npc.delayed and npc.script is not None:
                self.resume_npc(npc)
        self.npc_lifecycle(npc)
        if not npc.is_valid():
            return
        nt = self.t.npc_types[npc.type]
        if npc.hunt_mode != -1:
            hunt = nt['hunt']
            if hunt is not None and (hunt['nobodyNear'] != 0 or npc.observers > 0 or hunt['type'] == 1):
                if hunt['type'] != 1:
                    raise Unsupported(f'npc hunt of type {hunt["type"]}')
                npc.hunt_clock += 1
        self.npc_consume_hunt_target(npc, nt)
        self.npc_regen(npc, nt)
        if npc.timer_interval > 0:
            npc.timer_clock += 1
            if npc.timer_clock >= npc.timer_interval:
                script = nt['triggers'].get('AI_TIMER')
                if script and self.decide('npc_timer', True, npc):
                    self.npc_script(npc, script)
                    npc.timer_clock = 0
        self.npc_process_queue(npc, nt)
        self.npc_movement(npc, nt)

    def npc_consume_hunt_target(self, npc, nt):
        hunt = nt['hunt']
        if npc.hunt_target is None or hunt is None or hunt['type'] == 0:
            return
        if 22 <= hunt['findNewMode'] <= 41:  # NpcMode.QUEUE1..QUEUE20
            raise Unsupported('hunt that runs a queue')
        npc.target, npc.mode = npc.hunt_target, hunt['findNewMode']
        npc.hunt_target, npc.hunt_clock = None, 0
        if not hunt['findKeepHunting']:
            npc.hunt_mode = -1

    def npc_regen(self, npc, nt):
        npc.regen_clock += 1
        if npc.regen_clock >= nt['regenrate']:
            npc.regen_clock = 0
            base = nt['stats']
            npc.levels = [lv + (lv < b) - (lv > b) for lv, b in zip(npc.levels, base)]

    def npc_queue(self, npc, queue_id, arg, delay, source=None):
        """NPC_QUEUE; `source` (who queued it) is evidence for the kill audit, not engine state."""
        npc.queue.append([queue_id, arg, delay, source])

    def remove_npc(self, npc):
        """World.removeNpc with the type's respawnrate, scaled for one player."""
        self._occupy(npc, False)
        npc.active = False
        npc.lifecycle_tick = (4000 - 1) * self.t.npc_types[npc.type]['respawnrate'] // 4000

    def npc_magic_attack_roll(self, npc):
        return (npc.levels[NPC_MAGIC] + 9) * (self.npc_param(npc, 'magicattack') + 64)

    def player_damage(self, p, damage, source=None):
        """Player.applyDamage, sampled for death as the native monitor samples it (Player.dead). Each hit, a zero
        included, is recorded with the hitpoints before and after it and the npc that queued it (evidence, as the
        native capture's agent-hp rows; sim.audit ignores it)."""
        before = p.levels[HITPOINTS]
        p.levels[HITPOINTS] = max(0, p.levels[HITPOINTS] - damage)
        p.dead |= p.levels[HITPOINTS] <= 0
        self.events.append((self.tick, 'agent_hp', dict(cause='damage', planned=damage, before=before,
                                                        after=p.levels[HITPOINTS], source=source)))

    def stat_sub(self, p, stat, constant, percent):
        """STAT_SUB."""
        current = p.levels[stat]
        subbed = current - int(constant + p.base_levels[stat] * percent / 100)
        p.levels[stat] = max(subbed, 0)
        p.dead |= p.levels[HITPOINTS] <= 0
        if subbed != current:
            changestat(self, p)

    def stat_add(self, p, stat, constant, percent):
        """STAT_ADD."""
        current = p.levels[stat]
        added = current + int(constant + p.base_levels[stat] * percent / 100)
        p.levels[stat] = min(added, 255)
        if added != current:
            changestat(self, p)

    def stat_heal(self, p, stat, constant, percent, cause=None):
        """STAT_HEAL: up to the base level; [changestat] runs even when nothing healed. A hitpoints gain is recorded
        with its cause (eat, regen) as evidence."""
        current = p.levels[stat]
        healed = current + int(constant + p.base_levels[stat] * percent / 100)
        p.levels[stat] = max(min(healed, p.base_levels[stat]), current)
        if stat == HITPOINTS and p.levels[stat] != current:
            self.events.append((self.tick, 'agent_hp', dict(cause=cause, planned=healed - current, before=current,
                                                            after=p.levels[stat], source=None)))
        if healed != current:
            changestat(self, p)

    def obj_add(self, p, level, x, z, obj, count, duration):
        """OBJ_ADD for the script's player, or OBJ_ADDALL (p None: no receiver): a members obj needs a members world;
        one obj per unstackable item; a stack joins the receiver's stack of that obj on the tile."""
        receiver = None if p is None else p.name
        if obj == -1 or count == -1:
            return
        t = self.t.obj_types[obj]
        if t['dummyitem']:
            raise Unsupported(f"dummy obj {t['debugname']}")
        if t['members'] and not self.members:
            return
        for amount in ([1] * count if not t['stackable'] or count == 1 else [count]):
            mine = next((o for o in self.objs if not o.static and o.active and o.type == obj
                         and (o.level, o.x, o.z) == (level, x, z) and o.receiver == receiver), None) if t['stackable'] else None
            if mine is not None and mine.count + amount <= STACK_LIMIT:
                mine.count += amount
                mine.ticks = duration
                continue
            self.objs.append(Obj(obj, amount, x, z, static=False, ticks=duration, receiver=receiver, level=level))

    def take_obj(self, p, obj):
        """OBJ_TAKEITEM into the inventory: a spawn respawns after its type's respawnrate."""
        if not obj.is_valid():
            return
        added = p.inv.add(self.t, obj.type, obj.count)
        self.events.append((self.tick, 'pickup', dict(obj=obj.type, count=added)))
        self.remove_obj(obj)

    def remove_obj(self, obj):
        """World.removeObj with the type's respawnrate (OBJ_TAKEITEM, OBJ_DEL): a drop is gone, a spawn respawns after
        it; nothing for an obj already gone."""
        if not obj.active:
            return
        obj.active = False
        if obj.static:
            obj.ticks = (4000 - 1) * self.t.obj_types[obj.type]['respawnrate'] // 4000
        else:
            self.objs.remove(obj)

    def npc_process_queue(self, npc, nt):
        """Npc.processQueue: a request's delay counts down while the npc is not delayed."""
        if not npc.active:
            return
        for request in _links(npc.queue):
            if not npc.delayed:
                request[2] -= 1
            if not npc.delayed and request[2] <= 0:
                _unlink(npc.queue, request)
                nt = self.t.npc_types[npc.type]
                name = nt['triggers'].get(f'AI_QUEUE{request[0]}')
                if name:
                    self.npc_run(npc, self.npc_queue_script(name, request[0])(self, npc, request[1], request[3]))

    def npc_movement(self, npc, nt):
        """Npc.processMovementInteraction."""
        if npc.delayed or not npc.active:
            return
        if npc.mode == MODE_NULL:
            npc.mode = nt['defaultmode']
        if npc.mode == MODE_NONE:
            self.npc_update_movement(npc, nt)
            return
        if npc.mode == MODE_WANDER:
            if self._wander[0] != self.tick:
                self._wander = (self.tick, self.rng.prefix(R_WANDER_GO, self.tick))
            if nt['moverestrict'] != NOMOVE and self.rng.draw(self._wander[1], npc.nid) < 0.125:
                r = nt['wanderrange']
                # Npc.wander: Math.round(Math.random() * 2r - r) on each axis.
                dx = _js_round(self.rng(R_WANDER_DX, self.tick, npc.nid) * (r * 2) - r)
                dz = _js_round(self.rng(R_WANDER_DZ, self.tick, npc.nid) * (r * 2) - r)
                dest = (npc.start_x + dx, npc.start_z + dz)
                if dest != (npc.x, npc.z):
                    npc.path = [dest]
            self.npc_update_movement(npc, nt)
            on_spawn = (npc.x, npc.z) == (npc.start_x, npc.start_z)
            npc.stuck += 1
            if npc.stuck - 1 > 500:
                if not on_spawn:
                    self.npc_teleport(npc, npc.start_x, npc.start_z)
                npc.stuck = 0
            return
        if npc.mode == MODE_PATROL:
            raise Unsupported(f'patrolling npc {nt["debugname"]}')
        if npc.target is None or not self.npc_validate_target(npc, nt):
            self.npc_reset_defaults(npc, nt)
            return
        if npc.mode == MODE_PLAYERESCAPE:
            self.npc_escape(npc, nt)
        elif npc.mode == MODE_PLAYERFOLLOW:
            raise Unsupported('npc following a player')
        elif npc.mode == MODE_PLAYERFACE:
            pass
        elif npc.mode == MODE_PLAYERFACECLOSE:
            if _distance_to(npc, npc.target) > 1:
                self.npc_reset_defaults(npc, nt)
        else:
            self.npc_ai_mode(npc, nt)

    def npc_validate_target(self, npc, nt):
        """Npc.validateTarget + targetWithinMaxRange for a player target."""
        p = npc.target
        if p is not self.player or p.level != npc.level:  # a replaced or logged-out player, another floor
            return False
        if self._op_mode(npc.mode):
            dx, dz = abs(p.x - npc.start_x), abs(p.z - npc.start_z)
            if max(dx, dz) > nt['maxrange'] + 1 or (dx == nt['maxrange'] + 1 and dz == nt['maxrange'] + 1):
                return False
        elif self._ap_mode(npc.mode):
            if _distance_sw(npc.start_x, npc.start_z, p) > nt['maxrange'] + nt['attackrange']:
                return False
        elif npc.mode == MODE_PLAYERESCAPE:
            if _distance_sw(npc.start_x, npc.start_z, p) > nt['maxrange'] and \
                    max(abs(npc.x - npc.start_x), abs(npc.z - npc.start_z)) > nt['maxrange']:
                return False
        elif _distance_sw(npc.start_x, npc.start_z, p) > nt['maxrange'] + 1:
            return False
        return True

    @staticmethod
    def _op_mode(mode):
        return MODE_OPPLAYER1 <= mode <= MODE_OPPLAYER1 + 4

    @staticmethod
    def _ap_mode(mode):
        return MODE_APPLAYER1 <= mode <= MODE_APPLAYER1 + 4

    def npc_param(self, npc, name):
        params = self.t.npc_types[npc.type]['params']
        if name in params:
            return params[name]
        return self.t.meta['param_defaults'][name]

    def npc_ai_mode(self, npc, nt):
        """Npc.aiMode: interact before moving, else a naive step toward the target, then again."""
        npc.stuck = 0
        if self.npc_try_interact(npc, nt):
            return
        p = npc.target
        if nt['moverestrict'] != NOMOVE:
            waypoint = self.naive_path(npc.level, npc.x, npc.z, p.x, p.z, 1, 1, size=npc.width)
            if waypoint[0] != (npc.x, npc.z):
                npc.path = waypoint
        moved = self.npc_update_movement(npc, nt)
        if moved and not nt['givechase']:
            self.npc_reset_defaults(npc, nt)
            return
        if npc.target is not None:
            self.npc_try_interact(npc, nt)

    def npc_try_interact(self, npc, nt):
        p = npc.target
        if self._op_mode(npc.mode):
            if npc.width != 1:
                raise Unsupported('a large npc in melee')
            if (npc.x, npc.z) in self.entity_reaching(p.level, p.x, p.z):
                self.npc_ai_script(npc, nt, f'AI_OPPLAYER{npc.mode - MODE_OPPLAYER1 + 1}')
                return True
        elif self._ap_mode(npc.mode):
            # PathingEntity.inApproachDistance: not from under the target; an npc's line of
            # sight is taken from the target back to it.
            if npc.width != 1:
                raise Unsupported('a large npc approaching')
            if _distance_to(npc, p) <= nt['attackrange'] and (npc.x, npc.z) != (p.x, p.z) \
                    and self.line_of_sight(p.level, p.x, p.z, npc.x, npc.z):
                self.npc_ai_script(npc, nt, f'AI_APPLAYER{npc.mode - MODE_APPLAYER1 + 1}')
                return True
        return False

    def npc_ai_script(self, npc, nt, trigger):
        name = nt['triggers'].get(trigger)
        if not name:
            return
        fn = NPC_AI.get(name)
        if fn is None:
            raise Unsupported(f'npc script {name}')
        self.npc_run(npc, fn(self, npc, npc.target))

    def npc_queue_script(self, name, queue_id):
        if queue_id == 3:  # death: the area's death scripts are extracted (sim.rs2)
            return lambda world, npc, arg, source: npc_death_script(world, npc, name)
        fn = NPC_QUEUES.get(name)
        if fn is None:
            raise Unsupported(f'npc script {name}')
        return fn

    def npc_run(self, npc, gen):
        """Npc.executeScript: a script that suspends becomes the npc's active script."""
        npc.script = gen
        self.resume_npc(npc)

    def npc_escape(self, npc, nt):
        """Npc.playerEscapeMode: a step away from the player, diagonal first, within maxrange of
        the spawn; reset after 5 stuck ticks unless at maxrange on both axes."""
        p = npc.target
        if _distance_sw(npc.x, npc.z, p) > 25:
            self.npc_reset_defaults(npc, nt)
            return
        dx = -1 if p.x >= npc.x else 1
        dz = -1 if p.z >= npc.z else 1
        collision = {0: 0, 1: 1, 3: 3, 4: 4}.get(nt['moverestrict'])
        if collision is None or npc.width != 1:
            raise Unsupported('a retreating npc of this kind')
        within = lambda x, z: max(abs(x - npc.start_x), abs(z - npc.start_z)) <= nt['maxrange']
        if self.can_travel(npc.level, npc.x, npc.z, dx, dz, 1, FLAG_NPC, collision) and within(npc.x + dx, npc.z + dz):
            npc.path = [(npc.x + dx, npc.z + dz)]
        elif self.can_travel(npc.level, npc.x, npc.z, dx, 0, 1, FLAG_NPC, collision) and within(npc.x + dx, npc.z):
            npc.path = [(npc.x + dx, npc.z)]
        elif self.can_travel(npc.level, npc.x, npc.z, 0, dz, 1, FLAG_NPC, collision) and within(npc.x, npc.z + dz):
            npc.path = [(npc.x, npc.z + dz)]
        if not self.npc_update_movement(npc, nt):
            npc.stuck += 1
        at_max = abs(npc.x - npc.start_x) >= nt['maxrange'] and abs(npc.z - npc.start_z) >= nt['maxrange']
        if npc.stuck >= 5 and not at_max:
            self.npc_reset_defaults(npc, nt)
            npc.stuck = 0

    def npc_set_mode(self, npc, mode, target=None):
        """NPC_SETMODE: none/wander/patrol clear the interaction, null resets the defaults, a
        targeted mode sets an interaction with the script's player (or resets without one)."""
        nt = self.t.npc_types[npc.type]
        if mode in (MODE_NONE, MODE_WANDER, MODE_PATROL):
            npc.target, npc.mode = None, mode
        elif mode == MODE_NULL:
            self.npc_reset_defaults(npc, nt)
        elif target is None or not target.is_valid():
            self.npc_reset_defaults(npc, nt)
        else:
            npc.target, npc.mode = target, mode

    def npc_reset_defaults(self, npc, nt):
        """Npc.resetDefaults: clear the interaction, default mode, hunt and timer reset."""
        npc.target = None
        npc.mode = nt['defaultmode']
        npc.hunt_mode, npc.huntrange, npc.hunt_clock, npc.hunt_target = nt['huntmode'], nt['huntrange'], 0, None
        npc.timer_interval = nt['timer']

    def npc_update_movement(self, npc, nt):
        if nt['moverestrict'] == NOMOVE or not npc.path:
            moved = False
        else:
            collision = {0: 0, 1: 1, 3: 3, 4: 4}.get(nt['moverestrict'])
            if collision is None:
                raise Unsupported(f'move restrict {nt["moverestrict"]}')
            self.process_movement(npc, collision, FLAG_NPC)
            moved = (npc.x, npc.z) != (npc.last_tick_x, npc.last_tick_z)
        if moved:
            npc.last_movement = self.tick + 1
            npc.stuck = 0
        return moved

    def npc_teleport(self, npc, x, z):
        self._occupy(npc, False)
        npc.x, npc.z = x, z
        self._occupy(npc, True)
        npc.path = []
        npc.tele = True

    def npc_script(self, npc, name):
        fn = NPC_SCRIPTS.get(name)
        if fn is None:
            raise Unsupported(f'npc script {name}')
        self.npc_run(npc, fn(self, npc))

    def resume_npc(self, npc):
        try:
            step = next(npc.script)
        except StopIteration:
            npc.script = None
            return
        if step[0] == 'npc_delay':
            npc.delayed, npc.delayed_until = True, self.tick + 1 + step[1]
        elif step[0] == 'npc_arrivedelay':
            npc.delayed, npc.delayed_until = True, step[1]
        else:
            raise Unsupported(f'npc script suspension {step}')

    # ---- player scripts (ScriptRunner semantics the slice uses)
    def execute(self, p, gen, protect=False, force=False, name=''):
        """Player.executeScript. Generators yield ('pausebutton',) or ('delay', n)."""
        if gen is None:
            return
        if not force and protect and (p.protect or p.delayed):
            return
        if protect:
            p.protect = True
        self._run(p, gen, protect)

    def _run(self, p, gen, protect):
        try:
            step = next(gen)
        except StopIteration:
            if protect:
                p.protect = False
            if p.script is not None and p.script[0] is gen:
                p.script = None
                p.resume_buttons = []
                if 'main' not in p.modal_state:  # close chat dialogues, leave a main modal
                    p.close_modal(False)
            return
        if protect:
            p.protect = False
        if step[0] == 'pausebutton':
            p.script = (gen, 'pausebutton')
        elif step[0] == 'delay':
            p.delayed, p.delayed_until = True, self.tick + 1 + step[1]
            p.script = (gen, 'suspended')
            p.protect = protect
        elif step[0] == 'world_delay':  # WORLD_SUSPENDED (Player.executeScript): the script leaves the player for the world queue
            if p.script is not None and p.script[0] is gen:
                p.script = None  # natively the reference stays, read by nothing in that state
            self.queue.append([step[1] + 1, gen])
        else:
            raise Unsupported(f'script suspension {step}')

    def resume(self, p):
        gen = p.script[0]
        p.script = (gen, 'running')
        self._run(p, gen, True)

    def decide(self, kind, default, obj=None):
        """A random outcome the replay check may control: 'gather' (a gathering roll for obj)
        and 'deplete' (a resource's depletion roll); otherwise the keyed draw stands."""
        return default if self.control is None else self.control(kind, default, obj, self.tick)

    def random(self, n, *keys):
        """RuneScript random(n): floor(JavaRandom.nextDouble() * n)."""
        return int(self.rng(R_SCRIPT, self.tick, *keys) * n)

    def randominc(self, n, *keys):
        """RuneScript randominc(n): floor(JavaRandom.nextDouble() * (n + 1))."""
        return int(self.rng(R_SCRIPT, self.tick, *keys) * (n + 1))

    def stat_random(self, p, stat, low, high, *keys):
        """PlayerOps STAT_RANDOM: success when value > floor(rand * 256)."""
        return stat_random_value(p.levels[stat], low, high) > int(self.rng(R_SCRIPT, self.tick, *keys) * 256)

    def add_xp(self, p, stat, xp):
        """Player.addXp with the world XP rate; a level gained queues [advancestat] (engine queue)."""
        if xp == 0:
            return
        p.stats[stat] = min(p.stats[stat] + xp * self.xp_rate, 2_000_000_000)
        before, current = p.base_levels[stat], p.levels[stat]
        if p.levels[stat] == p.base_levels[stat]:
            p.levels[stat] = level_for(p.stats[stat])
        p.base_levels[stat] = level_for(p.stats[stat])
        if p.base_levels[stat] > before:
            if p.levels[stat] < before:
                p.levels[stat] += p.base_levels[stat] - before
            if stat == HITPOINTS:  # a level gained raises current hitpoints by as much: evidence of a heal
                self.events.append((self.tick, 'agent_hp', dict(cause='levelup', planned=p.levels[stat] - current,
                                                                before=current, after=p.levels[stat], source=None)))
            changestat(self, p)
            p.enqueue(f'[advancestat,{SKILLS[stat]}]', lambda s=stat: levelup(self, p, s), kind='engine')

    # ---- interaction (entity/Player.ts processInteraction, tryInteract)
    def in_operable(self, p, target):
        if target.level != p.level:
            return False
        if isinstance(target, Npc):
            return (p.x, p.z) in self.entity_reaching(p.level, target.x, target.z)
        if isinstance(target, Loc):
            return (p.x, p.z) in self.loc_reaching(target)
        # Player.inOperableDistance for an obj: beside it as for an npc, or on it
        return (p.x, p.z) == (target.x, target.z) or (p.x, p.z) in self.entity_reaching(p.level, target.x, target.z)

    def in_approach(self, p, rng, target):
        """PathingEntity.inApproachDistance for a 1x1 player."""
        if isinstance(target, Npc) and target.x <= p.x < target.x + target.width and target.z <= p.z < target.z + target.length:
            return False
        dist = _distance_to(p, target)
        if dist > rng:
            return False
        # isApproached -> LineValidator: the ray ends at the target's tile nearest the player
        return self.line_of_sight(p.level, p.x, p.z, *_nearest_tile(p, target))

    def trigger(self, target, op):
        triggers = self._types(target)['triggers']
        return triggers.get(_trigger_name(self.t, op))

    def _types(self, target):
        if isinstance(target, Npc):
            return self.t.npc_types[target.type]
        if isinstance(target, Loc):
            return self.t.loc_types[target.type]
        return self.t.obj_types[target.type]

    def validate_target(self, p):
        t = p.target
        if t is None or t.level != p.level:
            return False
        if t.type != p.target_type:
            return False
        return t.is_valid()

    def try_interact(self, p, allow_op_scenery):
        if p.target is None or not p.can_access():
            return False
        target = p.target
        op_name = self.trigger(target, p.target_op + AP_TO_OP)
        ap_name = self.trigger(target, p.target_op)
        if op_name and (isinstance(target, Npc) or allow_op_scenery) and self.in_operable(p, target):
            p.target = None
            p.clear_waypoints()
            self.run_trigger(p, op_name, target)
            p.next_target = p.target
            p.target = target
            return True
        if ap_name and self.in_approach(p, p.ap_range, target):
            p.ap_range_called = False
            saved = list(p.path)
            p.target = None
            p.clear_waypoints()
            self.run_trigger(p, ap_name, target)
            p.next_target = p.target
            p.target = target
            if p.next_target is not None:
                p.clear_waypoints()
            elif p.ap_range_called:
                p.path = saved
                p.target = target
                return False
            return True
        if self.in_approach(p, p.ap_range, target):
            p.ap_range = -1
            return False
        if (isinstance(target, Npc) or allow_op_scenery) and self.in_operable(p, target):
            self.default_op(p, target)
            return True
        return False

    def default_op(self, p, target):
        """Player.defaultOp outside production: the debug line, then the default message."""
        t = self._types(target)
        prefix = 'oplocu' if p.target_op == APLOCU else 'opnpcu' if p.target_op == APNPCU else None
        if prefix is None:
            raise Unsupported(f'no script for op {p.target_op} on {t["debugname"]}')
        self.mes(f'No trigger for [{prefix},{t["debugname"]}]')
        self.mes(self.text('dm_default'))
        p.clear_waypoints()

    def run_trigger(self, p, name, target):
        fn = SCRIPTS.get(name) or archive_script(self, name)
        if fn is None:
            raise Unsupported(f'engine script {name}')
        self.execute(p, fn(self, p, target), protect=True, name=name)

    def path_to_pathing_target(self, p):
        """Player.pathToPathingTarget for the NAIVE strategy players get with clientRoutefinder:
        from under the npc a random step, else on the last waypoint, when it ends more than a
        tile from the npc, a naive path (routefinder/NaivePathFinder.ts)."""
        t = p.target
        if not isinstance(t, Npc) or not p.can_access():
            return
        if t.x <= p.x < t.x + t.width and t.z <= p.z < t.z + t.length:
            if self.rng(R_SCRIPT, self.tick, 900) < 0.5:
                step = (-1 if self.rng(R_SCRIPT, self.tick, 901) < 0.5 else 1, 0)
            else:
                step = (0, -1 if self.rng(R_SCRIPT, self.tick, 901) < 0.5 else 1)
            p.queue_waypoint(p.x + step[0], p.z + step[1])
            return
        if len(p.path) <= 1 and p.allow_repath:
            if not p.path or abs(t.x - p.path[-1][0]) > 1 or abs(t.z - p.path[-1][1]) > 1:
                p.queue_waypoints(self.naive_path(p.level, p.x, p.z, t.x, t.z, t.width, t.length))

    def naive_path(self, level, sx, sz, dx, dz, dw, dl, size=1):
        """NaivePathFinder.findNaivePath for a 1x1 source, no extra flag, NORMAL collision."""
        if size != 1:
            raise Unsupported('naive path of a large mover')
        if not (dx >= sx + 1 or dx + dw <= sx or dz >= sz + 1 or dz + dl <= sz):
            d = (((-1, 0), (1, 0), (0, 1), (0, -1)))[int(self.rng(R_SCRIPT, self.tick, 902) * 4)]
            return [(sx + d[0], sz + d[1])]
        x, z = _naive_destination(sx, sz, dx, dz)
        if _is_diagonal(x, z, dx, dz, dw, dl) or not (dx >= x + 1 or dx + dw <= x or dz >= z + 1 or dz + dl <= z):
            return [(x, z)]
        while x != dx and z != dz:
            step_x, step_z = (dx > x) - (dx < x), (dz > z) - (dz < z)
            if self.can_travel(level, x, z, step_x, step_z, 1, 0, 0):
                x, z = x + step_x, z + step_z
            elif step_x and self.can_travel(level, x, z, step_x, 0, 1, 0, 0):
                x += step_x
            elif step_z and self.can_travel(level, x, z, 0, step_z, 1, 0, 0):
                z += step_z
            else:
                break
        return [(x, z)]

    def path(self, query):
        try:
            return query()
        except paths.OutsideTables as error:
            raise Unsupported(str(error)) from error

    def process_interaction(self, p):
        p.next_target = None
        interacted = False
        if p.target is not None and p.can_access():
            if not self.validate_target(p):
                p.clear_interaction()
                return
            interacted = self.try_interact(p, False)
        if not interacted:
            self.path_to_pathing_target(p)
            self.update_movement(p)
            if p.target is not None and p.can_access():
                interacted = self.try_interact(p, p.steps_taken == 0)
                if not interacted and not p.ap_range_called and not p.path and p.steps_taken == 0:
                    self.mes("I can't reach that!")
                    p.clear_interaction()
        if p.next_target is not None:
            p.target = p.next_target
        elif interacted and not p.ap_range_called:
            p.clear_interaction()

    def update_movement(self, p):
        """Player.updateMovement: no step at moveSpeed INSTANT (a teleJump this cycle); a
        waypoint may lie outside the supported area, the step check catches leaving it."""
        if p.path and not p.instant:
            self.process_movement(p, 0, 0)
            if self.t.source(p.level, p.x, p.z) < 0:
                raise Unsupported(f'player left the supported area at {p.level},{p.x},{p.z}')
        if p.steps_taken > 0:
            p.last_movement = self.tick + 1

    def tele_jump(self, p, x, z, level):
        """PathingEntity.teleJump: teleport (collision and zone follow the player; waypoints
        stay) and moveSpeed INSTANT, so no step until the cycle's cleanup."""
        if self.t.source(level, x, z) < 0:
            raise Unsupported(f'a jump to {level},{x},{z}, outside the supported tiles')
        self._occupy(p, False)
        p.x, p.z, p.level = x, z, level
        self._occupy(p, True)
        p.instant = True

    def teleport(self, p, x, z):
        """PathingEntity.teleport on the player's level: the tile changes at once (collision and zone follow the player;
        waypoints stay). No moveSpeed INSTANT: only a change of level or a teleJump sets it."""
        if self.t.source(p.level, x, z) < 0:
            raise Unsupported(f'a teleport to {p.level},{x},{z}, outside the supported tiles')
        self._occupy(p, False)
        p.x, p.z = x, z
        self._occupy(p, True)

    # ---- bank (interface_bank)
    def open_bank(self, p):
        """[label,openbank]: reorganize, open bank_main with bank_side."""
        p.varps['bankcert'] = 0
        reorganize(p.bank)
        p.open_main_side('bank_main')

    def close_bank(self, p):
        """[if_close,bank_main] -> [label,closebank]: queue reorganize_bank (delay 0)."""
        p.enqueue('[queue,reorganize_bank]', lambda: _gen(reorganize, p.bank))

    def bank_transfer(self, p, slot, requested, withdraw):
        """[label,bank_withdraw] / [label,bank_deposit] via the agent's itemTransfer (protected, forced)."""
        src, dst = (p.bank, p.inv) if withdraw else (p.inv, p.bank)
        item = src.items[slot]
        if item is None:
            return
        obj = item[0]
        number = min(requested, src.total(obj))
        if withdraw:
            uncert = self.uncert(obj)
            if p.inv.itemspace2(self.t, uncert, number) and p.inv.freespace() == 0:
                self.mes(self.text('bank_no_space'))
                return
            if p.inv.itemspace2(self.t, uncert, number):
                self.mes(self.text('bank_carry_all' if self.t.obj_types[uncert]['stackable'] else 'bank_withdraw_many'))
            move = number - p.inv.itemspace2(self.t, uncert, number)
            if move > 0:
                self.move_item(p, BANK, INV, obj, move)
        else:
            if self.t.obj_types[obj]['params'].get('unbankable'):
                raise Unsupported('unbankable item')
            uncert = self.uncert(obj)
            if STACK_LIMIT - number < p.bank.total(uncert):
                self.mes(self.text('bank_full_stack'))
                return
            overflow = p.bank.itemspace2(self.t, uncert, number)
            if overflow >= number:
                self.mes(self.text('bank_full'))
                return
            if number - overflow > 0:
                self.move_item(p, INV, BANK, obj, number - overflow)

    def uncert(self, obj):
        t = self.t.obj_types[obj]
        return t['certlink'] if t['certtemplate'] >= 0 and t['certlink'] >= 0 else obj

    def move_item(self, p, src, dst, obj, count):
        """INV_MOVEITEM_UNCERT."""
        done = p.inventory(src).remove(obj, count)
        if done:
            target = self.uncert(obj)
            p.inventory(dst).add(self.t, target, done)
            self.events.append((self.tick, 'transfer', dict(src=src, dst=dst, obj=obj, to=target, count=done)))

    def run_held(self, p, name):
        """An OPHELD or OPHELDU script, executed protected (env-ts opHeld and useOnHeld)."""
        fn = held_script(self, name)
        if fn is None:
            raise Unsupported(f'held script {name}')
        self.execute(p, fn(self, p, name), protect=True, name=name)

    def produce(self, p, obj, script):
        """inv_add from a gathering script: the only source of production credit."""
        added = p.inv.add(self.t, obj, 1)
        self.events.append((self.tick, 'produce', dict(obj=obj, count=added, script=script)))
        return added


# ---- helpers -----------------------------------------------------------------------------------
def stat_random_value(level, low, high):
    """STAT_RANDOM's threshold: a roll succeeds on value > floor(rand * 256)."""
    return (low * (99 - level)) // 98 + (high * (level - 1)) // 98 + 1


def _js_round(v):
    """Math.round: halves round up."""
    import math
    return int(math.floor(v + 0.5))


def _distance_to(p, target):
    """CoordGrid.distanceTo: Chebyshev distance between rectangles."""
    w, l = getattr(target, 'width', 1), getattr(target, 'length', 1)
    if isinstance(target, Loc) and target.angle & 1:
        w, l = l, w
    dx = max(target.x - p.x, p.x - (target.x + w - 1), 0)
    dz = max(target.z - p.z, p.z - (target.z + l - 1), 0)
    return max(dx, dz)


def _nearest_tile(p, target):
    """The tile of the target's rectangle nearest to p (LineValidator's ray end)."""
    w, l = getattr(target, 'width', 1), getattr(target, 'length', 1)
    if isinstance(target, Loc) and target.angle & 1:
        w, l = l, w
    return min(max(p.x, target.x), target.x + w - 1), min(max(p.z, target.z), target.z + l - 1)


def _naive_destination(sx, sz, dx, dz):
    """NaivePathFinder.naiveDestination for a 1x1 source and a 1x1 destination box."""
    diagonal, anti = sx - dx + (sz - dz), sx - dx - (sz - dz)
    sw, nw, ne, se = anti < 0, diagonal >= 0, anti > 0, diagonal <= 0
    if sw and not nw:
        off = min(diagonal + 1, 0) if diagonal >= -1 else -(1 + anti) if anti > -1 else 0
        return dx - 1, dz + off
    if nw and not ne:
        off = min(anti + 1, 0) if anti >= -1 else max(diagonal - 1, 0) if diagonal < 1 else 0
        return dx + off, dz + 1
    if ne and not se:
        off = 1 - anti if anti <= 1 else max(diagonal - 1, 0) if diagonal < 1 else 0
        return dx + 1, dz + off
    if not (se and not sw):
        raise Unsupported('naive destination with no side')
    off = min(diagonal + 1, 0) if diagonal > -1 else max(anti - 1, 0) if anti < 1 else 0
    return dx + off, dz - 1


def _is_diagonal(sx, sz, dx, dz, dw, dl):
    return ((sx + 1 == dx and sz + 1 == dz) or (sx - 1 == dx + dw - 1 and sz - 1 == dz + dl - 1)
            or (sx + 1 == dx and sz - 1 == dz + dl - 1) or (sx - 1 == dx + dw - 1 and sz + 1 == dz))


def _distance_sw(x, z, target):
    """CoordGrid.distanceToSW: Chebyshev distance from a tile to a 1x1 target."""
    return max(abs(target.x - x), abs(target.z - z))


_TRIGGER_NAMES = {}


def _trigger_name(t, op):
    if not _TRIGGER_NAMES:
        _TRIGGER_NAMES.update({v: k for k, v in t.meta['trigger_ids'].items()})
    return _TRIGGER_NAMES[op]


def _gen(fn, *args):
    yield from ()
    fn(*args)


def _links(items):
    """LinkList.all() over a list: each link's successor is read when the link is visited, so
    a request added while the last one runs waits for the next pass (the authentic queue
    timing quirk Player.processQueue describes), and a successor unlinked meanwhile is still
    visited, after which the pass ends."""
    node = items[0] if items else None
    while node is not None:
        k = next((i for i, item in enumerate(items) if item is node), -1)
        following = items[k + 1] if 0 <= k < len(items) - 1 else None
        yield node
        if k < 0:
            return
        node = following


def _unlink(items, node):
    k = next((i for i, item in enumerate(items) if item is node), -1)
    if k >= 0:
        del items[k]


def reorganize(inv):
    """[proc,reorganize_inv]: move later stacks into earlier empty slots, in order."""
    for count in range(inv.capacity):
        if inv.items[count] is None:
            later = next((i for i in range(count + 1, inv.capacity) if inv.items[i] is not None), -1)
            if later < 0:
                break
            inv.items[count], inv.items[later] = inv.items[later], None


# ---- content scripts -----------------------------------------------------------------------------
def tree_row(world, loc):
    rows = world.t.meta.get('content', {}).get('trees', {})
    row = rows.get(world.t.loc_types[loc.type]['debugname'])
    if row is None:
        raise Unsupported(f'tree {world.t.loc_types[loc.type]["debugname"]} has no woodcutting row')
    return row


AXES = ('rune_axe', 'adamant_axe', 'mithril_axe', 'black_axe', 'steel_axe', 'iron_axe', 'bronze_axe')


def axe_checker(world, p, print_mes):
    """[proc,woodcutting_axe_checker]: the best axe worn in the right hand or carried
    (only axes the area can hold are in the tables)."""
    rhand = p.worn.items[WEARPOS_RHAND][0] if p.worn.items[WEARPOS_RHAND] else None
    for name in AXES:
        obj = world.t.obj_ids.get(name)
        if obj is not None and (rhand == obj or p.inv.total(obj) > 0):
            return obj
    if print_mes:
        world.mes(world.text('wc_noaxe'))
    return None


def p_oploc(world, p, loc, op):
    """P_OPLOC: stopAction, walk onto the loc if not operable, set a SCRIPT interaction."""
    if not world.t.loc_types[loc.type]['op'][op - 1]:
        return
    p.stop_action()
    if not world.in_operable(p, loc):
        p.queue_waypoint(loc.x, loc.z)
    p.set_interaction(loc, APLOC1 + op - 1)


def p_opnpc(world, p, npc, op):
    if not world.t.npc_types[npc.type]['op'][op - 1]:
        return
    p.stop_action()
    p.set_interaction(npc, APNPC1 + op - 1)


def p_opobj(world, p, obj, op):
    """P_OPOBJ (script/handlers/PlayerOps.ts:1038-1054): nothing for a type without that op; else stopAction, a waypoint
    on the obj's tile and a SCRIPT interaction."""
    if not (world.t.obj_types[obj.type]['op'] or [None] * 5)[op - 1]:
        return
    p.stop_action()
    p.queue_waypoint(obj.x, obj.z)
    p.set_interaction(obj, APOBJ1 + op - 1)


def attempt_cut_tree(world, p, loc):
    """[oploc1,_tree] -> [label,attempt_cut_tree] (skill_woodcutting/scripts/woodcut.rs2)."""
    row = tree_row(world, loc)
    level = row['levelrequired']
    if p.levels[WOODCUTTING] < level:
        world.mes(world.text('wc_level', level=level))
        return
    if p.inv.freespace() < 1:
        world.mes(world.text('wc_full'))
        return
    if p.afk_ready:  # afk_event (debug world): a random woodcutting event follows
        p.afk_ready = False
        raise Unsupported('random event (afk_event) during woodcutting')
    if p.varps['action_delay'] < world.tick:
        if level > 0 and axe_checker(world, p, True) is None:
            return
        p.varps['action_delay'] = world.tick + 3
        p_oploc(world, p, loc, 1)
        return
    axe = axe_checker(world, p, True)
    if axe is None:
        return
    world.mes(world.text('wc_swing'))
    if p.varps['action_delay'] == world.tick:
        yield from get_logs(world, p, loc, row, axe)
        return
    p_oploc(world, p, loc, 3)
    yield from ()


def cut_tree(world, p, loc):
    """[oploc3,_tree] -> [label,cut_tree]."""
    axe = axe_checker(world, p, True)
    if axe is None:
        return
    row = tree_row(world, loc)
    if p.levels[WOODCUTTING] < row['levelrequired']:
        world.mes(world.text('wc_level', level=row['levelrequired']))
        return
    if p.inv.freespace() < 1:
        world.mes(world.text('wc_full'))
        return
    if p.varps['action_delay'] < world.tick:
        p.varps['action_delay'] = world.tick + 3
    elif p.varps['action_delay'] == world.tick:
        yield from get_logs(world, p, loc, row, axe)
        return
    p_oploc(world, p, loc, 3)


def get_logs(world, p, loc, row, axe):
    """[label,get_logs]: one roll; a normal tree always depletes, others 1 in 8."""
    low, high = row['successchance'][world.t.obj_types[axe]['debugname']]
    respawn = row['respawnrate']
    deplete = 8
    if respawn == 0:
        respawn = 120 + world.random(80, 1)
        deplete = 0
    product = world.t.obj_id(row['product'])
    if world.decide('gather', world.stat_random(p, WOODCUTTING, low, high, 2), product):
        world.mes(world.text('wc_get', product=world.t.obj_types[product]['name'].lower()))
        world.add_xp(p, WOODCUTTING, row['productexp'])
        world.produce(p, product, '[label,get_logs]')
        if world.decide('deplete', world.random(deplete, 3) == 0):
            respawn = world.decide('respawn', (respawn * (4000 - 1)) // 4000)  # ~scale_by_playercount, one player
            stage = world.t.loc_types[loc.type]['params']['next_loc_stage']
            world.change_loc(loc, stage, loc.shape, loc.angle, respawn)
            return
    p_oploc(world, p, loc, 3)
    yield from ()


def use_on_tree(world, p, loc):
    """[oplocu,_tree]: an axe starts op1 (delayed a tick, as the script notes); anything else
    shows the default message."""
    if world.t.obj_types[p.last_useitem]['debugname'] in AXES:
        p_oploc(world, p, loc, 1)
    else:
        world.mes(world.text('dm_default'))
    yield from ()



# ---- mining (skill_mining/scripts/mining.rs2, pickaxe_checker.rs2, prospect.rs2) ------------------------
def compact(world):
    return world.t.meta['content']['compact']


def rock_row(world, loc):
    """db_find(mining_table:rock, loc_type) (configs/mine.dbrow)."""
    return compact(world)['rocks'].get(world.t.loc_types[loc.type]['debugname'])


def pickaxe_checker(world, p):
    """[proc,pickaxe_checker]: the best pickaxe worn in the right hand or carried whose
    levelrequire the Mining level meets."""
    rhand = p.worn.items[WEARPOS_RHAND][0] if p.worn.items[WEARPOS_RHAND] else None
    for name in compact(world)['pickaxes']:
        obj = world.t.obj_ids.get(name)
        if obj is not None and p.levels[MINING] >= world.obj_param(obj, 'levelrequire') \
                and (rhand == obj or p.inv.total(obj) > 0):
            return obj
    return None


def mining(get_ore, first):
    """[oploc1,_mining_rock_*] @mining_firstswing and [oploc3,_mining_rock_*] @mining_continue: the
    refusals (empty rock, inventory, level, pickaxe) in the script's order, then a roll each
    mining_rate + 1 ticks, as woodcutting's action_delay cadence."""
    def script(world, p, loc):
        if first and world.loc_param_or_default(loc, 'mining_rock_empty') == 1:
            if world.t.loc_types[loc.type]['debugname'] == 'loc_4030':
                raise Unsupported('the empty loc_4030 rock')
            world.mes(world.text('mine_none'))
            return
        row = rock_row(world, loc)
        if row is None:
            world.mes(world.text('dm_default'))
            return
        if p.inv.freespace() < 1:
            yield from mesbox(world, p, world.text('mine_full', ore=row['ore']))
            return
        if p.levels[MINING] < row['level']:
            yield from mesbox(world, p, world.text('mine_level', level=row['level']))
            return
        if first and p.afk_ready:
            p.afk_ready = False
            raise Unsupported('random event (afk_event) during mining')
        pickaxe = pickaxe_checker(world, p)
        if pickaxe is None:
            yield from mesbox(world, p, world.text('mine_nopick'))
            return
        rate = world.obj_param(pickaxe, 'mining_rate')
        if first:
            if p.varps['action_delay'] < world.tick:
                p.varps['action_delay'] = world.tick + rate
                p_oploc(world, p, loc, 1)
                return
            world.mes(world.text('mine_swing'))
            if p.varps['action_delay'] == world.tick:
                yield from get_ore(world, p, loc, row)
                return
        elif p.varps['action_delay'] < world.tick:
            p.varps['action_delay'] = world.tick + rate
        elif p.varps['action_delay'] == world.tick:
            yield from get_ore(world, p, loc, row)
            return
        p_oploc(world, p, loc, 3)
    return script


def get_ore(fast):
    """[label,get_ore_normal] / [label,get_ore_fast]: a 1 in 256 gem roll (mining_gem_table), else a
    Mining roll; success (after a tick's p_delay for normal rocks) depletes the rock for its
    respawnrate scaled by one player, adds the ore and the xp; otherwise op3 again."""
    name = '[label,get_ore_fast]' if fast else '[label,get_ore_normal]'

    def script(world, p, loc, row):
        c = compact(world)
        chance = c['gem_chance']
        neck = p.worn.items[WEARPOS_FRONT]
        if neck and world.t.obj_types[neck[0]]['category'] == world.t.meta['content']['categories']['category_557'] \
                and world.members:
            raise Unsupported('a charged amulet of glory')
        product = world.t.obj_id(row['output'])
        low, high = row['chance']
        if world.random(chance, 4) == 1:  # random($chance) = ^true: a gem (or nothing), and mining goes on
            roll = world.random(c['gem_roll'], 6)
            gem = next((g for below, g in c['gems'] if roll < below), None)
            if gem is not None:
                world.produce(p, world.t.obj_id(gem), name)
                world.mes(world.text('mine_gem', gem=add_article(world.t.obj_types[world.t.obj_id(c['cut_gems'][gem])]['name'])))
        elif world.decide('gather', world.stat_random(p, MINING, low, high, 5), product):
            if not fast:
                yield ('delay', 0)
            respawn = world.decide('respawn', (row['respawn'] * (4000 - 1)) // 4000)  # ~scale_by_playercount
            world.change_loc(loc, world.loc_param_or_default(loc, 'next_loc_stage_mining'), loc.shape, loc.angle, respawn)
            world.produce(p, product, name)
            world.add_xp(p, MINING, row['exp'])
            world.mes(world.text('mine_get', ore=row['ore']))
            return
        p_oploc(world, p, loc, 3)
    return script


def add_article(word):
    """[proc,add_article]."""
    return f"{'an' if word[:1].lower() in 'aeiou' else 'a'} {word}"


def prospect(world, p, loc):
    """[oploc2,_mining_rock_*] @prospect_rock(finish_prospect_normal): the message, p_delay(0) and
    p_delay(1), then what the rock holds now."""
    world.mes(world.text('prospect_start'))
    yield ('delay', 0)
    yield ('delay', 1)
    if world.loc_param_or_default(loc, 'mining_rock_empty') == 1:
        world.mes(world.text('prospect_empty'))
        return
    row = rock_row(world, loc)
    if row is None:
        world.mes(world.text('dm_default'))
        return
    world.mes(world.text('prospect_ore', ore=row['ore']))

# ---- doors and gates (doors/scripts/doors.rs2, door_procs.rs2; general_use/scripts/gates.rs2) --------
WALL_STRAIGHT, WALL_DIAGONAL, CENTREPIECE_STRAIGHT = 0, 9, 10
LOC_DURATION = 500  # the scripts' loc_del and loc_add durations
# [proc,door_open] and [proc,door_close]: the new door's offset by shape and angle (W, N, E, S).
DOOR_OPEN = {WALL_STRAIGHT: ((-1, 0), (0, 1), (1, 0), (0, -1)), WALL_DIAGONAL: ((0, 1), (1, 0), (0, -1), (-1, 0))}
DOOR_CLOSE = {WALL_STRAIGHT: ((0, 1), (1, 0), (0, -1), (-1, 0)), WALL_DIAGONAL: ((1, 0), (0, -1), (-1, 0), (0, 1))}


def door(offsets, turn):
    """[oploc1,_door_closed] (turn 1) and [oploc1,_door_opened] (turn 3): the door goes for
    500 ticks and its next stage stands on the tile in front, turned a quarter."""
    def script(world, p, loc):
        if loc.shape not in offsets:
            raise Unsupported(f'a door of shape {loc.shape}')
        dx, dz = offsets[loc.shape][loc.angle]
        x, z = loc.x + dx, loc.z + dz
        if (p.x, p.z) == (x, z) and loc.shape == WALL_DIAGONAL:
            raise Unsupported('a diagonal door that moves the player out of its way (p_teleport)')
        stage = world.loc_param(loc, 'next_loc_stage')
        world.remove_loc(loc, LOC_DURATION)
        world.loc_add(loc.level, x, z, stage, (loc.angle + turn) % 4, loc.shape, LOC_DURATION)
        yield from ()
    return script


def _gate_close_offset(angle, mod):
    """[proc,gate_set_close]."""
    return ((-mod, 0), (0, mod), (mod, 0), (0, -mod))[angle]


def _gate_open_offset(angle, outer):
    """[proc,gate_set_open]."""
    x, z = ((0, -1), (-1, 0), (0, 1), (1, 0))[angle]
    if outer:
        x, z = ((x + 1, z), (x, z - 1), (x - 1, z), (x, z + 1))[angle]
    return x, z


def _gate_pair(x, z, angle, outer):
    """[proc,get_pair_coord]: the tile whose zone holds the other gate."""
    d = -1 if outer else 1
    return ((x, z + d), (x + d, z), (x, z - d), (x - d, z))[angle]


def _gate_partner(world, loc, outer, category):
    """loc_findallzone over the pair tile's zone: the first active loc within a tile of this
    gate in the named category."""
    cat = world.t.meta['content']['categories'][category]
    for other in world.locs_in_zone(loc.level, *_gate_pair(loc.x, loc.z, loc.angle, outer)):
        if max(abs(other.x - loc.x), abs(other.z - loc.z)) <= 1 and world.t.loc_types[other.type]['category'] == cat:
            return other
    return None


def _swing_gate(world, loc, opening):
    """[proc,open_gate] / [proc,close_gate] with the main gate active: it and its outer half
    each go for 500 ticks, their next stages standing beside, turned a quarter."""
    angle, x, z = loc.angle, loc.x, loc.z
    turn = 3 if opening else 1
    main = _gate_close_offset(angle, 1) if opening else _gate_open_offset(angle, False)
    outer = _gate_close_offset(angle, 2) if opening else _gate_open_offset(angle, True)
    stage = world.loc_param(loc, 'next_loc_stage')
    world.remove_loc(loc, LOC_DURATION)
    world.loc_add(loc.level, x + main[0], z + main[1], stage, (angle + turn) % 4, WALL_STRAIGHT, LOC_DURATION)
    partner = _gate_partner(world, loc, False, 'gate_outer_closed' if opening else 'gate_outer_open')
    if partner is not None:
        stage = world.loc_param(partner, 'next_loc_stage')
        world.remove_loc(partner, LOC_DURATION)
        world.loc_add(loc.level, x + outer[0], z + outer[1], stage, (angle + turn) % 4, WALL_STRAIGHT, LOC_DURATION)


def gate(opening, outer):
    """[oploc1,_gate_main_closed] ~open_gate, [oploc1,_gate_main_open] ~close_gate, and the
    outer halves, which swing the main gate found beside them (nothing if there is none)."""
    def script(world, p, loc):
        if not outer:
            _swing_gate(world, loc, opening)
        else:
            main = _gate_partner(world, loc, True, 'gate_main_closed' if opening else 'gate_main_open')
            if main is not None:
                _swing_gate(world, main, opening)
        yield from ()
    return script


def stairs(name):
    """[oploc1,loc_1722] (climb up) and [oploc1,loc_1723] (climb down) (ladders+stairs/scripts/
    stairs.rs2): p_arrivedelay, then p_telejump to switch_coord(loc_coord)'s case (sim/build.py
    stairs); a coord with no literal case goes to @unhandled_stairs, which is not ported."""
    def script(world, p, loc):
        if p.last_movement >= world.tick:  # p_arrivedelay: a tick's wait after a step this cycle
            yield ('delay', 0)
        dest = world.t.stairs.get((name, loc.level, loc.x, loc.z))
        if dest is None:
            raise Unsupported(f'{name} at {loc.level},{loc.x},{loc.z} (@unhandled_stairs or a computed case)')
        world.tele_jump(p, *dest)
    return script


def check_fish_equipment(world, p, equipment):
    """[proc,check_fish_equipment]: the tool's fail message in a mesbox."""
    if p.inv.total(world.t.obj_id(equipment)) < 1:
        yield from mesbox(world, p, world.text(f'fish_fail_{equipment}'))
        return False
    if equipment == 'fishing_rod':
        if p.inv.total(world.t.obj_id('fishing_bait')) < 1:
            yield from mesbox(world, p, world.text('fish_no_bait'))
            return False
    return True


def attempt_fish_shrimp(world, p, npc):
    """[opnpc1,_saltfish] -> [label,attempt_fish_shrimp] (fishing_spots/saltfish.rs2)."""
    if not (yield from check_fish_equipment(world, p, 'net')):
        return
    if p.inv.freespace() < 1:
        yield from mesbox(world, p, world.text('fish_full'))
        return
    if p.varps['action_delay'] < world.tick:
        p.varps['action_delay'] = world.tick + 4
        p_opnpc(world, p, npc, 1)
        return
    if p.varps['action_delay'] == world.tick + 3:
        world.mes(world.text('fish_cast'))
        p_opnpc(world, p, npc, 4)
    elif p.varps['action_delay'] == world.tick:
        fish_roll(world, p, 'raw_shrimp', 'raw_anchovies' if p.levels[FISHING] >= 15 else None)
    if p.afk_ready:
        p.afk_ready = False
        raise Unsupported('random event (afk_event) during fishing')


def fish_shrimp(world, p, npc):
    """[opnpc4,_saltfish] -> [label,fish_shrimp]."""
    if not (yield from check_fish_equipment(world, p, 'net')):
        return
    if p.inv.freespace() < 1:
        yield from mesbox(world, p, world.text('fish_full'))
        return
    if p.varps['action_delay'] < world.tick:
        p.varps['action_delay'] = world.tick + 5
    elif p.varps['action_delay'] == world.tick:
        fish_roll(world, p, 'raw_shrimp', 'raw_anchovies' if p.levels[FISHING] >= 15 else None)
    p_opnpc(world, p, npc, 4)


def fish_roll(world, p, fish1, fish2):
    """[proc,fish_roll]: the better fish first, then the common one (no bait for nets)."""
    for k, fish in enumerate((fish2, fish1)):
        if fish is None:
            continue
        s = world.t.meta['content']['fish'][fish]
        if world.decide('gather', world.stat_random(p, FISHING, s['success_low'], s['success_high'], 10 + k),
                        world.t.obj_id(fish)):
            world.produce(p, world.t.obj_id(fish), '[proc,fish_roll]')
            world.add_xp(p, FISHING, s['productexp'])
            world.mes(s['productmessage'])
            return


def attempt_fish_sardine(world, p, npc):
    """[opnpc3,_saltfish] -> [label,attempt_fish_sardine]: the slice has no rod, so only the
    level and equipment refusals can run."""
    if p.levels[FISHING] < 5:
        yield from mesbox(world, p, world.text('fish_level5'))
        return
    if not (yield from check_fish_equipment(world, p, 'fishing_rod')):
        return
    raise Unsupported('bait fishing with a rod')


def use_on_saltfish(world, p, npc):
    """[opnpcu,_saltfish]: a net fishes; an item without a fishing-equipment struct shows
    the default message (fishing_wrong_spot_message); fishing equipment is not in the slice."""
    t = world.t.obj_types[p.last_useitem]
    if t['debugname'] == 'net':
        yield from attempt_fish_shrimp(world, p, npc)
    elif 'fish_equipment_struct' in t['params'] or t['debugname'] in ('fly_fishing_rod', 'fishing_rod', 'big_net'):
        raise Unsupported(f"{t['debugname']} on a fishing spot")
    else:
        world.mes(world.text('dm_default'))


def move_fishing_spot(world, npc):
    """[ai_timer,_saltfish] -> [label,move_fishing_spot] (skill_fishing/scripts/fishing_movement.rs2)."""
    nt = world.t.npc_types[npc.type]
    coords = world.t.meta['enums'][str(nt['params']['fishing_movement_enum'])]['values']
    dest = (npc.x, npc.z)
    for i in range(50):
        _, x, z, level = coords[int(world.rng(R_NPC_SCRIPT, world.tick, npc.nid, i) * len(coords))]
        if not any(o is not npc and (o.level, o.x, o.z) == (level, x, z)
                   and world.t.npc_types[o.type]['category'] == nt['category'] for o in world.npcs):
            if level != npc.level:
                raise Unsupported('a fishing spot moving to another level')
            dest = (x, z)
            break
    yield ('npc_delay', 2)
    world.npc_teleport(npc, *dest)
    npc.timer_interval = 280 + int(world.rng(R_NPC_SCRIPT, world.tick, npc.nid, 99) * 250)


def imp_timer(world, npc):
    """[ai_timer,imp]: half the time, out of combat, a teleport; then npc_setmode(null) and a
    timer of random_range(50, 200)."""
    if int(world.rng(R_NPC_SCRIPT, world.tick, npc.nid, 0) * 2) == 0 and npc.vars['lastcombat'] + 8 < world.tick:
        npc_imp_teleport(world, npc)
    world.npc_set_mode(npc, MODE_NULL)
    npc.timer_interval = 50 + int(world.rng(R_NPC_SCRIPT, world.tick, npc.nid, 100) * 151)
    yield from ()


def flavour_timer(base, spread):
    """Sheep, cow and duck timers (general/scripts/flavour_text): overhead speech, which the
    agent does not observe, and the next interval."""
    def script(world, npc):
        npc.timer_interval = base + int(world.rng(R_NPC_SCRIPT, world.tick, npc.nid, 0) * spread)
        yield from ()
    return script


def npc_poison_timer(world, npc):
    """[ai_timer,_] -> npc_poison_timer: nothing for an npc that is not poisoned."""
    yield from ()


# ---- combat (skill_combat/scripts) -----------------------------------------------------------------
# The player fights in melee with what it wields (or unarmed), with no prayers, spells or special
# attacks; npcs fight with the default melee scripts or, for the dark wizards, spells.
BONUSES = ('stabattack', 'slashattack', 'crushattack', 'magicattack', 'rangeattack', 'stabdefence', 'slashdefence',
           'crushdefence', 'magicdefence', 'rangedefence', 'strengthbonus', 'prayerbonus')


def equip_get_bonuses(world, p):
    """[proc,equip_get_bonuses] (player/scripts/equip.rs2): every worn obj's bonuses summed; a quiver's
    rangebonus counts only beside a matching bow or crossbow."""
    totals = dict.fromkeys(BONUSES + ('rangebonus',), 0)
    cats = world.t.meta['content']['categories']
    rhand = p.worn.items[WEARPOS_RHAND][0] if p.worn.items[WEARPOS_RHAND] else None
    for slot, obj, _ in p.worn.listed():
        cat = world.t.obj_types[obj]['category']
        weapon = world.t.obj_types[rhand]['category'] if rhand is not None else None
        if slot != WEARPOS_QUIVER or rhand is None or (cat == cats['arrows'] and weapon == cats['weapon_bow']) \
                or (cat == cats['bolts'] and weapon == cats['weapon_crossbow']):
            totals['rangebonus'] += world.obj_param(obj, 'rangebonus')
        for name in BONUSES:
            totals[name] += world.obj_param(obj, name)
    return totals


def weapon_styles(world, weapon):
    """[proc,combat_get_weapon_style_data]: the combat_style_table row of the weapon's category."""
    c = compact(world)
    if weapon is None or (not world.members and world.t.obj_types[weapon]['members']):
        return c['styles']['weapon_unarmed_table']
    names = {v: k for k, v in world.t.meta['content']['categories'].items()}
    table = c['style_of'].get(names.get(world.t.obj_types[weapon]['category']), 'weapon_unarmed_table')
    return c['styles'][table]


def player_combat_stat(world, p):
    """[proc,player_combat_stat] (player_combat_stat.rs2): rolls and max hit from the current levels,
    the worn bonuses and the wielded weapon's style (no prayer is on: combat_effective_stat is the level)."""
    b = equip_get_bonuses(world, p)
    rhand = p.worn.items[WEARPOS_RHAND][0] if p.worn.items[WEARPOS_RHAND] else None
    styles = weapon_styles(world, rhand)
    p.varps['com_mode'] = min(p.varps['com_mode'], len(styles['damagestyle']) - 1)
    damagetype = styles['damagetype'][p.varps['com_mode']]
    damagestyle = styles['damagestyle'][p.varps['com_mode']]
    attack, strength, defence, ranged = STYLE_BONUSES.get(damagestyle, (0, 0, 0, 0))
    lv = p.levels
    magic_defence = (7 * lv[MAGIC] + 3 * lv[DEFENCE]) // 10 + 8
    eff_attack, eff_defence, eff_ranged = lv[ATTACK] + 8 + attack, lv[DEFENCE] + 8 + defence, lv[RANGED] + 8 + ranged
    stat = lambda effective, bonus: effective * (bonus + 64)  # [proc,combat_stat]
    p.varps.update(damagetype=damagetype, damagestyle=damagestyle,
                   com_stabattack=stat(eff_attack, b['stabattack']), com_slashattack=stat(eff_attack, b['slashattack']),
                   com_crushattack=stat(eff_attack, b['crushattack']), com_magicattack=stat(lv[MAGIC] + 9, b['magicattack']),
                   com_rangeattack=stat(eff_ranged, b['rangeattack']), com_stabdef=stat(eff_defence, b['stabdefence']),
                   com_slashdef=stat(eff_defence, b['slashdefence']), com_crushdef=stat(eff_defence, b['crushdefence']),
                   com_magicdef=stat(magic_defence, b['magicdefence']), com_rangedef=stat(eff_defence, b['rangedefence']))
    if damagetype in (STAB, SLASH, CRUSH):  # [proc,combat_maxhit]
        p.varps['com_maxhit'] = (stat(lv[STRENGTH] + 8 + strength, b['strengthbonus']) + 320) // 640
    elif damagetype == RANGED_STYLE:
        p.varps['com_maxhit'] = (stat(eff_ranged, b['rangebonus']) + 320) // 640


def changestat(world, p):
    """Player.changeStat queues [changestat,_] (gosub player_combat_stat) on the engine queue."""
    p.enqueue('[changestat,_]', lambda: _gen(player_combat_stat, world, p), kind='engine')


ROLL_VARPS = {STAB: 'stab', SLASH: 'slash', CRUSH: 'crush', RANGED_STYLE: 'range', MAGIC_STYLE: 'magic'}


def player_in_combat_check(world, p, npc):
    """[proc,player_in_combat_check]: single-way combat allows one opponent each."""
    tick = world.tick
    if not world.is_multi(npc.level, npc.x, npc.z):
        if p.varps['lastcombat_pvp'] + 8 > tick:
            world.mes(world.text('under_attack'))
            return False
        if p.varps['lastcombat'] + 8 > tick and p.varps['aggressive_npc'] not in (npc.uid, None):
            world.mes(world.text('under_attack'))
            return False
        if npc.vars['lastcombat'] + 8 > tick and npc.vars['aggressive_player'] != p.name:
            world.mes(world.text('fighting_other'))
            return False
    if npc.vars['macro_target'] not in (None, p.name):
        raise Unsupported('an npc of a random event')
    return True


def player_attackrange(world, p):
    """[proc,player_attackrange] without autocast: the wielded weapon's attackrange (melee: 0)."""
    rhand = p.worn.items[WEARPOS_RHAND][0] if p.worn.items[WEARPOS_RHAND] else None
    if rhand is None:
        return 0
    attackrange = min(world.obj_param(rhand, 'attackrange'), 10)
    return min(attackrange + 2, 10) if p.varps['damagestyle'] == 6 else attackrange  # ^style_ranged_longrange


def player_combat_start_ap(world, p, npc):
    """[apnpc2,_] -> [label,player_combat_start_ap]: in melee the attack range is 0, so this
    waits for melee distance (p_aprange(0)), after the in-combat check if it fails."""
    attackrange = player_attackrange(world, p)
    if (attackrange <= 1 and not player_in_combat_check(world, p, npc)) or _distance_to(npc, p) > attackrange:
        p.ap_range, p.ap_range_called = attackrange, True
        return
    yield from player_combat_start(world, p, npc)


def player_combat_start(world, p, npc):
    """[opnpc2,_] -> [label,player_combat_start]: no special attack (%sa_attack stays false) or
    autocast; a ranged style is not ported."""
    if not player_in_combat_check(world, p, npc):
        return
    if p.varps['damagetype'] == RANGED_STYLE:
        raise Unsupported('a ranged attack')
    yield from player_melee_attack(world, p, npc)


def npc_is_attackable(world, p, npc):
    """[proc,npc_is_attackable]: its quest and minigame cases name npcs outside the slice."""
    nt = world.t.npc_types[npc.type]
    if nt['debugname'] in ('wormbrain', 'priestperilguarddog', 'grip', 'ikov_lucien2', 'gorad', 'desertminingcaptain',
                           'ikov_firewarrior', 'delrith', 'count_draynor', 'general_khazard'):
        raise Unsupported(f"attacking {nt['debugname']}")
    return len(nt['op']) > 1 and bool(nt['op'][1])


def player_melee_attack(world, p, npc):
    """[label,player_melee_attack] (player_melee.rs2): a hit roll, damage and xp, the npc's retaliation
    and damage queued, the attack delay (4 unarmed, else the weapon's attackrate), and op2 again."""
    p.stop_action()
    if not npc_is_attackable(world, p, npc):
        return
    if p.varps['action_delay'] > world.tick:
        p_opnpc(world, p, npc, 2)
        return
    if npc.levels[NPC_HP] == 0:
        p.stop_action()
        return
    damage = 0
    if player_npc_hit_roll(world, p, npc, p.varps['damagetype']):
        maxhit = min(p.varps['com_maxhit'], world.npc_param(npc, 'max_dealt'))  # ~npc_max_dealt
        damage = world.randominc(maxhit, npc.nid, 3)
        capped = min(damage, npc.levels[NPC_HP])
        give_combat_experience(world, p, p.varps['damagestyle'], capped, npc.vars['xp_mult'])
        if capped >= 1:
            npc.heroes[p.name] = npc.heroes.get(p.name, 0) + capped
    weapon = p.worn.items[WEARPOS_RHAND][0] if p.worn.items[WEARPOS_RHAND] else None
    if weapon is not None and damage > 0 and world.obj_param(weapon, 'poison_severity') > 0:
        raise Unsupported('a poisoned weapon')
    npc_retaliate(world, p, npc, 0)
    world.npc_queue(npc, 2, damage, 0, source=p.name)
    p.varps['action_delay'] = world.tick + (4 if weapon is None else world.obj_param(weapon, 'attackrate'))
    p_opnpc(world, p, npc, 2)
    yield from ()


def player_npc_hit_roll(world, p, npc, damagetype):
    attack_roll = p.varps[f'com_{ROLL_VARPS[damagetype]}attack']
    return world.randominc(attack_roll, npc.nid, 1) > world.randominc(npc_defence_roll(world, npc, damagetype), npc.nid, 2)


def npc_defence_roll(world, npc, damagetype):
    """[proc,npc_defence_roll_specific]: style bonus 1 (+9), the npc's defence bonus."""
    stat = npc.levels[NPC_MAGIC if damagetype == MAGIC_STYLE else NPC_DEFENCE]
    bonus = world.npc_param(npc, ('stab', 'slash', 'crush', 'range', 'magic')[damagetype] + 'defence')
    return (stat + 9) * (bonus + 64)


def give_combat_experience(world, p, damagestyle, damage, multiplier):
    """[proc,give_combat_experience] for the melee styles (its special npcs are not in the area)."""
    base = damage * 10
    xp = lambda percent: (base * percent // 100) * multiplier // 1000
    if damagestyle == 0:
        world.add_xp(p, ATTACK, xp(400))
    elif damagestyle == 1:
        world.add_xp(p, STRENGTH, xp(400))
    elif damagestyle == 2:
        world.add_xp(p, DEFENCE, xp(400))
    elif damagestyle == 3:
        for stat in (ATTACK, STRENGTH, DEFENCE):
            world.add_xp(p, stat, xp(133))
    else:
        raise Unsupported(f'combat xp for damage style {damagestyle}')
    world.add_xp(p, HITPOINTS, xp(133))


def npc_retaliate(world, p, npc, delay):
    """[proc,npc_retaliate]."""
    world.npc_queue(npc, 1, 0, delay)
    npc.vars['aggressive_player'] = p.name
    npc.vars['lastcombat'] = max(npc.vars['lastcombat'], world.tick)


def npc_found(world, uid):
    """NPC_FINDUID: the npc in that slot, if it still has that type (an npc waiting to
    respawn is found too)."""
    npc = world.npc_by_nid.get(uid & 0xffff)
    return npc if npc is not None and npc.type == (uid >> 16) & 0xffff else None


def combat_damage_player(world, p, damage, source=None):
    """[queue,combat_damage_player] (no ring of recoil is worn) -> [proc,damage_self]; `source`, the npc that queued
    it, is evidence only."""
    if p.levels[HITPOINTS] == 0:
        return
    p.close_modal()
    world.player_damage(p, damage, source)
    if p.levels[HITPOINTS] == 0:
        p.enqueue('[queue,player_death]', lambda: player_death(world, p))
    yield from ()  # ~ring_of_life_check: no ring is worn


def player_death(world, p):
    """[queue,player_death] -> [proc,player_death]: %death, a tick, stop and close, three
    ticks. A goal episode has ended by then (at most a chunk after the death); the rest
    (the message, item loss, the teleport to Lumbridge) is not in the slice."""
    p.varps['death'] = 1
    yield ('delay', 1)
    p.stop_action()
    p.close_modal()
    yield ('delay', 3)
    p.varps['death'] = 0
    raise Unsupported('the player death sequence after its first ticks')


def playerhit_n_retaliate(world, p, uid):
    """[queue,playerhit_n_retaliate]: auto-retaliate, when the player is otherwise idle."""
    npc, rhand = npc_found(world, uid), p.worn.items[WEARPOS_RHAND]
    if p.varps['option_nodef'] == 0 and npc is not None and not p.busy2():
        if p.varps['action_delay'] < world.tick:
            p.varps['action_delay'] = world.tick + (world.obj_param(rhand[0], 'attackrate') if rhand else 4) // 2
        p_opnpc(world, p, npc, 2)
    yield from ()


# npc side (skill_combat/scripts/npc)
def npc_check_notcombat(world, npc, p):
    if not world.is_multi(npc.level, npc.x, npc.z):
        if p.varps['lastcombat_pvp'] + 8 > world.tick:
            return False
        if p.varps['lastcombat'] + 8 > world.tick and p.varps['aggressive_npc'] not in (None, npc.uid):
            return False
    return True


def npc_check_notcombat_self(world, npc, p):
    if not world.is_multi(npc.level, npc.x, npc.z):
        if npc.vars['lastcombat'] + 8 > world.tick and npc.vars['aggressive_player'] not in (None, p.name):
            return False
    return True


def npc_set_attack_vars(world, npc, p):
    p.varps['lastcombat'] = world.tick
    npc.vars['attacking'] = p.name
    p.varps['aggressive_npc'] = npc.uid


def npc_default_attack(world, npc, p):
    """[ai_opplayer2,_] -> [proc,npc_default_attack] -> [proc,npc_meleeattack]."""
    if p.varps['death'] == 1:
        world.npc_set_mode(npc, MODE_NULL)
        return
    if npc.vars['action_delay'] > world.tick:
        return
    if not npc_check_notcombat(world, npc, p):
        world.npc_set_mode(npc, MODE_NULL)
        return
    if not npc_check_notcombat_self(world, npc, p):
        return
    attack_roll = (npc.levels[NPC_ATTACK] + 9) * (world.npc_param(npc, 'attackbonus') + 64)
    damagetype = world.npc_param(npc, 'damagetype')
    defence_roll = p.varps[f'com_{ROLL_VARPS[damagetype]}def']
    maxhit = ((npc.levels[NPC_STRENGTH] + 9) * (world.npc_param(npc, 'strengthbonus') + 64) + 320) // 640
    if world.randominc(attack_roll, npc.nid, 11) > world.randominc(defence_roll, npc.nid, 12):
        playerhit_n_melee(world, npc, p, world.randominc(maxhit, npc.nid, 13), world.npc_param(npc, 'attackrate'))
        npc_poison_player(world, npc)
        return
    playerhit_n_melee(world, npc, p, 0, world.npc_param(npc, 'attackrate'))
    yield from ()


def playerhit_n_melee(world, npc, p, damage, delay):
    """[proc,playerhit_n_melee]: close the player's interface, queue the damage and the
    retaliation (no protect prayer is on)."""
    p.close_modal()
    npc_set_attack_vars(world, npc, p)
    npc.vars['action_delay'] = world.tick + delay
    if npc.levels[NPC_HP] == 0:
        return
    p.enqueue('[queue,combat_damage_player]', lambda: combat_damage_player(world, p, damage, npc.nid))
    uid = npc.uid
    p.enqueue('[queue,playerhit_n_retaliate]', lambda: playerhit_n_retaliate(world, p, uid))


def npc_poison_player(world, npc):
    if world.npc_param(npc, 'poison_severity') > 0:
        raise Unsupported('poison')


def npc_default_retaliate(ap):
    """[ai_queue1,_] -> [proc,npc_default_retaliate] (melee, opplayer2); the _ap variant for
    ranged and magic npcs sets applayer2."""
    mode = MODE_APPLAYER2 if ap else MODE_OPPLAYER2

    def script(world, npc, arg, source=None):
        p = world.player
        if p is None or npc.vars['aggressive_player'] != p.name:  # finduid
            return
        retreat = world.npc_param(npc, 'retreat')
        if npc.levels[NPC_HP] <= retreat:
            world.npc_set_mode(npc, MODE_PLAYERESCAPE, p)
        ready = npc.vars['action_delay'] + 8 < world.tick or npc.mode not in (MODE_APPLAYER2, MODE_OPPLAYER2)
        if not ready:
            return
        npc.vars['action_delay'] = world.tick + world.npc_param(npc, 'attackrate') // 2
        npc.vars['attacking'] = npc.vars['aggressive_player']
        if npc.levels[NPC_HP] > retreat:
            world.npc_set_mode(npc, mode, p)
            if not world.npc_validate_target(npc, world.t.npc_types[npc.type]):  # npc_inrange
                world.npc_set_mode(npc, MODE_PLAYERESCAPE, p)
        yield from ()
    return script


def npc_default_damage(world, npc, damage, source=None):
    """[ai_queue2,_] -> [proc,npc_default_damage]: a hit (or block), and the death queue at 0. Each
    hit is recorded with the life's hitpoints before it, and a defeat with its source (sim.audit)."""
    if npc.levels[NPC_HP] == 0:
        return
    full = world.t.npc_types[npc.type]['stats'][NPC_HP]
    world.events.append((world.tick, 'npc_damage', dict(npc=npc.nid, type=npc.type, life=npc.life, source=source,
                                                        damage=damage, hp=npc.levels[NPC_HP], full=full)))
    npc.levels[NPC_HP] = max(0, npc.levels[NPC_HP] - damage)
    if npc.levels[NPC_HP] > 0:
        return
    world.events.append((world.tick, 'npc_defeated', dict(npc=npc.nid, type=npc.type, life=npc.life, source=source)))
    world.npc_queue(npc, 3, 0, 0)
    yield from ()


def imp_damage(world, npc, damage, source=None):
    """[ai_queue2,imp]: a 1 in 10 teleport when it survives the hit, then the default damage."""
    if world.random(10, npc.nid, 20) == 0 and npc.levels[NPC_HP] - damage > 0:
        npc_imp_teleport(world, npc)
    yield from npc_default_damage(world, npc, damage, source)


def npc_death(world, npc):
    """[proc,npc_death]: stop, wait for the arrival, clear the player's %lastcombat, a tick,
    then npc_del (respawn after respawnrate, scaled by the player count)."""
    npc.path = [(npc.x, npc.z)]  # npc_walk(npc_coord)
    world.npc_set_mode(npc, MODE_NONE)
    if npc.last_movement >= world.tick - 1:  # npc_arrivedelay
        yield ('npc_arrivedelay', world.tick + 1 if npc.last_movement == world.tick - 1 else world.tick + 2)
    p = world.player
    if p is not None and npc.vars['aggressive_player'] == p.name:
        p.varps['lastcombat'] = -1
    yield ('npc_delay', 1)
    world.remove_npc(npc)


def npc_death_script(world, npc, name):
    """An [ai_queue3,...] death script of the area, run from the extracted source (sim.rs2)."""
    yield from rs2.run(world.t.meta['content']['combat']['scripts'], name, DeathContext(world, npc))


def dark_wizard_water_attack(world, npc, p):
    """[proc,dark_wizard_water_attack]: confuse half the time while attack is not lowered,
    else water strike, every 4 ticks from approach distance."""
    if npc.vars['action_delay'] > world.tick:
        world.npc_set_mode(npc, MODE_APPLAYER2, p)
        return
    if not npc_check_notcombat(world, npc, p):
        world.npc_set_mode(npc, MODE_NULL)
        return
    if not npc_check_notcombat_self(world, npc, p):
        return
    ids = world.t.meta['content']['combat']['spell_ids']
    if p.base_levels[ATTACK] <= p.levels[ATTACK] and world.random(2, npc.nid, 30) == 0:
        npc_cast_spell(world, npc, p, ids['confuse'], 4)
    else:
        npc_cast_spell(world, npc, p, ids['water_strike'], 4)
    world.npc_set_mode(npc, MODE_APPLAYER2, p)
    yield from ()


def npc_cast_spell(world, npc, p, spell, speed):
    """[proc,npc_cast_spell_with_forced_max_hit] with the spell's own max hit."""
    if p.levels[HITPOINTS] == 0:
        return
    if npc.vars['action_delay'] > world.tick:
        world.npc_set_mode(npc, MODE_APPLAYER2, p)
        return
    if not npc_check_notcombat(world, npc, p):
        world.npc_set_mode(npc, MODE_NULL)
        return
    if not npc_check_notcombat_self(world, npc, p):
        return
    data = world.t.meta['content']['combat']['spells'][str(spell)]
    if data['freeze_time'] > 0:
        raise Unsupported('a freezing spell')
    for stat, _, _ in data['stat_change']:  # npc_debuff_allowed
        if p.levels[SKILLS.index(stat)] < p.base_levels[SKILLS.index(stat)]:
            return
    # npc_spell_cast
    p.close_modal()
    npc.vars['action_delay'] = world.tick + speed
    duration = 64
    if data['proj']:
        _, _, delay, _, length, _, step = data['proj']
        duration = delay + length + _distance_sw(npc.x, npc.z, p) * step
    hit = world.randominc(world.npc_magic_attack_roll(npc), npc.nid, 31) > world.randominc(p.varps['com_magicdef'], npc.nid, 32)
    uid = npc.uid
    if hit:
        # npc_spell_success
        p.close_modal()
        damage = world.randominc(data['maxhit'] if data['maxhit'] is not None else -1, npc.nid, 33)
        if data['maxhit'] is not None:
            p.enqueue('[queue,combat_damage_player]', lambda: combat_damage_player(world, p, damage, npc.nid),
                      delay=duration // 30)
        npc_poison_player(world, npc)
        p.enqueue('[queue,playerhit_n_retaliate]', lambda: playerhit_n_retaliate(world, p, uid),
                  delay=duration // 30)
        for i, (stat, constant, percent) in enumerate(data['stat_change']):  # npc_stat_change_effect
            if constant > 0 and percent > 0:
                raise Unsupported('a spell that raises a stat')
            if i == 0:
                world.mes(world.text('weakened'))
            world.stat_sub(p, SKILLS.index(stat), abs(constant), abs(percent))
    else:
        # npc_spell_fail
        p.close_modal()
        p.enqueue('[queue,playerhit_n_retaliate]', lambda: playerhit_n_retaliate(world, p, uid))
    npc_set_attack_vars(world, npc, p)


def highwayman_approach(world, npc, p):
    """[ai_applayer2,highwayman]: a shout, then melee."""
    world.npc_set_mode(npc, MODE_OPPLAYER2, p)
    yield from ()


def npc_imp_teleport(world, npc):
    """[proc,npc_imp_teleport]: map_findsquare within 20 tiles, then npc_setmode(null)."""
    for i, x in enumerate(range(npc.x - 20, npc.x + 21)):
        dz = int(world.rng(R_NPC_SCRIPT, world.tick, npc.nid, 1 + i) * 41) - 20
        if not (world.flags(npc.level, x, npc.z + dz, npc_bits=False) & WALK_BLOCKED) and (x, npc.z + dz) != (npc.x, npc.z):
            world.npc_teleport(npc, x, npc.z + dz)
            break
    world.npc_set_mode(npc, MODE_NULL)


# Native behavior a task's declared world may leave out (World `exclusions`), each because the slice does not model what
# follows it: `random_events`, the readiness roll every AFK_EVENTRATE ticks (a later skilling action would start a macro
# event); `clue_drops`, a death script's treasure-trail drop (picking one up starts the trail); `loot_drops`, the loot a death
# script adds that has a held or use script outside the slice (LOOT_OUTSIDE: drinking beer, which changes stats and leaves a
# glass; using anything on beer, bolts, a cabbage, goblin mail, a knife (fletching logs) or a spear; locating with a
# talisman; identifying a herb), found by exercising every held option, item-on-item use and pickup of the goblin and
# mugger loot (sim/tests/test_combat.py). An excluded drop is recorded ('excluded' event); its roll is made as natively;
# the random-event roll is simply not made.
EXCLUSIONS = ('random_events', 'clue_drops', 'loot_drops')
LOOT_OUTSIDE = ('beer', 'bolt', 'cabbage', 'goblin_armour', 'knife', 'bronze_spear', 'air_talisman')


def loot_outside(t, obj):
    """A loot object of LOOT_OUTSIDE, or an unidentified herb ([proc,randomherb])."""
    name = t.obj_types[obj]['debugname']
    return name in LOOT_OUTSIDE or name.startswith('unidentified_')


def trail_categories(t):
    """The category ids of treasure-trail objects (clues, caskets, challenge scrolls, puzzle boxes)."""
    return {v for k, v in t.meta['content']['categories'].items() if k.startswith('trail_')}


class DeathContext:
    """What an extracted death script (sim.rs2) can touch: its npc, the hero, ground items,
    the player's inventories and varps, and keyed random draws."""

    def __init__(self, world, npc):
        self.world, self.npc, self.draws = world, npc, 0
        self.active = None  # the player npc_findhero found

    def unsupported(self, what):
        raise Unsupported(f'death script of {self.world.t.npc_types[self.npc.type]["debugname"]}: {what}')

    def constant(self, name):
        consts = self.world.t.meta['content']['combat']['constants']
        if name not in consts:
            self.unsupported(f'constant ^{name}')
        return consts[name]

    def varp(self, name):
        p = self.player()
        return p.varps.get(name, 0)  # varps the slice never sets keep their default, 0

    def set_varp(self, name, value):
        self.player().varps[name] = value

    def player(self):
        if self.active is None:
            self.unsupported('a player command before npc_findhero')
        return self.active

    def obj(self, value):
        if isinstance(value, str):
            obj = self.world.t.obj_ids.get(value)
            if obj is None:
                self.unsupported(f'obj {value}')
            return obj
        return value

    def name(self, name):
        w, npc = self.world, self.npc
        if name == 'npc_findhero':
            hero = max(npc.heroes.items(), key=lambda kv: kv[1], default=(None, 0))[0]
            self.active = w.player if w.player is not None and hero == w.player.name else None
            return int(self.active is not None)
        if name == 'map_members':
            return int(w.members)
        if name in ('null',):
            return -1
        if name in ('true', 'false'):
            return int(name == 'true')
        if name in ('npc_coord', 'coord', 'uid'):
            return name
        if name == 'npc_type':
            return w.t.npc_types[npc.type]['debugname']
        if name == 'npc_name':
            return w.t.npc_types[npc.type]['name']
        if name == 'npc_category':
            cat = w.t.npc_types[npc.type]['category']
            return next((k for k, v in w.t.meta['content']['categories'].items() if v == cat), cat)
        return name  # an obj, inventory, enum, category or type name, read by the command

    def native(self, name):
        if name == 'npc_death':
            return lambda: npc_death(self.world, self.npc)
        return None

    def random(self, n, inclusive=False):
        self.draws += 1
        draw = self.world.rng(R_NPC_SCRIPT, self.world.tick, self.npc.nid, 1000 + self.draws)
        return int(draw * (n + 1 if inclusive else n))

    def command(self, name, args):
        yield from ()  # a generator, as rs2 runs commands that may suspend
        w, npc = self.world, self.npc
        if name == 'random':
            return self.random(args[0])
        if name == 'randominc':
            return self.random(args[0], inclusive=True)
        if name == 'npc_param':
            return w.npc_param(npc, args[0])
        if name == 'obj_add':
            coord, obj, count, duration = args
            if coord != 'npc_coord':
                self.unsupported('an obj added away from the npc')
            obj = self.obj(obj)
            if 'clue_drops' in w.exclusions and w.t.obj_types[obj]['category'] in trail_categories(w.t):
                w.events.append((w.tick, 'excluded', dict(what='clue_drop', obj=obj, npc=npc.nid)))
                return 0
            if 'loot_drops' in w.exclusions and loot_outside(w.t, obj):
                w.events.append((w.tick, 'excluded', dict(what='loot_drop', obj=obj, npc=npc.nid)))
                return 0
            w.obj_add(self.player(), npc.level, npc.x, npc.z, obj, count, duration)
        elif name == 'inv_total':
            inv, obj = args
            if inv not in (INV, WORN, BANK):
                self.unsupported(f'inventory {inv}')
            return self.player().inventory(inv).total(self.obj(obj))
        elif name == 'inv_totalcat':
            inv, category = args
            cat = w.t.meta['content']['categories'].get(category)
            items = self.player().inventory(inv).listed()
            return sum(c for _, o, c in items if w.t.obj_types[o]['category'] == cat) if cat is not None else 0
        elif name == 'if_close':
            self.player().close_modal()
        elif name == 'p_finduid':
            return int(args[0] == 'uid' and self.player().can_access())
        elif name == 'enum':
            _, _, enum, key = args
            table = w.t.meta['named_enums'].get(enum)
            if table is None:
                self.unsupported(f'enum {enum}')
            return dict(table['values']).get(key, table['defaultInt'])
        elif name == 'enum_getoutputcount':
            table = w.t.meta['named_enums'].get(args[0])
            if table is None:
                self.unsupported(f'enum {args[0]}')
            return len(table['values'])
        elif name == 'compare':  # StringOps javaStringCompare
            a, b = args
            return next((ord(x) - ord(y) for x, y in zip(a, b) if x != y), len(a) - len(b))
        elif name == 'clearbit_range':
            value, low, high = args
            return value & ~(((1 << (high - low + 1)) - 1) << low)
        else:
            self.unsupported(f'command {name}')
        return 0



# ---- held items: OPHELD and OPHELDU scripts, run as env-ts/engine/agent.ts opHeld and useOnHeld do -------
def inv_delslot(world, p, slot, why):
    """INV_DELSLOT on the inventory: the items leave the player's holdings (sim.audit 'remove')."""
    obj, count = p.inv.items[slot]
    p.inv.items[slot] = None
    world.events.append((world.tick, 'remove', dict(obj=obj, count=count, why=why)))
    return obj, count


LEVELREQUIRE = dict(levelrequire_attack=ATTACK, levelrequire_attack_caps=ATTACK, levelrequire_defence=DEFENCE,
                    levelrequire_defence_caps=DEFENCE, levelrequire_ranged=RANGED, levelrequire_magic=MAGIC)


def wield(world, p, name):
    """[opheld2,X]: its wield rule (levelrequire/scripts/tier*.rs2; @tutorial_island_equip, tutorial.rs2,
    jumps to @levelrequire_attack(1) once %tutorial is past ^newbie_combat_instructor_worn_inventory),
    then [proc,equip]. The refusal messages are not extracted: a refusal is not supported."""
    c = compact(world)
    label, args = c['wield'].get(name, ('equip', ['last_slot']))  # [opheld2,_] ~equip(last_slot)
    if label == 'tutorial_island_equip':
        if p.varps['tutorial'] <= c['tutorial_equip']:
            raise Unsupported('equipping before the tutorial allows it')
        label, args = 'levelrequire_attack', [1, 'last_slot']
    if label in LEVELREQUIRE:
        if p.base_levels[LEVELREQUIRE[label]] < args[0]:
            raise Unsupported(f'a {label} refusal')
    elif label != 'equip':
        raise Unsupported(f'wield rule {label}')
    equip(world, p, p.last_slot)
    yield from ()


def equip(world, p, slot):
    """[proc,equip] away from the duel arena, the desert mining camp and gnomeball pitches."""
    item = p.inv.items[slot]
    if item and world.t.obj_types[item[0]]['debugname'] in ('elemental_shield', 'ibanstaff'):
        raise Unsupported(f"equipping {world.t.obj_types[item[0]]['debugname']}")
    previous = p.worn.items[WEARPOS_RHAND][0] if p.worn.items[WEARPOS_RHAND] else None
    if not try_equip(world, p, slot):
        world.mes(world.text('equip_space'))
        return
    update_all(world, p, previous)


def wearpos(world, obj):
    t = world.t.obj_types[obj]
    return [w for w in (t['wearpos'], t['wearpos2'], t['wearpos3']) if w is not None and w >= 0]


def try_equip(world, p, slot):
    """[proc,try_equip]: a stackable joins its worn stack; otherwise, if the conflicting worn items fit
    in the inventory, inv_movetoslot swaps the obj into its wearpos, inv_movefromslot moves what came
    back to the first free inventory slot, and the remaining conflicts are unequipped."""
    item = p.inv.items[slot]
    if item is None or not wearpos(world, item[0]):
        return False
    obj, count = item
    pos = world.t.obj_types[obj]['wearpos']
    if world.t.obj_types[obj]['stackable'] and p.worn.total(obj) > 0:
        overflow = p.worn.itemspace2(world.t, obj, count)
        if overflow >= count:
            return False
        world.move_item(p, INV, WORN, obj, count - overflow)
        return True
    conflicts = 0
    for _, other, number in p.worn.listed():
        if not set(wearpos(world, obj)) & set(wearpos(world, other)):
            continue
        if world.t.obj_types[other]['stackable'] and p.inv.total(other) > 0:
            if p.inv.itemspace2(world.t, other, number):
                return False
        else:
            conflicts += 1
    if conflicts > p.inv.freespace() + 1:
        return False
    back = p.worn.items[pos]
    p.worn.items[pos], p.inv.items[slot] = [obj, count], (list(back) if back else None)
    world.events.append((world.tick, 'transfer', dict(src=INV, dst=WORN, obj=obj, to=obj, count=count)))
    if back:
        world.events.append((world.tick, 'transfer', dict(src=WORN, dst=INV, obj=back[0], to=back[0], count=back[1])))
        p.inv.items[slot] = None
        if p.inv.add(world.t, back[0], back[1]) < back[1]:
            raise Unsupported('an unequipped item dropped for want of space')
    for _, other, number in p.worn.listed():
        if other != obj and set(wearpos(world, obj)) & set(wearpos(world, other)):
            world.move_item(p, WORN, INV, other, number)
    return True


def update_all(world, p, previous):
    """[proc,update_all]: of what it updates (appearance, bonus texts, weight, the attack tab and an
    autocast reset) only ~player_combat_stat changes play here."""
    if any(world.t.obj_types[o]['debugname'] == 'ikov_bootsoflightness' for _, o, _ in p.worn.listed()):
        raise Unsupported('weight-reducing equipment')
    player_combat_stat(world, p)


def eat(world, p, name):
    """[opheld1,...] @player_consume_item(effect, eat_delay, skill_delay) (consumption/consume.rs2):
    no food again before %eat_delay passes, a skill action delayed by skill_delay, then the effect."""
    effect, eat_delay, skill_delay = compact(world)['eat_triggers'][name]
    if eat_delay is not None:
        if p.varps['eat_delay'] >= world.tick:
            return
        p.varps['eat_delay'] = world.tick + eat_delay
    if skill_delay is not None:
        p.varps['action_delay'] += skill_delay
    if effect != 'consume_effect_stat':
        raise Unsupported(f'food effect {effect}')
    consumable = p.last_item
    inv_delslot(world, p, p.last_slot, 'eat')
    if world.obj_param(consumable, 'next_obj_stage') != -1:
        raise Unsupported('food with a next stage')
    consume_effect_apply(world, p, consumable)
    yield from ()


def consume_effect_apply(world, p, consumable):
    """[proc,consume_effect_apply] and [proc,consume_effect_messages] for food that only heals."""
    c, t = compact(world), world.t.obj_types[consumable]
    hitpoints = p.levels[HITPOINTS]
    row = c['eat'].get(t['debugname'])
    if row is None:
        return
    if row['stat_change'] or row['healenergy']:
        raise Unsupported('food that changes stats or energy')
    for stat, constant, percent in row['stat_heal']:
        world.stat_heal(p, SKILLS.index(stat), constant, percent, cause='eat')
    if t['category'] == world.t.meta['content']['categories']['potion']:
        raise Unsupported('a potion')
    m = c['eat_messages'].get(t['debugname'], {})
    if 'message_delay' in m:
        raise Unsupported('a delayed food message')
    c1, c2, r1, r2 = (m.get(k, '') for k in ('consume_message1', 'consume_message2', 'restore_message1', 'restore_message2'))
    healed = hitpoints < p.levels[HITPOINTS]
    if r1 != '0' and healed and r1:
        world.mes(r1)
    elif c1 != '0':
        world.mes(c1 or world.text('eat', food=t['name'].lower()))
    if r2 != '0' and healed:
        world.mes(r2 or world.text('eat_heal'))
    if c2 != '0' and c2 and (not healed or r2 == '0'):
        world.mes(c2)


def bury_bones(world, p, name):
    """[opheld1,_bones] @bury_bones(last_slot) (skill_prayer/scripts/bury_bone.rs2)."""
    slot = p.last_slot
    world.mes(world.text('bury_dig'))
    p.stop_action()
    yield ('delay', 0)
    obj, _ = inv_delslot(world, p, slot, 'bury')
    world.add_xp(p, PRAYER, world.obj_param(obj, 'bone_exp'))
    world.mes(world.text('bury_done'))
    if p.afk_ready:
        p.afk_ready = False
        raise Unsupported('random event (afk_event) burying bones')


def drop_slot(world, p, name):
    """[opheld5,_] ~dropslot(last_slot) (player/scripts/drop.rs2) away from the duel arena, the trawler
    wreck and the games room: inv_dropslot for 1.5 lootdrop durations."""
    obj, count = inv_delslot(world, p, p.last_slot, 'drop')
    world.obj_add(p, p.level, p.x, p.z, obj, count, compact(world)['lootdrop_duration'] * 3 // 2)
    yield from ()


def use_default(cases, categories=(), oil=False):
    """An OPHELDU script whose cases (switch_obj(last_useitem)) need items the compact world lacks:
    the default branch, ~displaymessage(^dm_default) (weapon_poison.rs2 [opheldu,_weapon_stab],
    skill_fletching/scripts/darts.rs2 [opheldu,feather], cut_logs.rs2 [opheldu,_category_22])."""
    def script(world, p, name):
        used = world.t.obj_types[p.last_useitem]
        cats = world.t.meta['content']['categories']
        if used['debugname'] in cases or any(used['category'] == cats[c] for c in categories) \
                or (oil and world.t.obj_types[p.last_item]['category'] == cats['category_111']):
            raise Unsupported(f"{name} with {used['debugname']}")
        world.mes(world.text('dm_default'))
        yield from ()
    return script


HELD = {'[opheld1,_bones]': bury_bones, '[opheld5,_]': drop_slot,
        '[opheldu,_weapon_stab]': use_default(('weapon_poison', 'tbwt_cleaning_cloth')),
        '[opheldu,feather]': use_default(('ogre_arrow_shaft', 'arrow_shaft', 'thprotodarttip'), ('category_969',)),
        '[opheldu,_category_22]': use_default(('knife', 'tinderbox'), oil=True)}


def held_script(world, name):
    """The port of an OPHELD/OPHELDU script, or None."""
    c = compact(world)
    if name in HELD:
        return HELD[name]
    if name.startswith('[opheld2,') and (name in c['wield'] or name == '[opheld2,_]'):
        return wield
    return eat if name in c['eat_triggers'] else archive_script(world, name)


# ---- firemaking and cooking (skill_firemaking/scripts/firemaking.rs2, skill_cooking/scripts/cooking.rs2) ----------
# The phase 2 simulator port. Both read rows that only draynor-v3.npz holds (sim/build.py firemaking, cooking), so
# their triggers name a port only over such an archive (ARCHIVE_SCRIPTS); over an older one they name none, as before.
def in_zones(zones, level, x, z):
    """~inzone_coord_pair_table (general/scripts/misc/coord_procs.rs2:49-61) over INZONE's boxes (ServerOps.ts:48-64)."""
    return any(x0 <= x <= x1 and z0 <= z <= z1 and l0 <= level <= l1 for x0, z0, l0, x1, z1, l1 in zones)


def tinderbox_use(world, p, name):
    """[opheldu,tinderbox] (firemaking.rs2:13-28), which a tinderbox and another item run in either order
    (OpHeldUHandler.ts:93-101): logs light; bark and any other item show the default message. Its other branches
    (tutorial logs, unlit arrows, jogre bones, the firecracker, light sources) are declared exclusions."""
    used = world.t.obj_types[p.last_useitem]
    cats = world.t.meta['content']['categories']
    if used['debugname'] in ('newbielogs', 'tbwt_jogre_bones', 'viking_firecracker') \
            or world.obj_param(p.last_useitem, 'unlit_arrow') == 1 or used['category'] == cats['light_source_unlit']:
        raise Unsupported(f"a tinderbox used with {used['debugname']}")
    if used['category'] == cats['category_22'] and used['debugname'] != 'hollow_bark':
        yield from light_logs_inv(world, p, p.last_useslot)
    else:
        world.mes(world.text('dm_default'))


def light_refused(world, p, log, level, x, z):
    """The checks both light scripts start with (firemaking.rs2:37-51, 81-96), on the tile the logs lie on or will lie
    on: the log's Firemaking level, [proc,area_allow_loc_add] (134-144) and a tinderbox carried. area_allow_loc_add
    refuses a tile under a loc of an active type (MAP_LOCADDUNSAFE), one that blocks a walk (MAP_BLOCKED) and a player
    inside a bank zone: the zone check reads the player's coord, not the tile's (line 141). MAP_BLOCKED and LINEOFWALK
    also refuse every tile outside the free area in a free world (ServerOps.ts:77, 133), which the archive does not
    name."""
    if not world.members:
        raise Unsupported('lighting a fire in a free-to-play world')
    need = world.obj_param(log, 'levelrequire')
    if p.levels[FIREMAKING] < need:
        world.mes(world.text('fm_level', level=need, log=world.t.obj_types[log]['name']))
    elif world.map_locaddunsafe(level, x, z) or world.flags(level, x, z, npc_bits=False) & WALK_BLOCKED \
            or in_zones(world.t.meta['content']['firemaking']['bank_zones'], p.level, p.x, p.z):
        world.mes(world.text('fm_here'))
    elif p.inv.total(world.t.obj_id('tinderbox')) < 1:
        world.mes(world.text('fm_tinderbox'))
    else:
        return False
    return True


def light_logs_inv(world, p, slot):
    """[label,light_logs_inv] (firemaking.rs2:79-117): after the checks the logs drop on the player's tile for
    lootdrop_duration (INV_DROPSLOT, InvOps.ts:213-262; no log type stacks), and the attempt runs on them."""
    log = p.inv.items[slot][0]
    if light_refused(world, p, log, p.level, p.x, p.z):
        return
    inv_delslot(world, p, slot, 'light')
    obj = Obj(log, 1, p.x, p.z, static=False, ticks=compact(world)['lootdrop_duration'], receiver=p.name, level=p.level)
    world.objs.append(obj)
    yield from light_attempt(world, p, obj)


def light_logs_ground(world, p, obj):
    """[opobj4,_category_22] @light_logs_ground (firemaking.rs2:2-3, 30-77): light_logs_inv's checks on the logs' tile,
    then the same attempt. Only light_logs_inv's p_opobj(4) reaches it: the interface offers no Light on a ground item."""
    if world.t.obj_types[obj.type]['debugname'] == 'hollow_bark':
        world.mes(world.text('dm_default'))
    elif not light_refused(world, p, obj.type, obj.level, obj.x, obj.z):
        yield from light_attempt(world, p, obj)


def light_attempt(world, p, obj):
    """What both light scripts end with (firemaking.rs2:52-77, 98-117), and op 4 on the ground logs runs again each
    cycle (P_OPOBJ): past %action_delay, the attempt's message and a first roll 3 ticks on; on %action_delay, a roll,
    stat_random(firemaking, 64, 512), and after a failure the next one 4 ticks on. A success also sets %action_delay
    4 ticks on, so a light begun within them shows no message and rolls on that tick."""
    if p.varps['action_delay'] < world.tick:
        p.stop_action()
        world.mes(world.text('fm_attempt'))
        p.varps['action_delay'] = world.tick + 3
    elif p.varps['action_delay'] == world.tick:
        p.varps['action_delay'] = world.tick + 4
        if world.decide('light', world.stat_random(p, FIREMAKING, 64, 512, 40), obj.type):
            yield from push_player(world, p, obj)
            yield from firemaking_success(world, p, obj)
            return
    p_opobj(world, p, obj, 4)


def push_player(world, p, obj):
    """[proc,push_player] (firemaking.rs2:146-175): a player still on the logs' tile moves to the first of west, east,
    south and north with a line of walk (p_teleport) and waits a tick (p_delay(0)), so the fire comes a tick after the
    roll; with no such tile it stays and the fire comes at once. For a step of one tile LINEOFWALK reads what canTravel
    reads: the next tile's wall toward the player and WALK_BLOCKED (LineValidator.ts:27-47, Line.ts:9-12). The duel
    arena's case (no move, a tick's wait) is a declared exclusion."""
    if in_zones(world.t.meta['content']['firemaking']['duel_zones'], obj.level, obj.x, obj.z):
        raise Unsupported('a fire lit in the duel arena')
    if (p.x, p.z) != (obj.x, obj.z):
        return
    for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if world.can_travel(p.level, p.x, p.z, dx, dz, 1, 0, 0):
            world.teleport(p, p.x + dx, p.z + dz)
            yield ('delay', 0)
            return


def firemaking_success(world, p, obj):
    """[proc,firemaking_success] (firemaking.rs2:119-132): the logs go, the xp, a Fire (loc_2732) on their tile for
    100 + random(100) ticks, then world_delay(that count - 2) and ashes for 100 ticks, with no receiver (obj_addall).
    The fire's count runs out in a zone phase, and the world queue adds the ashes in the first phase of the next cycle."""
    world.remove_obj(obj)
    world.add_xp(p, FIREMAKING, world.obj_param(obj.type, 'productexp'))
    world.mes(world.text('fm_catch'))
    life = world.decide('burn', 100 + world.random(100, 41))
    fire = next(k for k, t in world.t.loc_types.items() if t['debugname'] == 'loc_2732')
    world.loc_add(obj.level, obj.x, obj.z, fire, 1, CENTREPIECE_STRAIGHT, life)
    yield ('world_delay', life - 2)
    world.obj_add(None, obj.level, obj.x, obj.z, world.t.obj_id('ashes'), 1, 100)


def cook_on_fire(world, p, loc):
    """[oplocu,_cooking_fire] (cooking.rs2:2-15): attempt_cook_item, after its own cases, each a declared exclusion:
    the Fremennik's fires, an unlit arrow, and Tutorial Island's script."""
    if world.t.loc_types[loc.type]['debugname'] in ('viking_house_fire', 'viking_longhall_fire'):
        raise Unsupported('a fire of the Fremennik')
    if world.obj_param(p.last_useitem, 'unlit_arrow') == 1:
        raise Unsupported('an unlit arrow on a fire')
    if in_zones(world.t.meta['content']['cooking']['tutorial_zones'], p.level, p.x, p.z):
        raise Unsupported('cooking on Tutorial Island')
    yield from attempt_cook_item(world, p, loc)


def attempt_cook_item(world, p, loc):
    """[oplocu,_cooking_oven] and a fire's last case -> [label,attempt_cook_item] (cooking.rs2:1, 17-154) -> [label,cook_item]
    (158-195), for the extracted rows (sim/build.py cooking): p_arrivedelay; an item the table does not name gets the
    mesbox; the roll, stat_random(cooking, low, high) with the row's pair, is made now; the cook message of an oven or
    a fire; p_delay(1), which resumes 2 ticks on; then one raw item goes (inv_del: the first held) and the cooked or
    the burnt one comes, with xp only for the cooked. Declared exclusions: a random event (afk_event), the jogre bones
    and karambwan the script treats apart, and a raw item whose row is not extracted. No extracted row asks for a
    Cooking level above 1, so the level refusal is not ported either."""
    if p.last_movement >= world.tick:  # p_arrivedelay
        yield ('delay', 0)
    cooking, raw = world.t.meta['content']['cooking'], p.last_useitem
    name = world.t.obj_types[raw]['debugname']
    row = cooking['rows'].get(name)
    if name in cooking['special'] or (row is None and name in cooking['uncooked']):
        raise Unsupported(f'cooking {name}')
    if row is None:
        yield from mesbox(world, p, world.text('cook_cant'))
        return
    if p.levels[COOKING] < row['level']:
        raise Unsupported(f'a Cooking level too low for {name}')
    if p.afk_ready:
        p.afk_ready = False
        raise Unsupported('random event (afk_event) during cooking')
    passes = world.decide('cook', world.stat_random(p, COOKING, *row['chance'], 42), raw)
    oven = world.t.loc_types[loc.type]['category'] == world.t.meta['content']['categories']['cooking_oven']
    world.mes(row['range' if oven else 'fire'])
    yield ('delay', 1)
    world.events.append((world.tick, 'remove', dict(obj=raw, count=p.inv.remove(raw, 1), why='cook')))
    world.produce(p, world.t.obj_id(row['cooked' if passes else 'burnt']), '[label,cook_item]')
    world.mes(row['success' if passes else 'burn'])
    if passes:
        world.add_xp(p, COOKING, row['exp'])


ARCHIVE_SCRIPTS = dict(firemaking={'[opheldu,tinderbox]': tinderbox_use, '[opobj4,_category_22]': light_logs_ground},
                       cooking={'[oplocu,_cooking_oven]': attempt_cook_item, '[oplocu,_cooking_fire]': cook_on_fire})


def archive_script(world, name):
    """The port of a firemaking or cooking trigger over an archive that holds its rows, else None."""
    content = world.t.meta.get('content', {})
    return next((scripts[name] for key, scripts in ARCHIVE_SCRIPTS.items() if key in content and name in scripts), None)

# ---- ground items (player/scripts/pickup.rs2) -----------------------------------------------------
def pickup_obj(world, p, obj):
    """[opobj3,_] -> [label,pickup_obj]: from the floor at once, from a table a tick later,
    if the inventory can hold all of it."""
    if p.inv.itemspace2(world.t, obj.type, obj.count):  # ~pickup_obj_check_for_space
        world.mes(world.text('pickup_no_space'))
        return
    if (p.x, p.z) != (obj.x, obj.z):
        yield ('delay', 0)
    world.take_obj(p, obj)


# ---- dialogs (interface_chat/scripts/chat.rs2) -------------------------------------------------
def openchat(world, p, interface):
    p.open_chat(interface)


def mesbox(world, p, text):
    """[proc,mesbox]: one page (the slice's texts fit), message1:com_0."""
    _one_page(text)
    world.settext(('message1', 0), text)
    openchat(world, p, 'message1')
    yield ('pausebutton',)


def objbox(world, p, text, interface='objbox1'):
    _one_page(text)
    world.settext((interface, 1), text)
    openchat(world, p, interface)
    yield ('pausebutton',)


def doubleobjbox(world, p, text):
    lines = text.split('|')
    for i, line in enumerate(lines):
        world.settext(('doubleobjbox', 2 + i), line)
    openchat(world, p, 'doubleobjbox')
    yield ('pausebutton',)


def _one_page(text):
    if len(text) > 200:
        raise Unsupported('a dialog text that may need several pages')


def chatnpc(world, p, npc, text):
    """[proc,chatnpcrange] for one page of up to 4 lines: npcchatN, name then lines."""
    _one_page(text)
    name = world.t.npc_types[npc.type]['name']
    world.settext(('npcchat', 1), name)
    world.settext(('npcchat', 2), _strip_anim(text))
    openchat(world, p, 'npcchat')
    if npc.mode not in (MODE_OPPLAYER2, MODE_APPLAYER2):
        world.npc_set_mode(npc, MODE_PLAYERFACE, p)
    yield ('pausebutton',)


def chatplayer(world, p, text):
    _one_page(text)
    world.settext(('chat', 1), p.display)
    world.settext(('chat', 2), _strip_anim(text))
    openchat(world, p, 'chat')
    yield ('pausebutton',)


def p_choice2(world, p, first, second):
    world.settext(('multi2', 0), world.text('select_option'))
    world.settext(('multi2', 1), first)
    world.settext(('multi2', 2), second)
    openchat(world, p, 'multi2')
    p.resume_buttons = [('multi2', 1), ('multi2', 2)]
    yield ('pausebutton',)
    return 1 if p.last_com == ('multi2', 1) else 2


def _strip_anim(text):
    return re.sub(r'^<p,[a-z_]+>', '', text)


def talk_to_banker(world, p, npc):
    """[label,talk_to_banker] (interface_bank/scripts/banker.rs2)."""
    d = world.t.meta['content']['banker']
    yield from chatnpc(world, p, npc, d['greet'])
    option = yield from p_choice2(world, p, d['choice_bank'], d['choice_what'])
    if option == 1:
        yield from chatplayer(world, p, d['choice_bank'])
        world.open_bank(p)
        return
    yield from chatplayer(world, p, d['choice_what'])
    yield from chatnpc(world, p, npc, d['branch'])
    option = yield from p_choice2(world, p, d['choice_do'], d['choice_varrock'])
    if option == 1:
        yield from chatplayer(world, p, d['choice_do'])
        yield from chatnpc(world, p, npc, d['answer_do'])
    else:
        yield from chatplayer(world, p, d['choice_varrock'])
        yield from chatnpc(world, p, npc, d['answer_varrock'])


def bank_teller_near(world, loc):
    """npc_findcat(loc_coord, bank_teller, 2, 0): a bank teller within 2 tiles of the booth."""
    cat = world.t.meta['content'].get('bank_teller_category')
    return next((n for n in world.npcs if n.active and world.t.npc_types[n.type]['category'] == cat
                 and n.level == loc.level and max(abs(n.x - loc.x), abs(n.z - loc.z)) <= 2), None)


def booth_use(world, p, loc):
    """[oploc1,bankbooth]: talk to a nearby banker, else open the bank."""
    banker = bank_teller_near(world, loc)
    if banker is not None:
        yield from talk_to_banker(world, p, banker)
        return
    world.open_bank(p)


def booth_quick(world, p, loc):
    world.open_bank(p)
    yield from ()


def teller_talk(world, p, npc):
    """[apnpc1,_bank_teller]: within 2 tiles, else p_aprange(2)."""
    if _distance_to(p, npc) > 2:
        p.ap_range, p.ap_range_called = 2, True
        return
    yield from talk_to_banker(world, p, npc)


def teller_bank(world, p, npc):
    if _distance_to(p, npc) > 2:
        p.ap_range, p.ap_range_called = 2, True
        return
    world.open_bank(p)
    yield from ()


# ---- level up (levelup/scripts/levelup.rs2) -------------------------------------------------------
def levelup(world, p, stat):
    rows = world.t.meta['content']['levelup']
    row = rows.get(SKILLS[stat])
    if row is None:
        return
    level = p.base_levels[stat]
    world.mes(row['message'])
    world.settext((row['interface'], 1), '@dbl@' + row['message'])
    world.settext((row['interface'], 2), f"{row['level_prefix']} {level}.")
    openchat(world, p, row['interface'])
    yield ('pausebutton',)
    if level not in row['unlocks']:
        return
    unlock = row['unlock_texts'].get(str(level))
    if unlock is None:
        return
    kind, text = unlock
    if kind == 'objbox':
        yield from objbox(world, p, text)
    elif kind == 'doubleobjbox':
        yield from doubleobjbox(world, p, text)
    else:
        yield from mesbox(world, p, text)


SCRIPTS = {
    '[oploc1,_tree]': attempt_cut_tree, '[oploc3,_tree]': cut_tree, '[oplocu,_tree]': use_on_tree,
    '[opnpc1,_saltfish]': attempt_fish_shrimp, '[opnpc4,_saltfish]': fish_shrimp,
    '[opnpc3,_saltfish]': attempt_fish_sardine, '[opnpcu,_saltfish]': use_on_saltfish,
    '[oploc1,bankbooth]': booth_use, '[oploc2,bankbooth]': booth_quick,
    '[apnpc1,_bank_teller]': teller_talk, '[apnpc3,_bank_teller]': teller_bank,
    '[apnpc2,_]': player_combat_start_ap, '[opnpc2,_]': player_combat_start,
    '[opobj3,_]': pickup_obj,
    '[oploc1,_door_closed]': door(DOOR_OPEN, 1), '[oploc1,_door_opened]': door(DOOR_CLOSE, 3),
    '[oploc1,_gate_main_closed]': gate(True, False), '[oploc1,_gate_outer_closed]': gate(True, True),
    '[oploc1,_gate_main_open]': gate(False, False), '[oploc1,_gate_outer_open]': gate(False, True),
    '[oploc1,loc_1722]': stairs('[oploc1,loc_1722]'), '[oploc1,loc_1723]': stairs('[oploc1,loc_1723]'),
    '[oploc1,_mining_rock_normal]': mining(get_ore(False), True), '[oploc3,_mining_rock_normal]': mining(get_ore(False), False),
    '[oploc1,_mining_rock_fast]': mining(get_ore(True), True), '[oploc3,_mining_rock_fast]': mining(get_ore(True), False),
    '[oploc2,_mining_rock_normal]': prospect, '[oploc2,_mining_rock_fast]': prospect,
}
NPC_SCRIPTS = {'[ai_timer,_saltfish]': move_fishing_spot, '[ai_timer,imp]': imp_timer,
               '[ai_timer,_]': npc_poison_timer, '[ai_timer,sheepunsheered]': flavour_timer(20, 20),
               '[ai_timer,cow]': flavour_timer(15, 20), '[ai_timer,_duck]': flavour_timer(50, 50)}

NPC_QUEUES = {'[ai_queue1,_]': npc_default_retaliate(False), '[ai_queue1,young_dark_wizard]': npc_default_retaliate(True),
              '[ai_queue1,highwayman]': npc_default_retaliate(True), '[ai_queue1,wizard]': npc_default_retaliate(True),
              '[ai_queue2,_]': npc_default_damage, '[ai_queue2,imp]': imp_damage}
NPC_AI = {'[ai_opplayer2,_]': npc_default_attack, '[ai_applayer2,young_dark_wizard]': dark_wizard_water_attack,
          '[ai_opplayer2,young_dark_wizard]': dark_wizard_water_attack, '[ai_applayer2,highwayman]': highwayman_approach}

# each evaluation and training header names the engine's auto-retaliate. retaliate-v1 queued
# it on the weak queue, which combat_damage_player cleared before it ran; results of the two are never pooled.
RETALIATE_CONTRACT = 'retaliate-v2'
