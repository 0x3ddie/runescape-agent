"""The native port's equivalence harness: `python -m sim.native diff`.

A run builds two sides with CpuBatch's interface (sim/vec.py) on the same pools, task, goals, n, first and key and
steps them in lockstep under one action source. After every decision it compares, world by world: the 34 observation
arrays bitwise (float32 as uint32; the mask is one of them), the reward bits, done, every info field (made, kills and
facts included), the unsupported flag and its evidence, and each new episode's start. Each side also gives one Row
per world cycle (a Python side from its hooks, a native one from its worlds' traces, sim/native/canon.c), taken where
rs.h takes the tick hash: after the watched cycle (task.py _watch) and before capture_text. A Row holds the canonical
state record (Canon), its zlib.crc32, the world's draws since the previous Row (the world's Rng wrapped: each key tuple
and its 24-bit m), and the events, chat lines and component texts appended since then. Rows are compared before the
decision's outputs, so a divergence is placed at its first tick; equal Rows with differing outputs place it in the
agent or observation layer. The first divergence stops the run. Its report gives the world slot, key, episode,
decision and tick, the world's last 20 actions, the field diff of the two records, the last draws, events and chat of
both sides, and a one-line repro: the same diff on CpuBatch(n=1, first=slot) from that episode, replaying its logged
actions. An unsupported world ends the run once both sides flag it alike (CpuBatch refuses to step it again): the
equivalence target there is the flag and its evidence, not the partial state.

The oracle is never edited. For the duration of a batch call the Python side hooks two calls: GoalEpisode.__init__,
which binds the new episode to its world slot's trace and wraps its world's Rng, and Agent.capture_text, the call that
follows world.cycle() in wait_ticks (agent.py:240-244), where the Row is taken. (Wrapping world.cycle per instance as
vec.trace does would hash before the terminal check once the episode arms: task.py _watch wraps whatever cycle it
finds.) Scenario starts override the start draws at their CpuBatch call sites (vec.draw_start, vec.draw_combat,
search.draw_supply) through the draws' own parameters (layout, group, variant) and edits of the drawn record (levels,
hp, food, worn), the record the native side receives through rs_episode_begin (NativeSide). A native side's Rows come
from its worlds' traces (sim/native/canon.c).

Private data stays private: records, rows and reports hold string ids (a pack's, or crc32 without one), never text.

    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native diff --archive logs/sim/data/draynor-nc1.npz \\
        --pool logs/sim/world/train-8-nc1-derived.npz --worlds 4 --steps 200 --source random
"""
from __future__ import annotations

import ast
from collections import OrderedDict, deque
import contextlib
import copy
from dataclasses import dataclass, field
from functools import lru_cache
import inspect
import json
from pathlib import Path
import re
import shlex
import struct
import sys
import weakref
import zlib

import numpy as np

from sim import native

R_ACTION, R_BIASED = 9, 10  # Rng purposes of the keyed action sources (9: sim/tests/test_workers.py run)
HISTORY, TAIL = 20, 8       # actions in a report; Rows kept per world for its last draws, events and chat
# Biased keyed play weights these action families (agent.FAMILIES) to reach content; the others weigh 1.
BIAS = dict(act=4.0, held=3.0, use=2.0, dialog=3.0)
# Play under attack (the gate's danger cases): the auto-retaliate's body runs only while an attacked player is otherwise
# idle. The other families keep their weight of 1, so a rare other request stays possible.
UNDER_ATTACK = dict(idle=dict(wait=1e6, dialog=1e6), walk=dict(click=1e6, dialog=1e6), eat=dict(held=100.0, wait=100.0, dialog=1e6))
# Named scenario starts: draw parameters and edits of the drawn record (Scenario).
SCENARIOS = dict(
    danger=dict(group='danger'),                  # a combat slot with a mugger
    hurt=dict(group='hurt_fed', hp=2),            # low hitpoints with food
    unfed=dict(group='hurt_unfed'),
    worn=dict(worn=True),                         # the weapon starts worn (the Wield setup step)
    scarce=dict(variant='scarce'), ample=dict(variant='ample'),
    under=dict(levels=dict(woodcutting=14, fishing=14, mining=14)),     # one under the oak, anchovy, iron unlocks
    unlocked=dict(levels=dict(woodcutting=15, fishing=15, mining=15)),
)
# free-play-v1's scenario starts (Scenario.village: its draw is draw_village's, its group VILLAGE's)
FREE_SCENARIOS = dict(
    freehurt=dict(hp=2),                          # a life at 2 hitpoints: deaths come early
    freeworn=dict(worn=True, levels=dict(woodcutting=15, fishing=15, mining=15)),
)


class CanonError(RuntimeError):
    """Oracle state the canonical record cannot express: a gap of the rs.h contract, reported, never dropped."""


# ---- rs.h: the enums and orders the record uses (the module exports field and script ids, not these) -------------
@lru_cache(maxsize=None)
def header():
    """({enum constant: value} of rs.h's literal enums, RS_VARPS names in order)."""
    text = (native.SOURCE / 'rs.h').read_text()
    code = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    enums = {}
    for body in re.findall(r'\benum\s*\{([^{}]*)\}', code):
        if '(' in body:  # an X-macro enum (the module exports those)
            continue
        names, value = {}, -1
        try:
            for item in filter(None, (s.strip() for s in body.split(','))):
                name, _, expr = (s.strip() for s in item.partition('='))
                value = int(expr, 0) if expr else value + 1
                names[name] = value
        except ValueError:  # an expression (bit flags): not a record enum
            continue
        enums.update(names)
    varps = re.search(r'#define RS_VARPS\(X\)((?:[^\n]*\\\n)*[^\n]*)', text)
    return enums, tuple(re.findall(r'X\((\w+)\)', varps.group(1)))


@lru_cache(maxsize=None)
def contract():
    """The compiled module's canonical field names and script rows (rs.h RS_CANON_FIELDS, RS_SCRIPTS)."""
    m = native.load()
    scripts = m.scripts()
    by_qualname = {}
    for k, (_, qualname, _) in enumerate(scripts):
        if qualname:
            by_qualname.setdefault(qualname, []).append(k)
    return (tuple(m.canon_fields()), {name: k for k, (name, _, _) in enumerate(scripts)}, by_qualname,
            [status for _, _, status in scripts], m.constants()['RS_FRAME_DEPTH'])


def crc_strings(text):
    """A string's id without a pack: its crc32 as a positive int32 (equal on two Python sides)."""
    return zlib.crc32(text.encode()) & 0x7fffffff


class PackStrings:
    """String ids of a pack (rs.h STRINGS): the id of each string the pack holds; another string is a CanonError."""

    def __init__(self, pack):
        self.ids = {pack.string(k): k for k in range(pack.n_strings)}

    def __call__(self, text):
        found = self.ids.get(text)
        if found is None:  # never the text itself: game strings are private
            raise CanonError(f'a string of {len(text)} characters (crc32 {zlib.crc32(text.encode()):08x}) is not in '
                             'the pack')
        return found


# ---- script frames (rs.h SCRIPT FRAMES) ------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _module_ast(filename):
    return ast.parse(Path(filename).read_text())


_SITES, _POSITIONS = {}, {}


def _sites(code):
    """A generator function's suspension sites in source order: its Yield and YieldFrom nodes (not those of nested
    functions or lambdas), except `yield from ()`, which never suspends, as (lineno, end_lineno, col, end_col)."""
    sites = _SITES.get(code)
    if sites is None:
        fn = next(node for node in ast.walk(_module_ast(code.co_filename))
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == code.co_name
                  and node.lineno == code.co_firstlineno)
        found = []

        def visit(node):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                never = isinstance(child, ast.YieldFrom) and isinstance(child.value, ast.Tuple) and not child.value.elts
                if isinstance(child, (ast.Yield, ast.YieldFrom)) and not never:
                    found.append((child.lineno, child.end_lineno, child.col_offset, child.end_col_offset))
                visit(child)
        visit(fn)
        sites = _SITES[code] = sorted(found, key=lambda s: (s[0], s[2]))
    return sites


def pc_of(gen):
    """rs.h's pc of a generator: 0 not started (or finished), k >= 1 suspended at its k-th suspension site. The
    suspended instruction's source position (CPython 3.11+ co_positions) is the Yield or YieldFrom node's span."""
    if gen.gi_frame is None or inspect.getgeneratorstate(gen) == inspect.GEN_CREATED:
        return 0
    code = gen.gi_code
    positions = _POSITIONS.get(code)
    if positions is None:
        positions = _POSITIONS[code] = list(code.co_positions())
    position = positions[gen.gi_frame.f_lasti // 2]
    try:
        return _sites(code).index(position[:4]) + 1
    except ValueError:
        raise CanonError(f'{code.co_qualname} is suspended off its suspension sites (line {position[0]})') from None


def _cells(fn):
    """A function's closure: {free variable: value}."""
    return {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ())}


# rs.h names a closure by its bound arguments (RS_SCRIPTS): the script of each closure qualname, from the free
# variables of its (started) frame. The oracle's closures: engine.py mining, get_ore, gate, use_default,
# npc_default_retaliate, door. The others sharing a qualname (stairs, flavour_timer, npc_death_script's unported
# roots) are RS_NP or decided by the exporter: the first ported id stands for them.
CLOSURES = {
    'mining.<locals>.script': lambda v: 'RS_S_MINE_{}_{}'.format('FIRST' if v['first'] else 'CONT',
                                                               'FAST' if _cells(v['get_ore'])['fast'] else 'NORMAL'),
    'get_ore.<locals>.script': lambda v: 'RS_S_GET_ORE_FAST' if v['fast'] else 'RS_S_GET_ORE_NORMAL',
    'gate.<locals>.script': lambda v: 'RS_S_GATE_{}_{}'.format('OPEN' if v['opening'] else 'CLOSE',
                                                             'OUTER' if v['outer'] else 'MAIN'),
    'use_default.<locals>.script': lambda v: {'weapon_poison': 'RS_S_USE_DEFAULT_WEAPON_STAB',
                                              'ogre_arrow_shaft': 'RS_S_USE_DEFAULT_FEATHER',
                                              'knife': 'RS_S_USE_DEFAULT_CATEGORY_22',
                                              # engine.FIELD_USE's (the pick archive's), by their first case
                                              'half_baked_batta': 'RS_S_USE_DEFAULT_CABBAGE',
                                              'bowl_water': 'RS_S_USE_DEFAULT_POTATO',
                                              'bowl_empty': 'RS_S_USE_DEFAULT_ONION',
                                              'reddye': 'RS_S_USE_DEFAULT_BLACK_CAPE'}[v['cases'][0]],
    'npc_default_retaliate.<locals>.script': lambda v: 'RS_S_NPC_DEFAULT_RETALIATE_AP' if v['ap'] else
    'RS_S_NPC_DEFAULT_RETALIATE',
    'door.<locals>.script': lambda v: 'RS_S_DOOR_OPEN' if v['turn'] == 1 else 'RS_S_DOOR_CLOSE',
}


def script_id(gen):
    """The RsScript id of a generator's function."""
    _, ids, by_qualname, status, _ = contract()
    qualname = gen.gi_code.co_qualname
    found = by_qualname.get(qualname)
    if not found:
        raise CanonError(f'script frame {qualname} has no rs.h script id')
    if len(found) == 1:
        return found[0]
    resolve = CLOSURES.get(qualname)
    if resolve is not None and gen.gi_frame is not None and inspect.getgeneratorstate(gen) != inspect.GEN_CREATED:
        return ids[resolve(gen.gi_frame.f_locals)]
    port, sub = header()[0]['RS_PORT'], header()[0]['RS_SUB']
    return next((k for k in found if status[k] in (port, sub)), found[0])


def frame_chain(gen):
    """[(script id, pc)] of a suspended script, outermost first, one level per `yield from` (rs.h Frame). The rs2
    interpreter's generators between a death script and npc_death (sim/rs2.py run, _block, _expr, _call and
    DeathContext.command) have no C counterpart and are dropped."""
    from sim import rs2
    interpreter = Path(rs2.__file__).resolve()
    out = []
    while gen is not None:
        code = gen.gi_code
        if Path(code.co_filename).resolve() != interpreter and code.co_qualname != 'DeathContext.command':
            out.append((script_id(gen), pc_of(gen)))
        inner = gen.gi_yieldfrom
        gen = inner if inspect.isgenerator(inner) else None
    if len(out) > contract()[4]:
        raise CanonError(f'a script chain {len(out)} deep (RS_FRAME_DEPTH {contract()[4]})')
    return out


# Request names (engine.py enqueue sites: 1475, 1665, 2363, 2507, 2589, 2591, 2718, 2721, 2732) -> (script, the
# closure variables of the factory that are its bound arguments, rs.h Req.arg order).
REQUESTS = {'[changestat,_]': ('RS_S_CHANGESTAT', ()), '[queue,reorganize_bank]': ('RS_S_REORGANIZE_BANK', ()),
            '[queue,player_death]': ('RS_S_PLAYER_DEATH', ()),
            '[queue,combat_damage_player]': ('RS_S_COMBAT_DAMAGE_PLAYER', ('damage', 'npc')),
            '[queue,playerhit_n_retaliate]': ('RS_S_PLAYERHIT_N_RETALIATE', ('uid',))}


def request_row(request):
    """[script, delay, args[3]] of a queued Request (rs.h Req: the factory's bound arguments, an npc as its nid)."""
    ids = contract()[1]
    if request.name.startswith('[advancestat,'):  # lambda s=stat: levelup(self, p, s) (engine.py:1475)
        return [ids['RS_S_LEVELUP'], request.delay, int(request.script.__defaults__[0]), 0, 0]
    if request.name not in REQUESTS:
        raise CanonError(f'request {request.name} has no rs.h script')
    script, names = REQUESTS[request.name]
    cells = _cells(request.script)
    args = [getattr(cells[n], 'nid', cells[n]) for n in names]
    return [ids[script], request.delay, *map(int, args), *[0] * (3 - len(args))]


# ---- the canonical record ----------------------------------------------------------------------------------------
NPC_FIELDS = ('nid', 'type', 'base', 'x', 'z', 'level', 'active', 'delayed', 'delayed_until', 'mode', 'target',
              'timer_interval', 'timer_clock', 'hunt_clock', 'regen_clock', 'stuck', 'tele', 'observers', 'last_tick_x',
              'last_tick_z', 'hunt_mode', 'huntrange', 'hunt_target', 'lifecycle', 'last_movement', 'life', 'has_path',
              'path_x', 'path_z', 'has_hero', 'hero_damage', 'v_action_delay', 'v_lastcombat', 'v_aggressive_player',
              'v_attacking', 'v_xp_mult', 'v_stunned', 'level_attack', 'level_defence', 'level_strength', 'level_hp',
              'level_ranged', 'level_magic')
# Fixed-stride fields: the names of each group's values (a diff names them; rs.h RS_CANON_FIELDS orders them). The
# other fields are variable: see Canon.
STRIDES = dict(
    RS_CF_TICK=('tick', 'n_stored_locs', 'next_loc', 'loc_version', 'pending_login', 'has_player'),
    RS_CF_PLAYER=('x', 'z', 'level', 'instant', 'steps_taken', 'last_movement', 'delayed', 'delayed_until', 'protect',
                  'dead', 'afk_ready', 'origin_x', 'origin_z', 'allow_repath'),
    RS_CF_PLAYER_PATH=('x', 'z'),
    RS_CF_PLAYER_TARGET=('kind', 'ident', 'target_op', 'target_type', 'ap_range', 'ap_range_called', 'next_kind',
                         'next_ident'),
    RS_CF_PLAYER_STATS=('level', 'base_level', 'xp'),
    RS_CF_PLAYER_LAST=('last_useitem', 'last_com_iface', 'last_com_com', 'last_item', 'last_slot', 'last_useslot'),
    RS_CF_PLAYER_INV=('slot', 'id', 'count'), RS_CF_PLAYER_WORN=('slot', 'id', 'count'),
    RS_CF_PLAYER_BANK=('slot', 'id', 'count'),
    RS_CF_PLAYER_TIMERS=('script', 'kind', 'interval', 'clock'),
    RS_CF_PLAYER_QUEUE=('script', 'delay', 'arg0', 'arg1', 'arg2'), RS_CF_PLAYER_WEAK=('script', 'delay', 'arg0',
                                                                                        'arg1', 'arg2'),
    RS_CF_PLAYER_ENGINE=('script', 'delay', 'arg0', 'arg1', 'arg2'),
    RS_CF_NPC=NPC_FIELDS,
    RS_CF_LOCS=('index', 'base', 'type', 'x', 'z', 'shape', 'angle', 'width', 'length', 'static', 'active', 'lifecycle',
                'layer'),
    RS_CF_OBJS=('type', 'count', 'x', 'z', 'level', 'static', 'ticks', 'receiver', 'active'),
    RS_CF_OCCUPIED=('x', 'z', 'count'), RS_CF_CHANGED=('x', 'z'),
    RS_CF_LOGS=('chat_count', 'chat_last', 'chat_last_tick', 'text_count', 'text_iface', 'text_com', 'text_last',
                'event_count'),
    RS_CF_EVENT_LAST=('kind', 'tick', 'a8', 'b8', 'has_source', 'obj', 'count', 'v0', 'v1', 'v2', 'v3'),
    RS_CF_AGENT=('decision', 'charged', 'episode_ticks', 'last_mask_sum', 'pace', 'inv_page', 'bank_page', 'bank_pages',
                 'style_stale', 'using_slot', 'using_id', 'death_tick', 'visited', 'chat_seq', 'chat_covered',
                 'chat_kept'),
    RS_CF_AGENT_REFUSED=('kind', 'id', 'x', 'z', 'level', 'option_lc', 'until'),
    RS_CF_AGENT_TEXTS=('iface', 'com', 'text', 'tick'),
    RS_CF_EPISODE=('armed', 'active', 'start_tick', 'end', 'event_start', 'terminal', 'reason', 'term_tick',
                   'decisions', 'carried_before', 'banked', 'paid_bits'),
)


def _bits(value):
    """A double's IEEE bits as int64 (EPISODE paid)."""
    return struct.unpack('<q', struct.pack('<d', float(value)))[0]


class Canon:
    """canon(world, agent, task): the canonical state record after a tick, a self-describing int64 array of fields
    [field id, count, count values] in rs.h RS_CANON_FIELDS order, which canon.c rs_canon writes too; its tick hash is
    zlib.crc32 of the little-endian bytes. Conventions for the C side:
    - strings are string ids (`strings`: PackStrings of the pack, crc_strings without one); a component is (iface
      string id, com); entity references are (rs.h RS_REF_* kind, identity): npc nid, Loc.index, position in
      world.objs (-1 when gone), -1 for none; bools are 0/1, None ints -1 unless a field says otherwise;
    - a "(has, ...)" group is [0] alone when absent, else 1 and its values;
    - the player fields are empty (count 0) while world.player is None;
    - fixed-stride fields are STRIDES (one group per entry, e.g. RS_CF_PLAYER_STATS per skill: level, base level,
      xp); the variable ones:
      RS_CF_PLAYER_VARPS: RS_VARPS order (aggressive_npc -1 for None);
      RS_CF_PLAYER_MODAL: modal bits (RS_MODAL_*), modal_main string id or -1, modal_chat string id or -1, then
        iface, com per resume button;
      RS_CF_PLAYER_SCRIPT and RS_CF_NPC_SCRIPT: [nid for an npc], mode (RS_SCRIPT_*), depth, then script, pc per level;
      RS_CF_NPC_QUEUE: per npc nid, n, then id, arg, delay, source (1 the player, 0 None) per entry;
      RS_CF_NPC_EVENTS, RS_CF_LOC_EVENTS: nids, Loc.index values in order;
      RS_CF_LOCS: map locs, then added locs in loc_at, by Loc.index;
      RS_CF_ZONES: per non-empty zone in (zone x, zone z) order, zone coordinates absolute (x >> 3): zx, zz, n, then
        Loc.index per entry in engine order;
      RS_CF_OCCUPIED, RS_CF_CHANGED: inside the layout, sorted by (x, z);
      RS_CF_LOGS: chat count, the last line's string id and tick (-1, -1 when none), text_log count (this tick's,
        before capture_text drains it), its last line's iface, com, text (-1 each when none), event count;
      RS_CF_EVENT_LAST: [] without events, else kind (RS_EV_*), tick, then RsEvent's a8, b8, has_source, obj, count,
        v0..v3 as rs.h documents them (unused 0);
      RS_CF_AGENT: using slot and id -1 for none, death_tick -1 for None, visited count, kept chat count;
      RS_CF_AGENT_OPTION: (has, label (RS_LABEL_*), k (0 for pick), start, last_x, last_z, last_progress, action (-1),
        done kind (RS_DONE_*), done_start (act_done's start, else 0));
      RS_CF_AGENT_ACT: (has, kind (RS_KIND_*), id, x, z, level, option_lc, xp, last_xp, (has, type, x, z, level) of the
        loc, n, then slot, id, count per item, n, then the same per last item);
      RS_CF_AGENT_OPTION_END: (has, action, reason (END_REASONS index), refused, reported);
      RS_CF_AGENT_REFUSED: entries with until > tick, sorted by their values;
      RS_CF_AGENT_TEXTS: sorted by (iface string, com);
      RS_CF_EPISODE: GoalEpisode's values, 0 for those not yet set (before arm), paid as its double's bits.
    Oracle state outside the contract (an unknown varp, npc var, modal, script, request or event) is a CanonError."""

    def __init__(self, strings=crc_strings):
        self.fields, self.ids, _, _, _ = contract()
        self.e, self.varps = header()
        self.strings = strings
        self.id = {name: k for k, name in enumerate(self.fields)}

    def __call__(self, world, agent, task):
        from sim.engine import NPC_VARS
        out, e, sid, p = [], self.e, self.strings, world.player
        name = agent.name

        def put(field, values):
            out.extend((self.id[field], len(values)))
            out.extend(values)

        def ref(target):
            from sim.engine import Loc, Npc, Obj, Player
            if target is None:
                return [e['RS_REF_NONE'], -1]
            if isinstance(target, Npc):
                return [e['RS_REF_NPC'], target.nid]
            if isinstance(target, Loc):
                return [e['RS_REF_LOC'], target.index]
            if isinstance(target, Obj):
                return [e['RS_REF_OBJ'], next((k for k, o in enumerate(world.objs) if o is target), -1)]
            if isinstance(target, Player):
                return [e['RS_REF_PLAYER'], 0]
            raise CanonError(f'a reference to a {type(target).__name__}')

        def player_flag(value):  # None or the agent's player (or its name): rs.h's one-player flags
            if value is None:
                return 0
            if value is p or value == name:
                return 1
            raise CanonError('a reference to another player')

        def component(c):
            return [sid(c[0]), int(c[1])]

        put('RS_CF_TICK', [world.tick, world.n_stored_locs, world._next_loc, world.loc_version,
                           int(world.pending_login is not None), int(p is not None)])
        if p is None:
            for f in ('RS_CF_PLAYER', 'RS_CF_PLAYER_PATH', 'RS_CF_PLAYER_TARGET', 'RS_CF_PLAYER_STATS',
                      'RS_CF_PLAYER_VARPS', 'RS_CF_PLAYER_LAST', 'RS_CF_PLAYER_INV', 'RS_CF_PLAYER_WORN',
                      'RS_CF_PLAYER_BANK', 'RS_CF_PLAYER_MODAL', 'RS_CF_PLAYER_SCRIPT', 'RS_CF_PLAYER_TIMERS',
                      'RS_CF_PLAYER_QUEUE', 'RS_CF_PLAYER_WEAK', 'RS_CF_PLAYER_ENGINE'):
                put(f, [])
        else:
            put('RS_CF_PLAYER', [p.x, p.z, p.level, int(p.instant), p.steps_taken, p.last_movement, int(p.delayed),
                                 p.delayed_until, int(p.protect), int(p.dead), int(p.afk_ready), p.origin_x, p.origin_z,
                                 int(p.allow_repath)])
            put('RS_CF_PLAYER_PATH', [v for step in p.path for v in step])
            put('RS_CF_PLAYER_TARGET', [*ref(p.target), p.target_op, p.target_type, p.ap_range, int(p.ap_range_called),
                                        *ref(p.next_target)])
            put('RS_CF_PLAYER_STATS', [v for k in range(len(p.stats)) for v in (p.levels[k], p.base_levels[k],
                                                                                 p.stats[k])])
            if set(p.varps) != set(self.varps):
                raise CanonError(f'player varps {sorted(set(p.varps) ^ set(self.varps))} outside RS_VARPS')
            put('RS_CF_PLAYER_VARPS', [-1 if p.varps[v] is None else int(p.varps[v]) for v in self.varps])
            last_com = [-1, -1] if p.last_com == -1 else component(p.last_com)
            put('RS_CF_PLAYER_LAST', [p.last_useitem, *last_com, p.last_item, p.last_slot, p.last_useslot])
            for f, inv in (('RS_CF_PLAYER_INV', p.inv), ('RS_CF_PLAYER_WORN', p.worn), ('RS_CF_PLAYER_BANK', p.bank)):
                put(f, [v for row in inv.listed() for v in row])
            bits = {'main': e['RS_MODAL_MAIN'], 'side': e['RS_MODAL_SIDE'], 'chat': e['RS_MODAL_CHAT']}
            if not p.modal_state <= set(bits):
                raise CanonError(f'modal states {sorted(p.modal_state - set(bits))} outside RS_MODAL_*')
            put('RS_CF_PLAYER_MODAL', [sum(bits[s] for s in p.modal_state),
                                       -1 if p.modal_main is None else sid(p.modal_main),
                                       -1 if p.modal_chat is None else sid(p.modal_chat),
                                       *[v for c in p.resume_buttons for v in component(c)]])
            put('RS_CF_PLAYER_SCRIPT', self.script(p.script and p.script[0], p.script and p.script[1]))
            put('RS_CF_PLAYER_TIMERS', [v for timer_name, timer in p.timers.items() for v in (
                self.ids[f'RS_S_{timer_name.upper()}'], {'normal': 0, 'soft': 1}[timer['kind']], timer['interval'],
                timer['clock'])])
            for f, queue in (('RS_CF_PLAYER_QUEUE', p.queue), ('RS_CF_PLAYER_WEAK', p.weak_queue),
                             ('RS_CF_PLAYER_ENGINE', p.engine_queue)):
                put(f, [v for request in queue for v in request_row(request)])

        npcs, scripts, queues = [], [], []
        for npc in world.npcs:
            if set(npc.vars) != set(NPC_VARS) or npc.vars['macro_target'] is not None:
                raise CanonError(f'npc {npc.nid} vars outside rs.h Npc (v_*)')
            if len(npc.path) > 1 or len(npc.heroes) > 1 or len(npc.levels) != 6:
                raise CanonError(f'npc {npc.nid}: {len(npc.path)} waypoints, {len(npc.heroes)} heroes, '
                                 f'{len(npc.levels)} levels (rs.h Npc holds 1, 1, 6)')
            if npc.heroes and next(iter(npc.heroes)) != name:
                raise CanonError(f'npc {npc.nid} has another player as hero')
            v = npc.vars
            npcs += [npc.nid, npc.type, npc.base, npc.x, npc.z, npc.level, int(npc.active), int(npc.delayed),
                     npc.delayed_until, npc.mode, player_flag(npc.target), npc.timer_interval, npc.timer_clock,
                     npc.hunt_clock, npc.regen_clock, npc.stuck, int(npc.tele), npc.observers, npc.last_tick_x,
                     npc.last_tick_z, npc.hunt_mode, npc.huntrange, player_flag(npc.hunt_target), npc.lifecycle_tick,
                     npc.last_movement, npc.life, int(bool(npc.path)), *(npc.path[0] if npc.path else (0, 0)),
                     int(bool(npc.heroes)), sum(npc.heroes.values()), v['action_delay'], v['lastcombat'],
                     player_flag(v['aggressive_player']), player_flag(v['attacking']), v['xp_mult'], v['stunned'],
                     *npc.levels]
            scripts += [npc.nid, *self.script(npc.script, None if npc.script is None else 'suspended')]
            queues += [npc.nid, len(npc.queue), *[int(x) for entry in npc.queue
                                                  for x in (*entry[:3], player_flag(entry[3]))]]
        put('RS_CF_NPC', npcs)
        put('RS_CF_NPC_SCRIPT', scripts)
        put('RS_CF_NPC_QUEUE', queues)
        put('RS_CF_NPC_EVENTS', [npc.nid for npc in world.npc_events])
        added = sorted({id(loc): loc for at in world.loc_at.values() for loc in at
                        if loc.index >= world.n_stored_locs}.values(), key=lambda loc: loc.index)
        put('RS_CF_LOCS', [v for loc in world.locs + added for v in (
            loc.index, loc.base, loc.type, loc.x, loc.z, loc.shape, loc.angle, loc.width, loc.length, int(loc.static),
            int(loc.active), loc.lifecycle_tick, loc.layer)])
        zones = []
        for (level, zx, zz), locs in sorted(world.loc_zones.items()):
            if locs:
                if level != 0:
                    raise CanonError(f'a loc zone on level {level}')
                zones += [zx, zz, len(locs), *[loc.index for loc in locs]]
        put('RS_CF_ZONES', zones)
        put('RS_CF_LOC_EVENTS', [loc.index for loc in world.loc_events])
        put('RS_CF_OBJS', [v for o in world.objs for v in (o.type, o.count, o.x, o.z, o.level, int(o.static), o.ticks,
                                                            player_flag(o.receiver), int(o.active))])
        inside = lambda x, z: world._x0 <= x <= world._x1 and world._z0 <= z <= world._z1
        put('RS_CF_OCCUPIED', [v for x, z, c in sorted((x, z, c) for (level, x, z), c in world.occupied.items()
                                                       if c and level == 0 and inside(x, z)) for v in (x, z, c)])
        put('RS_CF_CHANGED', [v for x, z in sorted((x, z) for level, x, z in world.changed) for v in (x, z)])
        chat, text = world.chat_log, world.text_log
        put('RS_CF_LOGS', [len(chat), *((sid(chat[-1][1]), chat[-1][0]) if chat else (-1, -1)), len(text),
                           *((*component(text[-1][1]), sid(text[-1][2])) if text else (-1, -1, -1)), len(world.events)])
        put('RS_CF_EVENT_LAST', self.event(world.events[-1], name) if world.events else [])
        self.agent(put, world, agent)
        term = getattr(task, 'terminal', None)
        put('RS_CF_EPISODE', [int(getattr(task, 'armed', False)), int(task.active), getattr(task, 'start_tick', 0),
                              getattr(task, 'end', 0), getattr(task, 'event_start', 0), int(term is not None),
                              e[f"RS_TERM_{term['reason'].upper()}"] if term else 0, term['tick'] if term else 0,
                              getattr(task, 'decisions', 0), getattr(task, 'carried_before', 0),
                              getattr(task, 'banked', 0), _bits(getattr(task, 'paid', 0.0))])
        record = np.asarray(out)
        if record.dtype.kind not in 'iub':  # a float or an object would be truncated or refused by the int64 record
            bad = next(k for k, v in enumerate(out) if not isinstance(v, (int, np.integer)))
            raise CanonError(f'a {type(out[bad]).__name__} value in the record (value {bad})')
        return record.astype('<i8')

    def script(self, gen, state):
        """[mode, depth, then script, pc per level] of p.script / npc.script (rs.h RS_SCRIPT_*)."""
        if gen is None:
            return [self.e['RS_SCRIPT_NONE'], 0]
        mode = {'pausebutton': 'RS_SCRIPT_PAUSE', 'suspended': 'RS_SCRIPT_SUSPENDED', 'running': 'RS_SCRIPT_RUNNING'}
        chain = frame_chain(gen)
        return [self.e[mode[state]], len(chain), *[v for level in chain for v in level]]

    def event(self, event, name):
        """An event as rs.h RsEvent: [kind, tick, a8, b8, has_source, obj, count, v0, v1, v2, v3]."""
        e = self.e
        tick, kind, d = event
        a8 = b8 = has = obj = count = v0 = v1 = v2 = v3 = 0
        source = lambda s: 0 if s is None else 1 if s == name else -1
        if kind == 'agent_hp':
            a8 = e['RS_HP_NONE' if d['cause'] is None else f"RS_HP_{d['cause'].upper()}"]
            count, v0, v1, has = d['planned'], d['before'], d['after'], int(d['source'] is not None)
            v2 = d['source'] if has else 0
        elif kind == 'pickup':
            obj, count = d['obj'], d['count']
        elif kind == 'produce':
            obj, count, v0 = d['obj'], d['count'], self.strings(d['script'])
        elif kind == 'transfer':
            a8, b8 = e[f"RS_LOCN_{d['src'].upper()}"], e[f"RS_LOCN_{d['dst'].upper()}"]
            obj, count, v0 = d['obj'], d['count'], d['to']
        elif kind == 'remove':
            a8, obj, count = e[f"RS_WHY_{d['why'].upper()}"], d['obj'], d['count']
        elif kind in ('npc_damage', 'npc_defeated'):
            a8, obj, count, v0 = source(d['source']), d['npc'], d['type'], d['life']
            if kind == 'npc_damage':
                v1, v2, v3 = d['damage'], d['hp'], d['full']
            if a8 < 0:
                raise CanonError('an npc event from another player')
        elif kind == 'excluded':
            a8 = e[{'clue_drop': 'RS_EXCL_CLUE', 'loot_drop': 'RS_EXCL_LOOT'}[d['what']]]
            obj, v0 = d['obj'], d['npc']
        elif kind == 'xp':  # World.add_xp: the stat, its cause (RS_XPC_*), the amount and the stat before and after
            cause = d.get('cause')
            if cause not in (None, 'produce', 'hit', 'bury', 'light'):
                raise CanonError(f'an xp cause {cause} outside RS_XPC_*')
            a8, b8, count, v0, v1 = d['stat'], e[f"RS_XPC_{(cause or 'none').upper()}"], d['xp'], d['before'], d['after']
            if cause == 'hit':
                obj, v2 = d['npc'], d['life']
            elif cause == 'produce':
                obj, v2 = d['obj'], -1 if d['src'] is None else d['src']
            elif cause is not None:
                obj = d['obj']
        elif kind == 'hit':  # player_melee_attack's hit, the xp's cause
            a8, obj, count, v0, v1, v2, v3 = source(d['source']), d['npc'], d['type'], d['life'], d['damage'], d['hp'], d['style']
            if a8 < 0:
                raise CanonError('a hit by another player')
        elif kind == 'npc_respawn':  # World.npc_respawn's record (the XP check's): the npc, its base type, the new life
            obj, count, v0 = d['npc'], d['type'], d['life']
        else:
            raise CanonError(f'event kind {kind} outside RS_EV_*')
        return [e[f'RS_EV_{kind.upper()}'], tick, a8, b8, has, obj, count, v0, v1, v2, v3]

    def agent(self, put, world, agent):
        """The RS_CF_AGENT* fields (agent.py: the option dicts of begin and act and their closures)."""
        from feedback_contract import END_REASONS
        e, sid = self.e, self.strings
        using = agent.using
        put('RS_CF_AGENT', [agent.decision, agent.charged, agent.episode_ticks, agent.last_mask_sum, agent.pace,
                            agent.inv_page, agent.bank_page, agent.bank_pages, int(agent.style_stale),
                            *((using['slot'], using['id']) if using else (-1, -1)),
                            -1 if agent.death_tick is None else agent.death_tick, len(agent.visited), agent.chat_seq,
                            agent.chat_covered, len(getattr(agent, 'chat', ()))])
        o = agent.option
        if o is None:
            put('RS_CF_AGENT_OPTION', [0])
            put('RS_CF_AGENT_ACT', [0])
        else:
            label = re.fullmatch(r'(act|nav_click)_(\d+)', o['label'])
            if label:
                kind, k = f'RS_LABEL_{label[1].upper()}', int(label[2])
            elif o['label'] == 'pick_item':
                kind, k = 'RS_LABEL_PICK', 0
            else:
                raise CanonError(f"option label {o['label']} outside RS_LABEL_*")
            done = o['done'].__code__.co_qualname
            if done == 'Agent.act_done.<locals>.done':
                done_kind, done_start = e['RS_DONE_ACT'], _cells(o['done'])['start']
            elif done == 'Agent.pick_up.<locals>.<lambda>':
                done_kind, done_start = e['RS_DONE_PICK'], 0
            elif done == 'Agent.click.<locals>.<lambda>':
                done_kind, done_start = e['RS_DONE_WALK'], 0
            else:
                raise CanonError(f'option done test {done} outside RS_DONE_*')
            put('RS_CF_AGENT_OPTION', [1, e[kind], k, o['start'], *o['last_pos'], o['last_progress'],
                                       o.get('action', -1), done_kind, done_start])
            act = o.get('act')
            if act is None:
                put('RS_CF_AGENT_ACT', [0])
            else:
                items = lambda state: [int(v) for part in state.split(',') if part for v in part.split(':')]
                now, last = items(act['items']), items(act['last_items'])
                put('RS_CF_AGENT_ACT', [1, *self.key(act['key']), act['xp'], act['last_xp'],
                                        *([1, *act['loc']] if act['loc'] else [0]), len(now) // 3, *now,
                                        len(last) // 3, *last])
        end = agent.option_end
        put('RS_CF_AGENT_OPTION_END', [0] if end is None else [
            1, end['action'], END_REASONS.index(end['reason']), int(end['refused']), int(end['reported'])])
        refused = sorted(tuple(self.key(key)) + (until,) for key, until in agent.refused.items() if until > world.tick)
        put('RS_CF_AGENT_REFUSED', [v for row in refused for v in row])
        put('RS_CF_AGENT_TEXTS', [v for (iface, com), (text, tick) in sorted(agent.texts.items())
                                  for v in (sid(iface), com, sid(text), tick)])

    def key(self, key):
        """An affordance key f'{kind}:{id}:{x}:{z}:{level}:{option.lower()}' (agent.py:1108) as rs.h RsAffKey."""
        kind, ident, x, z, level, option = key.split(':', 5)
        return [self.e[f'RS_KIND_{kind.upper()}'], int(ident), int(x), int(z), int(level), self.strings(option)]


def decode(record, fields=None):
    """{field name: int64 values} of a canonical record, refusing a malformed one."""
    fields = contract()[0] if fields is None else fields
    out, k, record = OrderedDict(), 0, np.asarray(record, np.int64)
    while k < len(record):
        fid, n = int(record[k]), int(record[k + 1])
        if not 0 <= fid < len(fields) or k + 2 + n > len(record):
            raise ValueError(f'a malformed canonical record at value {k}')
        out[fields[fid]] = record[k + 2:k + 2 + n]
        k += 2 + n
    return out


def field_diff(a, b, limit=8):
    """Lines naming the first differing values of two canonical records, field by field."""
    a, b = decode(a), decode(b)
    lines = []
    for name in dict.fromkeys([*a, *b]):
        va, vb = a.get(name, np.zeros(0, np.int64)), b.get(name, np.zeros(0, np.int64))
        if np.array_equal(va, vb):
            continue
        stride = STRIDES.get(name) or (header()[1] if name == 'RS_CF_PLAYER_VARPS' else None)
        if stride and len(va) == len(vb) and len(va) % len(stride) == 0:
            for k in np.flatnonzero(va != vb)[:limit]:
                group, at = divmod(int(k), len(stride))
                lines.append(f'{name}[{group}].{stride[at]}: {va[k]} != {vb[k]}')
        else:
            k = next((i for i in range(min(len(va), len(vb))) if va[i] != vb[i]), min(len(va), len(vb)))
            lines.append(f'{name}: {len(va)} vs {len(vb)} values, first difference at {k}: '
                         f'{va[k:k + 6].tolist()} vs {vb[k:k + 6].tolist()}')
        if len(lines) >= limit:
            break
    return lines


# ---- the Python side's trace -------------------------------------------------------------------------------------
@dataclass
class Row:
    """One world cycle: the record after it (rs.h hash point), its crc32, and what the world appended since the
    previous Row: draws (tick, keys, m), events (Canon.event rows), chat (tick, string id), texts (tick, iface, com,
    string id). tick is world.tick after the cycle (rs_trace_tick's); episode the slot's episode counter."""
    tick: int
    crc: int
    record: np.ndarray
    draws: list
    events: list
    chat: list
    texts: list
    episode: int


class Prefix(int):
    """Rng.prefix's chain state carrying the keys it folded, so a draw from it logs its whole key tuple."""
    keys = ()


class TracedRng:
    """A world's Rng (engine.py:94-117) that logs each draw: world.rng(*keys), world.rng.prefix and
    world.rng.draw(h, *keys), the three ways the oracle draws from a world."""

    def __init__(self, rng, log):
        self.rng, self.seed, self.log = rng, rng.seed, log

    def __call__(self, *keys):
        value = self.rng(*keys)
        self.log(keys, value)
        return value

    def prefix(self, *keys):
        h = Prefix(self.rng.prefix(*keys))
        h.keys = keys
        return h

    def draw(self, h, *keys):
        value = self.rng.draw(h, *keys)
        self.log(tuple(getattr(h, 'keys', ('?',))) + keys, value)
        return value


class WorldTrace:
    """One world slot of a Python side: the Rows of the current call, and the last TAIL Rows for reports."""

    def __init__(self, side, b):
        self.side, self.b = side, b
        self.task, self.rows, self.pending, self.tail = None, [], [], deque(maxlen=TAIL)
        self.events_seen = self.chat_seen = 0
        self.episode = 0
        self.most_objs = 0  # the most ground objs (len(world.objs)) at a tick's end since run() began

    def bind(self, task):
        """A new episode of this slot (GoalEpisode.__init__): its agent's Rows and its world's draws come here."""
        if self.pending and self.rows:  # a previous world's draws after its last cycle
            self.rows[-1].draws.extend(self.pending)
        self.pending, self.task, self.events_seen, self.chat_seen = [], task, 0, 0
        self.episode = self.side.env.episodes[self.b]
        _TRACERS[task.agent] = self
        world = task.world
        world.rng = TracedRng(world.rng, lambda keys, value: self.pending.append(
            (world.tick, tuple(int(k) for k in keys), int(value * 16777216))))

    def tick(self, agent):
        task = self.task
        if task is None or task.agent is not agent:
            raise CanonError('a cycle of an agent this trace did not bind')
        world, side = task.world, self.side
        if side.perturb is not None:
            side.perturb(self, side.step_index, world, agent, task)
        record, sid = side.canon(world, agent, task), side.canon.strings
        events = [side.canon.event(ev, agent.name) for ev in world.events[self.events_seen:]]
        chat = [(t, sid(text)) for t, text in world.chat_log[self.chat_seen:]]
        texts = [(t, sid(iface), com, sid(text)) for t, (iface, com), text in world.text_log]
        self.events_seen, self.chat_seen = len(world.events), len(world.chat_log)
        self.most_objs = max(self.most_objs, len(world.objs))
        row = Row(world.tick, zlib.crc32(record.tobytes()), record, self.pending, events, chat, texts, self.episode)
        self.pending = []
        self.rows.append(row)
        self.tail.append(row)


_TRACERS = weakref.WeakKeyDictionary()  # Agent -> WorldTrace
_BINDING = []                           # the WorldTrace of the CpuBatch._episode call running


class Scenario:
    """A scenario start: `layout` (compact pool index) and `levels` for draw_start, `group`, `hp`,
    `food` and `worn` for draw_combat, `supply_layout`, `variant` (and `levels`) for draw_supply. Draw parameters go
    through the draws' own arguments, so every other draw is the one CpuBatch makes; hp is kept inside 1..base."""

    KEYS = ('layout', 'levels', 'group', 'hp', 'food', 'worn', 'supply_layout', 'variant')

    def __init__(self, spec):
        """spec: a SCENARIOS name, a JSON object's text or a dict."""
        if isinstance(spec, str):
            spec = {**SCENARIOS, **FREE_SCENARIOS}[spec] if spec in SCENARIOS or spec in FREE_SCENARIOS else json.loads(spec)
        spec = dict(spec)
        unknown = set(spec) - set(self.KEYS)
        if unknown:
            raise ValueError(f'unknown scenario keys {sorted(unknown)}')
        self.spec = spec

    def start(self, draw, key, slot, episode, layouts, layout=None):
        start = draw(key, slot, episode, layouts, layout=self.spec.get('layout', layout))
        start['levels'] = {**start['levels'], **self.spec.get('levels', {})}
        return start

    def combat(self, draw, key, slot, episode, layout, start, tables, group=None, region=None):
        c = draw(key, slot, episode, layout, start, tables, group=self.spec.get('group', group), region=region)
        if 'hp' in self.spec:
            c['hp'] = max(1, min(start['levels']['hitpoints'], int(self.spec['hp'])))
        for name in ('food', 'worn'):
            if name in self.spec:
                c[name] = self.spec[name]
        return c

    def village(self, draw, key, slot, episode, layouts, tables, muggers, layout=None, region=None):
        """A free-play start (sim/combat.py draw_village): its draws, then the spec's levels, hp, food and worn
        (its group is VILLAGE's; a `layout` replaces the layout draw)."""
        if 'group' in self.spec or 'variant' in self.spec or 'supply_layout' in self.spec:
            raise ValueError('a free-play scenario sets layout, levels, hp, food or worn')
        start = draw(key, slot, episode, layouts, tables, muggers, layout=self.spec.get('layout', layout), region=region)
        start['levels'] = {**start['levels'], **self.spec.get('levels', {})}
        v = start['village']
        if 'hp' in self.spec:
            v['hp'] = max(1, min(start['levels']['hitpoints'], int(self.spec['hp'])))
        for name in ('food', 'worn'):
            if name in self.spec:
                v[name] = self.spec[name]
        return start

    def supply(self, draw, key, slot, episode, layouts, sites, layout=None, variant=None):
        start = draw(key, slot, episode, layouts, sites, layout=self.spec.get('supply_layout', layout),
                     variant=self.spec.get('variant', variant))
        start['levels'] = {**start['levels'], **self.spec.get('levels', {})}
        return start


@contextlib.contextmanager
def hooks(scenario=None):
    """The Python side's hooks for one batch call (module docstring), restored on exit."""
    from sim import search, vec
    from sim.agent import Agent
    from sim.task import FreePlayEpisode, GoalEpisode
    init, free_init, capture = GoalEpisode.__init__, FreePlayEpisode.__init__, Agent.capture_text
    draws = vec.draw_start, vec.draw_combat, search.draw_supply, vec.draw_village

    def episode_init(self, world, *args, **kwargs):
        init(self, world, *args, **kwargs)
        if _BINDING:
            _BINDING[-1].bind(self)

    def life_init(self, world, *args, **kwargs):  # FreePlayEpisode's own __init__ (it does not call GoalEpisode's)
        free_init(self, world, *args, **kwargs)
        if _BINDING:
            _BINDING[-1].bind(self)

    def capture_text(self):
        trace = _TRACERS.get(self)
        if trace is not None:
            trace.tick(self)
        capture(self)
    GoalEpisode.__init__, FreePlayEpisode.__init__, Agent.capture_text = episode_init, life_init, capture_text
    if scenario is not None:
        vec.draw_start = lambda *a, **k: scenario.start(draws[0], *a, **k)
        vec.draw_combat = lambda *a, **k: scenario.combat(draws[1], *a, **k)
        search.draw_supply = lambda *a, **k: scenario.supply(draws[2], *a, **k)
        vec.draw_village = lambda *a, **k: scenario.village(draws[3], *a, **k)
    try:
        yield
    finally:
        GoalEpisode.__init__, FreePlayEpisode.__init__, Agent.capture_text = init, free_init, capture
        vec.draw_start, vec.draw_combat, search.draw_supply, vec.draw_village = draws


# ---- sides ----------------------------------------------------------------------------------------------------------
def twin(env):
    """A second CpuBatch equal to `env` for a self-check: a fresh batch's state, sharing env's layout tables, which are
    pure memo (their LRU caches return what a rebuild returns), so the check does not hold the tables twice."""
    other = copy.copy(env)
    other.__dict__.pop('_episode', None)
    other.key, other.evidence, other.regions = None, [], dict(env.regions)
    if hasattr(env, 'variant_tables'):
        other.variant_tables = OrderedDict()
    return other


class PythonSide:
    """A CpuBatch as a side, with tick Rows. `perturb(trace, step, world, agent, task)`, if given, runs in each tick
    hook before the record (the harness's own tests); step is the run's step index (-1 at reset)."""

    def __init__(self, env, canon=None, scenario=None, perturb=None, name='cpu'):
        self.env, self.canon, self.perturb, self.name = env, canon or Canon(), perturb, name
        self.scenario = None if scenario is None else Scenario(scenario)
        self.n, self.first, self.step_index = env.n, env.first, -1
        self.traces = [WorldTrace(self, b) for b in range(env.n)]

        def episode(b):
            _BINDING.append(self.traces[b])
            try:
                return type(env)._episode(env, b)
            finally:
                _BINDING.pop()
        env._episode = episode
        self.obs = None

    def reset(self, key):
        self.step_index = -1
        with hooks(self.scenario):
            state, self.obs = self.env.reset(key)
        return state, self.obs

    def step(self, state, action):
        with hooks(self.scenario):
            out = self.env.step(state, action)
        self.obs = out[1]
        return out

    def start_episode(self, b, episode):
        """Rebuild world b at episode `episode` (a repro: CpuBatch's counter set, then its _episode)."""
        from sim.spec import dtype
        self.env.episodes[b] = episode
        with hooks(self.scenario):
            o = self.env._episode(b)
        obs = {name: value.copy() for name, value in self.obs.items()}
        for name in self.env.shapes:
            obs[name][b] = np.asarray(o[name], dtype(name))
        self.obs = obs
        return self.env._state(), obs

    def take(self):
        """Each world's Rows since the last take."""
        rows = [trace.rows for trace in self.traces]
        for trace in self.traces:
            trace.rows = []
        return rows

    def tail(self, b):
        return list(self.traces[b].tail)

    @property
    def goals(self):
        """Each world's goal, for the reports ('no goal' in free play)."""
        return ['no goal'] * self.n if self.env.free else list(self.env.goals)

    @property
    def starts(self):
        return self.env.starts

    @property
    def evidence(self):
        return self.env.evidence

    @property
    def episodes(self):
        return list(self.env.episodes)

    @property
    def decisions(self):
        return [getattr(task, 'decisions', 0) for task in self.env.tasks]


class NativeSide:
    """A NativeBatch as a side, every world traced (canon.c rows, taken after each call). With a scenario the native
    side plays the start records the Python side `python` drew under it (CpuBatch.starts, read after the Python side's
    call: run() calls side a first): after reset and after a step that ended an episode, each new world waits
    (NativeBatch.external_starts) and is begun from that record. A repro sets the episode counter and begins the
    episode."""

    def __init__(self, batch, scenario=None, name='native', python=None):
        if scenario is not None and python is None:
            raise ValueError('a native scenario side plays the Python side\'s start records: pass python')
        self.batch, self.name, self.n, self.first, self.step_index = batch, name, batch.n, batch.first, -1
        self.python = python if scenario is not None else None
        self.tails = [deque(maxlen=TAIL) for _ in range(batch.n)]
        batch.trace(True)
        batch.external_starts(self.python is not None)

    def _rebegin(self, worlds):
        """Begin these worlds from the Python side's starts (scenario)."""
        for b in worlds:
            self.batch.begin(int(b), self.python.env.starts[int(b)])

    def reset(self, key):
        if self.python is None:
            return self.batch.reset(key)
        self.batch.reset_into(key, check=False)
        self._rebegin(range(self.n))
        self.batch.check()
        return self.batch.state(), self.batch.observations()

    def step(self, state, action):
        if self.python is None:
            return self.batch.step(state, action)
        if state['key'] != self.batch.key:
            raise ValueError('a state of another reset')
        try:
            self.batch.step_into(np.asarray(action, np.int32), check=False)
        except Exception as error:  # the worlds below the failing one stepped: begin those whose life ended, as the
            last = getattr(error, 'world', None)  # Python side's step began them before its own raise, then raise
            try:
                if isinstance(last, int) and last > 0:
                    self._rebegin(np.flatnonzero(self.batch.views['done'][:last]))
            except Exception:  # noqa: BLE001 - the step's own error is the one both sides are compared on
                pass
            raise error
        self._rebegin(np.flatnonzero(self.batch.views['done']))
        self.batch.check()
        return self.batch.outputs()

    def start_episode(self, b, episode):
        self.batch.set_episode(b, episode)
        self.batch.begin(b, None if self.python is None else self.python.env.starts[b])
        return self.batch.state(), self.batch.observations()

    def take(self):
        rows = [[Row(*r) for r in self.batch.trace_rows(b)] for b in range(self.n)]
        for b, mine in enumerate(rows):
            self.tails[b].extend(mine)
        return rows

    def tail(self, b):
        return list(self.tails[b])

    @property
    def goals(self):
        return ['no goal'] * self.n if self.batch.free else list(self.batch.goals)

    @property
    def starts(self):
        return self.batch.starts

    @property
    def evidence(self):
        return self.batch.evidence

    @property
    def episodes(self):
        return self.batch.episodes

    @property
    def decisions(self):
        return None


# ---- action sources (keyed or recorded, never wall-clock random) -------------------------------------------------
class RandomLegal:
    """Keyed random legal actions: legal[int(Rng(key)(R_ACTION, step, slot) * len(legal))], sim/tests/test_workers.py
    run's rule with the world's global slot (equal to its index for first 0), so a slice plays as its whole batch."""

    def __init__(self, key):
        from sim.engine import Rng
        self.rng = Rng(key)

    def __call__(self, step, obs, slots, starts):
        out = []
        for b, slot in enumerate(slots):
            legal = np.flatnonzero(obs['mask'][b])
            out.append(legal[int(self.rng(R_ACTION, step, int(slot)) * len(legal))])
        return np.asarray(out, np.int32)

    def __str__(self):
        return 'random'


class Biased(RandomLegal):
    """Keyed random legal play weighting the act, held, use and dialog families (BIAS): one draw (R_BIASED, step,
    slot) over the legal actions' cumulative weights."""

    def __init__(self, key, bias=None, name='biased'):
        from sim.agent import N_ACTIONS, family
        super().__init__(key)
        self.name = name
        bias = BIAS if bias is None else bias

        def weight(a):
            try:
                return bias.get(family(a), 1.0)
            except ValueError:  # row 71 has no family under act-v3 and is never legal
                return 1.0
        self.weight = np.asarray([weight(a) for a in range(N_ACTIONS)])

    def __call__(self, step, obs, slots, starts):
        out = []
        for b, slot in enumerate(slots):
            legal = np.flatnonzero(obs['mask'][b])
            edges = np.cumsum(self.weight[legal])
            k = int(np.searchsorted(edges, self.rng(R_BIASED, step, int(slot)) * edges[-1], side='right'))
            out.append(legal[min(k, len(legal) - 1)])
        return np.asarray(out, np.int32)

    def __str__(self):
        return self.name


class Listed:
    """A recorded list (Hypothesis): world b's action at step s is legal[values[(s * n + b) % len] % len(legal)]."""

    def __init__(self, values):
        self.values = [int(v) for v in values]

    def __call__(self, step, obs, slots, starts):
        n, out = len(slots), []
        for b in range(n):
            legal = np.flatnonzero(obs['mask'][b])
            out.append(legal[self.values[(step * n + b) % len(self.values)] % len(legal)])
        return np.asarray(out, np.int32)

    def __str__(self):
        return 'list:' + ','.join(map(str, self.values))


class Replay:
    """Exact actions per step (a repro of one world: one action each)."""

    def __init__(self, actions):
        self.actions = [np.atleast_1d(np.asarray(a, np.int32)) for a in actions]

    def __call__(self, step, obs, slots, starts):
        return self.actions[step]


class PolicySource:
    """A policy's sampled actions (design 5, source 3): the actor of a MaskedRecurrentPolicy on the Python side's
    observation, its LSTM state carried per world and reset at episode starts, each action drawn by torch.multinomial
    from the masked distribution with a seeded generator. The other side gets the same actions after its observation
    compared equal; the policy is never recomputed in C. A goal policy reads obs-v5 (check_names), an array the batch
    lacks filled with zeros; a free-play-v1 one reads obs-v6 (`exact`): loaded by sim/hour.py load_free_policy with its
    checks, it plays only observations of exactly its own arrays, so it never acts on a goal task's."""

    def __init__(self, policy, spaces, seed=0, exact=False):
        import torch
        self.torch, self.policy, self.spaces, self.seed, self.exact = torch, policy, spaces, seed, exact
        self.generator = torch.Generator().manual_seed(seed)
        self.state = None
        policy.set_training_mode(False)

    @classmethod
    def load(cls, path, seed=0):
        from sb3_contrib import RecurrentPPO
        from baseline.policy import check_names
        from sim.task import FREE_PLAY
        model = RecurrentPPO.load(path, device='cpu')
        if getattr(model.policy, 'observation', None) == FREE_PLAY['observation']:
            from sim.hour import load_free_policy
            model, _ = load_free_policy(path)
            return cls(model.policy, model.observation_space.spaces, seed, exact=True)
        check_names(model.policy)
        return cls(model.policy, model.observation_space.spaces, seed)

    def __call__(self, step, obs, slots, starts):
        torch, policy, n = self.torch, self.policy, len(slots)
        if self.exact and set(obs) != set(self.spaces):
            raise ValueError(f"an observation with {sorted(set(obs) ^ set(self.spaces))} unlike the policy's own arrays")
        if self.state is None:
            lstm = policy.lstm_actor
            self.state = tuple(torch.zeros(lstm.num_layers, n, lstm.hidden_size) for _ in range(2))
        full = {k: obs[k] if k in obs else np.zeros((n, *space.shape), space.dtype) for k, space in self.spaces.items()}
        with torch.no_grad():
            tensors, _ = policy.obs_to_tensor(full)
            dist, self.state = policy.get_distribution(tensors, self.state, torch.as_tensor(starts, dtype=torch.float32))
            action = torch.multinomial(dist.distribution.probs, 1, generator=self.generator)
        return action.squeeze(1).numpy().astype(np.int32)


class ControlSource:
    """sim/free.py's controls on a free-play Python side (a yardstick's action sequence for both engines, never a
    label): world b plays control names[b % len(names)], which reads that world's life on the Python side (privileged)
    and acts through its legal actions; each new life resets it, keyed by (key, slot, episode). `period`: the
    rotation's ticks a member."""

    def __init__(self, side, names, key, period=None):
        from sim import free
        self.side, self.key, self.names = side, key, tuple(names)
        self.actors = [free.make(self.names[b % len(self.names)], period) for b in range(side.n)]

    def __call__(self, step, obs, slots, starts):
        env = self.side.env
        out = np.zeros(len(slots), np.int32)
        for b in range(len(slots)):
            if starts[b]:
                self.actors[b].reset([int(self.key), int(slots[b]), int(env.episodes[b])])
            out[b] = self.actors[b].act({name: value[b] for name, value in obs.items()}, env.tasks[b])
        return out

    def __str__(self):
        return f'control:{",".join(self.names)}'


def make_source(spec, key):
    """random | biased | idle | walk | eat | list:1,2,3 | policy:PATH[:seed]."""
    if spec == 'random':
        return RandomLegal(key)
    if spec == 'biased':
        return Biased(key)
    if spec in UNDER_ATTACK:
        return Biased(key, UNDER_ATTACK[spec], spec)
    if spec.startswith('list:'):
        return Listed(spec[5:].split(','))
    if spec.startswith('policy:'):
        path, _, seed = spec[7:].partition(':')
        return PolicySource.load(path, int(seed or 0))
    raise ValueError(f'unknown action source {spec}')


# ---- the lockstep diff -------------------------------------------------------------------------------------------
@dataclass
class Divergence:
    """The first divergence of a run (module docstring). where: start, ticks (one side ran another number of cycles),
    tick (records differ), draws, events, chat, texts, observation, reward, done, info, evidence, exception."""
    where: str
    local: int
    slot: int
    key: int
    goal: str
    episode: int
    step: int
    decision: int
    tick: int
    detail: list
    actions: list
    tails: dict = field(default_factory=dict)
    repro: str = ''

    def text(self):
        lines = [f'[native diff] first divergence: {self.where}',
                 f'  world slot {self.slot} (local {self.local}), goal {self.goal}, key {self.key}, episode {self.episode}, '
                 f'step {self.step}, decision {self.decision}, tick {self.tick}',
                 f'  last {len(self.actions)} actions: {self.actions}']
        lines += [f'  {line}' for line in self.detail]
        for name, tail in self.tails.items():
            for what in ('draws', 'events', 'chat'):
                values = [v for row in tail for v in getattr(row, what)][-TAIL:]
                lines.append(f'  {name} last {what}: {values}')
        lines.append(f'  repro: {self.repro}')
        return '\n'.join(lines)


@dataclass
class Result:
    """A run: decisions stepped, worlds, Rows compared (traced: both sides gave Rows), episodes ended; the first
    divergence, why the run stopped early (both sides raised alike, an unsupported world), and a gap: a path one side
    has not ported (NotPorted), which stops the comparison and is never a divergence. most_objs: the most ground objs
    one world of the Python side held at a tick's end (len(world.objs); rs.h RS_MAX_OBJS bounds the C side's)."""
    steps: int = 0
    worlds: int = 0
    ticks: int = 0
    episodes: int = 0
    traced: bool = False
    divergence: Divergence | None = None
    stopped: str = ''
    gap: str = ''
    most_objs: int = 0


def _plain(value):
    """A start or evidence value as plain JSON types (numpy scalars, tuples), for comparison."""
    return json.loads(json.dumps(value, sort_keys=True, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))


def _bits_of(array):
    array = np.ascontiguousarray(array)
    return array.view(f'u{array.dtype.itemsize}') if array.dtype.kind in 'fiub' else array


def _array_diff(name, a, b):
    """Lines for differing values of one array or scalar (a 0-d value: an info field of one world), bitwise, floats
    shown with their bits."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return [f'{name}: shape {a.shape} vs {b.shape}']
    where = np.flatnonzero(_bits_of(a.reshape(-1)) != _bits_of(b.reshape(-1)))
    if not len(where):
        return []
    k = tuple(int(i) for i in np.unravel_index(int(where[0]), a.shape))
    shown = (lambda v: f'{float(v)!r} ({int(np.asarray(v).view(np.uint32)):08x})') if a.dtype == np.float32 \
        else (lambda v: repr(v.item()))
    return [f'{name}{list(k)}: {shown(a[k])} != {shown(b[k])} ({len(where)} values differ)']


def _compare_rows(ra, rb):
    """(where, detail, k) of the first differing Row pair (k: its index in this call's Rows), or None. Draws compare
    as a multiset per Row (they are pure functions of their keys: only the set of key tuples must match)."""
    for k in range(max(len(ra), len(rb))):
        if k >= len(ra) or k >= len(rb):
            return 'ticks', [f'{len(ra)} vs {len(rb)} cycles this decision'], k
        x, y = ra[k], rb[k]
        if x.tick != y.tick:
            return 'ticks', [f'cycle {k}: tick {x.tick} vs {y.tick}'], k
        if x.crc != y.crc:
            return 'tick', [f'tick hash {x.crc:08x} vs {y.crc:08x}', *field_diff(x.record, y.record)], k
        for what in ('draws', 'events', 'chat', 'texts'):
            u, v = getattr(x, what), getattr(y, what)
            if (sorted(u) if what == 'draws' else u) != (sorted(v) if what == 'draws' else v):
                only = lambda p, q: [i for i in p if i not in q][:TAIL]
                return what, [f'{what} differ at tick {x.tick}: only first {only(u, v)}, only second {only(v, u)}'], k
    return None


def _until(tail, rows, k):
    """A side's last TAIL Rows up to its k-th Row of this call: the tail's earlier Rows, then this call's to k."""
    mine = {id(row) for row in rows}
    return ([row for row in tail if id(row) not in mine] + list(rows[:k + 1]))[-TAIL:]


def _compare_outputs(b, out_a, out_b, info_names):
    """(where, detail) of world b's first differing decision output, or None. The observation arrays are the sides'
    table's (obs-v5, or obs-v6 for free play, whose info also holds last_obs: each ended life's last observation)."""
    _, oa, ra, da, ia = out_a
    _, ob, rb, db, ib = out_b
    if list(oa) != list(ob):
        return 'observation', [f'observation keys {list(oa)} vs {list(ob)}']
    lines = [line for name in oa for line in _array_diff(name, oa[name][b], ob[name][b])]
    if lines:
        return 'observation', lines
    if ra is None:  # a reset
        return None
    if np.float32(ra[b]).view(np.uint32) != np.float32(rb[b]).view(np.uint32):
        return 'reward', [f'reward {ra[b]!r} != {rb[b]!r}']
    if bool(da[b]) != bool(db[b]):
        return 'done', [f'done {bool(da[b])} != {bool(db[b])}']
    lines = [line for name in info_names for line in _array_diff(f'info.{name}', ia[name][b], ib[name][b])]
    if ('last_obs' in ia) != ('last_obs' in ib):
        lines.append(f"info.last_obs on one side only ({'last_obs' in ia} vs {'last_obs' in ib})")
    elif 'last_obs' in ia:
        la, lb = ia['last_obs'], ib['last_obs']
        lines += [f'info.last_obs keys {list(la)} vs {list(lb)}'] if list(la) != list(lb) else \
            [line for name in la for line in _array_diff(f'info.last_obs.{name}', la[name][b], lb[name][b])]
    return ('info', lines) if lines else None


def _evidence(rows, slot):
    return [_plain({k: v for k, v in row.items() if k != 'observation'}) for row in rows if row['world'] == slot]


# The check failures (sim/audit.py AuditError) whose C text (rs.h RS_E_AUDIT_*) carries less than the oracle's: the
# oracle's text cut to the fields the C reason carries (an XP event's stat for its dict, a production's or a hit's cause
# and tick for its dict and stats, a holding's place for its two counts, no values for the pay's). Every other check
# failure's C text is the oracle's own. XP_WHY maps the XpCheck._fail reasons that C words without their values.
AUDIT_CUTS = (
    (re.compile(r'(production|hit|bones buried) at tick (-?\d+) \(.*\) ha(?:s|ve) no xp event(?: for stats \[[\d, ]*\])?',
                re.S), lambda m: f'{m[1]} at tick {m[2]} has no xp event'),
    (re.compile(r'xp event at tick (-?\d+) \((\{.*\})\): (.*)', re.S), lambda m: _xp_cut(*m.groups())),
    (re.compile(r'(\w+) holdings not explained by the events: .*', re.S),
     lambda m: f'{m[1]} holdings not explained by the events'),
    (re.compile(r'paid \S+, the fractional level gain \S+'), lambda m: 'paid differs from the fractional level gain'),
)
XP_WHY = (
    (re.compile(r'production by .* trains no such skill', re.S), 'production by a script that trains no such skill'),
    (re.compile(r'a hit in style -?\d+ gives that stat no more XP'), 'a hit in its style gives that stat no more XP'),
    (re.compile(r'the stat held -?\d+, not -?\d+'), "the stat's XP before is not its checked XP"),
    (re.compile(r'the tables give -?\d+ at rate -?\d+'), 'the tables give another amount at its rate'),
)


def _xp_cut(tick, data, why):
    """XpCheck._fail's text as rs.h RS_E_AUDIT_XP renders it: the event's stat for its dict, XP_WHY's wording."""
    try:
        stat = ast.literal_eval(data)['stat']
    except (ValueError, SyntaxError, TypeError, KeyError):
        return None
    why = next((c_why for pattern, c_why in XP_WHY if pattern.fullmatch(why)), why)
    return f'xp event at tick {tick} (stat {stat}): {why}'


def audit_cut(text):
    """The C text of an oracle check failure's text (AUDIT_CUTS), or None when no cut applies."""
    for pattern, cut in AUDIT_CUTS:
        found = pattern.fullmatch(text)
        if found:
            return cut(found)
    return None


def raised_alike(error_a, error_b):
    """How two sides' raises compare: 'text' (one class, one text), 'cut' (a check failure, sim/audit.py AuditError,
    whose C text is the oracle's text cut to what its C reason carries: audit_cut) or None. Side a is the oracle."""
    from sim.audit import AuditError
    if type(error_a) is not type(error_b):
        return None
    if str(error_a) == str(error_b):
        return 'text'
    if isinstance(error_a, AuditError) and audit_cut(str(error_a)) == str(error_b):
        return 'cut'
    return None


def _call(fn, *args):
    try:
        return fn(*args), None
    except Exception as error:  # compared as type and text: both sides must raise alike (raised_alike)
        return None, error


def run(a, b, source, key, steps, episode=0, repro=None, observe=None):
    """Step sides a (a PythonSide: its observations feed the source, its decisions and goals label the report) and b
    in lockstep under `source` (step, obs, slots, starts) -> actions for up to `steps` decisions from reset(key): a
    Result whose divergence is the first one (None: none). With `episode` (a repro), world 0 is rebuilt at that episode
    after the reset, and only that episode is compared. repro(slot, episode, actions) renders the report's repro
    line. observe(step, out_a, out_b), if given, sees both sides' outputs once they compared equal (step -1: the
    first observations). The Result's most_objs is the most ground objs a world of side a held at a tick's end in this
    run."""
    traces = getattr(a, 'traces', [])
    for trace in traces:
        trace.most_objs = 0
    result = _run(a, b, source, key, steps, episode, repro, observe)
    result.most_objs = max((trace.most_objs for trace in traces), default=0)
    return result


def _run(a, b, source, key, steps, episode, repro, observe):
    from sim.spec import INFO_FIELDS
    n, slots, goals = a.n, a.first + np.arange(a.n), a.goals
    info_names = [*INFO_FIELDS, 'made', 'kills', 'facts']
    result = Result(worlds=n)
    history, recent = [[] for _ in range(n)], [deque(maxlen=HISTORY) for _ in range(n)]
    seen = [len(a.evidence), len(b.evidence)]

    def diverge(where, w, detail, step, decision, tick, ep, new, cut=None):
        """new: the divergence lies in the episode this step began (its setup), whose repro replays no action; cut:
        (each side's Rows of this call, the divergent Row's index) to end the tails there."""
        actions = [] if new else list(history[w])
        tails = {side.name: side.tail(w) if cut is None else _until(side.tail(w), rows, cut[2])
                 for side, rows in zip((a, b), cut[:2] if cut else (None, None))}
        result.divergence = Divergence(where, w, int(slots[w]), key, goals[w], ep + new, step, 0 if new else decision,
                                       tick, detail, list(recent[w]), tails,
                                       repro(int(slots[w]), ep + new, actions) if repro else '')
        return result

    def check(step, out_a, out_b, before, done):
        rows_a, rows_b = a.take(), b.take()
        traced = rows_a[0] is not None and rows_b[0] is not None
        result.traced = result.traced or traced
        new_a, new_b = a.evidence[seen[0]:], b.evidence[seen[1]:]
        seen[:] = len(a.evidence), len(b.evidence)
        for w in range(n):
            ep, decision = before[w]
            tick = rows_a[w][-1].tick if traced and rows_a[w] else -1
            if traced:
                result.ticks += len(rows_a[w])
                found = _compare_rows(rows_a[w], rows_b[w])
                if found:
                    where, detail, k = found
                    row = (rows_a[w] or rows_b[w])[min(k, len(rows_a[w] or rows_b[w]) - 1)]
                    return diverge(where, w, detail, step, decision, row.tick, ep, int(row.episode != ep),
                                   (rows_a[w], rows_b[w], k))
            found = _compare_outputs(w, out_a, out_b, info_names)
            if found:  # a done world's observation is its next episode's first
                return diverge(found[0], w, found[1], step, decision, tick, ep,
                               int(found[0] == 'observation' and bool(done[w])))
            ea, eb = _evidence(new_a, int(slots[w])), _evidence(new_b, int(slots[w]))
            if ea != eb:
                return diverge('evidence', w, [f'evidence rows {len(ea)} vs {len(eb)}', *map(str, ea + eb)][:4], step,
                               decision, tick, ep, 0)
            if done[w]:
                sa, sb = _plain(a.starts[w]), _plain(b.starts[w])
                if sa != sb:
                    keys = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
                    return diverge('start', w, [f'start differs in {keys}: {[sa.get(k) for k in keys]} vs '
                                                f'{[sb.get(k) for k in keys]}'], step, decision, tick, ep, int(step >= 0))
        return None

    def both(call_a, call_b, step, before):
        """Both sides' outputs, or a Result: stopped when both raised alike, diverged when they did not."""
        (out_a, err_a), (out_b, err_b) = call_a(), call_b()
        if err_a is None and err_b is None:
            return out_a, out_b, None
        gap = next((f'{side.name}: {err}' for side, err in ((a, err_a), (b, err_b)) if isinstance(err, native.NotPorted)),
                   None)
        if gap:
            result.gap = f'{gap} (step {step})'
            return None, None, result
        alike = raised_alike(err_a, err_b)
        texts = [f'{side.name}: {"no exception" if err is None else f"{type(err).__name__}: {err}"}'
                 for side, err in ((a, err_a), (b, err_b))]
        if alike == 'cut':  # the detail the C text leaves out lies in the step's tick rows: they must agree, up to the
            rows_a, rows_b = a.take(), b.take()  # world the C side names (later worlds may not have stepped)
            if rows_a[0] is not None and rows_b[0] is not None:
                last = getattr(err_b, 'world', n - 1)
                last = min(n - 1, last if isinstance(last, int) and last >= 0 else n - 1)
                # the oracle steps its worlds in order and stops at its raise: rows past C's world mean it raised later
                if any(rows_a[w] for w in range(last + 1, n)):
                    return None, None, diverge('exception', 0, texts, step, before[0][1], -1, before[0][0], 0)
                for w in range(last + 1):
                    found = _compare_rows(rows_a[w], rows_b[w])
                    if found:
                        where, detail, k = found
                        row = (rows_a[w] or rows_b[w])[min(k, len(rows_a[w] or rows_b[w]) - 1)]
                        ep, decision = before[w]
                        return None, None, diverge(where, w, detail, step, decision, row.tick, ep,
                                                   int(row.episode != ep), (rows_a[w], rows_b[w], k))
            result.stopped = f'both sides raised {type(err_a).__name__} at step {step}: {err_a} (C text: {err_b})'
            return None, None, result
        if alike:
            result.stopped = f'both sides raised {type(err_a).__name__} at step {step}: {err_a}'
            return None, None, result
        a.take(), b.take()
        return None, None, diverge('exception', 0, texts, step, before[0][1], -1, before[0][0], 0)

    a.take(), b.take()  # Rows from before this run's reset are not its
    out_a, out_b, stop = both(lambda: _call(a.reset, key), lambda: _call(b.reset, key), -1, [(0, 0)] * n)
    if stop:
        return stop
    if episode:
        a.take(), b.take()
        out_a, out_b, stop = both(lambda: _call(a.start_episode, 0, episode),
                                  lambda: _call(b.start_episode, 0, episode), -1, [(episode, 0)] * n)
        if stop:
            return stop
    reset = lambda out: (*out, None, None, None)
    found = check(-1, reset(out_a), reset(out_b), [(e, 0) for e in a.episodes], np.ones(n, bool))
    if found:
        return found
    if observe is not None:
        observe(-1, out_a, out_b)
    (state_a, obs), state_b, starts = out_a, out_b[0], np.ones(n, bool)
    for step in range(steps):
        a.step_index = b.step_index = step
        before = list(zip(a.episodes, [d + 1 for d in a.decisions]))
        action = np.asarray(source(step, obs, slots, starts), np.int32)
        for w in range(n):
            history[w].append(int(action[w]))
            recent[w].append(int(action[w]))
        out_a, out_b, stop = both(lambda: _call(a.step, state_a, action), lambda: _call(b.step, state_b, action), step,
                                  before)
        if stop:
            return stop
        result.steps, done = step + 1, np.asarray(out_a[3], bool)
        found = check(step, out_a, out_b, before, done)
        if found:
            return found
        if observe is not None:
            observe(step, out_a, out_b)
        (state_a, obs, _, _, info), state_b = out_a, out_b[0]
        result.episodes += int(done.sum())
        for w in np.flatnonzero(done):
            history[w] = []
        starts = done
        if np.asarray(info['unsupported']).any():
            w = int(np.flatnonzero(info['unsupported'])[0])
            result.stopped = f'world slot {int(slots[w])} reached unsupported content at step {step}, alike on both sides'
            return result
    return result


# ---- inputs, the command line and repros ---------------------------------------------------------------------------
@dataclass
class Inputs:
    """What a diff is built from (and its repro line names)."""
    pool: str
    archive: str | None = None
    supply: str | None = None
    supply_count: int = 0
    task: str = 'goal-activity-v2'
    goals: str = 'all'
    horizon: int | None = None
    scenario: str | None = None
    against: str = 'cpu'
    pack: str | None = None
    # free-play-v1's batch settings (sim/vec.py CpuBatch): the XP rate, muggers and the life length as JSON text
    xp_rate: int | None = None
    muggers: bool | None = None
    life: str | None = None

    @property
    def free(self):
        return self.task == native.FREE_TASK

    def batch_settings(self):
        """free-play-v1's CpuBatch and NativeBatch settings ({} for a goal task)."""
        if not self.free:
            return {}
        return dict(xp_rate=self.xp_rate, muggers=self.muggers, life=None if self.life is None else tuple(json.loads(self.life)))

    def goal_set(self):
        from baseline.resource_task import GOALS
        return GOALS if self.goals == 'all' else (self.goals,)

    def repro(self, key):
        """repro(slot, episode, actions) -> the one-line command that rebuilds CpuBatch(n=1, first=slot) there."""
        def line(slot, episode, actions):
            args = ['--task', self.task, '--pool', self.pool, '--key', str(key), '--slot', str(slot), '--episode',
                    str(episode), '--actions', ','.join(map(str, actions)) or 'none', '--against', self.against]
            for flag, value in (('--archive', self.archive), ('--supply', self.supply), ('--pack', self.pack),
                                ('--scenario', self.scenario), ('--horizon', self.horizon), ('--xp-rate', self.xp_rate),
                                ('--muggers', None if self.muggers is None else int(self.muggers)),
                                ('--life', self.life)):
                if value is not None:
                    args += [flag, str(value)]
            if self.supply_count:
                args += ['--supply-count', str(self.supply_count)]
            if self.goals != 'all':
                args += ['--goals', self.goals]
            return 'PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native diff ' + shlex.join(args)
        return line


def build(inputs, n, first=0, tables_=None):
    """The two sides of a diff: a CpuBatch, and its twin (against cpu) or a NativeBatch on the pack (against native)."""
    from sim.vec import CpuBatch
    from sim.workers import load_tables
    from sim.world import load_pool
    tables_ = tables_ or load_tables(inputs.archive)
    layouts, derived = load_pool(inputs.pool)
    search = None
    if inputs.supply:
        supply, none = load_pool(inputs.supply)
        count = inputs.supply_count or len(supply)
        search = (supply[:count], none[:count])
    kwargs = dict(derived=None if derived[0] is None else derived, first=first, task=inputs.task, search=search)
    if not inputs.free:
        kwargs['goals'] = inputs.goal_set()
    if inputs.horizon is not None:
        kwargs['horizon'] = inputs.horizon
    env = CpuBatch(tables_, layouts, n, **kwargs, **inputs.batch_settings())
    a = PythonSide(env, scenario=inputs.scenario, name='cpu')
    if inputs.against == 'cpu':
        return a, PythonSide(twin(env), scenario=inputs.scenario, name='cpu-twin')
    if inputs.against != 'native':
        raise ValueError(f'against cpu or native, not {inputs.against}')
    from sim.build import DATA
    opened = {name: path for name, path in (('tables', inputs.archive or DATA), ('pool', inputs.pool),
                                            ('supply_pool', inputs.supply)) if path}  # as the training workers open it
    batch = native.NativeBatch(inputs.pack, n, first=first, task=inputs.task, inputs=opened,
                               **{k: v for k, v in kwargs.items() if k in ('goals', 'horizon')}, **inputs.batch_settings())
    a.canon = Canon(PackStrings(batch.pack))  # the pack's string ids, as canon.c writes them
    return a, NativeSide(batch, scenario=inputs.scenario, python=a)


def add_arguments(parser):
    parser.add_argument('--pool', required=True, help='the compact pool (sim.world pool file)')
    parser.add_argument('--archive', help='the simulator archive (sim/data.py), loaded with check_pins=False')
    parser.add_argument('--supply', help='the supply pool of goal-activity-v5 and v11')
    parser.add_argument('--supply-count', type=int, default=0, help='use the first N supply layouts (0: all)')
    parser.add_argument('--task', default='goal-activity-v2')
    parser.add_argument('--goals', default='all', help='all, or one goal in every slot')
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--xp-rate', type=int, help='free-play-v1: the worlds\' XP rate (1 or 25)')
    parser.add_argument('--muggers', type=int, choices=(0, 1), help='free-play-v1: muggers placed (1) or not (0)')
    parser.add_argument('--life', help='free-play-v1: the life length as JSON, e.g. \'["uniform", 200, 600]\'')
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--key', type=int, default=7)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--source', default='random', help='random, biased, idle, walk, eat, list:1,2,3, policy:PATH[:seed] or '
                        '(free-play-v1) control:NAME[,NAME...] (sim/free.py controls)')
    parser.add_argument('--period', type=int, default=None, help="control:rotation's ticks a member")
    parser.add_argument('--scenario', help=f'a scenario start: {", ".join(SCENARIOS)}, or a JSON object')
    parser.add_argument('--against', default='cpu', help='cpu (a twin CpuBatch) or native (a NativeBatch on --pack)')
    parser.add_argument('--pack')
    parser.add_argument('--slot', type=int, help='repro: world slot (n=1, first=slot)')
    parser.add_argument('--episode', type=int, default=0, help='repro: the episode to rebuild')
    parser.add_argument('--actions', help="repro: the episode's logged actions, comma separated (or 'none')")


def main(args):
    """`python -m sim.native diff` (sim/native.py main): prints a summary line, and the report of a divergence (exit
    status 1); a gap (a path the native side has not ported) exits 2."""
    if args.scenario:
        Scenario(args.scenario)  # refuse an unknown one before loading anything
    inputs = Inputs(pool=args.pool, archive=args.archive, supply=args.supply, supply_count=args.supply_count,
                    task=args.task, goals=args.goals, horizon=args.horizon, scenario=args.scenario, against=args.against,
                    pack=args.pack, xp_rate=args.xp_rate, muggers=None if args.muggers is None else bool(args.muggers),
                    life=args.life)
    if args.slot is not None:
        actions = [] if args.actions in (None, 'none') else [int(v) for v in args.actions.split(',')]
        a, b = build(inputs, 1, first=args.slot)
        result = run(a, b, Replay(actions), args.key, len(actions), episode=args.episode, repro=inputs.repro(args.key))
    else:
        a, b = build(inputs, args.worlds, first=args.first)
        source = ControlSource(a, args.source[8:].split(','), args.key, args.period) if args.source.startswith('control:') \
            else make_source(args.source, args.key)
        result = run(a, b, source, args.key, args.steps, repro=inputs.repro(args.key))
    print(f'[native diff] {a.name} vs {b.name}: {result.worlds} worlds, {result.steps} decisions, '
          f'{result.ticks if result.traced else "untraced"} ticks, {result.episodes} episodes ended, '
          f'{"diverged" if result.divergence else "no divergence"}{"; " + result.stopped if result.stopped else ""}'
          f'{"; not ported: " + result.gap if result.gap else ""}')
    if result.divergence is not None:
        print(result.divergence.text())
        sys.exit(1)
    if result.gap:
        sys.exit(2)


# ---- the release gate: python -m sim.native gate [--quick] ----------------------------------------------------------
# The gate is frozen: later rounds may add cases, never remove or shrink one. Every case is the full comparison above
# (tick records with the native trace rows, draws, events, chat, texts, the observation arrays and mask bitwise, reward
# bits, done, info and the start dicts). Per configuration (task, goal set, first slot of n = GATE_N worlds) the sources
# are keyed random legal play, biased keyed play, two trained policies' sampled actions, Hypothesis action lists and
# every named scenario under biased play. A scenario runs on every configuration: where its draw is not made (combat
# draws on a goal set without combat starts, supply draws without supply starts) it plays the biased case's starts, but
# through the native side's external starts (NativeSide). Added on 6 November: the danger start under idle, walking and
# eating play (UNDER_ATTACK), sized as a scenario. Beside the cases: FOUR_GOAL_DIGESTS through the workers path,
# thread-count invariance, and the native soak whose sampled episodes are replayed in Python.
GATE_N = 8
GATE_TASKS = ('goal-activity-v2', 'goal-activity-v3', 'goal-activity-v5')
SINGLE_GOAL_TASKS = ('goal-activity-v5', 'goal-activity-v2')
# The policy sources: fresh obs-v3 builds (sim/learn.py build, N's initialization) from these seeds. act-v3 has no
# trained policy before its launch freeze, and check_names refuses the act-v2 checkpoints the gate played before.
FRESH_POLICIES = dict(A=101, S=202)
# (keys 1..k, decisions per run); hypothesis (examples, most decisions); threads (worlds, decisions); soak world ticks
GATE = dict(
    full=dict(firsts=(0, 8, 16), random=(8, 400), biased=(8, 600), policy=(4, 600), hypothesis=(200, 300),
              scenario=(4, 300), threads=(64, 200), soak=10 ** 8),
    quick=dict(firsts=(0,), random=(2, 50), biased=(2, 70), policy=(2, 50), hypothesis=(20, 300), scenario=(2, 30),
               threads=(64, 40), soak=0),
)
# free-play-v1's configurations (the gate's `goals` column names them): the XP rate, muggers and the life length.
# Short lives end by their length and, with muggers or a hurt start, by a death within a case; the long ones (full
# gate only) play whole 36,000-tick lives, past RS_MAX_EVENTS events a life, so their settlement runs in steps, at both
# XP rates (the 25x one added on 12 November: high levels and large XP counts in long lives).
FREE_CONFIGS = {
    'x1-none': dict(xp_rate=1, muggers=False, life=('uniform', 100, 1000)),
    'x1-muggers': dict(xp_rate=1, muggers=True, life=('uniform', 100, 1000)),
    'x25-none': dict(xp_rate=25, muggers=False, life=('uniform', 100, 1000)),
    'x25-muggers': dict(xp_rate=25, muggers=True, life=('uniform', 100, 1000)),
    'x1-none-long': dict(xp_rate=1, muggers=False, life=('fixed', 36000)),
    'x25-none-long': dict(xp_rate=25, muggers=False, life=('fixed', 36000)),
}
# village-world-v2's configurations, an opt-in world set of its own (gate --world village-world-v2; added 13 November):
# the village-world-v1 set above (the goal tasks and FREE_CONFIGS: the gate's default) and its recorded results do not
# change. v2 holds no muggers, and the C engine plays no goal task on its layouts, so the set is free play alone. No name
# holds a v1 name, so an --only pattern of the v1 set matches no v2 case.
FREE_CONFIGS_V2 = {
    'w2-x1': dict(xp_rate=1, muggers=False, life=('uniform', 100, 1000)),
    'w2-x25': dict(xp_rate=25, muggers=False, life=('uniform', 100, 1000)),
    'w2-x1-long': dict(xp_rate=1, muggers=False, life=('fixed', 36000)),
    'w2-x25-long': dict(xp_rate=25, muggers=False, life=('fixed', 36000)),
}
FREE_ALL = {**FREE_CONFIGS, **FREE_CONFIGS_V2}  # every free-play configuration by name
FREE_LONG = ('x1-none-long', 'x25-none-long', 'w2-x1-long', 'w2-x25-long')
GATE_WORLDS = ('village-world-v1', 'village-world-v2')  # the gate's world sets; the first is the default
FREE_SOURCES = ('random', 'biased', 'hypothesis')  # obs-v6 has no trained or fresh policy on this line
# The long lives' play: sim/free.py controls (ControlSource), whose fights fill a life with events (the chickens
# control passes 4,096 by about tick 8,200), the rotation at one of the soak's declared periods (not the study's).
FREE_LONG_CONTROLS, ROTATION_PERIOD = ('chickens', 'rotation'), 1500
GATE['full'].update(long=(1, 7000), free_soak=2 * 10 ** 7)
GATE['quick'].update(long=(0, 0), free_soak=0)
# village-world-v2's training-workers part (gate_workers; added 13 November): worlds and decisions a run, as the digests
# part checks the goal tasks' workers path for village-world-v1
GATE['full'].update(workers=(32, 250))
GATE['quick'].update(workers=(8, 100))
WORKERS_CONFIGS, WORKERS_COUNTS, WORKERS_KEY = ('w2-x1', 'w2-x25'), (1, 2, 4), 1
PARTS = ('cases', 'digests', 'threads', 'soak', 'workers', 'policy')
SOAK_N, SOAK_KEY, SOAK_EVERY = 128, 100_000, 1000  # soak worlds per batch, first key, 1 in SOAK_EVERY episodes replayed
SOAK_SOURCES = ('biased', 'policy:A', 'policy:S')
SOAK_FAILURES = 50  # a soak stops resetting after this many errors and unsupported worlds (each one is reported)
# The compact pool FOUR_GOAL_DIGESTS' v5 batch was recorded on (sim/tests/test_workers.py v5_inputs): the digests part
# reads it (gate --digest-pool's default) whatever the gate's --pool, so a gate on another pool (an earlier study's
# layouts arm) can pass that part. A checkout without logs/ (a remote host's) names its uploaded copy.
DIGEST_POOL = native.ROOT / 'logs' / 'sim' / 'world' / 'train-8-nc1-derived.npz'
# Each world set's default inputs (gate --archive, --pool, --pack): village-world-v1's are the gate's as before;
# village-world-v2's are its training pool (plain), its archive with the pick rows and their pack without trees.
GATE_INPUTS = dict(archive=native.ROOT / 'logs/sim/data/draynor-nc1.npz',
                   pool=native.ROOT / 'logs/sim/world/train-8-nc1-derived.npz', pack=native.PACK)
GATE_INPUTS_V2 = dict(archive=native.ROOT / 'logs/sim/data/draynor-nc2.npz',
                      pool=native.ROOT / 'logs/sim/world/train-128-20000-v2.npz', pack=native.PACK_V2)
# The policy part (gate_policy; added 14 November, for a learned policy's state that no other part reached): each
# world set's pinned free-play-v1 checkpoint (a file under ignored logs/, its sha256 pinned here), sampled, in both
# engines side by side under its frozen plan's batch (sim/gate.py FREE['plans']: the world, XP rate, muggers and life) on
# that plan's training pool. village-world-v2's set plays its own inputs (its pool is that plan's); village-world-v1's,
# whose cases play train-8-nc1, plays the first pilot's 128-layout pool through that pool's pack. free_policy: the first
# world slot of each run (run k plays key k + 1), worlds a run, decisions a run (full: past the longest life, 36,000 ticks
# at about 14 ticks a decision, so each world's first life ends; measured on the Mac alone at full size, 14 November:
# 688 s and 469 s, v2 and v1, 105 and 154 world decisions a second, loads included).
GATE_POLICY = {
    GATE_WORLDS[0]: dict(plan='pilot', checkpoint=native.ROOT / 'logs/native/policies/pilot-run77-latest.zip',
                         sha256='cc77f60ff61ed93a09f545a72a688f3e63bcbc639731d5f214c451081be4bb48',
                         pool=native.ROOT / 'logs/sim/world/train-128-1000-nc1.npz',
                         pack=native.ROOT / 'logs/native/train-128-1000-nc1-a6c325a-ids3-item3.pack'),
    GATE_WORLDS[1]: dict(plan='wider_pilot',
                         checkpoint=native.ROOT / 'logs/native/policies/wider-pilot-s79-a1-ppo_1572864_steps.zip',
                         sha256='ffd81cf49e79d3ac336da55fee5ce721b110df119a0d0d6c8e0b43c78b60fabb'),
}
GATE['full'].update(free_policy=((0, 8, 16), 8, 3000))
GATE['quick'].update(free_policy=((0,), 4, 200))
# The respawn-cycle life (14 November): the wider pilot's update-24 checkpoint, sampled on the C engine (key 81, torch
# seed 4, 1,024 worlds), reached it after 5,695,488 decisions; replayed, both engines raise at its last decision (an
# agent's hit in its npc's respawn cycle, before [ai_spawn], owed XP by the check). The policy part of
# village-world-v2's set plays it on that set's training pool: it must play every action with no stop, no divergence and
# no gap.
RESPAWN_LIFE = dict(world=GATE_WORLDS[1], key=81, slot=295, episode=7, actions=(
    27, 94, 87, 79, 68, 1, 91, 8, 2, 28, 20, 6, 5, 1, 10, 14, 14, 8, 1, 8, 1, 7, 13, 3, 10, 0, 9, 10, 7, 1, 1, 7, 9,
    2, 15, 10, 2, 3, 1, 7, 16, 0, 9, 14, 1, 6, 1, 14, 10, 8, 0, 15, 6, 2, 2, 61, 3, 3, 95, 7, 2, 0, 61, 2, 5, 1, 2,
    2, 44, 3, 0, 0, 6, 12, 25, 68, 3, 40, 5, 61, 38, 37, 5, 1, 42, 39, 4, 44, 6, 40, 43, 4, 62, 8, 24, 42, 68, 3, 0,
    78, 7, 1, 11, 0, 9, 3, 13, 7, 40, 14, 7, 1, 8, 13, 5, 12, 4, 11, 1, 1, 15, 0, 61, 7, 12, 8, 6, 0, 0, 37, 13, 13,
    9, 15, 1, 4, 13, 8, 16, 3, 11, 0, 13, 13, 9, 1, 10, 4, 1, 1, 3, 0, 78, 0, 2, 15, 39, 38, 2, 5, 4, 5, 4, 8, 61,
    37, 40, 39, 5, 38, 4, 3, 38, 13, 61, 38, 40, 2, 61, 38, 6, 40, 39, 3, 37, 40, 4, 1, 75, 91, 9, 1, 12, 73, 1, 9,
    10, 1, 6, 10, 5, 37, 11, 90, 9, 0, 8, 2, 1, 72, 15, 13, 14, 2, 75, 0, 0, 77, 1, 0, 15, 16, 18, 5, 5, 6, 0, 13,
    1, 15, 13, 1, 1, 15, 12, 16, 14, 2, 14, 16, 9, 1, 16, 4, 1, 14, 0, 73, 1, 0, 10, 1, 5, 4, 6, 0, 1, 38, 3, 0, 0,
    1, 2, 3, 4, 0, 7, 0, 10, 2, 4, 0, 4, 12, 5, 16, 1, 2, 4, 5, 0, 7, 76, 9, 0, 1, 76, 3, 5, 76, 5, 1, 3, 2, 8, 12,
    12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61))


def gate_parts(mode, world=GATE_WORLDS[0]):
    """The part rows (besides the cases) a world set's gate writes in a mode: village-world-v1's digests (its goal tasks'
    workers path), village-world-v2's training workers (gate_workers), the thread check, the soak in the full gate (the
    quick gate has none) and the policy part (gate_policy), in both modes and both sets."""
    if world not in GATE_WORLDS:
        raise ValueError(f'the gate plays the world sets {GATE_WORLDS}, not {world!r}')
    return ['digests' if world == GATE_WORLDS[0] else 'workers', 'threads', *(['soak'] if GATE[mode]['soak'] else []),
            'policy']


def gate_configs(mode, world=GATE_WORLDS[0]):
    """(task, goals, first) of every configuration of a world set, in run order; free-play-v1's goals column is its
    FREE_CONFIGS (FREE_CONFIGS_V2) name (its long lives in the full gate only). village-world-v2's set is its free-play
    configurations alone."""
    firsts = GATE[mode]['firsts']
    from baseline.resource_task import GOALS
    if world not in GATE_WORLDS:
        raise ValueError(f'the gate plays the world sets {GATE_WORLDS}, not {world!r}')
    if world != GATE_WORLDS[0]:
        return [(native.FREE_TASK, name, first) for name in FREE_CONFIGS_V2
                if name not in FREE_LONG or GATE[mode]['long'][0] for first in firsts]
    free = [name for name in FREE_CONFIGS if name not in FREE_LONG or GATE[mode]['long'][0]]
    return [(task, 'all', first) for task in GATE_TASKS for first in firsts] + \
        [(task, goal, first) for task in SINGLE_GOAL_TASKS for goal in GOALS for first in firsts] + \
        [(native.FREE_TASK, name, first) for name in free for first in firsts]


def gate_cases(task=None, goals=None):
    """(source, scenario) of a configuration's cases: the goal tasks' list; free-play-v1's keyed and Hypothesis play,
    its scenarios under biased play and the hurt start under attack (UNDER_ATTACK: with muggers the attacker comes),
    and on a long-life configuration keyed random play alone."""
    if task == native.FREE_TASK:
        if goals in FREE_LONG:
            return [(f'control:{name}', None) for name in FREE_LONG_CONTROLS]
        return [*((source, None) for source in FREE_SOURCES), *(('biased', name) for name in FREE_SCENARIOS),
                *((source, 'freehurt') for source in UNDER_ATTACK)]
    cases = [('random', None), ('biased', None), *((f'policy:{name}', None) for name in FRESH_POLICIES), ('hypothesis', None)]
    return cases + [('biased', name) for name in SCENARIOS] + [(source, 'danger') for source in UNDER_ATTACK]


def case_id(task, goals, first, source, scenario):
    return f'{task}|{goals}|{first}|{source}|{scenario or "-"}'


def retarget(env, first, goals=None):
    """A CpuBatch like `env` whose world 0 is global slot `first` (and goal set `goals`), sharing env's layout tables
    (twin): CpuBatch.__init__ derives only first and goals from those arguments."""
    from baseline.resource_task import GOALS
    from sim.spec import goal_set, slot_goal
    other = twin(env)
    declared = goal_set(GOALS if goals in (None, 'all') else (goals,))
    other.first, other.goals = first, [slot_goal(declared, first + b) for b in range(env.n)]
    return other


def world_digest(obs, reward, done, info, b):
    """sha256 of world b's row of a step's outputs (every observation array bitwise, reward bits, done, info)."""
    import hashlib
    from sim.spec import INFO_FIELDS
    h = hashlib.sha256()
    for name in obs:  # the batch's table, in its order (obs-v5, or free play's obs-v6)
        h.update(np.ascontiguousarray(obs[name][b]).tobytes())
    if reward is not None:
        h.update(np.float32(reward[b]).tobytes() + bytes([bool(done[b])]))
        for name in (*INFO_FIELDS, 'made', 'kills', 'facts'):
            h.update(np.ascontiguousarray(info[name][b]).tobytes())
        for name in info.get('last_obs', {}):  # free play: each ended life's last observation
            h.update(np.ascontiguousarray(info['last_obs'][name][b]).tobytes())
    return h.hexdigest()


class FastBiased:
    """The soak's biased play: Biased's family weights over the legal actions, one numpy draw per world (seeded), so a
    large batch is sampled at once. Its actions are logged per episode; replays play the log."""

    def __init__(self, seed):
        self.weight, self.rng = Biased(0).weight, np.random.default_rng(seed)

    def __call__(self, step, obs, slots, starts):
        edges = np.cumsum(obs['mask'] * self.weight, axis=1)
        u = self.rng.random(len(edges)) * edges[:, -1]
        return np.minimum((edges <= u[:, None]).sum(1), edges.shape[1] - 1).astype(np.int32)


@dataclass
class GateContext:
    """What every part of a gate run reads, loaded once."""
    archive: str
    pool: str
    supply: str
    pack_path: str
    work: Path
    threads: int
    world: str = GATE_WORLDS[0]
    tables: object = None
    layouts: list = None
    derived: list = None
    search: tuple = None
    pack: object = None
    policies: dict = None
    bases: dict = field(default_factory=dict)

    def load(self, policies=True):
        from sim.workers import load_tables
        from sim.world import load_pool
        from sim.combat import village_of
        self.tables = load_tables(self.archive)
        self.layouts, derived = load_pool(self.pool)
        self.derived = None if derived[0] is None else derived
        found = {(village_of(layout) or {}).get('version') for layout in self.layouts}
        if found != {self.world}:  # a world set plays its own world's pool (its free-play batches name it too)
            raise ValueError(f"the gate's {self.world} set plays that world's pool, not {self.pool} "
                             f"({sorted(map(str, found))} layouts)")
        if self.world != GATE_WORLDS[0]:  # the wider world's set plays training layouts only, never held-out ones
            from sim.world import SPLITS
            lo, hi = SPLITS['train']
            outside = sorted({int(layout.seed) for layout in self.layouts if not lo <= layout.seed < hi})
            if outside:
                raise ValueError(f"the gate's {self.world} set plays layouts of the train split only (seeds {lo} to "
                                 f"{hi - 1}): {self.pool} holds {len(outside)} others, {outside[:5]}")
        supply, none = load_pool(self.supply)
        self.search = (supply, none)
        self.pack = native.open_pack(self.pack_path, inputs=dict(tables=self.archive, pool=self.pool,
                                                                 supply_pool=self.supply))
        self.policies = self.load_policies() if policies else {}
        return self

    def load_policies(self):
        """Each fresh policy's actor (FRESH_POLICIES: sim/learn.py build from its seed, as sim.train builds N's), after
        checking it reads obs-v3 through this name table under act-v3's contract (check_names); the mask is the env's,
        applied as given."""
        import torch
        from baseline.policy import check_names
        from sim.learn import build
        from sim.train import make_env
        torch.set_num_threads(1)
        out = {}
        for name, seed in FRESH_POLICIES.items():
            model = build(make_env('fake', 4, seed, 'cpu'), seed=seed, device='cpu')
            check_names(model.policy)
            out[name] = (model.policy, model.observation_space.spaces, f'{model.policy.observation}/fresh-{seed}')
        return out

    def base(self, task, goals='all'):
        """One CpuBatch of GATE_N worlds per task (free-play-v1: per configuration; its layout tables built once;
        retarget gives the others)."""
        from sim.combat import task_spec
        from sim.vec import CpuBatch
        k = (task, goals) if task == native.FREE_TASK else task
        if k not in self.bases:
            if task == native.FREE_TASK:
                self.bases[k] = CpuBatch(self.tables, self.layouts, GATE_N, derived=self.derived, task=task,
                                         world=self.world, **FREE_ALL[goals])
            else:
                self.bases[k] = CpuBatch(self.tables, self.layouts, GATE_N, derived=self.derived, task=task,
                                         search=self.search if task_spec(task)['search'] else None)
        return self.bases[k]

    def inputs(self, task, goals, scenario=None):
        if task == native.FREE_TASK:
            c = FREE_ALL[goals]
            return Inputs(pool=self.pool, archive=self.archive, task=task, scenario=scenario, against='native',
                          pack=self.pack_path, xp_rate=c['xp_rate'], muggers=c['muggers'], life=json.dumps(c['life']))
        from sim.combat import task_spec
        supply = task_spec(task)['supply']
        return Inputs(pool=self.pool, archive=self.archive, supply=self.supply if supply else None, task=task, goals=goals,
                      scenario=scenario, against='native', pack=self.pack_path)

    def batch(self, task, goals, n, first=0, threads=1):
        """The configuration's NativeBatch on the gate's pack."""
        if task == native.FREE_TASK:
            return native.NativeBatch(self.pack, n, first=first, task=task, threads=threads, world=self.world,
                                      **FREE_ALL[goals])
        return native.NativeBatch(self.pack, n, first=first, task=task, goals=self.inputs(task, goals).goal_set(),
                                  threads=threads)

    def source(self, spec, key):
        if spec.startswith('policy:'):
            policy, spaces, _ = self.policies[spec[7:]]
            return PolicySource(policy, spaces, seed=key)
        return make_source(spec, key)


@dataclass
class CaseResult:
    """One case of the gate: its runs summed (decisions are steps of GATE_N worlds; world decisions count each world),
    the first-divergence report of each failed run, and the runs that stopped alike (an unsupported world)."""
    id: str
    task: str
    goals: str
    first: int
    source: str
    scenario: str | None
    runs: int = 0
    decisions: int = 0
    world_decisions: int = 0
    ticks: int = 0
    episodes: int = 0
    divergences: int = 0
    gaps: int = 0
    stopped: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    seconds: float = 0.0
    most_objs: int = 0  # the most ground objs one Python world held at a tick's end over the runs (Result.most_objs)
    compared: int = 0  # decisions both sides' outputs compared equal at (compared_steps), over the runs
    census: dict | None = None  # free play: what its Python lives reached (FreeCensus), over the runs
    deaths: int | None = None  # free play: lives that ended by a death (not by their length), over the runs

    def add(self, result, key, compared=None):
        """A run's counts (compared: its compared decisions, default compared_steps(result))."""
        self.runs += 1
        self.most_objs = max(self.most_objs, result.most_objs)
        self.compared += compared_steps(result) if compared is None else compared
        self.decisions += result.steps
        self.world_decisions += result.steps * result.worlds
        self.ticks += result.ticks
        self.episodes += result.episodes
        if result.stopped:
            self.stopped.append(f'key {key}: {result.stopped}')
        if result.gap:
            self.gaps += 1
            self.reports.append(f'key {key}: not ported: {result.gap}')
        if result.divergence is not None:
            self.divergences += 1
            self.reports.append(result.divergence.text())

    def line(self):
        return (f'[gate] {self.task} {self.goals} first {self.first} {self.source} {self.scenario or "-"}: {self.runs} '
                f'runs, {self.decisions} decisions ({self.world_decisions} world, {self.compared} compared), {self.ticks} '
                f'ticks, {self.episodes} episodes ended{"" if self.deaths is None else f" ({self.deaths} deaths)"}, '
                f'{self.divergences} divergences, {self.gaps} gaps, at most {self.most_objs} ground objs'
                f'{f", {len(self.stopped)} stopped alike" if self.stopped else ""} ({self.seconds:.0f} s)')


def compared_steps(result):
    """The decisions a run compared equal on both sides: its steps, less the one a divergence was found at (a divergence
    at the reset, step -1, leaves none). A run stopped alike compared the steps before its stop (an unsupported world's
    step included: its outputs compared equal); a gap's run, those before the gap."""
    if result.divergence is None:
        return result.steps
    return max(0, min(result.steps, result.divergence.step))


def gate_case(ctx, env, batch, canon, mode, task, goals, first, source, scenario, log):
    """One case on this configuration's batches: fresh sides around them, every key (or Hypothesis example) run."""
    import time
    g, t0 = GATE[mode], time.time()
    out = CaseResult(case_id(task, goals, first, source, scenario), task, goals, first, source, scenario)
    inputs = ctx.inputs(task, goals, scenario)
    census, deaths = (FreeCensus(ctx.tables), [0]) if task == native.FREE_TASK else (None, None)

    def observe(step, out_a, out_b):
        """Free play's coverage, read off the Python side once both sides' outputs compared equal: each life's census
        and the lives that ended by a death."""
        if step >= 0:
            done, terminal = np.asarray(out_a[3], bool), np.asarray(out_a[4]['terminal'])
            deaths[0] += int((done & (terminal != 1)).sum())
        for w, life in enumerate(env.tasks):
            census.read(w, life)
    observing = observe if census is not None else None

    def sides():
        a = PythonSide(env, canon=canon, scenario=scenario)
        return a, NativeSide(batch, scenario=scenario, python=a)
    if source == 'hypothesis':
        from hypothesis import HealthCheck, given, settings, strategies as st
        examples, most = g['hypothesis']
        a, b = sides()

        @settings(max_examples=examples, derandomize=True, database=None, deadline=None, report_multiple_bugs=False,
                  suppress_health_check=list(HealthCheck), print_blob=False)
        @given(key=st.integers(-2 ** 31, 2 ** 31 - 1), values=st.lists(st.integers(0, 95), min_size=1, max_size=most))
        def example(key, values):
            result = run(a, b, Listed(values), key, len(values), repro=inputs.repro(key), observe=observing)
            failed = result.divergence is not None or bool(result.gap)
            last[:] = [(result, key)] if failed else last
            out.add(Result(result.steps, result.worlds, result.ticks, result.episodes, result.traced, None,
                           result.stopped, '', result.most_objs), key,
                    compared_steps(result))  # every example's counts; the failure once below
            assert not failed, 'a divergence or a gap'
        last = []
        try:
            example()
        except AssertionError:
            pass
        if last:  # Hypothesis replays the shrunk failing example last
            result, key = last[0]
            out.add(Result(divergence=result.divergence, gap=result.gap), key)
            out.runs -= 1
    else:
        keys, steps = g['scenario' if scenario else 'long' if source.startswith('control:') else source.split(':')[0]]
        for key in range(1, keys + 1):
            a, b = sides()
            play = ControlSource(a, source[8:].split(','), key, ROTATION_PERIOD) if source.startswith('control:') \
                else ctx.source(source, key)
            out.add(run(a, b, play, key, steps, repro=inputs.repro(key), observe=observing), key)
    if census is not None:
        out.census, out.deaths = dict(census.counts), deaths[0]
    out.seconds = time.time() - t0
    log(out.line())
    for report in out.reports:
        log(report)
    return out


def gate_cases_part(ctx, mode, shard, skip, emit, log):
    """Every configuration of this shard (index % k == i), each case once (skip(id): leave it out)."""
    import gc
    i, k = shard
    for index, (task, goals, first) in enumerate(gate_configs(mode, ctx.world)):
        if index % k != i:
            continue
        todo = [(src, sc) for src, sc in gate_cases(task, goals) if not skip(case_id(task, goals, first, src, sc))]
        if not todo:
            continue
        free = task == native.FREE_TASK
        env = retarget(ctx.base(task, goals), first, None if free else goals)
        batch = ctx.batch(task, goals, GATE_N, first=first)
        canon = Canon(PackStrings(ctx.pack))
        for source, scenario in todo:
            emit(gate_case(ctx, env, batch, canon, mode, task, goals, first, source, scenario, log))
        del env, batch
        gc.collect()


# ---- the gate's other parts --------------------------------------------------------------------------------------
@contextlib.contextmanager
def synthetic_names(work):
    """The test suite's synthetic name table (conftest.py ROWS), under which FOUR_GOAL_DIGESTS were recorded."""
    import os
    import conftest
    import names
    path = work / 'synthetic-names.npz'
    kinds, rows = zip(*conftest.ROWS)
    vectors = np.random.default_rng(0).normal(size=(len(conftest.ROWS), 8)).astype(np.float16)
    np.savez(path, kind=np.array(kinds), name=np.array(rows), vector=vectors)
    saved = os.environ.get('RL_NAME_TABLE')
    os.environ['RL_NAME_TABLE'] = str(path)
    names.name_table.cache_clear()
    try:
        yield path
    finally:
        if saved is None:
            os.environ.pop('RL_NAME_TABLE')
        else:
            os.environ['RL_NAME_TABLE'] = saved
        names.name_table.cache_clear()


def _gate_pack(path, pool, supply, archive, log):
    """The pack of these inputs at `path`, exported again unless it opens with their hashes and this name table."""
    from sim.native_export import export
    try:
        native.open_pack(path, inputs=dict(tables=archive, pool=pool, supply_pool=supply))
        return path
    except (OSError, ValueError, RuntimeError, native.load().Error) as why:
        log(f'[gate] exporting {path.name} ({type(why).__name__})')
    export(path, pool, supply, archive, log=lambda line: None)
    return path


def gate_digests(ctx, log, pool=DIGEST_POOL):
    """sim/tests/test_workers.py FOUR_GOAL_DIGESTS from NativeBatch through the workers path (CpuWorkers engine='native',
    one worker process): the v2 workers run on the suite's six development layouts (horizon 40) and the v5 batch on
    `pool` (gate --digest-pool: DIGEST_POOL, the training pool the digest was recorded on, whatever the gate's --pool),
    with the first four train supply layouts, both under the synthetic name table. The packs are exported from pool
    files holding those layouts (the exporter needs a compact pool's derived tables; the digest is of the outputs
    only)."""
    import time
    from sim import search, world as W
    from sim.tests.test_workers import FOUR_GOAL_DIGESTS, digest, run as keyed_run
    from sim.workers import CpuWorkers
    t0, out = time.time(), dict(id='digests', got={}, expected=dict(FOUR_GOAL_DIGESTS))
    with synthetic_names(ctx.work):
        dev, supply4 = ctx.work / 'digest-dev6.npz', ctx.work / 'digest-supply4.npz'
        if not dev.exists():
            layouts = [W.generate(seed, ctx.tables) for seed in W.split_seeds('development', 6)]
            W.save_pool(dev, layouts, [W.derive(layout, ctx.tables) for layout in layouts])
        if not supply4.exists():
            W.save_pool(supply4, search.supply_layouts('train', 4, ctx.tables)[0], version=search.SUPPLY['world'])
        pack_v2 = _gate_pack(ctx.work / 'digest-v2.pack', dev, supply4, ctx.archive, log)
        pack_v5 = _gate_pack(ctx.work / 'digest-v5.pack', Path(pool), supply4, ctx.archive, log)
        for name, kwargs, key, steps in (
                ('v2_workers', dict(pool=dev, horizon=40, pack=pack_v2), 5, 50),
                ('v5_batch', dict(pool=Path(pool), task='goal-activity-v5', search_pool=supply4, pack=pack_v5), 3, 60)):
            workers = CpuWorkers(8, 1, tables=ctx.archive, engine='native', **kwargs)
            try:
                out['got'][name] = digest(keyed_run(workers, key, steps))
            finally:
                workers.close()
    out['failures'] = sorted(k for k, v in out['expected'].items() if out['got'].get(k) != v)
    out['pool'] = dict(path=str(Path(pool).absolute()), sha256=native.sha256_file(pool))  # the v5 batch's (--digest-pool)
    out['seconds'] = time.time() - t0
    log(f'[gate] digests: {", ".join(f"{k} {"equal" if k not in out["failures"] else "DIFFERS"}" for k in out["expected"])} '
        f'({out["seconds"]:.0f} s)')
    return out


def gate_threads(ctx, mode, log):
    """NativeBatch with 1, 3 and 8 threads on the same worlds and actions: every output (observations, reward, done,
    info), the starts and the evidence equal bitwise at every decision (biased play, key 1, v2 and v5)."""
    import hashlib
    import time
    n, steps = GATE[mode]['threads']
    t0, out = time.time(), dict(id='threads', worlds=n, decisions=steps, failures=[])
    tasks = (('goal-activity-v2', 'all'), ('goal-activity-v5', 'all'), (native.FREE_TASK, 'x1-muggers')) \
        if ctx.world == GATE_WORLDS[0] else ((native.FREE_TASK, 'w2-x1'),)
    for task, goals in tasks:
        runs = {}
        for threads in (1, 3, 8):
            batch, source = ctx.batch(task, goals, n, threads=threads), Biased(1)
            state, obs = batch.reset(1)
            rows = [world_digest(obs, None, None, None, b) for b in range(n)]
            for step in range(steps):
                state, obs, reward, done, info = batch.step(state, source(step, obs, range(n), None))
                rows.append(hashlib.sha256(''.join(world_digest(obs, reward, done, info, b) for b in range(n)).encode()
                                           + json.dumps(_plain(batch.starts)).encode()).hexdigest())
            rows.append(json.dumps(_plain(batch.evidence), sort_keys=True))
            runs[threads] = rows
        for threads in (3, 8):
            bad = [k for k, (x, y) in enumerate(zip(runs[1], runs[threads])) if x != y]
            if bad:
                out['failures'].append(f'{task}: {threads} threads differ from 1 at row {bad[0]} ({len(bad)} rows)')
    out['seconds'] = time.time() - t0
    played = 'v2, v5 and free play' if ctx.world == GATE_WORLDS[0] else f'free play ({tasks[0][1]})'
    log(f'[gate] threads: {n} worlds, {steps} decisions, {played} at 1, 3 and 8 threads: '
        f'{"; ".join(out["failures"]) or "identical"} ({out["seconds"]:.0f} s)')
    return out


def gate_workers(ctx, mode, log):
    """village-world-v2's training workers (sim/workers.py CpuWorkers, the path sim.train collects through) on the set's
    pool, archive and pack: the C engine in 1, 2 and 4 worker processes against the Python reference in one, under keyed
    random legal play from one reset (WORKERS_KEY) on each of WORKERS_CONFIGS (lives of 100-1000 ticks, so lives end and
    begin within a run): the settings in effect equal (the world among them), and every world's outputs (world_digest:
    observations, reward, done, info, each ended life's last observation) equal at the reset and every decision. As the
    digests part checks the goal tasks' workers path for village-world-v1."""
    import time
    from sim.engine import Rng
    from sim.spec import OBSERVATIONS
    from sim.task import FREE_PLAY
    from sim.vec import R_ACTION
    from sim.workers import CpuWorkers
    n, steps = GATE[mode]['workers']
    observation = FREE_PLAY['observation']
    t0 = time.time()
    out = dict(id='workers', worlds=n, decisions=steps, configs=list(WORKERS_CONFIGS), counts=list(WORKERS_COUNTS),
               key=WORKERS_KEY, compared=0, ticks=0, episodes=0, failures=[])
    for goals in WORKERS_CONFIGS:
        runs, settings = {}, {}
        for engine, count in (('cpu', 1), *(('native', c) for c in WORKERS_COUNTS)):
            env = CpuWorkers(n, count, ctx.pool, tables=ctx.archive, task=native.FREE_TASK, goals=None, engine=engine,
                             pack=ctx.pack_path if engine == 'native' else None, observation=observation,
                             shapes=OBSERVATIONS[observation], batch=dict(world=ctx.world, **FREE_ALL[goals]))
            rows = []
            try:
                settings[engine, count] = env.settings_in_effect
                rng = Rng(WORKERS_KEY)
                state, obs = env.reset(WORKERS_KEY)
                rows.append([world_digest(obs, None, None, None, b) for b in range(n)])
                for step in range(steps):
                    legal = [np.flatnonzero(obs['mask'][b]) for b in range(n)]
                    action = np.array([row[int(rng(R_ACTION, step, b) * len(row))] for b, row in enumerate(legal)],
                                      np.int32)
                    state, obs, reward, done, info = env.step(state, action)
                    rows.append([world_digest(obs, reward, done, info, b) for b in range(n)])
                    if engine == 'cpu':
                        out['ticks'] += int(np.asarray(info['decision_ticks']).sum())
                        out['episodes'] += int(np.asarray(done).sum())
            except Exception as error:  # an unsupported world or a worker's error: the part fails with its words
                out['failures'].append(f'{goals}: {engine} in {count} workers raised at decision {len(rows) - 1}: '
                                       f'{type(error).__name__}: {error}'[:2000])
            finally:
                env.close()
            runs[engine, count] = rows
        for (engine, count), rows in runs.items():
            if engine == 'cpu':
                continue
            if settings.get((engine, count)) != settings.get(('cpu', 1)):
                out['failures'].append(f'{goals}: the C engine in {count} workers has other settings in effect '
                                       f'({settings.get((engine, count))} vs {settings.get(("cpu", 1))})')
            bad = [(k, b) for k, (x, y) in enumerate(zip(runs['cpu', 1], rows)) for b in range(n) if x[b] != y[b]]
            if len(rows) != steps + 1 or len(runs['cpu', 1]) != steps + 1:
                continue  # a run that raised (its failure above) compared no whole run
            if bad:
                out['failures'].append(f'{goals}: the C engine in {count} workers differs from the Python reference '
                                       f'first at decision {bad[0][0] - 1} world {bad[0][1]} ({len(bad)} world outputs)')
            else:
                out['compared'] += n * steps
    out['seconds'] = time.time() - t0
    log(f'[gate] workers: {n} worlds, {steps} decisions, {", ".join(WORKERS_CONFIGS)}; the C engine in '
        f'{", ".join(map(str, WORKERS_COUNTS))} workers against the Python reference in one: '
        f'{"; ".join(out["failures"]) or "identical"} ({out["episodes"]} lives ended; {out["seconds"]:.0f} s)')
    return out


def gate_policy(ctx, mode, log):
    """The world set's pinned free-play checkpoint (GATE_POLICY): its sampled actions (PolicySource, seeded by the run's
    key, one torch thread) played by both engines side by side (run, traced, as a case is) under its frozen plan's batch
    (sim/gate.py FREE['plans']) on that plan's training pool and the pool's pack, GATE[mode]['free_policy'] runs; then,
    in village-world-v2's set, the respawn-cycle life (RESPAWN_LIFE), which must play all its actions with no stop, no
    divergence and no gap. Nothing is played, and the part fails, unless the checkpoint has its pinned sha256, the plan
    names this world, the pool and archive are the plan's (its run's pool and tables sha256) and the pack opens on them
    (its oracle, exporter, ids and recorded inputs); an error raised outside the runs (a load, a build) fails the part
    with its words. The row holds the files it played with their sha256, a case's counts (compared: the decisions both
    sides compared equal, over the runs), what the Python lives reached (PolicyCensus), the lives that ended by a death,
    and the found life's run."""
    import time
    import torch
    from sim.gate import FREE as STUDY, free_plan_world
    t0, pinned = time.time(), GATE_POLICY[ctx.world]
    plan = STUDY['plans'][pinned['plan']]
    firsts, n, steps = GATE[mode]['free_policy']
    v1 = ctx.world == GATE_WORLDS[0]
    files = dict(checkpoint=pinned['checkpoint'], pool=pinned['pool'] if v1 else ctx.pool,
                 pack=pinned['pack'] if v1 else ctx.pack_path, archive=ctx.archive)
    batch = dict(xp_rate=plan['xp_rate'], muggers=plan['run']['muggers'], life=list(plan['run']['life']))
    out = dict(id='policy', plan=pinned['plan'], batch=batch, firsts=list(firsts), worlds=n, decisions=steps,
               files={k: dict(path=str(Path(v).absolute()), sha256=native.sha256_file(v) if Path(v).exists() else None)
                      for k, v in files.items()},
               runs=0, compared=0, world_decisions=0, ticks=0, episodes=0, deaths=0, divergences=0, gaps=0, stopped=[],
               reports=[], census=None, most_objs=0, found_life=None, failures=[])
    got = {k: v['sha256'] for k, v in out['files'].items()}
    for name, want in (('checkpoint', pinned['sha256']), ('pool', plan['run']['pool']), ('archive', plan['run']['tables'])):
        if got[name] != want:
            out['failures'].append(f'the {name} {files[name]} has sha256 {got[name]}, not the pinned {want}')
    if free_plan_world(plan) != ctx.world:
        out['failures'].append(f"plan {pinned['plan']} is {free_plan_world(plan)}'s, not {ctx.world}'s")
    if not out['failures']:  # the pack opens on these inputs (its oracle, exporter, ids and recorded inputs)
        try:
            native.open_pack(files['pack'], inputs=dict(tables=files['archive'], pool=files['pool']))
        except (OSError, ValueError, RuntimeError) as error:
            out['failures'].append(f"the pack {files['pack']} does not open on the plan's inputs: {error}"[:2000])
    if not out['failures']:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            _policy_play(ctx, out, files, batch, firsts, n, steps)
        except Exception as error:  # a load or build error: the part fails with its words, as the workers part does
            out['failures'].append(f'the play raised {type(error).__name__}: {error}'[:2000])
        finally:
            torch.set_num_threads(threads)
    out['seconds'] = time.time() - t0
    log(policy_line(out))
    for report in out['reports']:
        log(report)
    return out


def _policy_play(ctx, out, files, batch, firsts, n, steps):
    """gate_policy's runs and the found life, counted into its row `out`."""
    from sim.workers import load_tables
    inputs = Inputs(pool=str(files['pool']), archive=str(files['archive']), task=native.FREE_TASK, against='native',
                    pack=str(files['pack']), xp_rate=batch['xp_rate'], muggers=batch['muggers'],
                    life=json.dumps(batch['life']))
    tables = load_tables(inputs.archive)
    census, deaths = PolicyCensus(tables), [0]
    for k, first in enumerate(firsts):
        key = k + 1
        a, b = build(inputs, n, first=first, tables_=tables)

        def observe(step, out_a, out_b, a=a):
            if step >= 0:
                done, terminal = np.asarray(out_a[3], bool), np.asarray(out_a[4]['terminal'])
                deaths[0] += int((done & (terminal != 1)).sum())
            for w, life in enumerate(a.env.tasks):
                census.read(w, life)
        result = run(a, b, PolicySource.load(str(files['checkpoint']), key), key, steps, repro=inputs.repro(key),
                     observe=observe)
        out['runs'] += 1
        out['compared'] += compared_steps(result)
        out['world_decisions'] += result.steps * result.worlds
        for name in ('ticks', 'episodes'):
            out[name] += getattr(result, name)
        out['most_objs'] = max(out['most_objs'], result.most_objs)
        if result.stopped:
            out['stopped'].append(f'key {key} first {first}: {result.stopped}')
        if result.gap:
            out['gaps'] += 1
            out['reports'].append(f'key {key} first {first}: not ported: {result.gap}')
        if result.divergence is not None:
            out['divergences'] += 1
            out['reports'].append(result.divergence.text())
        del a, b
    out['census'], out['deaths'] = dict(census.counts), deaths[0]
    if ctx.world == RESPAWN_LIFE['world']:
        life = RESPAWN_LIFE
        a, b = build(inputs, 1, first=life['slot'], tables_=tables)
        result = run(a, b, Replay([[x] for x in life['actions']]), life['key'], len(life['actions']),
                     episode=life['episode'], repro=inputs.repro(life['key']))
        out['found_life'] = dict(key=life['key'], slot=life['slot'], episode=life['episode'],
                                 actions=len(life['actions']), steps=result.steps, ticks=result.ticks,
                                 stopped=result.stopped, gap=result.gap,
                                 divergence=None if result.divergence is None else result.divergence.where)
        if result.divergence is not None:
            out['divergences'] += 1
            out['reports'].append(result.divergence.text())
        if result.gap:
            out['gaps'] += 1
        if result.divergence is not None or result.gap or result.stopped or result.steps != len(life['actions']):
            out['failures'].append(
                f"the respawn-cycle life (key {life['key']}, slot {life['slot']}, episode {life['episode']}) played "
                f"{result.steps} of its {len(life['actions'])} actions: "
                f"{result.stopped or result.gap or (result.divergence and result.divergence.where) or 'cut short'}"[:2000])


def policy_line(row):
    """The policy part's line (gate_policy, gate_summary)."""
    found = row.get('found_life')
    life = '' if found is None else (f"; the respawn-cycle life: {found['steps']} of {found['actions']} actions"
                                     f"{', stopped' if found['stopped'] else ''}"
                                     f"{', diverged at ' + found['divergence'] if found['divergence'] else ''}"
                                     f"{', a gap' if found['gap'] else ''}")
    checkpoint = row['files']['checkpoint']
    return (f"[gate] policy: {row['plan']} {Path(checkpoint['path']).name} ({str(checkpoint['sha256'])[:16]}) on "
            f"{Path(row['files']['pool']['path']).name}: {row['runs']} runs, {row['world_decisions']} world decisions "
            f"({row['compared']} of {row['runs'] * row['decisions']} decisions compared), {row['ticks']} ticks, "
            f"{row['episodes']} lives ended ({row['deaths']} deaths), {row['divergences']} divergences, {row['gaps']} "
            f"gaps, {len(row['stopped'])} stopped alike{life}; "
            f"{'; '.join(row['failures']) or 'no failure'} ({row.get('seconds', 0):.0f} s)")


def soak_selected(key, slot, episode):
    """The soak's replay rule: 1 in SOAK_EVERY episodes, chosen by the crc32 of its key (key, slot, episode)."""
    return zlib.crc32(struct.pack('<qqq', int(key), int(slot), int(episode))) % SOAK_EVERY == 0


class SoakReplays:
    """Episodes of the soak replayed in Python from their start (key, slot, episode) and action log: the harness's
    run on CpuBatch(n=1, first=slot) against a traced NativeBatch(n=1), tick by tick, and the soak's own recorded
    outputs (world_digest) against the Python side's at every decision."""

    def __init__(self, ctx):
        self.ctx, self.canon = ctx, Canon(PackStrings(ctx.pack))
        self.envs = {}
        self.replayed = self.divergences = self.gaps = self.mismatches = self.most_objs = 0
        self.stopped, self.reports = [], []

    def replay(self, task, key, slot, episode, actions, recorded=None, goals='all'):
        """goals: free-play-v1's FREE_CONFIGS name (its settings), 'all' for a goal task."""
        from sim.vec import CpuBatch
        from sim.combat import task_spec
        ctx = self.ctx
        k = (task, goals)
        if k not in self.envs:
            self.envs[k] = CpuBatch(ctx.tables, ctx.layouts, 1, derived=ctx.derived, task=task, world=ctx.world,
                                    **FREE_ALL[goals]) \
                if task == native.FREE_TASK else CpuBatch(ctx.tables, ctx.layouts, 1, derived=ctx.derived, task=task,
                                                          search=ctx.search if task_spec(task)['search'] else None)
        a = PythonSide(retarget(self.envs[k], slot), canon=self.canon)
        b = NativeSide(ctx.batch(task, goals, 1, first=slot))
        differs = []

        def observe(step, out_a, out_b):
            if recorded is None or differs or step + 1 >= len(recorded):
                return
            mine = world_digest(*out_a[1:], 0) if step >= 0 else world_digest(out_a[1], None, None, None, 0)
            if mine != recorded[step + 1]:
                differs.append(step)
        result = run(a, b, Replay([[x] for x in actions]), key, len(actions), episode=episode,
                     repro=ctx.inputs(task, goals).repro(key), observe=observe)
        self.replayed += 1
        self.most_objs = max(self.most_objs, result.most_objs)
        if result.divergence is not None:
            self.divergences += 1
            self.reports.append(result.divergence.text())
        if result.gap:
            self.gaps += 1
            self.reports.append(f'{task} key {key} slot {slot} episode {episode}: not ported: {result.gap}')
        if result.stopped:
            self.stopped.append(f'{task} key {key} slot {slot} episode {episode}: {result.stopped}')
        if differs:
            self.mismatches += 1
            self.reports.append(f'{task} key {key} slot {slot} episode {episode}: the soak\'s outputs differ from the '
                                f'Python replay\'s at decision {differs[0]} (the traced rerun agreed: '
                                f'{result.divergence is None})')
        return result


def gate_soak(ctx, mode, log):
    """At least GATE[mode]['soak'] world ticks (decision ticks, setup cycles not counted) natively over v2, v3 and v5
    with biased play and both policies, SOAK_N worlds per batch on ctx.threads threads: check_batch on every output
    (NativeBatch.step), every exception and unsupported world recorded and its episode replayed in Python, and 1 in
    SOAK_EVERY episodes (soak_selected) replayed in Python from its start and action log (SoakReplays)."""
    import time
    target, free_target = GATE[mode]['soak'], GATE[mode]['free_soak']
    v1 = ctx.world == GATE_WORLDS[0]
    segments = [(task, 'all', source) for task in GATE_TASKS for source in SOAK_SOURCES] if v1 else []
    target = target if v1 else 0  # village-world-v2's set: free play alone
    per = -(-target // len(segments)) if segments else 0
    # free-play-v1's own segments (added on 12 November; the goal tasks' share is as it was): biased play on its short
    # configurations, which replay quickly, and their own world-tick target
    free = [(native.FREE_TASK, name, 'biased') for name in (FREE_CONFIGS if v1 else FREE_CONFIGS_V2) if name not in FREE_LONG]
    free_per = -(-free_target // len(free)) if free_target else 0
    replays = SoakReplays(ctx)
    t0 = time.time()
    out = dict(id='soak', target=target, free_target=free_target, ticks=0, free_ticks=0, decisions=0, episodes=0,
               errors=[], unsupported=[], segments=[])
    for index, (task, goals, spec) in enumerate(segments + (free if free_per else [])):
        if task == native.FREE_TASK:
            per = free_per
        batch, key, ticks, t1 = ctx.batch(task, goals, SOAK_N, threads=ctx.threads), SOAK_KEY + 1000 * index, 0, time.time()
        slots, episodes0 = np.arange(SOAK_N), out['episodes']
        while ticks < per and len(out['errors']) + len(out['unsupported']) < SOAK_FAILURES:
            source = FastBiased(key) if spec == 'biased' else ctx.source(spec, key)
            state, obs = batch.reset(key)
            actions = [[] for _ in range(SOAK_N)]
            recorded = [[world_digest(obs, None, None, None, b)] if soak_selected(key, b, 0) else None
                        for b in range(SOAK_N)]
            starts, step = np.ones(SOAK_N, bool), 0
            while ticks < per:
                action = source(step, obs, slots, starts)
                for b in range(SOAK_N):
                    actions[b].append(int(action[b]))
                episodes = batch.episodes
                try:
                    state, obs, reward, done, info = batch.step(state, action)
                except Exception as error:  # surfaced: recorded, its world's episode replayed in Python
                    world = getattr(error, 'world', -1)
                    out['errors'].append(f'{task} {spec} key {key} step {step} world {world}: '
                                         f'{type(error).__name__}: {error}')
                    if 0 <= world < SOAK_N:
                        replays.replay(task, key, world, episodes[world], actions[world], goals=goals)
                    break
                step += 1
                ticks += int(info['decision_ticks'].sum())
                out['decisions'] += SOAK_N
                for b in np.flatnonzero(info['unsupported']):
                    reason = next((e['reason'] for e in reversed(batch.evidence) if e['world'] == b), '?')
                    out['unsupported'].append(f'{task} {goals} {spec} key {key} world {b} episode {episodes[b]}: {reason}')
                    replays.replay(task, key, int(b), episodes[b], actions[b], goals=goals)
                if info['unsupported'].any():
                    break
                for b in range(SOAK_N):
                    if recorded[b] is not None:
                        recorded[b].append(world_digest(obs, reward, done, info, b))
                for b in np.flatnonzero(done):
                    out['episodes'] += 1
                    if recorded[b] is not None:
                        replays.replay(task, key, int(b), episodes[b], actions[b], recorded[b], goals=goals)
                    actions[b] = []
                    recorded[b] = [world_digest(obs, None, None, None, b)] if soak_selected(key, b, episodes[b] + 1) \
                        else None
                starts = done
            key += 1
        out['free_ticks' if task == native.FREE_TASK else 'ticks'] += ticks
        out['segments'].append(dict(task=task, goals=goals, source=spec, ticks=ticks, episodes=out['episodes'] - episodes0,
                                    seconds=round(time.time() - t1, 1)))
        log(f'[gate] soak {task} {goals} {spec}: {ticks} ticks, {out["episodes"] - episodes0} episodes, '
            f'{replays.replayed} replayed so far ({time.time() - t1:.0f} s)')
    out.update(replayed=replays.replayed, divergences=replays.divergences, gaps=replays.gaps,
               mismatches=replays.mismatches, stopped=replays.stopped, reports=replays.reports,
               most_objs=replays.most_objs, seconds=time.time() - t0)  # most_objs: over the replayed episodes
    log(f'[gate] soak: {out["ticks"]} ticks (free play {out["free_ticks"]}), {out["decisions"]} world decisions, '
        f'{out["episodes"]} episodes, '
        f'{len(out["errors"])} errors, {len(out["unsupported"])} unsupported, {out["replayed"]} replayed: '
        f'{out["divergences"]} divergences, {out["gaps"]} gaps, {out["mismatches"]} output mismatches '
        f'({out["seconds"]:.0f} s)')
    for line in out['errors'] + out['unsupported'] + out['reports']:
        log(line)
    return out


# ---- free-play-v1's soak in the C engine (the reachability soak in each engine) ------------------------------------
# Random legal play and sim/free.py's controls on free-play lives of village-world-v1 Stage A, each run in lockstep
# with the Python reference through `run`: every decision's outputs and every tick's record, draws and events compared.
# A divergence, a gap or an unsupported world ends its run there; it is reported with its evidence, never dropped.
FREE_SOAK_CONTROLS = ('wood', 'fish', 'mine', 'melee', 'chickens', 'rotation@1500', 'rotation@9000')
_SOAK_INPUTS = {}


def free_soak_run(job):
    """One soak run: job (pack, pool, archive, muggers, xp_rate, life, source, key, worlds, steps) -> its JSONL row."""
    import time
    from sim.vec import CpuBatch
    from sim.workers import load_tables
    from sim.world import load_pool
    pack_path, pool, archive, muggers, xp_rate, life, source, key, worlds, steps = job
    if (pack_path, pool, archive) not in _SOAK_INPUTS:  # the pack's recorded archive and pool must be these
        _SOAK_INPUTS[pack_path, pool, archive] = (load_tables(archive), load_pool(pool)[0],
                                                  native.open_pack(pack_path, inputs=dict(tables=archive, pool=pool)))
    tables, layouts, pack = _SOAK_INPUTS[pack_path, pool, archive]
    t0 = time.time()
    settings = dict(xp_rate=xp_rate, muggers=muggers, life=tuple(life))
    env = CpuBatch(tables, layouts, worlds, task=native.FREE_TASK, **settings)
    a = PythonSide(env, canon=Canon(PackStrings(pack)))
    b = NativeSide(native.NativeBatch(pack, worlds, task=native.FREE_TASK, **settings))
    name, _, period = source.partition('@')
    play = RandomLegal(key) if name == 'random' else ControlSource(a, [name], key, int(period) if period else None)
    ends = dict(horizon=0, death=0, ticks=0, events=0)
    seen = [0] * worlds
    census, played = FreeCensus(tables), set()

    def observe(step, out_a, out_b):
        if step >= 0:
            done, info = out_a[3], out_a[4]
            for w in np.flatnonzero(done):
                ends['horizon' if info['terminal'][w] == 1 else 'death'] += 1
                ends['ticks'] = max(ends['ticks'], int(info['episode_ticks'][w]))
                ends['events'] = max(ends['events'], seen[w])
        for w, task in enumerate(env.tasks):
            seen[w] = len(task.world.events) - task.event_start
            census.read(w, task)
        played.update(s['layout'] for s in env.starts if s is not None)
    result = run(a, b, play, key, steps, observe=observe)
    return dict(form='muggers' if muggers else 'none', world=env.world, layouts=sorted(played), census=census.counts,
                xp_rate=xp_rate, life=list(life), source=source, key=key, worlds=worlds, steps=result.steps, world_decisions=result.steps * result.worlds, ticks=result.ticks,
                lives=result.episodes, horizons=ends['horizon'], deaths=ends['death'], longest=ends['ticks'],
                most_events=max([ends['events'], *seen]), most_objs=result.most_objs,
                divergence=None if result.divergence is None else result.divergence.text(), gap=result.gap or None,
                stopped=result.stopped or None, seconds=round(time.time() - t0, 1))


class FreeCensus:
    """What a free-play run's Python lives reached (read(b, task) after each step, world slot b's life now): picks and
    the Picks refused on a full inventory (their message), willow and yew logs, pen door openings (a step that finds a
    door open which the last step found closed) and the steps with one open, the highwayman's hits on the player, the
    player's on it and its defeats, and the black cape (its drop) picked up and worn; a coverage count, not an output of
    either engine. A life that ended in a step (CpuBatch starts the next one in it) is read to its end first, so its last
    decision counts (a lethal hit, say)."""

    def __init__(self, tables):
        self.counts = dict(picks=0, pick_full=0, willow_yew_logs=0, door_openings=0, door_open_steps=0,
                           highwayman_damage=0, highwayman_hits=0, highwayman_defeated=0, cape_pickups=0, cape_worn=0)
        self.slots = {}  # world slot: (its life's World, events read, chat lines read, a pen door open at the last read)
        self.logs = {tables.obj_ids[name] for name in ('willow_logs', 'yew_logs') if name in tables.obj_ids}
        self.cape = tables.obj_ids.get('black_cape')
        self.cross = {k for k, v in tables.loc_types.items() if v['debugname'] == 'poordoor_cross'}
        self.hunter = {k for k, v in tables.npc_types.items() if v['debugname'] == 'highwayman'}
        self.full = None  # the full-inventory messages of the archive's plants (pick: pick_full_<plant>)

    def read(self, b, task):
        w, c = task.world, self.counts
        last, seen, said, was_open = self.slots.get(b, (None, 0, 0, False))
        if last is not None and last is not w:
            self._events(last, seen, said)
            seen, said, was_open = 0, 0, False
        self._events(w, seen, said)
        now_open = any(loc.active and loc.type in self.cross for row in w.loc_at.values() for loc in row)
        c['door_open_steps'] += now_open
        c['door_openings'] += now_open and not was_open
        self.slots[b] = (w, len(w.events), len(w.chat_log), now_open)

    def _events(self, w, seen, said):
        from sim.engine import WORN
        c = self.counts
        if self.full is None:
            plants = (w.t.meta.get('content') or {}).get('picks') or {}
            self.full = {w.text(f'pick_full_{name}') for name in plants}
        c['pick_full'] += sum(text in self.full for _, text in w.chat_log[said:])
        hunters = {n.nid for n in w.npcs if n.type in self.hunter}
        for _, kind, d in w.events[seen:]:
            if kind == 'produce':
                c['picks'] += d.get('script') == '[proc,pickup_loc_floor]'
                c['willow_yew_logs'] += d.get('obj') in self.logs
            elif kind == 'hit':
                c['highwayman_hits'] += d.get('type') in self.hunter
            elif kind == 'npc_defeated':
                c['highwayman_defeated'] += d.get('type') in self.hunter
            elif kind == 'agent_hp':
                c['highwayman_damage'] += d.get('cause') == 'damage' and d.get('source') in hunters
            elif kind == 'pickup':
                c['cape_pickups'] += self.cape is not None and d.get('obj') == self.cape
            elif kind == 'transfer':
                c['cape_worn'] += self.cape is not None and d.get('dst') == WORN and d.get('obj') == self.cape


class PolicyCensus(FreeCensus):
    """FreeCensus with what a policy's play reached in either world (the policy part, gate_policy): the agent's hits and
    the npcs it defeated, its productions, XP events and level-ups (an xp event that raised the stat's level, which
    queues its dialog)."""

    def __init__(self, tables):
        super().__init__(tables)
        self.counts.update(agent_hits=0, agent_defeats=0, productions=0, xp_events=0, level_ups=0)

    def _events(self, w, seen, said):
        from sim.engine import level_for
        super()._events(w, seen, said)
        c = self.counts
        for _, kind, d in w.events[seen:]:  # the world's one player is the agent: a hit or defeat with a source is its
            if kind == 'hit':
                c['agent_hits'] += d.get('source') is not None
            elif kind == 'npc_defeated':
                c['agent_defeats'] += d.get('source') is not None
            elif kind == 'produce':
                c['productions'] += 1
            elif kind == 'xp':
                c['xp_events'] += 1
                c['level_ups'] += level_for(d['after']) > level_for(d['before'])


def free_soak(output, pack, pool, archive, jobs, processes):
    """Run the jobs (free_soak_run) on `processes` processes; lives.jsonl gets one row a run in completion order and
    manifest.json the settings, counts and times, with the commit (marked -dirty for a changed tree), the native library's
    file (its tag digests the C source, the generated header, the flags and the interpreter) and the pack's sha256.
    Returns the manifest."""
    import multiprocessing as mp
    import time
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    t0, rows = time.time(), []
    with (output / 'runs.jsonl').open('w') as stream, mp.get_context('spawn').Pool(processes) as workers:
        for row in workers.imap_unordered(free_soak_run, [(str(pack), str(pool), str(archive), *job) for job in jobs]):
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            rows.append(row)
            print(f"[free-soak] {row['form']} {row['source']} key {row['key']}: {row['world_decisions']} world decisions, "
                  f"{row['ticks']} ticks, {row['lives']} lives ({row['horizons']} cut, {row['deaths']} deaths), "
                  f"{'DIVERGED' if row['divergence'] else row['gap'] or row['stopped'] or 'equal'} "
                  f"({row['seconds']:.0f} s)", flush=True)
    counts = {}
    for row in rows:
        c = counts.setdefault(f"{row['form']}/{row['source']}", dict(runs=0, world_decisions=0, ticks=0, lives=0,
                                                                     horizons=0, deaths=0, divergences=0, gaps=0,
                                                                     stopped=0, longest=0, most_events=0,
                                                                     most_objs=0))
        for k in ('world_decisions', 'ticks', 'lives', 'horizons', 'deaths'):
            c[k] += row[k]
        c['runs'] += 1
        c['divergences'] += row['divergence'] is not None
        c['gaps'] += row['gap'] is not None
        c['stopped'] += row['stopped'] is not None
        c['longest'], c['most_events'] = max(c['longest'], row['longest']), max(c['most_events'], row['most_events'])
        c['most_objs'] = max(c['most_objs'], row['most_objs'])
        c['census'] = {k: c.get('census', {}).get(k, 0) + v for k, v in row['census'].items()}
        c['layouts'] = sorted(set(c.get('layouts', [])) | set(row['layouts']))
    manifest = dict(version='free-soak', commit=_commit(), library=native.library()[0].name, pack=str(pack),
                    pack_sha256=native.sha256_file(pack), pool=str(pool), archive=str(archive), worlds=sorted({row['world'] for row in rows}),
                    processes=processes, jobs=[list(j) for j in jobs], wall_seconds=round(time.time() - t0, 1),
                    counts=counts)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def free_soak_jobs(none_random, none_controls, muggers_random, muggers_controls, worlds, xp_rate, life, key0):
    """The soak's runs: (muggers, xp_rate, life, source, key, worlds, steps); *_random: (runs, steps), *_controls:
    (runs a control, steps), each run its own key from key0 on."""
    jobs, key = [], key0
    for muggers, (runs, steps), (control_runs, control_steps) in ((False, none_random, none_controls),
                                                                 (True, muggers_random, muggers_controls)):
        for _ in range(runs):
            jobs.append((muggers, xp_rate, life, 'random', key, worlds, steps))
            key += 1
        for name in FREE_SOAK_CONTROLS:
            for _ in range(control_runs):
                jobs.append((muggers, xp_rate, life, name, key, worlds, control_steps))
                key += 1
    return jobs


def free_soak_arguments(parser):
    logs = native.ROOT / 'logs'
    parser.add_argument('--pack', required=True, help='the pack of --pool (sim.native export)')
    parser.add_argument('--pool', required=True, help='a pool of one village world: compact-world-v1 or village-world-v2 '
                        '(its layouts; derived tables unused); village-world-v2 holds no muggers (--muggers-* 0,0)')
    parser.add_argument('--archive', default=str(logs / 'sim/data/draynor-nc1.npz'),
                        help='the archive of the pack (village-world-v2: draynor-nc2.npz)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--xp-rate', type=int, default=1)
    parser.add_argument('--life', default='["fixed", 36000]')
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--key', type=int, default=500_000, help='the first run\'s key (each run its own)')
    for name, default in (('none-random', '12,4000'), ('none-controls', '2,6000'), ('muggers-random', None),
                          ('muggers-controls', None)):
        parser.add_argument(f'--{name}', default=default, help='runs,steps (controls: runs a control,steps); the mugger '
                            "runs' default: 4,1500 and 1,1500 in a world with muggers, 0,0 in one without")
    parser.add_argument('--processes', type=int, default=6)


def free_soak_main(args):
    from sim.combat import village_of
    from sim.world import load_pool
    pair = lambda text: tuple(int(v) for v in text.split(','))
    village = village_of(load_pool(args.pool)[0][0])
    muggers = village is not None and village.get('muggers') is not False
    defaults = dict(muggers_random='4,1500', muggers_controls='1,1500') if muggers else \
        dict(muggers_random='0,0', muggers_controls='0,0')
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
        elif not muggers and pair(getattr(args, name))[0]:
            raise SystemExit(f"{village['version'] if village else args.pool} holds no muggers: --{name.replace('_', '-')} "
                             'runs must be 0')
    jobs = free_soak_jobs(pair(args.none_random), pair(args.none_controls), pair(args.muggers_random),
                          pair(args.muggers_controls), args.worlds, args.xp_rate, json.loads(args.life), args.key)
    manifest = free_soak(args.output, args.pack, args.pool, args.archive, jobs, args.processes)
    bad = sum(c['divergences'] + c['gaps'] + c['stopped'] for c in manifest['counts'].values())
    for name, c in manifest['counts'].items():
        print(f"[free-soak] {name}: {c['runs']} runs, {c['world_decisions']} world decisions, {c['ticks']} ticks, "
              f"{c['lives']} lives ({c['horizons']} cut, {c['deaths']} deaths; longest {c['longest']} ticks, most "
              f"events {c['most_events']}, most ground objs {c['most_objs']}), {c['divergences']} divergences, "
              f"{c['gaps']} gaps, {c['stopped']} stopped; {len(c['layouts'])} layouts; reached {c['census']}")
    print(f"[free-soak] {'CLEAN' if bad == 0 else 'NOT CLEAN'} in {manifest['wall_seconds']:.0f} s")
    sys.exit(0 if bad == 0 else 1)


# ---- the gate's command line -------------------------------------------------------------------------------------
def _commit():
    """HEAD's commit, with '-dirty' when a tracked file differs from it; 'unknown' outside a checkout (or without git)."""
    import subprocess
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=native.ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=native.ROOT,
                               capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    if head.returncode or dirty.returncode or not COMMIT.fullmatch(head.stdout.strip()):
        return 'unknown'
    return head.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')


COMMIT = re.compile(r'[0-9a-f]{40}')  # a gate's rows are of one such commit, clean (gate_summary)


# run()'s stops (Result.stopped, in a case row's `stopped` or a soak replay's): an unsupported world, flagged alike or
# raised alike by a setup, passes; any other exception both sides raised alike stopped the run before its decisions.
STOP = re.compile(r'both sides raised (\w+) at step -?\d+: |world slot \d+ reached unsupported content at step ')


def unsupported_stop(text):
    """Whether a stop is an unsupported world's (the gate passes it; every other stop alike fails the gate)."""
    from sim.engine import Unsupported
    found = STOP.search(text)
    if found is None:
        raise ValueError(f'not a stop of run(): {text[:200]}')
    if found.group(1) is None:
        return True
    names, todo = set(), [Unsupported]
    while todo:
        cls = todo.pop()
        names.add(cls.__name__)
        todo += cls.__subclasses__()
    return found.group(1) in names


def gate_clean(rows):
    """(clean, divergences, gaps, halted, empty): every case, the digests or the training workers, the thread check, the
    soak and the policy part (no failure: its pins and the respawn-cycle life) passed; halted: the runs and replays that
    stopped alike on an exception other than an unsupported world's (unsupported_stop), whose remaining decisions were
    never compared; empty: the cases and policy parts that compared no decision (every run stopped before one, or a row
    without the count), which fail the gate too."""
    divergences = gaps = halted = 0
    empty = []
    clean = True
    for row in rows:
        if row['id'] in ('digests', 'threads', 'workers'):
            clean &= not row['failures']
            continue
        if row['id'] == 'policy':
            clean &= not row['failures']
        if ('task' in row or row['id'] == 'policy') and not row.get('compared'):
            empty.append(row['id'])
        halted += sum(not unsupported_stop(text) for text in row.get('stopped') or [])
        if row['id'] == 'soak':
            divergences += row['divergences'] + row['mismatches']
            gaps += row['gaps']
            # an unsupported world passes when its replay flags it alike (a replay that does not is a divergence)
            clean &= not row['errors'] and row['ticks'] >= row['target'] and row.get('free_ticks', 0) >= row.get('free_target', 0)
        else:
            divergences += row['divergences']
            gaps += row['gaps']
    clean = clean and divergences == 0 and gaps == 0 and halted == 0 and not empty
    return clean, divergences, gaps, halted, empty


def gate_expected(mode, shard=(0, 1), only=None, world=GATE_WORLDS[0], parts=None):
    """The row ids a gate run writes: with the cases part, the cases (gate_cases) of its world set's shard's
    configurations (gate_configs, index % k == i), and the set's other parts in this mode (gate_parts) that `parts`
    names (default: every part), each that `only` matches."""
    i, k = shard
    parts = PARTS if parts is None else parts
    pattern = re.compile(only) if only else None
    ids = [case_id(task, goals, first, source, scenario)
           for index, (task, goals, first) in enumerate(gate_configs(mode, world))
           if index % k == i for source, scenario in gate_cases(task, goals)] if 'cases' in parts else []
    ids += [name for name in gate_parts(mode, world) if name in parts]
    return {name for name in ids if pattern is None or pattern.search(name)}


def claimed_cases(rows, parts=None):
    """The row ids a merged set of result rows claims: its mode's whole gate of its world set, cases and parts (a row
    that names no world is village-world-v1's, as every row before the world sets), or, when every row comes from an
    --only rerun, the cases and parts its patterns match; `parts` narrows the claim to the parts named (gate --summary
    --parts: a merge that leaves a part out on purpose). None when the rows hold no mode, more than one, or more than one
    world set."""
    modes = {row.get('mode') for row in rows}
    worlds = {row.get('world', GATE_WORLDS[0]) for row in rows}
    if len(modes) != 1 or None in modes or len(worlds) != 1:
        return None
    mode, world = modes.pop(), worlds.pop()
    patterns = {row.get('only') for row in rows}
    if None in patterns:
        return gate_expected(mode, world=world, parts=parts)
    return set().union(*(gate_expected(mode, only=pattern, world=world, parts=parts) for pattern in patterns))


def provenance_key(provenance):
    """A provenance's identity: all of it but the files' paths (each file is its sha256; a path is where it lay)."""
    if not provenance:
        return provenance
    files = {name: {k: v for k, v in entry.items() if k != 'path'} for name, entry in (provenance.get('files') or {}).items()}
    return json.dumps(dict(provenance, files=files), sort_keys=True)


def provenance_problems(rows):
    """Why the rows are not of one gate's inputs: each row must carry the same provenance (gate_provenance, compared
    without its paths: provenance_key) that names their world set, and the pack must record the archive, pool and supply
    pool the rows played ([] when they are)."""
    found = {provenance_key(row.get('provenance')) for row in rows}
    if len(found) != 1:
        return [f'the rows name {len(found)} sets of inputs (provenance), not one']
    given = rows[0].get('provenance')
    if not given:
        return ['the rows name no inputs (provenance)']
    problems = []
    if given.get('world') != rows[0].get('world', GATE_WORLDS[0]):
        problems.append(f"the inputs are of {given.get('world')}, the rows of {rows[0].get('world', GATE_WORLDS[0])}")
    files, recorded = given.get('files') or {}, given.get('pack_inputs') or {}
    for name, key in (('archive', 'tables'), ('pool', 'pool'), ('supply', 'supply_pool')):
        if recorded.get(key) != (files.get(name) or {}).get('sha256'):
            problems.append(f'the pack records another {key} than the {name} the rows played')
    return problems


def gate_summary(rows, log, expected=None, parts=None):
    """The table per task (and goal set), source and scenario, summed over first slots, free play's census, and the
    totals. Clean only when gate_clean holds, the rows hold every expected row id (cases and parts) and no other of their
    mode's gate (expected: a run's own ids, gate_expected; None: those the rows claim, claimed_cases, narrowed to `parts`
    when given), and, when there are rows, they are all of one clean commit (40 hex digits, not '-dirty', not 'unknown')
    and of one set of inputs (provenance_problems)."""
    groups = OrderedDict()
    for row in rows:
        if 'task' not in row:
            continue
        k = (row['task'], row['goals'], row['source'], row['scenario'] or '-')
        g = groups.setdefault(k, dict(firsts=[], runs=0, decisions=0, compared=0, world_decisions=0, ticks=0, episodes=0,
                                      divergences=0, gaps=0, stopped=0, objs=None))
        g['firsts'].append(row['first'])
        for name in ('runs', 'decisions', 'world_decisions', 'ticks', 'episodes', 'divergences', 'gaps'):
            g[name] += row[name]
        g['compared'] += row.get('compared') or 0
        g['stopped'] += len(row['stopped'])
        if 'most_objs' in row:
            g['objs'] = max(g['objs'] or 0, row['most_objs'])
    log('| task | goals | source | scenario | firsts | runs | decisions | compared | world decisions | ticks | episodes | '
        'divergences | gaps | stopped alike | most ground objs |')
    log('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for (task, goals, source, scenario), g in groups.items():
        log(f'| {task} | {goals} | {source} | {scenario} | {",".join(map(str, sorted(g["firsts"])))} | {g["runs"]} | '
            f'{g["decisions"]} | {g["compared"]} | {g["world_decisions"]} | {g["ticks"]} | {g["episodes"]} | '
            f'{g["divergences"]} | {g["gaps"]} | {g["stopped"]} | {"-" if g["objs"] is None else g["objs"]} |')
    free = [row for row in rows if row.get('task') == native.FREE_TASK]
    if free:  # what free play's Python lives reached over the cases (FreeCensus), and the deaths
        census = {}
        for row in free:
            for name, count in (row.get('census') or {}).items():
                census[name] = census.get(name, 0) + count
        log(f'[gate] free play reached: {", ".join(f"{k} {v}" for k, v in census.items()) or "no census"}; deaths '
            f'{sum(row.get("deaths") or 0 for row in free)}')
    for row in rows:
        if row['id'] == 'policy' and 'files' in row:  # the policy part's line and what its Python lives reached
            log(policy_line(row))
            if row.get('census'):
                log(f'[gate] policy reached: {", ".join(f"{k} {v}" for k, v in row["census"].items())}; deaths '
                    f'{row.get("deaths") or 0}')
    clean, divergences, gaps, halted, empty = gate_clean(rows)
    modes = {(row.get('mode'), row.get('world', GATE_WORLDS[0])) for row in rows}
    ids = {row['id'] for row in rows}
    claim = ''
    if expected is None and parts is not None and len(modes) == 1:  # a merge that names its parts claims no other
        mode, world = next(iter(modes))
        foreign = sorted(set(parts) - {'cases', *(gate_parts(mode, world) if mode in GATE and world in GATE_WORLDS else [])})
        left = [p for p in ['cases', *gate_parts(mode, world)] if p not in parts] if mode in GATE and world in GATE_WORLDS \
            else []
        if foreign:
            log(f'[gate] --parts names parts that {world} {mode} does not have: {foreign}')
            clean = False
        claim = f'; claims {", ".join(parts)}' + (f' (leaves out {", ".join(left)})' if left else '')
    reruns = sorted({str(row.get('only')) for row in rows if row.get('only') is not None})
    if reruns and all(row.get('only') is not None for row in rows):
        claim += f'; a rerun of {reruns}'
    expected = claimed_cases(rows, parts) if expected is None else expected
    declared = set().union(*(gate_expected(mode, world=world) for mode, world in modes
                             if mode in GATE and world in GATE_WORLDS))
    missing, undeclared = sorted((expected or set()) - ids), sorted(ids - declared)
    if expected is None:
        log(f'[gate] the rows are of modes and worlds {sorted(map(str, modes))}, not of one gate')
    for name in missing[:20]:
        log(f'[gate] missing: {name}')
    for name in undeclared[:20]:
        log(f'[gate] undeclared: {name}')
    for name in empty[:20]:
        log(f'[gate] compared no decision: {name}')
    commits = sorted({str(row.get('commit')) for row in rows})
    one_commit = not rows or (len(commits) == 1 and COMMIT.fullmatch(commits[0]) is not None)
    if not one_commit:
        log(f'[gate] the rows are not of one clean commit: {commits[:5]}')
    problems = provenance_problems(rows) if rows else []
    for line in problems:
        log(f'[gate] inputs: {line}')
    if rows and not problems:
        given = rows[0]['provenance']
        pack, env = given['files']['pack'], given.get('env') or {}
        log(f"[gate] inputs: {given['world']}, pack {Path(pack['path']).name} {pack['sha256'][:16]}, library "
            f"{given['library']}{''.join(f', {k}={v}' for k, v in env.items())}")
    clean = clean and expected is not None and not missing and not undeclared and one_commit and not problems
    objs = [row['most_objs'] for row in rows if 'most_objs' in row]
    log(f'[gate] {len(rows)} rows, commits {commits}: {"CLEAN" if clean else "NOT CLEAN"}, {divergences} divergences, '
        f'{gaps} gaps, {halted} stopped on an exception, {len(empty)} compared nothing, {len(missing)} of '
        f'{len(expected or ())} rows missing, {len(undeclared)} undeclared; most ground objs '
        f'{max(objs) if objs else "-"} of RS_MAX_OBJS {native.load().constants()["RS_MAX_OBJS"]}{claim}')
    return clean


def gate_arguments(parser):
    logs = native.ROOT / 'logs'
    parser.add_argument('--quick', action='store_true', help='the fixed quick subset (about 15 minutes, no soak)')
    parser.add_argument('--world', default=GATE_WORLDS[0], choices=GATE_WORLDS,
                        help='the world set: village-world-v1 (the goal tasks and its free play: the gate) or '
                        'village-world-v2 (its free play alone, an opt-in set of its own; its inputs default to '
                        'GATE_INPUTS_V2)')
    parser.add_argument('--archive', help='default: draynor-nc1.npz (village-world-v2: draynor-nc2.npz)')
    parser.add_argument('--pool', help='default: train-8-nc1-derived.npz (village-world-v2: train-128-20000-v2.npz)')
    parser.add_argument('--supply', default=str(logs / 'supply/pools/train-128.npz'))
    parser.add_argument('--digest-pool', default=str(DIGEST_POOL),
                        help="the digests part's v5 pool (default: the one FOUR_GOAL_DIGESTS were recorded on)")
    parser.add_argument('--pack', help='default: sim/native.py PACK (village-world-v2: PACK_V2)')
    parser.add_argument('--work', default=str(logs / 'native/gate'), help='the digest packs and the results (ignored)')
    parser.add_argument('--out', help='results, one JSON line per case (default WORK/gate-MODE[-SHARD].jsonl)')
    parser.add_argument('--parts', help=f'comma separated: {", ".join(PARTS)} (default: all of the world set: '
                        'village-world-v1 all but workers, village-world-v2 all but digests); with --summary, the parts '
                        'the merged rows claim (default: every part of their world set and mode)')
    parser.add_argument('--shard', default='0/1', help='i/k: the configurations whose index is i modulo k')
    parser.add_argument('--resume', action='store_true', help='skip the cases already in --out')
    parser.add_argument('--only', help='run only the cases and parts whose id this regular expression matches, searched '
                        '(TASK|GOALS|FIRST|SOURCE|SCENARIO; digests, threads, soak, workers, policy): a rerun, not the '
                        "gate. The policy part's id is in the policy-source cases' ids too (policy:A): ^policy$ selects "
                        'the part alone')
    parser.add_argument('--threads', type=int, default=4, help="the soak's NativeBatch threads")
    parser.add_argument('--summary', nargs='*', help='print the merged table and verdict of these result files and exit')


def summary_parts(text):
    """The parts a gate command names (gate --parts; with --summary, the parts a merge claims; None: every part of its
    world set), refusing unknown names and an empty list."""
    if text is None:
        return None
    parts = [p for p in text.split(',') if p]
    if not parts:
        raise ValueError('--parts names no part')
    if set(parts) - set(PARTS):
        raise ValueError(f'unknown parts {sorted(set(parts) - set(PARTS))}')
    return parts


def gate_settings(args):
    """(mode, world, parts, shard, out) of a gate command, its --archive, --pool and --pack filled with its world set's
    defaults (GATE_INPUTS, GATE_INPUTS_V2) where not given. village-world-v2's set has no digests part (FOUR_GOAL_DIGESTS
    are the goal tasks'), village-world-v1's no workers part (its digests check the workers path): each refuses the other's
    named; v2's default --out names the world."""
    mode = 'quick' if args.quick else 'full'
    world = getattr(args, 'world', None) or GATE_WORLDS[0]
    v1 = world == GATE_WORLDS[0]
    for name, value in (GATE_INPUTS if v1 else GATE_INPUTS_V2).items():
        if getattr(args, name, None) is None:
            setattr(args, name, str(value))
    other = 'workers' if v1 else 'digests'
    parts = summary_parts(args.parts or ','.join(p for p in PARTS if p != other))
    if not v1 and 'digests' in parts:
        raise ValueError("village-world-v2's set has no digests part (FOUR_GOAL_DIGESTS are the goal tasks')")
    if v1 and 'workers' in parts:
        raise ValueError("village-world-v1's set has no workers part (its digests part checks the workers path)")
    shard = tuple(int(v) for v in args.shard.split('/'))
    out = Path(args.out or Path(args.work) / (f'gate-{mode}{"" if v1 else "-" + world}'
                                              f'{"" if shard == (0, 1) else f"-{shard[0]}of{shard[1]}"}.jsonl'))
    return mode, world, parts, shard, out


def gate_provenance(ctx):
    """What a gate process plays, written into each of its rows (gate_row) and required alike of every row by the
    summary: the world set; each input file's absolute path and sha256 (the pack, the archive, the pool, the supply
    pool); the inputs and the name table the pack's manifest records; the native library's file (its tag digests the C
    source, the generated header, the flags and the interpreter); and SIM_NATIVE_SOURCE and SIM_NATIVE_LTO where set.
    The digests part's own pool is in its row."""
    import os
    meta = json.loads(ctx.pack.manifest)
    files = dict(pack=ctx.pack_path, archive=ctx.archive, pool=ctx.pool, supply=ctx.supply)
    return dict(world=ctx.world,
                files={k: dict(path=str(Path(v).absolute()), sha256=native.sha256_file(v)) for k, v in files.items()},
                pack_inputs=meta.get('inputs'), pack_names=meta.get('names'), library=native.library()[0].name,
                env={k: os.environ[k] for k in ('SIM_NATIVE_SOURCE', 'SIM_NATIVE_LTO') if k in os.environ})


def gate_row(row, commit, mode, only, world, provenance=None):
    """A result row as the gate writes it: the part's or case's fields, the commit, mode, --only and the process's
    provenance (gate_provenance); a row of village-world-v2's set names its world (a village-world-v1 row names none, as
    every row before the world sets)."""
    row = dict(row.__dict__ if hasattr(row, '__dict__') else row, commit=commit, mode=mode, only=only,
               provenance=provenance)
    if world != GATE_WORLDS[0]:
        row['world'] = world
    return row


def resume_done(before, out, commit, mode, world, provenance):
    """The row ids a resumed run skips: every row already in `out`, which must all be of this run's gate (its commit,
    mode, world set and provenance), so that the file stays one gate's; any other row refuses the resume."""
    mine = (commit, mode, world, provenance_key(provenance))
    other = sorted({row['id'] for row in before if (row.get('commit'), row.get('mode'), row.get('world', GATE_WORLDS[0]),
                                                    provenance_key(row.get('provenance'))) != mine})
    if other:
        raise ValueError(f'--resume: {out} holds {len(other)} rows of another commit, mode, world set or inputs than this '
                         f'run ({commit}, {mode}, {world}), such as {other[0]}: give a new --out')
    return {row['id'] for row in before}


def resume_rows(out, log):
    """The rows already in a results file (gate --resume): a last line that an interruption cut short is cut from the
    file (truncated at its start, so the rows before it stay), and a whole last row whose newline was lost gets it, so
    that the next row starts a line of its own; any other line that is not a row refuses the resume."""
    import os
    data = out.read_bytes()
    lines, rows, offset = data.splitlines(keepends=True), [], 0
    for k, line in enumerate(lines):
        last = k == len(lines) - 1
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not last or line.endswith(b'\n'):
                    raise ValueError(f'--resume: line {k + 1} of {out} is not a result row') from None
                os.truncate(out, offset)
                log(f'[gate] --resume: cut the last line of {out}, which an interruption left unfinished')
                break
            if last and not line.endswith(b'\n'):
                with out.open('ab') as stream:
                    stream.write(b'\n')
        offset += len(line)
    return rows


def gate_main(args):
    """`python -m sim.native gate`: runs the parts, prints a line per case (and each failure's first-divergence
    report), appends each result to --out, then the table; exit status 0 only when clean."""
    if args.summary is not None:
        rows = [json.loads(line) for path in args.summary for line in Path(path).read_text().splitlines() if line]
        sys.exit(0 if gate_summary(rows, print, parts=summary_parts(args.parts)) else 1)
    mode, world, parts, shard, out = gate_settings(args)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    def log(line):
        print(line, flush=True)
    before = resume_rows(out, log) if args.resume and out.exists() else []
    ctx = GateContext(args.archive, args.pool, args.supply, args.pack, work, args.threads, world).load(
        policies=('cases' in parts or 'soak' in parts) and world == GATE_WORLDS[0])  # v2's set plays no policy
    commit, provenance = _commit(), gate_provenance(ctx)
    done = resume_done(before, out, commit, mode, world, provenance)
    if not args.resume:
        out.write_text('')
    only = re.compile(args.only) if args.only else None

    def skip(name):
        return name in done or (only is not None and not only.search(name))

    def emit(row):
        row = gate_row(row, commit, mode, args.only, world, provenance)
        with out.open('a') as f:
            f.write(json.dumps(row, default=str) + '\n')
    log(f'[gate] {mode}{"" if world == GATE_WORLDS[0] else " " + world} at {commit}: parts {parts}, '
        f'shard {shard[0]}/{shard[1]}, results {out}')
    log(f'[gate] checkpoints: {", ".join(f"{k} {v[2]}" for k, v in ctx.policies.items())}')
    if 'cases' in parts:
        gate_cases_part(ctx, mode, shard, skip, emit, log)
    if 'digests' in parts and not skip('digests'):
        emit(gate_digests(ctx, log, args.digest_pool))
    if 'threads' in parts and not skip('threads'):
        emit(gate_threads(ctx, mode, log))
    if 'soak' in parts and GATE[mode]['soak'] and not skip('soak'):
        emit(gate_soak(ctx, mode, log))
    if 'workers' in parts and not skip('workers'):
        emit(gate_workers(ctx, mode, log))
    if 'policy' in parts and not skip('policy'):
        emit(gate_policy(ctx, mode, log))
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    sys.exit(0 if gate_summary(rows, log, gate_expected(mode, shard, args.only, world, parts)) else 1)
