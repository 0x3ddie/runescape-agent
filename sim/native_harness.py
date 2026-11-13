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
                                              'knife': 'RS_S_USE_DEFAULT_CATEGORY_22'}[v['cases'][0]],
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
            spec = SCENARIOS[spec] if spec in SCENARIOS else json.loads(spec)
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
    from sim.task import GoalEpisode
    init, capture = GoalEpisode.__init__, Agent.capture_text
    draws = vec.draw_start, vec.draw_combat, search.draw_supply

    def episode_init(self, world, *args, **kwargs):
        init(self, world, *args, **kwargs)
        if _BINDING:
            _BINDING[-1].bind(self)

    def capture_text(self):
        trace = _TRACERS.get(self)
        if trace is not None:
            trace.tick(self)
        capture(self)
    GoalEpisode.__init__, Agent.capture_text = episode_init, capture_text
    if scenario is not None:
        vec.draw_start = lambda *a, **k: scenario.start(draws[0], *a, **k)
        vec.draw_combat = lambda *a, **k: scenario.combat(draws[1], *a, **k)
        search.draw_supply = lambda *a, **k: scenario.supply(draws[2], *a, **k)
    try:
        yield
    finally:
        GoalEpisode.__init__, Agent.capture_text = init, capture
        vec.draw_start, vec.draw_combat, search.draw_supply = draws


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
        from sim.spec import OBSERVATION_SHAPES, dtype
        self.env.episodes[b] = episode
        with hooks(self.scenario):
            o = self.env._episode(b)
        obs = {name: value.copy() for name, value in self.obs.items()}
        for name in OBSERVATION_SHAPES:
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
        return list(self.env.goals)

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
        self.batch.step_into(np.asarray(action, np.int32), check=False)
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
        return list(self.batch.goals)

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
    """A policy's sampled actions: the actor of an obs-v2/mask-v2 MaskedRecurrentPolicy on the
    Python side's observation, its LSTM state carried per world and reset at episode starts, each action drawn by
    torch.multinomial from the masked distribution with a seeded generator. The other side gets the same actions after
    its observation compared equal; the policy is never recomputed in C."""

    def __init__(self, policy, spaces, seed=0):
        import torch
        self.torch, self.policy, self.spaces, self.seed = torch, policy, spaces, seed
        self.generator = torch.Generator().manual_seed(seed)
        self.state = None
        policy.set_training_mode(False)

    @classmethod
    def load(cls, path, seed=0):
        from sb3_contrib import RecurrentPPO
        from baseline.policy import check_names
        model = RecurrentPPO.load(path, device='cpu')
        check_names(model.policy)
        return cls(model.policy, model.observation_space.spaces, seed)

    def __call__(self, step, obs, slots, starts):
        torch, policy, n = self.torch, self.policy, len(slots)
        if self.state is None:
            lstm = policy.lstm_actor
            self.state = tuple(torch.zeros(lstm.num_layers, n, lstm.hidden_size) for _ in range(2))
        full = {k: obs[k] if k in obs else np.zeros((n, *space.shape), space.dtype) for k, space in self.spaces.items()}
        with torch.no_grad():
            tensors, _ = policy.obs_to_tensor(full)
            dist, self.state = policy.get_distribution(tensors, self.state, torch.as_tensor(starts, dtype=torch.float32))
            action = torch.multinomial(dist.distribution.probs, 1, generator=self.generator)
        return action.squeeze(1).numpy().astype(np.int32)


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
    has not ported (NotPorted), which stops the comparison and is never a divergence."""
    steps: int = 0
    worlds: int = 0
    ticks: int = 0
    episodes: int = 0
    traced: bool = False
    divergence: Divergence | None = None
    stopped: str = ''
    gap: str = ''


def _plain(value):
    """A start or evidence value as plain JSON types (numpy scalars, tuples), for comparison."""
    return json.loads(json.dumps(value, sort_keys=True, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))


def _bits_of(array):
    array = np.ascontiguousarray(array)
    return array.view(f'u{array.dtype.itemsize}') if array.dtype.kind in 'fiub' else array


def _array_diff(name, a, b):
    """Lines for differing rows of one array (bitwise), floats shown with their bits."""
    ba, bb = _bits_of(np.asarray(a)), _bits_of(np.asarray(b))
    if ba.shape != bb.shape:
        return [f'{name}: shape {ba.shape} vs {bb.shape}']
    where = np.argwhere(ba != bb)
    if not len(where):
        return []
    k = tuple(int(i) for i in where[0])
    va, vb = np.asarray(a)[k], np.asarray(b)[k]
    shown = (lambda v: f'{float(v)!r} ({int(np.asarray(v).view(np.uint32)):08x})') if np.asarray(a).dtype == np.float32 \
        else (lambda v: repr(v.item()))
    return [f'{name}{list(k)}: {shown(va)} != {shown(vb)} ({len(where)} values differ)']


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
    """(where, detail) of world b's first differing decision output, or None."""
    from sim.spec import OBSERVATION_SHAPES
    _, oa, ra, da, ia = out_a
    _, ob, rb, db, ib = out_b
    lines = [line for name in OBSERVATION_SHAPES for line in _array_diff(name, oa[name][b], ob[name][b])]
    if lines:
        return 'observation', lines
    if ra is None:  # a reset
        return None
    if np.float32(ra[b]).view(np.uint32) != np.float32(rb[b]).view(np.uint32):
        return 'reward', [f'reward {ra[b]!r} != {rb[b]!r}']
    if bool(da[b]) != bool(db[b]):
        return 'done', [f'done {bool(da[b])} != {bool(db[b])}']
    lines = [line for name in info_names for line in _array_diff(f'info.{name}', ia[name][b], ib[name][b])]
    return ('info', lines) if lines else None


def _evidence(rows, slot):
    return [_plain({k: v for k, v in row.items() if k != 'observation'}) for row in rows if row['world'] == slot]


def _call(fn, *args):
    try:
        return fn(*args), None
    except Exception as error:  # compared as type and text: both sides must raise alike
        return None, error


def run(a, b, source, key, steps, episode=0, repro=None, observe=None):
    """Step sides a (a PythonSide: its observations feed the source, its decisions and goals label the report) and b
    in lockstep under `source` (step, obs, slots, starts) -> actions for up to `steps` decisions from reset(key): a
    Result whose divergence is the first one (None: none). With `episode` (a repro), world 0 is rebuilt at that episode
    after the reset, and only that episode is compared. repro(slot, episode, actions) renders the report's repro
    line. observe(step, out_a, out_b), if given, sees both sides' outputs once they compared equal (step -1: the
    first observations)."""
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
        if type(err_a) is type(err_b) and str(err_a) == str(err_b):
            result.stopped = f'both sides raised {type(err_a).__name__} at step {step}: {err_a}'
            return None, None, result
        texts = [f'{side.name}: {"no exception" if err is None else f"{type(err).__name__}: {err}"}'
                 for side, err in ((a, err_a), (b, err_b))]
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

    def goal_set(self):
        from baseline.resource_task import GOALS
        return GOALS if self.goals == 'all' else (self.goals,)

    def repro(self, key):
        """repro(slot, episode, actions) -> the one-line command that rebuilds CpuBatch(n=1, first=slot) there."""
        def line(slot, episode, actions):
            args = ['--task', self.task, '--pool', self.pool, '--key', str(key), '--slot', str(slot), '--episode',
                    str(episode), '--actions', ','.join(map(str, actions)) or 'none', '--against', self.against]
            for flag, value in (('--archive', self.archive), ('--supply', self.supply), ('--pack', self.pack),
                                ('--scenario', self.scenario), ('--horizon', self.horizon)):
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
    kwargs = dict(derived=None if derived[0] is None else derived, first=first, task=inputs.task, search=search,
                  goals=inputs.goal_set())
    if inputs.horizon is not None:
        kwargs['horizon'] = inputs.horizon
    env = CpuBatch(tables_, layouts, n, **kwargs)
    a = PythonSide(env, scenario=inputs.scenario, name='cpu')
    if inputs.against == 'cpu':
        return a, PythonSide(twin(env), scenario=inputs.scenario, name='cpu-twin')
    if inputs.against != 'native':
        raise ValueError(f'against cpu or native, not {inputs.against}')
    batch = native.NativeBatch(inputs.pack, n, first=first, task=inputs.task, goals=inputs.goal_set(),
                               horizon=inputs.horizon)
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
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--key', type=int, default=7)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--source', default='random', help='random, biased, idle, walk, eat, list:1,2,3 or policy:PATH[:seed]')
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
                    pack=args.pack)
    if args.slot is not None:
        actions = [] if args.actions in (None, 'none') else [int(v) for v in args.actions.split(',')]
        a, b = build(inputs, 1, first=args.slot)
        result = run(a, b, Replay(actions), args.key, len(actions), episode=args.episode, repro=inputs.repro(args.key))
    else:
        a, b = build(inputs, args.worlds, first=args.first)
        result = run(a, b, make_source(args.source, args.key), args.key, args.steps, repro=inputs.repro(args.key))
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
PARTS = ('cases', 'digests', 'threads', 'soak')
SOAK_N, SOAK_KEY, SOAK_EVERY = 128, 100_000, 1000  # soak worlds per batch, first key, 1 in SOAK_EVERY episodes replayed
SOAK_SOURCES = ('biased', 'policy:A', 'policy:S')
SOAK_FAILURES = 50  # a soak stops resetting after this many errors and unsupported worlds (each one is reported)
# The compact pool FOUR_GOAL_DIGESTS' v5 batch was recorded on (sim/tests/test_workers.py v5_inputs): the digests part
# reads it (gate --digest-pool's default) whatever the gate's --pool, so a gate on another pool (the native gap study's
# arm Layouts) can pass that part. A checkout without logs/ (a cloud machine's) names its uploaded copy.
DIGEST_POOL = native.ROOT / 'logs' / 'sim' / 'world' / 'train-8-nc1-derived.npz'


def gate_configs(mode):
    """(task, goals, first) of every configuration, in run order."""
    firsts = GATE[mode]['firsts']
    from baseline.resource_task import GOALS
    return [(task, 'all', first) for task in GATE_TASKS for first in firsts] + \
        [(task, goal, first) for task in SINGLE_GOAL_TASKS for goal in GOALS for first in firsts]


def gate_cases():
    """(source, scenario) of every configuration's cases."""
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
    from sim.spec import INFO_FIELDS, OBSERVATION_SHAPES
    h = hashlib.sha256()
    for name in OBSERVATION_SHAPES:
        h.update(np.ascontiguousarray(obs[name][b]).tobytes())
    if reward is not None:
        h.update(np.float32(reward[b]).tobytes() + bytes([bool(done[b])]))
        for name in (*INFO_FIELDS, 'made', 'kills', 'facts'):
            h.update(np.ascontiguousarray(info[name][b]).tobytes())
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
        self.tables = load_tables(self.archive)
        self.layouts, derived = load_pool(self.pool)
        self.derived = None if derived[0] is None else derived
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

    def base(self, task):
        """One CpuBatch of GATE_N worlds per task (its layout tables built once; retarget gives the others)."""
        from sim.combat import task_spec
        from sim.vec import CpuBatch
        if task not in self.bases:
            self.bases[task] = CpuBatch(self.tables, self.layouts, GATE_N, derived=self.derived, task=task,
                                        search=self.search if task_spec(task)['search'] else None)
        return self.bases[task]

    def inputs(self, task, goals, scenario=None):
        from sim.combat import task_spec
        supply = task_spec(task)['supply']
        return Inputs(pool=self.pool, archive=self.archive, supply=self.supply if supply else None, task=task, goals=goals,
                      scenario=scenario, against='native', pack=self.pack_path)

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

    def add(self, result, key):
        self.runs += 1
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
                f'runs, {self.decisions} decisions ({self.world_decisions} world), {self.ticks} ticks, {self.episodes} '
                f'episodes ended, {self.divergences} divergences, {self.gaps} gaps'
                f'{f", {len(self.stopped)} stopped alike" if self.stopped else ""} ({self.seconds:.0f} s)')


def gate_case(ctx, env, batch, canon, mode, task, goals, first, source, scenario, log):
    """One case on this configuration's batches: fresh sides around them, every key (or Hypothesis example) run."""
    import time
    g, t0 = GATE[mode], time.time()
    out = CaseResult(case_id(task, goals, first, source, scenario), task, goals, first, source, scenario)
    inputs = ctx.inputs(task, goals, scenario)

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
            result = run(a, b, Listed(values), key, len(values), repro=inputs.repro(key))
            failed = result.divergence is not None or bool(result.gap)
            last[:] = [(result, key)] if failed else last
            out.add(Result(result.steps, result.worlds, result.ticks, result.episodes, result.traced, None,
                           result.stopped, ''), key)  # every example's counts; the failure is recorded once below
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
        keys, steps = g['scenario' if scenario else source.split(':')[0]]
        for key in range(1, keys + 1):
            a, b = sides()
            out.add(run(a, b, ctx.source(source, key), key, steps, repro=inputs.repro(key)), key)
    out.seconds = time.time() - t0
    log(out.line())
    for report in out.reports:
        log(report)
    return out


def gate_cases_part(ctx, mode, shard, skip, emit, log):
    """Every configuration of this shard (index % k == i), each case once (skip(id): leave it out)."""
    import gc
    i, k = shard
    for index, (task, goals, first) in enumerate(gate_configs(mode)):
        if index % k != i:
            continue
        todo = [(src, sc) for src, sc in gate_cases() if not skip(case_id(task, goals, first, src, sc))]
        if not todo:
            continue
        env = retarget(ctx.base(task), first, goals)
        batch = native.NativeBatch(ctx.pack, GATE_N, first=first, task=task, goals=ctx.inputs(task, goals).goal_set())
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
    for task in ('goal-activity-v2', 'goal-activity-v5'):
        runs = {}
        for threads in (1, 3, 8):
            batch, source = native.NativeBatch(ctx.pack, n, task=task, threads=threads), Biased(1)
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
    log(f'[gate] threads: {n} worlds, {steps} decisions, v2 and v5 at 1, 3 and 8 threads: '
        f'{"; ".join(out["failures"]) or "identical"} ({out["seconds"]:.0f} s)')
    return out


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
        self.replayed = self.divergences = self.gaps = self.mismatches = 0
        self.stopped, self.reports = [], []

    def replay(self, task, key, slot, episode, actions, recorded=None):
        from sim.vec import CpuBatch
        from sim.combat import task_spec
        ctx = self.ctx
        if task not in self.envs:
            self.envs[task] = CpuBatch(ctx.tables, ctx.layouts, 1, derived=ctx.derived, task=task,
                                       search=ctx.search if task_spec(task)['search'] else None)
        a = PythonSide(retarget(self.envs[task], slot), canon=self.canon)
        b = NativeSide(native.NativeBatch(ctx.pack, 1, first=slot, task=task))
        differs = []

        def observe(step, out_a, out_b):
            if recorded is None or differs or step + 1 >= len(recorded):
                return
            mine = world_digest(*out_a[1:], 0) if step >= 0 else world_digest(out_a[1], None, None, None, 0)
            if mine != recorded[step + 1]:
                differs.append(step)
        result = run(a, b, Replay([[x] for x in actions]), key, len(actions), episode=episode,
                     repro=ctx.inputs(task, 'all').repro(key), observe=observe)
        self.replayed += 1
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
    target = GATE[mode]['soak']
    segments = [(task, source) for task in GATE_TASKS for source in SOAK_SOURCES]
    per = -(-target // len(segments))
    replays = SoakReplays(ctx)
    t0 = time.time()
    out = dict(id='soak', target=target, ticks=0, decisions=0, episodes=0, errors=[], unsupported=[], segments=[])
    for index, (task, spec) in enumerate(segments):
        batch, key, ticks, t1 = native.NativeBatch(ctx.pack, SOAK_N, task=task, threads=ctx.threads), \
            SOAK_KEY + 1000 * index, 0, time.time()
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
                        replays.replay(task, key, world, episodes[world], actions[world])
                    break
                step += 1
                ticks += int(info['decision_ticks'].sum())
                out['decisions'] += SOAK_N
                for b in np.flatnonzero(info['unsupported']):
                    reason = next((e['reason'] for e in reversed(batch.evidence) if e['world'] == b), '?')
                    out['unsupported'].append(f'{task} {spec} key {key} world {b} episode {episodes[b]}: {reason}')
                    replays.replay(task, key, int(b), episodes[b], actions[b])
                if info['unsupported'].any():
                    break
                for b in range(SOAK_N):
                    if recorded[b] is not None:
                        recorded[b].append(world_digest(obs, reward, done, info, b))
                for b in np.flatnonzero(done):
                    out['episodes'] += 1
                    if recorded[b] is not None:
                        replays.replay(task, key, int(b), episodes[b], actions[b], recorded[b])
                    actions[b] = []
                    recorded[b] = [world_digest(obs, None, None, None, b)] if soak_selected(key, b, episodes[b] + 1) \
                        else None
                starts = done
            key += 1
        out['ticks'] += ticks
        out['segments'].append(dict(task=task, source=spec, ticks=ticks, episodes=out['episodes'] - episodes0,
                                    seconds=round(time.time() - t1, 1)))
        log(f'[gate] soak {task} {spec}: {ticks} ticks, {out["episodes"] - episodes0} episodes, '
            f'{replays.replayed} replayed so far ({time.time() - t1:.0f} s)')
    out.update(replayed=replays.replayed, divergences=replays.divergences, gaps=replays.gaps,
               mismatches=replays.mismatches, stopped=replays.stopped, reports=replays.reports,
               seconds=time.time() - t0)
    log(f'[gate] soak: {out["ticks"]} ticks, {out["decisions"]} world decisions, {out["episodes"]} episodes, '
        f'{len(out["errors"])} errors, {len(out["unsupported"])} unsupported, {out["replayed"]} replayed: '
        f'{out["divergences"]} divergences, {out["gaps"]} gaps, {out["mismatches"]} output mismatches '
        f'({out["seconds"]:.0f} s)')
    for line in out['errors'] + out['unsupported'] + out['reports']:
        log(line)
    return out


# ---- the gate's command line -------------------------------------------------------------------------------------
def _commit():
    import subprocess
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=native.ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=native.ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    return head + ('-dirty' if dirty else '')


def gate_clean(rows):
    """(clean, divergences, gaps): every case, the digests, the thread check and the soak passed."""
    divergences = gaps = 0
    clean = True
    for row in rows:
        if row['id'] in ('digests', 'threads'):
            clean &= not row['failures']
        elif row['id'] == 'soak':
            divergences += row['divergences'] + row['mismatches']
            gaps += row['gaps']
            # an unsupported world passes when its replay flags it alike (a replay that does not is a divergence)
            clean &= not row['errors'] and row['ticks'] >= row['target']
        else:
            divergences += row['divergences']
            gaps += row['gaps']
    return clean and divergences == 0 and gaps == 0, divergences, gaps


def gate_summary(rows, log):
    """The table per task (and goal set), source and scenario, summed over first slots, and the totals."""
    groups = OrderedDict()
    for row in rows:
        if 'task' not in row:
            continue
        k = (row['task'], row['goals'], row['source'], row['scenario'] or '-')
        g = groups.setdefault(k, dict(firsts=[], runs=0, decisions=0, world_decisions=0, ticks=0, episodes=0,
                                      divergences=0, gaps=0, stopped=0))
        g['firsts'].append(row['first'])
        for name in ('runs', 'decisions', 'world_decisions', 'ticks', 'episodes', 'divergences', 'gaps'):
            g[name] += row[name]
        g['stopped'] += len(row['stopped'])
    log('| task | goals | source | scenario | firsts | runs | decisions | world decisions | ticks | episodes | '
        'divergences | gaps | stopped alike |')
    log('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for (task, goals, source, scenario), g in groups.items():
        log(f'| {task} | {goals} | {source} | {scenario} | {",".join(map(str, sorted(g["firsts"])))} | {g["runs"]} | '
            f'{g["decisions"]} | {g["world_decisions"]} | {g["ticks"]} | {g["episodes"]} | {g["divergences"]} | '
            f'{g["gaps"]} | {g["stopped"]} |')
    clean, divergences, gaps = gate_clean(rows)
    commits = sorted({row.get('commit', '?') for row in rows})
    log(f'[gate] {len(rows)} rows, commits {commits}: {"CLEAN" if clean else "NOT CLEAN"}, {divergences} divergences, '
        f'{gaps} gaps')
    return clean


def gate_arguments(parser):
    logs = native.ROOT / 'logs'
    parser.add_argument('--quick', action='store_true', help='the fixed quick subset (about 15 minutes, no soak)')
    parser.add_argument('--archive', default=str(logs / 'sim/data/draynor-nc1.npz'))
    parser.add_argument('--pool', default=str(logs / 'sim/world/train-8-nc1-derived.npz'))
    parser.add_argument('--supply', default=str(logs / 'supply/pools/train-128.npz'))
    parser.add_argument('--digest-pool', default=str(DIGEST_POOL),
                        help="the digests part's v5 pool (default: the one FOUR_GOAL_DIGESTS were recorded on)")
    parser.add_argument('--pack', default=str(native.PACK))
    parser.add_argument('--work', default=str(logs / 'native/gate'), help='the digest packs and the results (ignored)')
    parser.add_argument('--out', help='results, one JSON line per case (default WORK/gate-MODE[-SHARD].jsonl)')
    parser.add_argument('--parts', default=','.join(PARTS), help=f'comma separated: {", ".join(PARTS)}')
    parser.add_argument('--shard', default='0/1', help='i/k: the configurations whose index is i modulo k')
    parser.add_argument('--resume', action='store_true', help='skip the cases already in --out')
    parser.add_argument('--only', help='run only the cases (and parts) whose id matches this regular expression '
                        '(TASK|GOALS|FIRST|SOURCE|SCENARIO; digests, threads, soak): a rerun, not the gate')
    parser.add_argument('--threads', type=int, default=4, help="the soak's NativeBatch threads")
    parser.add_argument('--summary', nargs='*', help='print the merged table of these result files and exit')


def gate_main(args):
    """`python -m sim.native gate`: runs the parts, prints a line per case (and each failure's first-divergence
    report), appends each result to --out, then the table; exit status 0 only when clean."""
    if args.summary is not None:
        rows = [json.loads(line) for path in args.summary for line in Path(path).read_text().splitlines() if line]
        sys.exit(0 if gate_summary(rows, print) else 1)
    mode = 'quick' if args.quick else 'full'
    parts = [p for p in args.parts.split(',') if p]
    if set(parts) - set(PARTS):
        raise ValueError(f'unknown parts {sorted(set(parts) - set(PARTS))}')
    shard = tuple(int(v) for v in args.shard.split('/'))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out or work / f'gate-{mode}{"" if shard == (0, 1) else f"-{shard[0]}of{shard[1]}"}.jsonl')
    done = {json.loads(line)['id'] for line in out.read_text().splitlines() if line} if args.resume and out.exists() \
        else set()
    if not args.resume:
        out.write_text('')
    ctx = GateContext(args.archive, args.pool, args.supply, args.pack, work, args.threads).load(
        policies='cases' in parts or 'soak' in parts)
    commit = _commit()
    only = re.compile(args.only) if args.only else None

    def skip(name):
        return name in done or (only is not None and not only.search(name))

    def log(line):
        print(line, flush=True)

    def emit(row):
        row = dict(row.__dict__ if hasattr(row, '__dict__') else row, commit=commit, mode=mode)
        with out.open('a') as f:
            f.write(json.dumps(row, default=str) + '\n')
    log(f'[gate] {mode} at {commit}: parts {parts}, shard {shard[0]}/{shard[1]}, results {out}')
    log(f'[gate] checkpoints: {", ".join(f"{k} {v[2]}" for k, v in ctx.policies.items())}')
    if 'cases' in parts:
        gate_cases_part(ctx, mode, shard, skip, emit, log)
    if 'digests' in parts and not skip('digests'):
        emit(gate_digests(ctx, log, args.digest_pool))
    if 'threads' in parts and not skip('threads'):
        emit(gate_threads(ctx, mode, log))
    if 'soak' in parts and GATE[mode]['soak'] and not skip('soak'):
        emit(gate_soak(ctx, mode, log))
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    sys.exit(0 if gate_summary(rows, log) else 1)
