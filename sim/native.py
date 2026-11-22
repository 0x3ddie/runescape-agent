"""The native simulator's Python side: build and load of sim/native/*.c, the generated header, the pack format and
NativeBatch, CpuBatch's interface over the C batch (the port's design: one CPython extension, one read-only pack per
pool set, fixed-capacity worlds; the contract is sim/native/rs.h, whose top comment holds the conventions).

`load()` builds the extension once per source set, flags and interpreter into SIM_BUILD (default ~/.cache/rs-sim), as
sim/cpaths.py builds sim/_paths.c, with -O2 -std=c11 -ffp-contract=off -fno-fast-math (Apple clang contracts a*b+c
into an FMA by default, which breaks float equality with the oracle), -flto (LINK) and zlib, through the interpreter's
LDSHARED compiler (PLATFORM and LIBS add what Linux needs). The build writes
rs_gen.h, the Python constants the C side reads (action ids, R_* purposes, the observation and info layouts of
sim/spec.py, EPISODE_FACTS, FAMILIES, engine, agent and task constants), into its own include directory; it is
generated, never edited. A build failure raises: the native engine has no fallback.

A pack (`PackWriter`, `read_pack`; format v1 in rs.h) holds everything C reads from the archive and the pools: types,
content rows, messages as token hashes, names-v1 rows, layouts with their scarce variants, cones and constants. Its
manifest records the sha256 of the oracle sources (ORACLE) and of the inputs and the spec of the name table its rows
come from; `open_pack` refuses a pack whose oracle hash differs from the working tree's unless allow_stale, and with
no override one whose name table differs from rl.names.name_table() or whose recorded input differs from a file the
caller names. Packs are extracted private data: they live under logs/native/ (ignored) and are never published.
Record sections are numpy structured arrays whose dtypes come from the extension's own struct layouts
(`record_dtype`), so writer and reader cannot disagree on offsets. The log1p feature tables are not in the pack:
`log1p_table` builds them in the process that runs the batch, with that host's numpy (rs.h FLOATS).

The exporter that writes a pool set's pack from the archive and the pools is sim/native_export.py.

    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native build
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native export --pool logs/sim/world/train-8-nc1-derived.npz \
        --supply-pool logs/supply/pools/train-128.npz --tables logs/sim/data/draynor-nc1.npz --output logs/native/train-8-nc1-a6c325a-ids3-item3.pack
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native pack-info logs/native/train-<hash>.pack
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native diff --pool logs/sim/world/train-8-nc1-derived.npz ...  (the
        equivalence harness, sim/native_harness.py)
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native gate [--quick]  (the release gate, sim/native_harness.py)
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native free-soak --pack PACK --pool POOL --output DIR  (free-play-v1's
        soak in lockstep with the Python reference, sim/native_harness.py)
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native bench --pack logs/native/train-8-nc1-a6c325a-ids3-item3.pack ...  (throughput,
        sim/native_bench.py)
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native env --engines cpu,native ...  (both engines through the training
        workers, timed and compared, sim/native_bench.py)
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The default pack of the gate, the bench and the native tests (sim/tests PACK): train-8-nc1-rv2's inputs exported again,
# with trees, by the exporter with the respawn fix (15 November 2026: an npc life's npc_respawn event and the XP check's
# respawn cycle), with the record layouts of village-world-v2's C port (each layout record names its world), by the
# exporter whose source it records and against the ids it records ("-ids3": rs.h's RS_EV_NPC_RESPAWN;
# open_pack checks both; the ids hash covers the section table since 13 November); the pack is of draynor-nc1.npz, over
# which firemaking, cooking and the field's Pick do not run. Kept as evidence, each refused on this tree:
# train-8-nc1-a6c325a-ids2.pack (the same inputs before the respawn fix: the oracle and ids of 13 November),
# train-8-nc1-a6c325a-ids.pack (the same pack, its ids without the section table),
# train-8-nc1-a6c325a-worlds.pack (the same inputs and record layouts, its manifest without the exporter check and the
# ids), train-8-nc1-a6c325a.pack (the record
# layouts before the port), train-8-nc1-e2a61c1.pack (the free-play oracle: XP events, free-play-v1, obs-v6),
# train-8-nc1-cc0614f.pack (the phase 2 port's oracle), train-8-nc1-5c14c8b.pack and train-8-nc1-p2.pack (the same inputs
# under the phase 2 contracts, at two earlier commits), train-8-nc1-gap1.pack (at an earlier study's freeze) and rv2
# (older run records name it).
# The -item3 exports (20 November 2026) are the -ids3 packs' inputs exported again after item-v3 changed three ORACLE
# files (rl/obs.py, rl/item_contract.py, rl/act_contract.py): their 34 sections are byte-equal to the -ids3 ones; their
# manifests' oracle record differs, so the -ids3 packs are refused on this tree.
PACK = ROOT / 'logs' / 'native' / 'train-8-nc1-a6c325a-ids3-item3.pack'
# village-world-v2's default pack (the gate's v2 world set, the bench): its training pool (logs/sim/world/train-128-20000-v2.npz,
# plain) over draynor-nc2.npz without trees, exported again as PACK was (15 November 2026; its 13 November export,
# -ids2, is refused on this tree)
PACK_V2 = ROOT / 'logs' / 'native' / 'train-128-20000-v2-a6c325a-ids3-item3.pack'
# SIM_NATIVE_SOURCE: a copy of sim/native/ to build in its place, for tests only (a copy with another rs.h limit, built
# in a subprocess into its own SIM_BUILD; sim/tests/test_native_free_play.py); the library tag digests the copy's files.
SOURCE = Path(os.environ.get('SIM_NATIVE_SOURCE') or Path(__file__).with_name('native'))
# The exporter whose source a pack records (manifest 'exporter') and open_pack checks against this tree's.
EXPORTER = Path(__file__).with_name('native_export.py')
FLAGS = ('-O2', '-std=c11', '-ffp-contract=off', '-fno-fast-math', '-fPIC')
# The extension alone is also built with link-time optimization: calls between its files (rs_tile, rs_wm, the walk
# reads, can_travel) inline as calls within one file do. It changes no operation (the FLAGS above hold at compile
# time, so no a*b+c is fused); the tests' partial builds keep FLAGS alone. SIM_NATIVE_LTO=0 builds without it (a
# toolchain without the LTO plugin), under its own library tag.
LINK = ('-flto',)
# Elsewhere than macOS (Linux x86_64, gcc through LDSHARED): glibc declares batch.c's pthread calls under
# strict -std=c11 only with a feature macro (pack.c sets the same value itself, as does Python.h's pyconfig.h); an
# undeclared call fails the build instead of linking as one returning int; and libm is linked by name (floor is a call
# on baseline x86-64). None of them changes an operation, and macOS keeps the flags its release gate passed with.
DARWIN = sys.platform == 'darwin'
PLATFORM = () if DARWIN else ('-D_POSIX_C_SOURCE=200809L', '-Werror=implicit-function-declaration')
LIBS = ('-lz', '-lpthread') if DARWIN else ('-lz', '-lpthread', '-lm')
# The oracle the pack is exported from and the C port must reproduce (frozen during the port).
ORACLE = tuple(f'sim/{name}.py' for name in ('engine', 'agent', 'task', 'vec', 'combat', 'search', 'audit', 'rs2',
                                             'paths', 'world', 'data', 'spec', 'run')) + \
    ('rl/obs.py', 'rl/act_contract.py', 'rl/feedback_contract.py', 'rl/item_contract.py', 'rl/names.py', 'baseline/resource_task.py', 'baseline/native_value.py', 'baseline/config.py')
PACK_FORMAT = 'rs-native-pack-v1'
PACK_MAGIC, PACK_VERSION, PACK_ALIGN = b'RSNPACK\x01', 1, 64
HEADER = struct.Struct('<8sIIQQQQ16s')    # RsPackHeader, 64 bytes
SECTION = struct.Struct('<IIQQQII')       # RsSection, 40 bytes
# RsDtype codes by numpy dtype (little-endian scalars); RECORD and BYTES are 11 and 0.
DTYPES = {np.dtype('u1'): 1, np.dtype('i1'): 2, np.dtype('<u2'): 3, np.dtype('<i2'): 4, np.dtype('<u4'): 5,
          np.dtype('<i4'): 6, np.dtype('<u8'): 7, np.dtype('<i8'): 8, np.dtype('<f4'): 9, np.dtype('<f8'): 10}
DT_BYTES, DT_RECORD = 0, 11
C_TYPES = {'uint8_t': 'u1', 'int8_t': 'i1', 'uint16_t': '<u2', 'int16_t': '<i2', 'uint32_t': '<u4', 'int32_t': '<i4',
           'uint64_t': '<u8', 'int64_t': '<i8', 'double': '<f8', 'float': '<f4'}
# rs.h RS_TASK_*; v4 (search) refused. v11 is v5 with bindings (sim/combat.py BOUND) that NativeBatch checks in Python,
# so C runs it as v5. free-play-v1 (sim/task.py FREE_PLAY) is RS_TASK_FREE.
TASKS = {'goal-activity-v2': 2, 'goal-activity-v3': 3, 'goal-activity-v5': 5, 'goal-activity-v11': 5, 'free-play-v1': 16}
FREE_TASK = 'free-play-v1'


class NotPorted(RuntimeError):
    """A path the oracle implements and the native port does not (RS_NOT_PORTED): always fatal, never unsupported."""


# ---- the generated header ------------------------------------------------------------------------------------------
def _c(value):
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return value.hex()
    return str(int(value))


def generated_header():
    """rs_gen.h: the Python constants the C side reads, in a fixed order (the build hashes this text)."""
    from baseline import config, resource_task
    import act_contract
    import feedback_contract
    import item_contract
    import obs as rl_obs
    from sim import agent, combat, data, engine, paths, search, spec, task, vec, world

    if not agent.N_ACTIONS == spec.N_ACTIONS == rl_obs.N_ACTIONS:
        raise ValueError('the action tables of sim/agent.py, sim/spec.py and rl/obs.py disagree')
    out = ['/* rs_gen.h: generated by sim/native.py from the Python constants; never edited (sim/native/rs.h). */',
           '#ifndef RS_GEN_H', '#define RS_GEN_H', '#include <stdint.h>',
           '#if defined(__GNUC__)', '#define RS_GEN_UNUSED __attribute__((unused))', '#else', '#define RS_GEN_UNUSED',
           '#endif']

    def section(title):
        out.append(f'\n/* ---- {title} */')

    def define(name, value):
        out.append(f'#define {name} {_c(value)}')

    def table(ctype, name, values):
        out.append(f'static const {ctype} {name}[{len(values)}] RS_GEN_UNUSED = {{{", ".join(map(_c, values))}}};')

    def strings(name, values):  # C string literals (json escapes are C's for these ASCII texts)
        if not all(v.isascii() for v in values):
            raise ValueError(f'{name}: a C string table holds ASCII only')
        out.append(f'static const char *const {name}[{len(values)}] RS_GEN_UNUSED = '
                   f'{{{", ".join(json.dumps(v) for v in values)}}};')

    def names(prefix, items, start=0):
        for k, item in enumerate(items, start):
            define(f'{prefix}{item.upper().replace("-", "_")}', k)

    section('actions (sim/agent.py:19-24) and decision constants (agent.py:25-41)')
    for name in ('N_ACTIONS', 'A_WAIT', 'A_ACT0', 'N_ACT', 'A_HELD0', 'N_HELD', 'N_INV_SEL', 'A_DROP0', 'A_DEPOSIT0',
                 'A_WITHDRAW0', 'A_CLOSE_IF', 'A_DIALOG_CONTINUE', 'A_DIALOG_CHOICE0', 'N_DIALOG_CHOICE', 'A_INV_PREV',
                 'A_INV_NEXT', 'A_BANK_PREV', 'A_BANK_NEXT', 'A_STYLE0', 'A_USE0', 'DROP_MIN_STACKS',
                 'K_TICKS', 'NAV_STALL_TICKS', 'ACT_STALL_TICKS', 'OPTION_HARD_TIMEOUT_TICKS', 'DECISION_WAIT_TICKS',
                 'MAX_EPISODE_TICKS', 'REFUSED_TICKS', 'IDLE_GRACE_TICKS', 'ACT_SCAN_TILES', 'ACT_CLUSTER_TILES',
                 'LOC_SCAN_TILES', 'GROUND_SCAN_TILES', 'NPC_SCAN_TILES', 'N_NPC_OBS', 'N_LOC_OBS', 'N_GROUND_OBS',
                 'VISIT_CELL', 'CHAT_KEEP'):
        define(f'RS_{name}', getattr(agent, name))
    # act-v3's clicks (rl/act_contract.py): slot k's offset and row
    for name in ('N_CLICK', 'A_CLICK_LOW', 'A_CLICK_HIGH', 'N_CLICK_LOW'):
        define(f'RS_{name}', getattr(act_contract, name))
    table('int8_t', 'RS_CLICK_DX', [o[0] for o in act_contract.CLICK_OFFSETS])
    table('int8_t', 'RS_CLICK_DZ', [o[1] for o in act_contract.CLICK_OFFSETS])
    table('int8_t', 'RS_CLICK_ROWS', list(act_contract.CLICK_ROWS))
    if agent.CLICK_OFFSETS != act_contract.CLICK_OFFSETS or agent.CLICK_ROWS != act_contract.CLICK_ROWS:
        raise ValueError('sim/agent.py and rl/act_contract.py hold different click tables')
    define('RS_AGENT_MAX_DECISIONS', agent.MAX_DECISIONS)
    define('RS_LOGIN_X', agent.LOGIN_TILE[0])
    define('RS_LOGIN_Z', agent.LOGIN_TILE[1])
    names('RS_ACTFAM_', agent.FAMILIES)
    define('RS_N_ACTFAMS', len(agent.FAMILIES))
    if agent.FAMILIES != feedback_contract.FAMILIES:
        raise ValueError('sim/agent.py and rl/feedback_contract.py name the action families differently')

    section('engine (sim/engine.py:24-117, 2078-2079)')
    for name in ('APNPC1', 'APNPCU', 'APOBJ1', 'APLOC1', 'APLOCU', 'AP_TO_OP', 'XP_RATE', 'WALK_BLOCKED', 'AFK_EVENTRATE',
                 'NPC_ACTIVE_RADIUS', 'WEARPOS_FRONT', 'WEARPOS_RHAND', 'WEARPOS_QUIVER', 'NOMOVE', 'STACK_LIMIT',
                 'LOC_DURATION', 'WALL_STRAIGHT', 'WALL_DIAGONAL', 'PLAYER_COUNT', 'STAB', 'SLASH', 'CRUSH', 'RANGED_STYLE',
                 'MAGIC_STYLE', 'NPC_ATTACK', 'NPC_DEFENCE', 'NPC_STRENGTH', 'NPC_HP', 'NPC_MAGIC'):
        define(f'RS_{name}', getattr(engine, name))
    define('RS_AFK_CHANCE1', float(engine.AFK_CHANCE1))
    for name in sorted(n for n in dir(engine) if n.startswith('MODE_')):
        define(f'RS_{name}', getattr(engine, name))
    names('RS_SKILL_', engine.SKILLS)
    define('RS_N_SKILLS', len(engine.SKILLS))
    for name in ('R_AFK', 'R_WANDER_GO', 'R_WANDER_DX', 'R_WANDER_DZ', 'R_SCRIPT', 'R_NPC_SCRIPT', 'R_PACE'):
        define(f'RS_{name}', getattr(engine, name))
    table('int8_t', 'RS_STYLE_BONUS_STYLES', list(engine.STYLE_BONUSES))
    table('int8_t', 'RS_STYLE_BONUSES', [b for v in engine.STYLE_BONUSES.values() for b in v])  # 4 per style row

    section('collision and paths (sim/data.py:17-22, sim/paths.py:23-34, 261, 278)')
    define('RS_FLAG_NPC', data.FLAG_NPC)
    define('RS_FLAG_NULL', data.FLAG_NULL)
    for name, value in data.BLOCK.items():
        define(f'RS_BLOCK_{name}', value)
    for name in ('D_NORTH', 'D_EAST', 'D_SOUTH', 'D_WEST', 'SEARCH', 'HALF'):
        define(f'RS_{name}', getattr(data, name))
    for name in ('MAX_WAYPOINTS', 'WALL_NORTH', 'WALL_EAST', 'WALL_SOUTH', 'WALL_WEST', 'PROJ_NORTH', 'PROJ_EAST',
                 'PROJ_SOUTH', 'PROJ_WEST', 'REACH_BOX'):
        define(f'RS_{name}', getattr(paths, name))
    # the client's tryMove (paths.py client_*): its scene, fallback, collision bits and the walls a step crosses
    for name in ('SCENE', 'NEAREST_STEPS', 'SCENE_UNREACHED', 'C_W_N', 'C_W_E', 'C_W_S', 'C_W_W', 'C_GRND', 'C_BOUNDS'):
        define(f'RS_{name}', getattr(paths, name))
    for name, value in paths.PL_WALK.items():
        define(f'RS_PL_WALK_{name}', value)
    define('RS_FLAG_LOC', paths.LOC)
    define('RS_FLAG_LOC_PROJ', paths.LOC_PROJ)
    table('int8_t', 'RS_STEP_DX', [s[0] for s in paths.STEPS])
    table('int8_t', 'RS_STEP_DZ', [s[1] for s in paths.STEPS])
    table('uint8_t', 'RS_STEP_BACK', [int(b) for b in paths.BACK])
    table('int8_t', 'RS_NEIGHBOUR_DX', [d[0] for d in paths.NEIGHBOURS])
    table('int8_t', 'RS_NEIGHBOUR_DZ', [d[1] for d in paths.NEIGHBOURS])

    section('tasks (sim/task.py:55-64, sim/combat.py:30-71, sim/search.py:77, 241-255, sim/vec.py:40-46, 229)')
    for name in ('R_LAYOUT', 'R_TILE', 'R_WEAPON', 'R_LEVEL', 'R_WORLD', 'WORLD_START_TICK'):
        define(f'RS_{name}', getattr(task, name))
    names('RS_FAM_', task.FAMILIES)
    define('RS_N_FAMILIES', len(task.FAMILIES))
    names('RS_GOAL_', resource_task.GOALS)
    define('RS_N_GOALS', len(resource_task.GOALS))
    define('RS_GOAL_VEC', len(resource_task.GOAL_VEC))  # goal-v3's entries
    table('int32_t', 'RS_GOAL_TARGETS', [resource_task.GOAL_TARGETS[g] for g in resource_task.GOALS])
    for name in ('R_GROUP', 'R_HP', 'R_FOOD_N', 'R_FOOD', 'R_WORN', 'R_MUGGER'):
        define(f'RS_{name}', getattr(combat, name))
    c = combat.COMBAT
    define('RS_COMBAT_TARGET', c['target'])
    define('RS_COMBAT_HORIZON', c['horizon'])
    define('RS_COMBAT_WORN', float(c['worn']))
    define('RS_COMBAT_CAMP_LO', c['mugger_camp'][0])
    define('RS_COMBAT_CAMP_HI', c['mugger_camp'][1])
    define('RS_COMBAT_MUGGER_START', c['mugger_start'])
    define('RS_COMBAT_N_FOOD', len(c['food']))
    names('RS_GROUP_', combat.GROUPS)
    define('RS_N_GROUPS', len(combat.GROUPS))
    groups = [c['groups'][g] for g in combat.GROUPS]
    table('int32_t', 'RS_GROUP_HP_LO', [g['hp'][0] for g in groups])
    table('int32_t', 'RS_GROUP_HP_HI', [g['hp'][1] for g in groups])
    table('uint8_t', 'RS_GROUP_HP_ABS', [bool(g.get('abs')) for g in groups])
    table('int32_t', 'RS_GROUP_FOOD_LO', [g['food'][0] for g in groups])
    table('int32_t', 'RS_GROUP_FOOD_HI', [g['food'][1] for g in groups])
    table('int32_t', 'RS_GROUP_MUGGERS', [g['muggers'] for g in groups])
    define('RS_R_SEARCH', search.R_SEARCH)
    for name in ('R_SUPPLY_GROVE', 'R_SUPPLY_TILE', 'R_SUPPLY_VARIANT'):
        define(f'RS_{name}', getattr(search, name))
    for name in ('near', 'apart', 'linkage', 'ample', 'horizon', 'cap', 'quota'):
        define(f'RS_SUPPLY_{name.upper()}', search.SUPPLY[name])
    names('RS_VARIANT_', search.SUPPLY['variants'])
    define('RS_SUPPLY_TABLES', vec.SUPPLY_TABLES)
    # every non-combat slot's enemy set (GoalEpisode.enemies, info kills, the v2 melee reward)
    table('int32_t', 'RS_MELEE_ENEMIES', sorted(resource_task.MELEE_ENEMIES))
    define('RS_N_MELEE_ENEMIES', len(resource_task.MELEE_ENEMIES))
    # World.exclusions as bits (rs.h World.excl): bit k is engine.EXCLUSIONS[k]; the tasks' declared sets
    for k, name in enumerate(engine.EXCLUSIONS):
        define(f'RS_EXCLUDE_{name.upper()}', 1 << k)
    bits = lambda names: sum(1 << engine.EXCLUSIONS.index(name) for name in names)
    define('RS_COMBAT_EXCLUSIONS', bits(combat.COMBAT['exclusions']))
    define('RS_SUPPLY_EXCLUSIONS', bits(search.SUPPLY['exclusions']))
    define('RS_R_ACTION', vec.R_ACTION)
    names('RS_FACT_', vec.EPISODE_FACTS)
    define('RS_N_FACTS', len(vec.EPISODE_FACTS))
    stats = list(spec.START_LEVELS)
    table('int8_t', 'RS_START_LEVEL_SKILL', [engine.SKILLS.index(s) for s in stats])
    table('int8_t', 'RS_START_LEVEL_N', [len(spec.START_LEVELS[s]) for s in stats])
    width = max(len(v) for v in spec.START_LEVELS.values())
    table('int8_t', 'RS_START_LEVEL_CHOICES', [v for s in stats for v in
                                               list(spec.START_LEVELS[s]) + [0] * (width - len(spec.START_LEVELS[s]))])
    define('RS_START_LEVEL_WIDTH', width)
    define('RS_N_START_WEAPONS', len(spec.START_WEAPONS))
    define('RS_N_START_KIT', len(spec.START_KIT))
    define('RS_HORIZON_TICKS', spec.HORIZON_TICKS)
    for name in ('TERMINAL_RUNNING', 'TERMINAL_HORIZON', 'TERMINAL_DEATH'):
        define(f'RS_{name}', getattr(spec, name))
    # the door scripts' offsets (engine.py DOOR_OPEN, DOOR_CLOSE): row k * 4 + angle, k 0 a straight wall, 1 a diagonal
    if list(engine.DOOR_OPEN) != [engine.WALL_STRAIGHT, engine.WALL_DIAGONAL] or list(engine.DOOR_CLOSE) != list(engine.DOOR_OPEN):
        raise ValueError('engine.DOOR_OPEN and DOOR_CLOSE name other shapes than the C door')
    for name in ('DOOR_OPEN', 'DOOR_CLOSE'):
        rows = [o for shape in getattr(engine, name).values() for o in shape]
        table('int8_t', f'RS_{name}_DX', [o[0] for o in rows])
        table('int8_t', f'RS_{name}_DZ', [o[1] for o in rows])
    section('free-play-v1 (sim/task.py FREE_PLAY, draw_life; sim/combat.py VILLAGES, village_of, draw_village)')
    define('RS_R_LIFE', task.R_LIFE)
    free = [engine.SKILLS.index(s) for s in task.FREE_PLAY['skills']]
    define('RS_FREE_N_SKILLS', len(free))
    table('int8_t', 'RS_FREE_SKILLS', free)  # S, in FREE_PLAY's order (the pay's sum order)
    # a layout record's world (rs.h RsLayoutRec.world: 1 + its sim/world.py WORLDS index) and the village world that
    # plays it (combat.village_of: its VILLAGES index, -1 none); each village world's start group, exclusions, whether
    # it places muggers (and so lets a batch ask for them), whether it declares its own loot list, and draw_village's
    # refusal texts
    declared, villages = list(world.WORLDS), list(combat.VILLAGES.values())
    define('RS_N_DECLARED', len(declared))
    define('RS_N_VILLAGES', len(villages))
    table('int8_t', 'RS_DECLARED_VILLAGE', [next((i for i, v in enumerate(villages) if v['layouts'] == name), -1)
                                            for name in declared])
    table('int8_t', 'RS_VILLAGE_GROUP', [combat.GROUPS.index(v['start_group']) for v in villages])
    table('uint8_t', 'RS_VILLAGE_EXCLUSIONS', [bits(v['exclusions']) for v in villages])
    table('uint8_t', 'RS_VILLAGE_PLACE_MUGGERS', [v.get('muggers') is not False for v in villages])
    table('uint8_t', 'RS_VILLAGE_OWN_LOOT', [v.get('loot_outside') is not None for v in villages])
    strings('RS_VILLAGE_NO_MUGGERS', [f"{v['version']} holds no muggers: pass False" for v in villages])
    out.append(f'#define RS_VILLAGE_NONE {json.dumps("free play plays the layouts of " + str([v["layouts"] for v in villages]))}')
    define('RS_ENGINE_XP_RATE', engine.XP_RATE)  # World's default: the goal tasks' rate
    define('RS_MAX_HORIZON', resource_task.MAX_HORIZON)  # a life's bound (check_life)
    cfg = config.TASK_CFG
    define('RS_CFG_PACE_JITTER', cfg.get('paceJitterTicks', 0))
    define('RS_CFG_DECISION_POINTS', bool(cfg.get('decisionPoints')))
    define('RS_CFG_MAX_DECISIONS', cfg.get('maxDecisions', agent.MAX_DECISIONS))
    define('RS_CFG_MAX_EPISODE_TICKS', cfg.get('maxEpisodeTicks', agent.MAX_EPISODE_TICKS))

    section('the projection (rl/obs.py and the contracts it imports)')
    for name in ('N_NPC', 'N_INV', 'N_WORN', 'PLAYER_VEC', 'TASK_VEC', 'PROG_VEC'):
        define(f'RS_OBSV_{name}', getattr(rl_obs, name))
    define('RS_OBS_MAX_DECISIONS', rl_obs.MAX_DECISIONS)
    for name in ('TEXT_DIM', 'CHOICE_DIM', 'N_CHOICES', 'FEEDBACK_FEATS', 'TEXT_FEATS'):
        define(f'RS_{name}', getattr(feedback_contract, name))
    names('RS_END_', feedback_contract.END_REASONS)
    define('RS_ACT_FEATS', act_contract.ACT_FEATS)
    define('RS_HELD_FEATS', act_contract.HELD_FEATS)
    define('RS_CLICK_FEATS', act_contract.CLICK_FEATS)
    define('RS_CLICK_SCALE', act_contract.CLICK_SCALE)
    define('RS_BANK_SCALE', act_contract.BANK_SCALE)
    define('RS_NO_BANK', float(act_contract.NO_BANK))
    for name in ('N_BANK_SEL', 'BANK_CAPACITY', 'MAX_BANK_PAGES'):
        define(f'RS_{name}', getattr(item_contract, name))

    section('observation layout (sim/spec.py OBSERVATION_SHAPES and dtype(); a packed row: keys in order)')
    keys, offset = list(spec.OBSERVATION_SHAPES), 0
    elems, offsets, nbytes = [], [], []
    for k, key in enumerate(keys):
        shape, dt = spec.OBSERVATION_SHAPES[key], np.dtype(spec.dtype(key))
        n = int(np.prod(shape))
        offset = -(-offset // dt.itemsize) * dt.itemsize
        define(f'RS_OBS_{key.upper()}', k)
        define(f'RS_OBS_{key.upper()}_N', n)
        define(f'RS_OBS_{key.upper()}_OFF', offset)
        elems.append(n)
        offsets.append(offset)
        nbytes.append(n * dt.itemsize)
        offset += n * dt.itemsize
    define('RS_OBS_N_KEYS', len(keys))
    define('RS_OBS_BYTES', -(-offset // 8) * 8)
    out.append(f'static const char *const RS_OBS_NAMES[{len(keys)}] RS_GEN_UNUSED = '
               f'{{{", ".join(json.dumps(k) for k in keys)}}};')
    out.append(f'static const char RS_OBS_FORMAT[{len(keys)}] RS_GEN_UNUSED = '
               f'{{{", ".join(repr(np.dtype(spec.dtype(k)).char) for k in keys)}}};')
    table('uint8_t', 'RS_OBS_NDIM', [len(spec.OBSERVATION_SHAPES[k]) for k in keys])
    table('int32_t', 'RS_OBS_SHAPE', [d for k in keys for d in (list(spec.OBSERVATION_SHAPES[k]) + [1])[:2]])
    table('int32_t', 'RS_OBS_NELEM', elems)
    table('int32_t', 'RS_OBS_OFFSET', offsets)
    table('int32_t', 'RS_OBS_NBYTES', nbytes)
    # obs-v6 (rl/obs.py project_v6, sim/spec.py OBSERVATION_SHAPES_V6): the packed row above holds obs-v5's arrays and,
    # after them, each SKILL_ORDER skill's level progress (RS_OBS_PROGRESS_OFF); a v6 key is its obs-v5 key's bytes
    # (RS_OBS6_SRC, from RS_OBS6_SKIP bytes on: task_vec drops its decision count), and prog_vec is followed by the
    # progress block. The client's level table (rl/obs.py LEVEL_XP) is the progress's.
    v6 = spec.OBSERVATION_SHAPES_V6
    if [k for k in keys if k in v6] != list(v6):
        raise ValueError('obs-v6 keys are not obs-v5 keys in obs-v5 order')
    define('RS_OBS_PROGRESS_OFF', -(-offset // 8) * 8)
    define('RS_OBS_PROGRESS_N', len(rl_obs.SKILL_ORDER))
    define('RS_OBS_ROW_BYTES', -(-(-(-offset // 8) * 8 + 4 * len(rl_obs.SKILL_ORDER)) // 8) * 8)
    define('RS_OBSV_PROG_VEC_V6', rl_obs.PROG_VEC_V6)
    define('RS_OBSV_TASK_VEC_V6', rl_obs.TASK_VEC_V6)
    if rl_obs.PROG_VEC_V6 != rl_obs.PROG_VEC + len(rl_obs.SKILL_ORDER) or rl_obs.TASK_VEC_V6 != rl_obs.TASK_VEC - 1:
        raise ValueError('obs-v6 prog_vec or task_vec is not obs-v5\'s with the progress added, the decision count gone')
    table('int32_t', 'RS_OBS_LEVEL_XP', list(rl_obs.LEVEL_XP))
    v6keys = list(v6)
    for k, key in enumerate(v6keys):
        define(f'RS_OBS6_{key.upper()}', k)
    define('RS_OBS6_N_KEYS', len(v6keys))
    out.append(f'static const char *const RS_OBS6_NAMES[{len(v6keys)}] RS_GEN_UNUSED = '
               f'{{{", ".join(json.dumps(k) for k in v6keys)}}};')
    out.append(f'static const char RS_OBS6_FORMAT[{len(v6keys)}] RS_GEN_UNUSED = '
               f'{{{", ".join(repr(np.dtype(spec.dtype(k)).char) for k in v6keys)}}};')
    table('uint8_t', 'RS_OBS6_NDIM', [len(v6[k]) for k in v6keys])
    table('int32_t', 'RS_OBS6_SHAPE', [d for k in v6keys for d in (list(v6[k]) + [1])[:2]])
    table('int32_t', 'RS_OBS6_NELEM', [int(np.prod(v6[k])) for k in v6keys])
    table('int32_t', 'RS_OBS6_NBYTES', [int(np.prod(v6[k])) * np.dtype(spec.dtype(k)).itemsize for k in v6keys])
    table('int32_t', 'RS_OBS6_SRC', [keys.index(k) for k in v6keys])
    table('int32_t', 'RS_OBS6_SKIP', [4 if k == 'task_vec' else 0 for k in v6keys])

    section('step info (sim/spec.py INFO_FIELDS order; sim/workers.py fields adds made, kills, facts, reward, done)')
    info = list(spec.INFO_FIELDS)
    for k, name in enumerate(info):
        define(f'RS_INFO_{name.upper()}', k)
    define('RS_INFO_N', len(info))
    out.append(f'static const char *const RS_INFO_NAMES[{len(info)}] RS_GEN_UNUSED = '
               f'{{{", ".join(json.dumps(k) for k in info)}}};')
    out.append(f'static const char RS_INFO_FORMAT[{len(info)}] RS_GEN_UNUSED = '
               f'{{{", ".join(repr(np.dtype(spec.INFO_FIELDS[k]).char) for k in info)}}};')
    out += ['', '#endif /* RS_GEN_H */', '']
    return '\n'.join(out)


# ---- build and load ------------------------------------------------------------------------------------------------
def build_dir():
    return Path(os.environ.get('SIM_BUILD') or Path.home() / '.cache' / 'rs-sim')


def sources():
    return sorted(SOURCE.glob('*.c'))


def build_flags():
    """The extension's compile and link flags on this host: FLAGS, PLATFORM and LINK (unless SIM_NATIVE_LTO=0)."""
    return (*FLAGS, *PLATFORM, *(() if os.environ.get('SIM_NATIVE_LTO') == '0' else LINK))


def linker():
    """The interpreter's own LDSHARED command (its compiler and shared-object flags), or the system `cc` with them
    when that compiler is not installed here (an interpreter built on another machine)."""
    link = shlex.split(sysconfig.get_config_var('LDSHARED') or 'cc -shared')
    if shutil.which(link[0]) is None:
        link = ['cc', *(['-bundle', '-undefined', 'dynamic_lookup'] if DARWIN else ['-shared'])]
    return link


def _command(header_dir, out):
    return [*linker(), *build_flags(), '-I', sysconfig.get_paths()['include'], '-I', str(header_dir), '-I', str(SOURCE),
            *map(str, sources()), '-o', str(out), *LIBS]


def compiler():
    """The first line of the linker's compiler's --version (the build's record), or None."""
    try:
        done = subprocess.run([linker()[0], '--version'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return (done.stdout or done.stderr).strip().splitlines()[0] if done.returncode == 0 else None


def library(header=None):
    """The built file for this source set, generated header, flags and interpreter (and the header it needs)."""
    header = generated_header() if header is None else header
    digest = hashlib.sha256()
    for path in sorted(SOURCE.glob('*.[ch]')):
        digest.update(path.name.encode() + b'\0' + path.read_bytes() + b'\0')
    digest.update(header.encode() + ' '.join(build_flags()).encode() + sys.version.encode()
                  + sysconfig.get_platform().encode())
    tag = digest.hexdigest()[:16]
    return build_dir() / f'_native-{tag}{sysconfig.get_config_var("EXT_SUFFIX")}', header


def build(out, header):
    """Compile and link every sim/native/*.c into `out` (written to a temporary name, then renamed into place)."""
    include = out.with_name(out.name.split('.')[0] + '-include')
    include.mkdir(parents=True, exist_ok=True)
    gen = include / 'rs_gen.h'
    temp_gen = include / f'rs_gen.h.{os.getpid()}.tmp'
    temp_gen.write_text(header)
    os.replace(temp_gen, gen)
    temp = out.with_name(f'{out.name}.{os.getpid()}.tmp')
    command = _command(include, temp)
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        temp.unlink(missing_ok=True)
        raise OSError(f'{" ".join(command)} failed: {done.stderr.strip()[-4000:]}')
    os.replace(temp, out)


_MODULE = None


def check_interpreter(total=None):
    """Refuse an interpreter whose sum() of floats is not compensated. The oracle sums floats with sum() where they
    reach a compared value (free-play-v1's pay: the fractional level gains over S), and CPython sums floats with
    Neumaier's compensation since 3.12, plainly before; the C twin mirrors the compensated sum (sim/native/pysum.h), so
    under a plain sum() the two engines' pay would differ in its last bits with no other error. The probe: 1e16 + 1.0
    rounds the 1.0 away, which only the compensation keeps. total: the sum to probe (default: the builtin, looked up
    at the call)."""
    if (sum if total is None else total)([1e16, 1.0, -1e16]) != 1.0:
        raise RuntimeError(f'this interpreter (Python {sys.version.split()[0]}) adds floats in sum() plainly, not '
                           "with the compensation of CPython 3.12 and later that the C twin mirrors (sim/native/pysum.h): "
                           "the oracle's free-play pay would differ from the C engine's; use Python 3.12 or later")


def load():
    """The extension module sim._native (built on first use), in an interpreter whose sum() the C twin mirrors
    (check_interpreter)."""
    global _MODULE
    if _MODULE is None:
        check_interpreter()
        out, header = library()
        if not out.exists():
            build(out, header)
        loader = importlib.machinery.ExtensionFileLoader('sim._native', str(out))
        spec = importlib.util.spec_from_file_location('sim._native', out, loader=loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        _MODULE = module
    return _MODULE


# ---- pack format v1 ------------------------------------------------------------------------------------------------
def record_dtype(name):
    """The numpy dtype of a pack record struct (rs.h), from the extension's own layout of it."""
    size, fields = load().struct_layouts()[name]
    names, formats, offsets = [], [], []
    for field, ctype, offset, count in fields:
        names.append(field)
        formats.append((C_TYPES[ctype], (count,)) if count > 1 else C_TYPES[ctype])
        offsets.append(offset)
    return np.dtype(dict(names=names, formats=formats, offsets=offsets, itemsize=size))


def layout_hash():
    """sha256 of the record layouts a pack is written against (a struct change invalidates every pack)."""
    layouts = load().struct_layouts()
    return hashlib.sha256(json.dumps(layouts, sort_keys=True).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def oracle_hashes(root=ROOT):
    return {name: sha256_file(Path(root) / name) for name in ORACLE}


def oracle_hash(hashes=None):
    hashes = oracle_hashes() if hashes is None else hashes
    return hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest()


def input_hashes(paths):
    """{name: sha256} of the exporter's input files ({name: path}: the archive, the pools), for manifest(inputs=)."""
    return {name: sha256_file(path) for name, path in paths.items()}


def names_spec(table):
    """The manifest's record of a names-v1 table (rl/names.py NameTable.spec: rows, vector width, file digest)."""
    rows, dim, digest = table.spec
    return dict(rows=int(rows), dim=int(dim), digest=str(digest))


def exporter_hash():
    """sha256 of this tree's exporter source (sim/native_export.py), which a pack records (manifest 'exporter')."""
    return sha256_file(EXPORTER)


def ids_hash():
    """sha256 of the C-side id tables a pack's rows are written against beyond the record layouts: the library's script
    table (RsScript ids: name, Python qualname, port status, which the exporter maps the oracle's scripts through), its
    section table (the ids a pack's sections are filed under), rs.h's integer contract (sim/native_export.py contract:
    the enum values, flags, text keys, capacities and other #defines the exporter reads) and the declared worlds
    (sim/world.py WORLDS, whose order numbers a layout record's world). A pack written against other ids runs other
    scripts, texts, flags or worlds with no other error, so open_pack refuses it with no override. Not here: the ids the
    Python side gives a pack's rows (engine.SKILLS' order, the goal tables), which the oracle hash alone covers
    (allow_stale overrides it), and the archive's own rows, which the pack's recorded inputs pin (open_pack checks them
    against the files a caller names, with no override)."""
    from sim.native_export import contract
    from sim.world import WORLDS
    module = load()
    table = dict(scripts=[list(row) for row in module.scripts()], sections=module.constants()['sections'],
                 contract=contract(), declared=list(WORLDS))
    return hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()


def manifest(inputs=None, names=None, **extra):
    """A pack manifest's required part: format, the oracle's file and combined hashes, the record layouts' hash, the
    exporter (its version and this tree's exporter_hash) and the id tables' hash (ids_hash), the exporter's input hashes
    (input_hashes: archive meta, pools) and the name table its STR_ROW rows come from (a NameTable, or None for a pack
    without name rows); `extra` adds exporter fields (counts)."""
    from sim.native_export import EXPORT_VERSION
    hashes = oracle_hashes()
    return dict(format=PACK_FORMAT, version=PACK_VERSION, oracle=hashes, oracle_hash=oracle_hash(hashes),
                layout_hash=layout_hash(), exporter=dict(version=EXPORT_VERSION, sha256=exporter_hash()), ids=ids_hash(),
                inputs=dict(inputs or {}), names=None if names is None else names_spec(names), **extra)


class PackWriter:
    """Writes a pack: add() sections (ids from the extension's section table), then close(). Every section is 64-byte
    aligned; the file is written to a temporary name and renamed into place. Arrays are kept as given and written
    through their buffers (no byte copies), so a pack of large heaps needs its data in memory once."""

    def __init__(self, path, manifest):
        self.path, self.manifest, self.sections = Path(path), dict(manifest), {}

    def add(self, sid, data, record=None):
        """A section: bytes (BYTES), a numpy array of a supported scalar dtype, a list of such arrays of one dtype
        (written one after another as one section: the exporter's heaps), or `record` (a struct name of
        struct_layouts) with an array of record_dtype(record)."""
        sid = int(sid)
        if sid in self.sections:
            raise ValueError(f'section {sid} written twice')
        if record is not None:
            array = np.ascontiguousarray(data)
            if array.dtype != record_dtype(record):
                raise ValueError(f'section {sid}: dtype {array.dtype} is not {record} {record_dtype(record)}')
            entry = (DT_RECORD, [array], array.size, array.dtype.itemsize)
        elif isinstance(data, (bytes, bytearray)):
            entry = (DT_BYTES, [bytes(data)], len(data), 1)
        else:
            parts = [np.ascontiguousarray(part) for part in (data if isinstance(data, list) else [data])]
            kinds = {part.dtype.newbyteorder('<') if part.dtype.byteorder in '=|' else part.dtype for part in parts}
            code = DTYPES.get(kinds.pop()) if len(kinds) == 1 else None
            if code is None:
                raise ValueError(f'section {sid}: unsupported or mixed dtypes {sorted(map(str, kinds))}')
            parts = [part.astype(part.dtype.newbyteorder('<'), copy=False) for part in parts]
            entry = (code, parts, sum(part.size for part in parts), parts[0].dtype.itemsize)
        self.sections[sid] = entry

    def close(self):
        text = json.dumps(self.manifest, sort_keys=True).encode()
        table_off = -(-(HEADER.size + len(text)) // 8) * 8
        offset = table_off + SECTION.size * len(self.sections)
        rows, blobs = [], []
        for sid in sorted(self.sections):
            code, parts, count, elem = self.sections[sid]
            nbytes = sum(len(part) if isinstance(part, bytes) else part.nbytes for part in parts)
            offset = -(-offset // PACK_ALIGN) * PACK_ALIGN
            rows.append(SECTION.pack(sid, code, offset, nbytes, count, elem, 0))
            blobs.append((offset, parts))
            offset += nbytes
        size = -(-offset // PACK_ALIGN) * PACK_ALIGN
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f'{self.path.name}.{os.getpid()}.tmp')
        with open(temp, 'wb') as stream:
            stream.write(HEADER.pack(PACK_MAGIC, PACK_VERSION, len(rows), HEADER.size, len(text), table_off, size,
                                     bytes(16)))
            stream.write(text)
            stream.write(bytes(table_off - HEADER.size - len(text)))
            stream.write(b''.join(rows))
            for at, parts in blobs:
                stream.write(bytes(at - stream.tell()))
                for part in parts:
                    stream.write(part if isinstance(part, bytes) else part.reshape(-1).view(np.uint8).data)
            stream.write(bytes(size - stream.tell()))
        os.replace(temp, self.path)
        return self.path


def read_pack(path):
    """(header dict, manifest dict, {section id: (dtype code, count, elem, read-only array or bytes)}), memory-mapped."""
    path = Path(path)
    raw = np.memmap(path, np.uint8, 'r')
    magic, version, n, manifest_off, manifest_len, table_off, size, _ = HEADER.unpack(raw[:HEADER.size].tobytes())
    if magic != PACK_MAGIC or version != PACK_VERSION or size != raw.size:
        raise ValueError(f'{path}: not a v{PACK_VERSION} pack of {raw.size} bytes')
    header = dict(version=version, n_sections=n, manifest_off=manifest_off, manifest_len=manifest_len,
                  table_off=table_off, file_size=size)
    meta = json.loads(raw[manifest_off:manifest_off + manifest_len].tobytes())
    sections = {}
    for k in range(n):
        sid, code, off, nbytes, count, elem, _ = SECTION.unpack(raw[table_off + k * SECTION.size:
                                                                    table_off + (k + 1) * SECTION.size].tobytes())
        blob = raw[off:off + nbytes]
        if code == DT_BYTES:
            value = blob.tobytes()
        elif code == DT_RECORD:
            value = blob
        else:
            value = blob.view(next(dt for dt, c in DTYPES.items() if c == code))
        sections[sid] = (code, count, elem, value)
    return header, meta, sections


def pack_digest(path):
    """A pack's hashes, for comparing two hosts' exports of the same inputs: the file's sha256 and size, its manifest's
    (format, oracle, layouts, inputs, names, counts) and each section's sha256 and size by name; never its contents."""
    _, meta, sections = read_pack(path)
    names = {v: k for k, v in load().constants()['sections'].items()}
    hashed = {}
    for sid, (_, count, elem, value) in sorted(sections.items()):
        data = value if isinstance(value, bytes) else value.tobytes()
        hashed[names.get(sid, str(sid))] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), count=count,
                                                elem=elem)
    text = json.dumps(meta, sort_keys=True)
    return dict(sha256=sha256_file(path), bytes=Path(path).stat().st_size, manifest_sha256=hashlib.sha256(text.encode()).hexdigest(),
                manifest=meta, sections=hashed)


def compare_digests(mine, other):
    """Where two pack_digest results differ: the file, the manifest's keys and the sections (by name)."""
    keys = sorted(set(mine['manifest']) | set(other['manifest']))
    sections = sorted(set(mine['sections']) | set(other['sections']))
    return dict(file_equal=mine['sha256'] == other['sha256'],
                manifest_keys=[k for k in keys if mine['manifest'].get(k) != other['manifest'].get(k)],
                sections=[k for k in sections if mine['sections'].get(k, {}).get('sha256') != other['sections'].get(k, {}).get('sha256')],
                sections_compared=len(sections))


def open_pack(path, allow_stale=False, names=None, inputs=None):
    """The extension's Pack over `path`, refused (a ValueError that names the file) when the reader refuses it (a record
    section of another struct size, a damaged file), when its record layouts or id tables (ids_hash) differ from this
    tree's, when its oracle or its exporter's source (exporter_hash) differs (an explicit allow_stale accepts a stale
    oracle and exporter, never other layouts or ids), when its names-v1 rows come from another table than `names`
    (default rl.names.name_table(), the one the Python projection reads), and when a recorded input differs from the file
    `inputs` ({name: path}) names. The last two have no override: a pack with other rows or inputs gives other
    observations than CpuBatch with no error."""
    module = load()
    try:
        pack = module.Pack(str(path))
    except module.Error as error:  # the reader's refusal (another struct size, a damaged file), named by its file
        raise ValueError(f'{path}: {as_python(error)}; export it again') from error
    meta = json.loads(pack.manifest)
    if meta.get('format') != PACK_FORMAT:
        raise ValueError(f'{path}: format {meta.get("format")!r}, not {PACK_FORMAT}')
    if meta.get('layout_hash') != layout_hash():
        raise ValueError(f'{path} was written against other record layouts; export it again')
    if meta.get('ids') != ids_hash():
        raise ValueError(f'{path} was written against other ids (the script or section table, rs.h\'s contract or the '
                         'declared worlds); export it again')
    if meta.get('oracle_hash') != oracle_hash() and not allow_stale:
        current = oracle_hashes()
        changed = sorted(k for k in current if meta.get('oracle', {}).get(k) != current[k])
        raise ValueError(f'{path} was exported from another oracle ({changed}); export it again or pass allow_stale')
    exporter = meta.get('exporter') or {}
    if exporter.get('sha256') != exporter_hash() and not allow_stale:
        raise ValueError(f'{path} was exported by another exporter ({exporter.get("version")}, source '
                         f'{str(exporter.get("sha256"))[:12]}, not {exporter_hash()[:12]}); export it again or pass '
                         'allow_stale')
    rows = module.constants()['sections']['STR_ROW'] in {row[0] for row in pack.sections()}
    if rows and meta.get('names') is None:
        raise ValueError(f'{path} has names-v1 rows but does not record their name table; export it again')
    if meta.get('names') is not None:
        if names is None:
            from names import name_table
            names = name_table()
        if meta['names'] != names_spec(names):
            raise ValueError(f'{path} was exported with name table {meta["names"]}, not {names_spec(names)} '
                             '(RL_NAME_TABLE); export it again')
    for name, file in (inputs or {}).items():
        recorded = meta.get('inputs', {}).get(name)
        if recorded != sha256_file(file):
            raise ValueError(f'{path} was exported from another {name} than {file}; export it again')
    return pack


@lru_cache(maxsize=None)
def log1p_table():
    """The log1p feature tables of rs.h (RsSlot.log1p): float32 [len(RS_LOG1P_K)][RS_LOG1P_COUNTS], row k at count c
    float32(np.log1p(c) / K[k]) as rl/obs.py, act_contract and feedback_contract compute it, with one scalar np.log1p
    call per count as they make it (an array call may take another SIMD path), in this process with this numpy.
    Read-only."""
    c = load().constants()
    logs = np.array([np.log1p(count) for count in range(c['RS_LOG1P_COUNTS'])], np.float64)
    table = np.stack([(logs / k).astype(np.float32) for k in c['RS_LOG1P_K']])
    table.setflags(write=False)
    return table


@lru_cache(maxsize=1 << 16)
def log1p_feature(k, count):
    """A late log1p feature (rs.h RsLate, a count past the tables): float32(np.log1p(count) / K[k]), the expression
    rl/obs.py, act_contract and feedback_contract evaluate (np.log1p of the int, a float64 divide, one float32
    rounding), in this process, so its value is the oracle's whatever the count."""
    return np.float32(np.log1p(count) / load().constants()['RS_LOG1P_K'][k])


@lru_cache(maxsize=None)
def _obs_layout():
    """The packed row's arrays (rs_gen.h RS_OBS_*): (name, byte offset, byte size) by offset."""
    return tuple((name, off, nbytes) for name, _, _, _, _, off, nbytes in load().constants()['obs'])


# ---- errors ----------------------------------------------------------------------------------------------------------
def _exceptions():
    from sim.audit import AuditError
    from sim.engine import Unsupported
    from sim.paths import OutsideTables
    return dict(EXC_UNSUPPORTED=Unsupported, EXC_KEYERROR=KeyError, EXC_VALUEERROR=ValueError,
                EXC_OUTSIDE=OutsideTables, EXC_AUDIT=AuditError, EXC_RUNTIME=RuntimeError, EXC_INDEX=IndexError,
                EXC_NOT_PORTED=NotPorted, EXC_ZERODIV=ZeroDivisionError)


def render(error, pack=None):
    """str(error) of the Python exception an RsErr mirrors: (code, exc, reason, args, text, world) as module.Error
    carries it; 's' arguments are the pack's strings."""
    _, _, reason, args, text, _ = error.args
    _, _, form, kinds = load().reasons()[reason]
    values, pending = [], list(args)
    for kind in kinds:
        if kind == 'c':
            values.append(text or '')
        else:
            value = pending.pop(0)
            values.append(value if kind == 'i' else pack.string(value) if pack is not None else f'<string {value}>')
    return form.format(*values)


def as_python(error, pack=None):
    """The Python exception (class and text) of a module.Error."""
    _, exc, _, _, _, _ = error.args
    cls = _exceptions()[load().exceptions()[exc]]
    text = render(error, pack)
    return cls(text) if cls is not KeyError else KeyError(text)


# ---- the batch ------------------------------------------------------------------------------------------------------
START_KINDS = ('short', 'combat', 'supply', 'village')  # rs.h RS_START_*
LAYOUT_KINDS = ('compact', 'supply', 'variant')  # rs.h RS_LAYOUT_*


def compact_layouts(pack):
    """(seed, declared world) of a Pack's compact pool layouts in pool order, from its LAYOUTS records: the world each
    record names (RsLayoutRec.world: 1 + its sim/world.py WORLDS index), which the exporter writes from world_of of the
    layout built as its pool's class, as CpuBatch reads it. A generator: the section is read when iterated."""
    from sim.world import WORLDS
    names, sid = list(WORLDS), load().constants()['sections']['LAYOUTS']
    if sid not in {row[0] for row in pack.sections()}:  # a pack of no layouts (a test pack) holds none
        return
    records = np.frombuffer(pack.section(sid), record_dtype('RsLayoutRec'))
    records = records[records['kind'] == LAYOUT_KINDS.index('compact')]
    for r in records[np.argsort(records['pool_index'], kind='stable')]:
        yield int(r['seed']), names[int(r['world']) - 1]



def village_world(declared):
    """The village world (sim/combat.py VILLAGES key) that plays layouts of this declared world, or None: village_of's
    rule on a record's world name."""
    from sim.combat import VILLAGES
    return next((name for name, v in VILLAGES.items() if v['layouts'] == declared), None)


def fields(n):
    from sim.workers import fields as worker_fields
    return worker_fields(n)


def free_fields(n, shapes):
    """The arrays a free-play batch writes (bind): the training workers' own table, sim/workers.py fields(n, shapes,
    last_obs=True), which allocates their shared buffer: obs/<key> and last_obs/<key> in the batch's table (obs-v6's),
    the info fields, made, kills, facts, reward, done and action."""
    from sim.workers import fields as worker_fields
    return worker_fields(n, shapes, last_obs=True)


def decode_start(raw):
    """CpuBatch.starts[b] from a world's RsStart (Batch.start): draw_start's dict, with draw_combat's under 'combat'
    for a combat start, and draw_supply's grove, variant and supply flag for a supply start."""
    from sim.combat import COMBAT, GROUPS
    from sim.search import SUPPLY
    from sim.spec import START_LEVELS, START_WEAPONS
    start = dict(layout=raw['layout'], tile=(raw['tile_x'], raw['tile_z']), weapon=START_WEAPONS[raw['weapon']],
                 levels=dict(zip(START_LEVELS, raw['levels'])), world_seed=raw['world_seed'])
    kind = START_KINDS[raw['kind']]
    if kind == 'village':  # draw_village's dict (sim/combat.py) and draw_life's length (sim/vec.py _episode)
        start['village'] = dict(group=GROUPS[raw['group']], hp=raw['hp'], food=[COMBAT['food'][k] for k in raw['food']],
                                worn=bool(raw['worn']), muggers=list(zip(raw['mugger_x'], raw['mugger_z'])),
                                placed=bool(raw['placed']))
        start['length'] = raw['length']
    elif kind == 'combat':
        start['combat'] = dict(group=GROUPS[raw['group']], hp=raw['hp'], food=[COMBAT['food'][k] for k in raw['food']],
                               worn=bool(raw['worn']), muggers=list(zip(raw['mugger_x'], raw['mugger_z'])))
    elif kind == 'supply':
        start.update(grove=raw['grove'], variant=SUPPLY['variants'][raw['variant']], supply=True)
    return start


def encode_start(start):
    """decode_start's inverse: a CpuBatch start dict as the raw ints Batch.begin reads (the layout record it plays is
    the C side's rs_start_rec)."""
    from sim.combat import COMBAT, GROUPS
    from sim.search import SUPPLY
    from sim.spec import START_LEVELS, START_WEAPONS
    kind = 'village' if 'village' in start else 'combat' if 'combat' in start else 'supply' if start.get('supply') else 'short'
    raw = dict(kind=START_KINDS.index(kind),
               layout=int(start['layout']), tile_x=int(start['tile'][0]), tile_z=int(start['tile'][1]),
               weapon=START_WEAPONS.index(start['weapon']), levels=[int(start['levels'][s]) for s in START_LEVELS],
               world_seed=int(start['world_seed']), rec=-1)
    if 'combat' in start or 'village' in start:
        c = start.get('combat') or start['village']
        raw.update(group=GROUPS.index(c['group']), hp=int(c['hp']), food=[COMBAT['food'].index(f) for f in c['food']],
                   worn=int(bool(c['worn'])), mugger_x=[int(x) for x, _ in c['muggers']],
                   mugger_z=[int(z) for _, z in c['muggers']])
    if 'village' in start:
        raw.update(length=int(start['length']), placed=int(bool(start['village']['placed'])))
    if start.get('supply'):
        raw.update(grove=int(start['grove']), variant=SUPPLY['variants'].index(start['variant']))
    return raw


UNSET = object()  # NativeBatch's goals and horizon when not given: a goal task's defaults, or free play's (CpuBatch's)


class NativeBatch:
    """CpuBatch's interface (sim/vec.py) over the native batch: B worlds of the four-goal task on a pack's pools, world
    slot b asking for its goal of the declared goal set (sim/spec.py goal_set), keyed starts and episodes as CpuBatch's,
    so the two reproduce each other. `reset` and `step` return fresh arrays as CpuBatch does; `bind` plus `reset_into`
    and `step_into` write a worker's slice of the shared buffer (sim/workers.py fields) in place. Both check the written
    observations with sim.spec.check_batch, as CpuBatch._stack does. `starts` and `evidence` are CpuBatch's, decoded
    from the C slots; an evidence row has no 'observation' (the raw observation dict is not built natively;
    sim/workers.py drops it anyway). A pack given by path is opened with open_pack's checks (`inputs`: {name: path} of
    the archive and pool files the caller would load). `threads` > 1 steps the worlds on that many threads (results do
    not depend on it). Task v4 (search) is refused; v11 runs as v5 once its bindings hold (sim/combat.py
    check_bindings on the goals and the pack's compact layouts). For the equivalence harness: `trace`, `trace_rows`,
    `begin`, `set_episode` and `external_starts` (sim/native_harness.py NativeSide).

    task='free-play-v1' is CpuBatch's free-play batch (sim/vec.py, sim/task.py FreePlayEpisode): the lives of the
    village world that the pack's compact records declare (village-world-v1 Stage A on compact-world-v1 layouts,
    village-world-v2 on its own: each record names its world), obs-v6 observations, `xp_rate` (1 or 25; None: 1) and,
    with no default, `muggers` (True or False; False in village-world-v2) and `life` (sim/task.py check_life); CpuBatch's
    check_village runs over the archive content the pack's manifest records; `horizon` and `goals` keep CpuBatch's
    defaults and `cfg` defaults to free_play_cfg. It refuses what CpuBatch refuses, here at build where CpuBatch's World
    and FreePlayEpisode refuse at the first life: an XP rate that is not a plain int, and agent limits that could cut the
    longest life. Its step's info has goal -1 and info['last_obs'] (each ended life's last observation, zero elsewhere);
    it binds free_fields(n, shapes): obs/<key> and last_obs/<key> in obs-v6's table. `observation` and `shapes` name
    the table of every batch; `xp_rate`, `muggers`, `life` and `world` are free play's. `world` is CpuBatch's: the
    village world (sim/combat.py VILLAGES) that the pool's layouts declare, held as the batch's `world` (None for a goal
    task); when it is given it must be that one. A goal task is played on compact-world-v1 layouts only: on a pack of
    village-world-v2's it is refused (native only; CpuBatch plays the goal tasks on v2 layouts)."""

    def __init__(self, pack, n, first=0, task='goal-activity-v5', goals=UNSET, horizon=UNSET, cfg=None, allow_stale=False,
                 inputs=None, threads=1, xp_rate=None, muggers=None, life=None, world=None):
        from baseline import config
        from baseline.resource_task import GOALS
        from obs import OBSERVATION
        from sim import agent
        from sim.combat import VILLAGE, VILLAGES, check_bindings, check_village, task_spec
        from sim.spec import HORIZON_TICKS, OBSERVATIONS, goal_set, slot_goal
        from sim.task import FREE_PLAY, check_life, free_play_cfg
        if n < 1 or first < 0:  # rs_batch_new's rule, raised as CpuBatch's ValueError for n = 0 (at its reset)
            raise ValueError(f'a batch needs n >= 1 worlds, first >= 0: n {n!r}, first {first!r}')
        if task not in TASKS:
            raise ValueError(f'the native engine runs {sorted(TASKS)}, not {task}')
        self.free = task == FREE_TASK
        self.pack = pack if isinstance(pack, load().Pack) else open_pack(pack, allow_stale, inputs=inputs)
        life_cfg = dict(life_kind=0, life_lo=0, life_hi=0)
        self.world = None
        if self.free:  # CpuBatch's checks, in its order (sim/vec.py), then its first life's (World, FreePlayEpisode)
            goals, horizon = GOALS if goals is UNSET else goals, HORIZON_TICKS if horizon is UNSET else horizon
            if horizon != HORIZON_TICKS or tuple(goals) != GOALS:
                raise ValueError(f'{task} has no goal, no goal horizon and no search pool')
            if type(muggers) is not bool:
                raise ValueError(f"{VILLAGE['version']} Stage A places muggers or not: pass muggers=True or False")
            declared = [name for _, name in compact_layouts(self.pack)]
            found = {village_world(name) for name in declared}
            if len(found) != 1 or None in found:
                worlds = ', '.join(f"{name} on {v['layouts']}" for name, v in VILLAGES.items())
                raise ValueError(f'{task} plays the layouts of one village world ({worlds}), not {sorted(set(declared))}')
            self.world = found.pop()
            if world is not None and world != self.world:
                raise ValueError(f'the pool holds {self.world} layouts, not {world!r}')
            # check_village over the archive the pack was exported from (its manifest's content keys)
            content = json.loads(self.pack.manifest).get('content') or []
            check_village(VILLAGES[self.world], SimpleNamespace(meta=dict(content=dict.fromkeys(content))), muggers)
            self.spec, self.muggers = dict(combat=False, search=False, supply=False), muggers
            self.xp_rate = FREE_PLAY['xp_rates'][0] if xp_rate is None else xp_rate
            if self.xp_rate not in FREE_PLAY['xp_rates']:
                raise ValueError(f"a free-play XP rate is one of {FREE_PLAY['xp_rates']}, not {xp_rate!r}")
            if life is None:
                raise ValueError(f"{task}'s life-length distribution is a setting with no default: pass life")
            self.life = check_life(life)
            life_cfg = dict(life_kind=int(self.life[0] == 'uniform'), life_lo=self.life[1], life_hi=self.life[-1])
            self.observation = FREE_PLAY['observation']
            cfg = free_play_cfg() if cfg is None else cfg
            if type(self.xp_rate) is not int:  # World's check (sim/engine.py), at CpuBatch's first life
                raise ValueError(f'xp_rate {self.xp_rate!r}: a positive whole multiplier')
            limits = dict(cfg or config.TASK_CFG)  # FreePlayEpisode's check against the longest life
            if limits.get('maxEpisodeTicks', 0) < life_cfg['life_hi'] + 2 * agent.DECISION_WAIT_TICKS or \
                    limits.get('maxDecisions', 0) < life_cfg['life_hi']:
                raise ValueError('agent limits that could cut a free-play life (free_play_cfg)')
            horizon, goals, rate = HORIZON_TICKS, goal_set(GOALS), self.xp_rate
        else:
            from sim.engine import XP_RATE
            if (xp_rate, muggers, life, world) != (None, None, None, None):
                raise ValueError(f'xp_rate, muggers, life and world are {FREE_TASK} settings, not {task}\'s')
            if {village_world(name) for _, name in compact_layouts(self.pack)} - {VILLAGE['version'], None}:
                raise ValueError('the native engine plays the goal tasks on compact-world-v1 layouts only')
            self.spec = task_spec(task)
            goals, horizon = None if goals is UNSET else goals, None if horizon is UNSET else horizon
            horizon = HORIZON_TICKS if horizon is None else horizon
            if self.spec['combat'] and horizon != HORIZON_TICKS:
                raise ValueError(f'{task} keeps the gathering goals\' {HORIZON_TICKS}-tick horizon')
            goals = goal_set(GOALS if goals is None else goals)
            check_bindings(self.spec, compact_layouts(self.pack), goals)
            self.observation, rate = OBSERVATION, XP_RATE
        self.shapes = OBSERVATIONS[self.observation]
        self.cfg = cfg = dict(cfg or config.TASK_CFG)
        self.n, self.first, self.horizon, self.task = n, first, horizon, task
        self.goals = [slot_goal(goals, first + b) for b in range(n)]
        # the agent's cfg as Agent._login copies it, with agent.py's defaults for absent keys (truncated, pace_ticks)
        self._batch = load().Batch(self.pack, dict(
            task=TASKS[task], n=n, first=first, n_goals=len(goals), goal=GOALS.index(goals[0]), horizon=horizon,
            pace_jitter=cfg.get('paceJitterTicks', 0), decision_points=int(bool(cfg.get('decisionPoints'))),
            max_decisions=cfg.get('maxDecisions', agent.MAX_DECISIONS),
            max_episode_ticks=cfg.get('maxEpisodeTicks', agent.MAX_EPISODE_TICKS), threads=int(threads),
            xp_rate=rate, muggers=int(bool(muggers)), **life_cfg), log1p_table())
        self.key, self.views, self.evidence = None, None, []
        table = free_fields(n, self.shapes) if self.free else fields(n)
        self.bind({name: np.zeros(shape, kind) for name, (shape, kind) in table.items()})

    def bind(self, views):
        """Write into these arrays from now on: sim/workers.py fields(n) names, shapes and dtypes, C-contiguous."""
        self._batch.bind(views)
        self.views = views

    def _call(self, method, *args):
        """A batch call; its error raised as the Python exception it mirrors, with the failing world's index (`world`)."""
        try:
            method(*args)
        except load().Error as error:
            from sim.engine import Unsupported
            exception = as_python(error, self.pack)
            exception.world = error.args[5]
            if isinstance(exception, Unsupported) and 0 <= exception.world < self.n:
                self._setup_evidence(exception.world, exception)
            raise exception from None

    def _setup_evidence(self, b, error):
        """CpuBatch._start's evidence row for world b's setup that reached unsupported content (a step's Unsupported is
        flagged, never raised, so a raised one is a setup's: reset, an auto-reset or begin): the start as drawn, its
        layout seed, the slot's episode counter, then the call raises it."""
        raw = self._batch.start(b)
        self.evidence.append(dict(reason=str(error), world=self.first + b, key=self.key, episode=self._batch.episodes()[b],
                                  start=None if raw is None or raw['seed'] is None else decode_start(raw),
                                  seed=None if raw is None else raw['seed'], setup=True, decision=0, tick=None,
                                  action=None))

    def _fill_late(self, last=False):
        """Write each world's late log1p features (log1p_feature) into its observation row of the bound arrays (with
        `last`, its last_obs row's). An obs-v6 key holds its obs-v5 key's values from the same place; task_vec, the one
        key obs-v6 shortens (its decision count is dropped), holds no log1p feature, so a late one there is refused."""
        for b, off, k, count in (self._batch.late_last() if last else self._batch.late()):
            name, start, _ = next(entry for entry in _obs_layout() if entry[1] <= off < entry[1] + entry[2])
            if name == 'task_vec':
                raise RuntimeError(f'a late log1p feature in task_vec (byte {off}), which holds none')
            self.views[f'{"last_obs" if last else "obs"}/{name}'][b].flat[(off - start) // 4] = log1p_feature(k, count)

    def check(self, step=False):
        """CpuBatch._stack's check_batch on the written observations, with its ValueErrors: the keys, shapes and dtypes
        are the bound arrays' (bind checks them against sim/spec.py), and the value checks (finite floats, a 0/1 mask
        with a legal action in every row) run in C over those arrays (rs_batch_check). With `step`, free play's
        last_obs too, as sim/workers.py _write checks a CpuBatch reply, here in Python over the bound arrays: finite
        floats in every ended life's row, and a legal action in the row of each life cut by its length."""
        bad = self._batch.check()
        if bad is not None:
            names = list(self.shapes)
            if bad < len(names):
                raise ValueError(f'{names[bad]}: non-finite values')
            raise ValueError('invalid or all-empty action mask')
        if not (step and self.free):
            return
        from sim.spec import TERMINAL_HORIZON
        ended = np.flatnonzero(self.views['done'])
        for name in self.shapes:
            value = self.views[f'last_obs/{name}'][ended]
            if value.dtype.kind == 'f' and not np.isfinite(value).all():
                raise ValueError(f'last_obs {name}: non-finite values')
        cut = self.views['info/terminal'] == TERMINAL_HORIZON
        if not self.views['last_obs/mask'][cut].any(-1).all():
            raise ValueError('the last observation of a life cut by its length offers no legal action')

    def _record_failures(self):
        """CpuBatch.step's evidence rows for the worlds this step flagged unsupported."""
        for b in np.flatnonzero(self.views['info/unsupported']):
            error, decision, tick, action, episode = self._batch.failure(int(b))
            raw = self._batch.start(int(b))
            self.evidence.append(dict(reason=render(load().Error(*error), self.pack), world=self.first + int(b),
                                      key=self.key, episode=episode, start=decode_start(raw), seed=raw['seed'],
                                      decision=decision, tick=tick, action=action))

    def state(self):
        return dict(key=self.key, episode=np.asarray(self._batch.episodes(), np.int32))

    def observations(self):
        """Copies of the bound observation arrays."""
        return {name: self.views[f'obs/{name}'].copy() for name in self.shapes}

    def outputs(self):
        """step's return value from the bound arrays (copies): state, obs, reward, done, info (free play: with
        last_obs)."""
        from sim.spec import INFO_FIELDS
        info = {name: self.views[f'info/{name}'].copy() for name in INFO_FIELDS}
        info.update(made=self.views['info/made'].copy(), kills=self.views['info/kills'].copy(),
                    facts=self.views['info/facts'].copy())
        if self.free:
            info['last_obs'] = {name: self.views[f'last_obs/{name}'].copy() for name in self.shapes}
        return self.state(), self.observations(), self.views['reward'].copy(), self.views['done'].copy(), info

    def reset_into(self, key, check=True):
        """reset without copies: the bound arrays hold the first observations (with external_starts, none: begin each
        world before checking)."""
        self.key, self.evidence = int(key), []
        self._call(self._batch.reset, self.key)
        self._fill_late()
        if check:
            self.check()

    def reset(self, key):
        self.reset_into(key)
        return self.state(), self.observations()

    def step_into(self, action, check=True):
        """step without copies: the bound arrays hold the result (valid until the next call)."""
        if self.views['action'] is not action:
            self.views['action'][:] = action
        self._call(self._batch.step, self.views['action'])
        self._record_failures()
        self._fill_late()
        if self.free:
            self._fill_late(last=True)
        if check:
            self.check(step=True)

    def step(self, state, action):
        if state['key'] != self.key:
            raise ValueError('a state of another reset')
        self.step_into(np.asarray(action, np.int32))
        return self.outputs()

    @property
    def episodes(self):
        return list(self._batch.episodes())

    @property
    def cycles(self):
        """World cycles every world has run so far, setup ones included (sim/native_bench.py's tick count)."""
        return self._batch.cycles()

    @property
    def starts(self):
        """CpuBatch.starts: each world's current start (decode_start); None before its first draw and after a draw
        that failed, as CpuBatch's."""
        return [None if raw is None else decode_start(raw) for raw in map(self._batch.start, range(self.n))]

    # ---- the equivalence harness's calls
    def trace(self, on=True, **caps):
        """Trace every world (canon.c): its tick rows from its next episode on (caps: draws_cap, rows_cap, data_cap)."""
        for b in range(self.n):
            self._batch.trace(b, on, *[caps[k] for k in ('draws_cap', 'rows_cap', 'data_cap') if k in caps])

    def trace_rows(self, b):
        """World b's rows since the last call, as sim/native_harness.py Row fields (the record an int64 array)."""
        return [(tick, crc, np.frombuffer(record, '<i8').copy(), draws, events, chat, texts, episode)
                for tick, crc, record, draws, events, chat, texts, episode in self._batch.trace_rows(b)]

    def begin(self, b, start=None):
        """World b's next episode from a CpuBatch start dict (None: drawn from the key and its episode counter)."""
        self._call(self._batch.begin, int(b), None if start is None else encode_start(start))
        self._fill_late()

    def set_episode(self, b, episode):
        self._batch.set_episode(int(b), int(episode))

    def external_starts(self, on=True):
        """Begin no episode at reset or auto-reset: each world waits for begin(b, start) (the harness's scenario
        starts)."""
        self._batch.external_starts(bool(on))


# ---- command line -----------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build', help='build (or find) the extension for this source set and print its path, compiler and flags '
                   '(one JSON line)')
    info = sub.add_parser('pack-info', help="a pack's format, hashes and section sizes (never its contents)")
    info.add_argument('pack', type=Path)
    info.add_argument('--digest', type=Path, help='write pack_digest (the file, manifest and section hashes) here as JSON')
    info.add_argument('--against', type=Path, help='with --digest: another host\'s pack_digest JSON, compared with this '
                      'one (compare_digests) in the same file')
    export = sub.add_parser('export', help='write the pack of a compact pool, a supply pool and the archive '
                            '(sim/native_export.py)')
    export.add_argument('--pool', type=Path, required=True, help='a sim.world pool with derived tables')
    export.add_argument('--supply-pool', type=Path, required=True, help='a sim.search supply pool')
    export.add_argument('--tables', type=Path, required=True, help='the simulator archive (pins unchecked)')
    export.add_argument('--output', type=Path, required=True, help='logs/native/NAME.pack (never published)')
    export.add_argument('--no-trees', action='store_true', help='leave out the derived state-0 trees')
    export.add_argument('--variants', type=int, default=None,
                        help='at most this many scarce variants per supply layout (a partial pack, for checks)')
    from sim import native_bench, native_harness
    native_harness.add_arguments(sub.add_parser('diff', help='step two batches in lockstep and report the first '
                                                 'divergence (sim/native_harness.py)'))
    native_harness.gate_arguments(sub.add_parser('gate', help='the release gate: every frozen case, the digests, thread '
                                                 'invariance and the soak (sim/native_harness.py)'))
    native_harness.free_soak_arguments(sub.add_parser('free-soak', help="free-play-v1's soak: random legal play and the "
                                                      'free-play controls in lockstep with the Python reference'))
    native_bench.add_arguments(sub.add_parser('bench', help='world ticks and decisions per second by activity, action '
                                              'source, thread count and check (sim/native_bench.py)'))
    native_bench.env_arguments(sub.add_parser('env', help='random legal play through the training workers with either '
                                              'engine, timed and, given both, compared (sim/native_bench.py)'))
    args = parser.parse_args()
    if args.command == 'diff':
        native_harness.main(args)
        return
    if args.command == 'gate':
        native_harness.gate_main(args)
        return
    if args.command == 'free-soak':
        native_harness.free_soak_main(args)
        return
    if args.command == 'bench':
        native_bench.main(args)
        return
    if args.command == 'env':
        native_bench.env_main(args)
        return
    if args.command == 'build':
        load()
        print(json.dumps(dict(library=str(library()[0]), compiler=compiler(), command=linker(), flags=build_flags(),
                              libs=LIBS, platform=sysconfig.get_platform())))
        return
    if args.command == 'export':
        from sim.native_export import main as export_main
        export_main(args)
        return
    header, meta, sections = read_pack(args.pack)
    names = {v: k for k, v in load().constants()['sections'].items()}
    print(f'[native] {args.pack}: {header["file_size"]} bytes, oracle {meta.get("oracle_hash", "?")[:16]}'
          f'{" (current)" if meta.get("oracle_hash") == oracle_hash() else " (stale)"}')
    for sid, (code, count, elem, _) in sorted(sections.items()):
        print(f'  {names.get(sid, sid)}: {count} x {elem} bytes')
    if args.digest:
        mine = pack_digest(args.pack)
        other = json.loads(args.against.read_text()) if args.against else None
        compared = None if other is None else compare_digests(mine, other)
        args.digest.write_text(json.dumps(dict(digest=mine, against=compared), indent=1, sort_keys=True) + '\n')
        if compared is not None:
            print(f'[native] against {args.against}: file {"equal" if compared["file_equal"] else "differs"}, manifest keys '
                  f'{compared["manifest_keys"] or "equal"}, {len(compared["sections"])} of {compared["sections_compared"]} '
                  f'sections differ {compared["sections"]}')


if __name__ == '__main__':
    main()
