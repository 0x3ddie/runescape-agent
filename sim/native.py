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
        --supply-pool logs/supply/pools/train-128.npz --tables logs/sim/data/draynor-nc1.npz --output logs/native/train-8-nc1-cc0614f.pack
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native pack-info logs/native/train-<hash>.pack
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native diff --pool logs/sim/world/train-8-nc1-derived.npz ...  (the
        equivalence harness, sim/native_harness.py)
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native gate [--quick]  (the release gate, sim/native_harness.py)
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native bench --pack logs/native/train-8-nc1-cc0614f.pack ...  (throughput,
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The default pack of the gate, the bench and the native tests (sim/tests PACK): train-8-nc1-rv2's inputs exported again,
# with trees (9 November 2026), the oracle with the firemaking and cooking scripts; the pack is of
# draynor-nc1.npz, over which they do not run. Kept as evidence, each stale on this tree: train-8-nc1-5c14c8b.pack and
# train-8-nc1-p2.pack (the same inputs at two earlier phase 2 states), train-8-nc1-gap1.pack
# (at the native gap study's Stage 3 freeze) and rv2 (M's records name it).
PACK = ROOT / 'logs' / 'native' / 'train-8-nc1-cc0614f.pack'
SOURCE = Path(__file__).with_name('native')
FLAGS = ('-O2', '-std=c11', '-ffp-contract=off', '-fno-fast-math', '-fPIC')
# The extension alone is also built with link-time optimization: calls between its files (rs_tile, rs_wm, the walk
# reads, can_travel) inline as calls within one file do. It changes no operation (the FLAGS above hold at compile
# time, so no a*b+c is fused); the tests' partial builds keep FLAGS alone. SIM_NATIVE_LTO=0 builds without it (a
# toolchain without the LTO plugin), under its own library tag.
LINK = ('-flto',)
# Elsewhere than macOS (a Linux x86_64 cloud machine, gcc through LDSHARED): glibc declares batch.c's pthread calls under
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
# so C runs it as v5.
TASKS = {'goal-activity-v2': 2, 'goal-activity-v3': 3, 'goal-activity-v5': 5, 'goal-activity-v11': 5}


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
    from sim import agent, combat, data, engine, paths, search, spec, task, vec

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
                 'LOC_DURATION', 'WALL_STRAIGHT', 'WALL_DIAGONAL', 'STAB', 'SLASH', 'CRUSH', 'RANGED_STYLE',
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


def load():
    """The extension module sim._native (built on first use)."""
    global _MODULE
    if _MODULE is None:
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


def manifest(inputs=None, names=None, **extra):
    """A pack manifest's required part: format, the oracle's file and combined hashes, the record layouts' hash, the
    exporter's input hashes (input_hashes: archive meta, pools) and the name table its STR_ROW rows come from (a
    NameTable, or None for a pack without name rows); `extra` adds exporter fields (counts, version)."""
    hashes = oracle_hashes()
    return dict(format=PACK_FORMAT, version=PACK_VERSION, oracle=hashes, oracle_hash=oracle_hash(hashes),
                layout_hash=layout_hash(), inputs=dict(inputs or {}),
                names=None if names is None else names_spec(names), **extra)


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
    """The extension's Pack over `path`, refused when its oracle or record layouts differ from this tree's (an
    explicit allow_stale accepts a stale oracle, never other layouts), when its names-v1 rows come from another table
    than `names` (default rl.names.name_table(), the one the Python projection reads), and when a recorded input
    differs from the file `inputs` ({name: path}) names. The last two have no override: a pack with other rows or
    inputs gives other observations than CpuBatch with no error."""
    module = load()
    pack = module.Pack(str(path))
    meta = json.loads(pack.manifest)
    if meta.get('format') != PACK_FORMAT:
        raise ValueError(f'{path}: format {meta.get("format")!r}, not {PACK_FORMAT}')
    if meta.get('layout_hash') != layout_hash():
        raise ValueError(f'{path} was written against other record layouts; export it again')
    if meta.get('oracle_hash') != oracle_hash() and not allow_stale:
        current = oracle_hashes()
        changed = sorted(k for k in current if meta.get('oracle', {}).get(k) != current[k])
        raise ValueError(f'{path} was exported from another oracle ({changed}); export it again or pass allow_stale')
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
START_KINDS = ('short', 'combat', 'supply')  # rs.h RS_START_*
LAYOUT_KINDS = ('compact', 'supply', 'variant')  # rs.h RS_LAYOUT_*


def compact_layouts(pack):
    """(seed, declared world) of a Pack's compact pool layouts in pool order, from its LAYOUTS records: the world is the
    first declared one of the record's size, as sim/world.py world_of reads a layout's. A generator: the section is read
    when iterated."""
    from sim.world import WORLDS
    records = np.frombuffer(pack.section(load().constants()['sections']['LAYOUTS']), record_dtype('RsLayoutRec'))
    records = records[records['kind'] == LAYOUT_KINDS.index('compact')]
    for r in records[np.argsort(records['pool_index'], kind='stable')]:
        size = (int(r['W']), int(r['H']))
        yield int(r['seed']), next((name for name, spec in WORLDS.items() if spec['size'] == size), None)


def fields(n):
    from sim.workers import fields as worker_fields
    return worker_fields(n)


def decode_start(raw):
    """CpuBatch.starts[b] from a world's RsStart (Batch.start): draw_start's dict, with draw_combat's under 'combat'
    for a combat start, and draw_supply's grove, variant and supply flag for a supply start."""
    from sim.combat import COMBAT, GROUPS
    from sim.search import SUPPLY
    from sim.spec import START_LEVELS, START_WEAPONS
    start = dict(layout=raw['layout'], tile=(raw['tile_x'], raw['tile_z']), weapon=START_WEAPONS[raw['weapon']],
                 levels=dict(zip(START_LEVELS, raw['levels'])), world_seed=raw['world_seed'])
    kind = START_KINDS[raw['kind']]
    if kind == 'combat':
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
    raw = dict(kind=START_KINDS.index('combat' if 'combat' in start else 'supply' if start.get('supply') else 'short'),
               layout=int(start['layout']), tile_x=int(start['tile'][0]), tile_z=int(start['tile'][1]),
               weapon=START_WEAPONS.index(start['weapon']), levels=[int(start['levels'][s]) for s in START_LEVELS],
               world_seed=int(start['world_seed']), rec=-1)
    if 'combat' in start:
        c = start['combat']
        raw.update(group=GROUPS.index(c['group']), hp=int(c['hp']), food=[COMBAT['food'].index(f) for f in c['food']],
                   worn=int(bool(c['worn'])), mugger_x=[int(x) for x, _ in c['muggers']],
                   mugger_z=[int(z) for _, z in c['muggers']])
    if start.get('supply'):
        raw.update(grove=int(start['grove']), variant=SUPPLY['variants'].index(start['variant']))
    return raw


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
    `begin`, `set_episode` and `external_starts` (sim/native_harness.py NativeSide)."""

    def __init__(self, pack, n, first=0, task='goal-activity-v5', goals=None, horizon=None, cfg=None, allow_stale=False,
                 inputs=None, threads=1):
        from baseline import config
        from baseline.resource_task import GOALS
        from sim import agent
        from sim.combat import check_bindings, task_spec
        from sim.spec import HORIZON_TICKS, goal_set, slot_goal
        if task not in TASKS:
            raise ValueError(f'the native engine runs {sorted(TASKS)}, not {task}')
        self.spec = task_spec(task)
        horizon = HORIZON_TICKS if horizon is None else horizon
        if self.spec['combat'] and horizon != HORIZON_TICKS:
            raise ValueError(f'{task} keeps the gathering goals\' {HORIZON_TICKS}-tick horizon')
        goals = goal_set(GOALS if goals is None else goals)
        cfg = dict(cfg or config.TASK_CFG)
        self.pack = pack if isinstance(pack, load().Pack) else open_pack(pack, allow_stale, inputs=inputs)
        check_bindings(self.spec, compact_layouts(self.pack), goals)
        self.n, self.first, self.horizon, self.task = n, first, horizon, task
        self.goals = [slot_goal(goals, first + b) for b in range(n)]
        # the agent's cfg as Agent._login copies it, with agent.py's defaults for absent keys (truncated, pace_ticks)
        self._batch = load().Batch(self.pack, dict(
            task=TASKS[task], n=n, first=first, n_goals=len(goals), goal=GOALS.index(goals[0]), horizon=horizon,
            pace_jitter=cfg.get('paceJitterTicks', 0), decision_points=int(bool(cfg.get('decisionPoints'))),
            max_decisions=cfg.get('maxDecisions', agent.MAX_DECISIONS),
            max_episode_ticks=cfg.get('maxEpisodeTicks', agent.MAX_EPISODE_TICKS), threads=int(threads)), log1p_table())
        self.key, self.views, self.evidence = None, None, []
        self.bind({name: np.zeros(shape, kind) for name, (shape, kind) in fields(n).items()})

    def bind(self, views):
        """Write into these arrays from now on: sim/workers.py fields(n) names, shapes and dtypes, C-contiguous."""
        self._batch.bind(views)
        self.views = views

    def _call(self, method, *args):
        """A batch call; its error raised as the Python exception it mirrors, with the failing world's index (`world`)."""
        try:
            method(*args)
        except load().Error as error:
            exception = as_python(error, self.pack)
            exception.world = error.args[5]
            raise exception from None

    def _fill_late(self):
        """Write each world's late log1p features (log1p_feature) into its observation row of the bound arrays."""
        for b, off, k, count in self._batch.late():
            name, start, _ = next(entry for entry in _obs_layout() if entry[1] <= off < entry[1] + entry[2])
            self.views[f'obs/{name}'][b].flat[(off - start) // 4] = log1p_feature(k, count)

    def check(self):
        """CpuBatch._stack's check_batch on the written observations, with its ValueErrors: the keys, shapes and dtypes
        are the bound arrays' (bind checks them against sim/spec.py), and the value checks (finite floats, a 0/1 mask
        with a legal action in every row) run in C over those arrays (rs_batch_check)."""
        from sim.spec import OBSERVATION_SHAPES
        bad = self._batch.check()
        if bad is None:
            return
        names = list(OBSERVATION_SHAPES)
        if bad < len(names):
            raise ValueError(f'{names[bad]}: non-finite values')
        raise ValueError('invalid or all-empty action mask')

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
        from sim.spec import OBSERVATION_SHAPES
        return {name: self.views[f'obs/{name}'].copy() for name in OBSERVATION_SHAPES}

    def outputs(self):
        """step's return value from the bound arrays (copies): state, obs, reward, done, info."""
        from sim.spec import INFO_FIELDS
        info = {name: self.views[f'info/{name}'].copy() for name in INFO_FIELDS}
        info.update(made=self.views['info/made'].copy(), kills=self.views['info/kills'].copy(),
                    facts=self.views['info/facts'].copy())
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
        if check:
            self.check()

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
        """World cycles every world has run so far, setup ones included (sim/bench.py's tick count)."""
        return self._batch.cycles()

    @property
    def starts(self):
        """CpuBatch.starts: each world's current start (decode_start)."""
        return [decode_start(self._batch.start(b)) for b in range(self.n)]

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
