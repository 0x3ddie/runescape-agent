"""The native simulator's exporter: one read-only pack per pool set, holding everything the C
engine reads from the simulator archive and the pools, built by calling the oracle's own functions so every table is
exact by construction. The pack format and the record structs are sim/native/rs.h's; sim/native.py writes and opens
packs. Sections:

- strings: every name, option, script, interface and message text, with code-point ranks, lowercase ids, option
  name ids and names-v1 rows (rl/names.py) for each of loc, npc and obj;
- types: loc, npc and obj types with resolved params, op states, regex flags and their triggers mapped to rs.h script
  ids (RS_S_UNKNOWN where the oracle has no port of the named script, RS_NP ids where it has one C lacks);
- content rows: trees, rocks, gems, fish, eat triggers and rows, wield rules, weapon styles, level-up rows and unlocks;
- messages: every chat and dialog string a ported path can emit, as text_vector's token hashes (rl/feedback_contract);
- death drop programs: the reachable [ai_queue3] roots (chicken, goblin, mugger) compiled per npc type into rs.h's
  register machine (DropCompiler), with the sha256 of each root's parsed source;
- layouts: the compact pool (with its derived state-0 trees), the supply pool and every scarce variant
  (sim/search.py variant_layout), each through sim/world.py LayoutTables as CpuBatch reads it, with start tiles,
  combat camps and regions, groves and supply sites;
- LEVEL_XP, the click option strings and the other constants (RsConsts).

The declared-world check (`declared_world`) refuses, before anything is written, pools whose reachable loc, npc or obj
types have a trigger that names a script C does not port (RS_NP: banks, stairs, imps, dark wizards) or a death
root the compiler cannot carry; an archive with script rows the C port does not run (firemaking and cooking:
PORTED_ARCHIVE_SCRIPTS holds the picks alone); and a village-world-v2 pool over an archive without its pick rows. Each
layout record names its world (world_of of the layout built as its pool's class), and the manifest the pools' worlds,
the archive's content keys and the village world's own loot list. Packs are extracted private data: logs/native/
(ignored), never published.

A village-world-v2 pool is plain (no derived tables): export it with --no-trees over draynor-nc2.npz.

    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.native export --pool logs/sim/world/train-8-nc1-derived.npz \
        --supply-pool logs/supply/pools/train-128.npz --tables logs/sim/data/draynor-nc1.npz \
        --output logs/native/train-8-nc1-a6c325a-ids2.pack

With --no-trees the compact pool may also hold no derived tables (gap1-x's pool; gap1-k's keeps its derived tables, without
the trees).
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import inspect
import json
import re
from types import SimpleNamespace
import zlib

import numpy as np

from sim import native

EXPORT_VERSION = 'rs-native-export-v1'
PLAYER = 'wd1a0'  # the agent's player name (GoalEpisode's default), the hero a death script finds
# compact-pool arrays of a derived pool file that LayoutTables and the pack read (state 0 = the gate closed)
DERIVED_STATE = ('steps', 'entity_reach', 'los', 'loc_reach', 'order', 'dist', 'parent')
DERIVED_PLAIN = ('snapshot_steps', 'sides', 'snapshot_dist')
TREES = ('order', 'dist', 'parent', 'snapshot_dist')  # dropped by trees=False


# the archive scripts (engine.ARCHIVE_SCRIPTS) the C port runs: an archive with other script rows is refused
PORTED_ARCHIVE_SCRIPTS = {'picks'}
# the rs.h ids of the oracle's use_default closures by trigger (engine.HELD's and the pick archive's FIELD_USE)
USE_DEFAULT_IDS = {'[opheldu,_weapon_stab]': 'RS_S_USE_DEFAULT_WEAPON_STAB', '[opheldu,feather]': 'RS_S_USE_DEFAULT_FEATHER',
                   '[opheldu,_category_22]': 'RS_S_USE_DEFAULT_CATEGORY_22', '[opheldu,cabbage]': 'RS_S_USE_DEFAULT_CABBAGE',
                   '[opheldu,potato]': 'RS_S_USE_DEFAULT_POTATO', '[opheldu,onion]': 'RS_S_USE_DEFAULT_ONION',
                   '[opheldu,black_cape]': 'RS_S_USE_DEFAULT_BLACK_CAPE'}
# the obj flags of the FIELD_USE closures' cases (rs.h RS_OF_CASE_*)
FIELD_USE_FLAGS = {'[opheldu,cabbage]': 'CASE_CABBAGE', '[opheldu,potato]': 'CASE_POTATO', '[opheldu,onion]': 'CASE_ONION',
                   '[opheldu,black_cape]': 'CASE_BLACK_CAPE'}


class ExportError(ValueError):
    """The archive or the pools hold something the pack cannot carry exactly: the export refuses."""


# ---- the contract (sim/native/rs.h) ----------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def contract():
    """rs.h's integer #defines and enum values by name, plus the X-lists RS_TEXT_KEYS and RS_VARPS (in order)."""
    text = (native.SOURCE / 'rs.h').read_text()
    code = re.sub(r'\\\n', ' ', re.sub(r'/\*.*?\*/', ' ', text, flags=re.S))
    values = {'INT32_MIN': -2 ** 31}

    def evaluate(expr):
        names = set(re.findall(r'[A-Za-z_]\w*', expr))
        return int(eval(expr, {'__builtins__': {}}, {n: values[n] for n in names}))  # rs.h's own constant expressions

    for name, expr in re.findall(r'^[ \t]*#define[ \t]+([A-Z_][A-Z0-9_]*)[ \t]+([^\n]+)$', code, re.M):
        try:
            values[name] = evaluate(expr)
        except (KeyError, NameError, SyntaxError, TypeError, ValueError):
            pass
    for body in re.findall(r'\benum\s*\w*\s*\{([^{}]*)\}', code):
        if re.search(r'\w\s*\(', body):  # an X-macro expansion (reasons, sections, scripts): the module exports those
            continue
        following = 0
        for item in filter(None, (part.strip() for part in body.split(','))):
            name, _, expr = item.partition('=')
            values[name.strip()] = following = evaluate(expr) if expr else following
            following += 1
    for name in ('RS_TEXT_KEYS', 'RS_VARPS'):
        found = re.search(rf'#define\s+{name}\(X\)([^\n]*)', code)
        values[name] = tuple(re.findall(r'X\((\w+)\)', found.group(1)))
    values.update({f'RS_TX_{key}': k for k, key in enumerate(values['RS_TEXT_KEYS'])})  # RsTextKey
    return values


def record(name, n):
    """n zeroed records of an rs.h struct (record_dtype)."""
    return np.zeros(n, native.record_dtype(name))


def stack(name, rows):
    """Records of an rs.h struct from one-record arrays (np.concatenate would drop the struct's offsets)."""
    out = record(name, len(rows))
    for k, row in enumerate(rows):
        out[k] = row[0]
    return out


# ---- inputs --------------------------------------------------------------------------------------------------------
def load_tables(path):
    """The archive's Tables with its meta only (sim/data.py Tables; its pins unchecked, as sim/workers.py load_tables
    in a checkout without vendor/rs-sdk): the pack reads types and content, and LayoutTables carries each layout's own arrays."""
    from sim.data import Tables
    with np.load(path) as data:
        raw = data['meta'].tobytes()
    tables = Tables(json.loads(raw), {})
    tables.meta_sha256 = hashlib.sha256(raw).hexdigest()  # the archive JSON meta, for the manifest
    return tables


def load_pool(path, derived=None):
    """sim/world.py load_pool for the Layout fields, without the derived tables it would decompress whole: `derived`
    names the derived arrays to read (state-indexed ones cut to state 0), each layout's dict, else None."""
    from dataclasses import fields
    from sim import world
    names = [f.name for f in fields(world.Layout)]
    with np.load(path) as data:
        if str(data['version']) not in world.WORLDS:
            raise ValueError(f"{path} holds {data['version']}, not one of {sorted(world.WORLDS)}")
        cls = world.LAYOUT_CLASSES.get(str(data['version']), world.Layout)  # the class names the world (world_of)
        arrays = {k: data[k] for k in data.files if k in names or k.startswith('n_')}
        n = len(arrays['seed'])
        rows = [{} for _ in range(n)]
        for name in derived or ():
            value = data[name]
            for k in range(n):
                v = value[k]
                if name in DERIVED_STATE:
                    v = v[:1]
                if name in world.PADDED:  # loc_reach: [state, loc, 24] padded along its locs
                    v = np.take(v, range(arrays[f'n_{name}'][k]), axis=world.PADDED[name])
                rows[k][name] = np.ascontiguousarray(v)
            del value
    layouts = []
    for k in range(n):
        row = {}
        for name in names:
            value = arrays[name][k]
            if name in world.PADDED:
                count = arrays[f'n_{name}'][k]
                value = value[:count] if world.PADDED[name] == 0 else value[:, :count]
            row[name] = int(value) if name in ('seed', 'attempt') else value
        layouts.append(cls(**row))
    return layouts, (rows if derived else None)


# ---- strings and messages -------------------------------------------------------------------------------------------
class Strings:
    """The pack's interned strings: id 0 is ''; every one NUL-free ASCII (rs.h STRINGS)."""

    def __init__(self):
        self.ids, self.text = {'': 0}, ['']

    def __call__(self, s):
        if s is None:
            raise ExportError('a string field is None')
        s = str(s)
        k = self.ids.get(s)
        if k is None:
            if not s.isascii() or '\0' in s:
                raise ExportError(f'string {len(self.text)} is not NUL-free ASCII')
            k = self.ids[s] = len(self.text)
            self.text.append(s)
        return k

    def sections(self, names):
        """STRINGS, STR_OFF, STR_RANK, STR_LOWER, STR_NAME_ID and STR_ROW (kinds in RS_KIND_* order), lowercase forms
        interned first so STR_LOWER names a string of the table."""
        from act_contract import name_id
        for s in list(self.text):
            self(s.lower())
        blob = b''.join(s.encode() + b'\0' for s in self.text)
        offsets = np.cumsum([0] + [len(s) + 1 for s in self.text[:-1]]).astype(np.uint32)
        rank = np.empty(len(self.text), np.int32)
        rank[sorted(range(len(self.text)), key=self.text.__getitem__)] = np.arange(len(self.text), dtype=np.int32)
        c = contract()
        kinds = sorted(((c['RS_KIND_LOC'], 'loc'), (c['RS_KIND_NPC'], 'npc'), (c['RS_KIND_OBJ'], 'obj')))
        rows = np.asarray([[names.row(kind, s) for s in self.text] for _, kind in kinds], np.int32)
        return dict(STRINGS=blob, STR_OFF=offsets, STR_RANK=rank,
                    STR_LOWER=np.asarray([self.ids[s.lower()] for s in self.text], np.int32),
                    STR_NAME_ID=np.asarray([name_id(s) for s in self.text], np.int32), STR_ROW=rows.ravel())


def chat_counts(tokens, dim):
    """text_vector's signed bucket counts (rl/feedback_contract.py) of one text's tokens and its in-text pairs."""
    counts = np.zeros(dim, np.int64)
    for feature in tokens + [f'{a} {b}' for a, b in zip(tokens, tokens[1:])]:
        h = zlib.crc32(feature.encode())
        counts[h % dim] += 1 if (h >> 16) & 1 else -1
    return counts


class Messages:
    """RsMsg rows (deduplicated by text), their tokens, and the (key, fill) -> message map (rs.h Messages)."""

    def __init__(self, strings):
        from feedback_contract import TEXT_DIM
        self.s, self.dim = strings, TEXT_DIM
        self.rows, self.by_text, self.tokens, self.token_rows, self.msg_tokens, self.keys = [], {}, {}, [], [], {}

    def __call__(self, text):
        """The message id of a text."""
        from feedback_contract import words
        from sim.engine import Unsupported, _one_page
        if text in self.by_text:
            return self.by_text[text]
        if '<' in text or '>' in text:  # tags: a dialog's tokens would then not be its parts' tokens joined
            raise ExportError(f'message {len(self.rows)} holds a tag bracket')
        tokens = words(text)
        counts = chat_counts(tokens, self.dim)
        if np.abs(counts).max(initial=0) > 127:
            raise ExportError(f'message {len(self.rows)} has a bucket count beyond int8')
        try:
            _one_page(text)
            long_text = 0
        except Unsupported:
            long_text = 1
        ids = []
        for token in tokens:
            if token not in self.tokens:
                self.tokens[token] = len(self.token_rows)
                self.token_rows.append((zlib.crc32(token.encode()), zlib.crc32((token + ' ').encode()), len(token)))
            ids.append(self.tokens[token])
        self.by_text[text] = k = len(self.rows)
        self.rows.append((self.s(text), int(bool(text.strip())), long_text, len(ids), len(self.msg_tokens), counts))
        self.msg_tokens += ids
        return k

    def key(self, key, fill, text):
        """MSG_KEYS row (key, fill): the message of `text`, or for a text key the archive lacks (text None) the marker
        -(string id of the key) - 1 that rs_msg raises as RS_E_GAME_TEXT."""
        self.keys[(key, fill)] = self(text) if isinstance(text, str) else -self.s(text[1]) - 1

    def sections(self):
        msgs = record('RsMsg', len(self.rows))
        for k, (text, nonblank, long_text, n_tok, tok0, counts) in enumerate(self.rows):
            msgs[k]['text'], msgs[k]['nonblank'], msgs[k]['long_text'] = text, nonblank, long_text
            msgs[k]['n_tok'], msgs[k]['tok0'], msgs['chat'][k] = n_tok, tok0, counts
        tokens = record('RsToken', len(self.token_rows))
        for k, (crc, crc_sp, n) in enumerate(self.token_rows):
            tokens[k]['crc'], tokens[k]['crc_sp'], tokens[k]['len'] = crc, crc_sp, n
        keys = record('RsMsgKeyRow', len(self.keys))
        for k, ((key, fill), msg) in enumerate(sorted(self.keys.items())):
            keys[k]['key'], keys[k]['fill'], keys[k]['msg'] = key, fill, msg
        return dict(MSGS=msgs, MSG_KEYS=keys, TOKENS=tokens, MSG_TOKENS=np.asarray(self.msg_tokens, np.int32))


# ---- the oracle seen as a world ---------------------------------------------------------------------------------------
class _View:
    """What the oracle's content helpers read of a World (its tables, texts and members flag), for calling them at
    export with no world running."""

    def __init__(self, tables):
        self.t = tables
        self.texts = tables.meta.get('content', {}).get('texts', {})
        self.members = bool(tables.meta['environment']['members'])
        self.lines = []

    def mes(self, text):
        self.lines.append(('mes', None, text))

    def settext(self, component, text):
        self.lines.append(('settext', component, text))

    def text(self, key, **fill):
        from sim.engine import World
        return World.text(self, key, **fill)

    def _types(self, target):
        return target.types

    def obj_param(self, obj, name):
        from sim.engine import World
        return World.obj_param(self, obj, name)

    def npc_param(self, npc, name):
        from sim.engine import World
        return World.npc_param(self, npc, name)

    def loc_param(self, loc, name):
        from sim.engine import World
        return World.loc_param(self, loc, name)

    def loc_param_or_default(self, loc, name):
        from sim.engine import World
        return World.loc_param_or_default(self, loc, name)


def _value(fn, *args):
    """fn(*args), or RS_NONE where the oracle raises (a param with no default: the C side raises there too) or gives
    None."""
    from sim.engine import Unsupported
    try:
        v = fn(*args)
    except (KeyError, Unsupported):
        return contract()['RS_NONE']
    return contract()['RS_NONE'] if v is None else int(v)


# ---- the exporter -----------------------------------------------------------------------------------------------------
class Exporter:
    """Builds the pack's sections from the archive's tables and the pools (module docstring)."""

    def __init__(self, tables, names=None):
        from names import name_table
        from sim import engine
        self.t, self.names = tables, names or name_table()
        self.c, self.m = contract(), native.load()
        self.s = Strings()
        self.msg = Messages(self.s)
        self.view = _View(tables)
        self.scripts = self.m.scripts()
        self.script = {name: k for k, (name, _, _) in enumerate(self.scripts)}
        self.compact = tables.meta['content']['compact']
        self.content = tables.meta['content']
        self.categories = self.content['categories']
        self.npc_ids = {v['debugname']: k for k, v in tables.npc_types.items()}  # sim/combat.py's map: the last wins
        if self.c['RS_VARPS'] != tuple(engine.Player(None, PLAYER).varps):
            raise ExportError('rs.h RS_VARPS is not Player.varps in order')
        self.drop_rows, self.drop_ops, self.drop_roots, self.enum_rows = {}, [], [], {}
        self.drop_objs = set()
        self.wide_eat = set()  # eat rows that heal more stats than RsEatRow holds (declared_world refuses them)
        self.loot_own = None  # the loot list of the pack's village world when it declares its own (export sets it)

    # -- triggers to scripts
    def script_id(self, fn):
        """The rs.h script id of an oracle script function; closures by their bound cells (RS_SCRIPTS notes)."""
        from sim import engine
        q = fn.__qualname__
        cells = inspect.getclosurevars(fn).nonlocals if fn.__closure__ else {}
        S = self.script
        if q == 'mining.<locals>.script':
            fast = inspect.getclosurevars(cells['get_ore']).nonlocals['fast']
            return S[f"RS_S_MINE_{'FIRST' if cells['first'] else 'CONT'}_{'FAST' if fast else 'NORMAL'}"]
        if q == 'gate.<locals>.script':
            return S[f"RS_S_GATE_{'OPEN' if cells['opening'] else 'CLOSE'}_{'OUTER' if cells['outer'] else 'MAIN'}"]
        if q == 'door.<locals>.script':
            return S['RS_S_DOOR_OPEN' if cells['turn'] == 1 else 'RS_S_DOOR_CLOSE']
        if q == 'stairs.<locals>.script':
            return S['RS_S_STAIRS_UP' if cells['name'] == '[oploc1,loc_1722]' else 'RS_S_STAIRS_DOWN']
        if q == 'use_default.<locals>.script':  # HELD's, then the pick archive's (engine.FIELD_USE)
            held = next((k for k, v in engine.HELD.items() if v is fn), None) or \
                next((k for k, v in engine.ARCHIVE_SCRIPTS['picks'].items() if v is fn), None)
            if held not in USE_DEFAULT_IDS:
                raise ExportError(f'no rs.h script id names the use script {held}')
            return S[USE_DEFAULT_IDS[held]]
        if q == 'npc_default_retaliate.<locals>.script':
            return S['RS_S_NPC_DEFAULT_RETALIATE_AP' if cells['mode'] == engine.MODE_APPLAYER2 else 'RS_S_NPC_DEFAULT_RETALIATE']
        if q == 'flavour_timer.<locals>.script':
            return S['RS_S_FLAVOUR_TIMER']
        found = [k for k, (_, qual, status) in enumerate(self.scripts) if qual == q and status]
        if len(found) != 1:
            raise ExportError(f'no rs.h script id names {q}')
        return found[0]

    def _named(self, name, table):
        """(script, arg, name string) of a trigger looked up in one of the oracle's script dicts."""
        if not name:
            return self.script['RS_S_NONE'], -1, 0
        fn = table.get(name)
        return (self.script['RS_S_UNKNOWN'] if fn is None else self.script_id(fn)), -1, self.s(name)

    def _loc_trigger(self, name):
        """(script, arg, name string) of a loc trigger as World.run_trigger finds its script: engine.SCRIPTS, else the
        archive's (engine.archive_script: the pick over an archive with content.picks)."""
        from sim import engine
        if not name:
            return self.script['RS_S_NONE'], -1, 0
        fn = engine.SCRIPTS.get(name) or engine.archive_script(self.view, name)
        return (self.script['RS_S_UNKNOWN'] if fn is None else self.script_id(fn)), -1, self.s(name)

    def _held(self, name):
        """(script, arg, name string) of an OPHELD/OPHELDU trigger (engine.held_script); arg the eat trigger or wield
        rule row the script reads."""
        from sim import engine
        if not name:
            return self.script['RS_S_NONE'], -1, 0
        fn = engine.held_script(self.view, name)
        if fn is None:
            return self.script['RS_S_UNKNOWN'], -1, self.s(name)
        arg = self.eat_trigger_row[name] if fn is engine.eat else self.wield_row(name) if fn is engine.wield else -1
        return self.script_id(fn), arg, self.s(name)

    def _ops(self, ops):
        """(op string ids, op states, visible count, visible indices) of a type's op list (agent.py scan's raw)."""
        c = self.c
        ops = list(ops or [])
        if len(ops) > 5:
            raise ExportError('an op list longer than 5')
        ids, states, visible = [0] * 5, [c['RS_OPS_NONE']] * 5, []
        for k, o in enumerate(ops):
            if o:
                ids[k] = self.s(o)
                states[k] = c['RS_OPS_HIDDEN'] if o == 'hidden' else c['RS_OPS_VISIBLE']
                if o != 'hidden':
                    visible.append(k)
        return ids, states, len(visible), visible + [0] * (5 - len(visible))

    def _names(self, row, k, t):
        from sim.agent import name_of
        n = name_of(t)
        row['debugname'][k], row['name'][k] = self.s(t['debugname']), self.s(t['name'] or '')
        row['name_of'][k], row['name_of_lc'][k] = self.s(n), self.s(n.lower())

    def _bank_listed(self, t):
        """act_contract.banks of a listed loc or npc of this type: BANK_NAME on its name (name_of), BANK_OPTION on a
        visible op."""
        from act_contract import BANK_NAME, BANK_OPTION
        from sim.agent import name_of
        options = [o for o in (t['op'] or []) if o and o != 'hidden']
        return int(bool(BANK_NAME.search(name_of(t)) and any(BANK_OPTION.search(o) for o in options)))

    def loc_types(self):
        from sim import engine, world
        c, t = self.c, self.t
        rows = record('RsLocType', max(t.loc_types) + 1)
        trees, rocks = sorted(self.content['trees']), sorted(self.compact['rocks'])
        for k, lt in t.loc_types.items():
            rows['present'][k] = 1
            for field in ('blockwalk', 'blockrange', 'active', 'width', 'length', 'forceapproach', 'category'):
                rows[field][k] = int(lt[field])
            self._names(rows, k, lt)
            rows['op'][k], rows['op_state'][k], rows['n_opt'][k], rows['opt_index'][k] = self._ops(lt['op'])
            rows['bank_listed'][k] = self._bank_listed(lt)
            for j in range(c['RS_LTRIG_N']):
                trig = engine._trigger_name(t, engine.APLOC1 + j % 6 + (engine.AP_TO_OP if j >= c['RS_TRIG_OP'] else 0))
                rows['trig_script'][k, j], rows['trig_arg'][k, j], rows['trig_name'][k, j] = \
                    self._loc_trigger(lt['triggers'].get(trig))
            loc = SimpleNamespace(type=k)
            rows['next_loc_stage'][k] = _value(self.view.loc_param, loc, 'next_loc_stage')
            rows['next_loc_stage_mining'][k] = _value(self.view.loc_param_or_default, loc, 'next_loc_stage_mining')
            rows['mining_rock_empty'][k] = _value(self.view.loc_param_or_default, loc, 'mining_rock_empty')
            rows['tree_row'][k] = trees.index(lt['debugname']) if lt['debugname'] in trees else -1
            rows['rock_row'][k] = rocks.index(lt['debugname']) if lt['debugname'] in rocks else -1
            rows['is_loc_4030'][k] = lt['debugname'] == 'loc_4030'
            rows['supply_tree'][k] = lt['debugname'] in (world.TREE, world.OAK)
            rows['pen_gate'][k] = lt['debugname'] in world.PEN_GATE
            pick = self.content.get('picks', {}).get(lt['debugname'])  # sim/build.py add_picks (draynor-nc2)
            if pick is not None:
                rows['has_pick'][k], rows['pick_scaled'][k] = 1, bool(pick['scaled'])
                rows['pick_item'][k], rows['pick_respawn'][k] = _value(t.obj_id, pick['item']), int(pick['respawn'])
        return rows

    def npc_types(self, drops):
        """npc type rows; `drops`: {npc type: DROP_ROOTS row} of the compiled death roots."""
        from sim import engine
        c, t = self.c, self.t
        rows = record('RsNpcType', max(t.npc_types) + 1)
        kinds = {'chicken': 'CHICKEN', 'chicken_brown': 'CHICKEN', 'goblin': 'GOBLIN', 'mugger': 'MUGGER',
                 '0_46_49_saltfish': 'SPOT'}
        hunt_fields = dict(hunt_id='id', hunt_type='type', hunt_rate='rate', hunt_check_vis='checkVis',
                           hunt_check_not_too_strong='checkNotTooStrong', hunt_check_not_combat='checkNotCombat',
                           hunt_check_not_combat_self='checkNotCombatSelf', hunt_check_inv='checkInv',
                           hunt_nobody_near='nobodyNear', hunt_find_new_mode='findNewMode')
        params = ('attackbonus', 'strengthbonus', 'damagetype', 'stabdefence', 'slashdefence', 'crushdefence',
                  'rangedefence', 'magicdefence', 'attackrate', 'max_dealt', 'combat_xp_multiplier', 'retreat',
                  'poison_severity', 'magicattack')
        for k, nt in t.npc_types.items():
            rows['present'][k] = 1
            for field in ('givechase', 'members', 'size', 'defaultmode', 'moverestrict', 'wanderrange', 'maxrange',
                          'attackrange', 'huntmode', 'huntrange', 'regenrate', 'respawnrate', 'timer', 'category'):
                rows[field][k] = int(nt[field])
            rows['kind'][k] = c[f"RS_NPCK_{kinds.get(nt['debugname'], 'OTHER')}"]
            rows['vislevel'][k] = c['RS_NONE'] if nt['vislevel'] is None else nt['vislevel']
            rows['vislevel0'][k] = nt['vislevel'] or 0
            if len(nt['stats']) != 6:
                raise ExportError(f'npc type {k} has {len(nt["stats"])} stats')
            rows['stats'][k] = nt['stats']
            hunt = nt['hunt']
            if hunt is not None:
                rows['has_hunt'][k] = 1
                rows['hunt_check_not_busy'][k] = bool(hunt['checkNotBusy'])
                rows['hunt_check_vars'][k] = bool(hunt['checkVars'])
                rows['hunt_find_keep'][k] = bool(hunt['findKeepHunting'])
                for field, key in hunt_fields.items():
                    rows[field][k] = int(hunt[key])
            npc = SimpleNamespace(type=k)
            for name in params:
                rows[name][k] = _value(self.view.npc_param, npc, name)
            self._names(rows, k, nt)
            rows['op'][k], rows['op_state'][k], rows['n_opt'][k], rows['opt_index'][k] = self._ops(nt['op'])
            options = [o for o in (nt['op'] or []) if o and o != 'hidden']
            rows['attack_op'][k] = any('attack' in str(o).lower() for o in options)
            try:
                rows['attackable'][k] = engine.npc_is_attackable(self.view, None, SimpleNamespace(type=k))
            except engine.Unsupported:
                rows['refuse_attack'][k] = 1
            rows['bank_listed'][k] = self._bank_listed(nt)
            for j in range(c['RS_NTRIG_N']):
                if j < c['RS_NTRIG_TIMER']:
                    trig = engine._trigger_name(t, engine.APNPC1 + j % 6 + (engine.AP_TO_OP if j >= c['RS_TRIG_OP'] else 0))
                    entry = self._named(nt['triggers'].get(trig), engine.SCRIPTS)
                elif j == c['RS_NTRIG_TIMER']:
                    entry = self._named(nt['triggers'].get('AI_TIMER'), engine.NPC_SCRIPTS)
                elif j < c['RS_NTRIG_OPPLAYER']:
                    queue = j - c['RS_NTRIG_QUEUE'] + 1
                    name = nt['triggers'].get(f'AI_QUEUE{queue}')
                    if queue < 3 or not name:
                        entry = self._named(name, engine.NPC_QUEUES)
                    elif k in drops:
                        entry = self.script['RS_S_NPC_DEATH_SCRIPT'], drops[k], self.s(name)
                    else:
                        entry = self.script['RS_S_NPC_DEATH_NP'], -1, self.s(name)
                else:
                    ap = j >= c['RS_NTRIG_APPLAYER']
                    n = j - (c['RS_NTRIG_APPLAYER'] if ap else c['RS_NTRIG_OPPLAYER']) + 1
                    entry = self._named(nt['triggers'].get(f"AI_{'AP' if ap else 'OP'}PLAYER{n}"), engine.NPC_AI)
                rows['trig_script'][k, j], rows['trig_arg'][k, j], rows['trig_name'][k, j] = entry
        return rows

    def obj_flags(self, k, ot):
        from sim import engine
        from sim.agent import TOOL_PROTECT_RE, name_of
        c, cats = self.c, self.categories
        cases = {key: inspect.getclosurevars(fn).nonlocals for key, fn in engine.HELD.items() if fn.__closure__}
        own = self.loot_own
        flags = [('TOOL_PROTECT', bool(TOOL_PROTECT_RE.search(name_of(ot)))),
                 ('COINS', name_of(ot).lower() == 'coins'),
                 ('LOOT_OUTSIDE', engine.loot_outside(self.t, k)),
                 ('LOOT_OUTSIDE_OWN', own is not None and engine.loot_outside(self.t, k, own)),
                 ('TRAIL', ot['category'] in engine.trail_categories(self.t)),
                 ('GLORY', ot['category'] == cats['category_557']),
                 ('POTION', ot['category'] == cats['potion']),
                 ('OIL', ot['category'] == cats['category_111'])]
        for key, flag in (('[opheldu,_weapon_stab]', 'CASE_WEAPON_STAB'), ('[opheldu,feather]', 'CASE_FEATHER'),
                          ('[opheldu,_category_22]', 'CASE_CATEGORY_22')):
            cell = cases[key]
            flags.append((flag, ot['debugname'] in cell['cases'] or any(ot['category'] == cats[x] for x in cell['categories'])))
        for key, flag in FIELD_USE_FLAGS.items():  # over an archive that resolves them only (engine.archive_script)
            fn = engine.archive_script(self.view, key)
            cell = inspect.getclosurevars(fn).nonlocals if fn is not None else None
            flags.append((flag, cell is not None and (ot['debugname'] in cell['cases']
                                                      or any(ot['category'] == cats[x] for x in cell['categories']))))
        flags += [('EQUIP_REFUSED', ot['debugname'] in ('elemental_shield', 'ibanstaff')),
                  ('WEIGHT', ot['debugname'] == 'ikov_bootsoflightness'),
                  ('FISH_EQUIPMENT', 'fish_equipment_struct' in ot['params']
                   or ot['debugname'] in ('fly_fishing_rod', 'fishing_rod', 'big_net')),
                  ('NET', ot['debugname'] == 'net'),
                  ('ARROWS', ot['category'] == cats['arrows']), ('BOLTS', ot['category'] == cats['bolts']),
                  ('BOW', ot['category'] == cats['weapon_bow']), ('CROSSBOW', ot['category'] == cats['weapon_crossbow']),
                  ('UNBANKABLE', bool(ot['params'].get('unbankable')))]
        return sum(c[f'RS_OF_{name}'] for name, on in flags if on)

    def obj_types(self):
        from baseline.resource_task import GOAL_FAMILIES, GOALS
        from sim import engine
        from sim.task import object_families
        c, t = self.c, self.t
        rows = record('RsObjType', max(t.obj_types) + 1)
        made = object_families(t)
        families = []
        for goal in GOALS:  # GoalEpisode.family: the goal items and their notes
            family = set(GOAL_FAMILIES.get(goal, ()))
            for base in list(family):
                note = t.obj_types.get(base, {}).get('certlink', -1)
                if note is not None and note >= 0:
                    family.add(note)
            families.append(family)
        axes = {t.obj_ids.get(name): r for r, name in enumerate(engine.AXES) if t.obj_ids.get(name) is not None}
        pickaxes = {t.obj_ids.get(name): r for r, name in enumerate(self.compact['pickaxes'])
                    if t.obj_ids.get(name) is not None}
        eats = sorted(self.compact['eat'])
        for k, ot in t.obj_types.items():
            rows['present'][k] = 1
            for field in ('stackable', 'members', 'dummyitem', 'category', 'certlink', 'certtemplate', 'respawnrate'):
                rows[field][k] = -1 if ot[field] is None else int(ot[field])
            rows['wearpos'][k] = [-1 if ot[f] is None else ot[f] for f in ('wearpos', 'wearpos2', 'wearpos3')]
            rows['take_op'][k] = next((i for i, o in enumerate(ot['op'] or []) if o and 'take' in o.lower()), 2)
            rows['flags'][k] = self.obj_flags(k, ot)
            self._names(rows, k, ot)
            rows['iop'][k], rows['iop_state'][k], rows['n_iopt'][k], rows['iopt_index'][k] = self._ops(ot['iop'])
            heldu = ot['triggers'].get('OPHELDU')
            rows['heldu_own'][k] = heldu is not None and not heldu.split(',')[1].startswith('_')
            rows['axe_rank'][k], rows['pickaxe_rank'][k] = axes.get(k, -1), pickaxes.get(k, -1)
            rows['made_family'][k] = made[k]
            rows['family_mask'][k] = sum(1 << g for g, family in enumerate(families) if k in family)
            rows['style_table'][k] = self.style_row(k)
            rows['eat_row'][k] = eats.index(ot['debugname']) if ot['debugname'] in eats else -1
            for field in ('levelrequire', 'mining_rate', 'bone_exp', 'next_obj_stage', 'attackrange', 'attackrate',
                          'poison_severity', 'rangebonus'):
                rows[field][k] = _value(self.view.obj_param, k, field)
            rows['bonus'][k] = [_value(self.view.obj_param, k, name) for name in engine.BONUSES]
            for j in range(c['RS_OTRIG_N']):
                if j < c['RS_OTRIG_HELD']:
                    trig = engine._trigger_name(t, engine.APOBJ1 + j % 6 + (engine.AP_TO_OP if j >= c['RS_TRIG_OP'] else 0))
                    entry = self._named(ot['triggers'].get(trig), engine.SCRIPTS)
                elif j < c['RS_OTRIG_HELDU']:
                    entry = self._held(ot['triggers'].get(f"OPHELD{j - c['RS_OTRIG_HELD'] + 1}"))
                else:
                    # agent.use_on_held never runs '[opheldu,_]' (its script() returns None for it): no script
                    entry = self._held(None if heldu == '[opheldu,_]' else heldu)
                rows['trig_script'][k, j], rows['trig_arg'][k, j], rows['trig_name'][k, j] = entry
        return rows

    # -- content rows
    def style_row(self, obj):
        """The STYLES row of weapon_styles(world, obj) (engine.py:2325) in this archive's members world."""
        from sim import engine
        table = engine.weapon_styles(self.view, obj)
        return next(r for r, name in enumerate(sorted(self.compact['styles'])) if self.compact['styles'][name] is table)

    def wield_row(self, name):
        """The WIELD_RULES row engine.wield reads for this trigger (c['wield'] by name; '[opheld2,_]' -> equip)."""
        rules = self.compact['wield']
        return sorted(rules).index(name) if name in rules else len(rules)

    def content_rows(self):
        from sim import engine
        c, t = self.c, self.t
        out = {}
        trees = record('RsTreeRow', len(self.content['trees']))
        for r, name in enumerate(sorted(self.content['trees'])):
            row = self.content['trees'][name]
            trees[r]['levelrequired'], trees[r]['respawnrate'] = row['levelrequired'], row['respawnrate']
            trees[r]['product'] = _value(t.obj_id, row['product'])
            trees[r]['productexp'] = row['productexp']
            for a, axe in enumerate(engine.AXES):
                if axe in row['successchance']:
                    trees[r]['axe_mask'] |= 1 << a
                    trees['low'][r, a], trees['high'][r, a] = row['successchance'][axe]
            self.msg.key(c['RS_MK_TEXT'] + c['RS_TX_wc_level'], row['levelrequired'],
                         self.text('wc_level', level=row['levelrequired']))
            product = t.obj_id(row['product'])
            self.msg.key(c['RS_MK_TEXT'] + c['RS_TX_wc_get'], product,
                         self.text('wc_get', product=t.obj_types[product]['name'].lower()))
        out['TREES'] = trees
        rocks = record('RsRockRow', len(self.compact['rocks']))
        for r, name in enumerate(sorted(self.compact['rocks'])):
            row = self.compact['rocks'][name]
            rocks[r]['level'], (rocks[r]['low'], rocks[r]['high']) = row['level'], row['chance']
            rocks[r]['output'] = _value(t.obj_id, row['output'])
            rocks[r]['exp'], rocks[r]['respawn'], rocks[r]['ore'] = row['exp'], row['respawn'], self.s(row['ore'])
            for key in ('mine_full', 'mine_get', 'prospect_ore'):
                self.msg.key(c['RS_MK_TEXT'] + c[f'RS_TX_{key}'], r, self.text(key, ore=row['ore']))
            self.msg.key(c['RS_MK_TEXT'] + c['RS_TX_mine_level'], row['level'], self.text('mine_level', level=row['level']))
        out['ROCKS'] = rocks
        gems = record('RsGemRow', len(self.compact['gems']))
        for r, (below, gem) in enumerate(self.compact['gems']):
            gems[r]['below'], gems[r]['gem'] = below, t.obj_id(gem)
            gems[r]['cut'] = t.obj_id(self.compact['cut_gems'][gem])
            self.msg.key(c['RS_MK_TEXT'] + c['RS_TX_mine_gem'], t.obj_id(gem),
                         self.text('mine_gem', gem=engine.add_article(t.obj_types[gems[r]['cut']]['name'])))
        if list(gems['below']) != sorted(gems['below']):
            raise ExportError('the gem rows are not ascending')
        out['GEMS'] = gems
        fish = record('RsFishRow', 2)
        for r, name in enumerate(('raw_shrimp', 'raw_anchovies')):
            row = self.content['fish'][name]
            fish[r]['obj'], fish[r]['success_low'], fish[r]['success_high'] = t.obj_id(name), row['success_low'], row['success_high']
            fish[r]['productexp'], fish[r]['message'] = row['productexp'], self.msg(row['productmessage'])
        out['FISH'] = fish
        triggers = record('RsEatTrigger', len(self.compact['eat_triggers']))
        self.eat_trigger_row = {}
        for r, name in enumerate(sorted(self.compact['eat_triggers'])):
            effect, eat_delay, skill_delay = self.compact['eat_triggers'][name]
            self.eat_trigger_row[name] = r
            triggers[r]['effect'], triggers[r]['effect_stat'] = self.s(effect), effect == 'consume_effect_stat'
            triggers[r]['eat_delay'] = c['RS_NONE'] if eat_delay is None else eat_delay
            triggers[r]['skill_delay'] = c['RS_NONE'] if skill_delay is None else skill_delay
        out['EAT_TRIGGERS'] = triggers
        eats = record('RsEatRow', len(self.compact['eat']))
        for r, name in enumerate(sorted(self.compact['eat'])):
            row, m = self.compact['eat'][name], self.compact['eat_messages'].get(name, {})
            eats[r]['has_row'], eats[r]['changes'] = 1, bool(row['stat_change'] or row['healenergy'])
            eats[r]['message_delay'], eats[r]['n_heal'] = 'message_delay' in m, min(len(row['stat_heal']), 4)
            if len(row['stat_heal']) > 4 and not eats[r]['changes']:  # carried only where no reachable obj eats it
                self.wide_eat.add(name)
            for h, (stat, constant, percent) in enumerate(row['stat_heal'][:4]):
                eats['heal_stat'][r, h] = engine.SKILLS.index(stat)
                eats['heal_constant'][r, h], eats['heal_percent'][r, h] = constant, percent
            for field, key in (('c1', 'consume_message1'), ('c2', 'consume_message2'), ('r1', 'restore_message1'),
                               ('r2', 'restore_message2')):
                text = m.get(key, '')
                eats[r][field] = c['RS_EAT_EMPTY'] if text == '' else c['RS_EAT_ZERO'] if text == '0' else self.msg(text)
            foods = [k for k, ot in t.obj_types.items() if ot['debugname'] == name]
            texts = {self.text('eat', food=t.obj_types[k]['name'].lower()) for k in foods}
            if len(texts) > 1:
                raise ExportError(f'eat row {r}: objs of one debugname with different eat messages')
            for k in foods:
                self.msg.key(c['RS_MK_TEXT'] + c['RS_TX_eat'], k, self.text('eat', food=t.obj_types[k]['name'].lower()))
            text = texts.pop() if texts else None
            eats[r]['eat_default'] = self.msg(text) if isinstance(text, str) else c['RS_NONE']
        out['EAT_ROWS'] = eats
        rules = self.compact['wield']
        wield = record('RsWieldRule', len(rules) + 1)
        for r, (label, args) in enumerate([rules[name] for name in sorted(rules)] + [('equip', ['last_slot'])]):
            if label == 'equip':
                wield[r]['kind'] = c['RS_WIELD_EQUIP']
            elif label == 'tutorial_island_equip':  # becomes @levelrequire_attack(1) past the tutorial (engine.py:2922)
                wield[r]['kind'], label = c['RS_WIELD_TUTORIAL'], 'levelrequire_attack'
                wield[r]['stat'], wield[r]['level'] = engine.LEVELREQUIRE[label], 1
            elif label in engine.LEVELREQUIRE:
                wield[r]['kind'], wield[r]['stat'], wield[r]['level'] = c['RS_WIELD_LEVELREQUIRE'], engine.LEVELREQUIRE[label], args[0]
            else:
                wield[r]['kind'] = c['RS_WIELD_OTHER']
            wield[r]['label'] = self.s(label)
        out['WIELD_RULES'] = wield
        styles = record('RsStyleTable', len(self.compact['styles']))
        for r, name in enumerate(sorted(self.compact['styles'])):
            row = self.compact['styles'][name]
            if len(row['damagestyle']) != len(row['damagetype']) or not 1 <= len(row['damagestyle']) <= 4:
                raise ExportError(f'style table {r} is not 1-4 rows of type and style')
            styles[r]['n'] = len(row['damagestyle'])
            styles['damagetype'][r, :styles[r]['n']] = row['damagetype']
            styles['damagestyle'][r, :styles[r]['n']] = row['damagestyle']
        out['STYLES'] = styles
        out['LEVELUP'], out['UNLOCKS'] = self.levelups()
        for key in self.c['RS_TEXT_KEYS']:  # the keys whose templates take no fill
            if key not in ('wc_level', 'wc_get', 'mine_full', 'mine_level', 'mine_gem', 'mine_get', 'prospect_ore', 'eat',
                           'pick_full', 'pick'):
                self.msg.key(c['RS_MK_TEXT'] + c[f'RS_TX_{key}'], 0, self.text(key))
        picks = self.content.get('picks', {})  # a plant's texts, by its loc type (engine.pick)
        for k, lt in t.loc_types.items():
            if lt['debugname'] in picks:
                for key in ('pick_full', 'pick'):
                    self.msg.key(c['RS_MK_TEXT'] + c[f'RS_TX_{key}'], k, self.text(f"{key}_{lt['debugname']}"))
        for k, lt in t.loc_types.items():
            self.msg.key(c['RS_MK_NO_TRIGGER_OPLOCU'], k, self.no_trigger(lt, engine.APLOCU))
        for k, nt in t.npc_types.items():
            self.msg.key(c['RS_MK_NO_TRIGGER_OPNPCU'], k, self.no_trigger(nt, engine.APNPCU))
        cant = [v for v in engine.World.process_interaction.__code__.co_consts if isinstance(v, str) and 'reach' in v]
        if len(cant) != 1:
            raise ExportError("process_interaction does not hold one \"can't reach\" text")
        self.msg.key(c['RS_MK_CANT_REACH'], 0, cant[0])
        from sim.agent import NO_HELDU_TRIGGER, NOTHING_INTERESTING  # a use on a held item with no script
        for k, ot in t.obj_types.items():
            self.msg.key(c['RS_MK_NO_TRIGGER_OPHELDU'], k, NO_HELDU_TRIGGER.format(ot['debugname']))
        self.msg.key(c['RS_MK_NOTHING_INTERESTING'], 0, NOTHING_INTERESTING)
        return out

    def text(self, key, **fill):
        """world.text(key, **fill), or ('missing', key) for a key the archive lacks (Unsupported there)."""
        from sim.engine import Unsupported
        try:
            return self.view.text(key, **fill)
        except Unsupported:
            return ('missing', key)

    def no_trigger(self, types, op):
        """World.default_op's first line for a type (engine.py:1553-1561), from the oracle itself."""
        from sim.engine import Unsupported, World
        self.view.lines.clear()
        try:
            World.default_op(self.view, SimpleNamespace(target_op=op, clear_waypoints=lambda: None),
                             SimpleNamespace(types=types))
        except Unsupported:  # dm_default missing: the first line is already out
            pass
        return self.view.lines[0][2]

    def levelups(self):
        """LEVELUP rows by SKILLS index and their UNLOCKS, from running the oracle's levelup script (engine.py:3237) for
        every level on a view that records its lines."""
        from sim import engine
        c = self.c
        rows, unlocks, seen = record('RsLevelupRow', len(engine.SKILLS)), [], {}
        content = self.content['levelup']
        for stat, skill in enumerate(engine.SKILLS):
            if skill not in content:
                continue
            row = content[skill]
            rows[stat]['present'], rows[stat]['iface'] = 1, self.s(row['interface'])
            for level in range(100):
                p = SimpleNamespace(base_levels={stat: level}, open_chat=lambda iface: None)
                self.view.lines.clear()
                try:
                    for _ in engine.levelup(self.view, p, stat):
                        pass
                    long_text = 0
                except engine.Unsupported:  # _one_page: the unlock text needs several pages
                    long_text = 1
                lines = self.view.lines
                (_, _, message), (_, (iface, one), dbl), (_, _, prefix) = lines[:3]
                if iface != row['interface'] or one != 1:
                    raise ExportError(f'levelup {skill} writes another interface')
                rows[stat]['message'], rows[stat]['dbl'] = self.msg(message), self.msg(dbl)
                rows['level_msg'][stat, level] = self.msg(prefix)
                rows['unlocks'][stat, level] = level in row['unlocks']
                if level not in row['unlocks'] or str(level) not in row['unlock_texts']:
                    rows['unlock'][stat, level] = -1
                    continue
                kind, text = row['unlock_texts'][str(level)]
                if (kind, text) not in seen:
                    parts = [line for line in lines[3:] if line[0] == 'settext']
                    if len(parts) > c['RS_UNLOCK_LINES']:
                        raise ExportError(f'an unlock text of {skill} has more than {c["RS_UNLOCK_LINES"]} lines')
                    seen[(kind, text)] = len(unlocks)
                    unlocks.append((c['RS_UNLOCK_OBJBOX'] if kind == 'objbox' else c['RS_UNLOCK_DOUBLEOBJBOX']
                                    if kind == 'doubleobjbox' else c['RS_UNLOCK_MESBOX'], long_text,
                                    [self.msg(line) for _, _, line in parts] if not long_text else [self.msg(text)]))
                rows['unlock'][stat, level] = seen[(kind, text)]
        table = record('RsUnlock', len(unlocks))
        for r, (kind, long_text, msgs) in enumerate(unlocks):
            table[r]['kind'], table[r]['long_text'], table[r]['n_lines'] = kind, long_text, len(msgs)
            table['msg'][r, :len(msgs)] = msgs
        return rows, table

    def consts(self):
        from sim import agent, combat, engine, world
        from sim.spec import START_KIT, START_WEAPONS
        from baseline.resource_task import MELEE_ENEMIES
        t = self.t
        out = record('RsConsts', 1)
        r = out[0]
        get = lambda name: -1 if t.obj_ids.get(name) is None else t.obj_ids[name]
        out['level_xp'][0] = engine.LEVEL_XP
        out['axes'][0] = [get(name) for name in engine.AXES]
        pickaxes = self.compact['pickaxes']
        if len(pickaxes) > 8:
            raise ExportError('more than 8 pickaxes')
        r['n_pickaxes'] = len(pickaxes)
        out['pickaxes'][0, :len(pickaxes)] = [get(name) for name in pickaxes]
        for field in ('gem_chance', 'gem_roll', 'lootdrop_duration', 'tutorial_equip'):
            r[field] = self.compact[field]
        r['style_unarmed'] = sorted(self.compact['styles']).index('weapon_unarmed_table')
        for name in ('gate_main_closed', 'gate_main_open', 'gate_outer_closed', 'gate_outer_open'):
            r[f'cat_{name}'] = self.categories[name]
        types = world._Types(t)
        r['npc_goblin'], r['npc_mugger'] = self.npc_ids['goblin'], self.npc_ids['mugger']
        r['npc_chicken'], r['npc_chicken_brown'] = self.npc_ids['chicken'], self.npc_ids['chicken_brown']
        if {int(r['npc_chicken']), int(r['npc_chicken_brown'])} != set(MELEE_ENEMIES):
            raise ExportError('the chicken types are not MELEE_ENEMIES')
        r['npc_spot'], r['loc_tree'], r['loc_oaktree'] = types.spot, types.tree, types.oak
        for field, name in (('obj_coins', 'coins'), ('obj_net', 'net'), ('obj_fishing_rod', 'fishing_rod'),
                            ('obj_fishing_bait', 'fishing_bait'), ('obj_raw_shrimp', 'raw_shrimp'),
                            ('obj_raw_anchovies', 'raw_anchovies')):
            r[field] = get(name)
        out['kit'][0] = [t.obj_id(name) for name, _ in START_KIT]
        out['kit_count'][0] = [count for _, count in START_KIT]
        out['weapons'][0] = [t.obj_id(name) for name in START_WEAPONS]
        out['combat_tools'][0] = [t.obj_id(name) for name, _ in combat.COMBAT['tools']]
        out['combat_tool_count'][0] = [count for _, count in combat.COMBAT['tools']]
        out['combat_food'][0] = [t.obj_id(name) for name in combat.COMBAT['food']]
        for field, text in (('s_take', agent.TAKE), ('s_use', agent.USE), ('s_drop', 'Drop'),
                            ('s_deposit', 'Deposit'), ('s_withdraw', 'Withdraw'), ('s_wield', 'Wield'),
                            ('s_message1', 'message1'), ('s_objbox1', 'objbox1'), ('s_doubleobjbox', 'doubleobjbox'),
                            ('s_bank_main', 'bank_main'), ('s_produce_get_logs', '[label,get_logs]'),
                            ('s_produce_ore_normal', '[label,get_ore_normal]'),
                            ('s_produce_ore_fast', '[label,get_ore_fast]'), ('s_produce_fish_roll', '[proc,fish_roll]'),
                            ('s_death_no_hero', 'a player command before npc_findhero')):
            r[field] = self.s(text)
        out['s_click'][0] = [self.s(f'{dx},{dz}') for dx, dz in agent.CLICK_OFFSETS]  # a click's meaning option
        # a pick's produce script, interned only over an archive with the pick rows (the other packs' strings unchanged)
        r['s_produce_pick'] = self.s('[proc,pickup_loc_floor]') if 'picks' in self.content else self.c['RS_NONE']
        r['members'] = self.view.members
        return out

    # -- death drop programs
    def drops(self, types, excluded):
        """DROP_ROOTS, DROP_OPS and ENUMS for the reachable npc types' [ai_queue3] roots: {type: DROP_ROOTS row}. A root
        that compiles to nothing the port can run raises ExportError naming why; `excluded`: the types whose worlds
        exclude clue and loot drops (combat's)."""
        out = {}
        for k in sorted(types):
            name = self.t.npc_types[k]['triggers'].get('AI_QUEUE3')
            if not name:
                continue
            ops, reached = DropCompiler(self, k, name, k in excluded).compile()
            self.drop_rows[k] = out[k] = len(self.drop_roots)
            scripts = self.content['combat']['scripts']
            source = json.dumps([[s, scripts.get(s)] for s in sorted(reached)], sort_keys=True).encode()
            self.drop_roots.append((self.s(name), len(self.drop_ops), len(ops), hashlib.sha256(source).digest()))
            self.drop_ops += ops
        return out

    def drop_sections(self):
        roots = record('RsDropRoot', len(self.drop_roots))
        for r, (name, first, n, digest) in enumerate(self.drop_roots):
            roots[r]['name'], roots[r]['first'], roots[r]['n_ops'] = name, first, n
            roots['ast_sha256'][r] = np.frombuffer(digest, np.uint8)
        ops = record('RsDropOp', len(self.drop_ops))
        for field, k in zip(('op', 'a', 'b', 'c', 'd'), range(5)):
            ops[field] = [op[k] for op in self.drop_ops]
        enums = record('RsEnumRow', len(self.enum_rows))
        for r, ((table, key), (value, default)) in enumerate(sorted(self.enum_rows.items())):
            enums[r]['table'], enums[r]['key'], enums[r]['value'], enums[r]['default_int'] = table, key, value, default
        return dict(DROP_ROOTS=roots, DROP_OPS=ops, ENUMS=enums)


# ---- death drop programs ----------------------------------------------------------------------------------------------
class _Dead(Exception):
    """A path that cannot continue: a return or jump left its frame, or an unsupported raised."""


# Compile-time values: ('int', n) and ('str', s) constants, ('reg', r, kind, temporary, range) a register (kind 'int',
# or 'obj' for an obj id made from an obj name; range (lo, hi) of the values it can hold, or None),
# ('tuple', [values]) a proc's several values, ('none',) npc_death's None.
def R(r, kind='int', temp=True, rng=None):
    return ('reg', r, kind, temp, rng)


def _kind(v):
    return 'obj' if v[0] == 'str' or (v[0] == 'reg' and v[2] == 'obj') else 'int'


def _range(v):
    return (v[1], v[1]) if v[0] == 'int' else v[4] if v[0] == 'reg' else None


def _hull(ranges):
    ranges = list(ranges)
    return None if not ranges or None in ranges else (min(r[0] for r in ranges), max(r[1] for r in ranges))


def _decide(op, a, b):
    """Whether `a op b` holds for every value of the ranges a and b (True / False), or None."""
    if a is None or b is None:
        return None
    (alo, ahi), (blo, bhi) = a, b
    if op in ('=', '!'):
        if alo == ahi == blo == bhi:
            return op == '='
        if ahi < blo or bhi < alo:
            return op == '!'
        return None
    holds, fails = dict(zip(('<', '<=', '>', '>='), ((ahi < blo, alo >= bhi), (ahi <= blo, alo > bhi),
                                                       (alo > bhi, ahi <= blo), (alo >= bhi, ahi < blo))))[op]
    return True if holds else False if fails else None


def _refine(op, rng, k, truth):
    """The range of x where `x op k` is `truth`, from x's range (None: unknown or empty)."""
    if rng is None:
        return None
    lo, hi = rng
    if not truth:
        op = {'<': '>=', '<=': '>', '>': '<=', '>=': '<', '=': '!', '!': '='}[op]
    if op == '<':
        hi = min(hi, k - 1)
    elif op == '<=':
        hi = min(hi, k)
    elif op == '>':
        lo = max(lo, k + 1)
    elif op == '>=':
        lo = max(lo, k)
    elif op == '=':
        lo, hi = max(lo, k), min(hi, k)
    elif lo == hi == k:
        return None
    elif lo == k:
        lo += 1
    elif hi == k:
        hi -= 1
    return (lo, hi) if lo <= hi else None


class DropCompiler:
    """One [ai_queue3] root for one npc type compiled to rs.h's drop program (DOP_*): rs2.run's semantics with
    DeathContext (engine.py:2775-2901), statement by statement and in Python's evaluation order, so the draws count in
    call order. Procs and labels are inlined, each frame with its own locals and result registers; strings are
    compile-time constants (an obj name becomes its obj id where it must go into a register); npc_param, the npc names
    and members are constants of the type and the archive; compare, enum_getoutputcount and constant arithmetic fold.
    Jumps name pcs of the program. Where DeathContext raises Unsupported the program has DOP_UNSUPPORTED with its text;
    a construct outside this set (or where Python raises another class) raises ExportError, and the declared-world
    check then refuses the pools."""

    def __init__(self, ex, npc_type, root, excluded):
        self.ex, self.type, self.root, self.excluded = ex, npc_type, root, excluded
        self.nt = ex.t.npc_types[npc_type]
        combat = ex.content['combat']
        self.scripts, self.consts, self.enums = combat['scripts'], combat['constants'], ex.t.meta['named_enums']
        self.ops, self.free, self.reached, self.arities = [], list(range(contract()['RS_DROP_REGS'])), set(), {}
        self.d = {name: value for name, value in contract().items() if name.startswith('DOP_')}

    def fail(self, why):
        raise ExportError(f'death program of npc type {self.type} ({self.root}): {why}')

    # -- registers
    def alloc(self):
        if not self.free:
            self.fail(f'more than {contract()["RS_DROP_REGS"]} registers')
        return self.free.pop(0)

    def give_back(self, regs):
        self.free = sorted(set(self.free) | set(regs))

    def release(self, v):
        if v[0] == 'reg' and v[3]:
            self.give_back([v[1]])
        elif v[0] == 'tuple':
            for x in v[1]:
                self.release(x)

    def emit(self, op, a=0, b=0, c=0, d=0):
        self.ops.append([self.d[op], int(a), int(b), int(c), int(d)])
        return len(self.ops) - 1

    def patch(self, at):
        self.ops[at][1] = len(self.ops)

    def obj_id(self, name):
        obj = self.ex.t.obj_ids.get(name)
        if obj is None:
            self.fail('an obj name that resolves to no obj must go into a register')
        self.ex.drop_objs.add(obj)
        return obj

    def constant(self, v, kind):
        if v[0] == 'int':
            return v[1]
        if v[0] == 'str' and kind == 'obj':
            return self.obj_id(v[1])
        self.fail('a string, a pair or None where a number is read')

    def reg(self, v, kind='int'):
        """(register, temporary) holding v's value, kind 'obj' resolving an obj name."""
        if v[0] == 'reg':
            if v[2] == 'obj' and kind != 'obj':
                self.fail('an obj id read as a number')
            return v[1], v[3]
        r = self.alloc()
        self.emit('DOP_CONST', r, self.constant(v, kind))
        return r, True

    def done_with(self, *pairs):
        self.give_back([r for r, temp in pairs if temp])

    def into(self, v, r, kind='int'):
        """r = v's value (v released)."""
        if v[0] == 'reg':
            if v[2] == 'obj' and kind != 'obj':
                self.fail('an obj id read as a number')
            if v[1] != r:
                self.emit('DOP_MOVE', r, v[1])
            self.release(v)
        else:
            self.emit('DOP_CONST', r, self.constant(v, kind))

    def unsupported(self, what):
        """DeathContext.unsupported(what) here: the path ends."""
        self.emit('DOP_UNSUPPORTED', self.ex.s(what))
        raise _Dead()

    def require_hero(self):
        """DeathContext.player() alone: a varp read into a scratch register (raises without a hero)."""
        r = self.alloc()
        self.emit('DOP_VARP', r, 0)
        self.give_back([r])

    # -- frames (rs2.run)
    def compile(self):
        """(ops, names of the scripts reached)."""
        try:
            self.run(self.root, [], discard=True)
            self.emit('DOP_END')
        except _Dead:
            pass
        if len(self.ops) > 1 << 15:
            self.fail('more than 32768 ops')
        return [tuple(op) for op in self.ops], self.reached

    def release_all(self, values):
        for v in values:
            self.release(v)

    def arity(self, name):
        """How many values a frame of `name` gives (its returns and jumps; refused if they differ)."""
        if name in self.arities:
            return self.arities[name]
        self.arities[name] = 1  # a recursive proc, if any, is refused below by the differing count
        script, found = self.scripts.get(name), set()

        def walk(body):
            for stmt in body:
                if stmt[0] == 'return':
                    found.add(sum(self.expr_arity(e) for e in stmt[1]))
                elif stmt[0] == 'jump':
                    found.add(0)
                elif stmt[0] == 'if':
                    for _, block in stmt[1]:
                        walk(block)
                    walk(stmt[2])
        if script is not None:
            walk(script['body'])
        if len(found) > 1:
            self.fail(f'{name} returns different numbers of values')
        self.arities[name] = found.pop() if found else 0
        return self.arities[name]

    def expr_arity(self, e):
        if e[0] == 'call' and e[1].startswith('~') and e[1] != '~npc_death':
            return self.arity(f'[proc,{e[1][1:]}]')
        return 1

    def run(self, name, args, discard=False):
        """rs2.run of `name` inlined: its values as temporaries in result registers (_Dead when no path returns).
        discard: the caller drops them (the root, a jump's label, a gosub or a call statement), so any number goes."""
        self.reached.add(name)
        script = self.scripts.get(name)
        if script is None:
            self.unsupported(f'script {name} was not extracted')
        n = 0 if discard else self.arity(name)
        frame = dict(results=[self.alloc() for _ in range(n)], kinds=['int'] * n, ranges=[[] for _ in range(n)], exits=[],
                     locals={}, discard=discard)
        local = dict(zip(script['params'], args))
        try:
            self.block(script['body'], local, frame)
            self.give(frame, [])  # the end of the body returns []
        except _Dead:
            pass
        self.give_back(frame['locals'].values())
        if not frame['exits']:
            self.give_back(frame['results'])
            raise _Dead()
        for at in frame['exits']:
            self.patch(at)
        return [R(r, kind, True, _hull(rngs)) for r, kind, rngs in zip(frame['results'], frame['kinds'], frame['ranges'])]

    def give(self, frame, values):
        """Return(values) from a frame: its values into the result registers, then a jump to the frame's end."""
        flat = []
        for v in values:
            flat += v[1] if v[0] == 'tuple' else [v]
        if frame['discard']:
            self.release_all(flat)
            flat = []
        if len(flat) != len(frame['results']):
            self.fail('a frame returns different numbers of values on its paths')
        for k, v in enumerate(flat):
            if _kind(v) == 'obj':
                frame['kinds'][k] = 'obj'
            frame['ranges'][k].append(_range(v))
            self.into(v, frame['results'][k], frame['kinds'][k])
        frame['exits'].append(self.emit('DOP_JUMP'))

    def local_reg(self, frame, name):
        if name not in frame['locals']:
            frame['locals'][name] = self.alloc()
        return frame['locals'][name]

    # -- statements (rs2._block)
    def block(self, body, local, frame):
        for stmt in body:
            self.statement(stmt, local, frame)

    def statement(self, stmt, local, frame):
        op = stmt[0]
        if op == 'set':
            v = self.single(self.expr(stmt[2], local, frame))
            r = self.local_reg(frame, stmt[1])
            self.into(v, r, _kind(v))
            local[stmt[1]] = R(r, _kind(v), False, _range(v))
        elif op == 'setvarp':
            v = self.single(self.expr(stmt[2], local, frame))
            varps = contract()['RS_VARPS']
            if stmt[1] not in varps or stmt[1] == 'aggressive_npc':
                self.fail(f'a write of varp {stmt[1]}, which rs.h Player does not hold as an int')
            pair = self.reg(v)
            self.emit('DOP_SET_VARP', varps.index(stmt[1]), pair[0])
            self.done_with(pair)
        elif op == 'if':
            self.branch(stmt, local, frame)
        elif op == 'return':
            values = [self.expr(e, local, frame) for e in stmt[1]]
            self.give(frame, values)
            raise _Dead()
        elif op == 'gosub':
            self.release(self.call(f'~{stmt[1]}', [], discard=True))
        elif op == 'jump':
            self.run(f'[label,{stmt[1]}]', [], discard=True)
            self.give(frame, [])
            raise _Dead()
        elif op == 'do':
            self.release(self.expr(stmt[1], local, frame, discard=True))
        else:
            self.unsupported(f"statement '{stmt[1]}'")

    @staticmethod
    def assigned(node):
        """The locals a statement may set, its nested arms included."""
        out = set()
        if isinstance(node, list):
            if node and node[0] == 'set':
                out.add(node[1])
            for child in node:
                out |= DropCompiler.assigned(child)
        return out

    def branch(self, stmt, local, frame):
        """rs2's if/else-if chain: conditions in order until one holds. The locals an arm may set get their frame
        register first (the value they hold copied in), so every arm leaves them where the join reads them."""
        for name in sorted(self.assigned(stmt)):
            r = self.local_reg(frame, name)
            v = local.get(name)
            if v is not None and v[0] != 'unset' and not (v[0] == 'reg' and v[1] == r):
                self.into(R(v[1], v[2], False, v[4]) if v[0] == 'reg' else v, r, _kind(v))
                local[name] = R(r, _kind(v), False, _range(v))
        free, ends, states, scope = list(self.free), [], [], dict(local)
        for cond, body in [*stmt[1], (None, stmt[2])]:
            self.free = list(free)
            skip, env = None, dict(scope)
            if cond is not None:
                narrowed = self.narrowing(cond, scope)
                try:
                    skip, holds = self.condition(cond, scope, frame)
                except _Dead:
                    break
                if skip is None and not holds:  # a constant false condition: the arm never runs
                    continue
                if narrowed is not None:  # the arm knows its condition holds; every later arm, that it failed
                    name, v, op, k = narrowed
                    env[name] = R(*v[1:4], _refine(op, v[4], k, True))
                    scope[name] = R(*v[1:4], _refine(op, v[4], k, False))
            try:
                self.block(body, env, frame)
                states.append(env)
                ends.append(self.emit('DOP_JUMP'))
            except _Dead:
                pass
            if skip is not None:
                self.patch(skip)
            elif cond is not None:  # a constant true condition: no later arm runs
                break
        self.free = list(free)
        for at in ends:
            self.patch(at)
        if not states:
            raise _Dead()
        for name in set().union(*states):
            seen = [env.get(name) for env in states]
            if any(v is None or v[0] == 'unset' for v in seen):
                local[name] = ('unset', name)
            elif len(set(seen)) == 1:
                local[name] = seen[0]
            elif all(v[0] == 'reg' for v in seen) and len({v[:4] for v in seen}) == 1:  # one register, its ranges joined
                local[name] = R(*seen[0][1:4], _hull(v[4] for v in seen))
            else:
                self.fail(f'local {name} holds different values after a branch')

    @staticmethod
    def narrowing(e, local):
        """(local name, its value, op, constant) of a condition `$local op constant` (either side) on a local register
        whose range is known, else None."""
        if e[0] != 'cmp':
            return None
        flip = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '=': '=', '!': '!'}
        for x, y, op in ((e[2], e[3], e[1]), (e[3], e[2], flip[e[1]])):
            if x[0] == 'local' and y[0] == 'num':
                v = local.get(x[1])
                if v is not None and v[0] == 'reg' and v[4] is not None:
                    return x[1], v, op, y[1]
        return None

    def condition(self, e, local, frame):
        """(the jump to patch to the next arm, or None for a constant; whether the constant holds)."""
        if e[0] == 'cmp':
            a = self.single(self.expr(e[2], local, frame))
            b = self.single(self.expr(e[3], local, frame))
            folded = self.fold_cmp(e[1], a, b)
            if folded is not None:
                return None, bool(folded)
            pa, pb = self.reg(a), self.reg(b)
            at = self.emit('DOP_JUMP_IF', 0, pa[0], self.cmp_code(e[1]), pb[0])
            self.done_with(pa, pb)
            return at, False
        v = self.single(self.expr(e, local, frame))
        if v[0] != 'reg':
            return None, self.truth(v)
        pair = self.reg(v)
        at = self.emit('DOP_JUMP_ZERO', 0, pair[0])
        self.done_with(pair)
        return at, False

    @staticmethod
    def cmp_code(op):
        return ('=', '!', '<', '>', '<=', '>=').index(op)

    def truth(self, v):
        """bool() of a constant (rs2's _single then `if`)."""
        if v[0] in ('int', 'str'):
            return bool(v[1])
        self.fail('the truth of a pair, None or an obj id')

    def fold_cmp(self, op, a, b):
        """rs2's cmp where its outcome is known: of constants, or of ranges that decide it (the operands, already
        evaluated, are released); None when the program must compare."""
        if a[0] not in ('int', 'str', 'reg') or b[0] not in ('int', 'str', 'reg'):
            self.fail('a comparison of a pair or None')
        if 'obj' in (_kind(a) if a[0] == 'reg' else '', _kind(b) if b[0] == 'reg' else ''):
            self.fail('a comparison of an obj id')
        # rs2.py evaluates every entry of {'=': a == b, ..., '<': a < b, ...}: a string against a number (a register
        # is a number) raises TypeError whatever the operator
        if 'str' in (a[0], b[0]) and a[0] != b[0]:
            self.fail('a comparison of a string and a number (a TypeError in Python)')
        if a[0] == 'reg' or b[0] == 'reg':
            decided = _decide(op, _range(a), _range(b))
            if decided is None:
                return None
            self.release(a)
            self.release(b)
            return int(decided)
        x, y = a[1], b[1]
        return int({'=': x == y, '!': x != y, '<': x < y, '>': x > y, '<=': x <= y, '>=': x >= y}[op])

    def single(self, v):
        if v[0] == 'tuple':
            self.fail('a pair where one value is expected')
        return v

    # -- expressions (rs2._expr)
    def expr(self, e, local, frame, discard=False):
        op = e[0]
        if op in ('num', 'str'):
            return ('int', e[1]) if op == 'num' else ('str', e[1])
        if op == 'const':
            if e[1] not in self.consts:
                self.unsupported(f'constant ^{e[1]}')
            value = self.consts[e[1]]
            return ('str', value) if isinstance(value, str) else ('int', int(value))
        if op == 'local':
            v = local.get(e[1])
            if v is None:
                self.unsupported(f'unset local {e[1]}')
            if v[0] == 'unset':
                self.fail(f'local {e[1]} is set on some paths only')
            return R(v[1], v[2], False, v[4]) if v[0] == 'reg' else v
        if op == 'varp':
            varps = contract()['RS_VARPS']
            if e[1] == 'aggressive_npc':
                self.fail('a read of varp aggressive_npc (None in Python)')
            r = self.alloc()
            self.emit('DOP_VARP', r, varps.index(e[1]) if e[1] in varps else 0)
            if e[1] not in varps:  # p.varps.get(name, 0): the hero first, then 0
                self.emit('DOP_CONST', r, 0)
            return R(r)
        if op == 'name':
            return self.name(e[1])
        if op in ('and', 'or'):
            return self.logic(e, local, frame)
        if op == 'cmp':
            a = self.single(self.expr(e[2], local, frame))
            b = self.single(self.expr(e[3], local, frame))
            folded = self.fold_cmp(e[1], a, b)
            if folded is not None:
                return ('int', folded)
            pa, pb = self.reg(a), self.reg(b)
            out = self.alloc()
            self.emit('DOP_CONST', out, 0)
            at = self.emit('DOP_JUMP_IF', 0, pa[0], self.cmp_code(e[1]), pb[0])
            self.emit('DOP_CONST', out, 1)
            self.patch(at)
            self.done_with(pa, pb)
            return R(out, rng=(0, 1))
        if op == 'arith':
            a = self.single(self.expr(e[2], local, frame))
            b = self.single(self.expr(e[3], local, frame))
            # rs2.py evaluates every entry of {'+': a + b, '-': a - b, '*': a * b, '/': int(a / b)}: strings raise
            # TypeError, and a zero right side ZeroDivisionError whatever the operator (DOP_ARITH raises it at run time)
            if 'str' in (a[0], b[0]):
                self.fail('arithmetic on a string (a TypeError in Python)')
            if a[0] == 'int' and b[0] == 'int' and b[1] != 0:
                x, y = a[1], b[1]
                return ('int', x + y if e[1] == '+' else x - y if e[1] == '-' else x * y if e[1] == '*' else int(x / y))
            pa, pb = self.reg(a), self.reg(b)
            self.done_with(pa, pb)
            out = self.alloc()
            self.emit('DOP_ARITH', out, pa[0], '+-*/'.index(e[1]), pb[0])
            return R(out)
        if op == 'call':
            args = []
            for a in e[2]:
                v = self.expr(a, local, frame)
                args += v[1] if v[0] == 'tuple' else [v]
            return self.call(e[1], args, discard)
        self.unsupported(f'expression {e}')

    def logic(self, e, local, frame):
        """int(bool(a) and bool(b)) / int(bool(a) or bool(b)): the right side runs only where Python's does."""
        conj = e[0] == 'and'
        a = self.single(self.expr(e[1], local, frame))
        if a[0] != 'reg':
            if self.truth(a) != conj:  # a false left of `and`, a true left of `or`: decided
                return ('int', int(not conj))
            b = self.single(self.expr(e[2], local, frame))
            if b[0] != 'reg':
                return ('int', int(self.truth(b)))
            return self.boolean(b)
        pa = self.reg(a)
        out = self.alloc()
        self.emit('DOP_CONST', out, int(not conj))
        if conj:
            decided = [self.emit('DOP_JUMP_ZERO', 0, pa[0])]
        else:
            left_false = self.emit('DOP_JUMP_ZERO', 0, pa[0])
            decided = [self.emit('DOP_JUMP')]
            self.patch(left_false)
        self.done_with(pa)
        b = self.single(self.expr(e[2], local, frame))
        if b[0] != 'reg':
            self.emit('DOP_CONST', out, int(self.truth(b)))
        else:
            pb = self.reg(b)
            self.emit('DOP_CONST', out, 0)
            at = self.emit('DOP_JUMP_ZERO', 0, pb[0])
            self.emit('DOP_CONST', out, 1)
            self.patch(at)
            self.done_with(pb)
        for at in decided:
            self.patch(at)
        return R(out, rng=(0, 1))

    def boolean(self, v):
        """int(bool(v)) of a register."""
        pair = self.reg(v)
        out = self.alloc()
        self.emit('DOP_CONST', out, 0)
        at = self.emit('DOP_JUMP_ZERO', 0, pair[0])
        self.emit('DOP_CONST', out, 1)
        self.patch(at)
        self.done_with(pair)
        return R(out, rng=(0, 1))

    def name(self, name):
        """DeathContext.name (engine.py:2812-2833)."""
        if name == 'npc_findhero':
            r = self.alloc()
            self.emit('DOP_FINDHERO', r)
            return R(r, rng=(0, 1))
        if name == 'map_members':
            return ('int', int(self.ex.view.members))
        if name == 'null':
            return ('int', -1)
        if name in ('true', 'false'):
            return ('int', int(name == 'true'))
        if name in ('npc_type', 'npc_name'):
            value = self.nt['debugname' if name == 'npc_type' else 'name']
            if not isinstance(value, str):
                self.fail(f'{name} is not a string')
            return ('str', value)
        if name == 'npc_category':
            cat = self.nt['category']
            found = next((k for k, v in self.ex.categories.items() if v == cat), cat)
            return ('str', found) if isinstance(found, str) else ('int', int(found))
        return ('str', name)  # npc_coord, coord, uid, and an obj, inventory, enum, category or type name

    def call(self, name, args, discard=False):
        """rs2._call: a ~proc (npc_death is the simulator's own), calc, or a DeathContext command."""
        if name.startswith('~'):
            if name == '~npc_death':
                if args:
                    self.fail('npc_death with arguments')
                self.emit('DOP_NPC_DEATH')
                return ('none',)
            values = self.run(f'[proc,{name[1:]}]', args, discard)
            self.release_all(args)
            return ('none',) if discard else values[0] if len(values) == 1 else ('tuple', values)
        if name == 'calc':
            if not args:
                self.fail('calc without an argument')
            self.release_all(args[1:])
            return args[0]
        return self.command(name, args)

    @staticmethod
    def inventory(v):
        return ('inv', 'worn', 'bank').index(v[1]) if v[0] == 'str' and v[1] in ('inv', 'worn', 'bank') else None

    def arguments(self, name, args, n):
        if len(args) != n:
            self.fail(f'{name} with {len(args)} arguments (a ValueError in Python)')
        return args

    def command(self, name, args):
        """DeathContext.command (engine.py:2845-2899), its checks in its order."""
        from sim import engine
        if name in ('random', 'randominc'):
            (n,) = self.arguments(name, args, 1)
            pair = self.reg(n)
            self.done_with(pair)
            out = self.alloc()
            self.emit('DOP_RANDOM' if name == 'random' else 'DOP_RANDOMINC', out, pair[0])
            top = None if n[0] != 'int' else n[1] if name == 'randominc' else n[1] - 1
            return R(out, rng=None if top is None or top < 0 else (0, top))  # int(draw * n), 0 <= draw < 1
        if name == 'npc_param':
            (param,) = self.arguments(name, args, 1)
            if param[0] != 'str':
                self.fail('npc_param of a computed name')
            try:
                value = engine.World.npc_param(self.ex.view, SimpleNamespace(type=self.type), param[1])
            except KeyError:
                self.fail(f'npc_param {param[1]} has no value (a KeyError in Python)')
            return ('str', value) if isinstance(value, str) else ('int', int(value))
        if name == 'obj_add':
            coord, obj, count, duration = self.arguments(name, args, 4)
            if coord != ('str', 'npc_coord'):
                self.unsupported('an obj added away from the npc')
            if obj[0] == 'str' and self.ex.t.obj_ids.get(obj[1]) is None:
                self.unsupported(f'obj {obj[1]}')
            if obj[0] == 'int' and obj[1] not in self.ex.t.obj_types and (self.excluded or obj[1] != -1):
                self.fail(f'obj_add of obj {obj[1]}, which indexes no obj type (a KeyError in Python)')
            if obj[0] == 'int' and obj[1] >= 0:
                self.ex.drop_objs.add(obj[1])
            po, pc, pd = self.reg(obj, 'obj'), self.reg(count), self.reg(duration)
            self.emit('DOP_OBJ_ADD', po[0], pc[0], pd[0])
            self.done_with(po, pc, pd)
            return ('int', 0)
        if name == 'inv_total':
            inv, obj = self.arguments(name, args, 2)
            code = self.inventory(inv)
            if code is None:
                if inv[0] not in ('int', 'str'):
                    self.fail('inv_total of a computed inventory')
                self.unsupported(f'inventory {inv[1]}')
            if obj[0] == 'str' and self.ex.t.obj_ids.get(obj[1]) is None:
                self.require_hero()
                self.unsupported(f'obj {obj[1]}')
            po = self.reg(obj, 'obj')
            self.done_with(po)
            out = self.alloc()
            self.emit('DOP_INV_TOTAL', out, code, po[0])
            return R(out)
        if name == 'inv_totalcat':
            inv, category = self.arguments(name, args, 2)
            code = self.inventory(inv)
            if code is None:
                self.fail('inv_totalcat of another inventory (an AttributeError in Python)')
            cat = self.ex.categories.get(category[1]) if category[0] == 'str' else None
            self.release(category)
            out = self.alloc()
            self.emit('DOP_INV_TOTALCAT', out, code, contract()['RS_NONE'] if cat is None else cat)
            return R(out)
        if name == 'if_close':
            self.release_all(args)
            self.emit('DOP_IF_CLOSE')
            return ('int', 0)
        if name == 'p_finduid':
            if not args:
                self.fail('p_finduid without an argument')
            if args[0] != ('str', 'uid'):  # args[0] == 'uid' is false: no player read
                self.release_all(args)
                return ('int', 0)
            out = self.alloc()
            self.emit('DOP_P_FINDUID', out)
            return R(out, rng=(0, 1))
        if name == 'enum':
            _, _, enum, key = self.arguments(name, args, 4)
            table = self.enums.get(enum[1]) if enum[0] == 'str' else None
            if table is None:
                if enum[0] not in ('int', 'str'):
                    self.fail('enum of a computed table')
                self.unsupported(f'enum {enum[1]}')
            values = dict(table['values'])
            if not all(type(k) is int and type(v) is int for k, v in values.items()) or type(table['defaultInt']) is not int:
                self.fail(f'enum {enum[1]} has other than int keys, values and default')
            if key[0] == 'int':
                return ('int', values.get(key[1], table['defaultInt']))
            if key[0] != 'reg' or key[2] != 'int':
                self.fail('an enum key that is not a number')
            sid = self.ex.s(enum[1])
            for k, v in values.items():
                self.ex.enum_rows[(sid, k)] = (v, table['defaultInt'])
                if table.get('outputtype') in ('obj', 'namedobj') and v in self.ex.t.obj_types:
                    self.ex.drop_objs.add(v)
            pk = self.reg(key)
            self.done_with(pk)
            out = self.alloc()
            self.emit('DOP_ENUM', out, sid, pk[0], table['defaultInt'])
            return R(out, rng=_hull([(v, v) for v in values.values()] + [(table['defaultInt'],) * 2]))
        if name == 'enum_getoutputcount':
            (enum,) = self.arguments(name, args, 1)
            table = self.enums.get(enum[1]) if enum[0] == 'str' else None
            if table is None:
                if enum[0] not in ('int', 'str'):
                    self.fail('enum_getoutputcount of a computed table')
                self.unsupported(f'enum {enum[1]}')
            return ('int', len(table['values']))
        if name == 'compare':
            a, b = self.arguments(name, args, 2)
            if a[0] != 'str' or b[0] != 'str':
                self.fail('compare of other than two constant strings')
            return ('int', next((ord(x) - ord(y) for x, y in zip(a[1], b[1]) if x != y), len(a[1]) - len(b[1])))
        if name == 'clearbit_range':
            value, low, high = self.arguments(name, args, 3)
            if all(v[0] == 'int' for v in (value, low, high)):
                return ('int', value[1] & ~(((1 << (high[1] - low[1] + 1)) - 1) << low[1]))
            pairs = [self.reg(v) for v in (value, low, high)]
            self.done_with(*pairs)
            out = self.alloc()
            self.emit('DOP_CLEARBIT_RANGE', out, pairs[0][0], pairs[1][0], pairs[2][0])
            return R(out)
        self.release_all(args)
        self.unsupported(f'command {name}')


# ---- layouts ------------------------------------------------------------------------------------------------------------
class Heaps:
    """The five typed heaps the layout records index (element offsets), kept as chunk lists until written."""
    KINDS = dict(i32=np.int32, i16=np.int16, u16=np.uint16, u8=np.uint8, f64=np.float64)

    def __init__(self):
        self.parts, self.size = {k: [] for k in self.KINDS}, dict.fromkeys(self.KINDS, 0)

    def add(self, kind, values):
        """The offset of `values` appended to heap `kind` (-1 for nothing)."""
        array = np.ascontiguousarray(values)
        if array.size == 0:
            return -1
        cast = array.astype(self.KINDS[kind], copy=False)
        if cast is not array and not np.array_equal(cast, array):
            raise ExportError(f'a layout array does not fit {kind}')
        offset = self.size[kind]
        self.parts[kind].append(cast.ravel())
        self.size[kind] += cast.size
        return offset

    def sections(self):
        names = dict(i32='HEAP_I32', i16='HEAP_I16', u16='HEAP_U16', u8='HEAP_U8', f64='HEAP_F64')
        return {names[k]: parts or [np.zeros(0, self.KINDS[k])] for k, parts in self.parts.items()}


def layout_tables(tables, layout, derived=None):
    """CpuBatch's LayoutTables of a layout (sim/world.py), checked against what the pack can carry."""
    from sim import world
    lt = world.LayoutTables(tables, layout, derived)
    if lt.meta['multiway']:
        raise ExportError(f'layout {layout.seed}: multi-combat zones (rs.h World is single-way)')
    return lt


def layout_record(ex, heaps, kind, index, layout, lt, derived=None, grove=-1, keep=-1, groves=None, sites=None):
    """One RsLayoutRec of a pool layout or variant (rs.h Layouts), from its LayoutTables `lt`."""
    from sim import combat, world
    c = ex.c
    rec = record('RsLayoutRec', 1)
    r = rec[0]
    W, H = layout.flags.shape
    x0, z0 = world.ORIGIN
    if W > c['RS_MAX_W'] or H > c['RS_MAX_W'] or x0 % 8 or z0 % 8:
        raise ExportError(f'layout {layout.seed}: {W}x{H} at {x0},{z0} is outside rs.h Layout')
    local = lambda x, z: (x - x0) * H + (z - z0)
    r['kind'], r['pool_index'], r['W'], r['H'], r['T'], r['x0'], r['z0'] = kind, index, W, H, W * H, x0, z0
    r['grove'], r['keep'], r['seed'], r['attempt'] = grove, keep, layout.seed, layout.attempt
    r['world'] = 1 + list(world.WORLDS).index(world.world_of(layout)['version'])  # the layout built as its pool's class
    sources = [local(x, z) for x, z, _ in lt.meta['sources']]
    r['n_src'], r['n_start'] = len(sources), int(layout.start.sum())
    r['n_locs'], r['n_open'], r['n_npcs'], r['n_fish'] = len(layout.locs), len(layout.open_locs), len(layout.npcs), len(layout.fish)
    if r['n_locs'] > c['RS_MAX_LOCS'] or r['n_npcs'] > c['RS_MAX_NPCS']:
        raise ExportError(f'layout {layout.seed}: more locs or npcs than a World holds')
    zones = {}
    for _, x, z, _, _ in layout.locs:
        zones[(int(x) >> 3, int(z) >> 3)] = zones.get((int(x) >> 3, int(z) >> 3), 0) + 1
    if max(zones.values(), default=0) + len(layout.open_locs) > c['RS_ZONE_CAP']:
        raise ExportError(f'layout {layout.seed}: a zone holds more locs than RS_ZONE_CAP')
    r['gate'] = layout.gate
    r['flags_boot'] = heaps.add('i32', lt.arrays['collision'])
    r['steps_boot'] = heaps.add('u8', lt.steps)
    r['steps_snap'] = heaps.add('u8', lt.snapshot_steps)
    r['sides'] = heaps.add('u8', lt.arrays['sides'])
    r['entity_reach'] = heaps.add('u16', lt.arrays['entity_reach'])
    r['tile_src'] = heaps.add('i16', lt.source_grid)
    r['src_tile'] = heaps.add('i16', sources)
    r['start_tile'] = heaps.add('i16', [int(x) * H + int(z) for x, z in np.argwhere(layout.start)])
    r['locs'] = heaps.add('i32', layout.locs)
    reach = np.full((len(layout.locs), 24), -1, np.int16)
    for k, tiles in enumerate(lt.meta['loc_reach']):
        if len(tiles) > 48:
            raise ExportError(f'layout {layout.seed}: a loc reached from more than 24 tiles')
        reach[k, :len(tiles) // 2] = [local(x, z) for x, z in zip(tiles[0::2], tiles[1::2])]
    r['loc_reach'] = heaps.add('i16', reach)
    r['open_locs'] = heaps.add('i32', layout.open_locs)
    r['npcs'] = heaps.add('i32', layout.npcs)
    spot = lt.npc_types[world._Types(ex.t).spot]
    values = lt.meta['enums'][str(spot['params']['fishing_movement_enum'])]['values']
    r['fish'] = heaps.add('i32', [(x - x0, z - z0) for _, x, z, _ in values])
    for name in ('order0', 'dist0', 'parent0', 'snap_dist'):
        r[name] = -1
    if derived is not None and 'order' in derived:
        r['order0'] = heaps.add('u16', derived['order'][0])
        r['dist0'] = heaps.add('u8', derived['dist'][0])
        r['parent0'] = heaps.add('u8', derived['parent'][0])
        r['snap_dist'] = heaps.add('u8', derived['snapshot_dist'])
    for name in ('camp', 'outside', 'grove_off', 'grove_row', 'grove_normal', 'site_grove', 'site_off', 'site_tile'):
        r[name] = -1
    if kind == c['RS_LAYOUT_COMPACT']:
        camp = combat.camp(layout, ex.t)
        region = [int(x) * H + int(z) for x, z in np.argwhere(combat.outside(layout))]
        r['n_camp'], r['camp'] = len(camp), heaps.add('i16', [x * H + z for x, z in camp])
        r['n_outside'], r['outside'] = len(region), heaps.add('i16', region)
    if groves is not None:
        from sim.search import is_normal
        if max(map(len, groves), default=0) > c['RS_GROVE_MAX']:
            raise ExportError(f'layout {layout.seed}: a grove of more than RS_GROVE_MAX trees')
        r['n_groves'] = len(groves)
        r['grove_off'] = heaps.add('i32', np.cumsum([0] + [len(g) for g in groves]))
        r['grove_row'] = heaps.add('i32', [k for g in groves for k in g])
        r['grove_normal'] = heaps.add('u8', [is_normal(layout, ex.t, k) for g in groves for k in g])
        r['n_sites'] = len(sites)
        r['site_grove'] = heaps.add('i32', [g for g, _ in sites])
        tiles = [[int(x) * H + int(z) for x, z in np.argwhere(found)] for _, found in sites]
        r['site_off'] = heaps.add('i32', np.cumsum([0] + [len(t) for t in tiles]))
        r['site_tile'] = heaps.add('i16', [t for ts in tiles for t in ts])
    return rec


def variant_keys(layout, tables, groves, sites):
    """The scarce variants a supply layout's starts can play: {(grove, keep): a local start tile with that keep}, as
    CpuBatch._supply_tables keys them (search.scarce_rows over every start tile of every site)."""
    from sim.search import scarce_rows
    out = {}
    for g, tiles in sites:
        for x, z in np.argwhere(tiles):
            _, keep = scarce_rows(layout, tables, groves[g], (int(x), int(z)))
            out.setdefault((g, keep), (int(x), int(z)))
    return dict(sorted(out.items()))


def variant(tables, layout, grove, tile):
    """search.variant_layout of a scarce start on (grove, local tile), or None where the oracle raises at the draw."""
    from sim import search, world
    record_ = dict(variant='scarce', grove=grove, tile=(world.ORIGIN[0] + tile[0], world.ORIGIN[1] + tile[1]))
    try:
        return search.variant_layout(layout, tables, record_)
    except RuntimeError:
        return None


# ---- the declared world ---------------------------------------------------------------------------------------------
def reachable(ex, compact, supply, variants=()):
    """(loc, npc, obj) type ids the pools can put in play: the layouts' locs, gate stages and npcs, the stages scripts
    change a loc to (next_loc_stage, next_loc_stage_mining), combat's goblins and muggers, the kits, products and
    drops (ex.drop_objs, complete after drops())."""
    from sim import combat, engine
    from sim.spec import START_KIT, START_WEAPONS
    t, view = ex.t, ex.view
    locs, npcs = set(), set()
    for layout in list(compact) + list(supply) + list(variants):
        locs |= {int(row[0]) for row in layout.locs} | {int(row[0]) for row in layout.open_locs}
        npcs |= {int(row[1]) for row in layout.npcs}
    if compact:
        npcs |= {ex.npc_ids[name] for name in combat.COMBAT['enemies']}
    todo = list(locs)
    while todo:
        k = todo.pop()
        for name, read in (('next_loc_stage', view.loc_param), ('next_loc_stage_mining', view.loc_param_or_default)):
            try:
                stage = read(SimpleNamespace(type=k), name)
            except (KeyError, engine.Unsupported):
                continue
            if isinstance(stage, int) and stage in t.loc_types and stage not in locs:
                locs.add(stage)
                todo.append(stage)
    objs = {t.obj_id(name) for name, _ in START_KIT} | {t.obj_id(name) for name in START_WEAPONS}
    objs |= {t.obj_id(name) for name, _ in combat.COMBAT['tools']} | {t.obj_id(name) for name in combat.COMBAT['food']}
    objs.add(t.obj_id('coins'))
    for k in locs:
        name = t.loc_types[k]['debugname']
        if name in ex.content['trees']:
            objs.add(t.obj_id(ex.content['trees'][name]['product']))
        if name in ex.compact['rocks']:
            objs.add(t.obj_id(ex.compact['rocks'][name]['output']))
        if name in ex.content.get('picks', {}):
            objs.add(t.obj_id(ex.content['picks'][name]['item']))
    objs |= {t.obj_id(gem) for _, gem in ex.compact['gems']}
    objs |= {t.obj_id(name) for name in ex.content['fish']}
    objs |= ex.drop_objs
    return locs, npcs, objs


def declared_world(ex, locs, npcs, objs, loc_rows, npc_rows, obj_rows):
    """Refuse (ExportError) when a reachable type's trigger names a script C does not port (RS_NP), or its death root
    did not compile; and an op list p_oploc / p_opnpc could index past (Python's IndexError)."""
    status = [s for _, _, s in ex.scripts]
    problems = [f'obj type {k}: eats a row that heals more than 4 stats' for k in sorted(objs)
                if ex.t.obj_types[k]['debugname'] in ex.wide_eat]
    for kind, ids, rows in (('loc', locs, loc_rows), ('npc', npcs, npc_rows), ('obj', objs, obj_rows)):
        for k in sorted(ids):
            for j, script in enumerate(rows['trig_script'][k]):
                if status[script] == ex.c['RS_NP']:
                    problems.append(f'{kind} type {k} trigger {j}: {ex.s.text[rows["trig_name"][k, j]]} '
                                    f'({ex.scripts[script][0]} is not ported)')
            if kind != 'obj':
                table = ex.t.loc_types if kind == 'loc' else ex.t.npc_types
                ops = table[k]['op'] or []
                runs = any(s not in (ex.script['RS_S_NONE'], ex.script['RS_S_UNKNOWN']) for s in rows['trig_script'][k][:12])
                if runs and len(ops) != 5:
                    problems.append(f'{kind} type {k}: a script runs on it but its op list has {len(ops)} entries')
    if problems:
        raise ExportError('the pools reach content the native port does not run: ' + '; '.join(problems))


# ---- export ---------------------------------------------------------------------------------------------------------------
def check_village(layout, tables, c):
    """Raise unless the free-play village world that plays this compact layout (sim/combat.py village_of) fits a World
    on it (village_npcs, rs.h rs_village_npcs): the layout's other npcs, a goblin at every chicken spawn, the start
    group's muggers where the world places any, and the chickens, within RS_MAX_NPCS. A layout of no village world is
    not played by free play."""
    from sim import combat
    village = combat.village_of(layout)
    if village is None:
        return
    chickens = len(combat.camp(layout, tables))
    muggers = combat.COMBAT['groups'][village['start_group']]['muggers'] if village.get('muggers') is not False else 0
    count = len(layout.npcs) + chickens + muggers
    if count > c['RS_MAX_NPCS']:
        raise ExportError(f'layout {layout.seed}: {count} village npcs ({len(layout.npcs) - chickens} others, {chickens} '
                          f'chickens and their goblins, {muggers} muggers), more than RS_MAX_NPCS {c["RS_MAX_NPCS"]}')


def export(output, pool, supply_pool, tables_path, names=None, trees=True, variants=None, log=print):
    """Write the pack of a compact pool (with derived tables), a supply pool and the archive to `output`; returns its
    path. trees=False leaves out the derived state-0 trees (C then computes them) and also takes a compact pool
    without derived tables, whose static tables LayoutTables computes as it does for supply layouts; variants: a
    limit on the scarce variants per supply layout (None: all; a partial pack is marked so in its manifest)."""
    from names import name_table
    from sim import combat, engine, search
    names = names or name_table()
    t = load_tables(tables_path)
    rows = sorted((set(engine.ARCHIVE_SCRIPTS) - PORTED_ARCHIVE_SCRIPTS) & set(t.meta['content']))
    if rows:  # until the C port of the phase 2 simulator port, step 5
        raise ExportError(f"{tables_path} holds the {' and '.join(rows)} rows: the oracle runs their scripts, the native port does not")
    ex = Exporter(t, names)
    with np.load(pool) as data, np.load(supply_pool) as other:
        worlds = dict(compact=str(data['version']), supply=str(other['version']))
    derived_keys = DERIVED_STATE + DERIVED_PLAIN if trees else tuple(k for k in DERIVED_STATE + DERIVED_PLAIN if k not in TREES)
    with np.load(pool) as data:
        plain = not trees and not set(derived_keys) & set(data.files)
    compact, derived = load_pool(pool, None if plain else derived_keys)
    if plain:
        derived = [None] * len(compact)
    elif derived is None or any(d is None for d in derived):
        raise ExportError(f'{pool} has no derived tables (sim.world pool --derived)')
    villages = {(combat.village_of(layout) or {}).get('version') for layout in compact}
    if len(villages) > 1:
        raise ExportError(f'{pool} holds the layouts of more than one world')
    village = combat.VILLAGES.get(next(iter(villages), None))
    if village is not None and village.get('content') and village['content'] not in t.meta['content']:
        raise ExportError(f"{pool} holds {village['version']} layouts, whose scripts read the {village['content']} rows "
                          f"of an archive ({village['tables']}, sim/build.py): {tables_path} has none")
    if village is not None and village.get('loot_outside') is not None:
        if not set(village['loot_outside']) <= set(engine.LOOT_OUTSIDE):  # World's ValueError (engine.py:482)
            raise ExportError(f"{village['version']}'s loot list is not within engine.LOOT_OUTSIDE")
        ex.loot_own = tuple(village['loot_outside'])
    for layout in compact:
        check_village(layout, t, ex.c)
    supply, _ = load_pool(supply_pool)
    log(f'[native] {len(compact)} compact and {len(supply)} supply layouts')
    sections = ex.content_rows()
    # the declared world first (a scarce variant only removes trees, so the pools' types are every variant's)
    _, npcs, _ = reachable(ex, compact, supply)
    drops = ex.drops(npcs, {ex.npc_ids[name] for name in combat.COMBAT['enemies']})
    locs, npcs, objs = reachable(ex, compact, supply)
    loc_rows, npc_rows, obj_rows = ex.loc_types(), ex.npc_types(drops), ex.obj_types()
    declared_world(ex, locs, npcs, objs, loc_rows, npc_rows, obj_rows)
    log(f'[native] declared world: {len(locs)} loc, {len(npcs)} npc and {len(objs)} obj types, {len(drops)} death programs')
    heaps, recs = Heaps(), []
    for k, layout in enumerate(compact):
        lt = layout_tables(t, layout, derived[k])
        recs.append(layout_record(ex, heaps, ex.c['RS_LAYOUT_COMPACT'], k, layout, lt, derived[k]))
        derived[k] = None
    keys = {}
    for k, layout in enumerate(supply):
        groves, sites = search.groves(layout, t), search.supply_sites(layout, t)
        lt = layout_tables(t, layout)
        recs.append(layout_record(ex, heaps, ex.c['RS_LAYOUT_SUPPLY'], k, layout, lt, groves=groves, sites=sites))
        found = variant_keys(layout, t, groves, sites)
        keys[k] = list(found.items())[:variants] if variants is not None else list(found.items())
    rows = []
    for k, items in keys.items():
        for (grove, keep), tile in items:
            v = variant(t, supply[k], grove, tile)
            if v is None:
                rows.append((k, grove, keep, ex.c['RS_VARIANT_FAILS']))
                continue
            rows.append((k, grove, keep, len(recs)))
            recs.append(layout_record(ex, heaps, ex.c['RS_LAYOUT_VARIANT'], k, v, layout_tables(t, v), grove=grove, keep=keep))
        if k % 16 == 15:
            log(f'[native] supply layouts 0-{k}: {len(rows)} scarce variant keys')
    sections.update(LOC_TYPES=loc_rows, NPC_TYPES=npc_rows, OBJ_TYPES=obj_rows, CONSTS=ex.consts(),
                    **ex.drop_sections(), LAYOUTS=stack('RsLayoutRec', recs), **heaps.sections())
    table = record('RsVariantKey', len(rows))
    for r, (layout, grove, keep, rec) in enumerate(sorted(rows)):
        table[r]['layout'], table[r]['grove'], table[r]['keep'], table[r]['rec'] = layout, grove, keep, rec
    sections['VARIANTS'] = table
    sections.update(ex.msg.sections())
    sections.update(ex.s.sections(names))  # last: every string is interned by now
    inputs = native.input_hashes(dict(tables=tables_path, pool=pool, supply_pool=supply_pool))
    manifest = native.manifest(  # its exporter (EXPORT_VERSION, this source's sha256) and ids: native.manifest
        inputs=inputs, names=names, meta_sha256=t.meta_sha256, worlds=worlds, content=sorted(t.meta['content']),
        loot_outside=None if ex.loot_own is None else list(ex.loot_own),
        partial=variants is not None, trees=trees,
        counts=dict(compact=len(compact), supply=len(supply), variants=sum(1 for row in rows if row[3] >= 0),
                    failing_variants=sum(1 for row in rows if row[3] < 0), strings=len(ex.s.text),
                    messages=len(ex.msg.rows), drop_roots=len(ex.drop_roots)))
    return write(output, manifest, sections)


def write(output, manifest, sections):
    """A pack of named sections (rs.h RS_SECTIONS names; record sections as their struct arrays)."""
    ids = native.load().constants()['sections']
    structs = dict(CONSTS='RsConsts', LOC_TYPES='RsLocType', NPC_TYPES='RsNpcType', OBJ_TYPES='RsObjType',
                   TREES='RsTreeRow', ROCKS='RsRockRow', GEMS='RsGemRow', FISH='RsFishRow', EAT_TRIGGERS='RsEatTrigger',
                   EAT_ROWS='RsEatRow', WIELD_RULES='RsWieldRule', STYLES='RsStyleTable', LEVELUP='RsLevelupRow',
                   UNLOCKS='RsUnlock', MSGS='RsMsg', MSG_KEYS='RsMsgKeyRow', TOKENS='RsToken', DROP_ROOTS='RsDropRoot',
                   DROP_OPS='RsDropOp', ENUMS='RsEnumRow', LAYOUTS='RsLayoutRec', VARIANTS='RsVariantKey')
    writer = native.PackWriter(output, manifest)
    for name, data in sections.items():
        writer.add(ids[name], data, record=structs.get(name))
    return writer.close()


def main(args):
    """`python -m sim.native export` (sim/native.py's command line)."""
    path = export(args.output, args.pool, args.supply_pool, args.tables, trees=not args.no_trees, variants=args.variants)
    print(f'[native] wrote {path} ({path.stat().st_size} bytes)')
