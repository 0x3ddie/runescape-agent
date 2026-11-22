"""The native exporter (sim/native_export.py) against the oracle it reads: every message's token features give
rl/feedback_contract.py text_vector bit for bit (alone, as a chat of several lines, and as a dialog of parts); the
type rows, name rows and trigger scripts are the oracle's own answers; each compiled death drop program, run with
rs.h's DOP semantics (the reference machine below, the contract combat.c implements), does what rs2.run with
DeathContext does on the same worlds; scarce variants are search.variant_layout's under CpuBatch's keys; the
declared-world check refuses unported content; and a pack of these sections opens in the C reader.

These read the archive and pools in logs/ (docs/RUNNING.md) and skip without them. Names are rows of the conftest's
synthetic name table, never the Jagex-derived one, and a failure reports type ids, message ids and hashes, never the
archive's strings."""
from pathlib import Path
import random
import zlib

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

from sim import native, native_export as nx

LOGS = Path(__file__).resolve().parents[2] / 'logs'
TABLES = LOGS / 'sim' / 'data' / 'draynor-nc1.npz'
POOL = LOGS / 'sim' / 'world' / 'train-8-nc1-derived.npz'
SUPPLY = LOGS / 'supply' / 'pools' / 'train-128.npz'
SLOW = settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(scope='module')
def tables():
    for path in (TABLES,):
        if not path.exists():
            pytest.skip(f'{path} is absent (docs/RUNNING.md says how to build it)')
    return nx.load_tables(TABLES)


def _digest(value):
    """A failure's evidence without the archive's strings."""
    import hashlib
    return hashlib.sha256(repr(value).encode()).hexdigest()[:12]


def same(got, expected, what):
    if got != expected:
        raise AssertionError(f'{what} differs (got {_digest(got)}, expected {_digest(expected)})')


def equal(got, expected, what):
    if not np.array_equal(got, expected):
        raise AssertionError(f'{what} differs')


@pytest.fixture(scope='module')
def ex(tables, name_table_path):
    """An exporter with every non-layout section built (content rows, the reachable death programs, types), its
    names-v1 rows from the conftest's synthetic table."""
    from names import NameTable
    from sim import combat
    from sim.world import _Types
    ex = nx.Exporter(tables, NameTable(name_table_path))
    ex.sections = ex.content_rows()
    types = _Types(tables)
    reach = {types.spot, *types.chickens, *(ex.npc_ids[name] for name in combat.COMBAT['enemies'])}
    ex.drop_types = ex.drops(reach, {ex.npc_ids[name] for name in combat.COMBAT['enemies']})
    ex.sections.update(LOC_TYPES=ex.loc_types(), NPC_TYPES=ex.npc_types(ex.drop_types), OBJ_TYPES=ex.obj_types(),
                       CONSTS=ex.consts(), **ex.drop_sections())
    ex.sections.update(ex.msg.sections())
    ex.sections.update(ex.s.sections(ex.names))
    return ex


# ---- messages ----------------------------------------------------------------------------------------------------------
def _normalized(counts):
    """C's text_vector from bucket counts (rs.h FLOATS): float counts, sqrtf of the integer sum of squares, a float32
    divide."""
    v = np.asarray(counts, np.float32)
    sumsq = int(np.sum(np.asarray(counts, np.int64) ** 2))
    norm = np.sqrt(np.float32(sumsq))
    return v / norm if sumsq else v


def _crc32_combine(crc1, crc2, len2):
    """zlib's crc32_combine (the pair hash C builds from per-token CRCs), by appending len2 zero bytes to crc1's
    stream: crc32(a + b) = crc32(b, crc32(a)) = combine(crc32(a), crc32(b), len(b))."""
    zero = zlib.crc32(bytes(len2), crc1) ^ zlib.crc32(bytes(len2))
    return zero ^ crc2


def _dialog_counts(ex, msgs, dim=64):
    """C's dialog features: the parts' tokens concatenated in order, pairs across parts, pair hashes combined."""
    sec = ex.sections
    toks = [int(t) for m in msgs for t in sec['MSG_TOKENS'][sec['MSGS'][m]['tok0']:sec['MSGS'][m]['tok0'] + sec['MSGS'][m]['n_tok']]]
    rows = sec['TOKENS']
    counts = np.zeros(dim, np.int64)
    hashes = [int(rows[t]['crc']) for t in toks]
    hashes += [_crc32_combine(int(rows[a]['crc_sp']), int(rows[b]['crc']), int(rows[b]['len'])) for a, b in zip(toks, toks[1:])]
    for h in hashes:
        counts[h % dim] += 1 if (h >> 16) & 1 else -1
    return counts


def test_every_message_gives_text_vector(ex):
    from feedback_contract import text_vector, words
    sec = ex.sections
    text = lambda m: ex.s.text[sec['MSGS'][m]['text']]
    assert len(sec['MSGS']) > 1000
    for m, row in enumerate(sec['MSGS']):
        t = text(m)
        assert _normalized(row['chat']).tobytes() == text_vector([t], 64).tobytes(), m
        assert np.array_equal(_dialog_counts(ex, [m]), row['chat']), m
        toks = sec['MSG_TOKENS'][row['tok0']:row['tok0'] + row['n_tok']]
        tokens = words(t)
        assert [ex.sections['TOKENS'][k]['len'] for k in toks] == [len(w) for w in tokens]
        assert [ex.sections['TOKENS'][k]['crc'] for k in toks] == [zlib.crc32(w.encode()) for w in tokens]
        assert row['nonblank'] == bool(t.strip())


@settings(deadline=None, max_examples=300)
@given(st.data())
def test_chat_and_dialog_combinations_give_text_vector(ex, data):
    """A chat of several lines sums per-line counts; a dialog joins its nonblank parts with spaces, pairs crossing."""
    from feedback_contract import text_vector
    sec = ex.sections
    msgs = data.draw(st.lists(st.integers(0, len(sec['MSGS']) - 1), max_size=6))
    texts = [ex.s.text[sec['MSGS'][m]['text']] for m in msgs]
    chat = sum((sec['MSGS'][m]['chat'].astype(np.int64) for m in msgs), np.zeros(64, np.int64))
    assert _normalized(chat).tobytes() == text_vector(texts, 64).tobytes()
    parts = [m for m in msgs if sec['MSGS'][m]['nonblank']]
    joined = ' '.join(ex.s.text[sec['MSGS'][m]['text']].strip() for m in parts)
    assert _normalized(_dialog_counts(ex, parts)).tobytes() == text_vector([joined], 64).tobytes()


def test_message_keys_are_world_text(ex, tables):
    """Each (key, fill) row is world.text's string for that fill; the literal kinds are the engine's lines."""
    from sim.engine import World
    c = nx.contract()
    sec, view = ex.sections, nx._View(tables)
    keys = c['RS_TEXT_KEYS']
    fills = dict(wc_level='level', mine_level='level', wc_get='product', eat='food', mine_gem='gem', mine_full='ore',
                 mine_get='ore', prospect_ore='ore')
    rocks, bad = sorted(ex.compact['rocks']), []
    for row in sec['MSG_KEYS']:
        key, fill, msg = int(row['key']), int(row['fill']), int(row['msg'])
        text = ex.s.text[sec['MSGS'][msg]['text']] if msg >= 0 else None
        if key >= c['RS_MK_NO_TRIGGER_OPLOCU']:
            continue
        name = keys[key - c['RS_MK_TEXT']]
        field = fills.get(name)
        if field is None:
            value = {}
        elif field == 'level':
            value = dict(level=fill)
        elif field == 'ore':
            value = dict(ore=ex.compact['rocks'][rocks[fill]]['ore'])
        elif field == 'gem':
            from sim.engine import add_article
            gem = tables.obj_types[fill]['debugname']
            value = dict(gem=add_article(tables.obj_types[tables.obj_id(ex.compact['cut_gems'][gem])]['name']))
        else:
            value = {field: tables.obj_types[fill]['name'].lower()}
        if text != World.text(view, name, **value):
            bad.append((key, fill))
    assert not bad, f'{len(bad)} (key, fill) rows are not world.text: {bad[:5]}'
    lines = {(int(r['key']), int(r['fill'])): ex.s.text[sec['MSGS'][r['msg']]['text']] for r in sec['MSG_KEYS']}
    for kind, types in ((c['RS_MK_NO_TRIGGER_OPLOCU'], tables.loc_types), (c['RS_MK_NO_TRIGGER_OPNPCU'], tables.npc_types)):
        prefix = 'oplocu' if types is tables.loc_types else 'opnpcu'
        bad = [k for k, t in types.items() if lines[(kind, k)] != f"No trigger for [{prefix},{t['debugname']}]"]
        assert not bad, f'no-trigger lines of types {bad[:5]}'
    assert lines[(c['RS_MK_CANT_REACH'], 0)] == "I can't reach that!"


# ---- types and names ---------------------------------------------------------------------------------------------------
def test_type_rows_and_name_rows_are_the_oracles(ex, tables):
    """Names, names-v1 rows (the projection's ident(kind, name)), ops and their states, and the oracle-derived flags
    of every type; string ranks are Python's sorted() order."""
    from act_contract import BANK_NAME, BANK_OPTION
    from sim import engine
    from sim.agent import TOOL_PROTECT_RE, name_of
    sec, c, strings = ex.sections, nx.contract(), ex.s.text
    rows = sec['STR_ROW'].reshape(3, -1)
    assert rows.shape[1] == len(strings) and (rows > 0).any()  # the synthetic table names a few of them
    bad = [(kind, k) for kind in ('loc', 'npc', 'obj') for k, s in enumerate(strings)
           if rows[c[f'RS_KIND_{kind.upper()}'], k] != ex.names.row(kind, s)]
    assert not bad, f'{len(bad)} names-v1 rows differ: {bad[:5]}'
    for kind, types, key in (('loc', tables.loc_types, 'LOC_TYPES'), ('npc', tables.npc_types, 'NPC_TYPES'),
                             ('obj', tables.obj_types, 'OBJ_TYPES')):
        rec, bad = sec[key], []
        for k, t in types.items():
            r = rec[k]
            ops = (t['iop'] if kind == 'obj' else t['op']) or []
            prefix = 'i' if kind == 'obj' else ''
            visible = [o for o in ops if o and o != 'hidden']
            states = [c['RS_OPS_NONE'] if not o else c['RS_OPS_HIDDEN'] if o == 'hidden' else c['RS_OPS_VISIBLE']
                      for o in list(ops) + [None] * (5 - len(ops))]
            if not (r['present'] and strings[r['name_of']] == name_of(t) and strings[r['debugname']] == t['debugname']
                    and strings[sec['STR_LOWER'][r['name_of']]] == name_of(t).lower() == strings[r['name_of_lc']]
                    and strings[r['name']] == (t['name'] or '')
                    and [strings[r[f'{prefix}op'][j]] for j in r[f'{prefix}opt_index'][:r[f'n_{prefix}opt']]] == visible
                    and list(r[f'{prefix}op_state']) == states):
                bad.append(k)
            if kind != 'obj' and r['bank_listed'] != bool(BANK_NAME.search(name_of(t))
                                                          and any(BANK_OPTION.search(o) for o in visible)):
                bad.append(k)
        assert not bad, f'{kind} types {bad[:5]} differ'
        assert not rec['present'][[k for k in range(len(rec)) if k not in types]].any()
    bad = []
    for k, t in tables.obj_types.items():
        r = sec['OBJ_TYPES'][k]
        if (bool(r['flags'] & c['RS_OF_TOOL_PROTECT']) != bool(TOOL_PROTECT_RE.search(name_of(t)))
                or bool(r['flags'] & c['RS_OF_LOOT_OUTSIDE']) != engine.loot_outside(tables, k)
                or bool(r['flags'] & c['RS_OF_COINS']) != (name_of(t).lower() == 'coins')
                or r['take_op'] != next((i for i, o in enumerate(t['op'] or []) if o and 'take' in o.lower()), 2)
                or ex.compact['styles'][sorted(ex.compact['styles'])[r['style_table']]] is not engine.weapon_styles(ex.view, k)):
            bad.append(k)
    assert not bad, f'obj types {bad[:5]} differ'
    bad = []
    for k in tables.npc_types:
        r = sec['NPC_TYPES'][k]
        try:
            if r['attackable'] != engine.npc_is_attackable(ex.view, None, engine.Npc(0, k, 0, 0, 1, type=k)):
                bad.append(k)
        except engine.Unsupported:
            if not r['refuse_attack']:
                bad.append(k)
        if r['max_dealt'] != engine.World.npc_param(ex.view, engine.Npc(0, k, 0, 0, 1, type=k), 'max_dealt'):
            bad.append(k)
    assert not bad, f'npc types {bad[:5]} differ'
    order = sorted(range(len(strings)), key=strings.__getitem__)
    assert [int(sec['STR_RANK'][k]) for k in order] == list(range(len(strings)))
    from act_contract import name_id
    assert [int(v) for v in sec['STR_NAME_ID']] == [name_id(s) for s in strings]


def test_triggers_name_the_oracles_scripts(ex, tables):
    """Every trigger slot names the script the oracle's dispatch would run: the RsScript's qualname is the function's,
    closures bound as the id says, and an absent or unknown script is none or RS_S_UNKNOWN as the dicts have it."""
    import inspect
    from sim import engine
    sec, scripts, c = ex.sections, ex.scripts, nx.contract()
    name = lambda r, j: ex.s.text[r['trig_name'][j]]
    checked = 0
    for k, t in tables.loc_types.items():
        r = sec['LOC_TYPES'][k]
        for j in range(c['RS_LTRIG_N']):
            trig = engine._trigger_name(tables, engine.APLOC1 + j % 6 + (engine.AP_TO_OP if j >= 6 else 0))
            script = t['triggers'].get(trig)
            sid = int(r['trig_script'][j])
            if not script:
                assert scripts[sid][0] == 'RS_S_NONE'
                continue
            same(name(r, j), script, f'loc type {k} trigger {j} name')
            fn = engine.SCRIPTS.get(script)
            if fn is None:
                assert scripts[sid][0] == 'RS_S_UNKNOWN'
                continue
            assert scripts[sid][1] == fn.__qualname__, (k, j)
            if fn.__qualname__ == 'mining.<locals>.script':
                cells = inspect.getclosurevars(fn).nonlocals
                assert ('FIRST' in scripts[sid][0]) == cells['first'], (k, j)
            checked += 1
    for k, t in tables.obj_types.items():
        r = sec['OBJ_TYPES'][k]
        for op in range(5):
            script = t['triggers'].get(f'OPHELD{op + 1}')
            sid = int(r['trig_script'][c['RS_OTRIG_HELD'] + op])
            fn = engine.held_script(nx._View(tables), script) if script else None
            assert scripts[sid][0] == ('RS_S_NONE' if not script else 'RS_S_UNKNOWN' if fn is None else scripts[sid][0]), k
            if fn is not None:
                assert scripts[sid][1] == fn.__qualname__, (k, op)
                checked += 1
    assert checked > 3000


# ---- death drop programs -------------------------------------------------------------------------------------------------
class _Deaths:
    """The world a death script touches (DeathContext's reads and npc_death's), recording what it does."""

    def __init__(self, tables, seed, tick, exclusions, player):
        from sim.engine import LOOT_OUTSIDE, Rng
        self.t, self.rng, self.tick, self.exclusions = tables, Rng(seed), tick, frozenset(exclusions)
        self.loot_outside = LOOT_OUTSIDE  # World.loot_outside of every world the C port plays
        self.members = bool(tables.meta['environment']['members'])
        self.player, self.events, self.calls = player, [], []

    def npc_param(self, npc, name):
        from sim.engine import World
        return World.npc_param(self, npc, name)

    def obj_add(self, p, level, x, z, obj, count, duration):
        self.calls.append(('obj_add', p is self.player, level, x, z, obj, count, duration))

    def npc_set_mode(self, npc, mode, target=None):
        self.calls.append(('mode', mode))

    def remove_npc(self, npc):
        self.calls.append(('remove',))


def run_program(ex, row, world, npc):
    """rs.h's drop program semantics (DOP_*), as a generator of npc_death's suspensions like npc_death_script."""
    from sim import engine
    c = nx.contract()
    D = {k: v for k, v in c.items() if k.startswith('DOP_')}
    D = {v: k for k, v in D.items()}
    root = ex.sections['DROP_ROOTS'][row]
    ops = ex.drop_ops[root['first']:root['first'] + root['n_ops']]
    debugname = world.t.npc_types[npc.type]['debugname']
    enums = {(table, key): value for (table, key), (value, _) in ex.enum_rows.items()}
    r, pc, draws, active = [0] * c['RS_DROP_REGS'], 0, 0, None
    inventories = ('inv', 'worn', 'bank')

    def player():
        if active is None:
            raise engine.Unsupported(f'death script of {debugname}: a player command before npc_findhero')
        return active
    while True:
        op, a, b, cc, d = ops[pc]
        pc += 1
        name = D[op]
        if name == 'DOP_END':
            return
        if name == 'DOP_UNSUPPORTED':
            raise engine.Unsupported(f'death script of {debugname}: {ex.s.text[a]}')
        if name == 'DOP_NPC_DEATH':
            yield from engine.npc_death(world, npc)
        elif name == 'DOP_CONST':
            r[a] = b
        elif name == 'DOP_MOVE':
            r[a] = r[b]
        elif name == 'DOP_ARITH':
            x, y = r[b], r[d]
            if y == 0:  # rs2.py divides whatever the operator
                raise ZeroDivisionError('division by zero')
            r[a] = (x + y, x - y, x * y, int(x / y))[cc]
        elif name == 'DOP_JUMP':
            pc = a
        elif name == 'DOP_JUMP_IF':
            x, y = r[b], r[d]
            if not (x == y, x != y, x < y, x > y, x <= y, x >= y)[cc]:
                pc = a
        elif name == 'DOP_JUMP_ZERO':
            if r[b] == 0:
                pc = a
        elif name == 'DOP_FINDHERO':
            hero = max(npc.heroes.items(), key=lambda kv: kv[1], default=(None, 0))[0]
            active = world.player if world.player is not None and hero == world.player.name else None
            r[a] = int(active is not None)
        elif name in ('DOP_RANDOM', 'DOP_RANDOMINC'):
            draws += 1
            n = r[b] + (1 if name == 'DOP_RANDOMINC' else 0)
            r[a] = int(world.rng(engine.R_NPC_SCRIPT, world.tick, npc.nid, 1000 + draws) * n)
        elif name == 'DOP_OBJ_ADD':
            obj = r[a]
            if 'clue_drops' in world.exclusions and world.t.obj_types[obj]['category'] in engine.trail_categories(world.t):
                world.events.append((world.tick, 'excluded', dict(what='clue_drop', obj=obj, npc=npc.nid)))
            elif 'loot_drops' in world.exclusions and engine.loot_outside(world.t, obj):
                world.events.append((world.tick, 'excluded', dict(what='loot_drop', obj=obj, npc=npc.nid)))
            else:
                world.obj_add(player(), npc.level, npc.x, npc.z, obj, r[b], r[cc])
        elif name == 'DOP_INV_TOTAL':
            r[a] = player().inventory(inventories[b]).total(r[cc])
        elif name == 'DOP_INV_TOTALCAT':
            items = player().inventory(inventories[b]).listed()
            r[a] = 0 if cc == c['RS_NONE'] else sum(n for _, o, n in items if world.t.obj_types[o]['category'] == cc)
        elif name == 'DOP_IF_CLOSE':
            player().close_modal()
        elif name == 'DOP_P_FINDUID':
            r[a] = int(player().can_access())
        elif name == 'DOP_ENUM':
            r[a] = enums.get((b, r[cc]), d)
        elif name == 'DOP_VARP':
            r[a] = player().varps[c['RS_VARPS'][b]]
        elif name == 'DOP_SET_VARP':
            player().varps[c['RS_VARPS'][a]] = r[b]
        elif name == 'DOP_CLEARBIT_RANGE':
            r[a] = r[b] & ~(((1 << (r[d] - r[cc] + 1)) - 1) << r[cc])
        else:
            raise AssertionError(f'{name} is not emitted by the compiler')


def _death_params(ex, tables, data):
    """A random world for a death script: the hero, the player's inventories, varps and access, exclusions, the tick."""
    pool = sorted(ex.drop_objs | {k for k, t in tables.obj_types.items() if t['category'] in
                                  {op[3] for op in ex.drop_ops if op[0] == nx.contract()['DOP_INV_TOTALCAT']}})
    items = [[(data.draw(st.sampled_from(pool)), data.draw(st.integers(1, 3))) for _ in range(data.draw(st.integers(0, 6)))]
             for _ in range(3)]
    tick = data.draw(st.integers(10, 10 ** 6))
    return dict(items=items, trail=data.draw(st.integers(0, (1 << 20) - 1)), protect=data.draw(st.booleans()),
                delayed=data.draw(st.booleans()), chat=data.draw(st.booleans()), tick=tick,
                seed=data.draw(st.integers(0, 2 ** 32 - 1)),
                exclusions=data.draw(st.sampled_from([(), ('clue_drops', 'loot_drops'), ('clue_drops',), ('loot_drops',)])),
                player=data.draw(st.integers(0, 9)) > 0, nid=data.draw(st.integers(0, 15)),
                aggressive=data.draw(st.sampled_from([nx.PLAYER, None])),
                hero=data.draw(st.sampled_from([nx.PLAYER, 'someone', None])), damage=data.draw(st.integers(1, 9)),
                moved=tick - data.draw(st.integers(-1, 3)))


def _death_world(tables, npc_type, q):
    from sim import engine
    p = engine.Player(None, nx.PLAYER)
    for inv, items in zip((p.inv, p.worn, p.bank), q['items']):
        for obj, count in items:
            inv.add(tables, obj, count)
    p.varps['trail_status'], p.protect, p.delayed = q['trail'], q['protect'], q['delayed']
    p.modal_state = {'chat'} if q['chat'] else set()
    world = _Deaths(tables, q['seed'], q['tick'], q['exclusions'], p if q['player'] else None)
    npc = engine.Npc(q['nid'], npc_type, 2900, 3340, 1, type=npc_type)
    npc.vars = dict(engine.NPC_VARS, aggressive_player=q['aggressive'])
    npc.heroes = {} if q['hero'] is None else {q['hero']: q['damage']}
    npc.last_movement = q['moved']
    return world, npc


def _drive(gen, world, schedule):
    """Every suspension of a death script, the ticks moving on between them as the schedule says; its end or raise."""
    steps = []
    try:
        for k, step in enumerate(gen):
            steps.append(step)
            world.tick += schedule[k % len(schedule)]
    except Exception as error:  # the oracle's raise (Unsupported, or a KeyError) is compared, class and text
        return steps, (type(error).__name__, str(error))
    return steps, None


@SLOW
@given(st.data())
def test_drop_programs_do_what_the_death_scripts_do(ex, data):
    """Each compiled program, on the reference machine, against npc_death_script on an identical world: the same
    suspensions, drops (obj, count, duration, where), excluded events, varps, closed modals and raises."""
    from sim import engine
    tables = ex.t  # never a test argument: Hypothesis would print the archive with a failing example
    npc_type = data.draw(st.sampled_from(sorted(ex.drop_types)))
    q = _death_params(ex, tables, data)
    world, npc = _death_world(tables, npc_type, q)
    twin_world, twin_npc = _death_world(tables, npc_type, q)
    schedule = data.draw(st.lists(st.integers(1, 3), min_size=1, max_size=4))
    name = tables.npc_types[npc_type]['triggers']['AI_QUEUE3']
    expected = _drive(engine.npc_death_script(world, npc, name), world, schedule)
    got = _drive(run_program(ex, ex.drop_types[npc_type], twin_world, twin_npc), twin_world, schedule)
    same(got, expected, f'npc type {npc_type}: suspensions and raise')
    same((twin_world.calls, twin_world.events), (world.calls, world.events), f'npc type {npc_type}: drops and events')
    p, q = world.player, twin_world.player
    if p is not None:
        same((q.varps, q.protect, q.modal_state, q.inv.items, q.worn.items),
             (p.varps, p.protect, p.modal_state, p.inv.items, p.worn.items), f'npc type {npc_type}: the player')


def test_drop_programs_reach_their_drops(ex, tables):
    """A sample of worlds where the hero is the player and nothing is excluded drops something from every program,
    so the equivalence above covers the drop tables, not only their refusals."""
    from sim import engine
    rng = random.Random(5)
    for npc_type, row in ex.drop_types.items():
        if not tables.npc_types[npc_type]['op'] or not any(o and 'attack' in o.lower() for o in tables.npc_types[npc_type]['op']):
            continue
        drops = 0
        for k in range(40):
            p = engine.Player(None, nx.PLAYER)
            world = _Deaths(tables, rng.randrange(2 ** 32), rng.randrange(10, 10 ** 6), (), p)
            npc = engine.Npc(k % 16, npc_type, 2900, 3340, 1, type=npc_type)
            npc.vars, npc.heroes, npc.last_movement = dict(engine.NPC_VARS), {nx.PLAYER: 3}, 0
            steps, error = _drive(run_program(ex, row, world, npc), world, [1])
            drops += sum(call[0] == 'obj_add' for call in world.calls)
        assert drops >= 40, npc_type


# ---- layouts -------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def compact():
    """The first two layouts of the compact pool with the derived arrays the pack reads (state 0)."""
    layouts, derived = nx.load_pool(POOL, nx.DERIVED_STATE + nx.DERIVED_PLAIN)
    return layouts[:2], derived[:2]


class _Quiet(list):
    """Pool layouts whose repr (in a failing example) holds none of their data."""
    def __repr__(self):
        return f'<{len(self)} pool layouts>'


@pytest.fixture(scope='module')
def supply():
    for path in (SUPPLY,):
        if not path.exists():
            pytest.skip(f'{path} is absent (docs/RUNNING.md says how to build it)')
    layouts, _ = nx.load_pool(SUPPLY)
    return _Quiet(layouts)


def _heap(heaps, kind, offset, n):
    data = np.concatenate(heaps.parts[kind]) if heaps.parts[kind] else np.zeros(0)
    return data[offset:offset + n]


def test_load_pool_is_the_oracles(supply, compact):
    """The exporter's lazy pool reader gives sim/world.py load_pool's layouts (and the derived rows it cuts)."""
    from sim import world
    layouts, _ = world.load_pool(SUPPLY)
    for a, b in zip(layouts, supply):
        for name, value in a.arrays().items():
            equal(value, b.arrays()[name], name)
    k = 1
    with np.load(POOL) as data:
        n = int(data['n_loc_reach'][k])
        equal(compact[1][k]['loc_reach'], data['loc_reach'][k][:1, :n], 'loc_reach rows')
        equal(compact[1][k]['order'], data['order'][k][:1], 'order rows')


def test_compact_records_answer_the_oracles_queries(ex, tables, compact):
    """A compact record read as C reads it gives what LayoutTables answers CpuBatch: sources, reach, stored trees and
    walk steps, start tiles in draw_start's order, the fishing enum, the combat camp and region."""
    from sim import combat, paths, world
    from sim.engine import World
    c = nx.contract()
    rng = random.Random(3)
    for k, (layout, derived) in enumerate(zip(*compact)):
        heaps = nx.Heaps()
        lt = nx.layout_tables(tables, layout, derived)
        rec = nx.layout_record(ex, heaps, c['RS_LAYOUT_COMPACT'], k, layout, lt, derived)[0]
        W, H, T = int(rec['W']), int(rec['H']), int(rec['T'])
        x0, z0 = world.ORIGIN
        get = lambda kind, field, n: _heap(heaps, kind, int(rec[field]), n)
        tile_src, src_tile = get('i16', 'tile_src', T), get('i16', 'src_tile', rec['n_src'])
        reach, steps, entity = get('u16', 'entity_reach', T), get('u8', 'steps_boot', T), get('i16', 'loc_reach', rec['n_locs'] * 24)
        order, dist = get('u16', 'order0', T * T), get('u8', 'dist0', T * T)
        snap = get('u8', 'snap_dist', T * T)
        tiles = [(x, z) for x in range(W) for z in range(H)]
        for x, z in rng.sample(tiles, 300):
            t = x * H + z
            same(int(tile_src[t]), lt.source(0, x0 + x, z0 + z), f'layout {k} source of a tile')
            if tile_src[t] >= 0:
                same(int(src_tile[tile_src[t]]), t, f'layout {k} tile of a source')
            try:
                found = paths.entity_reaching(lt, 0, x0 + x, z0 + z)
            except paths.OutsideTables:
                found = None
            if found is not None:
                same(sorted(found), sorted((x0 + x + dx, z0 + z + dz) for b, (dx, dz) in enumerate(paths.NEIGHBOURS)
                                           if reach[t] >> b & 1), f'layout {k} entity reach')
            same(int(steps[t]), int(lt.steps[x, z]), f'layout {k} steps')
        w = World(lt, seed=1)
        for i, loc in enumerate(w.locs):
            row = [v for v in entity[i * 24:(i + 1) * 24] if v >= 0]
            same(sorted(w.loc_reaching(loc)), sorted((x0 + v // H, z0 + v % H) for v in row), f'layout {k} loc {i} reach')
        for src in rng.sample(range(int(rec['n_src'])), 12):
            t = int(src_tile[src])
            o, d, p = lt.rows(src)
            equal(order[t * T:(t + 1) * T].reshape(W, H), o, 'a stored tree order')
            equal(dist[t * T:(t + 1) * T].reshape(W, H), d, 'a stored tree dist')
            sx, sz = x0 + t // H, z0 + t % H
            for x, z in rng.sample(tiles, 20):
                expect = paths.walk_steps(lt, src, sx, sz, x0 + x, z0 + z)
                same(int(snap[t * T + x * H + z]) if snap[t * T + x * H + z] != 255 else -1, expect, f'layout {k} walk steps')
        same(list(get('i16', 'start_tile', rec['n_start'])), [int(x) * H + int(z) for x, z in np.argwhere(layout.start)],
             f'layout {k} start tiles')
        enum = lt.meta['enums'][str(lt.npc_types[world._Types(tables).spot]['params']['fishing_movement_enum'])]['values']
        same(list(get('i32', 'fish', rec['n_fish'] * 2)), [v for _, x, z, _ in enum for v in (x - x0, z - z0)], f'layout {k} fish')
        same(list(get('i16', 'camp', rec['n_camp'])), [x * H + z for x, z in combat.camp(layout, tables)], f'layout {k} camp')
        same(list(get('i16', 'outside', rec['n_outside'])), [int(x) * H + int(z) for x, z in np.argwhere(combat.outside(layout))],
             f'layout {k} combat region')


def test_supply_records_hold_groves_and_sites(ex, tables, supply):
    """Groves and sites as draw_supply and SupplyTracker read them: grove rows in search.groves order, each site's
    start tiles in np.argwhere order."""
    from sim import search
    c = nx.contract()
    for k in (0, 7):
        layout = supply[k]
        heaps = nx.Heaps()
        groves, sites = search.groves(layout, tables), search.supply_sites(layout, tables)
        rec = nx.layout_record(ex, heaps, c['RS_LAYOUT_SUPPLY'], k, layout, nx.layout_tables(tables, layout),
                               groves=groves, sites=sites)[0]
        H = int(rec['H'])
        get = lambda kind, field, n: list(_heap(heaps, kind, int(rec[field]), n))
        off = get('i32', 'grove_off', rec['n_groves'] + 1)
        rows = get('i32', 'grove_row', off[-1])
        same([rows[off[g]:off[g + 1]] for g in range(rec['n_groves'])], groves, f'layout {k} groves')
        same(get('u8', 'grove_normal', off[-1]), [search.is_normal(layout, tables, r) for g in groves for r in g], 'normal')
        same(get('i32', 'site_grove', rec['n_sites']), [g for g, _ in sites], f'layout {k} site groves')
        soff = get('i32', 'site_off', rec['n_sites'] + 1)
        tiles = get('i16', 'site_tile', soff[-1])
        for s, (_, found) in enumerate(sites):
            same(tiles[soff[s]:soff[s + 1]], [int(x) * H + int(z) for x, z in np.argwhere(found)], f'layout {k} site {s}')


@settings(deadline=None, max_examples=8)
@given(st.integers(0, 2 ** 40), st.integers(0, 127), st.integers(0, 50))
def test_scarce_variants_are_variant_layouts_under_cpubatchs_keys(ex, supply, key, slot, episode):
    """A keyed scarce supply start (search.draw_supply) maps to the key CpuBatch._supply_tables builds
    ((layout, grove, keep) by scarce_rows); the pack's variant under that key is search.variant_layout's layout as
    LayoutTables carries it, or a failing row where the oracle raises."""
    from sim import search, world
    c, tables = nx.contract(), ex.t
    sites = {k: search.supply_sites(layout, tables) for k, layout in enumerate(supply)}
    start = search.draw_supply(key, slot, episode, supply, sites, variant='scarce')
    layout = supply[start['layout']]
    groves = search.groves(layout, tables)
    tile = (start['tile'][0] - world.ORIGIN[0], start['tile'][1] - world.ORIGIN[1])
    _, keep = search.scarce_rows(layout, tables, groves[start['grove']], tile)
    keys = nx.variant_keys(layout, tables, groves, sites[start['layout']])
    assert (start['grove'], keep) in keys
    found = nx.variant(tables, layout, start['grove'], keys[(start['grove'], keep)])
    try:
        expected = search.variant_layout(layout, tables, start)
    except RuntimeError:
        assert found is None
        return
    heaps = nx.Heaps()
    rec = nx.layout_record(ex, heaps, c['RS_LAYOUT_VARIANT'], start['layout'], found, nx.layout_tables(tables, found),
                           grove=start['grove'], keep=keep)[0]
    lt = world.LayoutTables(tables, expected)
    T = int(rec['T'])
    get = lambda kind, field, n: _heap(heaps, kind, int(rec[field]), n)
    equal(get('i32', 'flags_boot', T), lt.arrays['collision'], 'boot flags')
    equal(get('u8', 'steps_boot', T), lt.steps.ravel(), 'boot steps')
    equal(get('u8', 'steps_snap', T), lt.snapshot_steps.ravel(), 'snapshot steps')
    equal(get('u8', 'sides', T), lt.arrays['sides'], 'sides')
    equal(get('u16', 'entity_reach', T), lt.arrays['entity_reach'], 'entity reach')
    equal(get('i16', 'tile_src', T), lt.source_grid, 'sources')
    equal(get('i32', 'locs', rec['n_locs'] * 5), expected.locs.ravel(), 'locs')
    same((int(rec['grove']), int(rec['keep']), int(rec['pool_index']), int(rec['seed'])),
         (start['grove'], keep, start['layout'], layout.seed), 'the variant key')


def test_declared_world_refuses_unported_content(ex, tables, compact):
    """A pool whose layout holds a loc or an npc whose trigger names a script C does not port (a bank booth, stairs, a
    bank teller) is refused; the pools as they are pass."""
    import dataclasses
    sec, c = ex.sections, nx.contract()
    layouts = compact[0]
    status = [s for _, _, s in ex.scripts]
    locs, npcs, objs = nx.reachable(ex, layouts, [])
    nx.declared_world(ex, locs, npcs, objs, sec['LOC_TYPES'], sec['NPC_TYPES'], sec['OBJ_TYPES'])
    np_loc = next(k for k in tables.loc_types if any(status[s] == c['RS_NP'] for s in sec['LOC_TYPES']['trig_script'][k]))
    np_npc = next(k for k in tables.npc_types if any(status[s] == c['RS_NP'] for s in sec['NPC_TYPES']['trig_script'][k]))
    locs_ = layouts[0].locs.copy()
    locs_[0, 0] = np_loc
    npcs_ = layouts[0].npcs.copy()
    npcs_[-1, 1] = np_npc
    for bad in (dataclasses.replace(layouts[0], locs=locs_), dataclasses.replace(layouts[0], npcs=npcs_)):
        locs, npcs, objs = nx.reachable(ex, [bad], [])
        with pytest.raises(nx.ExportError, match='does not run'):
            nx.declared_world(ex, locs, npcs, objs, sec['LOC_TYPES'], sec['NPC_TYPES'], sec['OBJ_TYPES'])


def test_an_archive_with_the_firemaking_and_cooking_rows_is_refused(tmp_path):
    """Over draynor-v3 the oracle runs the firemaking and cooking scripts (engine.ARCHIVE_SCRIPTS), which C does not
    port yet: no pack is written from it."""
    for path in (POOL, SUPPLY, TABLES.with_name('draynor-v3.npz')):
        if not path.exists():
            pytest.skip(f'{path} is absent (docs/RUNNING.md says how to build it)')
    with pytest.raises(nx.ExportError, match='cooking and firemaking'):
        nx.export(tmp_path / 'v3.pack', POOL, SUPPLY, TABLES.with_name('draynor-v3.npz'))
    assert not (tmp_path / 'v3.pack').exists()


def test_a_plain_pool_exports_as_its_derived_pool_without_trees(tables, supply, name_table_path, tmp_path):
    """Without trees, a compact pool with no derived tables gives the pack of the same layouts' derived pool in every
    section: LayoutTables computes the static tables the derived pool stores. Only the pool's input hash differs."""
    from names import NameTable
    from sim import world
    layouts = [world.generate(seed, tables) for seed in world.split_seeds('train', 3)]
    plain = world.save_pool(tmp_path / 'plain.npz', layouts)
    derived = world.save_pool(tmp_path / 'derived.npz', layouts, [world.derive(layout, tables) for layout in layouts])
    with np.load(SUPPLY) as data:
        small = world.save_pool(tmp_path / 'supply.npz', supply[:1], version=str(data['version']))
    names = NameTable(name_table_path)
    digests = [native.pack_digest(nx.export(tmp_path / f'{pool.stem}.pack', pool, small, TABLES, names=names, trees=False,
                                            variants=2, log=lambda _: None)) for pool in (plain, derived)]
    found = native.compare_digests(*digests)
    assert found['sections'] == [] and found['sections_compared'] == 34
    assert found['manifest_keys'] == ['inputs']
    a, b = (d['manifest']['inputs'] for d in digests)
    assert a.pop('pool') != b.pop('pool') and a == b
    assert not digests[0]['manifest']['trees'] and digests[0]['manifest']['counts']['compact'] == 3


def test_a_layout_whose_village_holds_more_npcs_than_a_world_is_refused(tables, supply, name_table_path, tmp_path):
    """Free play places the layout's npcs, a goblin at every chicken spawn, the start group's mugger and the chickens
    (sim/combat.py village_npcs): a compact-world-v1 layout whose village would hold more than RS_MAX_NPCS is refused
    at export (the C engine would stop at its first life there), and the same layout less its extra chickens is not."""
    from dataclasses import replace
    from names import NameTable
    from sim import combat, world
    layout = world.generate(world.split_seeds('train', 1)[0], tables)
    kinds = {k for k, v in tables.npc_types.items() if v['debugname'] in ('chicken', 'chicken_brown')}
    chickens = [k for k, row in enumerate(layout.npcs) if int(row[1]) in kinds]
    assert len(chickens) == len(combat.camp(layout, tables)) >= 1
    village = len(layout.npcs) + len(chickens) + 1
    extra = (native.load().constants()['RS_MAX_NPCS'] - village) // 2 + 1  # each chicken adds itself and a goblin
    full = replace(layout, npcs=np.concatenate([layout.npcs, layout.npcs[[chickens[0]] * extra]]))
    with np.load(SUPPLY) as data:
        small = world.save_pool(tmp_path / 'supply.npz', supply[:1], version=str(data['version']))
    names = NameTable(name_table_path)
    with pytest.raises(nx.ExportError, match='village npcs'):
        nx.export(tmp_path / 'full.pack', world.save_pool(tmp_path / 'full.npz', [full]), small, TABLES, names=names,
                  trees=False, variants=1, log=lambda _: None)
    assert not (tmp_path / 'full.pack').exists()
    nx.check_village(layout, tables, native.load().constants())


def test_a_pack_of_these_sections_opens_in_c(ex, tables, compact, supply, tmp_path):
    """The sections, with one compact layout, one supply layout and its first variants, pass the C reader's checks
    (bounds, record layouts, sorted keys, variant rows) and its declared-world guard, and read back as written."""
    from sim import search
    c, sec = nx.contract(), dict(ex.sections)
    heaps, recs, rows = nx.Heaps(), [], []
    layout, derived = compact[0][0], compact[1][0]
    recs.append(nx.layout_record(ex, heaps, c['RS_LAYOUT_COMPACT'], 0, layout, nx.layout_tables(tables, layout, derived),
                                 derived))
    s = supply[0]
    groves, sites = search.groves(s, tables), search.supply_sites(s, tables)
    recs.append(nx.layout_record(ex, heaps, c['RS_LAYOUT_SUPPLY'], 0, s, nx.layout_tables(tables, s), groves=groves,
                                 sites=sites))
    for (grove, keep), tile in list(nx.variant_keys(s, tables, groves, sites).items())[:2]:
        v = nx.variant(tables, s, grove, tile)
        rows.append((0, grove, keep, c['RS_VARIANT_FAILS'] if v is None else len(recs)))
        if v is not None:
            recs.append(nx.layout_record(ex, heaps, c['RS_LAYOUT_VARIANT'], 0, v, nx.layout_tables(tables, v),
                                         grove=grove, keep=keep))
    table = nx.record('RsVariantKey', len(rows))
    for r, row in enumerate(sorted(rows)):
        table[r] = row
    sec.update(LAYOUTS=nx.stack('RsLayoutRec', recs), VARIANTS=table, **heaps.sections())
    path = nx.write(tmp_path / 'part.pack', native.manifest(names=ex.names, partial=True), sec)
    pack = native.open_pack(path, names=ex.names)
    pack.check_declared()
    _, meta, read = native.read_pack(path)
    ids = native.load().constants()['sections']
    assert meta['partial'] and pack.n_strings == len(ex.s.text)
    same(bytes(read[ids['STRINGS']][3]), sec['STRINGS'], 'the string blob')
    for name, struct in (('MSGS', 'RsMsg'), ('OBJ_TYPES', 'RsObjType'), ('LAYOUTS', 'RsLayoutRec'), ('DROP_OPS', 'RsDropOp')):
        same(np.frombuffer(pack.section(ids[name]), native.record_dtype(struct)).tobytes(), sec[name].tobytes(), name)
    equal(np.frombuffer(pack.section(ids['HEAP_U16']), np.uint16), np.concatenate(heaps.parts['u16']), 'the u16 heap')


# ---- village-world-v2 ------------------------------------------------------------------------------------------------
TABLES_V2 = TABLES.with_name('draynor-nc2.npz')
POOL_V2 = LOGS / 'sim' / 'world' / 'train-128-20000-v2.npz'


@pytest.fixture(scope='module')
def v2_pack(name_table_path, tmp_path_factory):
    """(pool file, pack path, decoded sections, manifest) of the v2 training pool's first 8 layouts over draynor-nc2."""
    if not (TABLES_V2.exists() and POOL_V2.exists() and SUPPLY.exists()):
        pytest.skip('private v2 archive, v2 pool or supply pool absent')
    from names import NameTable
    from sim import search, world
    work = tmp_path_factory.mktemp('v2-export')
    pool = world.save_pool(work / 'pool.npz', world.load_pool(POOL_V2)[0][:8])
    supply = world.save_pool(work / 'supply.npz', world.load_pool(SUPPLY)[0][:1], version=search.SUPPLY['world'])
    path = nx.export(work / 'v2.pack', pool, supply, TABLES_V2, names=NameTable(name_table_path), trees=False, variants=1,
                     log=lambda _: None)
    _, meta, sections = native.read_pack(path)
    names = {v: k for k, v in native.load().constants()['sections'].items()}
    decoded = {names[sid]: value for sid, (_, _, _, value) in sections.items()}
    return pool, path, decoded, meta


def _records(decoded, name, struct):
    return np.frombuffer(decoded[name] if isinstance(decoded[name], bytes) else decoded[name].tobytes(),
                         native.record_dtype(struct))


def test_a_village_world_v2_pack_names_its_world_and_holds_its_content(v2_pack):
    """Over draynor-nc2 the v2 pool's pack: each compact record names village-world-v2 and the supply record
    search-world-v1 (world_of); the plants' Pick is RS_S_PICK with each plant's picks row; the field items' and the
    cape's OPHELDU are their default use scripts, whose cases set their own flags only; the loot flag of the world's own
    list (LOOT_OUTSIDE less the cabbage) beside the default one; the manifest names the pools' worlds, the archive's
    content and that list."""
    from sim import combat, engine, world
    pool, path, decoded, meta = v2_pack
    c, t = nx.contract(), nx.load_tables(TABLES_V2)
    scripts = {name: k for k, (name, _, _) in enumerate(native.load().scripts())}
    records = _records(decoded, 'LAYOUTS', 'RsLayoutRec')
    ids = {name: 1 + k for k, name in enumerate(world.WORLDS)}
    assert {int(r['world']) for r in records if r['kind'] == c['RS_LAYOUT_COMPACT']} == {ids['village-world-v2']}
    assert {int(r['world']) for r in records if r['kind'] != c['RS_LAYOUT_COMPACT']} == {ids['search-world-v1']}
    locs = _records(decoded, 'LOC_TYPES', 'RsLocType')
    picks = t.meta['content']['picks']
    for k, lt in t.loc_types.items():
        row = picks.get(lt['debugname'])
        assert bool(locs[k]['has_pick']) == (row is not None), k
        if row is not None:
            assert (int(locs[k]['pick_item']), int(locs[k]['pick_respawn']), bool(locs[k]['pick_scaled'])) == \
                (t.obj_id(row['item']), row['respawn'], row['scaled'])
            assert int(locs[k]['trig_script'][c['RS_TRIG_OP'] + 1]) == scripts['RS_S_PICK']   # OPLOC2
    objs = _records(decoded, 'OBJ_TYPES', 'RsObjType')
    heldu = c['RS_OTRIG_HELDU']
    for name, script in (('cabbage', 'RS_S_USE_DEFAULT_CABBAGE'), ('potato', 'RS_S_USE_DEFAULT_POTATO'),
                         ('onion', 'RS_S_USE_DEFAULT_ONION'), ('black_cape', 'RS_S_USE_DEFAULT_BLACK_CAPE')):
        assert int(objs[t.obj_id(name)]['trig_script'][heldu]) == scripts[script], name
    for key, flag in nx.FIELD_USE_FLAGS.items():
        cases = set(engine.FIELD_USE[key])
        on = {t.obj_types[k]['debugname'] for k in t.obj_types if int(objs[k]['flags']) & c[f'RS_OF_{flag}']}
        assert on == cases & {v['debugname'] for v in t.obj_types.values()}, key
    own = combat.VILLAGE_V2['loot_outside']
    for k in t.obj_types:
        flags = int(objs[k]['flags'])
        assert bool(flags & c['RS_OF_LOOT_OUTSIDE_OWN']) == engine.loot_outside(t, k, own), k
        assert bool(flags & c['RS_OF_LOOT_OUTSIDE']) == engine.loot_outside(t, k), k
    cabbage = int(objs[t.obj_id('cabbage')]['flags'])
    assert cabbage & c['RS_OF_LOOT_OUTSIDE'] and not cabbage & c['RS_OF_LOOT_OUTSIDE_OWN']
    assert meta['worlds'] == dict(compact='village-world-v2', supply='supply-world-v1')
    assert meta['content'] == sorted(t.meta['content']) and 'picks' in meta['content']
    assert meta['loot_outside'] == list(own)
    consts = _records(decoded, 'CONSTS', 'RsConsts')[0]
    assert int(consts['s_produce_pick']) != c['RS_NONE']


def test_a_v1_pack_holds_none_of_village_world_v2(ex, tables):
    """Over draynor-nc1 nothing of village-world-v2 enters a pack: the plants' Pick stays unknown (Python raises
    'engine script [oploc2,cabbage]' there too), the field items' and the cape's OPHELDU stay unknown, no case flag of
    theirs or loot flag of an own list is set, the pick's produce string is not interned."""
    c = nx.contract()
    scripts = {name: k for k, (name, _, _) in enumerate(native.load().scripts())}
    locs, objs = ex.sections['LOC_TYPES'], ex.sections['OBJ_TYPES']
    for k, lt in tables.loc_types.items():
        if lt['debugname'] in ('cabbage', 'potato', 'onion'):
            assert int(locs['trig_script'][k, c['RS_TRIG_OP'] + 1]) == scripts['RS_S_UNKNOWN'] and not locs['has_pick'][k]
    for name in ('cabbage', 'potato', 'onion', 'black_cape'):
        assert int(objs['trig_script'][tables.obj_id(name), c['RS_OTRIG_HELDU']]) == scripts['RS_S_UNKNOWN'], name
    bits = sum(c[f'RS_OF_{flag}'] for flag in (*nx.FIELD_USE_FLAGS.values(), 'LOOT_OUTSIDE_OWN'))
    assert not np.any(objs['flags'] & bits)
    assert int(ex.sections['CONSTS'][0]['s_produce_pick']) == c['RS_NONE']
    assert '[proc,pickup_loc_floor]' not in ex.s.text


def test_a_village_world_v2_pool_over_an_archive_without_its_picks_is_refused(tmp_path):
    """The v2 pool's scripts read the pick rows: over draynor-nc1 (which CpuBatch refuses for free play) no pack is
    written; over draynor-v3 the firemaking and cooking rows are still refused first."""
    if not POOL_V2.exists():
        pytest.skip('private v2 pool absent')
    from sim import world
    pool = world.save_pool(tmp_path / 'pool.npz', world.load_pool(POOL_V2)[0][:2])
    with pytest.raises(nx.ExportError, match='picks rows'):
        nx.export(tmp_path / 'v2.pack', pool, SUPPLY, TABLES, trees=False, variants=1, log=lambda _: None)
    assert not (tmp_path / 'v2.pack').exists()


def test_the_use_default_closures_name_one_id_in_the_export_and_the_harness(ex):
    """Every use_default closure the oracle runs (engine.HELD's and the pick archive's FIELD_USE) maps to the same rs.h
    id in the exporter and in the harness's frame names, each its own."""
    import inspect
    from sim import engine, native_harness as H
    closures = {k: fn for k, fn in {**engine.HELD, **engine.ARCHIVE_SCRIPTS['picks']}.items()
                if fn.__qualname__ == 'use_default.<locals>.script'}
    assert len(closures) == 7
    names = {k: name for name, k in ex.script.items()}
    exported = {key: names[ex.script_id(fn)] for key, fn in closures.items()}
    harness = {key: H.CLOSURES['use_default.<locals>.script'](inspect.getclosurevars(fn).nonlocals)
               for key, fn in closures.items()}
    assert exported == harness and len(set(exported.values())) == 7
    assert exported == {key: nx.USE_DEFAULT_IDS[key] for key in closures}


def test_check_village_bounds_a_village_world_v2_layout(tables):
    """check_village counts a v2 layout's village npcs as village_npcs would (the layout's npcs, the highwayman among
    them, a goblin at each chicken spawn, no muggers): the training pool's layouts fit RS_MAX_NPCS; one with more rows
    is refused."""
    import dataclasses
    if not POOL_V2.exists():
        pytest.skip('private v2 pool absent')
    from sim import combat, world
    c = nx.contract()
    layouts = world.load_pool(POOL_V2)[0]
    counts = [len(combat.village_npcs(layout, tables, [])) for layout in layouts]
    for layout in layouts:
        nx.check_village(layout, tables, c)
    assert max(counts) <= c['RS_MAX_NPCS'] and max(counts) >= 12, (min(counts), max(counts))
    big = max(layouts, key=lambda layout: len(combat.village_npcs(layout, tables, [])))
    spot = next(row for row in big.npcs if tables.npc_types[int(row[1])]['debugname'] == world.SPOT)  # not a chicken
    grown = lambda k: dataclasses.replace(big, npcs=np.concatenate([big.npcs, [spot] * k]))
    room = c['RS_MAX_NPCS'] - len(combat.village_npcs(big, tables, []))
    nx.check_village(grown(room), tables, c)   # exactly RS_MAX_NPCS village npcs, no mugger among them, fit
    assert len(combat.village_npcs(grown(room), tables, [])) == c['RS_MAX_NPCS']
    with pytest.raises(nx.ExportError, match='village npcs'):
        nx.check_village(grown(room + 1), tables, c)
