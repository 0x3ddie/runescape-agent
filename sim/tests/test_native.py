"""The native simulator's foundation (sim/native.py, sim/native/rs.h): the extension builds from this source set; its
keyed RNG and numeric helpers equal the oracle's (sim/engine.py Rng and _js_round, Python's //, % and >>); the tables
it compiles agree with the Python they mirror (observation and info layouts, reasons, scripts, the log1p features);
and a pack round-trips through PackWriter and the C reader, which refuses damaged, stale and mis-laid-out packs and
packs whose name rows or inputs are not the ones this process would read."""
import json
from string import Formatter

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from sim import native
from sim.engine import Rng, _js_round
from sim.spec import OBSERVATION_SHAPES, check_batch, dtype

INT64 = st.integers(-2 ** 63, 2 ** 63 - 1)


@pytest.fixture(scope='module')
def module():
    return native.load()


@settings(deadline=None, max_examples=300)
@given(INT64, st.lists(INT64, max_size=8), st.integers(0, 8), st.integers(-10 ** 9, 10 ** 9))
def test_keyed_draws_equal_the_oracle(seed, keys, split, n):
    m = native.load()
    rng, split = Rng(seed), min(split, len(keys))
    h = m.rng_seed(seed)
    assert h == rng.seed
    assert m.rng_prefix(h, keys) == rng.prefix(*keys)
    assert m.rng_draw(m.rng_prefix(h, keys[:split]), keys[split:]) == rng(*keys)
    draw = Rng.draw(rng.seed, *keys)
    assert m.rng_m(h, keys) / 16777216.0 == draw
    assert m.rng_pick(m.rng_m(h, keys), n) == int(draw * n)


@settings(deadline=None, max_examples=300)
@given(st.floats(-1e9, 1e9, allow_nan=False) | st.integers(-10 ** 6, 10 ** 6).map(lambda k: k + 0.5),
       INT64, INT64.filter(lambda b: b != 0), st.integers(0, 70))
def test_numeric_helpers_equal_python(v, a, b, s):
    m = native.load()
    assert m.js_round(v) == _js_round(v)
    if not (a == -2 ** 63 and b == -1):
        assert m.floordiv(a, b) == a // b
        assert m.mod(a, b) == a % b
    assert m.asr(a, s) == a >> s


XP = st.integers(0, 2 * 10 ** 9)  # a stat's XP in tenths, 0 to 200,000,000 XP


@st.composite
def level_gains(draw):
    """The terms free-play-v1's pay sums in one decision (sim/task.py FreePlayEpisode._account) or its report over a
    life: one per skill of S in S's order, the change of fractional_level over a stat's XP gain (none, a decision's, a
    life's), with S's own length."""
    from sim.audit import fractional_level
    from sim.task import FREE_PLAY
    terms = []
    for _ in FREE_PLAY['skills']:
        before = draw(XP)
        gain = draw(st.sampled_from([0, 0, 0]) | st.integers(1, 5000) | st.integers(1, 2 * 10 ** 9))
        after = min(before + gain, 2 * 10 ** 9)
        terms.append(fractional_level(after) - fractional_level(before))
    return terms


def bits(value):
    return 'nan' if value != value else np.float64(value).tobytes()


@settings(deadline=None, max_examples=2000)
@given(level_gains() | st.lists(st.floats(), max_size=40) | st.lists(st.floats(-1e3, 1e3), min_size=3, max_size=12))
def test_the_c_sum_of_floats_is_the_oracles_sum(terms):
    """sim/native/pysum.h rs_py_sum (module py_sum) equals Python's own sum() of the same floats bit for bit: on the
    level gains free-play-v1's pay sums, and on any floats (infinities and NaN among them: NaN as NaN)."""
    assert bits(native.load().py_sum(terms)) == bits(float(sum(terms))), terms


def test_the_loader_refuses_an_interpreter_whose_sum_is_plain(monkeypatch):
    """The C twin sums as CPython 3.12+ sums floats (compensated): check_interpreter refuses a plain sum(), and load()
    asks it before it builds or loads anything; this interpreter passes."""
    import builtins

    def plain(values):
        total = 0
        for value in values:
            total += value
        return total
    native.check_interpreter()
    with pytest.raises(RuntimeError, match='adds floats in sum.. plainly'):
        native.check_interpreter(plain)
    monkeypatch.setattr(native, '_MODULE', None)
    monkeypatch.setattr(builtins, 'sum', plain)
    with pytest.raises(RuntimeError, match='Python 3.12 or later'):
        native.load()


def test_the_observation_and_info_layouts_are_the_spec(module):
    """The layouts C writes (rs_gen.h, through the compiled module) are sim/spec.py's, in sim/workers.py's order."""
    from sim.spec import INFO_FIELDS, OBSERVATION_SHAPES, dtype
    from sim.vec import EPISODE_FACTS
    c = module.constants()
    assert [row[0] for row in c['obs']] == list(OBSERVATION_SHAPES)
    for name, form, ndim, shape, elems, offset, nbytes in c['obs']:
        assert form == np.dtype(dtype(name)).char and shape[:ndim] == OBSERVATION_SHAPES[name]
        assert elems * np.dtype(dtype(name)).itemsize == nbytes and offset % np.dtype(dtype(name)).itemsize == 0
    assert c['RS_OBS_BYTES'] == sum(row[-1] for row in c['obs']) == 3528  # obs-v5 (obs-v3: 3520, obs-v2: 3960)
    assert c['info'] == [(name, np.dtype(kind).char) for name, kind in INFO_FIELDS.items()]
    assert c['RS_N_FACTS'] == len(EPISODE_FACTS)


def test_every_reason_names_its_fields_and_a_python_class(module):
    exceptions = module.exceptions()
    classes = native._exceptions()
    for k, (name, exc, form, kinds) in enumerate(module.reasons()):
        fields = [field for _, field, _, _ in Formatter().parse(form) if field is not None]
        assert fields == [str(i) for i in range(len(kinds))], name
        assert set(kinds) <= set('isc') and kinds.count('c') <= 1, name
        assert (exceptions[exc] == 'EXC_NONE') == (k == 0), name
        assert k == 0 or exceptions[exc] in classes, name


def test_errors_render_as_the_oracle_raises_them(module):
    """A C error becomes the oracle's exception class and text (engine.py:1642, :943; data.py:138)."""
    from sim.engine import Unsupported
    from sim.paths import OutsideTables
    reason = {name: k for k, (name, *_) in enumerate(module.reasons())}

    class Strings:
        def string(self, k):
            return {7: 'loc_1551'}[k]
    code = lambda exc, r, args, text=None: module.Error(1, module.exceptions().index(exc), reason[r],
                                                        tuple(args) + (0,) * (6 - len(args)), text, 0)
    error = native.as_python(code('EXC_UNSUPPORTED', 'RS_E_PLAYER_LEFT', (0, 2901, 3355)))
    assert type(error) is Unsupported and str(error) == 'player left the supported area at 0,2901,3355'
    error = native.as_python(code('EXC_UNSUPPORTED', 'RS_E_LOC_PARAM', (7,), 'next_loc_stage'), Strings())
    assert str(error) == 'param next_loc_stage of loc loc_1551'
    error = native.as_python(code('EXC_KEYERROR', 'RS_E_LOS_OUTSIDE', (0, 2890, 3340, 2905, 3341)))
    assert type(error) is KeyError and error.args == ('line of sight 0,2890,3340 -> 2905,3341 outside the tables',)
    error = native.as_python(code('EXC_OUTSIDE', 'RS_E_NO_PATH_TABLE', (0, 3094, 3106)))
    assert type(error) is OutsideTables and str(error) == 'no path table for 0,3094,3106'


def test_scripts_name_oracle_functions(module):
    """Every script id the port runs names a function of the oracle, so the harness can map frames to ids."""
    from sim import agent, engine
    rows = module.scripts()
    assert rows[0][0] == 'RS_S_NONE' and len({name for name, _, _ in rows}) == len(rows)
    for name, qualname, status in rows[2:]:
        outer = qualname.split('.')[0]
        assert status in (1, 2, 3) and (hasattr(engine, outer) or hasattr(agent, outer)), name


def _strings(*texts):
    blob, offsets = b'', []
    for text in texts:
        offsets.append(len(blob))
        blob += text.encode() + b'\0'
    return blob, np.asarray(offsets, np.uint32)


class Names:
    """A stand-in names-v1 table: only its spec matters to a pack (rl/names.py NameTable.spec)."""
    def __init__(self, digest):
        self.spec = (11, 4, digest)


def _pack(path, rows=False, inputs=None, user=None, **manifest):
    sec = native.load().constants()['sections']
    names = Names('d' * 16) if rows else None
    writer = native.PackWriter(path, {**native.manifest(inputs=inputs or dict(synthetic='test'), names=names),
                                      **manifest})
    blob, offsets = _strings('', 'tree', 'Chop down')
    writer.add(sec['STRINGS'], blob)
    writer.add(sec['STR_OFF'], offsets)
    if rows:
        writer.add(sec['STR_ROW'], np.zeros(3 * len(offsets), np.int32))
    writer.add(sec['USER'], np.arange(-5, 6, dtype=np.int16) if user is None else user)
    writer.add(sec['USER'] + 1, np.linspace(-1.0, 1.0, 7))
    cells = np.zeros(3, native.record_dtype('RsVariantKey'))   # three failing variant keys, sorted
    cells['layout'], cells['grove'], cells['keep'], cells['rec'] = (1, 2, 3), (-4, 5, -6), (7, 0, 2), -1
    writer.add(sec['VARIANTS'], cells, record='RsVariantKey')
    writer.add(sec['USER'] + 2, b'odd length')
    return writer.close(), cells


def test_a_pack_round_trips(tmp_path, module):
    path, cells = _pack(tmp_path / 'synthetic.pack', counts=dict(layouts=0))
    sec = module.constants()['sections']
    pack = native.open_pack(path)
    header, meta, sections = native.read_pack(path)
    assert json.loads(pack.manifest) == meta and meta['counts'] == dict(layouts=0)
    assert meta['oracle_hash'] == native.oracle_hash() and meta['layout_hash'] == native.layout_hash()
    assert header['file_size'] == path.stat().st_size and header['file_size'] % 64 == 0
    assert [pack.string(k) for k in range(4)] == ['', 'tree', 'Chop down', ''] and pack.n_strings == 3
    table = {sid: (dtype, count, elem, offset) for sid, dtype, count, elem, _, offset in pack.sections()}
    assert all(offset % 64 == 0 for *_, offset in table.values())
    assert np.array_equal(np.frombuffer(pack.section(sec['USER']), np.int16), np.arange(-5, 6))
    assert np.array_equal(sections[sec['USER'] + 1][3], np.linspace(-1.0, 1.0, 7))
    assert np.array_equal(np.frombuffer(pack.section(sec['VARIANTS']), native.record_dtype('RsVariantKey')), cells)
    assert pack.section(sec['USER'] + 2) == b'odd length' and table[sec['VARIANTS']][2] == 16


def test_two_exports_compare_by_file_manifest_and_section(tmp_path, module):
    """sim.native info --digest --against sets another host's export against this Mac's pack by pack_digest: equal
    packs compare equal, and a pack with one other section and another count names only those."""
    same, _ = _pack(tmp_path / 'a.pack', counts=dict(layouts=0))
    again, _ = _pack(tmp_path / 'b.pack', counts=dict(layouts=0))
    assert native.compare_digests(native.pack_digest(same), native.pack_digest(again)) == dict(
        file_equal=True, manifest_keys=[], sections=[], sections_compared=6)
    other, _ = _pack(tmp_path / 'c.pack', user=np.arange(5, -6, -1, dtype=np.int16), counts=dict(layouts=1))
    compared = native.compare_digests(native.pack_digest(other), native.pack_digest(same))
    assert compared == dict(file_equal=False, manifest_keys=['counts'], sections=['USER'], sections_compared=6)


def test_the_engine_comparison_finds_the_first_step_that_differs():
    """sim.native env's verdict (native_bench.compare_engines): exact when the per-step ticks, the episode ends and the
    outputs' hashes agree; otherwise the first differing step, or only the outputs, make it inexact."""
    from sim.native_bench import compare_engines

    def row(engine, ticks, ends, outputs='h'):
        return dict(case='v2', engine=engine, worlds=8, decisions_per_world=len(ticks), key=11, seed=1, ticks=sum(ticks),
                    episodes=sum(ends), tick_sha256=str((ticks, ends)), outputs_sha256=outputs, ticks_by_step=ticks,
                    ends_by_step=ends)
    same = compare_engines([row('cpu', [10, 12, 9], [0, 1, 0]), row('native', [10, 12, 9], [0, 1, 0])])
    assert [(r['exact'], r['first_step'], r['outputs_equal']) for r in same] == [(True, None, True)]
    later = compare_engines([row('cpu', [10, 12, 9], [0, 1, 0]), row('native', [10, 12, 9], [0, 0, 1])])
    assert [(r['exact'], r['first_step']) for r in later] == [(False, 1)]
    hashed = compare_engines([row('cpu', [10, 12], [0, 1]), row('native', [10, 12], [0, 1], outputs='other')])
    assert [(r['exact'], r['first_step'], r['outputs_equal']) for r in hashed] == [(False, None, False)]
    assert compare_engines([row('native', [1], [0])]) == []


def test_the_build_keeps_the_gated_flags_on_macos_and_tags_its_library_by_its_flags(monkeypatch):
    """macOS builds with the flags its release gate passed with; SIM_NATIVE_LTO=0 (a toolchain without the LTO plugin)
    drops -flto under another library; an interpreter whose LDSHARED names a compiler not installed here builds with cc."""
    if native.DARWIN:
        assert native.build_flags() == native.FLAGS + native.LINK and native.LIBS == ('-lz', '-lpthread')
    else:
        assert '-lm' in native.LIBS and '-Werror=implicit-function-declaration' in native.build_flags()
    header = native.generated_header()
    with_lto = native.library(header)[0]
    monkeypatch.setenv('SIM_NATIVE_LTO', '0')
    assert '-flto' not in native.build_flags() and native.library(header)[0] != with_lto
    monkeypatch.setattr(native.sysconfig, 'get_config_var', lambda name: 'no-such-compiler-here -shared -O2')
    assert native.linker()[0] == 'cc'


def test_damaged_stale_or_mislaid_packs_are_refused(tmp_path, module):
    path, _ = _pack(tmp_path / 'good.pack')
    data = path.read_bytes()
    for name, damaged in (('short', data[:-64]), ('magic', b'X' + data[1:])):
        (tmp_path / name).write_bytes(damaged)
        with pytest.raises(module.Error) as caught:
            module.Pack(str(tmp_path / name))
        assert str(native.as_python(caught.value)).startswith('native pack: ')
    sec = module.constants()['sections']
    writer = native.PackWriter(tmp_path / 'mislaid.pack', native.manifest())
    writer.add(sec['VARIANTS'], np.zeros(12, np.int32))  # the right bytes for 3 keys, not as RsVariantKey records
    writer.close()
    with pytest.raises(module.Error) as caught:
        module.Pack(str(tmp_path / 'mislaid.pack'))
    assert str(native.as_python(caught.value)) == 'native pack: a record section was written with another struct layout'
    with pytest.raises(ValueError, match=r'mislaid\.pack: native pack: a record section was written with another struct '
                       r'layout; export it again'):
        native.open_pack(tmp_path / 'mislaid.pack')
    stale, _ = _pack(tmp_path / 'stale.pack', oracle_hash='0' * 64)
    with pytest.raises(ValueError, match='another oracle'):
        native.open_pack(stale)
    assert native.open_pack(stale, allow_stale=True).n_strings == 3
    for k, exporter in enumerate((dict(version='rs-native-export-v1', sha256='0' * 64), None)):  # another source, none
        other, _ = _pack(tmp_path / f'exporter{k}.pack', exporter=exporter)
        with pytest.raises(ValueError, match='another exporter'):
            native.open_pack(other)
        assert native.open_pack(other, allow_stale=True).n_strings == 3
    for k, ids in enumerate(('0' * 64, None)):  # other id tables, or none recorded: no override
        other, _ = _pack(tmp_path / f'ids{k}.pack', ids=ids)
        with pytest.raises(ValueError, match='other ids'):
            native.open_pack(other, allow_stale=True)


def test_a_pack_is_refused_when_any_id_table_it_was_written_against_changes(tmp_path, module, monkeypatch):
    """ids_hash covers each id source the record layouts do not: the library's script and section tables, rs.h's
    contract and the order of the declared worlds. A pack written on this tree opens; with any one of the four changed
    (two script ids swapped, two section ids swapped, one text key renumbered, two declared worlds swapped) it is
    refused even with allow_stale."""
    from sim import native_export, world
    path, _ = _pack(tmp_path / 'ids.pack')
    assert native.open_pack(path).n_strings == 3

    class Swapped:  # the library with two script ids swapped
        def __getattr__(self, name):
            return getattr(module, name)

        def scripts(self):
            rows = module.scripts()
            return [rows[0], rows[2], rows[1], *rows[3:]]

    class Refiled(Swapped):  # the library with two section ids swapped, its scripts as they are
        scripts = property(lambda self: module.scripts)

        def constants(self):
            out = module.constants()
            sections = out['sections']
            sections['STRINGS'], sections['STR_OFF'] = sections['STR_OFF'], sections['STRINGS']
            return out

    contract = dict(native_export.contract())
    contract['RS_TX_pick'] += 1
    declared = list(world.WORLDS.items())
    for target, name, value in ((native, 'load', Swapped), (native, 'load', Refiled),
                                (native_export, 'contract', lambda: contract),
                                (world, 'WORLDS', dict([declared[1], declared[0], *declared[2:]]))):
        with monkeypatch.context() as patch:
            patch.setattr(target, name, value)
            with pytest.raises(ValueError, match='other ids'):
                native.open_pack(path, allow_stale=True)


def test_packs_name_their_rows_and_inputs(tmp_path, module):
    """A pack's names-v1 rows must come from the table the Python projection reads, and its inputs from the files the
    caller names; neither has an override. The rows are [3][n_strings]."""
    sec = module.constants()['sections']
    pool = tmp_path / 'pool.npz'
    pool.write_bytes(b'pool v1')
    path, _ = _pack(tmp_path / 'rows.pack', rows=True, inputs=native.input_hashes(dict(pool=pool)))
    assert native.open_pack(path, names=Names('d' * 16), inputs=dict(pool=pool)).n_strings == 3
    with pytest.raises(ValueError, match='name table'):
        native.open_pack(path, names=Names('e' * 16), allow_stale=True)
    pool.write_bytes(b'pool v2')
    with pytest.raises(ValueError, match='another pool'):
        native.open_pack(path, names=Names('d' * 16), inputs=dict(pool=pool))
    writer = native.PackWriter(tmp_path / 'unnamed.pack', native.manifest())
    writer.add(sec['STR_ROW'], np.zeros(0, np.int32))
    with pytest.raises(ValueError, match='does not record their name table'):
        native.open_pack(writer.close())
    writer = native.PackWriter(tmp_path / 'short.pack', native.manifest())
    blob, offsets = _strings('', 'tree')
    writer.add(sec['STRINGS'], blob)
    writer.add(sec['STR_OFF'], offsets)
    writer.add(sec['STR_ROW'], np.zeros(5, np.int32))
    with pytest.raises(module.Error) as caught:
        module.Pack(str(writer.close()))
    assert str(native.as_python(caught.value)) == 'native pack: name rows are not [3][n_strings]'


def test_variant_rows_record_failing_keys(tmp_path, module):
    """A variant key whose variant fails the oracle's check is a row with rec -1; any other rec must name a variant."""
    sec = module.constants()['sections']
    for rec, ok in ((-1, True), (0, False)):
        rows = np.zeros(1, native.record_dtype('RsVariantKey'))
        rows['rec'] = rec
        writer = native.PackWriter(tmp_path / f'variants{rec}.pack', native.manifest())
        writer.add(sec['VARIANTS'], rows, record='RsVariantKey')
        path = writer.close()
        if ok:
            module.Pack(str(path))
        else:
            with pytest.raises(module.Error, match='variant'):
                module.Pack(str(path))


def test_the_log1p_tables_are_the_projections_values(module):
    """log1p_table holds what rl/obs.py, act_contract and feedback_contract store into float32 for a count."""
    table = native.log1p_table()
    k = module.constants()['RS_LOG1P_K']
    assert table.shape == (len(k), 65536) and table.dtype == np.float32 and not table.flags.writeable
    for count in (0, 1, 2, 25, 999, 65535):
        expected = np.zeros(4, np.float32)
        expected[:] = [float(np.log1p(count) / k[0]), np.log1p(count) / k[1], np.log1p(count) / k[2],
                       float(np.log1p(count) / k[3])]
        assert table[:, count].tobytes() == expected.tobytes()


def test_native_batch_keeps_cpubatch_interface_and_reports_what_is_not_ported(tmp_path, module):
    from sim.workers import fields
    path, _ = _pack(tmp_path / 'batch.pack', rows=True)
    pack = native.open_pack(path, names=Names('d' * 16))
    with pytest.raises(ValueError, match='not goal-activity-v4'):
        native.NativeBatch(pack, 2, task='goal-activity-v4')
    cfg = dict(task=5, n=1, first=0, n_goals=4, goal=0, horizon=240, pace_jitter=1, decision_points=1,
               max_decisions=4000, max_episode_ticks=12000)
    with pytest.raises(ValueError, match='log1p'):
        module.Batch(pack, cfg, np.zeros((4, 100), np.float32))
    unnamed, _ = _pack(tmp_path / 'unnamed.pack')
    with pytest.raises(module.Error) as caught:
        module.Batch(native.open_pack(unnamed), cfg, native.log1p_table())
    assert 'STR_ROW' in str(native.as_python(caught.value))
    batch = native.NativeBatch(pack, 2, first=5, task='goal-activity-v5')
    assert batch.goals == ['fish', 'mine'] and batch.episodes == [0, 0] and batch.evidence == []
    # before its first reset no world has drawn a start (RsSlot.has_start): none, as a failed draw leaves none
    assert batch._batch.failure(1) is None and batch._batch.start(1) is None and batch.starts == [None, None]
    with pytest.raises(IndexError):
        batch._batch.start(2)
    wrong = {name: np.zeros(shape, kind) for name, (shape, kind) in fields(2).items()}
    wrong['obs/mask'] = np.zeros((2, 96), np.int32)
    with pytest.raises(ValueError, match='obs/mask'):
        batch.bind(wrong)
    with pytest.raises(RuntimeError, match='lacks the pool a start is drawn from'):  # this pack holds no pools
        batch.reset(3)


@pytest.fixture(scope='module')
def checked(tmp_path_factory):
    path, _ = _pack(tmp_path_factory.mktemp('check') / 'check.pack', rows=True)
    return native.NativeBatch(native.open_pack(path, names=Names('d' * 16)), 3, task='goal-activity-v5')


def _raised(fn, *args):
    try:
        fn(*args)
    except ValueError as error:
        return str(error)
    return None


FLOAT_KEYS = [k for k in OBSERVATION_SHAPES if np.dtype(dtype(k)).kind == 'f']


@settings(deadline=None, max_examples=300)
@given(st.lists(st.tuples(st.sampled_from(FLOAT_KEYS), st.integers(0, 2 ** 20),
                          st.sampled_from([np.nan, np.inf, -np.inf, -0.0, 3.4e38, -1.5, 7.0])), max_size=3),
       st.lists(st.tuples(st.integers(0, 2 ** 20), st.integers(-1, 2)), max_size=3))
def test_the_native_check_raises_what_check_batch_raises(checked, floats, masks):
    """NativeBatch.check (its value checks in C) on the bound arrays against sim.spec.check_batch on the same arrays:
    the same pass or the same ValueError, whichever key and world a non-finite float or a bad mask row sits in."""
    views = checked.views
    for name in OBSERVATION_SHAPES:
        views[f'obs/{name}'][...] = 0
    views['obs/mask'][:, 5] = 1
    for name, at, value in floats:
        flat = views[f'obs/{name}'].reshape(-1)
        flat[at % flat.size] = value
    mask = views['obs/mask'].reshape(-1)
    for at, value in masks:
        mask[at % mask.size] = value
    expected = _raised(check_batch, {name: views[f'obs/{name}'] for name in OBSERVATION_SHAPES}, checked.n)
    assert _raised(checked.check) == expected
