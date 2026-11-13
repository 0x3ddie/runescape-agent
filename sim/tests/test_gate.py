"""sim/gate.py, the skill gate command: its evaluation processes run every declared pass on the same records and
merge; each rule reads its cell and is invalid, never a pass or a failure, when the cell has an unscored trial or
another record count; the native check's rules count what the gate declares; verdicts follow the gating rules only;
`rules` and `native` score an output again only when its stored gate is this GATE; the overlap schedule runs the native
passes beside the simulator parts and changes no record. final4 (FINAL4, beside GATE, which stays gate-v2): its records
are their draw, frozen by digest; its trial seeds follow the design's rule; its rule reads only outputs of the declared
records and seeds, and an unscored trial makes the read incomplete."""
import gzip
import hashlib
import json

from hypothesis import given, strategies as st
import pytest

from sim import gate as G


@pytest.fixture
def tiny(monkeypatch):
    """Development records small enough for a test, with thresholds a WAIT control meets or misses on purpose."""
    records = dict(split='development', combat=dict(first=0, key=11, per_group=1), skills=dict(first=0, key=7, layouts=2),
                   supply=dict(first=0, key=13, layouts=2), search=dict(first=0, key=13, layouts=2))
    monkeypatch.setitem(G.GATE['records'], 'development', records)
    monkeypatch.setitem(G.GATE, 'parts', dict(combat=2, supply=1, search=1, skills=1))
    monkeypatch.setitem(G.GATE, 'rules', dict(wood=dict(successes=(0, 2), logs_mean=0.0, logs='recorded'),
                                              fish=dict(count='controlled_successes', at_least=(1, 2)),
                                              mine=dict(count='controlled_successes', at_least=(0, 2)),
                                              melee=dict(all=(0, 4), group=(0, 1))))


def test_the_wait_control_runs_every_pass_and_gets_one_result_per_rule(tiny, tmp_path, tables):
    found, merges = G.jobs(tmp_path, G.SKILLS, 'wait', None, 'development')
    names = [name for name, *_ in found]
    assert len(names) == 3 * (2 + 1 + 1 + 1) and len(merges) == 3 * 3  # combat in 2 parts, supply, search, skills
    assert all('--policies' in argv and argv[argv.index('--policies') + 1] == 'wait' for _, argv, _ in found)
    sampled = [argv for name, argv, _ in found if name.startswith('sampled2-combat')]
    assert all('--sampled' in argv and argv[argv.index('--seed') + 1] == '909001' for argv in sampled)
    assert not any('--sampled' in argv for name, argv, _ in found if name.startswith('greedy'))
    done = G.execute([(n, a) for n, a, _ in found], 4, tmp_path / 'logs', 900)
    done.update(G.execute([(f'merge-{k}', a) for k, a in enumerate(merges)], 4, tmp_path / 'logs', 300))
    assert all(v['exit'] == 0 for v in done.values()), done
    result = G.gate_json(tmp_path, G.SKILLS, 'wait', None, 'development', False, dict(simulator=done))
    assert json.loads((tmp_path / 'gate.json').read_text())['verdict'] == result['verdict']
    by = {(r['skill'], r['rule'], r['pass_name']): r for r in result['rules']}
    assert by[('fish', 'short.controlled_successes', 'sampled1')]['passed'] is False  # WAIT makes no fish
    assert by[('mine', 'short.controlled_successes', 'sampled2')]['passed'] is True   # a threshold of 0 of 2
    assert not by[('wood', 'short.controlled_successes', 'sampled1')]['gating']       # reported, never gating
    assert all(not r['gating'] for r in result['rules'] if r['pass_name'] == 'greedy')
    assert all(r['valid'] for r in result['rules'])
    stored = json.loads((tmp_path / 'sampled1' / 'skills' / 'results.json').read_text())
    assert not {'goals', 'policy_seed'} & set(stored)  # sim.run goals' defaults write gate-v2's keys only
    assert result['verdict'] == dict(wood=dict(simulator='pass', native='none'), fish=dict(simulator='fail', native='none'),
                                     mine=dict(simulator='pass', native='none'), melee=dict(simulator='pass', native='none'))
    cell = result['report']['simulator']['sampled1']
    assert cell['combat']['all']['trials'] == 4 and cell['combat']['all']['decisions_per_1000_ticks'] > 300  # WAIT's pace
    assert cell['skills']['fish']['refusals'] == 0 and cell['supply']['ample']['stalls'] >= 2
    # an unscored trial makes its cell's rules invalid, so the skill is incomplete, never passed
    path = tmp_path / 'sampled1' / 'supply' / 'results.json'
    merged = json.loads(path.read_text())
    merged['cells']['wait']['ample'].update(scored=1, unsupported=1, valid=False)
    path.write_text(json.dumps(merged))
    again = G.gate_json(tmp_path, G.SKILLS, 'wait', None, 'development', False, {})
    assert {r['valid'] for r in again['rules'] if r['rule'].startswith('supply.ample') and r['pass_name'] == 'sampled1'} == {False}
    assert again['verdict']['wood']['simulator'] == 'incomplete'
    # so does a gating pass that left no result, and a cell with another record count than its rule declares
    (tmp_path / 'sampled2' / 'skills' / 'results.json').unlink()
    G.GATE['rules']['melee'] = dict(all=(0, 800), group=(0, 200))
    again = G.gate_json(tmp_path, G.SKILLS, 'wait', None, 'development', False, {})
    missing = [r for r in again['rules'] if r['rule'] == 'short.controlled_successes' and r['pass_name'] == 'sampled2']
    assert len(missing) == 4 and not any(r['valid'] for r in missing) and again['verdict']['mine']['simulator'] == 'incomplete'
    assert {r['valid'] for r in again['rules'] if r['skill'] == 'melee' and r['rule'].startswith('combat')} == {False}
    assert again['verdict']['melee']['simulator'] == 'incomplete'


def native_trial(fixture, survived=True, logs=0, controlled=False):
    return dict(fixture=fixture, success=controlled, controlled_success=controlled, decisions=40, options_refused=1,
                decision_ticks_actions=[[12, 0], [14, 0], [16, 0]],
                report=dict(survived=survived, delivered_ticks=240, families={'1511': dict(retained_lower=logs)}))


def test_the_native_rules_count_eight_logs_and_survival_for_wood_and_controlled_successes_otherwise(tmp_path, monkeypatch):
    monkeypatch.setitem(G.GATE['native'], 'at_least', 26)
    wood = [native_trial('check_entrana_wood', logs=8)] * 14 + [native_trial('check_entrana_wood', survived=False, logs=12)] * 2 + \
        [native_trial('check_rimmington_wood', logs=9)] * 12 + [native_trial('check_rimmington_wood', logs=7)] * 4
    fish = [native_trial('check_arena_fish', controlled=True)] * 16 + [native_trial('check_entrana_fish', controlled=True)] * 10
    for p in G.GATE['native']['passes']:
        for engine, trials in (('pinned', wood + fish[16:]), ('arena', fish[:16])):
            path = tmp_path / 'native' / p['name'] / engine / 'results.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(status='completed', trials=trials)))
    rules, reports = G.native_rules(tmp_path, ('wood', 'fish', 'melee'), 'final')
    wood_rule = next(r for r in rules if r['skill'] == 'wood' and r['pass_name'] == 'sampled1')
    assert (wood_rule['value'], wood_rule['valid'], wood_rule['passed']) == (26, True, True)  # 14 + 12: deaths and 7 logs fail
    fish_rule = next(r for r in rules if r['skill'] == 'fish' and r['pass_name'] == 'sampled1')
    assert (fish_rule['value'], fish_rule['n'], fish_rule['valid'], fish_rule['passed']) == (26, 32, False, None)  # 26 trials
    assert reports['sampled1']['wood']['by_site'] == dict(check_entrana_wood=14, check_rimmington_wood=12)
    assert reports['sampled1']['wood']['deaths'] == 2 and not any(r['skill'] == 'melee' for r in rules)
    assert G.verdict(rules, 'wood', 'native') == 'pass' and G.verdict(rules, 'fish', 'native') == 'incomplete'
    assert G.verdict(rules, 'melee', 'native') == 'none'
    # the development record set reads the rehearsal fixtures: none of these trials, so every native rule is invalid
    rules, _ = G.native_rules(tmp_path, ('wood', 'fish'), 'development')
    assert rules and not any(r['valid'] for r in rules) and G.verdict(rules, 'wood', 'native') == 'incomplete'


# A stand-in for scripts/stack.py, its world service and baseline.evaluate, given to `sim.gate run` as --native-python:
# it starts no world and never takes logs/stack.lock. Each native pass copies its prepared results into its output and
# records how many simulator results.json files existed as it began (at most one pass at a time, or it fails); the last
# pass then waits for all of them, so it ends only if the simulator parts and merges run while it is open.
STAND_IN = """#!/bin/sh
here=$(dirname "$0")
while [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done
while [ $# -gt 0 ]; do [ "$1" = --output ] && out=$2; shift; done
gate=$(dirname "$(dirname "$out")"); name=$(basename "$out")
count() { ls "$gate"/*/*/results.json 2>/dev/null | wc -l | tr -d ' '; }
mkdir "$here/busy" || exit 3
echo "$name $(count)" >> "$here/events"
if [ "$name" = greedy ]; then
  n=0; while [ "$(count)" -lt TOTAL ] && [ $n -lt 1500 ]; do sleep 0.2; n=$((n + 1)); done
  echo "end $(count)" >> "$here/events"
fi
mkdir -p "$out" && cp -R "$here/$name/." "$out/" && rmdir "$here/busy"
"""


def records(out):
    """A gate output without its timing: gate.json without 'timing', each results*.json without timing_seconds and
    wall_seconds, each .jsonl.gz as its decompressed lines (the gzip header carries the write time), every other file
    as bytes; the process logs are left out."""
    got = {}
    for path in sorted(p for p in out.rglob('*') if p.is_file() and p.relative_to(out).parts[0] != 'logs'):
        name = path.relative_to(out).as_posix()
        if path.name == 'gate.json':
            got[name] = {k: v for k, v in json.loads(path.read_text()).items() if k != 'timing'}
        elif path.name.startswith('results') and path.suffix == '.json':
            got[name] = {k: v for k, v in json.loads(path.read_text()).items() if k not in ('timing_seconds', 'wall_seconds')}
        elif path.name.endswith('.jsonl.gz'):
            with gzip.open(path, 'rb') as f:
                got[name] = f.read().splitlines()
        else:
            got[name] = path.read_bytes()
    return got


def test_the_overlap_schedule_runs_the_native_passes_beside_the_simulator_and_changes_no_record(tiny, tmp_path, tables,
                                                                                                   monkeypatch):
    stand_in = tmp_path / 'stand-in'
    for p in G.GATE['native']['passes']:
        for engine, names in G.native_fixtures(G.SKILLS, 'development').items():
            path = stand_in / p['name'] / engine / 'results.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(status='completed', trials=[
                native_trial(name, survived=k != 5, logs=k % 12, controlled=k % 3 == 0) for name in names for k in range(16)])))
    total = len(G.GATE['passes']) * len({t for s in G.SKILLS for t in G.TASKS[s]})  # every pass's merged results
    (stand_in / 'python').write_text(STAND_IN.replace('TOTAL', str(total)))
    (stand_in / 'python').chmod(0o755)
    common = ['sim.gate', 'run', '--policy', 'wait', '--native', '--bun', 'bun', '--native-python', str(stand_in / 'python')]
    events = {}
    for schedule, extra in (('serial', ['--workers', '1']), ('overlap', ['--workers', '2', '--schedule', 'overlap'])):
        monkeypatch.setattr('sys.argv', [*common, *extra, '--output', str(tmp_path / schedule)])
        G.main()
        events[schedule] = [line.split() for line in (stand_in / 'events').read_text().splitlines()]
        (stand_in / 'events').unlink()
    # serial (the default): every simulator result exists before the first native pass
    assert events['serial'] == [['sampled1', str(total)], ['sampled2', str(total)], ['greedy', str(total)], ['end', str(total)]]
    # overlap: the native passes, one at a time in GATE's order, start before any simulator result and outlast them
    assert [e[0] for e in events['overlap']] == ['sampled1', 'sampled2', 'greedy', 'end']
    assert events['overlap'][0] == ['sampled1', '0'] and events['overlap'][-1] == ['end', str(total)]
    serial, overlap = records(tmp_path / 'serial'), records(tmp_path / 'overlap')
    assert serial == overlap
    assert sum(name.endswith('.jsonl.gz') for name in serial) == 3 * 5 and len(serial['gate.json']['rules']) == 51
    assert {r['kind'] for r in serial['gate.json']['rules'] if r['valid']} == {'simulator', 'native'}
    timing = {s: json.loads((tmp_path / s / 'gate.json').read_text())['timing'] for s in events}
    assert 'schedule' not in timing['serial'] and timing['overlap']['schedule'] == 'overlap'
    assert set(timing['serial']['native']) == set(timing['overlap']['native']) == \
        {f'native-{p["name"]}' for p in G.GATE['native']['passes']}


def test_a_rehearsal_plays_the_rehearsal_records_and_only_the_final_record_set_plays_final_data(monkeypatch, tmp_path):
    from baseline import config
    seen = []
    monkeypatch.setattr(G, 'execute', lambda commands, *args: seen.extend(commands) or {})
    for records in ('development', 'final'):
        G.native_trials(tmp_path, 'wait', None, True, 41001, ('wood', 'fish', 'mine', 'melee'), records)
    development, final = seen[:2], seen[2:]
    names = lambda commands: sorted(n for _, argv in commands for n in argv[argv.index('--fixtures') + 1:argv.index('--trials')])
    assert names(development) == sorted(config.REHEARSAL_FIXTURES) and names(final) == sorted(config.CHECK_FIXTURES)
    assert not any('--final' in argv for _, argv in development) and all('--final' in argv for _, argv in final)
    assert all(not f['final'] for f in config.REHEARSAL_FIXTURES.values())
    assert G.native(tmp_path, 'wait', None, ('melee',), 'development', 'bun', {}, 'python') == {}  # no native check
    assert len(seen) == 4 and not (tmp_path / 'native').exists()  # and no stack was asked for


def scored_fish_output(out):
    """A fish gate output on the final records with every pass's simulator and native results, scored with this GATE:
    170 of 200 controlled successes in each simulator pass, 16 + 10 of 32 in each native pass."""
    cells = {goal: dict(valid=True, trials=200, scored=200, controlled_successes=170) for goal in G.SKILLS}
    for p in G.GATE['passes']:
        path = out / p['name'] / 'skills' / 'results.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(trials=[], cells=dict(wait=cells))))
    fish = dict(arena=[native_trial('check_arena_fish', controlled=True)] * 16,
                pinned=[native_trial('check_entrana_fish', controlled=k < 10) for k in range(16)])
    for p in G.GATE['native']['passes']:
        for engine, trials in fish.items():
            path = out / 'native' / p['name'] / engine / 'results.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(status='completed', trials=trials)))
    return G.gate_json(out, ('fish',), 'wait', None, 'final', True, {})


def test_rules_scores_a_gate_json_with_this_gate_again_to_its_stored_verdict(tmp_path, monkeypatch):
    scored_fish_output(tmp_path)
    stored = json.loads((tmp_path / 'gate.json').read_text())
    assert stored['verdict'] == dict(fish=dict(simulator='pass', native='pass'))
    # the verdict comes from the results again, not from the file
    (tmp_path / 'gate.json').write_text(json.dumps(dict(stored, verdict={}, rules=[])))
    monkeypatch.setattr('sys.argv', ['sim.gate', 'rules', str(tmp_path)])
    G.main()
    again = json.loads((tmp_path / 'gate.json').read_text())
    assert again['verdict'] == stored['verdict'] and again['rules'] == stored['rules'] and again['gate'] == stored['gate']


@pytest.mark.parametrize('command', (['rules'], ['native', '--bun', 'bun']))
def test_rules_and_native_refuse_a_gate_json_scored_with_phase_0s_native_passes(tmp_path, monkeypatch, command):
    scored_fish_output(tmp_path)
    stamp = json.loads((tmp_path / 'gate.json').read_text())
    # phase 0's gate.json files carry 'gate-v1' with the native passes sampled1 and greedy (no sampled2)
    stamp['gate']['native']['passes'] = [p for p in stamp['gate']['native']['passes'] if p['name'] != 'sampled2']
    (tmp_path / 'gate.json').write_text(json.dumps(stamp))
    before = (tmp_path / 'gate.json').read_bytes()
    played = []
    monkeypatch.setattr(G, 'native', lambda *args: played.append(args) or {})
    monkeypatch.setattr('sys.argv', ['sim.gate', command[0], str(tmp_path), *command[1:]])
    with pytest.raises(SystemExit, match=r'these keys differ: gate\.native\.passes\. Score it again only with the code'):
        G.main()
    assert (tmp_path / 'gate.json').read_bytes() == before and not played  # nothing scored again, no native pass


rule = st.fixed_dictionaries(dict(skill=st.sampled_from(('wood', 'fish')), kind=st.sampled_from(('simulator', 'native')),
                                  gating=st.booleans(), valid=st.booleans(), value=st.integers(0, 200),
                                  threshold=st.integers(0, 200)))


@given(st.lists(rule, max_size=12))
def test_a_verdict_passes_only_when_every_gating_rule_is_valid_and_met(rows):
    for r in rows:  # as simulator_rules and native_rules fill them
        r['passed'] = r['value'] >= r['threshold'] if r['valid'] else None
    for skill in ('wood', 'fish'):
        for kind in ('simulator', 'native'):
            gating = [r for r in rows if r['skill'] == skill and r['kind'] == kind and r['gating']]
            got = G.verdict(rows, skill, kind)
            if not gating:
                assert got == 'none'
            elif any(r['valid'] and r['value'] < r['threshold'] for r in gating):
                assert got == 'fail'
            else:
                assert got == ('pass' if all(r['valid'] for r in gating) else 'incomplete')


def test_the_final2_native_records_are_a_new_frozen_draw_disjoint_from_every_earlier_native_record():
    """16 draws from the final2 seed over every earlier native draw (the study's,
    the transfer's, the check draws and native-hour-v1's nine), one draw at both wood sites, frozen by their digests."""
    from baseline import config
    from sim.hour import native_records
    key = lambda r: (r['set_stats'], r['seed_items'], r['reset_seed'])
    assert G.verify_final2_records() == G.GATE['native']['final2']['digests']
    sites = {site: [key(r) for r in G.final2_records(site)] for site in ('entrana', 'rimmington')}
    drawn = set(sites['entrana'])
    assert sites['entrana'] == sites['rimmington'] and len(drawn) == G.GATE['native']['final2']['records'] == 16
    earlier = {(r['set_stats'], r['seed_items'], r['reset_seed']) for r in config.earlier_native_draws()}
    earlier |= {key(r) for r in config.check_records('entrana')} | {key(r) for r in native_records()}
    assert not drawn & earlier
    original = G.FINAL2_DRAWS[5]
    try:
        G.FINAL2_DRAWS[5] = (original[0], original[1], original[2] + 1)
        with pytest.raises(ValueError, match='frozen'):
            G.verify_final2_records()
    finally:
        G.FINAL2_DRAWS[5] = original


def test_the_final2_native_pass_plays_in_process_after_every_earlier_fixture_with_the_declared_seed_offsets(tmp_path,
                                                                                                          monkeypatch):
    from baseline import config
    import baseline.evaluate
    monkeypatch.setattr(config, 'EVAL_FIXTURES', dict(config.EVAL_FIXTURES))
    before = list(config.EVAL_FIXTURES)
    calls = []
    monkeypatch.setattr(baseline.evaluate, 'evaluate', lambda output, **kwargs: calls.append((output, kwargs)))
    monkeypatch.setattr(G, 'execute', lambda *args: pytest.fail('final2 starts no baseline.evaluate process'))
    done = G.native_trials(tmp_path, 'wait', None, True, 909001, ('wood',), 'final2')
    names = ['gate2_entrana_wood', 'gate2_rimmington_wood']
    assert done['pinned']['exit'] == 0 and list(config.EVAL_FIXTURES) == before + names  # earlier positions unchanged
    (output, kwargs), = calls
    assert output == tmp_path / 'pinned' and kwargs['fixtures'] == names and kwargs['trials'] == 16
    assert kwargs['seed_offsets'] == dict(gate2_entrana_wood=91, gate2_rimmington_wood=92)
    assert (kwargs['policy'], kwargs['seed'], kwargs['deterministic'], kwargs['final'], kwargs['world_service']) == \
        ('wait', 909001, False, True, True)
    fixtures = {name: config.EVAL_FIXTURES[name] for name in names}
    assert all(f['final'] and f['goal'] == 'wood' and f['horizon'] == config.CHECK_HORIZONS['wood'] for f in fixtures.values())
    assert [f['start_set'] for f in fixtures.values()] == list(config.CHECK_SITES['wood'])
    # a failed evaluation is that engine's failed exit, as a failed baseline.evaluate process is
    monkeypatch.setattr(baseline.evaluate, 'evaluate', lambda output, **kwargs: 1 / 0)
    assert G.native_trials(tmp_path, 'wait', None, True, 909001, ('wood',), 'final2')['pinned']['exit'] == 1


def test_the_final2_native_rules_read_its_own_fixtures_and_count_32_records(tmp_path):
    wood = [native_trial('gate2_entrana_wood', logs=8)] * 16 + [native_trial('gate2_rimmington_wood', logs=9)] * 10 + \
        [native_trial('check_rimmington_wood', logs=9)] * 6  # another record set's trials never count
    for p in G.GATE['native']['passes']:
        path = tmp_path / 'native' / p['name'] / 'pinned' / 'results.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(status='completed', trials=wood)))
    rules, reports = G.native_rules(tmp_path, ('wood',), 'final2')
    assert {(r['value'], r['n'], r['valid'], r['passed']) for r in rules} == {(26, 32, False, None)}  # 26 of 32 trials
    assert reports['sampled1']['wood']['by_site'] == dict(gate2_entrana_wood=16, gate2_rimmington_wood=10)


def test_final2_serves_wood_only_and_a_gate_v1_output_is_scored_again_only_with_its_own_commit(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['sim.gate', 'run', '--policy', 'wait', '--records', 'final2', '--final',
                                     '--output', str(tmp_path / 'fish'), '--skills', 'wood', 'fish'])
    with pytest.raises(SystemExit):
        G.main()
    assert not (tmp_path / 'fish').exists()
    scored_fish_output(tmp_path)
    stamp = json.loads((tmp_path / 'gate.json').read_text())
    stamp['gate'] = dict(stamp['gate'], version='gate-v1', records={k: v for k, v in stamp['gate']['records'].items()
                                                                     if k != 'final2'},
                         native={k: v for k, v in stamp['gate']['native'].items() if k != 'final2'})
    (tmp_path / 'gate.json').write_text(json.dumps(stamp))
    monkeypatch.setattr('sys.argv', ['sim.gate', 'rules', str(tmp_path)])
    with pytest.raises(SystemExit, match=r'differ: gate\.native\.final2, gate\.records\.final2, gate\.version\.'):
        G.main()


def test_gate_stays_gate_v2_and_final4_is_as_frozen():
    """GATE as gate-v2 froze it, and FINAL4 as frozen in the native gap study's Stage 3 (8 November 2026)."""
    digest = lambda d: hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    assert G.GATE['version'] == 'gate-v2' and 'final4' not in G.GATE['records']
    assert digest(G.GATE) == '4eac45b0b4e93bf9f0183db22b240dcfeb95900c3c9f21301de43909f7b03cfb'
    assert digest(G.FINAL4) == 'ab1926935ddbb120bbe72bf76f8caab1660a387c6c7e2583f382cf770358bb35'


def test_the_final4_records_are_their_draw_frozen_by_digest(tables, monkeypatch):
    """final split 2,000,000 + 5,000-5,199, one start per layout, key 197, compact-world-v1;
    drawn as sim.run goals draws them, never played here."""
    from sim.run import start_records
    from sim.world import WORLD
    rec = G.FINAL4['records']
    assert rec['sha256'] == '8d65f581d365cbd091723994f8c6c1b1bfa7509484e56eac0a11246fc8c57132'
    pool, records = start_records(tables, split='final', layouts=200, first=5000, key=197, starts=200, one_per_layout=True)
    assert [layout.seed for layout in pool] == list(range(2_005_000, 2_005_200)) and WORLD['version'] == rec['world']
    assert [r['layout'] for r in records] == list(range(200))
    assert G.records_sha256(pool, records) == rec['sha256']
    moved = [dict(records[7], world_seed=records[7]['world_seed'] + 1) if k == 7 else r for k, r in enumerate(records)]
    assert G.records_sha256(pool, moved) != rec['sha256']  # the digest covers every record
    monkeypatch.setitem(G.FINAL4, 'records', dict(rec, key=198))
    with pytest.raises(ValueError, match='frozen digest'):
        G.verify_final4_records(tables)


def test_final4_trial_seeds_are_base_plus_1000_policy_seeds_plus_record(tables, tmp_path, monkeypatch):
    from sim import run as R
    jobs = dict(G.final4_jobs(tmp_path, 'ck.zip', 202, 'final4'))
    for p in G.FINAL4['passes']:
        argv = jobs[p['name']]
        flag = lambda name: argv[argv.index(name) + 1]
        assert (flag('--seed'), flag('--policy-seed'), flag('--goals'), flag('--split')) == (str(p['seed']), '202', 'mine', 'final')
        assert (flag('--first'), flag('--layouts'), flag('--starts'), flag('--key')) == ('5000', '200', '200', '197')
        assert '--final' in argv and '--one-per-layout' in argv and ('--sampled' in argv) == p['sampled']
    assert [p['seed'] for p in G.FINAL4['passes']] == [57_100_000, 67_100_000, 57_100_000]
    development = dict(G.final4_jobs(tmp_path, 'ck.zip', 202, 'development', 2))['sampled2']
    assert '--final' not in development and development[development.index('--split') + 1] == 'development'
    # sim.run goals plays them: one trial seed a record, the same for every goal; the default rule is unchanged
    seen = []

    class Recording(R.Actor):
        def reset(self, seed):
            seen.append((self.fixture['goal'], seed))
            super().reset(seed)
    monkeypatch.setattr(R, 'Actor', Recording)
    common = dict(policies=('wait',), split='development', layouts=2, starts=2, key=197, one_per_layout=True, sampled=True,
                  tables=tables)
    got = R.goals(tmp_path / 'final4', goals=('mine',), seed=67_100_000, policy_seed=202, **common)
    assert seen == [('mine', 67_100_000 + 202_000 + k) for k in range(2)]
    assert got['goals'] == ['mine'] and got['policy_seed'] == 202 and list(got['cells']['wait']) == ['mine']
    seen.clear()
    got = R.goals(tmp_path / 'default', goals=('mine', 'fish'), seed=202001, **common)
    assert seen == [(goal, 202001 + 1000 * k + g) for goal, g in (('fish', 1), ('mine', 2)) for k in range(2)]
    assert got['goals'] == ['fish', 'mine'] and got['policy_seed'] is None


def test_final4_holds_only_on_valid_reads_of_the_declared_records(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'ck.zip'
    checkpoint.write_bytes(b'final')

    played = hashlib.sha256(b'final').hexdigest()

    def output(name, counts, policy_seed=202, sha=played, **cell):
        out = tmp_path / name
        for p, count in zip(G.FINAL4['passes'], counts):
            path = out / p['name'] / 'results.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(split='final', first=5000, layouts=200, starts=200, key=197, one_per_layout=True,
                                            sampled=p['sampled'], seed=p['seed'], policy_seed=policy_seed, goals=['mine'],
                                            checkpoints=dict(final=dict(path=str(checkpoint), sha256=sha)),
                                            trials=[], cells=dict(final=dict(mine=dict(dict(valid=True, trials=200, scored=200),
                                                                                       controlled_successes=count, **cell))))))
        return G.final4_json(out, checkpoint, 202, 'final4', None, {})
    held = output('held', (170, 160, 90))
    assert held['verdict'] == 'pass' and [r['passed'] for r in held['rules']] == [True, True, False]
    assert [r['gating'] for r in held['rules']] == [True, True, False]  # greedy is reported
    assert json.loads((tmp_path / 'held' / 'final4.json').read_text())['verdict'] == 'pass'
    assert output('missed', (170, 159, 200))['verdict'] == 'fail'
    unscored = output('unscored', (199, 199, 199), valid=False, scored=199)  # an unsupported trial
    assert unscored['verdict'] == 'incomplete' and not any(r['valid'] for r in unscored['rules'])
    other = output('other', (170, 170, 170), policy_seed=101)  # another seed's trial seeds
    assert other['verdict'] == 'incomplete' and {r['value'] for r in other['rules']} == {None}
    stranger = output('stranger', (170, 170, 170), sha='0' * 64)  # another checkpoint's passes
    assert stranger['verdict'] == 'incomplete' and {r['value'] for r in stranger['rules']} == {None}
    # the command: the final4 records need --final, and an output is never written over
    for argv in (['--records', 'final4', '--output', str(tmp_path / 'new')],
                 ['--records', 'development', '--output', str(tmp_path / 'held')]):
        monkeypatch.setattr('sys.argv', ['sim.gate', 'final4', '--checkpoint', str(checkpoint), '--policy-seed', '202', *argv])
        with pytest.raises(SystemExit):
            G.main()
    assert not (tmp_path / 'new').exists() and not (tmp_path / 'held' / 'logs').exists()
