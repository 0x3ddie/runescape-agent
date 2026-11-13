"""The four-goal evaluation (sim/run.py goals): goal control from an episode's own events,
the certified kill ticks it reads, the controls, and the guard on final evaluation data."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from sim.audit import kill_ticks
from sim.run import goal_control, goals

ROOT = Path(__file__).resolve().parents[2]


def hit(tick, npc, source, hp, life=0, full=3, type_=41):
    return (tick, 'npc_damage', dict(npc=npc, life=life, type=type_, source=source, hp=hp, full=full))


def defeat(tick, npc, source, life=0):
    return (tick, 'npc_defeated', dict(npc=npc, life=life, source=source))


def test_kill_ticks_credit_a_fresh_life_only_this_agent_hit():
    events = [hit(10, 1, 'me', 3), defeat(12, 1, 'me'),                       # credited at its defeat
              hit(20, 2, 'other', 3), hit(21, 2, 'me', 2), defeat(22, 2, 'me'),  # another hit it first
              hit(30, 3, 'me', 2), defeat(31, 3, 'me'),                       # not at full health
              hit(40, 4, 'me', 3, type_=999), defeat(41, 4, 'me'),            # not an eligible enemy
              hit(50, 5, 'me', 3), defeat(52, 5, 'other')]                    # another's hit defeated it
    assert kill_ticks(events, 'me') == [12]


def episode(tables, goal, events, start=100, end=340):
    """A finished episode with these world events after arm, for goal_control."""
    world = SimpleNamespace(events=[(0, 'produce', dict(obj=-1, count=1, script='x'))] + events, t=tables)
    return SimpleNamespace(world=world, event_start=1, terminal=dict(tick=end), start_tick=start, goal=goal,
                           agent=SimpleNamespace(name='me'))


def test_goal_control_names_the_first_completion_a_tie_and_what_came_before(tables):
    logs, shrimp = tables.obj_id('logs'), tables.obj_id('raw_shrimp')
    produce = lambda tick, obj: (tick, 'produce', dict(obj=obj, count=1, script='woodcutting'))
    fish_first = [produce(110, shrimp), produce(120, shrimp), produce(130, logs), produce(150, logs)]
    control = goal_control(episode(tables, 'wood', fish_first), refusals=[105, 140, 160])
    assert control['first_completed'] == 'fish' and control['ticks_to_outcome'] == 50
    assert control['non_goal_before_outcome'] == dict(fish=2, mine=0, melee=0) and control['refusals_before_outcome'] == 2
    tie = [produce(110, logs), produce(110, shrimp), produce(120, logs), produce(120, shrimp)]
    assert goal_control(episode(tables, 'fish', tie), refusals=[])['first_completed'] == 'tie'
    late = [produce(110, logs), produce(400, logs)]  # after the terminal tick: outside the episode
    assert goal_control(episode(tables, 'wood', late), refusals=[])['first_completed'] is None
    kill = [hit(110, 7, 'me', 3), defeat(115, 7, 'me')]
    assert goal_control(episode(tables, 'melee', kill), refusals=[])['completion_ticks']['melee'] == 15


def test_the_reference_completes_the_requested_goal_first_and_wait_completes_nothing(tables, tmp_path):
    result = goals(tmp_path / 'eval', policies=('reference', 'wait'), layouts=2, starts=2, tables=tables)
    reference, wait = result['cells']['reference'], result['cells']['wait']
    assert all(reference[g]['valid'] and reference[g]['controlled_successes'] == 2 for g in reference), reference
    assert all(wait[g]['successes'] == 0 and wait[g]['first_completed'] == {'None': 2} for g in wait), wait
    trials = json.loads((tmp_path / 'eval' / 'results.json').read_text())['trials']
    starts = {(t['policy'], t['start']): set() for t in trials}
    for t in trials:
        starts[(t['policy'], t['start'])].add(json.dumps(t['record'], sort_keys=True))
    assert all(len(v) == 1 for v in starts.values())  # every goal of a start record shares its physical start


def test_a_checkpoint_and_its_controls_get_an_acceptance_and_pairing(tables, tmp_path, name_table_path):
    from sim.learn import DeviceEnv, FakeEnv, build
    from names import NameTable
    model = build(DeviceEnv(FakeEnv(4, id_rows=NameTable(name_table_path).spec[0]), seed=1), seed=1, device='cpu',
                  n_steps=8, batch_size=32, n_epochs=1)
    model.save(tmp_path / 'policy.zip')
    result = goals(tmp_path / 'eval', checkpoint=tmp_path / 'policy.zip', initial=tmp_path / 'policy.zip',
                   policies=('final', 'initial', 'wait'), layouts=1, starts=1, tables=tables)
    assert set(result['paired']) == {'initial', 'wait'} and result['acceptance']['passed'] is False
    assert result['checkpoints']['final']['sha256'] == result['checkpoints']['initial']['sha256']
    pairs = result['paired']['wait']['wood']
    assert sum(pairs.values()) == 1


def test_the_final_split_needs_the_final_flag(tmp_path):
    done = subprocess.run([sys.executable, '-m', 'sim.run', 'goals', '--split', 'final', '--policies', 'wait',
                           '--output', str(tmp_path / 'x')], cwd=ROOT, capture_output=True, text=True,
                          env={'PYTHONPATH': f'{ROOT}:{ROOT / "rl"}', 'PATH': '/usr/bin:/bin'})
    assert done.returncode == 2 and 'needs --final' in done.stderr and not (tmp_path / 'x').exists()


def test_goal_control_ignores_production_before_arm(tables):
    logs = tables.obj_id('logs')
    task = episode(tables, 'wood', [(105, 'produce', dict(obj=logs, count=2, script='woodcutting'))])
    task.world.events.insert(0, (50, 'produce', dict(obj=logs, count=5, script='woodcutting')))  # before event_start
    task.event_start = 2
    assert goal_control(task, refusals=[])['completion_ticks']['wood'] == 5


def test_one_start_per_layout_tests_each_layout_once_with_the_same_record_for_every_goal(tables, tmp_path):
    result = goals(tmp_path / 'eval', policies=('wait', 'random'), layouts=3, starts=3, first=5, one_per_layout=True,
                   tables=tables)
    trials = json.loads((tmp_path / 'eval' / 'results.json').read_text())['trials']
    assert result['layout_seeds'] == [1000005, 1000006, 1000007] and result['bar'] == 3
    assert {t['start']: t['layout'] for t in trials} == {0: 1000005, 1: 1000006, 2: 1000007}
    records = {}
    for t in trials:
        records.setdefault(t['start'], set()).add(json.dumps(t['record'], sort_keys=True))
    assert all(len(v) == 1 for v in records.values()) and len(trials) == 2 * 4 * 3
    import gzip
    with gzip.open(tmp_path / 'eval' / 'episodes.jsonl.gz', 'rt') as f:
        kept = [json.loads(line) for line in f]
    assert len(kept) == len(trials) and all(row['decisions'] and 'report' in row for row in kept)


def test_a_distraction_variant_adds_only_its_goblins_near_the_same_start(tables, tmp_path):
    """The npc cause test: the npcs variant is the same layout and start record plus three goblins 3-9 steps from it."""
    from sim.world import DISTRACTIONS, generate, with_distractions
    plain = goals(tmp_path / 'plain', policies=('wait',), layouts=2, starts=2, first=40, one_per_layout=True, tables=tables)
    with_npcs = goals(tmp_path / 'npcs', policies=('wait',), layouts=2, starts=2, first=40, one_per_layout=True,
                      distractions=('npcs',), tables=tables)
    read = lambda d: json.loads((tmp_path / d / 'results.json').read_text())['trials']
    assert [t['record'] for t in read('plain')] == [t['record'] for t in read('npcs')] and with_npcs['distractions'] == ['npcs']
    layout = generate(plain['layout_seeds'][0], tables)
    start = read('plain')[0]['record']['tile']
    around = (start[0] - 2880, start[1] - 3328)
    variant = with_distractions(layout, tables, ('npcs',), around)
    added = variant.npcs[len(layout.npcs):]
    assert (variant.npcs[:len(layout.npcs)] == layout.npcs).all() and len(added) == DISTRACTIONS['npcs']['count']
    assert all(tables.npc_types[int(ty)]['debugname'] == 'goblin' and 3 <= max(abs(x - around[0]), abs(z - around[1])) <= 9
               for _, ty, x, z in added)
    assert all((getattr(variant, f) == getattr(layout, f)).all() for f in ('flags', 'walkable', 'start', 'locs'))
    import gzip
    with gzip.open(tmp_path / 'npcs' / 'episodes.jsonl.gz', 'rt') as f:
        assert all(len(d) == 5 for row in map(json.loads, f) for d in row['decisions'])


def test_goals_command_line_passes_the_sampling_seed_through(monkeypatch, tmp_path):
    """a sampled pass declares its seed base on the command line; the default is the frozen one."""
    from sim import run
    seen = {}
    monkeypatch.setattr(run, 'goals', lambda output, **kw: seen.update(kw) or {})
    monkeypatch.setattr('sys.argv', ['sim.run', 'goals', '--checkpoint', 'final.zip', '--policies', 'final', '--sampled',
                                     '--seed', '909001', '--output', str(tmp_path)])
    run.main()
    assert seen['sampled'] is True and seen['seed'] == 909001
    monkeypatch.setattr('sys.argv', ['sim.run', 'goals', '--checkpoint', 'final.zip', '--policies', 'final', '--output', str(tmp_path)])
    run.main()
    assert seen['sampled'] is False and seen['seed'] == 202001
