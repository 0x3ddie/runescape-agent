"""sim/hour.py, hour-v1: one character persists past the task horizon for the whole hour under the declared
clock, death ends the hour at its tick, an unsupported step stops the trial unscored, the XP rate is the world's, and
inventory-full responses and the merged parts are accounted as declared. native-hour-v1: the native
episode shows hour-v1's clock, the windows sum to the audit's production, and a pass reports failed and unplayed hours."""
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from baseline import config
from baseline.native_value import audit_native_value
from baseline.resource_task import ResourceEpisode
from baseline.tests.test_goal_task import DECLARED
from baseline.tests.test_native_flow import CONTAINERS, item
from baseline.tests.test_resource_task import ARMED, CFG, raw as plain_raw
from feedback_contract import quiet
from sim import hour as H
from sim.agent import A_ACT0, A_DROP0, A_STYLE0, A_WAIT
from sim.combat import ControlActor
from sim.engine import WOODCUTTING, Unsupported, World


class Recorder(ControlActor):
    """WAIT, keeping each observation's task clock; `hook(task, k)` runs before decision k."""

    def __init__(self, hook=None):
        super().__init__('wait')
        self.hook, self.seen = hook, []

    def act(self, observation, task=None, trace=False):
        if self.hook:
            self.hook(task, len(self.seen))
        self.seen.append((task.world.tick - task.start_tick, *observation['budget_vec'], observation['task_vec'][0]))
        return A_WAIT


@pytest.fixture(scope='module')
def melee(tables):
    layouts, records = H.records('melee', 1, tables)
    return layouts[0], records[0]


def test_the_character_plays_the_whole_hour_past_its_task_horizon_under_the_window_clock(tables, melee):
    layout, record = melee
    for clock in ('window', 'hour'):
        task = H.hour_episode('melee', layout, record, tables, 25, clock)
        actor = Recorder()
        rows, info, tiles, full = H.trial(task, actor, 1)
        assert info['episode']['native_ticks'] == H.HOUR['ticks'] and info['report']['survived']
        assert task.world.tick - task.start_tick == H.HOUR['ticks']  # no reset at the 1,200-tick task horizon
        ticks, budget, scale, decision = (np.asarray(v) for v in zip(*actor.seen))
        window = H.HOUR['windows']['melee']
        if clock == 'window':  # the clock restarts at each window: the values a fresh 1,200-tick episode shows
            assert np.allclose(budget, 1 - (ticks % window) / window) and np.allclose(scale, window / 10000)
            first = [k for k, t in enumerate(ticks) if t >= window][0]
            assert decision[first] == decision[0] < decision[first - 1]
        else:
            assert np.allclose(budget, 1 - ticks / H.HOUR['ticks']) and np.allclose(scale, H.HOUR['ticks'] / 10000)
            assert (np.diff(decision) >= 0).all()
        assert len(tiles) == 1 and full.events == []


def test_death_ends_the_hour_at_its_tick_and_an_unsupported_step_stops_the_trial_unscored(tables, melee, tmp_path,
                                                                                         monkeypatch):
    layout, record = melee

    def lethal(task, k):  # a lethal hit inside the next cycle, through the engine's damage path as an enemy's
        if k == 300:
            cycle = task.world.cycle

            def hit():
                task.world.player_damage(task.agent.player, task.agent.player.hp)
                cycle()
            task.world.cycle = hit
    task = H.hour_episode('melee', layout, record, tables, 25)
    rows, info, tiles, full = H.trial(task, Recorder(lethal), 1)
    assert info['report']['reason'] == 'death' and 0 < info['episode']['native_ticks'] < H.HOUR['ticks']
    row = H.score(task, rows, info, tiles, full, H.skills(task.agent.player), tables)
    lethal_tick = next(t for t, kind, data in task.world.events if kind == 'agent_hp' and data['after'] == 0)
    # the hour ends with the cycle that applied the hit (World.tick advances after it), not a decision later
    assert row['died'] and row['death_tick'] == lethal_tick + 1 - task.start_tick == task.world.tick - task.start_tick
    # an unsupported step: reported with its tick, never scored, and the summary is not valid
    calls = []

    def unsupported(self, action):
        calls.append(action)
        raise Unsupported('a script the slice lacks')
    original = H.HourEpisode.step

    def step(self, action):
        if self.decisions == 50:
            unsupported(self, action)
        return original(self, action)
    monkeypatch.setattr(H.HourEpisode, 'step', step)
    result = H.run(tmp_path / 'u', goals=('melee',), trials=1, policies=('wait',), tables=tables)
    played = result['trials_played'][0]
    assert calls and 'unsupported' in played and played['decision'] == 50 and 'ticks' not in played
    summary = H.summarize(result['trials_played'])
    assert summary['unsupported'] == 1 and summary['scored'] == 0 and not summary['valid']


def test_the_xp_rate_is_the_worlds_multiplier(tables, melee):
    layout, record = melee
    for rate in (1, 25):
        task = H.hour_episode('melee', layout, record, tables, rate)
        p, before = task.agent.player, task.agent.player.stats[WOODCUTTING]
        task.world.add_xp(p, WOODCUTTING, 250)  # a normal tree's productexp, in tenths
        assert p.stats[WOODCUTTING] - before == 250 * rate and task.world.xp_rate == rate
    with pytest.raises(ValueError):
        World(task.world.t, xp_rate=0)
    with pytest.raises(ValueError):
        H.run('unused', xp_rate=5, policies=('wait',), tables=tables)
    with pytest.raises(ValueError, match='final'):
        H.run('unused', split='final', policies=('wait',), tables=tables)


def stub(free, events):
    """A task with a scripted inventory and event list for FullInventory."""
    world = SimpleNamespace(tick=10, events=events)
    inv = SimpleNamespace(freespace=lambda: free[0])
    return SimpleNamespace(world=world, start_tick=10, decisions=0, agent=SimpleNamespace(player=SimpleNamespace(inv=inv)))


def test_an_inventory_full_event_is_answered_by_what_frees_its_first_slot_in_time():
    events, free = [], [3]
    task = stub(free, events)
    full = H.FullInventory(task)
    window = H.HOUR['full_response_ticks']
    timeline = [(0, 'drop', 20), (5, 'deposit', 30), (0, 'eat', 15), (0, 'drop', window + 50)]
    for gap, cause, after in timeline:
        free[0] = 0
        task.world.tick += gap + 1
        full.observe()
        task.world.tick += 3  # a wield that swaps the worn weapon frees no slot
        events += [(task.world.tick, 'transfer', dict(src='inv', dst='worn', obj=2, to=2, count=1)),
                   (task.world.tick, 'transfer', dict(src='worn', dst='inv', obj=3, to=3, count=1))]
        full.observe()
        task.world.tick += after
        if cause == 'deposit':
            events.append((task.world.tick, 'transfer', dict(src='inv', dst='bank', obj=1, to=1, count=1)))
        else:
            events.append((task.world.tick, 'remove', dict(why=cause, obj=1, count=1)))
        free[0] = 1
        full.observe()
    free[0] = 0
    task.world.tick += 1
    full.observe()  # still full when the hour ends
    found = full.finish(task.world.tick - task.start_tick + 5)
    assert [e['response'] for e in found] == ['drop', 'bank', 'other', 'stall', 'open']
    assert [e['freed_by'] for e in found] == ['drop', 'bank', 'eat', 'drop', None]


def test_merge_refuses_an_incomplete_or_mixed_run(tables, tmp_path):
    for part in range(2):
        H.run(tmp_path / 'a', goals=('mine',), trials=2, policies=('wait',), part=(part, 2), tables=tables)
    merged = H.merge(tmp_path / 'a')
    assert merged['parts'] == 2 and merged['cells']['wait']['mine']['scored'] == 2
    assert merged['cells']['wait']['mine']['mean_ticks'] == H.HOUR['ticks']
    H.run(tmp_path / 'b', goals=('mine',), trials=2, policies=('wait',), part=(0, 2), tables=tables)
    with pytest.raises(ValueError, match='complete split'):
        H.merge(tmp_path / 'b')
    H.run(tmp_path / 'b', goals=('mine',), trials=2, policies=('wait',), part=(1, 2), xp_rate=1, tables=tables)
    with pytest.raises(ValueError, match='different runs'):
        H.merge(tmp_path / 'b')
    # a held-v1 part (written before held-v2: no `held` in its header) never pools with a held-v2 part
    part = tmp_path / 'a' / 'results-1.json'
    old = json.loads(part.read_text())
    assert old.pop('held') == 'held-v2'
    part.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='different runs'):
        H.merge(tmp_path / 'a')


START = 1000  # the native capture's start tick in these tests


def hour_trace(productions=(), start=START, ticks=H.NATIVE['ticks']):
    """A surviving native capture of the hour (baseline/tests/test_native_flow.py's form, every goal family declared) with
    one production of item `base` at each (tick from arm, base), each into the next inventory slot."""
    current, events, pending = dict(inventory=[], bank=[]), [], sorted(productions)
    for t in range(ticks + 1):
        if t:
            events.append(dict(kind='tick-boundary', tick=start + t, holdings=deepcopy(current)))
        while pending and pending[0][0] == t:
            after = dict(current, inventory=current['inventory'] + [item(len(current['inventory']), id=pending.pop(0)[1])])
            events.append(dict(kind='native-op', tick=start + t, before=deepcopy(current), after=deepcopy(after), opcode=4302,
                               productionSource=True, scriptId=51, scriptName='[label,get_logs]', pc=5,
                               stackTail=[93, after['inventory'][-1]['id'], 1], ground=[], resourceMutation=True,
                               ownedSlotSwap=None))
            current = after
    terminal = dict(reason='horizon', tick=start + ticks, holdings=deepcopy(current), toolAvailable=True,
                    lifecycle=dict(phase='alive', failure=None, deathSequence=0, respawnSequence=0, deathTick=None))
    events.append(dict(kind='terminal', **terminal))
    for sequence, event in enumerate(events, 1):
        event['sequence'] = sequence
    return dict(schema='native-resource-capture-v2', flowEvidenceVersion=1, slotSwapEvidenceVersion=1,
                ownedContainers=deepcopy(CONTAINERS), targetScope='gathering-v1', targets=deepcopy(DECLARED),
                identity=dict(worldBootId='boot', accountId='a', accountEpoch=1, incarnationId='p'),
                initial=dict(inventory=[], bank=[]), last=deepcopy(current), startTick=start, endTick=start + ticks,
                observedTick=start + ticks, events=events, terminal=terminal, failure=None, closed=True, completed=True,
                evidenceComplete=True, lifecycleObserved=True)


class Native:
    """A native world and its resource control: observations at the ticks of `shown` [(tick from arm, raw)], the capture
    `trace` (closed at the hour's end), every control operation in `calls`, and `fail` an operation that raises."""

    def __init__(self, shown, trace, fail=None):
        self.shown, self.trace, self.fail, self.k, self.calls, self.closed = shown, trace, fail, 0, [], False

    def reset(self, seed, cfg):
        self.k = 0
        return deepcopy(self.shown[0][1])

    def step(self, action):
        self.k += 1
        return deepcopy(self.shown[self.k][1])

    def close(self):
        self.closed = True

    def control(self, operation, **kwargs):
        self.calls.append(operation)
        if operation == self.fail:
            raise RuntimeError(f'{operation} failed')
        identity = deepcopy(self.trace['identity'])
        if operation == 'identity':
            return dict(type='resource_identity', identity=identity)
        if operation == 'arm':
            return dict(type='resource_capture', identity=identity, audit=dict({k: deepcopy(self.trace[k]) for k in ARMED},
                                                                                closed=False))
        if operation == 'budget':
            tick, end = START + self.shown[self.k][0], START + H.NATIVE['ticks']
            return dict(type='resource_budget', identity=identity, budget=dict(
                schema='resource-budget-v1', startTick=START, endTick=end, observedTick=tick, horizon=H.NATIVE['ticks'],
                remainingTicks=end - tick, status='horizon' if tick == end else 'active'))
        return dict(type='resource_capture', identity=identity, audit=deepcopy(self.trace))


def test_the_native_hour_shows_hour_v1s_window_clock_for_the_same_ticks_and_decisions(tables):
    layouts, records = H.records('wood', 1, tables)
    task = H.hour_episode('wood', layouts[0], records[0], tables, 25)
    actor, actions = ControlActor('random'), []
    actor.reset(7)
    shown, vectors, done = [(0, deepcopy(task.raw))], [task.projected], False
    while not done:  # random legal actions: decisions of varied length, across every 300-tick window of the hour
        actions.append(actor.act(task.projected, task))
        obs, _, done, info = task.step(actions[-1])
        shown.append((task.budget()[0] - task.start_tick, deepcopy(task.raw)))
        vectors.append(obs)
    assert info['report']['survived'] and shown[-1][0] == H.NATIVE['ticks'] and len({t // 300 for t, _ in shown}) == 21
    world = Native(shown, hour_trace())
    episode = H.NativeHourEpisode(ResourceEpisode(world, world.control, config.TASK_CFG, 1, 40, lambda *a, **k: None,
                                                  goal='wood'))
    native = [episode.reset(0)] + [episode.step(a)[0] for a in actions]
    for sim_obs, native_obs in zip(vectors, native, strict=True):
        assert np.array_equal(native_obs['budget_vec'], sim_obs['budget_vec'])
        assert native_obs['task_vec'][0] == sim_obs['task_vec'][0]
    assert episode.closed is False and episode.bound is False  # the hour closed at its horizon and released its capture


def test_the_window_counts_sum_to_the_audits_production_per_family():
    trace = hour_trace([(0, 1511), (1199, 1511), (1200, 1521), (2399, 1511), (4800, 1519), (6000, 1511)])
    counts = H.window_counts(trace)
    assert counts == {1511: [2, 1, 0, 0, 1], 1521: [0, 1, 0, 0, 0], 1519: [0, 0, 0, 0, 1]}  # the last tick in window 5
    report = audit_native_value(trace)
    assert {b: sum(c) for b, c in counts.items()} == {b: f['produced'] for b, f in report['families'].items()}


KIT = [dict(slot=s, id=900 + s, count=1, name=f'kit {s}') for s in range(25)] + [dict(slot=25, id=2309, count=1, name='Bread')]


def scripted_hour(productions, levels, every=100):
    """[(tick from arm, raw)] of a scripted native hour, one decision every `every` ticks: the kit leaves two slots free;
    each production adds a log the next observation shows; DROP_0 (the bread) is legal while the inventory is full and the
    bread is held, and then drops it; a tree to chop ('Dead tree' before tick 1200, 'Tree' from it); the woodcutting base
    level `levels` gives from each tick on."""
    shown, dropped = [], None
    for k, t in enumerate(range(0, H.NATIVE['ticks'] + 1, every)):
        logs = [dict(slot=0, id=base, count=1, name='Logs') for tick, base in productions if tick < t]
        inventory = [i for i in KIT if not (i['id'] == 2309 and dropped is not None and dropped < t)] + logs
        mask = [0] * 96
        mask[A_WAIT] = mask[A_ACT0] = 1
        if len(inventory) == H.INVENTORY_SLOTS and dropped is None:
            mask[A_DROP0], dropped = 1, t
        level = max(v for tick, v in levels.items() if tick <= t)
        shown.append((t, dict(plain_raw(), decision=k + 1, mask=mask, feedback=quiet(), actionNote='',
                              inventory=[dict(i, slot=s) for s, i in enumerate(inventory)],
                              player=dict(x=3096, z=3432, level=0, hp=10, maxHp=10),
                              skills=[dict(name='Woodcutting', level=level, baseLevel=level, xp=level * 1000),
                                      dict(name='Hitpoints', level=10, baseLevel=10, xp=1154)],
                              actSel=[dict(kind='loc', id=1276, name='Dead tree' if t < 1200 else 'Tree', option='Chop down',
                                           x=3100, z=3432, level=0, dx=4, dz=0, distance=4, steps=4, count=1,
                                           combatLevel=0)])))
    return shown


class Scripted:
    """A model that drops when it can, else chops (baseline.evaluate.Actor's interface)."""
    observation_space = SimpleNamespace(spaces={})

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        return np.asarray(A_DROP0 if observation['mask'][A_DROP0] else A_ACT0), None


def test_a_pass_reports_a_failed_hour_incomplete_scores_a_complete_one_and_stops_without_a_world(tmp_path, monkeypatch):
    productions = [(150, 1511), (1250, 1511), (1450, 1511)]
    hours = [Native(scripted_hour([], {0: 20}), hour_trace(), fail='read'),
             Native(scripted_hour(productions, {0: 20, 1300: 21, 4900: 22}), hour_trace(productions))]
    released = []

    class Worlds:
        def identity(self):
            return dict(engines={'pinned': dict(tree=config.ENGINE_TREES['pinned'])})

        def acquire(self, engine, timeout):
            if len(released) == len(hours):
                raise RuntimeError('world 3 failed to boot')
            return dict(id=len(released) + 1, engine=engine, tree=config.ENGINE_TREES['pinned'], block=0, boot_id='b')

        def release(self, world_id):
            released.append(world_id)

    def join(world, record):
        native = hours[world['id'] - 1]
        return native, native.control
    def setup(transport, observation, record):  # as prepare_declared's style step records before arm
        record('requested', action=A_STYLE0)
        record('observed', action=A_STYLE0, observation=observation)
        return observation
    monkeypatch.setenv('NODE_RANDOM_EVENTS', 'false')
    monkeypatch.setattr('baseline.world.make_episode', lambda transport, control, fixture, record: ResourceEpisode(
        transport, control, CFG, fixture['horizon'], 40, record, setup, goal=fixture['goal']))
    schedule = [('wait', None, 0), ('S_wood', 101, 1), ('S_wood', 101, 2), ('SH', 101, 3)]
    result = H.native_run(tmp_path / 'pass', 'sampled1', schedule, Worlds(), join,
                          policies=lambda name, seed: (Scripted(), dict(path=name)))
    failed, scored, unplayed, rest = result['hours']
    # the capture read failed at the horizon: incomplete, never scored; its capture retired and its world released
    assert failed['status'] == 'incomplete' and 'read failed' in failed['cause'] and failed['tick'] == H.NATIVE['ticks']
    assert 'logs' not in failed and hours[0].calls[-1] == 'retire' and hours[0].closed and released == [1, 2]
    assert scored['status'] == 'complete' and scored['trial_seed'] == 3101001 + 1 and scored['decisions'] == 60
    assert scored['logs_by_window'] == [1, 2, 0, 0, 0] and scored['primary'] == [1, -1, -1, -1]
    assert scored['woodcutting']['level'] == [20, 22] and scored['woodcutting']['levels_gained_by_window'] == [0, 1, 0, 0, 1]
    assert scored['logs_by_tree'] == {'Dead tree': 1, 'Tree': 2}
    assert [(e['tick'], e['response']) for e in scored['full']] == [(1300, 'drop'), (1500, 'stall')]
    assert scored['capture_events'] == H.NATIVE['ticks'] + len(productions) + 1 and scored['foreign_items'] == {}
    # no world for the third hour: the pass stops, and that hour and the rest are not played
    assert unplayed['status'] == rest['status'] == 'not_played' and 'failed to boot' in unplayed['cause']
    assert result['status'] == 'stopped' and (tmp_path / 'pass' / 'results.json').exists()


def test_the_final2_records_are_the_native_click_studys_wood_hours_on_the_final_split_only(tmp_path, monkeypatch):
    played = []
    monkeypatch.setattr(H, 'run', lambda output, **kwargs: played.append(kwargs) or dict(trials_played=[], wall_seconds=0))
    common = ['sim.hour', 'run', '--output', str(tmp_path), '--records', 'final2']
    monkeypatch.setattr('sys.argv', [*common, '--goals', 'wood', '--split', 'final', '--final'])
    H.main()
    assert (played[0]['split'], played[0]['first'], played[0]['key'], played[0]['goals']) == ('final', 6400, 173, ('wood',))
    for extra in (['--goals', 'wood', 'fish', '--split', 'final', '--final'], ['--goals', 'wood']):
        monkeypatch.setattr('sys.argv', [*common, *extra])
        with pytest.raises(SystemExit):
            H.main()
    assert len(played) == 1
