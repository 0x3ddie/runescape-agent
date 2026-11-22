"""sim/hour.py, hour-v1: one character persists past the task horizon for the whole hour under the declared
clock, death ends the hour at its tick, an unsupported step stops the trial unscored, the XP rate is the world's, and
inventory-full responses and the merged parts are accounted as declared."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim import hour as H
from sim.agent import A_WAIT
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
