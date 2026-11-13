"""Tick delivery decides which runs are comparable; run logs stay readable (baseline/world.py)."""
import io
import json
from types import SimpleNamespace

from hypothesis import given, settings, strategies as st
import pytest

from baseline import world
from baseline.world import JsonlEvidence, delivery_report, read_jsonl


def sample(tick, ms, **extra):
    return dict(schema=1, bootId='b', tick=tick, tickRateMs=48, monotonicMs=ms, xpRate=25, **extra)


def test_real_time_needs_measured_windows_and_lockstep_is_exact_by_construction():
    slow = [sample(i * 52, i * 5000.0) for i in range(6)]  # 52 ticks per 5 s: half the promised rate
    assert not delivery_report(SimpleNamespace(samples=slow), min_windows=3)['ok']
    fast = [sample(i * 5000, i * 5000.0, lockstep=True) for i in range(6)]  # wall clock is meaningless here
    assert delivery_report(SimpleNamespace(samples=fast), min_windows=3) == dict(ok=True, mode='lockstep', windows=[])


def test_a_compressed_log_reads_back_and_a_killed_runs_log_ends_at_its_last_line(tmp_path):
    log = JsonlEvidence(tmp_path / 'run.jsonl.gz')
    for i in range(3):
        log('step', i=i)
    # What a run killed now leaves: flushed lines, no gzip trailer.
    (tmp_path / 'killed.jsonl.gz').write_bytes((tmp_path / 'run.jsonl.gz').read_bytes())
    log.close()
    assert [row['i'] for row in read_jsonl(tmp_path / 'run.jsonl.gz')] == [0, 1, 2]
    assert [row['i'] for row in read_jsonl(tmp_path / 'killed.jsonl.gz')] == [0, 1, 2]


def test_a_demo_waits_for_its_declared_spectator(monkeypatch):
    rows = []
    def status(url, timeout):
        return io.BytesIO(json.dumps(rows.pop(0)).encode())
    monkeypatch.setattr(world, 'urlopen', status)
    monkeypatch.setattr(world.time, 'sleep', lambda seconds: None)
    demo = lambda *present, declared='viewer': dict(demo=dict(spectator=dict(declared=declared, present=[dict(name=n) for n in present])))
    said = []
    rows += [demo(), demo('wd1a0'), demo('viewer')]
    assert world.wait_for_spectator('http://127.0.0.1:43610/status', 'viewer', 60, say=lambda *a, **k: said.append(a)) >= 0
    assert not rows and len(said) == 1  # the login instructions, once in the first 15 s
    rows += [demo('viewer', declared='other')]
    with pytest.raises(ValueError, match='does not declare'):
        world.wait_for_spectator('http://127.0.0.1:43610/status', 'viewer', 60)
    rows += [dict(schema=1)]  # an ordinary world declares no spectator
    with pytest.raises(ValueError, match='does not declare'):
        world.wait_for_spectator('http://127.0.0.1:43610/status', 'viewer', 60)
    rows += [demo()]
    with pytest.raises(TimeoutError):
        world.wait_for_spectator('http://127.0.0.1:43610/status', 'viewer', 0, say=lambda *a, **k: None)


@settings(max_examples=200, deadline=None)
@given(actors=st.integers(0, 3), watching=st.integers(0, 1), declared=st.booleans(), v2=st.booleans())
def test_only_a_demo_worlds_declared_spectator_is_left_out_of_its_population(actors, watching, declared, v2):
    # The status an engine gives of a world with these players: actors (the agent, or any other player) and
    # the declared spectator 'viewer' when it watches. Only a population-v2 world declares a spectator.
    declared = declared and v2
    watching = watching if declared else 0
    row = sample(7, 1.0, players=actors + watching)
    if v2:
        row.update(population='population-v2', actors=actors)
    if declared:
        row['demo'] = dict(paceMs=450, spectator=dict(declared='viewer', present=[dict(name='viewer')] * watching))
    def accepted(**options):
        try:
            return world.check_status(row, actors=1, **options) is row
        except ValueError:
            return False
    assert accepted() == (actors == 1 and not watching)  # population-v1: every player acts
    assert accepted(spectator='viewer') == (actors == 1 and declared)
    assert not accepted(spectator='guest')
    if watching:
        row['demo']['spectator']['present'] = [dict(name='guest')]
        assert not accepted(spectator='viewer')
