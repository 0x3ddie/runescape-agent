"""sim.check replay's controls: which decisions count as npc or combat ones, and the player's npc list after the
resync has moved npcs to their native places, and the route-v1 records a replay compares and counts."""
from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baseline import config
from sim import check, paths
from sim.agent import Agent
from sim.engine import World
from sim.task import GoalEpisode


def test_a_hit_only_the_simulated_player_took_is_combat():
    obs = lambda hp: dict(player=dict(hp=hp, inCombat=False))
    click = dict(family='click', kind='')
    assert not check.npc_touched(obs(20), obs(20), obs(20), click)
    assert check.npc_touched(obs(20), obs(20), obs(19), click)


def test_an_npc_the_resync_moves_into_view_is_in_the_players_npc_list(tables):
    world = World(tables, seed=1, start_tick=6)
    task = GoalEpisode(world, dict(config.TASK_CFG), 'wood')
    task.reset(0)
    p = task.agent.player
    npc = next(n for n in world.npcs if n.active and n.level == p.level and max(abs(n.x - p.x), abs(n.z - p.z)) > 15)
    world.npc_teleport(npc, p.x + 8, p.z)  # as resync moves it
    check.observe_resynced(task.agent)
    assert npc.observers == 1
    task.agent.observe_npcs()  # the game's rule drops a teleported npc for one update; the resync's moves are no teleports
    assert npc.observers == 0


# ---- the route-v1 comparison and counts --------------------------------------------------------------------
CAPTURE = Path(__file__).resolve().parents[2] / 'rl' / 'tests' / 'fixtures' / 'reset.json'


def reset_capture():
    if not CAPTURE.exists():
        pytest.skip('needs rl/tests/fixtures/reset.json, a native reset capture that is not in this repository')
    return json.loads(CAPTURE.read_text())


def route_record(kind):
    """A route-v1 record as the Python reference writes it (Agent.record): a click 4 north, or a loc ACT."""
    agent = Agent.__new__(Agent)   # record reads only the player's tile and the world's tick
    agent.player, agent.world = SimpleNamespace(x=3200, z=3200, level=0), SimpleNamespace(tick=10)
    if kind == 'click':
        route = paths.Route([(3200, 3204)], 'target')
        agent.record('click', 17, (3200, 3204), route, offset=dict(dx=0, dz=4), waypoints=[dict(x=3200, z=3204)])
    else:
        route = paths.Route([(3201, 3203), (3201, 3204)], 'reach')
        agent.record('act', 1, (3201, 3205), route, thing=dict(kind='loc', id=1276, name='Tree', option='Chop down'),
                     refused=None, waypoints=[dict(x=a, z=b) for a, b in route])
    return agent.decision_route


def changed(value):
    if isinstance(value, dict):
        return dict(value, extra=1)
    if isinstance(value, list):
        return [*value, dict(x=0, z=0)]
    if isinstance(value, str):
        return value + '!'
    return 1 if value is None else value + 1


def test_a_decision_compares_every_route_field_but_the_version_tick_and_end():
    """replay_episode's comparison after each decision (DECISION_FIELDS) finds a route that differs in any route-v1
    field but its version, its tick and its end (the last turn point), and a decision that sent nothing against one
    that sent a route; equal routes give no difference. Dropping the route comparison, or one of its fields, fails."""
    for kind in ('click', 'act'):
        native = dict(reset_capture(), route=route_record(kind))
        assert check.compare(native, copy.deepcopy(native), check.DECISION_FIELDS) == {}
        compared = set(native['route']) - {'version', 'tick', 'end'}
        assert compared == set(check.ROUTE_FIELDS)
        for field in sorted(compared):
            sim = copy.deepcopy(native)
            sim['route'][field] = changed(sim['route'][field])
            assert set(check.compare(native, sim, check.DECISION_FIELDS)) == {'route'}, (kind, field)
        assert set(check.compare(native, dict(copy.deepcopy(native), route=None), check.DECISION_FIELDS)) == {'route'}


def test_replay_counts_the_native_route_records_by_kind_and_ending(tmp_path, monkeypatch):
    """replay's output counts the native route records of the replayed decisions by kind and ending (route_outcome),
    summed over its episodes, beside the mismatches."""
    click, act = route_record('click'), route_record('act')
    refused = dict(act, via='none', end=None, refused='no route', waypoints=[])
    assert [check.route_outcome(r) for r in (click, act, refused, None)] == ['click target', 'act reach', 'act no route',
                                                                            None]
    for name in ('a', 'b'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'trajectory.jsonl.gz').write_bytes(b'')
    episodes = {'a': [click, click, act], 'b': [refused, None]}

    def replay_episode(tables, episode, seed):
        records = episodes[episode['name']]
        return dict(decisions=len(records), replayed=len(records), mismatches=Counter(), npc_mismatches=Counter(),
                    routes=Counter(o for o in map(check.route_outcome, records) if o), stop=None)
    monkeypatch.setattr(check.Tables, 'load', staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(check, 'native_episodes', lambda path: [dict(start={}, name=path.parent.name)])
    monkeypatch.setattr(check, 'replay_episode', replay_episode)
    check.replay(tmp_path, '*', out=tmp_path / 'replay.json')
    out = json.loads((tmp_path / 'replay.json').read_text())
    assert out['routes'] == {'act no route': 1, 'act reach': 1, 'click target': 2}
    assert [e['routes'] for e in out['episodes']] == [{'click target': 2, 'act reach': 1}, {'act no route': 1}]
