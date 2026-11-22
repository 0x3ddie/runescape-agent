"""free-play-v1 in the Python reference: XP events with causes and their independent
check, the fractional level pay, obs-v6, the life (its own start, limits and hidden length), village-world-v1 Stage A,
the batch's seam with the learner, the controls and the soak. The existing tasks keep their outputs."""
import hashlib
import json
from types import SimpleNamespace

from hypothesis import given, strategies as st
import numpy as np
import pytest

from baseline import config
from baseline.resource_task import MAX_HORIZON
from obs import LEVEL_XP as CLIENT_LEVEL_XP, project_v6
from sim import free, world as W
from sim.audit import AuditError, XpCheck, check_stats, fractional_level
from sim.agent import A_DROP0, A_HELD0, A_INV_NEXT, A_STYLE0
from sim.combat import VILLAGE, draw_village, kit, outside, village_life, village_npcs
from sim.engine import (ATTACK, DEFENCE, FISHING, HITPOINTS, LEVEL_XP, MINING, PRAYER, SKILLS, STRENGTH, WOODCUTTING,
                        Rng, Unsupported, World, level_for, weapon_styles, xp_for, WEARPOS_RHAND)
from sim.spec import (OBSERVATION_SHAPES, OBSERVATION_SHAPES_V6, OBSERVATIONS, TERMINAL_DEATH, TERMINAL_HORIZON,
                      check_batch, dtype)
from sim.task import FREE_PLAY, WORLD_START_TICK, FreePlayEpisode, draw_life, free_play_cfg
from sim.vec import CpuBatch
from sim.world import ORIGIN, PEN_GATE, LayoutTables

KEY = 7


# ---- fractional levels and the XP check (no tables) ---------------------------------------------------------------
XP = st.integers(0, LEVEL_XP[-1] + 50_000)


@given(XP)
def test_the_fractional_level_lies_between_its_level_and_the_next(x):
    level, ell = level_for(x), fractional_level(x)
    assert level <= ell < level + 1 or (level == 99 and ell == 99.0)
    if level < 99:
        assert fractional_level(xp_for(level)) == level and fractional_level(xp_for(level + 1)) == level + 1


# S, the paid skills, and the paid level computed apart from sim/audit.py: from the client's copy of the level table
S = ('woodcutting', 'mining', 'fishing', 'attack', 'strength', 'defence', 'hitpoints')


def ell(x):
    """The paid level of XP x (tenths): L + (x - X(L)) / (X(L+1) - X(L)), capped at 99, from rl/obs.py LEVEL_XP."""
    level = min(99, next((k + 2 for k in range(98, -1, -1) if x >= CLIENT_LEVEL_XP[k]), 1))
    if level == 99:
        return 99.0
    low = CLIENT_LEVEL_XP[level - 2] if level >= 2 else 0
    return level + (x - low) / (CLIENT_LEVEL_XP[level - 1] - low)


def pay_of(before, after):
    """A step's pay from the stats before and after it: the change of ell over S only."""
    return sum(ell(after[SKILLS.index(s)]) - ell(before[SKILLS.index(s)]) for s in S)


TREE, ROCK, LOGS, ORE, FISH, BONES, NPC = 1276, 2091, 1511, 436, 317, 526, 41
TABLES = SimpleNamespace(
    loc_types={TREE: dict(debugname='tree'), ROCK: dict(debugname='copperrock1')},
    obj_types={LOGS: dict(debugname='logs', params={}), ORE: dict(debugname='copper_ore', params={}),
               FISH: dict(debugname='raw_shrimp', params={}), BONES: dict(debugname='bones', params=dict(bone_exp=45))},
    npc_types={NPC: dict(params=dict(combat_xp_multiplier=1000))},
    meta=dict(param_defaults=dict(bone_exp=0, productexp=0, combat_xp_multiplier=1000),
              content=dict(trees=dict(tree=dict(product='logs', productexp=250)),
                           compact=dict(rocks=dict(copperrock1=dict(output='copper_ore', exp=175))),
                           fish=dict(raw_shrimp=dict(productexp=100)), cooking=dict(rows={}))),
    obj_id=lambda name: dict(logs=LOGS, copper_ore=ORE, raw_shrimp=FISH)[name])
CAUSES = st.lists(st.tuples(st.sampled_from(['wood', 'mine', 'fish', 'bury', 'hit0', 'hit1', 'hit2', 'hit3']),
                            st.integers(0, 3), st.booleans()), max_size=25)


def life_log(causes, rate):
    """Events as the engine writes them for a run of causes (cause, damage, cause first), with the true amounts."""
    stats, log, tick = [0] * len(SKILLS), [], 0

    def xp(stat, amount, **cause):
        before = stats[stat]
        stats[stat] = min(before + amount * rate, 2_000_000_000)
        return (tick, 'xp', dict(stat=stat, xp=amount, before=before, after=stats[stat], **cause))
    for kind, damage, first in causes:
        tick += 1
        if kind in ('wood', 'mine', 'fish'):
            obj, script, stat, src, amount = dict(
                wood=(LOGS, '[label,get_logs]', WOODCUTTING, TREE, 250), mine=(ORE, '[label,get_ore_normal]', MINING, ROCK, 175),
                fish=(FISH, '[proc,fish_roll]', FISHING, None, 100))[kind]
            made = (tick, 'produce', dict(obj=obj, count=1, script=script))
            gain = xp(stat, amount, cause='produce', obj=obj, src=src)
            log += [gain, made] if first else [made, gain]  # woodcutting adds its XP before the logs
        elif kind == 'bury':
            log += [(tick, 'remove', dict(obj=BONES, count=1, why='bury')), xp(PRAYER, 45, cause='bury', obj=BONES)]
        elif damage:
            style = int(kind[3])
            log.append((tick, 'hit', dict(npc=3, type=NPC, life=0, source='me', damage=damage + 1, hp=damage, style=style)))
            stats_of = {0: [ATTACK], 1: [STRENGTH], 2: [DEFENCE], 3: [ATTACK, STRENGTH, DEFENCE]}[style]
            for stat in stats_of:
                log.append(xp(stat, damage * 10 * (133 if style == 3 else 400) // 100, cause='hit', npc=3, life=0))
            log.append(xp(HITPOINTS, damage * 10 * 133 // 100, cause='hit', npc=3, life=0))
    return log, stats


@given(CAUSES, st.sampled_from([1, 25]))
def test_the_xp_check_accounts_for_every_xp_event_with_its_cause(causes, rate):
    log, stats = life_log(causes, rate)
    check = XpCheck(TABLES, rate, 'me', [0] * len(SKILLS))
    half = next((k for k in range(len(log) // 2, len(log)) if k == 0 or log[k][0] != log[k - 1][0]), len(log))
    gained = [a + b for a, b in zip(check.observe(log[:half]), check.observe(log[half:]))]  # a step at a time
    assert check.xp == stats and gained == stats
    check_stats([0] * len(SKILLS), stats, gained)


@given(CAUSES.filter(lambda c: any(k != 'bury' and (d or k[:3] != 'hit') for k, d, _ in c)), st.sampled_from([1, 25]),
       st.sampled_from(['amount', 'cause', 'rate', 'chain', 'stat', 'missing']), st.randoms())
def test_an_xp_event_without_its_cause_or_amount_stops_the_run(causes, rate, fault, random):
    log, _ = life_log(causes, rate)
    k = random.choice([k for k, e in enumerate(log) if e[1] == 'xp' and e[2]['cause'] != 'bury'])
    tick, kind, data = log[k]
    data = dict(data)
    if fault == 'amount':
        data.update(xp=data['xp'] + 1, after=data['after'] + rate)
    elif fault == 'cause':
        log = [e for e in log if not (e[0] == tick and e[1] in ('produce', 'hit'))]
        k = next(j for j, e in enumerate(log) if e[0] == tick and e[1] == 'xp')
        data = dict(log[k][2])
    elif fault == 'rate':
        rate = 26 - rate
    elif fault == 'chain':
        data.update(before=data['before'] + 10, after=data['after'] + 10)
    elif fault == 'missing':  # the last XP of a stat never came: no chain breaks, only its cause is left without it
        k = max(j for j, e in enumerate(log) if e[1] == 'xp' and e[2]['stat'] == data['stat'])
        del log[k]
        with pytest.raises(AuditError):
            XpCheck(TABLES, rate, 'me', [0] * len(SKILLS)).observe(log)
        return
    else:
        data['stat'] = PRAYER
    log[k] = (tick, kind, data)
    with pytest.raises(AuditError):
        XpCheck(TABLES, rate, 'me', [0] * len(SKILLS)).observe(log)


def test_the_xp_check_is_two_sided_and_owes_nothing_for_a_gem():
    """A cause that gives XP must have it in its tick (an engine that drops XP pays too little): a log with no XP, a hit
    with attack XP and no hitpoints XP, bones buried with no XP all stop the run. A get_ore production of something that
    is not an ore (a gem) owes none."""
    hit = (1, 'hit', dict(npc=3, type=NPC, life=0, source='me', damage=3, hp=3, style=0))
    attack = (1, 'xp', dict(stat=ATTACK, xp=120, before=0, after=120, cause='hit', npc=3, life=0))
    for log in ([(1, 'produce', dict(obj=LOGS, count=1, script='[label,get_logs]'))], [hit, attack],
                [(1, 'remove', dict(obj=BONES, count=1, why='bury'))]):
        with pytest.raises(AuditError):
            XpCheck(TABLES, 1, 'me', [0] * len(SKILLS)).observe(log)
    hp = (1, 'xp', dict(stat=HITPOINTS, xp=39, before=0, after=39, cause='hit', npc=3, life=0))
    assert XpCheck(TABLES, 1, 'me', [0] * len(SKILLS)).observe([hit, attack, hp])[HITPOINTS] == 39
    gem = (1, 'produce', dict(obj=1623, count=1, script='[label,get_ore_normal]'))
    assert not any(XpCheck(TABLES, 1, 'me', [0] * len(SKILLS)).observe([gem]))


def test_a_hit_in_its_npc_lifes_respawn_cycle_owes_no_xp_and_may_have_none():
    """A hit on an npc life in the tick of that life's 'npc_respawn' event owes no XP, and an XP event for it
    stops the run. The check takes the cycle from the event's tick alone: a hit one tick later owes the type's XP as
    before (an engine that left the multiplier at 0 one cycle more is caught), and a second respawn of the same life
    stops the run."""
    respawn = (5, 'npc_respawn', dict(npc=3, type=NPC, life=1))
    hit = lambda tick: (tick, 'hit', dict(npc=3, type=NPC, life=1, source='me', damage=3, hp=3, style=0))
    xp = lambda stat, amount: (5, 'xp', dict(stat=stat, xp=amount, before=0, after=amount, cause='hit', npc=3, life=1))
    assert not any(XpCheck(TABLES, 1, 'me', [0] * len(SKILLS)).observe([respawn, hit(5)]))
    check = XpCheck(TABLES, 1, 'me', [0] * len(SKILLS))
    check.observe([respawn])
    with pytest.raises(AuditError, match='has no xp event'):
        check.observe([hit(6)])  # the next step's cycle
    with pytest.raises(AuditError, match="respawn cycle gives no XP"):
        XpCheck(TABLES, 1, 'me', [0] * len(SKILLS)).observe([respawn, hit(5), xp(ATTACK, 120), xp(HITPOINTS, 39)])
    check = XpCheck(TABLES, 1, 'me', [0] * len(SKILLS))
    check.observe([respawn, hit(5)])
    with pytest.raises(AuditError, match='a second npc_respawn'):
        check.observe([(9, 'npc_respawn', dict(npc=3, type=NPC, life=1))])


def test_the_bookkeeping_check_refuses_xp_no_event_explains():
    with pytest.raises(AuditError):
        check_stats([0, 0], [10, 0], [0, 0])


def test_the_client_level_table_is_the_servers():
    assert CLIENT_LEVEL_XP == LEVEL_XP


# ---- lives in the Python reference (private tables) ---------------------------------------------------------------
@pytest.fixture(scope='module')
def pool(tables):
    return [W.generate(seed, tables) for seed in W.split_seeds('development', 4)]


def life(tables, pool, record=0, muggers=False, length=600, rate=1, cfg=None):
    layout = pool[record % len(pool)]
    start = draw_village(KEY, 0, record, [layout], tables, muggers, layout=0)
    return village_life(start, layout, LayoutTables(tables, layout), tables, rate, length, cfg)


def play(task, actor, seed=0):
    actor.reset(seed)
    obs, rows = task.projected, []
    while True:
        obs, reward, done, info = task.step(int(actor.act(obs, task)))
        rows.append((obs, reward, done, info))
        if done:
            return rows


def test_stage_a_is_the_melee_world_with_the_layouts_chickens_added(tables, pool):
    from sim.combat import combat_npcs
    names = lambda rows: [tables.npc_types[r['type']]['debugname'] for r in rows]
    for layout in pool:
        hens = sum(tables.npc_types[int(ty)]['debugname'] in W.CHICKENS for _, ty, _, _ in layout.npcs)
        melee, village = combat_npcs(layout, tables, [(1, 1)]), village_npcs(layout, tables, [(1, 1)])
        assert village[:len(melee)] == melee and len(village) == len(melee) + hens
        assert [r['nid'] for r in village] == list(range(len(village))) and names(village).count('goblin') == hens
        assert sum(n in W.CHICKENS for n in names(village)) == hens and names(village).count('mugger') == 1
        a, b = (draw_village(KEY, 0, 3, [layout], tables, m, layout=0) for m in (True, False))
        assert a['village']['muggers'] and not b['village']['muggers']
        assert {k: v for k, v in a.items() if k != 'village'} == {k: v for k, v in b.items() if k != 'village'}
        assert {k: v for k, v in a['village'].items() if k not in ('muggers', 'placed')} == \
            {k: v for k, v in b['village'].items() if k not in ('muggers', 'placed')}
    with pytest.raises(ValueError):
        draw_village(KEY, 0, 0, pool, tables, None)


def test_each_step_pays_its_level_change_over_s_only_and_a_bury_pays_nothing(tables, pool):
    """S is the seven paid skills. A life with bones in its kit buries them (prayer XP, checked and paid 0), then
    gathers and fights (XP in S): each step pays the change of the paid level over S, computed here from the client's
    level table, and the life's pay is the gain over S of its XP events, which the check accepted as it went."""
    assert FREE_PLAY['skills'] == S
    layout = pool[1]
    start = draw_village(KEY, 0, 1, [layout], tables, False, layout=0)
    v = start['village']
    world = World(LayoutTables(tables, layout), seed=start['world_seed'], start_tick=WORLD_START_TICK,
                  npcs=village_npcs(layout, tables, []), exclusions=VILLAGE['exclusions'], xp_rate=25)
    task = FreePlayEpisode(world, 1500, 25)
    task.start(start['tile'], start['weapon'], start['levels'], kit=kit(start['weapon'], ['bones', 'bones']), hp=v['hp'],
               wield=v['worn'])
    p, paid, buried = task.agent.player, 0.0, 0
    arm = list(p.stats)
    actor = free.make('rotation', period=300)
    actor.reset([5, 1])
    while True:
        obs = task.projected
        bury = next((A_HELD0 + k for k, h in enumerate(task.raw['heldSel']) if h['option'] == 'Bury' and obs['mask'][A_HELD0 + k]),
                    None)
        before = list(p.stats)
        _, reward, done, info = task.step(bury if bury is not None else int(actor.act(obs, task)))
        assert reward == pytest.approx(pay_of(before, p.stats), abs=1e-9)
        if bury is not None and p.stats[PRAYER] > before[PRAYER]:
            buried += 1
            assert reward == 0 and all(p.stats[SKILLS.index(s)] == before[SKILLS.index(s)] for s in S)
        paid += reward
        if done:
            break
    gained = [0] * len(SKILLS)
    for _, kind, data in task.world.events[task.event_start:]:
        if kind == 'xp':
            gained[data['stat']] += data['after'] - data['before']
    events = sum(ell(arm[SKILLS.index(s)] + gained[SKILLS.index(s)]) - ell(arm[SKILLS.index(s)]) for s in S)
    assert buried == 2 and gained[PRAYER] > 0 and paid == pytest.approx(events, abs=1e-9) and paid > 0
    from obs import SKILL_ORDER  # rb2-3: prayer, outside S, shows its progress from its whole XP
    k, level = 25 + SKILL_ORDER.index('Prayer'), p.base_levels[PRAYER]
    want = ell(p.stats[PRAYER] // 10 * 10) - level
    assert 0 < want < 1 and task.projected['prog_vec'][k] == pytest.approx(want, abs=1e-6)
    assert info['report']['paid'] == pytest.approx(paid) and abs(paid - info['report']['gain_total']) < len(S)
    world = World(LayoutTables(tables, layout), seed=1, start_tick=WORLD_START_TICK, xp_rate=25)
    for rate in (1, 3):  # the declared rate must be one of the two and the world's
        with pytest.raises(ValueError):
            FreePlayEpisode(world, 600, rate)


def test_an_xp_gain_without_its_cause_or_without_an_event_stops_the_run(tables, pool):
    task = life(tables, pool)
    p, t = task.agent.player, tables
    tree = next(k for k, v in t.loc_types.items() if v['debugname'] == 'tree')
    task.world.add_xp(p, WOODCUTTING, 250, dict(cause='produce', obj=t.obj_id('logs'), src=tree))  # no logs made
    with pytest.raises(AuditError):
        task.step(0)
    task = life(tables, pool)
    task.agent.player.stats[MINING] += 10  # XP no event explains
    with pytest.raises(AuditError):
        task.step(0)


def test_a_life_of_36000_ticks_is_cut_or_changed_by_no_agent_limit(tables, pool):
    """Under TASK_CFG, truncated() would stop the decision-point wait after 12,000 ticks or 4,000 decisions (step A); the
    life passes its own limits, which give the same life as no limit at all."""
    assert free_play_cfg() == dict(config.TASK_CFG, maxDecisions=MAX_HORIZON, maxEpisodeTicks=2 * MAX_HORIZON)
    assert (config.TASK_CFG['maxDecisions'], config.TASK_CFG['maxEpisodeTicks']) == (4000, 12000)
    with pytest.raises(ValueError):
        life(tables, pool, length=36000, cfg=config.TASK_CFG)
    lives = [play(life(tables, pool, record=2, length=36000, cfg=cfg), free.make('wood'))  # nothing attacks it
             for cfg in (None, dict(config.TASK_CFG, maxDecisions=10 ** 9, maxEpisodeTicks=10 ** 9))]
    assert len(lives[0]) == len(lives[1])
    for (o1, r1, d1, _), (o2, r2, d2, _) in zip(*lives):
        assert r1 == r2 and d1 == d2 and all(np.array_equal(o1[k], o2[k]) for k in o1)
    report = lives[0][-1][3]['report']
    assert report['ticks'] == 36000 > config.TASK_CFG['maxEpisodeTicks'] and report['reason'] == 'horizon'


def test_no_obs_v6_array_changes_with_the_drawn_length_or_the_time_of_a_life(tables, pool):
    short, long_ = (play(life(tables, pool, record=3, length=n, rate=25), free.make('random'), [4, 3]) for n in (900, 1800))
    assert len(long_) > len(short)
    for (o1, _, _, _), (o2, _, _, _) in zip(short[:-1], long_):  # the short life's last step stops at its end tick
        assert set(o1) == set(OBSERVATION_SHAPES_V6) and all(np.array_equal(o1[k], o2[k]) for k in o1)
    task = life(tables, pool)
    raw = dict(task.raw, decision=1234, tick=task.raw['tick'] + 30000)
    assert all(np.array_equal(project_v6(raw)[k], task.projected[k]) for k in task.projected)
    assert draw_village(KEY, 0, 5, pool, tables, False) == draw_village(KEY, 0, 5, pool, tables, False)
    lengths = [draw_life(KEY, 0, e, ('uniform', 6000, 36000)) for e in range(200)]  # the proposal
    assert min(lengths) >= 6000 and max(lengths) <= 36000 and len(set(lengths)) > 150
    assert draw_life(KEY, 0, 0, ('fixed', 36000)) == 36000


def test_obs_v6_has_each_skills_progress_and_no_goal_or_time_input(tables, pool):
    assert set(OBSERVATION_SHAPES) - set(OBSERVATION_SHAPES_V6) == {'goal_vec', 'budget_vec', 'critic_vec'}
    assert OBSERVATION_SHAPES_V6['prog_vec'] == (46,) and OBSERVATION_SHAPES_V6['task_vec'] == (6,)
    assert OBSERVATIONS == {'obs-v5': OBSERVATION_SHAPES, 'obs-v6': OBSERVATION_SHAPES_V6}
    from obs import SKILL_ORDER
    task = life(tables, pool, record=1, length=1500, rate=25)
    p, actor, between = task.agent.player, free.make('rotation', period=300), set()
    actor.reset([5, 1])
    while True:  # every step, each skill of S: the progress from its whole XP, from the client's level table
        for s in S:
            stat, k = SKILLS.index(s), 25 + SKILL_ORDER.index(s.capitalize())
            want = ell(p.stats[stat] // 10 * 10) - p.base_levels[stat]
            assert task.projected['prog_vec'][k] == pytest.approx(want, abs=1e-6)
            between |= {s} if 0 < want < 1 else set()
        _, _, done, _ = task.step(int(actor.act(task.projected, task)))
        if done:
            break
    assert between == set(S)
    with pytest.raises(ValueError):
        project_v6(dict(task.raw, skills=[{k: v for k, v in s.items() if k != 'xp'} for s in task.raw['skills']]))


# ---- the batch: sim/spec.py's interface for the learner (the seam) ----------------------------------------------
def random_steps(env, key, steps):
    state, obs = env.reset(key)
    rng = Rng(key)
    for step in range(steps):
        legal = [np.flatnonzero(obs['mask'][b]) for b in range(env.n)]
        action = np.asarray([legal[b][int(rng(9, step, b) * len(legal[b]))] for b in range(env.n)], np.int32)
        before = list(env.tasks)
        state, obs, reward, done, info = env.step(state, action)
        yield before, obs, reward, done, info


def test_the_free_play_batch_pays_each_life_its_level_gain_and_gives_its_last_observation(tables, pool):
    env = CpuBatch(tables, pool, 4, task='free-play-v1', muggers=False, life=('fixed', 300))
    assert env.observation == 'obs-v6' and env.shapes is OBSERVATION_SHAPES_V6
    paid, ended = np.zeros(4), 0
    for before, obs, reward, done, info in random_steps(env, 3, 160):
        check_batch(obs, 4, OBSERVATION_SHAPES_V6)
        last = info['last_obs']
        assert set(last) == set(OBSERVATION_SHAPES_V6) and (info['goal'] == -1).all() and not info['unsupported'].any()
        assert all(last[k].shape == (4, *s) and last[k].dtype == dtype(k) for k, s in OBSERVATION_SHAPES_V6.items())
        paid += reward
        for b in range(4):
            if done[b]:
                assert all(np.array_equal(last[k][b], before[b].projected[k]) for k in last)  # after the last action
                assert env.tasks[b] is not before[b] and env.tasks[b].decisions == 0  # obs: the next life's first
                assert all(np.array_equal(obs[k][b], env.tasks[b].projected[k]) for k in obs)
                assert info['terminal'][b] == TERMINAL_HORIZON and info['episode_ticks'][b] == 300
                assert abs(paid[b] - info['episode_return'][b]) < 1e-4
                assert abs(info['episode_return'][b] - info['outcomes'][b]) < len(FREE_PLAY['skills'])
                assert info['decision_ticks'][b] >= 0
                paid[b], ended = 0, ended + 1
            else:
                assert not any(last[k][b].any() for k in last)
    assert ended >= 4


def test_a_death_ends_a_life_and_takes_nothing_back(tables, pool):
    """Random legal play with muggers dies; each life's pay is its level gain, never negative, and a death settles
    nothing: the life's return is what it was paid."""
    env = CpuBatch(tables, pool, 4, task='free-play-v1', muggers=True, xp_rate=25, life=('fixed', 6000))
    paid, deaths = np.zeros(4), 0
    for _, obs, reward, done, info in random_steps(env, 5, 300):
        assert (reward >= 0).all()
        paid += reward
        for b in np.flatnonzero(done):
            assert info['terminal'][b] == TERMINAL_DEATH and info['episode_ticks'][b] < 6000
            assert info['episode_return'][b] == pytest.approx(paid[b], abs=1e-4) and paid[b] > 0
            assert env.tasks[b].decisions == 0 and env.tasks[b].terminal is None  # the next life
            paid[b], deaths = 0, deaths + 1
    assert deaths >= 2


def test_an_unsupported_free_play_world_is_flagged_never_done_and_refused(tables, pool, monkeypatch):
    env = CpuBatch(tables, pool, 2, task='free-play-v1', muggers=False, life=('fixed', 600))
    state, obs = env.reset(1)

    def uncovered(action):
        raise Unsupported('engine script x')
    monkeypatch.setattr(env.tasks[1].agent, 'step', uncovered)
    state, obs, reward, done, info = env.step(state, np.zeros(2, np.int32))
    assert info['unsupported'].tolist() == [False, True] and not done[1] and reward[1] == 0
    assert env.evidence[-1]['reason'] == 'engine script x' and not info['last_obs']['mask'][1].any()
    with pytest.raises(RuntimeError):
        env.step(state, np.zeros(2, np.int32))


def test_an_unsupported_setup_keeps_its_evidence_and_is_raised_on_both_paths(tables, pool, monkeypatch):
    """No transition exists for a setup the simulator cannot model: the batch keeps the start as far as it was drawn and
    raises, in free play and in a goal task alike."""
    import sim.vec

    def refused(*args, **kwargs):
        raise Unsupported('spawn outside the supported area')
    monkeypatch.setattr(sim.vec, 'village_life', refused)
    env = CpuBatch(tables, pool, 2, task='free-play-v1', muggers=False, life=('fixed', 600))
    with pytest.raises(Unsupported):
        env.reset(4)
    row = env.evidence[-1]
    assert row['setup'] and row['episode'] == 0 and 'village' in row['start'] and row['start']['length'] == 600
    assert row['seed'] == pool[row['start']['layout']].seed and row['reason'] == 'spawn outside the supported area'
    monkeypatch.setattr(sim.vec.GoalEpisode, 'start', refused)
    env = CpuBatch(tables, pool, 4)
    with pytest.raises(Unsupported):
        env.reset(4)
    assert env.evidence[-1]['setup'] and env.evidence[-1]['start']['layout'] in range(len(pool))


def test_free_play_batches_are_keyed_and_refuse_goal_settings(tables, pool):
    first, second = (list(random_steps(CpuBatch(tables, pool, 2, task='free-play-v1', muggers=True, life=('fixed', 900)),
                                       11, 30)) for _ in range(2))
    for (_, o1, r1, d1, _), (_, o2, r2, d2, _) in zip(first, second):
        assert all(np.array_equal(o1[k], o2[k]) for k in o1) and np.array_equal(r1, r2) and np.array_equal(d1, d2)
    ok = dict(muggers=False, life=('fixed', 600))
    for bad in (dict(), dict(muggers=False), dict(life=('fixed', 600)), dict(ok, muggers=1), dict(ok, horizon=40),
                dict(ok, goals=('wood',)), dict(ok, xp_rate=3), dict(ok, life=('uniform', 6000))):
        with pytest.raises(ValueError):
            CpuBatch(tables, pool, 2, task='free-play-v1', **bad)
    with pytest.raises(ValueError):
        CpuBatch(tables, pool, 4, xp_rate=1)  # the XP rate, muggers and life are free-play-v1's settings


def test_the_goal_tasks_give_the_outputs_of_the_base(tables):
    """goal-activity-v2 and v3 on fixed seeds give the observations, rewards, done flags and info of the
    commit before the XP events, under the tests' name table (conftest.py): the events add evidence only."""
    pool = [W.generate(seed, tables) for seed in W.split_seeds('development', 6)]
    want = {'goal-activity-v2': (150, '1f04b9a1ab89471b263489cd7bad678a01e9d68940818822e2a2593736e5455f'),
            'goal-activity-v3': (400, '18ed08ba0c189c7dad1ed653d5aef0827e0648d7f16045d12992276af26cb8ad')}
    for task, (steps, digest) in want.items():
        env, h = CpuBatch(tables, pool, 8, task=task), hashlib.sha256()
        state, obs = env.reset(29)
        names = sorted(obs)
        for k in names:
            h.update(np.ascontiguousarray(obs[k]).tobytes())
        rng = Rng(29)
        for step in range(steps):
            legal = [np.flatnonzero(obs['mask'][b]) for b in range(env.n)]
            action = np.asarray([legal[b][int(rng(9, step, b) * len(legal[b]))] for b in range(env.n)], np.int32)
            state, obs, reward, done, info = env.step(state, action)
            for k in names:
                h.update(np.ascontiguousarray(obs[k]).tobytes())
            h.update(np.ascontiguousarray(reward).tobytes())
            h.update(np.ascontiguousarray(done).tobytes())
            for k in sorted(info):
                h.update(k.encode())
                h.update(np.ascontiguousarray(info[k]).tobytes())
        assert h.hexdigest() == digest, task


# ---- the controls and the soak (sim/free.py): yardsticks, never labels ----------------------------------------------
@pytest.mark.parametrize('name', free.CONTROLS)
def test_each_control_plays_a_life_of_legal_actions_in_both_forms(tables, pool, name):
    for muggers in (False, True):
        task = life(tables, pool, record=1, muggers=muggers, length=500, rate=25)
        rows = play(task, free.make(name, period=100), [3, 1])  # FreePlayEpisode.step refuses a masked action
        assert rows[-1][2] and rows[-1][3]['report']['reason'] in ('horizon', 'death')
    for bad in ('wood+drop', 'melee+lowest', 'rotation+'):  # one form each: the accepted one
        with pytest.raises(ValueError):
            free.make(bad)


def test_a_dropping_gatherer_drops_its_own_product_turning_pages(tables, pool):
    task = life(tables, pool)
    p, t = task.agent.player, tables
    bones, logs = t.obj_id('bones'), t.obj_id('logs')
    while p.inv.freespace() and len(p.inv.listed()) < 16:
        p.inv.add(t, bones, 1)
    while p.inv.freespace():
        p.inv.add(t, logs, 1)  # logs from slot 16, page 2
    family = {logs}
    task.step(0)
    assert free.drop_own(task, family) == A_INV_NEXT and task.step(A_INV_NEXT) and free.drop_own(task, family) == A_INV_NEXT
    task.step(A_INV_NEXT)
    action = free.drop_own(task, family)
    assert A_DROP0 <= action < A_DROP0 + 8 and task.raw['invSel'][action - A_DROP0]['id'] == logs
    task.step(action)
    assert p.inv.freespace() == 1 and task.raw['invPage'] == 2 and free.drop_own(task, family) is None  # no turn back


def worn_life(tables, pool):
    """A life whose drawn start wears the weapon (a style row can then be set)."""
    record = next(r for r in range(40) if draw_village(KEY, 0, r, [pool[r % 4]], tables, False, layout=0)['village']['worn'])
    return life(tables, pool, record=record)


def test_the_study_references_set_the_lowest_style_before_the_next_fight(tables, pool):
    """Items 3 and the exploit's definition: the melee reference and the chickens control themselves pick the style row
    that trains the lowest of attack, strength and defence when no fight is on."""
    for name in ('melee', 'chickens'):
        task = worn_life(tables, pool)
        p = task.agent.player
        p.base_levels[ATTACK], p.base_levels[STRENGTH], p.base_levels[DEFENCE] = 5, 3, 9
        task.step(0)
        control = free.make(name)
        control.reset(0)
        action = control.act(task.projected, task)
        styles = weapon_styles(task.world, p.worn.items[WEARPOS_RHAND][0])['damagestyle']
        assert A_STYLE0 <= action < A_STYLE0 + len(styles) and styles[action - A_STYLE0] == 1, name  # strength
        task.step(action)
        assert p.varps['com_mode'] == action - A_STYLE0 and control.act(task.projected, task) not in range(A_STYLE0, A_STYLE0 + 4)


def test_the_study_gatherers_drop_and_a_rotation_gatherer_drops_another_members_product(tables, pool):
    """Item 7: a gatherer with a full inventory drops one of its own product. In the rotation a gatherer drops any
    gatherer's product (a full inventory of logs would stop the fish phase); the fish reference alone does not."""
    task = life(tables, pool)
    p, logs = task.agent.player, tables.obj_id('logs')
    while p.inv.freespace():
        p.inv.add(tables, logs, 1)
    task.step(0)
    rotation = free.make('rotation', period=100)
    rotation.reset(0)
    for control in (free.make('wood'), rotation.members[rotation.order.index('fish')]):
        action = control.act(task.projected, task)
        assert A_DROP0 <= action < A_DROP0 + 8 and task.raw['invSel'][action - A_DROP0]['id'] == logs
    action = free.make('fish').act(task.projected, task)
    assert not A_DROP0 <= action < A_DROP0 + 8


@pytest.fixture(scope='module')
def pen(tables):
    """A development layout with a chicken pen, and an inside tile beside an outside one across the fence."""
    for seed in W.split_seeds('development', 40):
        layout = W.generate(seed, tables)
        if layout.gate[0] < 0:
            continue
        out = outside(layout)
        inside = layout.walkable & ~out
        for x, z in np.argwhere(inside):
            near = [(x + dx, z + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1) if (dx or dz)
                    and 0 <= x + dx < out.shape[0] and 0 <= z + dz < out.shape[1] and out[x + dx, z + dz]]
            if near:
                return layout, (int(x), int(z)), tuple(int(v) for v in near[0])
    raise AssertionError('no pen layout in 40 development seeds')


def in_pen_life(tables, pen):
    """A life on the pen layout, the player moved inside the pen with its gate closed (the boot state)."""
    layout, (x, z), _ = pen
    start = draw_village(KEY, 0, 0, [layout], tables, False, layout=0)
    task = village_life(start, layout, LayoutTables(tables, layout), tables, 25, 3000)
    task.world.teleport(task.agent.player, ORIGIN[0] + x, ORIGIN[1] + z)
    task.step(0)
    assert free.in_pen(task, {}) and any(loc.active and tables.loc_types[loc.type]['debugname'] in PEN_GATE
                                         for loc in task.world.locs)
    return task


def test_a_rotation_gatherer_leaves_a_closed_pen_and_gathers(tables, pen):
    """A melee phase can end inside a pen whose gate closes again; reference_action opens a pen gate only for its
    melee goal. The rotation's gatherer opens the gate, walks out and gathers."""
    task = in_pen_life(tables, pen)
    rotation = free.make('rotation', period=10 ** 6)
    rotation.reset(0)
    wood, memo, before = rotation.members[rotation.order.index('wood')], {}, task.agent.player.stats[WOODCUTTING]
    for _ in range(400):
        task.step(int(wood.act(task.projected, task)))
        if task.agent.player.stats[WOODCUTTING] > before:
            break
    assert not free.in_pen(task, memo) and task.agent.player.stats[WOODCUTTING] > before


def test_an_attacker_across_a_closed_pen_fence_is_not_chased_and_one_in_reach_is(tables, pen):
    """An npc can keep the player as its target from across a closed gate's fence, where no hit lands; an offered
    Attack does not show that. The fighter, the chickens control and a rotation gatherer leave it and go for the gate;
    an attacker inside the pen with the player is in reach."""
    from sim.combat import Reference
    task = in_pen_life(tables, pen)
    w, p, (_, _, (gx, gz)) = task.world, task.agent.player, pen
    goblins = [n for n in w.npcs if n.is_valid() and tables.npc_types[n.type]['debugname'] == 'goblin']
    far, near = goblins[0], goblins[1] if len(goblins) > 1 else None
    w.npc_teleport(far, ORIGIN[0] + gx, ORIGIN[1] + gz)
    far.target = p
    gate = {('loc', loc.x, loc.z, 'Open') for loc in w.locs if loc.active and tables.loc_types[loc.type]['debugname'] in PEN_GATE}
    assert far in free.attackers(task) and free.Reach().attackers(task) == []
    assert Reference.targets(task, [far]) == {('npc', far.x, far.z, 'Attack')}  # the goal tasks' Reference, unchanged
    assert free.ReachReference().targets(task, [far]) == gate and free.fight_back(task, free.Reach()) is None
    fighter = free.make('melee')  # rb2-2: the fighter itself takes the reach rule and goes for the gate
    fighter.reset(0)
    assert p.target is None and fighter.act(task.projected, task) == free.open_gate(task) is not None
    if near is not None and free.in_pen(task, {}) and not outside(pen[0])[near.x - ORIGIN[0], near.z - ORIGIN[1]]:
        near.target = p
        assert free.Reach().attackers(task) == [near]
        near.target = None
    # a chase begun with the gate open runs on once it has closed: the melee controls and a rotation gatherer open it
    far.target, p.target = None, far
    task.agent.last_observation['optionBusy'] = True  # the chase runs on: a reference waits while an option runs
    opening = free.open_gate(task)
    assert free.Reach().chasing(task) and opening is not None
    rotation = free.make('rotation', period=10 ** 6)
    rotation.reset(0)
    for control in (free.make('melee'), free.make('chickens'), rotation.members[rotation.order.index('fish')]):
        control.reset(0)
        assert control.act(task.projected, task) == opening


def closed(task):
    return any(loc.active and task.world.t.loc_types[loc.type]['debugname'] in PEN_GATE for loc in task.world.locs)


def chase_across_fence(tables, pen):
    """A real chase from inside a closed pen of a goblin outside its fence: (the life, the goblin, the ACT slot that
    started the running option)."""
    from sim.agent import A_ACT0
    task = in_pen_life(tables, pen)
    w, p, (_, _, (gx, gz)) = task.world, task.agent.player, pen
    far = next(n for n in w.npcs if n.is_valid() and tables.npc_types[n.type]['debugname'] == 'goblin')
    w.npc_teleport(far, ORIGIN[0] + gx, ORIGIN[1] + gz)
    task.step(0)
    obs = task.agent.last_observation
    task.step(A_ACT0 + next(k for k, s in enumerate(obs['actSel']) if s['option'] == 'Attack' and (s['x'], s['z']) == (far.x, far.z)))
    task.step(0)
    obs = task.agent.last_observation
    assert obs['optionBusy'] and p.target is far and closed(task)
    return task, far, obs['optionAction'] - A_ACT0


def put_in_slot(task, k, match):
    """Orders the published ACT entries, their targets with them, as a new scan can: the first that `match(entry,
    target)` picks in slot k, any other it picks after every other entry."""
    a = task.agent
    rows = a.last_observation['actSel']
    picked = [i for i in range(len(rows)) if match(rows[i], a.act_targets[i])]
    rest = [i for i in range(len(rows)) if i not in picked]
    order = rest[:k] + picked[:1] + rest[k:] + picked[1:]
    for entries in (a.last_observation['actSel'], a.act_targets):
        entries[:] = [entries[i] for i in order]
    assert match(rows[k], a.act_targets[k])


def test_an_open_on_the_slot_of_a_running_chase_ends_the_chase_first(tables, pen):
    """rb2-1: the agent goes on with a running option when the ACT slot that started it is chosen again, whatever that
    slot holds now. With the gate's Open in the chase's slot, the Open would be lost; the fighter first ends the chase
    (a click toward the gate or a page turn), then opens the gate."""
    from sim.agent import A_ACT0
    task, far, j = chase_across_fence(tables, pen)
    put_in_slot(task, j, lambda s, _: s['option'] == 'Open')
    assert free.swallowed(task, A_ACT0 + j)
    task.step(A_ACT0 + j)  # chosen as it stands: the chase goes on
    assert task.raw['actionNote'] == f'continue:act_{j}' and closed(task) and task.agent.player.target is far
    k = task.agent.last_observation['optionAction'] - A_ACT0
    put_in_slot(task, k, lambda s, _: s['option'] == 'Open')
    fighter = free.make('melee')
    fighter.reset(0)
    end = fighter.act(task.projected, task)
    assert not A_ACT0 <= end < A_ACT0 + 16 and fighter.enders == 1
    task.step(end)
    for _ in range(12):
        if not closed(task):
            break
        task.step(int(fighter.act(task.projected, task)))
    assert not closed(task)


def test_a_fight_back_on_the_slot_of_a_running_chase_ends_the_chase_first(tables, pen):
    """rb2-1, the same exposure for fight_back: an attacker in reach put in the slot of the running chase. The rotation's
    gatherer first ends the chase, then attacks it."""
    from sim.agent import A_ACT0
    task, far, j = chase_across_fence(tables, pen)
    p = task.agent.player
    near = next(n for n in task.world.npcs if n is not far and n.is_valid()
                and tables.npc_types[n.type]['debugname'] == 'goblin' and free.Reach().reachable(task, [n])
                and any(s['option'] == 'Attack' and t is n for s, t in zip(task.raw['actSel'], task.agent.act_targets)))
    near.target = p
    put_in_slot(task, j, lambda s, target: s['option'] == 'Attack' and target is near)
    assert free.fight_back(task, free.Reach()) == A_ACT0 + j and free.swallowed(task, A_ACT0 + j)
    rotation = free.make('rotation', period=10 ** 6)
    rotation.reset(0)
    fish = rotation.members[rotation.order.index('fish')]
    end = fish.act(task.projected, task)
    assert not A_ACT0 <= end < A_ACT0 + 16 and fish.enders == 1
    task.step(end)
    for _ in range(6):
        if p.target is near:
            break
        task.step(int(fish.act(task.projected, task)))
    assert p.target is near


def test_a_rotation_gatherer_that_takes_over_in_a_fight_finishes_it(tables, pool):
    """The timer can hand over in the middle of the fighter's fight. Here the fight starts, then a click stops the
    player's attack while the goblin still attacks: the rotation's gatherer attacks it back (as the four-way timer of 11
    November did) and finishes the fight; the wood reference alone goes on with its own work."""
    from act_contract import CLICK_ROWS
    from sim.agent import A_ACT0
    task = worn_life(tables, pool)
    fighter, p = free.make('melee'), task.agent.player
    fighter.reset(0)
    for _ in range(600):  # the fighter until a goblin fights back
        task.step(int(fighter.act(task.projected, task)))
        if free.attackers(task) and p.target in free.attackers(task):
            break
    click = next(row for row in CLICK_ROWS if task.projected['mask'][row])
    task.step(click)  # the player's attack stops; the goblin keeps the player as its target
    hit = free.attackers(task)
    assert hit and p.target not in hit
    rotation = free.make('rotation', period=10 ** 6)
    rotation.reset(0)
    wood = rotation.members[rotation.order.index('wood')]
    attack = {A_ACT0 + k for k, sel in enumerate(task.raw['actSel'])
              if sel['option'] == 'Attack' and (sel['x'], sel['z']) in {(n.x, n.z) for n in hit}}
    assert wood.act(task.projected, task) in attack and free.make('wood').act(task.projected, task) not in attack
    combat = lambda: sum(p.stats[k] for k in (ATTACK, STRENGTH, DEFENCE, HITPOINTS))
    before = combat()
    for _ in range(200):
        _, _, done, _ = task.step(int(wood.act(task.projected, task)))
        if done or not free.attackers(task):
            break
    assert not done and not free.attackers(task) and combat() > before


def test_the_chickens_control_fights_back_an_offered_attacker_and_no_other(tables, pool):
    """A hen can keep the player as its target from where it cannot reach (2 of 64 soak lives stopped on one outside a
    closed pen): the control fights back only an attacker whose Attack is offered, else plays the melee reference."""
    from sim.agent import A_ACT0
    from sim.run import reference_action
    for record in range(40):
        if not draw_village(KEY, 0, record, [pool[record % 4]], tables, False, layout=0)['village']['worn']:
            continue
        task = life(tables, pool, record=record)
        task.step(0)
        obs, p = task.agent.last_observation, task.agent.player
        offered = {(s['x'], s['z']): k for k, s in enumerate(obs['actSel']) if s['name'] == 'Chicken' and s['option'] == 'Attack'
                   and obs['mask'][A_ACT0 + k]}
        hens = [n for n in task.world.npcs if n.is_valid() and n.type in (41, 1017)]
        far = [n for n in hens if (n.x, n.z) not in offered]
        near = [n for n in hens if (n.x, n.z) in offered]
        if far and near:
            break
    far[0].target = p
    control = free.make('chickens')
    assert control.act(obs, task) == reference_action(free.view(task, 'melee'))
    near[0].target = p
    assert control.act(obs, task) == A_ACT0 + offered[(near[0].x, near[0].z)]


def test_the_rotation_hands_over_at_each_period_and_has_no_default_period():
    """Claim R's bar: the single-activity references, the fighter included, each for `period` ticks; its gatherers drop
    any gatherer's product. The study freezes the period later, so the control takes none by default."""
    rotation = free.Rotation(('wait', 'random', 'wood'), 10)
    assert [m.drops for m in rotation.members[2:]] == [('wood',)]
    rotation.members = [SimpleNamespace(act=lambda o, t, k=k: k) for k in range(3)]
    at = lambda tick: rotation.act(None, SimpleNamespace(world=SimpleNamespace(tick=tick), start_tick=100))
    assert [at(100 + d) for d in (0, 9, 10, 19, 20, 29, 30)] == [0, 0, 1, 1, 2, 2, 0]
    for period in (0, None, 9000.0):
        with pytest.raises(ValueError):
            free.Rotation(('wait',), period)
    with pytest.raises(ValueError):
        free.make('rotation')
    study = free.make('rotation', period=9000)
    assert study.order == free.ROTATION_ORDER == ('melee', *free.GATHERERS) and isinstance(study.members[0], free.Fighter)
    assert {m.drops for m in study.members if isinstance(m, free.Gatherer)} == {free.GATHERERS}


def test_a_soak_row_reports_a_life_and_an_unsupported_site_with_its_evidence(tables, monkeypatch):
    monkeypatch.setattr(free, '_TABLES', tables)
    row = free.soak_life((True, 25, 300, 0, KEY, 2, 'random', 9000, 1))
    assert row['reason'] in ('horizon', 'death') and row['unsupported'] is None and row['muggers_placed'] == 1
    assert row['gain_s'] == sum(row['gain'].get(s, 0) for s in FREE_PLAY['skills']) and row['form'] == 'muggers'
    assert (row['death'] is None) == (row['reason'] == 'horizon') and json.dumps(row)
    real = FreePlayEpisode.step

    def step(self, action):
        if self.decisions == 3:
            raise Unsupported('engine script x')
        return real(self, action)
    monkeypatch.setattr(FreePlayEpisode, 'step', step)
    row = free.soak_life((False, 25, 300, 0, KEY, 2, 'random', 9000, 1))
    u = row['unsupported']
    assert row['reason'] == 'unsupported' and row['death'] is None and row['decisions'] == 4
    assert u['message'] == 'engine script x' and u['file'] == 'sim/tests/test_free_play.py' and u['function'] == 'step'
    assert len(u['actions']) == 4 and u['start']['village']['muggers'] == [] and u['observation']['type'] == 'obs'


def test_the_soak_plays_random_play_to_its_decisions_then_each_control(tmp_path, tables):
    manifest = free.soak(tmp_path / 'soak', world='village-world-v1', forms=(False, True), xp_rate=25, ticks=200, first=0, records=2, key=KEY,
                         controls=('wait', 'random'), random_decisions=150, seed=1, processes=1, chunk=2)
    rows = [json.loads(line) for line in (tmp_path / 'soak' / 'lives.jsonl').read_text().splitlines()]
    for form in ('none', 'muggers'):
        played = [r for r in rows if r['form'] == form and r['actor'] == 'random']
        assert sum(r['decisions'] for r in played) >= 150 and len(played) % 2 == 0
        assert sum(r['decisions'] for r in played[:-2]) < 150 or len(played) == 2
        assert sorted(r['record'] for r in rows if r['form'] == form and r['actor'] == 'wait') == [0, 1]
    assert manifest['counts']['none/wait']['lives'] == 2 and json.loads((tmp_path / 'soak' / 'manifest.json').read_text())
    with pytest.raises(ValueError):  # the rotation needs its periods
        free.soak(tmp_path / 'none', world='village-world-v1', controls=('rotation',), processes=1)
    manifest = free.soak(tmp_path / 'rotation', world='village-world-v1', forms=(False,), xp_rate=25, ticks=200, records=2, key=KEY,
                         controls=('rotation',), periods=(20, 50), seed=1, processes=1)
    rows = [json.loads(line) for line in (tmp_path / 'rotation' / 'lives.jsonl').read_text().splitlines()]
    assert sorted((r['period'], r['record']) for r in rows) == [(20, 0), (20, 1), (50, 0), (50, 1)]
    for r in rows:  # each phase's start state and its XP in S, which add up to the life's
        phases = [ph['phase'] for ph in r['phases']]  # a phase with no decision in it has no mark
        assert phases[0] == 0 and phases == sorted(set(phases)) and len(phases) >= 2
        assert sum(ph['gained'] for ph in r['phases']) == sum(r['xp'].get(s, 0) for s in S)
    assert set(manifest['counts']) == {'none/rotation@20', 'none/rotation@50'} and manifest['periods'] == [20, 50]
