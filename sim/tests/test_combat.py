"""sustained-combat-v1 (sim/combat.py): persistent state across fights, the certified kills and their settlement, the
death the native monitor would record, the keyed starts, the v3 batch's per-goal horizons and goal-activity-v11's
bindings."""
from hypothesis import given, strategies as st
import numpy as np
import pytest

from baseline.resource_task import GOALS
from sim import check, combat, engine
from sim.engine import HITPOINTS
from sim.task import draw_start
from sim.world import generate, split_seeds


@pytest.fixture(scope='module')
def dev(tables):
    pool = [generate(s, tables) for s in split_seeds('development', 4)]
    return pool


def record(tables, pool, k, group, key=3):
    start = draw_start(key, 3, k, pool, layout=k % len(pool))
    start['combat'] = combat.draw_combat(key, 3, k, pool[start['layout']], start, tables, group=group)
    return start


def started(tables, pool, rec):
    from sim.world import LayoutTables
    layout = pool[rec['layout']]
    return combat.episode(rec, layout, LayoutTables(tables, layout), tables)


def play(task, actor, limit=2400):
    paid, info = 0.0, None
    for _ in range(limit):
        _, reward, done, info = task.step(actor.action(task) if actor else 0)
        paid += reward
        if done:
            return paid, info
    raise AssertionError('the episode did not end')


def test_hitpoints_food_and_kills_carry_across_fights_and_settle_to_the_audit(tables, dev):
    """The cautious reference fights several goblins in one world: every hitpoint gain is an eat or natural healing
    or a Hitpoints level-up at 25x XP (nothing restores the player after a kill), each certified life is paid once as it falls, and the rewards sum to
    the audited return, min(kills, target) / target alive."""
    rec = record(tables, dev, 0, 'hurt_fed')
    task = started(tables, dev, rec)
    assert task.agent.player.hp == rec['combat']['hp'] < task.agent.player.base_levels[HITPOINTS]
    paid, info = play(task, combat.Reference('sustain'))
    evidence = combat.fights(task)
    assert info['report']['survived'] and info['outcomes'] >= combat.COMBAT['target']
    assert paid == pytest.approx(info['episode']['r']) == 1.0
    assert {h['cause'] for h in evidence['heals']} <= {'eat', 'regen', 'levelup'} and evidence['summary']['eats'] >= 1
    received, gained = evidence['summary']['received'], sum(h['gained'] for h in evidence['heals'])
    assert task.agent.player.hp == task.arm_hp - received + gained
    lives = [(f['npc'], f['life']) for f in evidence['fights'] if f['certified']]
    assert len(lives) == len(set(lives)) == info['outcomes'] == len(evidence['kill_ticks'])


def test_a_death_is_terminal_even_when_a_heal_follows_in_the_same_tick(tables, dev):
    """The native monitor records a death when hitpoints reach 0 (env-ts/engine/death.ts); a regeneration later in the
    same tick must not hide it, or the death queue runs on into unsupported ticks. The kills paid so far settle to 0."""
    task = started(tables, dev, record(tables, dev, 1, 'healthy'))
    p, w = task.agent.player, task.world

    def hit_then_heal():  # inside one cycle: a lethal hit, then [timer,health_regen]'s heal
        w.player_damage(p, p.hp)
        w.stat_heal(p, HITPOINTS, 1, 0, cause='regen')
        yield from ()
    p.enqueue('[queue,test_hit_then_heal]', hit_then_heal, kind='engine')
    paid, info = play(task, None)
    assert p.hp == 1 and p.dead
    assert info['report']['reason'] == 'death' and info['episode']['r'] == 0.0 and paid == 0.0


def test_combat_draws_are_keyed_and_leave_the_four_goal_start_unchanged(tables, dev):
    a, b = record(tables, dev, 2, None), record(tables, dev, 2, None)
    assert a == b and a['combat']['group'] in combat.GROUPS
    plain = draw_start(3, 3, 2, dev, layout=2)
    assert {k: v for k, v in a.items() if k != 'combat'} == plain
    danger = record(tables, dev, 2, 'danger')['combat']
    assert len(danger['muggers']) == combat.COMBAT['groups']['danger']['muggers'] and danger['food']
    assert record(tables, dev, 2, 'hurt_unfed')['combat']['food'] == []


def test_waiting_earns_nothing(tables, dev):
    paid, info = play(started(tables, dev, record(tables, dev, 3, 'healthy')), None)
    assert info['outcomes'] == 0 and paid == info['episode']['r'] == 0.0


def test_the_v3_batch_ends_each_goal_at_its_own_horizon(tables):
    """CpuBatch goal-activity-v3: the gathering slots keep v2's 240-tick episodes, the melee slot plays sustained combat
    to its own horizon; certified returns follow each goal's target. The step that ends an episode reports that
    episode's facts (sim/vec.py EPISODE_FACTS), not the next one's: its success, and for combat its group, food and
    healing as sim.combat.fights finds them in the ended world."""
    from sim.vec import COMBAT_FACTS, EPISODE_FACTS, CpuBatch
    pool = [generate(s, tables) for s in split_seeds('development', 2)]
    env = CpuBatch(tables, pool, 4, task='goal-activity-v3')
    state, obs = env.reset(5)
    rng, ends = np.random.default_rng(0), {}
    for _ in range(900):
        action = np.asarray([rng.choice(np.flatnonzero(obs['mask'][b])) for b in range(4)], np.int32)
        tasks, starts = list(env.tasks), list(env.starts)
        try:
            state, obs, reward, done, info = env.step(state, action)
        except RuntimeError:  # a world reported unsupported content: its evidence names it
            pytest.fail(f'unsupported: {env.evidence}')
        assert not info['unsupported'].any(), env.evidence
        for b in np.flatnonzero(done):
            facts = dict(zip(EPISODE_FACTS, info['facts'][b].tolist()))
            assert facts['success'] == int(info['episode_return'][b] == 1.0)
            assert facts['search'] == facts['discovered'] == 0 and facts['discovery_ticks'] == -1  # v3 has no search
            assert facts['cells'] == len(tasks[b].agent.visited)
            if b == 3:
                s = combat.fights(tasks[b])['summary']
                assert {k: facts[k] for k in ('group', 'success', *COMBAT_FACTS)} == dict(
                    group=1 + combat.GROUPS.index(starts[b]['combat']['group']), success=facts['success'],
                    eaten=s['eats'] + s['eats_without_gain'], eat_gain=s['eat_gain'], regen_gain=s['regen_gain'],
                    levelup_gain=s['levelup_gain'], aggressed=s['aggressed'], eats_without_gain=s['eats_without_gain'])
            else:
                assert facts['group'] == facts['eaten'] == 0
        for b in np.flatnonzero(done & (info['terminal'] == 1)):
            ends.setdefault(int(b), int(info['episode_ticks'][b]))
        if len(ends) == 4:
            break
    assert ends == {0: 240, 1: 240, 2: 240, 3: combat.COMBAT['horizon']}
    assert env.tasks[3].combat and env.tasks[3].target == combat.COMBAT['target']


def test_goal_activity_v11_is_v5_with_its_two_bindings_and_v5_has_none():
    """goal-activity-v11 (arm Layouts, mine only) keeps v5's horizons, targets, combat, search and supply; only its
    version and its bound pool and goal set differ. No other version carries a binding."""
    v11 = combat.task_spec('goal-activity-v11')
    assert v11.pop('pool') == dict(world='compact-world-v1', split='train', first=1000, count=128)
    assert v11.pop('goals') == ('mine',)
    assert v11 == dict(combat.task_spec('goal-activity-v5'), version='goal-activity-v11')
    for task in combat.TASKS[:4]:
        assert set(combat.task_spec(task)) == {'version', 'horizons', 'targets', 'combat', 'search', 'supply'}


BOUND_POOL = [(seed, 'compact-world-v1') for seed in range(1000, 1128)]


@given(st.data())
def test_v11_takes_only_its_bound_pool_in_order_and_mine(data):
    """check_bindings: v11 takes compact-world-v1 seeds 1,000-1,127 in order with goals ('mine',). One layout dropped,
    added, moved, of another seed or of another world is refused, as is any other goal set; unbound versions take
    both."""
    spec = combat.task_spec('goal-activity-v11')
    combat.check_bindings(spec, iter(BOUND_POOL), ('mine',))
    pool, k = list(BOUND_POOL), data.draw(st.integers(0, 127))
    edit = data.draw(st.sampled_from(('drop', 'add', 'swap', 'seed', 'world')))
    if edit == 'drop':
        del pool[k]
    elif edit == 'add':
        pool.insert(k, (data.draw(st.integers(0, 2_009_999)), 'compact-world-v1'))
    elif edit == 'swap':
        j = data.draw(st.integers(0, 127).filter(lambda j: j != k))
        pool[k], pool[j] = pool[j], pool[k]
    elif edit == 'seed':
        pool[k] = (data.draw(st.integers(0, 2_009_999).filter(lambda s: s != pool[k][0])), 'compact-world-v1')
    else:
        pool[k] = (pool[k][0], data.draw(st.sampled_from(('search-world-v1', 'supply-world-v1', None))))
    with pytest.raises(ValueError, match='plays compact-world-v1 train seeds 1000-1127 in order'):
        combat.check_bindings(spec, pool, ('mine',))
    goals = data.draw(st.sampled_from((GOALS, ('wood',), ('fish',), ('melee',))))
    with pytest.raises(ValueError, match=r"plays goals \['mine'\]"):
        combat.check_bindings(spec, BOUND_POOL, goals)
    for task in combat.TASKS[:4]:
        combat.check_bindings(combat.task_spec(task), pool, goals)


def test_unknown_exclusions_are_refused(tables):
    from sim.engine import World
    with pytest.raises(ValueError):
        World(tables, exclusions=('respawn',))


def test_every_item_a_combat_world_can_hand_the_player_has_supported_interactions(tables, dev):
    """The goblin and mugger loot the declared world keeps (sim/engine.py LOOT_OUTSIDE excluded), the kit, the gathered
    products and mining's gems: every held option, every item used on another, every pickup and every item used on each
    npc and loc type of the world runs without reaching unsupported content, so none can stop a training rollout."""
    import itertools
    from sim.engine import APOBJ1, LOOT_OUTSIDE, loot_outside
    loot = ['bones', 'bodyrune', 'brass_necklace', 'bronze_sq_shield', 'chefs_hat', 'coins', 'earthrune', 'waterrune',
            'bronze_med_helm', 'copper_ore', 'fishing_bait', 'mindrune', 'rope', *LOOT_OUTSIDE]
    held = [n for n in dict.fromkeys(loot + ['bronze_axe', 'net', 'bronze_pickaxe', 'shrimp', 'bread', 'bronze_dagger',
                                             'bronze_sword', 'logs', 'oak_logs', 'raw_shrimp', 'raw_anchovies', 'tin_ore',
                                             'iron_ore', 'uncut_sapphire', 'uncut_emerald', 'uncut_ruby', 'uncut_diamond'])
            if not loot_outside(tables, tables.obj_id(n))]
    assert 'beer' not in held and 'knife' not in held and 'bronze_sq_shield' in held
    rec = record(tables, dev, 0, 'danger')

    def fresh(names):
        task = started(tables, dev, rec)
        p = task.agent.player
        p.inv.items = [None] * 28
        for name in names:
            p.inv.add(tables, tables.obj_id(name), 1)
        return task, p, {ob: s for s, ob, _ in p.inv.listed()}
    for name in held:
        for op in [x for x in (tables.obj_types[tables.obj_id(name)].get('iop') or []) if x and x != 'hidden']:
            task, p, slots = fresh([name])
            (task.agent.drop_slot if op.lower() == 'drop' else lambda s: task.agent.op_held(s, op))(slots[tables.obj_id(name)])
            task.agent.wait_ticks(12)
        task, p, _ = fresh([])
        task.world.obj_add(p, p.level, p.x, p.z, tables.obj_id(name), 1, 200)
        ops = tables.obj_types[tables.obj_id(name)]['op'] or []
        p.set_interaction(task.world.objs[-1], APOBJ1 + next((i for i, x in enumerate(ops) if x and 'take' in x.lower()), 2))
        task.agent.wait_ticks(12)
    for a, b in itertools.permutations(held, 2):
        task, p, slots = fresh([a, b])
        task.agent.use_on_held(slots[tables.obj_id(b)], dict(slot=slots[tables.obj_id(a)], id=tables.obj_id(a), name=a))
        task.agent.wait_ticks(12)
    task, _, _ = fresh([])
    kinds = {}
    for thing in [n for n in task.world.npcs] + task.world.locs:
        kind = tables.npc_types[thing.type]['debugname'] if hasattr(thing, 'nid') else tables.loc_types[thing.type]['debugname']
        kinds.setdefault(kind, thing)
    assert {'goblin', 'mugger', 'tree'} <= set(kinds)
    for name in held:
        for kind, thing in kinds.items():
            task, p, slots = fresh([name])
            using = dict(slot=slots[tables.obj_id(name)], id=tables.obj_id(name), name=name)
            if hasattr(thing, 'nid'):   # an item used on an npc or a loc: act-v3's ACT with `using`
                task.agent.interact_npc(next(n for n in task.world.npcs if n.nid == thing.nid), 'Use', 'use', using)
            else:
                task.agent.interact_loc(task.world.locs[thing.index], 'Use', 'use', using)
            task.agent.wait_ticks(60)


def test_after_the_setup_every_mask_offers_exactly_the_style_rows_that_change_the_style(tables, dev):
    """mask-v2 against the engine's own state at every decision of two episodes, each its own fresh start: the
    cautious reference fights (Hitpoints and Attack level-ups at 25x XP), random legal play wields and requests styles.
    The setup's STYLE_0 has run, so no observation is stale; a row is offered iff the style it selects is not the
    current one. An open bank, a use selection or a dialog offers its own actions only, under mask-v1 as now."""
    from sim.agent import A_STYLE0
    from sim.engine import WEARPOS_RHAND, weapon_styles
    rng, seen = np.random.default_rng(5), dict(levels=0, weapons=set(), styles=set(), checked=0, modal=0)
    for k, (group, actor) in enumerate((('hurt_fed', combat.Reference('sustain')), ('healthy', None))):
        task = started(tables, dev, record(tables, dev, k, group))
        p, w, levels = task.agent.player, task.world, sum(task.agent.player.base_levels)
        for _ in range(400):
            rhand = p.worn.items[WEARPOS_RHAND][0] if p.worn.items[WEARPOS_RHAND] else None
            count = len(weapon_styles(w, rhand)['damagestyle'])
            current = min(p.varps['com_mode'], count - 1)
            raw = task.raw
            assert raw['style'] == dict(mode=p.varps['com_mode'], count=count, stale=False)
            rows = raw['mask'][A_STYLE0:A_STYLE0 + 4]
            if raw['bankOpen'] or raw['using'] or raw['dialogOpen'] or raw['modalOpen']:
                assert rows == [0, 0, 0, 0]
                seen['modal'] += 1
            else:
                assert rows == [int(min(r, count - 1) != current) for r in range(4)]
                seen['checked'] += 1
            seen['weapons'].add(rhand)
            seen['styles'].add(current)
            action = actor.action(task) if actor else int(rng.choice(np.flatnonzero(task.raw['mask'])))
            _, _, done, _ = task.step(action)
            if done:
                break
        seen['levels'] += sum(p.base_levels) - levels
    assert seen['levels'] > 0 and len(seen['weapons']) >= 2 and len(seen['styles']) >= 2 and seen['checked'] > 200, seen


def test_evaluate_s_command_line_passes_the_sampling_seed_through(monkeypatch, tmp_path):
    """a sampled pass declares its seed base on the command line; the default is the frozen one."""
    seen = {}
    monkeypatch.setattr(combat, 'evaluate', lambda output, **kw: seen.update(kw))
    monkeypatch.setattr('sys.argv', ['sim.combat', 'evaluate', '--checkpoint', 'final.zip', '--policies', 'final', '--sampled',
                                     '--seed', '909001', '--output', str(tmp_path)])
    combat.main()
    assert seen['sampled'] is True and seen['seed'] == 909001
    monkeypatch.setattr('sys.argv', ['sim.combat', 'evaluate', '--checkpoint', 'final.zip', '--policies', 'final', '--output', str(tmp_path)])
    combat.main()
    assert seen['sampled'] is False and seen['seed'] == 303001


def idle_beside_a_goblin(tables, dev):
    task = started(tables, dev, record(tables, dev, 0, 'healthy'))
    goblins = [n for n in task.world.npcs if n.active and tables.npc_types[n.base]['debugname'] == 'goblin']
    return task.world, task.agent.player, goblins[0]


def struck(tables, dev, before=None, damage=0):
    """A combat start's idle player, after `before(world, p, npc)`, hit once through playerhit_n_melee: its damage and
    auto-retaliate requests after one pass of the queues. Returns the world, the player and the npc."""
    world, p, npc = idle_beside_a_goblin(tables, dev)
    if before:
        before(world, p, npc)
    engine.playerhit_n_melee(world, npc, p, damage, 4)
    world.process_queues(p)
    return world, p, npc


def test_an_idle_player_hit_once_fights_back_with_no_request(tables, dev):
    delay = {}
    world, p, npc = struck(tables, dev, lambda world, p, _: delay.update(before=p.varps['action_delay']))
    assert not p.queue and not p.weak_queue
    assert p.target is npc and p.target_op == engine.APNPC1 + 1
    assert delay['before'] < world.tick and p.varps['action_delay'] == world.tick + 2


@pytest.mark.parametrize('busy', ['walking', 'interacting'])
def test_a_walking_or_interacting_player_does_not_fight_back(tables, dev, busy):
    other = []

    def make_busy(world, p, npc):
        if busy == 'walking':
            p.queue_waypoint(p.x + 3, p.z)
        else:
            other.append(next(n for n in world.npcs if n.active and n is not npc and n.base == npc.base))
            p.set_interaction(other[0], engine.APNPC1 + 1)
        p.varps['action_delay'] = 0
    world, p, npc = struck(tables, dev, make_busy)
    assert p.target is (other[0] if other else None) and p.varps['action_delay'] == 0
    assert bool(p.path) == (busy == 'walking')


def test_auto_retaliate_off_leaves_the_player_idle(tables, dev):
    world, p, _ = struck(tables, dev, lambda world, p, npc: p.varps.update(option_nodef=1, action_delay=0))
    assert p.target is None and p.varps['action_delay'] == 0


def test_an_npc_no_longer_found_is_not_targeted(tables, dev):
    world, p, npc = idle_beside_a_goblin(tables, dev)
    p.varps['action_delay'] = 0
    assert list(engine.playerhit_n_retaliate(world, p, (npc.type + 1) << 16 | npc.nid)) == []
    assert p.target is None and p.varps['action_delay'] == 0


@pytest.mark.parametrize('later', [0, 9])
def test_an_action_delay_not_yet_past_stays_and_the_player_still_fights_back(tables, dev, later):
    tick = {}

    def delayed(world, p, npc):
        tick['value'] = world.tick + later
        p.varps['action_delay'] = tick['value']
    _, p, npc = struck(tables, dev, delayed)
    assert p.target is npc and p.varps['action_delay'] == tick['value']


@pytest.mark.parametrize('weapon,flinch', [(None, 2), ('bronze_sword', 2), ('bronze_dagger', 2), ('bronze_axe', 2),
                                           ('bronze_pickaxe', 2), ('bronze_2h_sword', 3)])
def test_the_flinch_is_half_the_weapons_rate_and_two_unarmed(tables, dev, weapon, flinch):
    def wear(world, p, npc):
        p.worn.items[engine.WEARPOS_RHAND] = (tables.obj_id(weapon), 1) if weapon else None
    world, p, _ = struck(tables, dev, wear)
    assert p.varps['action_delay'] == world.tick + flinch


def test_a_lethal_hit_runs_the_retaliate_before_the_death_script_stops_it(tables, dev):
    world, p, npc = struck(tables, dev, lambda world, p, _: p.levels.__setitem__(HITPOINTS, 1), damage=1)
    assert p.levels[HITPOINTS] == 0 and p.varps['death'] == 1 and p.target is npc
    world.tick += 2  # player_death's delay of 1
    world.process_players()
    assert p.target is None


def retaliate_trial(check='combat_idle', attacks=(10, 14, 18), hits=(12, 16, 20), walks=(), requests=0, end=40):
    return dict(check=check, trial=0, attacks=[(a, 7, 'mugger', 0) for a in attacks], agent_hits={'7:0': list(hits)} if hits else {},
                attack_requests=requests, walks=list(walks), end=end)


def test_rule_c6_idle_needs_a_hit_two_ticks_off_the_attacks_with_no_request():
    assert check.retaliate_rule([retaliate_trial()])['C6 idle']['passed']
    assert check.retaliate_rule([retaliate_trial(attacks=(13, 17), hits=(11, 15, 19))])['C6 idle']['passed']  # capture began late
    assert not check.retaliate_rule([retaliate_trial(hits=(), attacks=(10, 14))])['C6 idle']['passed']
    assert not check.retaliate_rule([retaliate_trial(hits=(10, 14, 18))])['C6 idle']['passed']  # a policy-started fight's offset
    assert not check.retaliate_rule([retaliate_trial(hits=(12, 15, 20))])['C6 idle']['passed']  # a gap below the rate
    assert not check.retaliate_rule([retaliate_trial(requests=1)])['C6 idle']['passed']


def test_rule_c6_idle_counts_no_trial_attacked_too_late_to_answer():
    late = retaliate_trial(attacks=(39,), hits=(), end=40)
    report = check.retaliate_rule([late])['C6 idle']
    assert report['too_late'] == 1 and not report['passed']
    assert check.retaliate_rule([late, retaliate_trial()])['C6 idle']['passed']


def test_rule_c6_walking_needs_an_attack_inside_a_walk_and_no_hit():
    walked = lambda **k: retaliate_trial('combat_walk', **k)
    assert check.retaliate_rule([walked(attacks=(25, 29), hits=(22,), walks=[(23, 35)])])['C6 walking']['passed']
    assert check.retaliate_rule([walked(attacks=(25, 29), hits=(23, 35), walks=[(23, 35)])])['C6 walking']['passed']  # the ends
    assert not check.retaliate_rule([walked(attacks=(25, 29), hits=(27,), walks=[(23, 35)])])['C6 walking']['passed']
    assert not check.retaliate_rule([walked(attacks=(25, 29), hits=(27,), walks=[(23, None)])])['C6 walking']['passed']
    assert not check.retaliate_rule([walked(attacks=(10,), hits=(), walks=[(23, 35)])])['C6 walking']['passed']  # untested
