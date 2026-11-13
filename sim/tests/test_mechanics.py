"""The four activities' mechanics in the reference (sim/engine.py ports), on generated layouts and through the
agent's actions as env-ts offers them: a success and a known refusal each, Prospect, reach, interruption,
equipment, eating, armed attack timing, kills and death."""
from dataclasses import replace

import numpy as np
import pytest

from baseline import config
from sim import world as W
from sim.agent import A_ACT0, A_DIALOG_CONTINUE, A_HELD0, A_INV_NEXT, A_STYLE0, A_USE0, CLICK_ROWS, Agent
from sim.audit import audit, kills
from sim.engine import ATTACK, HITPOINTS, WEARPOS_RHAND, World
from sim.spec import START_KIT
from sim.task import WORLD_START_TICK, GoalEpisode

KIT = list(START_KIT) + [('bronze_sword', 1)]


@pytest.fixture(scope='module')
def site(tables):
    """A development layout holding every resource type, with its LayoutTables."""
    for seed in W.split_seeds('development', 200):
        layout = W.generate(seed, tables)
        names = {tables.loc_types[int(t)]['debugname'] for t in layout.locs[:, 0]}
        if {'ironrock1', 'oaktree', 'tree', 'tinrock1'} <= names and names & set(W.ROCKS['copper']):
            return layout, W.LayoutTables(tables, layout)
    raise AssertionError('no development layout with every resource')


def begin(site, near, levels, kit=KIT, seed=1):
    """An agent started on a tile that operates the first loc named `near` (or beside an npc type)."""
    layout, lt = site
    world = World(lt, seed=seed, start_tick=WORLD_START_TICK)
    agent = Agent(world)
    loc = next(loc for loc in world.locs if world.t.loc_types[loc.type]['debugname'] == near)
    tile = next(xy for xy in world.loc_reaching(loc) if lt.source(0, *xy) >= 0)
    agent.start(0, config.TASK_CFG, tile, kit, levels)
    return world, agent, loc


def act(agent, loc_or_name, option):
    sel = agent.last_observation['actSel']
    k = next(i for i, s in enumerate(sel) if s['option'] == option and
             (s['name'] == loc_or_name if isinstance(loc_or_name, str) else (s['x'], s['z']) == (loc_or_name.x, loc_or_name.z)))
    return agent.step(A_ACT0 + k)


def held(agent, name, option):
    k = next(i for i, s in enumerate(agent.last_observation['heldSel']) if s['name'] == name and s['option'] == option)
    return agent.step(A_HELD0 + k)


def wait(agent, decisions, until=lambda: False):
    for _ in range(decisions):
        if until():
            return True
        mask = agent.last_observation['mask']
        agent.step(A_DIALOG_CONTINUE if mask[A_DIALOG_CONTINUE] else 0)
    return until()


def said(world, since=0):
    return [text for tick, text in world.chat_log if tick >= since]


def test_iron_refuses_mining_14_and_yields_at_15(tables, site):
    world, agent, rock = begin(site, 'ironrock1', dict(mining=14))
    obs = act(agent, rock, 'Mine')
    assert obs['feedback']['dialog']['text'] == 'You need a Mining level of 15 to mine this rock.'
    iron = tables.obj_id('iron_ore')
    assert not wait(agent, 20, lambda: agent.player.inv.total(iron))
    world, agent, rock = begin(site, 'ironrock1', dict(mining=15))
    act(agent, rock, 'Mine')
    assert wait(agent, 40, lambda: agent.player.inv.total(iron))
    assert 'You manage to mine some iron.' in said(world) and world.t.loc_types[rock.type]['debugname'] == 'plainrock1'


def test_oak_refuses_woodcutting_14_and_yields_at_15(tables, site):
    """The refusal is printed in the decision's first cycle, and reaches the policy (lines by sequence)."""
    world, agent, oak = begin(site, 'oaktree', dict(woodcutting=14))
    obs = act(agent, oak, 'Chop down')
    assert [c['text'] for c in obs['feedback']['chat']] == ['You need a Woodcutting level of 15 to chop down this tree.']
    logs = tables.obj_id('oak_logs')
    assert not wait(agent, 10, lambda: agent.player.inv.total(logs))
    world, agent, oak = begin(site, 'oaktree', dict(woodcutting=15))
    act(agent, oak, 'Chop down')
    assert wait(agent, 60, lambda: agent.player.inv.total(logs))


def test_mining_without_a_pickaxe_or_room_is_refused(tables, site):
    world, agent, rock = begin(site, 'tinrock1', dict(mining=1), kit=[('bronze_axe', 1), ('net', 1)])
    obs = act(agent, rock, 'Mine')
    assert obs['feedback']['dialog']['text'].startswith('You need a Pickaxe to mine this rock.')
    full = KIT + [('bronze_dagger', 1)] * (28 - len(KIT) - 1) + [('bread', 1)]
    world, agent, rock = begin(site, 'tinrock1', dict(mining=1), kit=full)
    assert agent.player.inv.freespace() == 0
    obs = act(agent, rock, 'Mine')
    assert obs['feedback']['dialog']['text'] == 'Your inventory is too full to hold any more tin.'
    world, agent, tree = begin(site, 'tree', dict(woodcutting=1), kit=full)
    act(agent, tree, 'Chop down')
    assert "Your inventory is too full to hold any more logs." in said(world)


def test_prospect_names_the_ore_after_its_delays_and_an_empty_rock_says_so(tables, site):
    world, agent, rock = begin(site, 'tinrock1', dict(mining=1))
    start = world.tick
    act(agent, rock, 'Prospect')
    lines = [(tick - start, text) for tick, text in world.chat_log if tick >= start]
    assert lines == [(0, 'You examine the rock for ores...'), (3, 'This rock contains tin.')]
    feedback = agent.last_observation['feedback']['chat']  # each line once, aged from its arrival
    assert [(c['text'], c['age']) for c in feedback] == [(text, world.tick - start - at) for at, text in lines]
    world.change_loc(rock, tables.loc_types[rock.type]['params']['next_loc_stage_mining'], rock.shape, rock.angle, 9)
    agent.step(0)
    act(agent, rock, 'Prospect')
    assert said(world)[-1] == 'There is no ore left in this rock.'
    wait(agent, 5)
    assert tables.loc_types[rock.type]['debugname'] == 'tinrock1'  # respawned after 10 * 3999 // 4000 ticks


def test_a_walled_rock_cannot_be_reached(tables, site):
    layout, _ = site
    k = next(i for i, t in enumerate(layout.locs[:, 0]) if tables.loc_types[int(t)]['debugname'] == 'tinrock1')
    x, z = (int(v) for v in layout.locs[k, 1:3])
    def wall(flags):
        flags = flags.copy()
        for dx, dz, own, other in ((1, 0, 0x8, 0x80), (-1, 0, 0x80, 0x8), (0, 1, 0x2, 0x20), (0, -1, 0x20, 0x2)):
            flags[x, z] |= own
            flags[x + dx, z + dz] |= other
        return flags
    walled = replace(layout, flags=wall(layout.flags), flags_open=wall(layout.flags_open),
                     flags_snapshot=wall(layout.flags_snapshot))
    world = World(W.LayoutTables(tables, walled), seed=1, start_tick=WORLD_START_TICK)
    agent = Agent(world)
    rock = world.locs[k]
    agent.start(0, config.TASK_CFG, (rock.x + 1, rock.z), KIT, dict(mining=1))
    act(agent, rock, 'Mine')
    assert "I can't reach that!" in said(world) and not agent.player.inv.total(tables.obj_id('tin_ore'))


def test_walking_away_stops_mining_before_its_roll(tables, site):
    world, agent, rock = begin(site, 'tinrock1', dict(mining=1))
    agent.cfg['decisionPoints'] = False  # one decision is K_TICKS (2) plus jitter
    act(agent, rock, 'Mine')
    last = agent.last_observation  # a click onto an open tile 4 away walks the player off
    move = next(CLICK_ROWS[k] for k, c in enumerate(last['clicks'][:8]) if c['open'] and last['mask'][CLICK_ROWS[k]])
    obs = agent.step(move)
    assert obs['feedback']['ended']['reason'] == 'superseded'
    wait(agent, 10)
    assert not agent.player.inv.total(tables.obj_id('tin_ore')) and 'You swing your pick at the rock.' in said(world)
    assert world.t.loc_types[rock.type]['debugname'] == 'tinrock1' and agent.player.target is None


def test_wielding_swaps_into_the_right_hand_and_sets_the_combat_bonuses(tables, site):
    world, agent, _ = begin(site, 'tree', dict(attack=5, strength=7, defence=3))
    p = agent.player
    assert p.varps['com_stabattack'] == (1 + 8 + 3) * 64  # [login,_]: a new account's level 1, until a recompute
    held(agent, 'Bronze sword', 'Wield')
    sword = tables.obj_id('bronze_sword')
    assert p.worn.items[WEARPOS_RHAND][0] == sword and not p.inv.total(sword)
    # accurate stab: (5 + 8 + 3) * (stabattack 4 + 64); strength (7 + 8) * (5 + 64) -> max hit 2
    assert (p.varps['com_stabattack'], p.varps['com_slashattack'], p.varps['com_maxhit']) == (16 * 68, 16 * 67, 2)
    assert p.varps['com_slashdef'] == (3 + 8) * (2 + 64)
    agent.step(A_STYLE0 + 3)  # the stab table's fourth style: defensive stab
    assert (p.varps['com_mode'], p.varps['damagestyle'], p.varps['damagetype'], p.varps['com_stabdef']) == (3, 2, 0, 14 * 64)
    slot = next(s for s, o, _ in p.inv.listed() if o == tables.obj_id('bronze_axe'))
    held(agent, 'Bronze axe', 'Wield')
    assert p.inv.items[slot][0] == sword and tables.obj_types[p.worn.items[WEARPOS_RHAND][0]]['debugname'] == 'bronze_axe'
    assert (p.varps['damagestyle'], p.varps['damagetype']) == (2, 1)  # the axe table's fourth: defensive slash
    events = [e for e in world.events if e[1] == 'transfer']
    assert [(e[2]['src'], e[2]['dst']) for e in events] == [('inv', 'worn'), ('inv', 'worn'), ('worn', 'inv')]


def test_a_style_row_is_legal_only_where_it_changes_the_style(tables, site):
    """mask-v2: row k selects min(k, styles - 1); the current style is no request, except while the start's
    levels wait for the recompute that the setup's STYLE_0 runs."""
    world, agent, _ = begin(site, 'tree', dict(attack=5, strength=7, defence=3))
    p = agent.player
    rows = lambda: agent.last_observation['mask'][A_STYLE0:A_STYLE0 + 4]
    assert rows() == [1, 1, 1, 1]
    agent.step(A_STYLE0)
    assert p.varps['com_stabattack'] == (5 + 8 + 3) * 64 and rows() == [0, 1, 1, 1]  # unarmed: three styles
    agent.step(A_STYLE0 + 3)  # past unarmed's styles: the last one
    assert p.varps['com_mode'] == 2 and rows() == [1, 1, 0, 0]
    assert agent.step(A_STYLE0 + 3)['actionNote'] == 'action not legal in published observation'
    held(agent, 'Bronze sword', 'Wield')  # four styles: row 3 is a style of its own
    assert p.varps['com_mode'] == 2 and rows() == [1, 1, 0, 1]


def chicken_fight(tables, site, weapon, rounds=40):
    layout, lt = site
    world = World(lt, seed=3, start_tick=WORLD_START_TICK)
    agent = Agent(world)
    hen = next(n for n in world.npcs if tables.npc_types[n.type]['debugname'] in W.CHICKENS)
    tile = next(xy for xy in world.entity_reaching(0, hen.x, hen.z) if lt.source(0, *xy) >= 0)
    agent.start(0, config.TASK_CFG, tile, list(START_KIT) + [(weapon, 1)], dict(attack=10, strength=10))
    if weapon != 'bronze_dagger':
        held(agent, tables.obj_types[tables.obj_id(weapon)]['name'], 'Wield')
    hits, queue = [], world.npc_queue

    def spy(npc, q, arg, delay, source=None):
        if q == 2:
            hits.append((world.tick, npc.nid, npc.life))
        queue(npc, q, arg, delay, source)
    world.npc_queue = spy
    act(agent, 'Chicken', 'Attack')
    wait(agent, rounds, lambda: kills(world.events, agent.name) >= 1)
    first = [tick for tick, nid, life in hits if (nid, life) == hits[0][1:]]
    return world, agent, np.diff(first)


def test_armed_attacks_come_at_the_weapon_attack_rate_and_a_kill_is_certified(tables, site):
    world, agent, gaps = chicken_fight(tables, site, 'bronze_pickaxe')
    assert len(gaps) and gaps.min() == 5  # bronze_pickaxe attackrate 5 (the sword's and the dagger's default 4)
    assert kills(world.events, agent.name) >= 1 and agent.player.stats[ATTACK] > 9800
    dead = [e for e in world.events if e[1] == 'npc_defeated'][0][2]
    wait(agent, 3)  # [proc,npc_death]'s delays, then the drop table
    assert any(o.type == tables.obj_id('bones') for o in world.objs) and dead['type'] in (41, 1017)
    world, agent, gaps = chicken_fight(tables, site, 'bronze_dagger')  # carried, not wielded: unarmed, 4 ticks
    assert len(gaps) and gaps.min() == 4 and agent.player.worn.items[WEARPOS_RHAND] is None


def test_eating_heals_with_its_messages_delay_and_skill_delay(tables, site):
    world, agent, _ = begin(site, 'tree', dict(hitpoints=12))
    p = agent.player
    p.levels[HITPOINTS] = 5
    start, delay = world.tick, p.varps['action_delay']
    obs = held(agent, 'Shrimps', 'Eat')
    assert p.hp == 8 and p.varps['eat_delay'] == start + 2 and p.varps['action_delay'] == delay + 3
    assert said(world, start)[:2] == ['You eat the shrimps.', 'They heal some health.']
    assert [c['text'] for c in obs['feedback']['chat']][:2] == said(world, start)[:2]  # printed during dispatch
    assert p.inv.total(tables.obj_id('shrimp')) == 1
    p.varps['eat_delay'] = world.tick  # within the delay: the next food is not eaten
    held(agent, 'Bread', 'Eat')
    assert p.hp == 8 and p.inv.total(tables.obj_id('bread')) == 1
    agent.step(0)
    held(agent, 'Bread', 'Eat')
    assert p.hp == 12 and said(world)[-2:] == ['You eat the bread.', 'It heals some health.']


def test_an_eat_ends_a_running_chop_and_one_refused_while_delayed_does_not(tables, site):
    """held-v2: OpHeldHandler clears the pending action, and a chop goes on only by setting its interaction
    again each cycle (woodcut.rs2), so no log comes after an Eat. The act's decision ends at its wait limit with the
    oak chop still running and no dialog open; the same start without the Eat gets another log, so the wait would see
    a chop that went on. While the player is delayed the handler refuses the option before it clears anything."""
    logs = tables.obj_id('oak_logs')

    def chop():
        world, agent, oak = begin(site, 'oaktree', dict(woodcutting=70, hitpoints=12), seed=2)
        p = agent.player
        p.levels[HITPOINTS] = 5
        act(agent, oak, 'Chop down')
        assert p.target is oak and not p.modal_state and p.inv.total(logs) > 0  # the chop runs on
        return world, agent, p, p.inv.total(logs)
    world, agent, p, cut = chop()
    assert wait(agent, 40, lambda: p.inv.total(logs) > cut)
    world, agent, p, cut = chop()
    hp = p.hp
    held(agent, 'Shrimps', 'Eat')
    assert p.hp == hp + 3 and p.target is None
    assert not wait(agent, 40, lambda: p.inv.total(logs) > cut)
    world, agent, p, cut = chop()
    p.delayed, p.delayed_until, hp = True, world.tick + 1, p.hp
    slot = next(s['slot'] for s in agent.last_observation['heldSel'] if s['name'] == 'Shrimps')
    assert agent.op_held(slot, 'Eat') == 'opheld: refused while delayed' and agent.drop_slot(slot) == 'drop: refused while delayed'
    assert p.target is not None and p.hp == hp and p.inv.total(tables.obj_id('shrimp')) == 2


def test_with_an_item_selected_a_page_turn_keeps_it_and_the_held_targets_follow_the_page(tables, site):
    """mask-v5: with the axe selected for use, INV_NEXT is legal, the selection stays,
    and the HELD rows name page 1's items, so the axe meets the bread there (OpHeldUHandler's lookup, nothing
    interesting happens)."""
    kit = KIT + [('bronze_dagger', 1), ('bread', 1)]  # 9 stacks: the bread is slot 8, page 1's first
    world, agent, _ = begin(site, 'tree', dict(woodcutting=1), kit=kit)
    p, axe, bread = agent.player, tables.obj_id('bronze_axe'), tables.obj_id('bread')
    k = next(i for i, s in enumerate(agent.last_observation['invSel']) if s['id'] == axe)
    obs = agent.step(A_USE0 + k)
    assert obs['using']['id'] == axe and obs['mask'][A_INV_NEXT] == 1 and obs['invPage'] == 0
    obs = agent.step(A_INV_NEXT)
    assert obs['using']['id'] == axe and obs['invPage'] == 1
    assert [(s['slot'], s['name'], s['option']) for s in obs['heldSel']] == [(8, 'Bread', 'Use')]
    agent.step(A_HELD0)
    assert (p.last_item, p.last_slot, p.last_useitem) == (bread, 8, axe) and agent.using is None


def test_death_settles_every_goal_to_zero(tables, site):
    layout, lt = site
    task = GoalEpisode(World(lt, seed=2, start_tick=WORLD_START_TICK), config.TASK_CFG, 'mine', scale=1.0)
    tile = tuple(int(v) + o for v, o in zip(np.argwhere(layout.start)[0], W.ORIGIN))
    task.start(tile, 'bronze_sword', dict(mining=1))
    task.world.produce(task.agent.player, tables.obj_id('copper_ore'), '[label,get_ore_normal]')
    task.world.produce(task.agent.player, tables.obj_id('tin_ore'), '[label,get_ore_normal]')
    _, paid, done, _ = task.step(0)
    assert not done and paid == 1.0
    task.world.player_damage(task.agent.player, 99)  # as a queued hit does, inside a cycle
    task.world.cycle()
    total, done = paid, False
    while not done:
        _, reward, done, info = task.step(0)
        total += reward
    assert info['report']['reason'] == 'death' and info['episode']['r'] == 0.0 and abs(total) < 1e-9


def test_picked_up_goal_items_are_not_production(tables, site):
    world, agent, rock = begin(site, 'tinrock1', dict(mining=1))
    task_initial = dict(inventory=agent.player.inv.listed(), bank=agent.player.bank.listed(), worn=[])
    first = len(world.events)
    act(agent, rock, 'Mine')
    tin = tables.obj_id('tin_ore')
    assert wait(agent, 40, lambda: agent.player.inv.total(tin))
    agent.drop_slot(next(s for s, o, _ in agent.player.inv.listed() if o == tin))  # DROP is masked below 24 stacks
    wait(agent, 2)
    ground = next(o for o in world.objs if o.type == tin and o.is_valid())
    world.take_obj(agent.player, ground)
    p = agent.player
    report = audit(world.events[first:], task_initial, dict(reason='horizon', tick=world.tick + 1, holdings=dict(
        inventory=p.inv.listed(), bank=p.bank.listed(), worn=p.worn.listed())), 'mine', tables, agent.name)
    assert report['produced'] == 1 and report['picked'] == 1 and report['retained'] == 0


def test_burying_bones_and_using_items_on_each_other(tables, site):
    from sim.agent import A_USE0
    world, agent, _ = begin(site, 'tree', dict(prayer=1), kit=KIT + [('bones', 1)])
    start = world.tick
    held(agent, 'Bones', 'Bury')
    assert said(world, start)[:2] == ['You dig a hole in the ground...', 'You bury the bones.']
    assert agent.player.stats[5] == 45 * 25 and not agent.player.inv.total(tables.obj_id('bones'))  # bone_exp at 25x
    wait(agent, 3)  # the Prayer level-up dialog
    use = lambda name: A_USE0 + next(i for i, s in enumerate(agent.last_observation['invSel']) if s['name'] == name)
    # The sword finds a script, whose message this is. The net finds none: OpHeldUHandler sends its debug line (native
    # worlds run with the engine's default node.debug) and its literal; before held-v2 no agent sent either.
    for used, target, lines in (('Bronze sword', 'Shrimps', ['Nothing interesting happens.']),
                                ('Small fishing net', 'Shrimps', ['No trigger for [opheldu,shrimp]',
                                                                  'Nothing interesting happens.'])):
        agent.step(use(used))
        start = world.tick
        obs = held(agent, target, 'Use')
        assert said(world, start)[:2] == lines, obs['actionNote']
        assert [c['text'] for c in obs['feedback']['chat']][:len(lines)] == lines  # the chat reaches the policy


def test_the_reference_controller_completes_each_goal(tables, tmp_path):
    from sim.run import reference
    summary = reference(tmp_path / 'reference', split='development', layouts=3, episodes=3, tables=tables)
    assert all(summary[goal]['successes'] >= 2 for goal in summary), summary


def test_net_fishing_needs_a_net_and_yields_with_one(tables, site):
    layout, lt = site
    for kit, net in ((KIT[:1] + KIT[2:], False), (KIT, True)):
        world = World(lt, seed=4, start_tick=WORLD_START_TICK)
        agent = Agent(world)
        agent.start(0, config.TASK_CFG, tuple(int(v) + o for v, o in zip(np.argwhere(layout.start)[0], W.ORIGIN)), kit, {})
        wait(agent, 3)  # the spots make their first move (timer 1), then stay for 280 ticks or more
        spot = next(n for n in world.npcs if tables.npc_types[n.type]['debugname'] == W.SPOT)
        tile = next(xy for xy in world.entity_reaching(0, spot.x, spot.z) if lt.source(0, *xy) >= 0)
        agent.start(0, config.TASK_CFG, tile, kit, dict(fishing=20))  # a new account beside it
        fish = [tables.obj_id(name) for name in ('raw_shrimp', 'raw_anchovies')]  # anchovies are rolled first from 15
        done = (lambda: sum(agent.player.inv.total(f) for f in fish)) if net else \
            (lambda: agent.dialog_text()['text'] == world.text('fish_fail_net'))
        for _ in range(60):  # a spot in its move delay is gone for the moment: net it again
            if done():
                break
            obs = agent.last_observation
            if obs.get('optionBusy') or obs['mask'][A_DIALOG_CONTINUE] or not any(s['name'] == 'Fishing spot' for s in obs['actSel']):
                wait(agent, 1)
            else:
                act(agent, 'Fishing spot', 'Net')
        assert done()


def test_single_way_combat_refuses_a_second_chicken_and_chickens_have_no_timer(tables, site):
    world, agent, _ = chicken_fight(tables, site, 'bronze_sword', rounds=1)
    first = next(n for n in world.npcs if n.vars['aggressive_player'] == agent.name)
    other = next(n for n in world.npcs if n.type in (41, 1017) and n is not first and n.is_valid())
    start = world.tick
    agent.player.set_interaction(other, 4)  # APNPC2 on another chicken within 8 ticks of the last hit
    world.cycle()
    assert "I'm already under attack!" in said(world, start)
    assert all(tables.npc_types[t]['timer'] == -1 for t in (41, 1017))  # [ai_timer,_chicken] never runs


def test_a_penned_chicken_is_killed_through_its_gate(tables):
    from sim.run import reference_action
    layout = next(L for L in (W.generate(s, tables) for s in W.split_seeds('development', 20)) if L.gate[0] >= 0)
    task = GoalEpisode(World(W.LayoutTables(tables, layout), seed=5, start_tick=WORLD_START_TICK), config.TASK_CFG, 'melee',
                       scale=1.0)
    task.start(tuple(int(v) + o for v, o in zip(np.argwhere(layout.start)[0], W.ORIGIN)), 'bronze_dagger', dict(attack=5))
    opened, done = False, False
    while not done:
        _, _, done, info = task.step(reference_action(task))
        opened |= bool(task.world.changed)
    assert opened and info['report']['kills']['lower'] >= 1 and info['episode']['r'] == 1.0


def test_an_episode_ends_at_its_terminal_cycle(tables, site):
    """a decision still waiting on its option at the horizon stops at the terminal cycle; no later
    tick is simulated (here any would raise)."""
    from sim.engine import Unsupported
    layout, lt = site
    world = World(lt, seed=2, start_tick=WORLD_START_TICK)
    task = GoalEpisode(world, config.TASK_CFG, 'wood', horizon=20, scale=1.0)
    cycle = world.cycle

    def guarded():
        if task.armed and world.tick >= task.end:
            raise Unsupported('a tick after the terminal cycle')
        cycle()
    world.cycle = guarded
    tree = next(loc for loc in world.locs if world.t.loc_types[loc.type]['debugname'] == 'tree')
    task.start(next(xy for xy in world.loc_reaching(tree) if lt.source(0, *xy) >= 0), 'bronze_sword', {})
    k = next(i for i, sel in enumerate(task.agent.last_observation['actSel']) if sel['option'] == 'Chop down')
    _, _, done, info = task.step(A_ACT0 + k)  # the chop keeps the decision waiting beyond 20 ticks
    while not done:
        _, _, done, info = task.step(0)
    assert world.tick == task.end and info['report']['reason'] == 'horizon' and info['episode']['native_ticks'] == 20


def test_setup_recomputes_combat_stats_so_the_first_attack_uses_the_set_levels(tables, site):
    """Native engine: at attack/strength/defence 10 and hitpoints 12 the STYLE_0 setup step moves the
    varps from a new player's (max hit 1, stab attack 768, stab defence 576) to 2, 1344 and 1152."""
    layout, lt = site
    levels = dict(attack=10, strength=10, defence=10, hitpoints=12)
    arms = []
    for seed in range(40):
        task = GoalEpisode(World(lt, seed=seed, start_tick=WORLD_START_TICK), config.TASK_CFG, 'melee', scale=1.0)
        hen = next(n for n in task.world.npcs if tables.npc_types[n.type]['debugname'] in W.CHICKENS)
        tile = next(xy for xy in task.world.entity_reaching(0, hen.x, hen.z) if lt.source(0, *xy) >= 0)
        task.start(tile, 'bronze_dagger', levels)
        arms.append(task.start_tick)
        p = task.agent.player
        assert (p.varps['com_maxhit'], p.varps['com_stabattack'], p.varps['com_stabdef']) == (2, 1344, 1152)
    assert set(arms) == {12, 13}  # K_TICKS plus 0 or 1 pace jitter after the reset observation at tick 10
    draws, randominc = [], task.world.randominc
    task.world.randominc = lambda n, *keys: draws.append((n, keys[-1])) or randominc(n, *keys)
    k = next(i for i, s in enumerate(task.agent.last_observation['actSel']) if s['option'] == 'Attack')
    task.step(A_ACT0 + k)
    rolls = dict((key, n) for n, key in draws)
    assert rolls[1] == 1344 and rolls.get(3, 2) == 2  # the hit roll, and the damage roll when it hits
