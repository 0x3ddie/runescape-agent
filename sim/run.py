"""Play a policy or a control in a simulated goal task (sim/task.py).

    .venv-sim/bin/python -m sim.run --policy random --goal wood --episodes 20 --output logs/sim/runs/random-wood
    .venv-sim/bin/python -m sim.run --policy checkpoint --checkpoint RUN/latest.zip \\
        --sampled --goal fish --episodes 20 --output logs/sim/runs/fish
    .venv-sim/bin/python -m sim.run reference --split development --layouts 50 --episodes 50 --output logs/sim/runs/reference
    .venv-sim/bin/python -m sim.run goals --checkpoint RUN/latest.zip --initial RUN/initial.zip --output logs/sim/runs/eval-s101

The policy is baseline.evaluate.Actor unchanged (PyTorch, recurrent memory reset per
episode). Each episode starts in its own fresh world, as the native evaluations do, unless
--persist keeps one world across episodes, as native training does. Writes results.json
(per-episode outcome, action families, coverage failures, timing) and episodes.jsonl.gz.

`reference` runs the four-goal task's reference controller on generated layouts: a validator of
task feasibility within the horizon, never a source of training labels. It reads the world to
choose its targets and acts only through legal published actions.

`goals` is the four-goal evaluation: a training seed's final checkpoint, its
initialization, and random-legal and WAIT controls, each on the same physical start records of
generated layouts for every goal (20 by default; --one-per-layout gives each layout one record), in
fresh worlds. Per trial: success (survived, target certified),
the first activity completed (baseline.evaluate's rule, from the episode's own events), controlled
success, ticks to the outcome, other activities' production and kills, refused options and repeated
actions before it. Per policy and goal: counts with a 95% interval; for the final checkpoint the
acceptance (controlled successes in at least 80% of the starts for every goal, 16 of 20) and its
per-start pairing with each control. The
development split is the default; the final split needs --final (final evaluation data). --goals plays a subset of
the goals. A trial's sampling seed is --seed + 1,000 x start + the goal's index in GOALS; with --policy-seed it is
--seed + 1,000 x the policy seed + start, the same for
every goal and policy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from baseline import config
from baseline.compare import wilson
from baseline.evaluate import Actor, first_completed, repeats
from baseline.resource_task import GOAL_FAMILIES, GOAL_TARGETS, GOALS, MELEE_ENEMIES
from baseline.world import JsonlEvidence
from sim.audit import kill_ticks
from act_contract import CLICK_ROWS
from sim import paths
from sim.agent import A_ACT0, A_CLOSE_IF, A_DIALOG_CONTINUE, A_HELD0, family
from sim.data import Tables
from sim.engine import MINING, NPC_HP, WEARPOS_RHAND, WOODCUTTING, Unsupported, World, rock_row
from sim.spec import HORIZON_TICKS, START_WEAPONS
from sim.task import WORLD_START_TICK, GoalEpisode, draw_start
from sim.world import OAK, ORIGIN, PEN_GATE, SPLITS, TREE, LayoutTables, generate, split_seeds, with_distractions

MAX_DECISIONS = 2400


def run(output, *, policy, goal, episodes, checkpoint=None, sampled=False, seed=101001, persist=False, tables=None):
    output.mkdir(parents=True, exist_ok=False)
    tables = tables or Tables.load()
    model = None
    if policy == 'checkpoint':
        from sb3_contrib import RecurrentPPO
        from baseline.policy import check_names
        model = RecurrentPPO.load(checkpoint, device='cpu')
        check_names(model.policy)
    actor = Actor(policy, model, deterministic=not sampled)
    evidence = JsonlEvidence(output / 'episodes.jsonl.gz')
    results, world, started = [], None, time.monotonic()
    cfg = dict(config.TASK_CFG)
    for k in range(episodes):
        if world is None or not persist:
            world = World(tables, seed=seed + 1000 * k)
        task = GoalEpisode(world, cfg, goal)
        t0, families, outcome = time.monotonic(), Counter(), None
        try:
            obs = task.reset(0)
        except Unsupported as error:
            results.append(dict(episode=k, coverage_failure=f'setup: {error}'))
            continue
        actor.fixture = dict(goal=goal)
        actor.reset(seed + 1000 * k)
        setup = time.monotonic() - t0
        paid = 0.
        for index in range(MAX_DECISIONS):
            action = actor.act(obs, task)
            families[family(action)] += 1
            try:
                obs, reward, done, info = task.step(action)
            except Unsupported as error:  # counted here; never a scored outcome
                outcome = dict(coverage_failure=str(error))
                break
            paid += reward
            evidence('effect', episode=k, decision=index, action=action, reward=reward, tick=task.world.tick,
                     note=task.raw.get('actionNote'), chat=[c['text'] for c in task.raw['feedback']['chat']],
                     chat_ages=[c['age'] for c in task.raw['feedback']['chat']])
            if done:
                outcome = info
                break
        row = dict(episode=k, decisions=task.decisions, paid=paid, setup_seconds=setup,
                   seconds=time.monotonic() - t0, families=dict(families),
                   coverage_failure=outcome.get('coverage_failure') if outcome else 'decision limit')
        if outcome and 'report' in outcome:
            report = outcome['report']
            row.update(report=report, episode_return=outcome['episode']['r'],
                       success=report['survived'] and report['retained'] >= 2)
            if abs(paid - outcome['episode']['r']) > 1e-6:
                raise RuntimeError('paid rewards disagree with the audited return')
        results.append(row)
        evidence('episode', **row)
        print(f"[sim] episode {k}: decisions {row['decisions']} return {row.get('episode_return')} "
              f"coverage {row['coverage_failure']} {row['seconds']:.1f}s", flush=True)
    evidence.close()
    done = [r for r in results if 'report' in r]
    summary = dict(policy=policy, checkpoint=str(checkpoint) if checkpoint else None, goal=goal, episodes=episodes,
                   sampled=sampled, persist=persist, completed=len(done),
                   coverage_failures=Counter(r['coverage_failure'] for r in results if r.get('coverage_failure')),
                   successes=sum(r['success'] for r in done), deaths=sum(not r['report']['survived'] for r in done),
                   mean_return=sum(r['episode_return'] for r in done) / max(1, len(done)),
                   decisions=sum(r.get('decisions', 0) for r in results), seconds=time.monotonic() - started,
                   trials=results)
    (output / 'results.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    return summary


# ---- the four-goal task's reference controller (a validator) -------------------------------------------
def reference_action(task):
    """The next action of the reference controller (module docstring)."""
    a, w, p = task.agent, task.world, task.agent.player
    obs = a.last_observation
    mask = obs['mask']
    if mask[A_DIALOG_CONTINUE]:
        return A_DIALOG_CONTINUE
    if obs['dialogOpen'] or obs['modalOpen']:
        return A_CLOSE_IF
    if task.goal == 'melee' and p.worn.items[WEARPOS_RHAND] is None:
        wield = [k for k, h in enumerate(obs['heldSel']) if h['option'] == 'Wield' and h['id'] in WEAPON_IDS(w)]
        if wield:
            return A_HELD0 + wield[0]
    if obs.get('optionBusy'):
        return 0
    targets = reference_targets(task)
    for k, sel in enumerate(obs['actSel']):
        if (sel['kind'], sel['x'], sel['z'], sel['option']) in targets:
            return A_ACT0 + k
    return click_toward(task, {(t[1], t[2]) for t in targets}) if targets else 0


def click_toward(task, tiles):
    """The reference's walk under act-v3 (privileged): of the target tiles, the one the player's client search reaches
    in the fewest steps (beside it; ties by position), then the legal click whose route ends the fewest steps from the
    tile beside it that the player reaches first (the client's search from that tile, so a target in a fence, a closed
    gate, is measured from outside it; ties to the lower slot), if that end is nearer than the player; else WAIT."""
    a, w, p = task.agent, task.world, task.agent.player
    obs = a.last_observation
    tree = w.player_scene(p)
    around = lambda t: [(tree.at(tree.dist, t[0] + dx, t[1] + dz, paths.SCENE_UNREACHED), (t[0] + dx, t[1] + dz))
                        for dx in (-1, 0, 1) for dz in (-1, 0, 1)]
    goal = min(tiles, key=lambda t: (min(around(t))[0], t))
    steps, near = min(around(goal))
    if steps >= paths.SCENE_UNREACHED:
        return 0
    back = w.scene_tree(p.level, *near, tree.base)
    steps_to = lambda x, z: back.at(back.dist, x, z, paths.SCENE_UNREACHED)
    best, choice = steps_to(p.x, p.z), 0
    for k, c in enumerate(obs['clicks']):
        if not obs['mask'][CLICK_ROWS[k]]:
            continue
        route = paths.client_route(tree, p.x, p.z, (p.x + c['dx'], p.z + c['dz']), nearest=True)
        if route and steps_to(*route[-1]) < best:
            best, choice = steps_to(*route[-1]), CLICK_ROWS[k]
    return choice


def WEAPON_IDS(world):
    return {world.t.obj_id(name) for name in START_WEAPONS}


def reference_targets(task):
    """(kind, x, z, option) of every thing that serves the goal now: a normal tree (an oak too from
    Woodcutting 15), a fishing spot, a rock with ore the Mining level allows, a closed pen gate, then the
    chicken it fights or one at full health."""
    w, p = task.world, task.agent.player
    if task.goal == 'fish':
        return {('npc', n.x, n.z, 'Net') for n in w.npcs if n.is_valid() and w.t.npc_types[n.type]['name'] == 'Fishing spot'}
    if task.goal == 'melee':  # a closed pen gate first; the chicken it fights (single-way combat refuses
        gate = {('loc', loc.x, loc.z, 'Open') for loc in w.locs if loc.active and w.t.loc_types[loc.type]['debugname'] in PEN_GATE}
        if gate:  # another), else a fresh one
            return gate
        hens = [n for n in w.npcs if n.is_valid() and n.type in MELEE_ENEMIES and n.target in (None, p)]
        engaged = [n for n in hens if p.name in n.heroes or n.vars['aggressive_player'] == p.name]
        return {('npc', n.x, n.z, 'Attack') for n in engaged or hens
                if engaged or n.levels[NPC_HP] == w.t.npc_types[n.type]['stats'][NPC_HP]}
    out = set()
    for loc in w.locs:
        name = w.t.loc_types[loc.type]['debugname']
        if not loc.active:
            continue
        if task.goal == 'wood' and (name == TREE or (name == OAK and p.levels[WOODCUTTING] >= 15)):
            out.add(('loc', loc.x, loc.z, 'Chop down'))
        if task.goal == 'mine':
            row = rock_row(w, loc)
            if row is not None and row['level'] <= p.levels[MINING]:
                out.add(('loc', loc.x, loc.z, 'Mine'))
    return out


def reference(output, *, split, layouts, episodes, key=1, horizon=HORIZON_TICKS, goals=GOALS, tables=None):
    """The controller on `episodes` starts per goal (task.draw_start over the split's first `layouts`)."""
    output.mkdir(parents=True, exist_ok=False)
    tables = tables or Tables.load()
    pool = [generate(seed, tables) for seed in split_seeds(split, layouts)]
    lts, rows = {}, []
    for goal in goals:
        for k in range(episodes):
            start = draw_start(key, GOALS.index(goal), k, pool)
            lt = lts.setdefault(start['layout'], LayoutTables(tables, pool[start['layout']]))
            task = GoalEpisode(World(lt, seed=start['world_seed'], start_tick=WORLD_START_TICK), config.TASK_CFG, goal,
                               horizon=horizon, scale=1.0)
            task.start(start['tile'], start['weapon'], start['levels'])
            done, completed = False, None
            try:
                while not done:
                    _, _, done, info = task.step(reference_action(task))
                    if completed is None and task.held() >= 1.0:
                        completed = task.world.tick - task.start_tick
            except Unsupported as error:  # counted apart, never scored
                rows.append(dict(goal=goal, episode=k, layout=pool[start['layout']].seed, start=start, success=None,
                                 completed_ticks=completed, unsupported=str(error)))
                continue
            rows.append(dict(goal=goal, episode=k, layout=pool[start['layout']].seed, start=start,
                             success=info['episode']['r'] == 1.0, completed_ticks=completed, outcomes=info['outcomes'],
                             decisions=info['episode']['l'], reason=info['report']['reason']))
    summary = {}
    for goal in goals:
        mine = [r for r in rows if r['goal'] == goal]
        # first seen at a decision's end, so an upper bound; successes only (a late decision may end past the horizon)
        ticks = sorted(r['completed_ticks'] for r in mine if r['success'])
        summary[goal] = dict(successes=sum(bool(r['success']) for r in mine), episodes=len(mine),
                             unsupported=sum(r['success'] is None for r in mine),
                             median_ticks=ticks[len(ticks) // 2] if ticks else None, max_ticks=ticks[-1] if ticks else None)
    (output / 'results.json').write_text(json.dumps(dict(split=split, layouts=layouts, key=key, horizon=horizon,
                                                         summary=summary, trials=rows), indent=1, default=str) + '\n')
    return summary


# ---- the four-goal evaluation ----------------------------------------------------------
EVAL_POLICIES = ('final', 'initial', 'random', 'wait')  # a training seed's checkpoint and its three controls


def bar(starts):
    """The acceptance bar: controlled successes in at least 80% of the starts, rounded up (16 of 20,
    160 of 200)."""
    return -(-4 * starts // 5)


class ReferenceActor:
    """The reference controller as an evaluation policy (tests of the evaluation; never labels)."""

    def reset(self, seed):
        pass

    def act(self, observation, task):
        return reference_action(task)


def family_ids(tables, goal):
    """A gathering goal's items and their notes (as GoalEpisode)."""
    family = set(GOAL_FAMILIES[goal])
    for base in list(family):
        note = tables.obj_types.get(base, {}).get('certlink', -1)
        if note is not None and note >= 0:
            family.add(note)
    return family


def goal_control(task, refusals, decisions=()):
    """Goal control of one finished episode from its world's events (baseline.evaluate.goal_control's rule):
    each activity's production (its families' produced items) and certified kill ticks inside the episode,
    the tick each first reached its target, the first activity to reach one (first_completed: 'tie' or
    None), ticks to the requested outcome, other activities' production and kills before it, the
    refused options before it and the repeated actions before it (baseline.evaluate.repeats, from the
    decisions' (tick, action))."""
    window = [e for e in task.world.events[task.event_start:] if e[0] < task.terminal['tick']]
    events = {}
    for goal in GOALS:
        if goal == 'melee':
            events[goal] = kill_ticks(window, task.agent.name)
        else:
            family = family_ids(task.world.t, goal)
            events[goal] = sorted(tick for tick, kind, data in window if kind == 'produce' and data['obj'] in family
                                  for _ in range(data['count']))
    reached = {g: ticks[GOAL_TARGETS[g] - 1] if len(ticks) >= GOAL_TARGETS[g] else None for g, ticks in events.items()}
    done = reached[task.goal]
    before = lambda tick: done is None or tick < done
    return dict(first_completed=first_completed(reached),
                ticks_to_outcome=None if done is None else done - task.start_tick,
                completion_ticks={g: None if t is None else t - task.start_tick for g, t in reached.items()},
                non_goal_before_outcome={g: sum(map(before, ticks)) for g, ticks in events.items() if g != task.goal},
                refusals_before_outcome=sum(1 for tick in refusals if done is None or tick <= done),
                **repeats(decisions, done))


def start_records(tables, *, split, layouts, first, key, starts, one_per_layout):
    """The layouts (generate over `layouts` seeds of the split from its index `first`) and the `starts` physical start
    records that goals plays on them (task.draw_start, slot 0, episode k; with one_per_layout record k on layout k)."""
    pool = [generate(s, tables) for s in split_seeds(split, layouts, first)]
    return pool, [draw_start(key, 0, k, pool, layout=k if one_per_layout else None) for k in range(starts)]


def trial_seed(seed, start, goal, policy_seed=None):
    """A goals trial's sampling seed (module docstring): seed + 1,000 x start + the goal's index, or with a policy seed
    seed + 1,000 x policy_seed + start."""
    return seed + 1000 * start + GOALS.index(goal) if policy_seed is None else seed + 1000 * policy_seed + start


def goals(output, *, checkpoint=None, initial=None, policies=EVAL_POLICIES, split='development', layouts=20, starts=20,
          key=7, sampled=False, seed=202001, first=0, one_per_layout=False, distractions=(), tables=None, previous=None,
          goals=GOALS, policy_seed=None):
    """Every policy on the same `starts` physical start records (task.draw_start over `layouts` layouts of the
    split from its seed index `first`; slot 0, episode k), each played for every goal of `goals` in a fresh world. The
    draw picks a layout for each record, with replacement; with `one_per_layout` record k uses layout k
    (starts must equal layouts), so the records test that many distinct layouts. A trial's seed is trial_seed's. A
    trial succeeds when it survives with the goal's target certified (return 1); it is a controlled success when the requested
    activity is also the first completed. An episode the simulator cannot model is counted apart, never
    scored, and makes its cell invalid. `final` is `checkpoint`, `initial` its initialization, `previous` another
    checkpoint on the same records (an earlier campaign's final policy).
    episodes.jsonl.gz keeps each trial's decisions (tick, action, reward, ended option, the action's note: what it
    requested of which target) and its audit report. `distractions` (with one_per_layout only): each layout becomes its
    sim.world with_distractions variant around its record's start, for a cause test on development layouts."""
    if one_per_layout and starts != layouts:
        raise ValueError('one start per layout needs as many starts as layouts')
    if not goals or not set(goals) <= set(GOALS):
        raise ValueError(f'goals are some of {GOALS}')
    played = tuple(g for g in GOALS if g in goals)
    output.mkdir(parents=True, exist_ok=False)
    tables = tables or Tables.load()
    seeds = split_seeds(split, layouts, first)
    pool, records = start_records(tables, split=split, layouts=layouts, first=first, key=key, starts=starts,
                                  one_per_layout=one_per_layout)
    if distractions:
        if not one_per_layout:
            raise ValueError('distractions are placed around each layout\'s one start: they need one_per_layout')
        pool = [with_distractions(lay, tables, distractions, (r['tile'][0] - ORIGIN[0], r['tile'][1] - ORIGIN[1]))
                for lay, r in zip(pool, records)]
    evidence = JsonlEvidence(output / 'episodes.jsonl.gz')
    lts, rows, timing = {}, [], {}
    from sb3_contrib import RecurrentPPO
    from baseline.policy import check_names
    from baseline.telemetry import sha256
    paths = dict(final=checkpoint, initial=initial, previous=previous)

    def actor_for(policy):
        if policy == 'reference':
            return ReferenceActor()
        if policy in paths:
            model = RecurrentPPO.load(paths[policy], device='cpu')
            check_names(model.policy)
            return Actor('checkpoint', model, deterministic=not sampled)
        return Actor(policy)
    for policy in policies:
        actor, t0 = actor_for(policy), time.monotonic()
        for goal in played:
            for k, start in enumerate(records):
                lt = lts.setdefault(start['layout'], LayoutTables(tables, pool[start['layout']]))
                task = GoalEpisode(World(lt, seed=start['world_seed'], start_tick=WORLD_START_TICK), config.TASK_CFG, goal,
                                   horizon=HORIZON_TICKS, scale=1.0)
                row = dict(policy=policy, goal=goal, start=k, layout=pool[start['layout']].seed, record=start)
                obs = task.start(start['tile'], start['weapon'], start['levels'])
                actor.fixture = dict(goal=goal)
                actor.reset(trial_seed(seed, k, goal, policy_seed))
                refusals, families, paid, info, trace = [], Counter(), 0., None, []
                try:
                    for _ in range(MAX_DECISIONS):
                        action, tick = actor.act(obs, task), task.world.tick
                        families[family(action)] += 1
                        obs, reward, done, info = task.step(action)
                        paid += reward
                        ended = task.raw['feedback'].get('ended') if not done else None
                        trace.append((tick, action, reward, ended, task.raw.get('actionNote')))
                        if ended and ended['refused']:
                            refusals.append(task.world.tick)
                        if done:
                            break
                except Unsupported as error:  # counted apart, never scored
                    rows.append(dict(row, unsupported=str(error), families=dict(families)))
                    evidence('trial', **row, decisions=trace, unsupported=str(error))
                    continue
                if not (info and 'report' in info):
                    rows.append(dict(row, decision_limit=True, families=dict(families)))
                    evidence('trial', **row, decisions=trace, decision_limit=True)
                    continue
                if abs(paid - info['episode']['r']) > 1e-6:
                    raise RuntimeError('paid rewards disagree with the audited return')
                succeeded = info['episode']['r'] == 1.0
                control = goal_control(task, refusals, [(tick, action) for tick, action, *_ in trace])
                evidence('trial', **row, decisions=trace, report=info['report'], outcomes=info['outcomes'])
                rows.append(dict(row, success=succeeded, controlled_success=succeeded and control['first_completed'] == goal,
                                 survived=info['report']['survived'], outcomes=info['outcomes'], episode_return=info['episode']['r'],
                                 decisions=info['episode']['l'], ticks=info['episode']['native_ticks'], families=dict(families),
                                 **control))
        timing[policy] = time.monotonic() - t0
        print(f'[sim] goals: {policy} done in {timing[policy]:.0f}s', flush=True)
    cells = {}
    for policy in policies:
        for goal in played:
            mine = [r for r in rows if r['policy'] == policy and r['goal'] == goal]
            scored = [r for r in mine if 'success' in r]
            controlled = sum(r['controlled_success'] for r in scored)
            ticks = sorted(r['ticks_to_outcome'] for r in scored if r['controlled_success'])
            cells.setdefault(policy, {})[goal] = dict(
                trials=len(mine), scored=len(scored), unsupported=sum('unsupported' in r for r in mine),
                decision_limit=sum('decision_limit' in r for r in mine), successes=sum(r['success'] for r in scored),
                controlled_successes=controlled, controlled_interval=wilson(controlled, len(scored)),
                deaths=sum(not r['survived'] for r in scored),
                first_completed=dict(Counter(str(r['first_completed']) for r in scored)),
                median_ticks_to_outcome=ticks[len(ticks) // 2] if ticks else None, valid=len(scored) == len(mine))
    evidence.close()
    from sim.agent import HELD_CONTRACT, MASK_CONTRACT
    from sim.engine import RETALIATE_CONTRACT
    result = dict(split=split, layouts=layouts, starts=starts, key=key, sampled=sampled, seed=seed, bar=bar(starts),
                  mask=MASK_CONTRACT, retaliate=RETALIATE_CONTRACT, held=HELD_CONTRACT,
                  first=first, one_per_layout=one_per_layout, layout_seeds=seeds, distractions=list(distractions),
                  checkpoints={name: dict(path=str(p), sha256=sha256(p)) for name, p in paths.items() if p},
                  cells=cells, timing_seconds=timing, decisions=sum(r.get('decisions', 0) for r in rows))
    if played != GOALS or policy_seed is not None:  # only beside the defaults, so a default output keeps its keys
        result.update(goals=list(played), policy_seed=policy_seed)
    if 'final' in policies:
        final = cells['final']
        result['acceptance'] = dict(passed=all(final[g]['valid'] and final[g]['controlled_successes'] >= bar(starts)
                                               for g in played),
                                    per_goal={g: final[g]['controlled_successes'] for g in played})
        by = {(r['policy'], r['goal'], r['start']): r.get('controlled_success') for r in rows}
        result['paired'] = {control: {goal: dict(
            final_only=sum(by[('final', goal, k)] is True and by[(control, goal, k)] is False for k in range(starts)),
            control_only=sum(by[('final', goal, k)] is False and by[(control, goal, k)] is True for k in range(starts)),
            both=sum(by[('final', goal, k)] is True and by[(control, goal, k)] is True for k in range(starts)),
            neither=sum(by[('final', goal, k)] is False and by[(control, goal, k)] is False for k in range(starts)))
            for goal in played} for control in policies if control != 'final'}
    (output / 'results.json').write_text(json.dumps(dict(result, trials=rows), indent=1, default=str) + '\n')
    return result


def main():
    import sys
    if sys.argv[1:2] == ['goals']:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('command')
        parser.add_argument('--checkpoint', type=Path, help='the final checkpoint (policy `final`)')
        parser.add_argument('--initial', type=Path, help='its initialization (policy `initial`)')
        parser.add_argument('--previous', type=Path, help='another checkpoint on the same records (policy `previous`)')
        parser.add_argument('--policies', nargs='+', choices=(*EVAL_POLICIES, 'previous', 'reference'), default=list(EVAL_POLICIES))
        parser.add_argument('--split', choices=SPLITS, default='development')
        parser.add_argument('--final', action='store_true', help='the final split is final evaluation data: say so')
        parser.add_argument('--layouts', type=int, default=20)
        parser.add_argument('--starts', type=int, default=20)
        parser.add_argument('--key', type=int, default=7)
        parser.add_argument('--first', type=int, default=0, help='the split\'s seed index of the first layout')
        parser.add_argument('--one-per-layout', action='store_true',
                            help='start record k on layout k (--starts equal to --layouts): distinct tested layouts')
        parser.add_argument('--distractions', nargs='+', default=[], choices=('npcs',),
                            help='development variants: the additions of sim.world DISTRACTIONS around each start')
        parser.add_argument('--sampled', action='store_true', help='sample the checkpoints\' actions (default: likeliest)')
        parser.add_argument('--seed', type=int, default=202001, help='the sampling-seed base of --sampled')
        parser.add_argument('--policy-seed', type=int,
                            help='a trial\'s seed is --seed + 1,000 x this + start (default: --seed + 1,000 x start + goal index)')
        parser.add_argument('--goals', nargs='+', choices=GOALS, default=list(GOALS), help='the goals played (default: all)')
        parser.add_argument('--tables', type=Path, help='the simulator archive as given, away from the checkout (its pins unchecked)')
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args()
        if (args.split == 'final') != args.final:
            parser.error('the final split needs --final, and --final needs the final split')
        for policy, path, flag in (('final', args.checkpoint, 'checkpoint'), ('initial', args.initial, 'initial'),
                                   ('previous', args.previous, 'previous')):
            if policy in args.policies and path is None:
                parser.error(f'policy {policy} needs --{flag}')
        result = goals(args.output, checkpoint=args.checkpoint, initial=args.initial, previous=args.previous,
                       policies=tuple(args.policies),
                       split=args.split, layouts=args.layouts, starts=args.starts, key=args.key, sampled=args.sampled, seed=args.seed,
                       first=args.first, one_per_layout=args.one_per_layout, distractions=tuple(args.distractions),
                       tables=Tables.load(args.tables, check_pins=False) if args.tables else None,
                       goals=tuple(args.goals), policy_seed=args.policy_seed)
        print(json.dumps({k: v for k, v in result.items() if k != 'paired'}, indent=1, default=str))
        return
    if sys.argv[1:2] == ['reference']:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('command')
        parser.add_argument('--split', choices=SPLITS, default='development')
        parser.add_argument('--layouts', type=int, default=50)
        parser.add_argument('--episodes', type=int, default=50, help='per goal')
        parser.add_argument('--key', type=int, default=1)
        parser.add_argument('--horizon', type=int, default=HORIZON_TICKS, help='a measurement, not the task horizon')
        parser.add_argument('--goals', nargs='+', choices=GOALS, default=list(GOALS))
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args()
        print(json.dumps(reference(args.output, split=args.split, layouts=args.layouts, episodes=args.episodes,
                                   key=args.key, horizon=args.horizon, goals=args.goals), indent=1))
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', choices=('checkpoint', 'random', 'wait'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--goal', choices=('wood', 'fish'), required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--sampled', action='store_true')
    parser.add_argument('--seed', type=int, default=101001)
    parser.add_argument('--persist', action='store_true', help='one world across episodes, as native training')
    args = parser.parse_args()
    summary = run(args.output, policy=args.policy, goal=args.goal, episodes=args.episodes, checkpoint=args.checkpoint,
                  sampled=args.sampled, seed=args.seed, persist=args.persist)
    print(json.dumps({k: v for k, v in summary.items() if k != 'trials'}, indent=1, default=str))


if __name__ == '__main__':
    main()
