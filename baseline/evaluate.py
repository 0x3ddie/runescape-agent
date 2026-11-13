"""Evaluate a checkpoint, or a WAIT/random control, on the fixed development fixtures.

Runs inside a world booted by scripts/stack.py. Every trial is a fresh first life:
death ends it with zero reward. Writes results.json (with the run manifest) and
trajectory.jsonl.gz (every row, with its episode id, plus 'effect' and 'episode_measures'
rows from baseline/telemetry.py) into --output and prints one line per trial plus a
per-fixture summary. Compare two evaluations with baseline/compare.py.


Success depends on the fixture's kind (from the original study design), counted in
certified items kept of any produced kind (value scoring reports them per item):
  short  survived, >=2 certified retained items, no banking actions executed
  long   survived, >=32 certified retained items, deposited, then produced more
  drop   survived, >=10 certified retained items (the starter kit leaves 10 slots free)
  goal   survived, the requested activity's target: >=2 certified retained items of its families, or for
         melee >=1 certified kill (goal-v2); banking allowed
The axe_banked and guard fixtures are long: they add a trip for the axe or the
walk from the jail road before that bar. Goal fixtures run only with goal checkpoints (they
read goal_vec) or controls, never mixed with the others. --policy reference plays a goal
fixture with the scripted teacher limited to the requested skill: a task check, never labels. --policy edge
plays the Draynor goal start with a walk by clicks to EDGE_POINT, past the simulator's supported tiles: the
native-click fidelity recording's click near the support edge (sim.check replay), never labels. --policy chase plays that
start with one attack on a Dark wizard about 8 tiles away (with RL_TICK_TRACE set on the stack, for sim.check chase).

A four-goal fixture (config.START_SET_FIXTURES) cycles through its start set's declared records, the same
for every goal of the set, and runs only on the set's engine build (config.ENGINE_TREES), only for a goal
the set supports and, being final evaluation data, only with --final. Each trial reports goal control
from the native trace: the first
activity to reach its threshold (two produced items of its families, or one certified kill; 'tie' when
two reach it on the same tick), controlled success (that activity is the requested one and the trial
succeeds), native ticks to the requested outcome, other activities' production and kills before it,
and the options refused before it.
"""
import argparse
from collections import Counter
import hashlib
import os
import subprocess
import json
from pathlib import Path
import time

import numpy as np

from baseline import config
from baseline.resource_task import GOAL_FAMILIES, GOAL_TARGETS, goal_items
from baseline.telemetry import EpisodeRecorder, manifest
from baseline.world import (FreshWorlds, JsonlEvidence, connect, delivery_report, join, make_episode, wait_for_spectator,
                            worlds)
from act_contract import CLICK_ROWS

POLICIES = ('checkpoint', 'wait', 'random', 'reference', 'edge', 'chase')
# 15 tiles west of the Draynor tables' supported tiles (x 3040-3150, z 3180-3300; Tables.source), on the start's row.
EDGE_POINT = (3025, 3243)
GOAL_SKILL = dict(wood='woodcutting', fish='fishing')


class Actor:
    """Chooses legal actions; recurrent state resets at the start of every trial."""
    def __init__(self, policy, model=None, *, deterministic=True):
        if policy not in POLICIES or (policy == 'checkpoint') != (model is not None):
            raise ValueError('checkpoint policy requires a model; controls take none')
        self.policy, self.model, self.deterministic = policy, model, deterministic

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.state, self.first = None, True
        if self.policy == 'reference' and 'start_set' in self.fixture:
            from baseline.scripted import GoalReference  # a four-goal start set: its validator
            self.teacher = GoalReference(self.fixture['goal'], config.START_SETS[self.fixture['start_set']])
        elif self.policy == 'edge':
            from baseline.scripted import Walker
            if 'start_set' in self.fixture or self.fixture.get('village', 'draynor') != 'draynor':
                raise ValueError('the edge walk starts at Draynor')
            self.teacher = Walker(EDGE_POINT)
        elif self.policy == 'chase':
            from baseline.scripted import Chaser
            self.teacher = Chaser('Dark wizard', 8)
        elif self.policy == 'reference':
            from baseline.scripted import SKILLS, Teacher
            skill = GOAL_SKILL[self.fixture['goal']]  # evaluate() sets the fixture before each trial
            self.teacher = Teacher({skill: config.GATHER_AREAS[skill][self.fixture.get('village', 'draynor')]},
                                   skills={skill: SKILLS[skill]})
        if self.model is not None:
            import torch
            torch.manual_seed(seed)

    def probabilities(self, observation):
        """The policy's action probabilities, from the memory it holds now."""
        import torch
        policy = self.model.policy
        observation = {**observation, **{k: np.zeros(space.shape, space.dtype)
                                         for k, space in self.model.observation_space.spaces.items() if k not in observation}}
        obs, _ = policy.obs_to_tensor(observation)
        state = self.state or (np.zeros(policy.lstm_hidden_state_shape),) * 2
        with torch.no_grad():
            distribution, _ = policy.get_distribution(
                obs, tuple(torch.tensor(s, dtype=torch.float32) for s in state), torch.tensor([float(self.first)]))
        return distribution.distribution.probs[0].numpy()

    def click_probability(self, observation):
        """The policy's probability of a click, where one is legal (act-v3); else None."""
        rows = list(CLICK_ROWS)
        if self.policy != 'checkpoint' or not observation['mask'][rows].any():
            return None
        return float(self.probabilities(observation)[rows].sum())

    def act(self, observation, task=None):
        if self.policy == 'wait':
            action = 0
        elif self.policy == 'random':
            action = int(self.rng.choice(np.flatnonzero(observation['mask'])))
        elif self.policy in ('reference', 'edge', 'chase'):
            action = self.teacher.action(task.raw, task.clock.end - task.clock.last_tick)
        else:
            # Checkpoints trained with the KL penalty also expect ref_probs; it is not a network input.
            missing = {k: np.zeros(space.shape, space.dtype)
                       for k, space in self.model.observation_space.spaces.items() if k not in observation}
            observation = {**observation, **missing}
            action, self.state = self.model.predict(observation, state=self.state,
                episode_start=np.asarray([self.first]), deterministic=self.deterministic)
            action = int(np.asarray(action).item())
        self.first = False
        return action


def run_trial(task, fixture, actor, seed, record, max_decisions):
    observation = task.reset(fixture['reset_seed'])
    actor.reset(seed)
    paid = 0.
    # The game's own progress measure, reported beside base value (it is not the reward).
    xp_start = {s['name']: s['xp'] for s in task.raw.get('skills', [])}
    decisions = []  # (native tick of the observation, action)
    for index in range(max_decisions):
        probability = actor.click_probability(observation)
        action = actor.act(observation, task)
        tick = getattr(getattr(task, 'clock', None), 'last_tick', None)
        decisions.append((tick, action))
        record('evaluation_decision', index=index, action=action, click_probability=probability, tick=tick)
        observation, reward, done, info = task.step(action)
        paid += reward
        if done:
            # Rewards are paid as earned; together they must equal the certified total.
            if abs(paid - info['reward_bounds'][0]) > 1e-6:
                raise RuntimeError('paid rewards disagree with the certified total')
            xp_end = {s['name']: s['xp'] for s in task.raw.get('skills', [])}
            xp_gained = {name: xp_end[name] - xp for name, xp in xp_start.items() if xp_end.get(name, xp) > xp}
            return dict(decisions=index + 1, reward=paid, reward_bounds=info['reward_bounds'],
                        report=info['resource_report'], xp_gained=xp_gained, decision_ticks_actions=decisions)
    raise RuntimeError('evaluation decision limit before native terminal')


def items(report):
    """Certified retained items; value scoring's retained_lower is in coins."""
    families = report.get('families')
    return report['retained_lower'] if families is None else sum(f['retained_lower'] for f in families.values())


def success(kind, report, bank_actions, goal=None):
    if not report['survived']:
        return False
    if kind == 'goal':
        return goal_items(report, goal) >= GOAL_TARGETS[goal]
    if kind == 'short':
        return items(report) >= 2 and bank_actions == 0
    if kind == 'drop':
        return items(report) >= 10
    return items(report) >= 32 and report['deposit_events'] >= 1 and report['production_after_deposit'] >= 1


def completions(trace, kills):
    """Native ticks of each activity's certified production events (by its families), and of its credited
    kills for melee; and the tick each first reached its threshold (None if it never did)."""
    events = {goal: [] for goal in GOAL_FAMILIES}
    for event in trace['events']:
        tail = event.get('stackTail') or []
        if event.get('kind') == 'native-op' and event.get('productionSource') is True and len(tail) >= 2:
            for goal, families in GOAL_FAMILIES.items():
                if tail[-2] in families:
                    events[goal].append(event['tick'])
    events['melee'] = list((kills or {}).get('kill_ticks', []))
    reached = {goal: ticks[GOAL_TARGETS[goal] - 1] if len(ticks) >= GOAL_TARGETS[goal] else None
               for goal, ticks in events.items()}
    return events, reached


def first_completed(reached):
    """The activity that reached its threshold first ({activity: tick or None}): 'tie' when two reached it on
    the same tick, None when none did. sim/run.py's goals evaluation decides the same way."""
    first = min((t for t in reached.values() if t is not None), default=None)
    firsts = sorted(g for g, t in reached.items() if t is not None and t == first)
    return None if first is None else firsts[0] if len(firsts) == 1 else 'tie'


def repeats(decisions, done):
    """Repeated actions before the requested outcome: of the decisions ((tick, action) in order) made before
    tick `done` (all of them when it was never reached), how many repeat the previous decision's action,
    and the longest run of one action (1 for a decision that repeats nothing). sim/run.py's goals evaluation
    counts the same way."""
    actions = [a for tick, a in decisions if done is None or tick < done]
    repeated, run, longest = 0, 0, 0
    for k, action in enumerate(actions):
        same = k > 0 and action == actions[k - 1]
        repeated += same
        run = run + 1 if same else 1
        longest = max(longest, run)
    return dict(decisions_before_outcome=len(actions), repeats_before_outcome=repeated,
                longest_repeat_before_outcome=longest)


def goal_control(goal, trace, report, effects, succeeded, decisions=()):
    """Goal control of one trial: what was completed first, and what came before it.
    decisions: (native tick at the decision, action) in order, for the repeated actions before the outcome."""
    events, reached = completions(trace, report.get('kills'))
    first = first_completed(reached)
    done = reached[goal]
    before = lambda tick: done is None or tick < done
    return dict(first_completed=first, controlled_success=first == goal and succeeded,
                ticks_to_outcome=None if done is None else done - trace['startTick'],
                completion_ticks={g: None if t is None else t - trace['startTick'] for g, t in reached.items()},
                non_goal_before_outcome={g: sum(map(before, ticks)) for g, ticks in events.items() if g != goal},
                refusals_before_outcome=sum(1 for e in effects if (e.get('ended') or {}).get('refused')
                                            and (done is None or e['tick'] <= done)),
                **repeats(decisions, done))


def summarize(trials):
    """Per fixture. final_value is the reward in base value (0 after a death); value_held_at_end
    counts what a dead player held too, so it is never a score."""
    summary = {}
    for name in sorted({t['fixture'] for t in trials}):
        rows = [t for t in trials if t['fixture'] == name]
        control = [t for t in rows if 'controlled_success' in t]
        summary[name] = dict(trials=len(rows), successes=sum(t['success'] for t in rows),
                             **(dict(controlled_successes=sum(t['controlled_success'] for t in control)) if control else {}),
                             survived=sum(t['report']['survived'] for t in rows),
                             mean_final_value=float(np.mean([t['report']['reward_eligible_lower'] for t in rows])),
                             mean_value_held_at_end=float(np.mean([t['report']['retained_lower'] for t in rows])),
                             mean_items_produced=float(np.mean([t['report']['produced'] for t in rows])))
    return summary


def engine_tree(engine=None):
    """The git tree of the engine checkout scripts/stack.py boots (vendor/rs-sdk); None if it has tracked
    changes, which no declared engine build has."""
    vendor = Path(__file__).resolve().parents[1] / 'vendor' / 'rs-sdk'
    git = lambda *args: subprocess.run(['git', '-C', str(vendor), *args], capture_output=True, text=True, check=True).stdout
    return None if git('status', '--porcelain', '--untracked-files=no').strip() else git('rev-parse', 'HEAD^{tree}').strip()


def check_start_sets(fixtures, final, tree=engine_tree):
    """Refuse a goal its start set does not support, final data without --final, native check records that differ from
    their frozen draw (config.verify_check_records), and an engine build other than the one the start sets need.
    Returns that build's name (None without start sets)."""
    sets = {config.EVAL_FIXTURES[name].get('start_set') for name in fixtures} - {None}
    if any(name in config.CHECK_FIXTURES for name in fixtures):
        config.verify_check_records()
    for name in fixtures:
        fixture = config.EVAL_FIXTURES[name]
        if 'start_set' in fixture and fixture['goal'] not in config.START_SETS[fixture['start_set']]['goals']:
            raise ValueError(f"start set {fixture['start_set']} does not support goal {fixture['goal']}")
        if fixture.get('final') and not final:
            raise ValueError(f'{name} is final evaluation data: pass --final')
    engines = {config.START_SETS[name]['engine'] for name in sets}
    if len(engines) > 1:
        raise ValueError('start sets of different engine builds cannot share an evaluation')
    engine = engines.pop() if engines else None
    if engine and (found := tree(engine)) != config.ENGINE_TREES[engine]:
        raise ValueError(f'the engine checkout ({found}) is not the build these start sets need')
    return engine


def evaluate(output, *, policy, checkpoint=None,
             fixtures=tuple(name for name, f in config.EVAL_FIXTURES.items() if 'goal' not in f), trials=1,
             deterministic=True, seed=41001, device='cpu', final=False, fresh_worlds=False, first_trial=0,
             world_service=False, seed_offsets=None, demo=False, demo_wait_seconds=600.):
    """world_service: every trial in a fresh world of the start sets' build from scripts/stack.py's world service,
    stopped after it (the engine identity is the service's). A trial's seed is seed + 1000 x trial + the fixture's
    position in config.EVAL_FIXTURES, or its declared offset in seed_offsets ({fixture name: offset}; sim/gate.py's
    final2 native check, whose fixtures are added in memory). demo: one trial of one development fixture in a demo
    world (scripts/stack.py --demo-spectator), started once its spectator is in the world and labelled a demo."""
    spectator = os.environ.get('RL_DEMO_SPECTATOR')
    if bool(spectator) != demo:
        raise ValueError('a demo world serves only --demo evaluations' if spectator else
                         '--demo runs inside scripts/stack.py --demo-spectator NAME --demo-pace-ms MS')
    if demo and (final or fresh_worlds or world_service or trials != 1 or len(fixtures) != 1
                 or any(config.EVAL_FIXTURES[name].get('final') for name in fixtures)):
        raise ValueError('a demo plays one trial of one development fixture, never final data, in its own world')
    worlds_ = FreshWorlds() if world_service else None
    engines = worlds_.identity()['engines'] if worlds_ else None
    engine = check_start_sets(fixtures, final, tree=(lambda name: engines[name]['tree']) if worlds_ else engine_tree)
    if worlds_ is not None and engine is None:
        raise ValueError('the world service serves start-set fixtures, whose engine build is declared')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    evidence = JsonlEvidence(output / 'trajectory.jsonl.gz')
    results = dict(status='started', policy=policy, checkpoint=str(checkpoint) if checkpoint else None,
                   decoding='greedy' if deterministic else 'sampled', seed=seed, trials=[], final=final,
                   fresh_worlds=fresh_worlds, first_trial=first_trial, world_service=world_service,
                   engines=engines, **({'seed_offsets': seed_offsets} if seed_offsets else {}))
    if demo:
        results['demo'] = dict(label='demo: not evaluation evidence', spectator=spectator,
                               pace_ms=int(os.environ['RL_DEMO_PACE_MS']))
    executed, effects, reads = Counter(), [], []
    def write(kind, **data):
        evidence(kind, **data)
        if kind == 'task_transition' and data['executed_action'] is not None:
            executed[data['executed_action']] += 1
        elif kind == 'effect':
            effects.append(data)
        elif kind == 'resource_control' and data['operation'] == 'read':
            reads.append(data['response']['audit'])
    record = EpisodeRecorder(write)
    task, links, world = None, [], None
    try:
        model = None
        if policy == 'checkpoint':
            from sb3_contrib import RecurrentPPO
            from baseline.policy import check_names
            results['checkpoint_sha256'] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            model = RecurrentPPO.load(checkpoint, device=device)
            check_names(model.policy)
        results['manifest'] = manifest(output, checkpoint_sha256=results.get('checkpoint_sha256'))
        # The goal is an input only goal-v1 checkpoints have; zero-filling it would hide the mismatch.
        goal_task = {'goal' in config.EVAL_FIXTURES[name] for name in fixtures}
        if len(goal_task) != 1:
            raise ValueError('goal fixtures and value fixtures cannot share an evaluation')
        if model is not None and ('goal_vec' in model.observation_space.spaces) != (goal_task == {True}):
            raise ValueError('checkpoint and fixtures disagree on the goal input (goal-v1)')
        if policy in ('reference', 'edge', 'chase') and goal_task != {True}:
            raise ValueError('the reference player, the edge walk and the chase play goal fixtures only')
        actor = Actor(policy, model, deterministic=deterministic)
        if demo:
            results['demo']['waited_seconds'] = wait_for_spectator(worlds()[0]['status_url'], spectator, demo_wait_seconds)
        links = ([] if world_service else connect(record, agents=len(worlds())) if fresh_worlds
                 else connect(record, spectator=spectator if demo else None))
        if fresh_worlds and len(links) < len(fixtures) * trials:
            raise ValueError(f'{len(fixtures) * trials} trials need as many worlds: scripts/stack.py --worlds N')
        transport, service = links[0] if links else (None, None)
        previous = None
        for name in fixtures:
            for index in range(first_trial, first_trial + trials):
                if fresh_worlds:
                    # A world of its own: no earlier episode changed its locs or npcs.
                    transport, service = links[len(results['trials'])]
                    previous = None
                elif world_service:
                    world = worlds_.acquire(engine)
                    if world['tree'] != config.ENGINE_TREES[engine]:
                        raise RuntimeError(f"world {world['id']} runs another engine build than {engine}")
                    transport, service = join(world, record)
                    previous = None
                fixture = config.EVAL_FIXTURES[name]
                if 'starts' in fixture:
                    # The declared start records, in turn: the same physical starts for every goal.
                    start = fixture['starts'][index % len(fixture['starts'])]
                    fixture = dict({k: v for k, v in fixture.items() if k != 'starts'}, **start)
                offset = (seed_offsets or {}).get(name)
                trial_seed = seed + 1000 * index + (list(config.EVAL_FIXTURES).index(name) if offset is None else offset)
                record('scheduled_episode', index=len(results['trials']), fixture=dict(fixture, name=name),
                       trial=index, seed=trial_seed)
                # A fresh episode object per trial; the identity chain proves a new life.
                task = make_episode(transport, service, fixture, record)
                task.previous_identity = previous
                executed.clear(); effects.clear(); reads.clear()
                actor.fixture = fixture
                row = run_trial(task, fixture, actor, trial_seed, record, config.MAX_EVAL_DECISIONS)
                previous = task.previous_identity
                bank_actions = sum(v for k, v in executed.items() if k in config.BANK_ACTIONS)
                row.update(fixture=name, trial=index, seed=trial_seed, bank_actions=bank_actions, world=getattr(service, 'boot', None),
                           executed_action_counts={str(k): v for k, v in sorted(executed.items())},
                           success=success(fixture['kind'], row['report'], bank_actions, fixture.get('goal')),
                           **record.last)
                if 'goal' in fixture:
                    # Production of each activity, requested or not: whether the goal changes behaviour.
                    produced = {g: sum(f['produced'] for base, f in row['report']['families'].items()
                                       if int(base) in families) for g, families in GOAL_FAMILIES.items()}
                    row.update(goal=fixture['goal'], goal_return=row['reward'], produced_by_activity=produced)
                if 'start' in fixture:
                    row.update(start=fixture['start'], kills=row['report'].get('kills'),
                               **goal_control(fixture['goal'], reads[-1], row['report'], effects, row['success'],
                                              row['decision_ticks_actions']))
                if world is not None:
                    row['service_world'] = {k: world[k] for k in ('id', 'engine', 'tree', 'boot_id')}
                results['trials'].append(row)
                if fresh_worlds or world_service:
                    task.close()  # this world's agent leaves; the next trial joins another world
                if world is not None:
                    worlds_.release(world['id'])
                    world = None
                print(f"[{name} {index}] success {row['success']} survived {row['survived']}"
                      f" final value {row['final_value']} (held at end {row['value_held_at_end']})"
                      f" produced {row['items_produced']} deposits {row['deposits']} bank actions {bank_actions}"
                      f" xp {row.get('xp_gained', {})} cells {row['decision_cells']}"
                      f" clicks {row['clicks_requested']} ({row['clicks_displaced']} displaced)"
                      f" gathering XP tick {row['gathering_xp_tick']}"
                      + (f" click p {row['click_probability']:.3f}" if row['click_probability'] is not None else ''),
                      flush=True)
        if checkpoint is not None and hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != results['checkpoint_sha256']:
            raise RuntimeError('evaluation modified the checkpoint')
        results.update(summary=summarize(results['trials']),
                       delivery=delivery_report(service, min_windows=1) if service else None, status='completed')
        for name, row in results['summary'].items():
            print(f"{name}: {row['successes']}/{row['trials']} successes, {row['survived']} survived,"
                  f" mean final value {row['mean_final_value']:.1f}, mean produced {row['mean_items_produced']:.1f}",
                  flush=True)
        if results['delivery'] and not results['delivery']['ok']:
            print('[warning] world tick delivery left 95-105% during evaluation; see results.json', flush=True)
    except BaseException as error:
        results.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if task is not None: task.close()
            for link, _ in links:
                link.close()
            if world is not None:
                worlds_.release(world['id'])
        finally:
            evidence.close()
            results['seconds'] = time.monotonic() - started
            (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='RecurrentPPO .zip; omit with --policy wait/random')
    parser.add_argument('--policy', choices=POLICIES, help='default: checkpoint when --checkpoint is given')
    parser.add_argument('--fixtures', nargs='+', choices=[*config.EVAL_FIXTURES, *config.DEMO_FIXTURES],
                        default=[name for name, f in config.EVAL_FIXTURES.items() if 'goal' not in f],
                        help='default: every value-scored fixture; goal fixtures need a goal-v1 checkpoint')
    parser.add_argument('--trials', type=int, default=1, help='trials per fixture')
    parser.add_argument('--sampled', action='store_true', help='sample actions instead of taking the most likely')
    parser.add_argument('--seed', type=int, default=41001)
    parser.add_argument('--device', default='cpu')

    parser.add_argument('--final', action='store_true',
                        help='allow final evaluation data (config.START_SET_FIXTURES): the reference, WAIT and the '
                             'final evaluation only')
    parser.add_argument('--fresh-worlds', action='store_true',
                        help='one world per trial (scripts/stack.py --worlds N, N >= fixtures x trials), as final '
                             'evaluation runs: no earlier episode changed the world')
    parser.add_argument('--first-trial', type=int, default=0,
                        help='index of the first trial: its start record and seed, for a batch of a longer run')
    parser.add_argument('--world-service', action='store_true',
                        help="a fresh world per trial from scripts/stack.py's world service (--engine): start sets only")
    parser.add_argument('--demo', action='store_true',
                        help='one trial of one development fixture in a demo world (scripts/stack.py --demo-spectator), '
                             'started once the spectator is in the world; the output is a demo, never evaluation evidence')
    parser.add_argument('--demo-wait-seconds', type=float, default=600., help='how long a demo waits for its spectator')
    args = parser.parse_args()
    if args.demo:
        config.EVAL_FIXTURES.update(config.DEMO_FIXTURES)
    elif set(args.fixtures) & set(config.DEMO_FIXTURES):
        parser.error('a demo start plays only under --demo')
    policy = args.policy or ('checkpoint' if args.checkpoint else None)
    if policy is None:
        parser.error('give --checkpoint or --policy wait/random/reference')
    if (policy == 'checkpoint') != (args.checkpoint is not None):
        parser.error('--checkpoint goes with --policy checkpoint only')
    evaluate(args.output, policy=policy, checkpoint=args.checkpoint, fixtures=args.fixtures,
             trials=args.trials, deterministic=not args.sampled, seed=args.seed, device=args.device, final=args.final,
             fresh_worlds=args.fresh_worlds, first_trial=args.first_trial, world_service=args.world_service,
             demo=args.demo, demo_wait_seconds=args.demo_wait_seconds)


if __name__ == '__main__':
    main()
