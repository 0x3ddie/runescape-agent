"""Compare two evaluations of the same starts, trial by trial (baseline/evaluate.py results.json).

usage: python -m baseline.compare BASELINE/results.json CANDIDATE/results.json [--json OUT]

Trials pair by start and trial number and must share their trial seeds. For final value (the
reward in base value, 0 after a death), items produced, survival and task success it prints
each start's means and change, then the pooled paired change with a sign-flip permutation p
(two-sided; each pair's difference may change sign) and a bootstrap 95% interval (pairs
resampled within each start). Losses at single starts are shown, not only the pooled mean.

One evaluation of one checkpoint describes that checkpoint; whether a training recipe is
reliable needs several independently trained seeds.

usage: python -m baseline.compare goals --sim 101=RESULTS ... --native final-101=DIR ... --expected N --bar K --output OUT

The four-goal evaluations' rule, for sim/run.py goals and baseline/evaluate.py's goal
fixtures: per condition (policy, site, goal) the evidence, controlled and terminal successes with Wilson
intervals, failure types and what came before the requested outcome; per trained condition its gate, with
the paired bootstrap against its initialization and random-legal (DRAWS resamples of ANALYSIS_SEED, pairs
resampled within each start profile; the improvement needs the 2.5th percentile above zero).
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

MEASURES = {
    'final_value': lambda t: t['report']['reward_eligible_lower'],
    'items_produced': lambda t: t['report']['produced'],
    'survived': lambda t: float(t['report']['survived']),
    'success': lambda t: float(t['success']),
}


def pairs(baseline, candidate):
    """[(start, baseline trial, candidate trial)] in baseline order; refuses unmatched runs."""
    index = {(t['fixture'], t['trial']): t for t in candidate}
    if len(index) != len(candidate) or len(baseline) != len(candidate) or set(index) != {(t['fixture'], t['trial']) for t in baseline}:
        raise ValueError('the two evaluations do not cover the same starts and trials')
    matched = [(t['fixture'], t, index[t['fixture'], t['trial']]) for t in baseline]
    if any(a['seed'] != b['seed'] for _, a, b in matched):
        raise ValueError('paired trials used different trial seeds')
    return matched


def paired_test(diffs, groups, rng, draws=20000):
    """Mean paired difference, two-sided sign-flip p, and a bootstrap 95% interval that
    resamples pairs within each group."""
    diffs, groups = np.asarray(diffs, float), np.asarray(groups)
    mean = float(diffs.mean())
    flips = rng.choice((-1., 1.), size=(draws, len(diffs)))
    p = float((np.abs((flips * diffs).mean(axis=1)) >= abs(mean) - 1e-12).mean())
    members = [np.flatnonzero(groups == g) for g in dict.fromkeys(groups)]
    boot = np.array([np.concatenate([diffs[rng.choice(m, len(m))] for m in members]).mean() for _ in range(2000)])
    return dict(mean=mean, p=p, low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)))


def compare(baseline, candidate, seed=0):
    rng = np.random.default_rng(seed)
    matched = pairs(baseline, candidate)
    starts = list(dict.fromkeys(start for start, _, _ in matched))
    result = dict(trials=len(matched), starts={}, pooled={})
    for name, measure in MEASURES.items():
        a = np.array([measure(x) for _, x, _ in matched])
        b = np.array([measure(y) for _, _, y in matched])
        groups = [start for start, _, _ in matched]
        for start in starts:
            keep = np.array([g == start for g in groups])
            result['starts'].setdefault(start, {})[name] = dict(baseline=float(a[keep].mean()), candidate=float(b[keep].mean()),
                                                                change=float((b - a)[keep].mean()))
        result['pooled'][name] = dict(baseline=float(a.mean()), candidate=float(b.mean()), **paired_test(b - a, groups, rng))
    return result



# ---- the four-goal evaluations' comparison rule ----------------------------------------
DRAWS, ANALYSIS_SEED = 10_000, 17017


def wilson(successes, trials, z=1.96):
    """A 95% Wilson interval for a proportion."""
    if not trials:
        return None
    p, n = successes / trials, trials
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def paired_bootstrap(diffs, groups=None, draws=DRAWS, seed=ANALYSIS_SEED):
    """The mean paired difference and its 95% bootstrap interval. Each resample draws, with replacement, as many
    pairs from each group as the group holds (without groups, from all pairs), so every group keeps its count;
    the interval is the 2.5th and 97.5th percentiles of the resampled means (numpy's linear rule). A fresh
    generator of `seed` for each call, so each comparison reproduces alone."""
    diffs = np.asarray(diffs, float)
    groups = np.zeros(len(diffs), int) if groups is None else np.asarray(groups)
    rng, total = np.random.default_rng(seed), np.zeros(draws)
    for group in dict.fromkeys(groups.tolist()):
        member = diffs[groups == group]
        total += member[rng.integers(0, len(member), size=(draws, len(member)))].sum(axis=1)
    low, high = np.quantile(total / len(diffs), [.025, .975], method='linear')
    return dict(mean=float(diffs.mean()), low=float(low), high=float(high), pairs=len(diffs), draws=draws, seed=seed)


def sim_rows(results, label):
    """The trials of a sim/run.py goals results.json as goal_report rows: policies `final` and `initial` become
    final-<label> and initial-<label>; a start record is a pair key (one group)."""
    for t in results['trials']:
        policy = t['policy'] if t['policy'] in ('random', 'wait', 'reference') else f"{t['policy']}-{label}"
        yield dict(t, policy=policy, site='simulator', pair=t['start'], group=None, pairing=json.dumps(t['record'], sort_keys=True))


def native_rows(results, policy):
    """The goal trials of a baseline/evaluate.py results.json as goal_report rows: `<set>_<goal>` fixtures, the
    trial index is the pair key and its declared start profile the group."""
    for t in results['trials']:
        site, goal = t['fixture'].rsplit('_', 1)
        yield dict(t, policy=policy, site=site, goal=goal, pair=t['trial'], group=t['start'], pairing=t['start'],
                   survived=t['report']['survived'])


def condition(rows, expected):
    """One condition's trials (a policy, site and goal): what evidence there is, controlled and terminal
    successes with intervals, how the others failed, and what came before the requested outcome."""
    scored = [r for r in rows if 'success' in r]
    goal = rows[0]['goal'] if rows else None
    first = Counter('requested' if r['first_completed'] == goal else 'none' if r['first_completed'] is None
                    else 'tie' if r['first_completed'] == 'tie' else 'other activity' for r in scored)
    controlled = sum(r['controlled_success'] for r in scored)
    ticks = sorted(r['ticks_to_outcome'] for r in scored if r['ticks_to_outcome'] is not None)
    quartile = lambda q: float(np.quantile(ticks, q, method='linear')) if ticks else None
    mean = lambda f: float(np.mean([f(r) for r in scored])) if scored else None
    by_group = {}
    for r in scored:
        if r['group'] is not None:
            cell = by_group.setdefault(r['group'], dict(trials=0, controlled_successes=0))
            cell['trials'] += 1
            cell['controlled_successes'] += r['controlled_success']
    return dict(expected=expected, present=len(rows), scored=len(scored), missing=expected - len(rows),
                unsupported=sum('unsupported' in r for r in rows), decision_limit=sum('decision_limit' in r for r in rows),
                valid=len(scored) == expected == len(rows),
                controlled_successes=controlled, controlled_interval=wilson(controlled, len(scored)),
                successes=sum(r['success'] for r in scored), success_interval=wilson(sum(r['success'] for r in scored), len(scored)),
                first_completed=dict(first), deaths=sum(not r['survived'] for r in scored),
                outcome_reached=len(ticks), ticks_to_outcome=dict(p25=quartile(.25), median=quartile(.5), p75=quartile(.75)),
                off_goal_before_outcome=dict(mean=mean(lambda r: sum(r['non_goal_before_outcome'].values())),
                                             trials=sum(any(r['non_goal_before_outcome'].values()) for r in scored)),
                repeats_before_outcome=dict(mean=mean(lambda r: r['repeats_before_outcome']),
                                            mean_longest=mean(lambda r: r['longest_repeat_before_outcome']),
                                            mean_decisions=mean(lambda r: r['decisions_before_outcome'])),
                refusals_before_outcome=dict(mean=mean(lambda r: r['refusals_before_outcome']),
                                             trials=sum(r['refusals_before_outcome'] > 0 for r in scored)),
                **({'by_profile': dict(sorted(by_group.items()))} if by_group else {}))


def goal_report(rows, expected, bar, controls=('initial', 'random')):
    """Every condition (policy, site, goal) of the rows, and for each trained policy final-<seed> its gate: valid
    evidence for every expected trial, at least `bar` controlled successes, and an improvement over each
    control (initial-<seed>, random) whose paired_bootstrap lower bound is above zero. Pairs share the pair key
    and its start (the start record, or the declared profile); a pair with a missing or unscored side, or with
    different starts, fails the gate."""
    cells = {}
    for r in rows:
        cells.setdefault((r['policy'], r['site'], r['goal']), []).append(r)
    conditions = {'|'.join(k): condition(v, expected) for k, v in sorted(cells.items())}
    gates = {}
    for (policy, site, goal), mine in sorted(cells.items()):
        if not policy.startswith('final-'):
            continue
        seed = policy.removeprefix('final-')
        cell = conditions['|'.join((policy, site, goal))]
        gate = dict(valid=cell['valid'], controlled_successes=cell['controlled_successes'], bar=bar,
                    meets_bar=cell['controlled_successes'] >= bar, comparisons={})
        for control in controls:
            name = f'{control}-{seed}' if control == 'initial' else control
            theirs = {r['pair']: r for r in cells.get((name, site, goal), [])}
            ours = {r['pair']: r for r in mine}
            matched = sorted(set(ours) & set(theirs))
            usable = [k for k in matched if 'success' in ours[k] and 'success' in theirs[k]]
            same_start = all(ours[k]['pairing'] == theirs[k]['pairing'] for k in matched)
            if not usable:
                gate['comparisons'][name] = dict(pairs=0, improved=False)
                continue
            diffs = [float(ours[k]['controlled_success']) - float(theirs[k]['controlled_success']) for k in usable]
            result = paired_bootstrap(diffs, None if ours[usable[0]]['group'] is None else [ours[k]['group'] for k in usable])
            complete = len(usable) == len(ours) == len(theirs) == expected and same_start
            gate['comparisons'][name] = dict(result, complete=complete, same_starts=same_start,
                                             final_only=sum(d > 0 for d in diffs), control_only=sum(d < 0 for d in diffs),
                                             improved=complete and result['low'] > 0)
        gate['passed'] = gate['valid'] and gate['meets_bar'] and all(
            gate['comparisons'].get(f'{c}-{seed}' if c == 'initial' else c, {}).get('improved') for c in controls)
        gates['|'.join((policy, site, goal))] = gate
    return dict(rule=dict(draws=DRAWS, seed=ANALYSIS_SEED, quantiles='2.5th and 97.5th, linear', bar=bar, expected=expected),
                conditions=conditions, gates=gates, passed=bool(gates) and all(g['passed'] for g in gates.values()),
                failed=[k for k, g in gates.items() if not g['passed']])


def goals_main(argv):
    parser = argparse.ArgumentParser(prog='python -m baseline.compare goals', description=goal_report.__doc__)
    parser.add_argument('--sim', nargs='+', default=[], metavar='LABEL=RESULTS',
                        help='sim/run.py goals results.json files; LABEL is the seed of their final and initial policies')
    parser.add_argument('--native', nargs='+', default=[], metavar='POLICY=DIRECTORY',
                        help='every baseline/evaluate.py results.json under DIRECTORY, of policy POLICY (final-101, random...)')
    parser.add_argument('--expected', type=int, required=True, help='trials each condition must have')
    parser.add_argument('--bar', type=int, required=True, help='controlled successes each trained condition needs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    rows, sources = [], dict(sim={}, native={})
    for item in args.sim:
        label, path = item.split('=', 1)
        results = json.loads(Path(path).read_text())
        rows += list(sim_rows(results, label))
        sources['sim'][label] = dict(path=path, checkpoints=results.get('checkpoints'), layout_seeds=results.get('layout_seeds'),
                                     one_per_layout=results.get('one_per_layout'), key=results.get('key'))
    for item in args.native:
        policy, directory = item.split('=', 1)
        found = dict(completed=0, other=[], checkpoint_sha256=set())
        for path in sorted(Path(directory).rglob('results.json')):
            results = json.loads(path.read_text())
            if results.get('status') != 'completed':  # its trials count as missing
                found['other'].append(dict(path=str(path), status=results.get('status'), error=results.get('error')))
                continue
            found['completed'] += 1
            found['checkpoint_sha256'].add(results.get('checkpoint_sha256'))
            rows += list(native_rows(results, policy))
        sources['native'][policy] = dict(found, directory=directory, checkpoint_sha256=sorted(map(str, found['checkpoint_sha256'])))
    report = dict(goal_report(rows, args.expected, args.bar), sources=sources)
    args.output.write_text(json.dumps(report, indent=1, default=str) + '\n', encoding='utf-8')
    for name, gate in report['gates'].items():
        comparisons = '  '.join(f"{c} {g.get('mean', 0):+.3f} [{g.get('low', 0):+.3f}, {g.get('high', 0):+.3f}]"
                                for c, g in gate['comparisons'].items())
        print(f"{name:32s} {gate['controlled_successes']:4d}/{args.expected} valid {gate['valid']}  {comparisons}  "
              f"{'passed' if gate['passed'] else 'NOT passed'}")
    print('all passed' if report['passed'] else f"not passed: {report['failed']}")


def main():
    import sys
    if sys.argv[1:2] == ['goals']:
        return goals_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--json', type=Path, help='also write the comparison here')
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in (args.baseline, args.candidate)]
    if any(run.get('status') != 'completed' for run in runs):
        raise SystemExit('both evaluations must have completed')
    result = compare(runs[0]['trials'], runs[1]['trials'])
    result.update(baseline=str(args.baseline), candidate=str(args.candidate),
                  checkpoints=[run.get('checkpoint_sha256') for run in runs])
    print(f"{result['trials']} paired trials; per start: baseline -> candidate (change)")
    for start, row in result['starts'].items():
        print(f"  {start:16s} " + '  '.join(f"{name} {m['baseline']:.1f} -> {m['candidate']:.1f} ({m['change']:+.1f})"
                                            for name, m in row.items()))
    print('pooled:')
    for name, m in result['pooled'].items():
        print(f"  {name:15s} {m['baseline']:8.2f} -> {m['candidate']:8.2f}  change {m['mean']:+.2f}"
              f"  95% [{m['low']:+.2f}, {m['high']:+.2f}]  sign-flip p {m['p']:.4f}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
