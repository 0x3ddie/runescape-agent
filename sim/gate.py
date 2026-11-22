"""The skill gate command: one checkpoint, or the WAIT control, through phase 1's simulator gates on
declared records in two sampled passes and greedy play, and through the native check, written as one gate.json with
every rule's result.

    .venv-sim/bin/python -m sim.gate run --checkpoint RUN/latest.zip --output OUT [--skills wood fish mine melee]
        [--records development | --records final --final | --records final2 --final --skills wood] [--workers 4]
        [--native --bun BUN] [--schedule overlap]
    .venv-sim/bin/python -m sim.gate run --policy wait --output OUT ...
    .venv-sim/bin/python -m sim.gate native OUT --bun BUN   # the native check for an existing output, gate.json again
    .venv-sim/bin/python -m sim.gate rules OUT          # gate.json again from an existing output
    .venv-sim/bin/python -m sim.gate final4 --checkpoint RUN/latest.zip --policy-seed 101 --records final4 --final --output OUT
    .venv-sim/bin/python -m sim.gate final4 ... --records development --layouts 4   # a rehearsal, never final data

The simulator evaluations are the frozen commands' (sim.search evaluate-supply and evaluate, sim.run goals, sim.combat
evaluate), each pass on the same records, in parts across `--workers` processes, then merged. The native check plays
baseline.evaluate in fresh worlds of scripts/stack.py's world service, one process per engine build, in GATE['native']'s
passes: with the final records config.CHECK_FIXTURES (the frozen check records, final data), with the development records
config.REHEARSAL_FIXTURES (the same sites and skills on the paused study's training draws), so a rehearsal never plays
final data. Melee has none. Each setting frozen for phase 1 is one
entry of GATE: the passes and their seed bases, which pass decodes how and which gate, the record sets, each rule's
threshold and count, and the native passes and thresholds.

gate-v2 is gate-v1 plus the final2 record set: new
final records of wood's simulator sets and a new native draw (FINAL2_DRAWS) at wood's two sites, for wood only.
baseline/config.py does not define the final2 native fixtures, so a baseline.evaluate process cannot play them: the
native pass appends them to config.EVAL_FIXTURES in its own process, after every earlier fixture, and plays them there
through baseline.evaluate with GATE's declared trial-seed offsets.

The schedule changes only when each process runs, never what it runs: `--schedule serial` (the default, phase 1's)
runs the simulator parts, then the merges, then the native passes; `--schedule overlap` runs the native passes, still
one scripts/stack.py world service at a time and in GATE's order, beside the simulator parts and merges. The simulator
records and rules are the same either way, and the native passes get the same commands; but two runs of one native
pass already differ under the serial schedule (the phase 1 rehearsal's repeated sampled1 and greedy passes).

Every gate.json stores the GATE it was scored with under 'gate'. `native` and `rules` refuse an output whose stored
gate differs from this GATE (after a JSON round trip) and name the keys that differ: score an old output again only
with the code of its own commit.

A rule reads one cell of one pass. A cell with an unsupported or unfinished trial, or with another record count than
the rule declares, makes that rule invalid, never a pass or a failure. A skill's simulator verdict is pass when every
gating rule passes in every gating pass, fail when a valid gating rule fails, and incomplete otherwise; its native
verdict likewise. Reported rules (greedy play, hidden-tree search, the short wood and melee tasks) never gate. Beside
the rules, every cell reports successes, deaths, decisions per 1,000 ticks, refusals, stalls and interface loops (each
None where its evaluator records no such evidence).

`free` reads the rows of sim.hour free (free-play-v1's evaluation) and writes free.json:
for each policy seed and each sampled pass, never pooled, every actor's mean integer level gain and deaths a life, the
paired differences policy - control with a 95% percentile bootstrap paired by record (baseline/compare.py
paired_bootstrap, FREE's 10,000 resamples and seed 17017), Claims L, B and R and No rescue with every number each used;
greedy play and the 6,000-tick read are descriptions with no claim; beside them the unpaid reads. A verdict (pass, fail,
match) is written only for outputs that match one of FREE's frozen plans exactly (its settings, tables, seeds and the
final checkpoint of a run with its frozen facts), of one commit with no change; any other outputs give the same numbers
as a description, and an XP rate other than 1 the differences only. The pilot's plan is frozen (12
November 2026); the final read's is not, until its decision budget and start digests. It plays nothing, so it runs
again on the same rows:

    .venv-sim/bin/python -m sim.gate free OUT_S101 OUT_S202 ... OUT_CONTROLS --output free.json

`final4` (FINAL4, beside GATE) is the native gap study's simulator read of one Stage 4 final: sim.run goals on the
frozen final4 records, mine only, in FINAL4's passes, written as final4.json with gate-v2's mine rule and verdict.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ('wood', 'fish', 'mine', 'melee')
# Any later change to GATE takes a new version name. 'gate-v1' kept its name when the second gating native pass was
# added (27 October), so phase 0's gate.json files carry 'gate-v1' with the native passes sampled1 and greedy, and
# phase 1's with sampled1, sampled2 and greedy; only the stored gate tells them apart (stale_keys). 'gate-v2' adds the
# final2 record set and its native check, so a gate-v1 output is scored again only with the code of its own commit.
GATE = dict(
    version='gate-v2',
    # The decoding rule (26 October): sampled play governs every gate (both passes must pass); greedy is reported.
    passes=(dict(name='sampled1', sampled=True, gating=True, seeds=dict(combat=303001, search=404001, skills=202001, supply=505001)),
            dict(name='sampled2', sampled=True, gating=True, seeds=dict(combat=909001, search=909001, skills=909001, supply=909001)),
            dict(name='greedy', sampled=False, gating=False, seeds=dict(combat=303001, search=404001, skills=202001, supply=505001))),
    # Record sets: each world's split, first layout index, start key and count. development: the development split from
    # index 0 with each evaluator's default key (rehearsals); final: frozen 27 October, seed
    # ranges no earlier final evaluation used in any world: combat 8800-9599 with the short tasks on its first 200
    # layouts, as earlier studies shared them, supply from 9600, search from 220; new keys. final2: the native-click
    # study's records, for wood only, each range and key used by no
    # earlier evaluation: supply 6200-6399, search from 3400 (it scans 3400-3792), the short task on compact 4600-4799.
    records=dict(development=dict(split='development', combat=dict(first=0, key=11, per_group=200),
                                  skills=dict(first=0, key=7, layouts=200), supply=dict(first=0, key=13, layouts=200),
                                  search=dict(first=0, key=13, layouts=200)),
                 final=dict(split='final', combat=dict(first=8800, key=131, per_group=200),
                            skills=dict(first=8800, key=137, layouts=200), supply=dict(first=9600, key=139, layouts=200),
                            search=dict(first=220, key=149, layouts=200)),
                 final2=dict(split='final', serves=('wood',), skills=dict(first=4600, key=167, layouts=200),
                             supply=dict(first=6200, key=157, layouts=200), search=dict(first=3400, key=163, layouts=200))),
    parts=dict(combat=8, supply=4, search=2, skills=1),
    # Simulator gates: wood on supply-wood-v1, per variant, the successes and the mean
    # certified logs (logs_mean 'recorded': each scored trial's retained logs, a death's included, as earlier supply
    # summaries; 'death_zero': a death counts 0); fish and mine the short task's controlled successes; melee
    # sustained-combat-v1 overall and in each start group. (count, out of).
    rules=dict(wood=dict(successes=(160, 200), logs_mean=8.0, logs='recorded'),
               fish=dict(count='controlled_successes', at_least=(160, 200)),
               mine=dict(count='controlled_successes', at_least=(160, 200)),
               melee=dict(all=(640, 800), group=(140, 200))),
    # The native check: at each skill's sites (config.CHECK_SITES), config.CHECK_RECORDS trials
    # each (2 x 16 = 32), wood counts survival with at least `wood_logs` certified logs, fish and mine a controlled
    # success; at least `at_least` of them, in both gating sampled passes (the second added on 27 October).
    # Worlds run at the training XP rate. final2: FINAL2_DRAWS, drawn from `seed`, at each wood site; a trial's seed is
    # the pass's seed + 1000 x record + the site's offset, declared because in-memory fixtures of M2b's development runs
    # (logs/sim-content/m2b/evaluate_dev.py) took config.EVAL_FIXTURES positions 87-90 under 41001; `digests` are each
    # site's records in config.check_records' form.
    native=dict(passes=(dict(name='sampled1', sampled=True, gating=True, seed=41001),
                        dict(name='sampled2', sampled=True, gating=True, seed=909001),
                        dict(name='greedy', sampled=False, gating=False, seed=41001)),
                at_least=26, wood_logs=8, count='controlled_success', xp_rate=25, timeout_seconds=5400,
                final2=dict(seed=20261102, records=16, offsets=dict(entrana=91, rimmington=92),
                            digests=dict(entrana='db7ea1258ba7bf2942f2ef8d28a0ab6afd4d219d2e500259087b0585cebb05ba',
                                         rimmington='3a918d1df12e7db25c3de0b87a3cc7f930582dd21c3c26e88a1060178342a036'))),
)
# The final2 native check's start records: GATE['native']['final2'] draws, drawn
# as the check records were (config.study_draws, START_KIT plus the weapon carried) from its seed, none equal to an
# earlier native draw (config.earlier_native_draws, plus the check draws and native-hour-v1's nine, which it does not
# hold), frozen here as (setStats, weapon, reset seed) and by GATE's digests. One draw serves both sites. Final data.
FINAL2_DRAWS = [
    ('woodcutting:16,fishing:15,mining:5,attack:5,strength:8,defence:9,hitpoints:10', 'bronze_sword', 3249),
    ('woodcutting:10,fishing:14,mining:5,attack:9,strength:6,defence:3,hitpoints:12', 'bronze_sword', 483562),
    ('woodcutting:5,fishing:14,mining:5,attack:7,strength:4,defence:6,hitpoints:12', 'bronze_dagger', 81740),
    ('woodcutting:5,fishing:16,mining:10,attack:4,strength:10,defence:9,hitpoints:10', 'bronze_sword', 538988),
    ('woodcutting:14,fishing:16,mining:1,attack:9,strength:7,defence:7,hitpoints:11', 'bronze_sword', 34369),
    ('woodcutting:15,fishing:16,mining:16,attack:2,strength:10,defence:9,hitpoints:11', 'bronze_dagger', 483627),
    ('woodcutting:15,fishing:15,mining:10,attack:9,strength:10,defence:4,hitpoints:11', 'bronze_sword', 843265),
    ('woodcutting:20,fishing:15,mining:14,attack:2,strength:10,defence:10,hitpoints:11', 'bronze_dagger', 182677),
    ('woodcutting:15,fishing:5,mining:5,attack:2,strength:1,defence:2,hitpoints:12', 'bronze_sword', 572885),
    ('woodcutting:5,fishing:10,mining:1,attack:7,strength:9,defence:2,hitpoints:11', 'bronze_sword', 722422),
    ('woodcutting:1,fishing:20,mining:10,attack:5,strength:10,defence:8,hitpoints:10', 'bronze_dagger', 102836),
    ('woodcutting:15,fishing:16,mining:1,attack:10,strength:3,defence:3,hitpoints:11', 'bronze_dagger', 205851),
    ('woodcutting:20,fishing:5,mining:16,attack:10,strength:8,defence:8,hitpoints:12', 'bronze_sword', 691059),
    ('woodcutting:1,fishing:1,mining:1,attack:7,strength:3,defence:5,hitpoints:12', 'bronze_dagger', 463162),
    ('woodcutting:20,fishing:1,mining:1,attack:1,strength:3,defence:6,hitpoints:12', 'bronze_dagger', 410386),
    ('woodcutting:14,fishing:20,mining:1,attack:1,strength:5,defence:7,hitpoints:12', 'bronze_dagger', 890462),
]
TASKS = dict(wood=('supply', 'search', 'skills'), fish=('skills',), mine=('skills',), melee=('combat', 'skills'))
NATIVE_FAMILY = dict(development='rehearse', final='check', final2='gate2')  # each record set's native fixture prefix
# final4 (built in Stage 3, 8 November 2026): the records on
# which each Stage 4 final must hold 160 of 200 in its own world in both sampled passes, or its seed is not valid. Under
# arm Layouts both arms train in compact-world-v1, so K's and X's finals read the same records, paired by record. Its
# own version beside GATE, which stays gate-v2, so no gate-v2 output changes. The records are final split seed indices
# 5,000-5,199, one start per layout, key 197, mine only, frozen by `sha256` (records_sha256). A trial's seed is its
# pass's seed + 1,000 x the policy seed (the final's training seed) + the record (sim.run trial_seed). The rule counts
# controlled successes as gate-v2's mine rule does; a cell with an unsupported or unfinished trial, or with another
# record count, makes its rule invalid, so the read is incomplete, never a pass or a failure. `development` is
# the same draw on the development split, for rehearsals only (its rules are invalid by count below 200 layouts).
FINAL4 = dict(
    version='final4',
    goal='mine',
    records=dict(split='final', first=5000, layouts=200, key=197, world='compact-world-v1',
                 sha256='8d65f581d365cbd091723994f8c6c1b1bfa7509484e56eac0a11246fc8c57132'),
    development=dict(split='development', first=0, layouts=200, key=197, world='compact-world-v1'),
    passes=(dict(name='sampled1', sampled=True, gating=True, seed=57_100_000),
            dict(name='sampled2', sampled=True, gating=True, seed=67_100_000),
            dict(name='greedy', sampled=False, gating=False, seed=57_100_000)),
    policy_seeds=(101, 202, 303),
    rule=dict(count='controlled_successes', at_least=(160, 200)),
)


def environment():
    return dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / 'rl')]))


def jobs(out, skills, policy, checkpoint, records):
    """(name, argv, log, weight) of every simulator evaluation part, longest first, and the merges (argv)."""
    rec, py = GATE['records'][records], sys.executable
    final = ['--final'] if rec['split'] == 'final' else []
    who = ['--checkpoint', str(checkpoint), '--policies', 'final'] if policy == 'final' else ['--policies', 'wait']
    tasks = sorted({t for s in skills for t in TASKS[s]})
    found, merges = [], []
    for p in GATE['passes']:
        mode = lambda task: (['--sampled'] if p['sampled'] else []) + ['--seed', str(p['seeds'][task])]
        for task in tasks:
            r, where, n = rec[task], out / p['name'] / task, GATE['parts'][task]
            if task == 'skills':
                found.append((f'{p["name"]}-skills', [py, '-m', 'sim.run', 'goals', *who, '--split', rec['split'], *final,
                                                       '--layouts', str(r['layouts']), '--starts', str(r['layouts']),
                                                       '--one-per-layout', '--first', str(r['first']), '--key', str(r['key']),
                                                       *mode(task), '--output', str(where)], 4))
                continue
            for i in range(n):
                if task == 'combat':
                    argv = [py, '-m', 'sim.combat', 'evaluate', *who, '--split', rec['split'], *final, '--per-group',
                            str(r['per_group']), '--key', str(r['key']), '--first', str(r['first'])]
                else:
                    argv = [py, '-m', 'sim.search', 'evaluate-supply' if task == 'supply' else 'evaluate', *who,
                            '--split', rec['split'], *final, '--layouts', str(r['layouts']), '--first', str(r['first']),
                            '--key', str(r['key'])]
                found.append((f'{p["name"]}-{task}-{i}', [*argv, '--part', str(i), str(n), *mode(task), '--output', str(where)],
                              dict(combat=3, supply=2, search=1)[task]))
            merges.append([py, '-m', 'sim.combat', 'merge', str(where)] if task == 'combat' else
                          [py, '-m', 'sim.search', 'merge-supply' if task == 'supply' else 'merge', '--output', str(where)])
    return sorted(found, key=lambda j: -j[2]), merges


def execute(commands, workers, logs, timeout):
    """Run (name, argv) commands, at most `workers` at once, each with its log and timeout: {name: {exit, seconds}}."""
    logs.mkdir(parents=True, exist_ok=True)

    def one(item):
        name, argv = item
        t0 = time.perf_counter()
        with open(logs / f'{name}.log', 'w') as log:
            try:
                code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=environment(), cwd=ROOT,
                                      timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        return name, dict(exit=code, seconds=round(time.perf_counter() - t0, 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, commands))


def timed(run):
    """run()'s result and its seconds."""
    t0 = time.perf_counter()
    done = run()
    return done, round(time.perf_counter() - t0, 1)


def final2_records(site):
    """The final2 native check's start records at wood site `site`, in their frozen order (config.check_records' form)."""
    from baseline import config
    x, z = config.START_SETS[site]['start']
    return [dict(start=f'{site}-gate2-{k}', spawnTele=f'0,{x >> 6},{z >> 6},{x & 63},{z & 63}', set_stats=stats,
                 seed_items=f'{config.START_KIT_ITEMS},{weapon}:1', reset_seed=reset_seed)
            for k, (stats, weapon, reset_seed) in enumerate(FINAL2_DRAWS)]


def verify_final2_records():
    """The final2 native records as frozen: equal to their draw, disjoint from every earlier native draw and hashing to
    GATE's digests; raises ValueError otherwise. Returns the digests."""
    import hashlib
    from baseline import config
    from sim.hour import native_records
    F = GATE['native']['final2']
    fields = lambda records: [{k: r[k] for k in ('set_stats', 'seed_items', 'reset_seed')} for r in records]
    earlier = config.earlier_native_draws() + fields(config.check_records('entrana')) + fields(native_records())
    frozen = fields(final2_records('entrana'))
    digests = {site: hashlib.sha256(json.dumps(final2_records(site), sort_keys=True).encode()).hexdigest()
               for site in F['offsets']}
    if frozen != config.study_draws(F['seed'], F['records'], earlier) or any(r in earlier for r in frozen) \
            or digests != F['digests']:
        raise ValueError('the final2 native records differ from their draw or their frozen digests')
    return digests


def native_set(records):
    """{name: fixture} of a record set's native check: the check records (final), the rehearsal records (development)
    or the final2 fixtures, wood's goal fixture at each of its sites as config.CHECK_FIXTURES are (final2)."""
    from baseline import config
    if records != 'final2':
        return config.CHECK_FIXTURES if records == 'final' else config.REHEARSAL_FIXTURES
    return {f'gate2_{site}_wood': dict(id=f'gate2-{site}-wood', kind='goal', horizon=config.CHECK_HORIZONS['wood'],
                                       capacity='declared', goal='wood', start_set=site, final=True, starts=final2_records(site))
            for site in GATE['native']['final2']['offsets']}


def native_fixtures(skills, records):
    """{engine build: [fixtures]} of the skills that have a native check in the record set (native_set)."""
    from baseline import config
    out = {}
    for name, fixture in native_set(records).items():
        if fixture['goal'] in skills:
            out.setdefault(config.START_SETS[fixture['start_set']]['engine'], []).append(name)
    return out


def final2_trials(out, policy, checkpoint, sampled, seed, skills):
    """Inside a scripts/stack.py world service: the final2 native check in this process, its fixtures appended to
    config.EVAL_FIXTURES after every earlier one and played by baseline.evaluate.evaluate with GATE's trial-seed
    offsets (16 trials each, the records in order). Returns its exit code by engine build."""
    import traceback
    from baseline import config
    from baseline.evaluate import evaluate
    verify_final2_records()
    fixtures, F = native_set('final2'), GATE['native']['final2']
    config.EVAL_FIXTURES.update(fixtures)
    done = {}
    for engine, names in native_fixtures(skills, 'final2').items():
        t0 = time.perf_counter()
        try:
            evaluate(out / engine, policy='checkpoint' if policy == 'final' else 'wait', checkpoint=checkpoint,
                     fixtures=names, trials=F['records'], deterministic=not sampled, seed=seed, final=True,
                     world_service=True, seed_offsets={name: F['offsets'][fixtures[name]['start_set']] for name in names})
            code = 0
        except Exception:
            traceback.print_exc()
            code = 1
        done[engine] = dict(exit=code, seconds=round(time.perf_counter() - t0, 1))
    return done


def native_trials(out, policy, checkpoint, sampled, seed, skills, records):
    """Inside a scripts/stack.py world service: one baseline.evaluate process per engine build, both at once, on the
    native fixtures of `skills` (16 trials each, the records in order); final2's in this process (final2_trials).
    Returns their exit codes."""
    from baseline import config
    if records == 'final2':
        return final2_trials(out, policy, checkpoint, sampled, seed, skills)
    who = ['--checkpoint', str(checkpoint)] if policy == 'final' else ['--policy', 'wait']
    final = ['--final'] if GATE['records'][records]['split'] == 'final' else []
    commands = [(engine, [sys.executable, '-m', 'baseline.evaluate', '--output', str(out / engine), '--world-service',
                          *final, '--fixtures', *names, '--trials', str(config.CHECK_RECORDS), *who,
                          *(['--sampled'] if sampled else []), '--seed', str(seed)])
                for engine, names in native_fixtures(skills, records).items()]
    return execute(commands, len(commands), out / 'logs', GATE['native']['timeout_seconds'])


def native(out, policy, checkpoint, skills, records, bun, engines, python):
    """Each native pass in its own scripts/stack.py world service (fresh worlds, lockstep, 25x XP), stopped after it.
    Skills without a native check run nothing."""
    from baseline import config
    fixtures = native_fixtures(skills, records)
    if not fixtures:
        return {}
    if (out / 'native').exists():
        raise SystemExit(f'{out / "native"} holds an earlier native attempt: move it aside first')
    config.verify_check_records()
    if records == 'final2':
        verify_final2_records()
    done = {}
    for p in GATE['native']['passes']:
        where = out / 'native' / p['name']
        child = [python, '-m', 'sim.gate', 'native-trials', '--output', str(where), '--policy', policy, '--seed', str(p['seed']),
                 '--skills', *skills, '--records', records, *(['--checkpoint', str(checkpoint)] if policy == 'final' else []),
                 *(['--sampled'] if p['sampled'] else [])]
        argv = [python, str(ROOT / 'scripts' / 'stack.py'), '--bun', str(bun), '--output', str(out / 'native' / f'stack-{p["name"]}'),
                '--timeout-seconds', str(GATE['native']['timeout_seconds']), '--lockstep', '--xp-rate', str(GATE['native']['xp_rate']),
                '--worlds', '0',
                *[a for name, path in engines.items() for a in ('--engine', f'{name}={path}')],
                '--blocks', str(len(fixtures)), '--', *child]
        done.update(execute([(f'native-{p["name"]}', argv)], 1, out / 'logs', GATE['native']['timeout_seconds'] + 600))
    return done


# ---- rules ------------------------------------------------------------------------------------------------------
def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def stream(path):
    """The rows of a .jsonl.gz evidence file (absent: none)."""
    if not Path(path).exists():
        return []
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def report(trials, refusals=None, interface=None, stalls=None):
    """One cell's per-policy fields."""
    scored = [t for t in trials if 'success' in t]
    ticks = sum(t['ticks'] for t in scored)
    return dict(trials=len(trials), scored=len(scored), successes=sum(t['success'] for t in scored),
                deaths=sum(not t['survived'] for t in scored),
                decisions_per_1000_ticks=round(1000 * sum(t['decisions'] for t in scored) / ticks, 2) if ticks else None,
                refusals=refusals, stalls=stalls, interface_loops=interface)


def skills_cells(where, merged, policy, goals=SKILLS):
    """{goal: report} of `policy` in one sim.run goals output (folder `where`, its results.json read as `merged`)."""
    from sim.search import interface_loops
    trials = [t for t in merged['trials'] if t['policy'] == policy]
    rows = [r for r in stream(where / 'episodes.jsonl.gz') if r.get('policy') == policy and 'decisions' in r]
    got = {}
    for goal in goals:
        mine = [r for r in rows if r['goal'] == goal]
        got[goal] = dict(report([t for t in trials if t['goal'] == goal],
                                refusals=sum(bool(d[3] and d[3].get('refused')) for r in mine for d in r['decisions']),
                                interface=sum(interface_loops([d[1] for d in r['decisions']]) for r in mine)),
                         controlled_successes=merged['cells'][policy][goal]['controlled_successes'])
    return got


def cells(out, p, skills, policy):
    """{task: {cell: report}} and the merged results of one pass."""
    from sim.search import interface_loops
    where, got, results = out / p['name'], {}, {}
    tasks = sorted({t for s in skills for t in TASKS[s]})
    for task in tasks:
        merged = read(where / task / 'results.json')
        results[task] = merged
        if merged is None:
            continue
        trials = [t for t in merged['trials'] if t['policy'] == policy]
        if task == 'combat':
            rows = [r for part in sorted((where / task).glob('trials-*.jsonl.gz')) for r in stream(part)
                    if r['policy'] == policy and 'decisions' in r]  # an unsupported trial's row keeps no decisions
            interface = sum(interface_loops([d[1] for d in r['decisions']]) for r in rows) if rows else None
            from sim.combat import GROUPS
            got[task] = {g: report([t for t in trials if g in (t['group'], 'all')], interface=interface if g == 'all' else None)
                         for g in (*GROUPS, 'all')}
        elif task == 'skills':
            got[task] = skills_cells(where / task, merged, policy)
        else:
            groups = ('ample', 'scarce') if task == 'supply' else ('all',)
            got[task] = {}
            for g in groups:
                mine = [t for t in trials if g in (t.get('variant'), 'all')]
                scored = [t for t in mine if 'success' in t]
                got[task][g] = report(mine, refusals=sum(t['refused'] for t in scored), stalls=sum(t['stalls'] for t in scored),
                                      interface=sum(t['interface_alternations'] for t in scored))
                if task == 'supply':
                    got[task][g].update(logs_mean=round(sum(t['logs'] for t in scored) / len(scored), 3) if scored else None,
                                        logs_mean_death_zero=round(sum(t['logs'] * t['survived'] for t in scored) / len(scored), 3)
                                        if scored else None)
    return got, results


def valid(merged_cell, n):
    """A rule's cell: every trial scored and the declared record count."""
    return merged_cell is not None and merged_cell['valid'] and merged_cell['scored'] == merged_cell['trials'] == n


def simulator_rules(out, skills, policy):
    rules, reports = [], {}
    R = GATE['rules']
    for p in GATE['passes']:
        got, results = cells(out, p, skills, policy)
        reports[p['name']] = got

        def rule(skill, name, value, threshold, merged_cell, n, gating=p['gating']):
            ok = valid(merged_cell, n)
            rules.append(dict(skill=skill, kind='simulator', rule=name, pass_name=p['name'], gating=gating, value=value,
                              threshold=threshold, n=n, valid=ok, passed=(value >= threshold) if ok else None))
        def cell(task, *keys):  # None: the evaluation left no result, so its rules are invalid
            c = None if results.get(task) is None else results[task]['cells'][policy]
            for k in keys:
                c = None if c is None else c[k]
            return c
        value = lambda c, key: None if c is None else c[key]
        if 'wood' in skills:
            count, n = R['wood']['successes']
            key = 'logs_mean' if R['wood']['logs'] == 'recorded' else 'logs_mean_death_zero'
            for v in ('ample', 'scarce'):
                c = cell('supply', v)
                rule('wood', f'supply.{v}.successes', value(c, 'successes'), count, c, n)
                rule('wood', f'supply.{v}.{key}', None if c is None else got['supply'][v][key], R['wood']['logs_mean'], c, n)
            c = cell('search')
            rule('wood', 'search.successes', value(c, 'successes'), R['wood']['successes'][0], c, R['wood']['successes'][1],
                 gating=False)
        for goal in ('fish', 'mine', 'wood', 'melee'):  # the short task gates fish and mine; wood's and melee's are reported
            if goal in skills:
                short = R[goal] if goal in ('fish', 'mine') else R['fish']
                (count, n), c = short['at_least'], cell('skills', goal)
                rule(goal, f'short.{short["count"]}', value(c, short['count']), count, c, n,
                     gating=p['gating'] and goal in ('fish', 'mine'))
        if 'melee' in skills:
            from sim.combat import GROUPS
            c = cell('combat', 'all')
            rule('melee', 'combat.all.successes', value(c, 'successes'), R['melee']['all'][0], c, R['melee']['all'][1])
            for g in GROUPS:
                c = cell('combat', g)
                rule('melee', f'combat.{g}.successes', value(c, 'successes'), R['melee']['group'][0], c, R['melee']['group'][1])
    return rules, reports


def native_rules(out, skills, records):
    """The native check's rules from each native pass's baseline.evaluate results (a missing or failed pass leaves its
    rules invalid)."""
    from baseline import config
    from baseline.resource_task import goal_items
    from sim.search import interface_loops
    N = GATE['native']
    family = NATIVE_FAMILY[records]
    rules, reports = [], {}
    for p in N['passes']:
        trials = []
        for engine in native_fixtures(skills, records):
            result = read(out / 'native' / p['name'] / engine / 'results.json')
            trials += [] if result is None or result.get('status') != 'completed' else result['trials']
        reports[p['name']] = {}
        for skill in ('wood', 'fish', 'mine'):
            if skill not in skills:
                continue
            names = [f'{family}_{site}_{skill}' for site in config.CHECK_SITES[skill]]
            mine = [t for t in trials if t['fixture'] in names]
            ok = [t['report']['survived'] and goal_items(t['report'], 'wood') >= N['wood_logs'] if skill == 'wood' else
                  bool(t[N['count']]) for t in mine]
            count, n = N['at_least'], len(config.CHECK_SITES[skill]) * (N['final2']['records'] if records == 'final2' else
                                                                         config.CHECK_RECORDS)
            ticks = sum(t['report']['delivered_ticks'] for t in mine)
            reports[p['name']][skill] = dict(
                trials=len(mine), successes=sum(ok), deaths=sum(not t['report']['survived'] for t in mine),
                by_site={name: sum(o for o, t in zip(ok, mine) if t['fixture'] == name) for name in names},
                decisions_per_1000_ticks=round(1000 * sum(t['decisions'] for t in mine) / ticks, 2) if ticks else None,
                refusals=sum(t['options_refused'] for t in mine), stalls=None, interface_loops=sum(
                    interface_loops([a for _, a in t['decision_ticks_actions']]) for t in mine))
            good = len(mine) == n
            rules.append(dict(skill=skill, kind='native', rule=f'native.{N["wood_logs"]}_logs_and_survival' if skill == 'wood'
                              else f'native.{N["count"]}', pass_name=p['name'], gating=p['gating'], value=sum(ok), threshold=count,
                              n=n, valid=good, passed=sum(ok) >= count if good else None))
    return rules, reports


def stale_keys(stored, current, key='gate'):
    """The keys, as dotted paths, at which a gate.json's stored gate differs from `current` (GATE after a JSON round
    trip, so its tuples read as lists). A key that only one side has differs."""
    if not (isinstance(stored, dict) and isinstance(current, dict)):
        return [] if stored == current else [key]
    return [k for name in sorted(set(stored) | set(current))
            for k in (stale_keys(stored[name], current[name], f'{key}.{name}') if name in stored and name in current
                      else [f'{key}.{name}'])]


def verdict(rules, skill, kind):
    gating = [r for r in rules if r['skill'] == skill and r['kind'] == kind and r['gating']]
    if not gating:
        return 'none'
    if any(r['passed'] is False for r in gating):
        return 'fail'
    return 'pass' if all(r['passed'] for r in gating) else 'incomplete'


def gate_json(out, skills, policy, checkpoint, records, native_run, timing):
    from baseline import config
    from baseline.telemetry import sha256
    import hashlib
    sim_rules, sim_reports = simulator_rules(out, skills, policy)
    nat_rules, nat_reports = native_rules(out, skills, records) if native_run else ([], {})
    rules = sim_rules + nat_rules
    fixtures = native_set(records)
    played = {name: fixtures[name] for names in native_fixtures(skills, records).values() for name in names}
    digests = dict(final=config.CHECK_DIGESTS, final2=GATE['native']['final2']['digests']).get(records)
    result = dict(version=GATE['version'], policy=policy, skills=list(skills),
                  checkpoint=None if checkpoint is None else dict(path=str(checkpoint), sha256=sha256(checkpoint)),
                  records=dict(name=records, **GATE['records'][records]), gate=GATE,
                  native_records=None if not native_run else dict(
                      fixtures=sorted(played), sha256=hashlib.sha256(json.dumps(played, sort_keys=True).encode()).hexdigest(),
                      **({'check_digests': digests} if digests else {})),
                  verdict={s: dict(simulator=verdict(rules, s, 'simulator'), native=verdict(rules, s, 'native'))
                           for s in skills},
                  rules=rules, report=dict(simulator=sim_reports, native=nat_reports), timing=timing)
    (out / 'gate.json').write_text(json.dumps(result, indent=1, default=str) + '\n', encoding='utf-8')
    return result


# ---- final4 -----------------------------------------------------------------------------------------------------
def final4_set(records, layouts=None):
    """The records final4 plays: FINAL4's frozen records ('final4', all of them), or for a rehearsal the same draw on
    the development split's first `layouts` ('development')."""
    if records == 'final4':
        if layouts not in (None, FINAL4['records']['layouts']):
            raise ValueError('the final4 records are all of their layouts')
        rec = FINAL4['records']
    else:
        rec = dict(FINAL4['development'], layouts=FINAL4['development']['layouts'] if layouts is None else layouts)
    return dict({k: rec[k] for k in ('split', 'first', 'layouts', 'key', 'world')}, sha256=rec.get('sha256'))


def records_sha256(pool, records):
    """A record set's digest in the native-click freeze's form (`534e6796…` is final2's short
    task): sha256 of the JSON list of [layout seed, layout sha256, record sha256] by record, the record's layout hashed
    array by array in name order (name, dtype, shape, bytes), the record as JSON with sorted keys."""
    import hashlib
    import numpy as np
    rows = []
    for record in records:
        layout, h = pool[record['layout']], hashlib.sha256()
        for name, a in sorted(layout.arrays().items()):
            h.update(name.encode())
            h.update(str(a.dtype).encode() + str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        rows.append([int(layout.seed), h.hexdigest(), hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()])
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def verify_final4_records(tables=None):
    """FINAL4's records drawn again as sim.run goals draws them (start_records) and checked against their frozen digest:
    raises ValueError when they differ, else returns the digest. It draws layouts and records and plays nothing."""
    from sim.data import Tables
    from sim.run import start_records
    rec = FINAL4['records']
    pool, records = start_records(tables or Tables.load(), split=rec['split'], layouts=rec['layouts'], first=rec['first'],
                                  key=rec['key'], starts=rec['layouts'], one_per_layout=True)
    digest = records_sha256(pool, records)
    if digest != rec['sha256']:
        raise ValueError(f'the final4 records differ from their frozen digest: {digest}')
    return digest


def final4_jobs(out, checkpoint, policy_seed, records, layouts=None):
    """(name, argv) of final4's sim.run goals processes, one a pass, all on the same records."""
    rec = final4_set(records, layouts)
    return [(p['name'], [sys.executable, '-m', 'sim.run', 'goals', '--checkpoint', str(checkpoint), '--policies', 'final',
                         '--split', rec['split'], *(['--final'] if rec['split'] == 'final' else []),
                         '--layouts', str(rec['layouts']), '--starts', str(rec['layouts']), '--one-per-layout',
                         '--first', str(rec['first']), '--key', str(rec['key']), '--goals', FINAL4['goal'],
                         *(['--sampled'] if p['sampled'] else []), '--seed', str(p['seed']), '--policy-seed', str(policy_seed),
                         '--output', str(out / p['name'])])
            for p in FINAL4['passes']]


def final4_json(out, checkpoint, policy_seed, records, layouts, timing):
    """final4.json: each pass's rule from its sim.run goals output, valid only when that output played the declared
    records and seeds with every trial scored and FINAL4's record count, and the verdict (verdict's rule: pass when
    both gating sampled passes hold, fail when a valid one misses, incomplete otherwise)."""
    from baseline.telemetry import sha256
    rec, goal, rule = final4_set(records, layouts), FINAL4['goal'], FINAL4['rule']
    checkpoint_sha256 = sha256(checkpoint)  # each pass must have played this checkpoint
    (count, n), rules, reports = rule['at_least'], [], {}
    for p in FINAL4['passes']:
        merged = read(out / p['name'] / 'results.json')
        declared = dict(split=rec['split'], first=rec['first'], layouts=rec['layouts'], starts=rec['layouts'], key=rec['key'],
                        one_per_layout=True, sampled=p['sampled'], seed=p['seed'], policy_seed=policy_seed, goals=[goal])
        played = merged is not None and all(merged.get(k) == v for k, v in declared.items()) and \
            (merged.get('checkpoints') or {}).get('final', {}).get('sha256') == checkpoint_sha256
        c = merged['cells']['final'][goal] if played else None
        ok = played and valid(c, n)
        value = None if c is None else c[rule['count']]
        rules.append(dict(skill=goal, kind='simulator', rule=f'final4.{rule["count"]}', pass_name=p['name'],
                          gating=p['gating'], value=value, threshold=count, n=n, valid=ok,
                          passed=(value >= count) if ok else None))
        reports[p['name']] = skills_cells(out / p['name'], merged, 'final', (goal,)) if played else None
    result = dict(version=FINAL4['version'], final4=FINAL4, records=dict(name=records, **rec), policy_seed=policy_seed,
                  checkpoint=dict(path=str(checkpoint), sha256=checkpoint_sha256),
                  verdict=verdict(rules, goal, 'simulator'), rules=rules, report=reports, timing=timing)
    (out / 'final4.json').write_text(json.dumps(result, indent=1, default=str) + '\n', encoding='utf-8')
    return result


# ---- free-play-v1: the statistics and the verdict from the rows of sim.hour free ------------------------------------
# A plan's settings: the world (sim/combat.py VILLAGES; free_plan_world), the record set and how it is played, each
# compared with the outputs' own.
FREE_PLAN_SETTINGS = ('world', 'split', 'first', 'records', 'key', 'xp_rate', 'muggers', 'ticks', 'read', 'period',
                      'control_seed')
FREE_PLAN = FREE_PLAN_SETTINGS + ('tables', 'run', 'start', 'pass_seeds', 'policy_seeds', 'claims')
RELAUNCH = 1_000_000  # one relaunch from the frozen initialization at environment seed + this
FREE_CLAIMS = ('L', 'B', 'R', 'no_rescue')
FREE = dict(
    statistics=dict(draws=10_000, seed=17017),  # baseline/compare.py paired_bootstrap: 95% percentile, paired by record
    passes=('sampled1', 'sampled2'),            # the claims' passes: each must hold, never pooled; greedy is described
    zero='random',                              # Claim L: policy - random legal play
    references=dict(wood=('woodcutting',), fish=('fishing',), mine=('mining',),  # Claim B: each reference's own skills
                    melee=('attack', 'strength', 'defence', 'hitpoints')),
    timer='rotation',                           # Claim R: policy - the fixed-timer rotation
    exploits=('chickens', 'once', 'eat'),       # No rescue
    # The frozen plans, as FINAL4 freezes its own (records and statistics, frozen before training).
    # A verdict (pass, fail, match) is written only for outputs that match one plan exactly (free_plan); other outputs
    # give their numbers as a description. A plan holds FREE_PLAN: its settings, the tables' sha256, the training run's
    # frozen facts ({fact: value} over sim/hour.py FREE['run_facts']: decisions, worlds, the learner's settings, the
    # batch's muggers and life, the layout pool's sha256, the tables' sha256; beside them FREE['engine_facts'], the C
    # engine and its pack's sha256, or none of the two for the Python reference: free_plan_engine), each policy seed's start ({seed: its initialization's tensor
    # digest, policy_sha256_at_start}), the sampled passes' seeds ({sampled1: seed, sampled2: seed}; greedy plays
    # sampled1's), the policy seeds and the claims it reads. Values are compared in the JSON form that the manifests
    # hold (a tuple equals a list). A plan that holds a None matches no output. final: the five-seed final read (Claims
    # L, B and R and No rescue); its records, seeds, claims and controls are fixed in advance, and a later commit may
    # add only its decision budget and start digests, so it holds no value yet. pilot: one seed on development
    # records, Claim L alone as a learning signal; frozen on 12 November 2026. Its digests were computed with the
    # code's own functions: the tables' sha256 (sim/build.py DATA), the 128-layout pool's
    # (logs/sim/world/train-128-1000-nc1.npz) and seed 77's start (sim.train train --initial-only at the plan's settings).
    # The world: a plan frozen after the world became a recorded setting names it (`world`; final holds the key); the
    # pilot's, frozen before (12 November 2026), names none and is village-world-v1's (free_plan_world), its values as
    # frozen. A village-world-v2 plan names its world. wider_pilot: the pilot of village-world-v2, one seed on its
    # development records, Claim L alone as a learning signal; frozen on 14 November 2026, its training on the C
    # engine. Its digests were computed with the code's own functions: the tables' sha256 (sim/hour.py
    # world_archive of village-world-v2: logs/sim/data/draynor-nc2.npz), the pool's (summary 'pool':
    # logs/sim/world/train-128-20000-v2.npz, train indices 20,000 to 20,127), the pack's (summary 'pack', the file's
    # sha256, exported from both without route trees) and seed 79's start (sim.train train --initial-only at the plan's
    # settings on this Mac's C engine with that pack and 8 workers; the same digest in two runs). The pack's pin is the
    # plan's second freeze (15 November 2026) and the one value it changed: the first launch stopped on the XP check's
    # rule (a hit in the cycle of its npc's respawn owes no XP), and every pack was exported again at the repair (the
    # npc_respawn event in both engines), so the first pin, train-128-20000-v2-a6c325a-ids2.pack's (3f17a1c1...), no
    # longer opens. The pin is now train-128-20000-v2-a6c325a-ids3.pack's; a start-only run after the repair with it
    # gives seed 79's frozen start digest.
    # Its run pins the engine facts beside the run facts; the run's world is no run fact but the batch's (summary
    # free_play batch_in_effect, read by free_plan_differs through sim/hour.py trained_world).
    plans=dict(
        final=dict.fromkeys(FREE_PLAN),
        pilot=dict(split='development', first=4000, records=64, key=223, xp_rate=1, muggers=False, ticks=36000,
                   read=6000, period=4500, control_seed=4253,
                   tables='307b77a7a58327782bc25654127048968d12c9e88ef1a74c221ca60b3d6c5771',
                   run=dict(decisions=4194304, worlds=1024, n_steps=64, batch_size=1024, n_epochs=4, learning_rate=0.0003,
                            gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                            lstm_hidden_size=128, half_life_ticks=1000.0,
                            value_norm=dict(method='popart', beta=0.1, min_std=0.01), muggers=False,
                            life=('uniform', 6000, 36000),
                            pool='1cccffbede7f2613b5baf31e9f436a22bc84202eb3ed99023c33e0cd553d9469',
                            tables='307b77a7a58327782bc25654127048968d12c9e88ef1a74c221ca60b3d6c5771', validate=True),
                   start={77: '96b1f9872b62bef7034b8502b227273ece84f6c0d988d50e3ccc6b8223105d65'},
                   pass_seeds=dict(sampled1=229, sampled2=233), policy_seeds=(77,), claims=('L',)),
        wider_pilot=dict(world='village-world-v2', split='development', first=7000, records=64, key=239, xp_rate=1,
                         muggers=False, ticks=36000, read=6000, period=4500, control_seed=4253,
                         tables='7dee885e663b89956f2416262582d34f8438caa3eeb6a3421b57bdad862eba19',
                         run=dict(decisions=4194304, worlds=1024, n_steps=64, batch_size=1024, n_epochs=4,
                                  learning_rate=0.0003, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
                                  max_grad_norm=0.5, lstm_hidden_size=128, half_life_ticks=1000.0,
                                  value_norm=dict(method='popart', beta=0.1, min_std=0.01), muggers=False,
                                  life=('uniform', 6000, 36000),
                                  pool='56e2607b7a0e5a7e4a19797a13b6b61774709adf69523d7ba6faf3ced8df69bb',
                                  tables='7dee885e663b89956f2416262582d34f8438caa3eeb6a3421b57bdad862eba19', validate=True,
                                  engine='native',
                                  # pack was 97640806ca29025e565bafd37bdd2fe6e3160700a47b02339761659fc08b683f, moved
                                  # on 20 November 2026; the 34 sections are byte-equal
                                  pack='21f147bfa83868ec5d5de456502cec7c97e2e72d875a45f337319dc5900cd04c'),
                         start={79: 'e355680184117f730fef1d9f3c6050c60c21922ca33218c5b5c4aad35410eb6c'},
                         pass_seeds=dict(sampled1=251, sampled2=257), policy_seeds=(79,), claims=('L',)),
        # wider_final: five seeds of village-world-v2 on its final records, Claims L, B and R and No rescue; frozen on
        # 16 November 2026. The records, pass and control seeds, claims and controls are the wider world's freeze of
        # 14 November; the run is wider_pilot's, its budget the pilot's 4,194,304 decisions (in place of the earlier
        # budget rule for this block), on the C engine with wider_pilot's pack. Each start digest was made by sim.train
        # train --initial-only at this run's settings on this Mac's C engine with that pack and 8 workers, the same
        # digest in two runs a seed; the same command gives seed 79's frozen start (e3556801...). A seed's tensors at
        # its start depend on the seed and the network alone: they equal the first world's freeze dry run's.
        wider_final=dict(world='village-world-v2', split='final', first=8000, records=256, key=241, xp_rate=1,
                         muggers=False, ticks=36000, read=6000, period=4500, control_seed=4253,
                         tables='7dee885e663b89956f2416262582d34f8438caa3eeb6a3421b57bdad862eba19',
                         run=dict(decisions=4194304, worlds=1024, n_steps=64, batch_size=1024, n_epochs=4,
                                  learning_rate=0.0003, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
                                  max_grad_norm=0.5, lstm_hidden_size=128, half_life_ticks=1000.0,
                                  value_norm=dict(method='popart', beta=0.1, min_std=0.01), muggers=False,
                                  life=('uniform', 6000, 36000),
                                  pool='56e2607b7a0e5a7e4a19797a13b6b61774709adf69523d7ba6faf3ced8df69bb',
                                  tables='7dee885e663b89956f2416262582d34f8438caa3eeb6a3421b57bdad862eba19', validate=True,
                                  engine='native',
                                  # pack was 97640806ca29025e565bafd37bdd2fe6e3160700a47b02339761659fc08b683f, moved
                                  # on 20 November 2026; the 34 sections are byte-equal
                                  pack='21f147bfa83868ec5d5de456502cec7c97e2e72d875a45f337319dc5900cd04c'),
                         start={101: '5dc9abefeb3349d6c37cb3dcb3d97bfa52e26df027bb00ab8efd1e00d67695e5',
                                202: '18ce6031a43d9207acd86220c12bbacdd075e728c051a8c8891058a99a011642',
                                303: '1be77d8c94ee4f97a6221cb90e78a1f5ece4e109c278a154ef86e522dbb65293',
                                404: '8be523c0832a682e9278ef15860e8e7ea06068d1754caf148ca43f85677f09ba',
                                505: '7e71409a474ed3d76ec45463c0b92f6d7558b2a175348d476a93573d1dff04e9'},
                         pass_seeds=dict(sampled1=251, sampled2=257), policy_seeds=(101, 202, 303, 404, 505),
                         claims=('L', 'B', 'R', 'no_rescue')),
    ),
    statement=("No claim is about generalization: the final records are new draws from the training world's own "
               "distribution, and new layouts and native sites are phase 4's."),
    readings=dict(no_rescue='the return is cheap in this world', B='learns, not shown beyond one activity',
                  R='not shown better than a timer'),
    off_rate='description (25x, out of the training distribution)',  # any XP rate but 1: differences only, no claim
)
# Each claim's own controls (free_claims): named in a seed's incomplete_controls when one is not complete.
FREE_CONTROLS = dict(L=(FREE['zero'],), B=tuple(FREE['references']), R=(FREE['timer'],), no_rescue=FREE['exploits'])
# The settings every output of one evaluation shares: the record set, the code and the tables (sim/hour.py free_run).
FREE_SHARED = ('task', 'rows', 'world', 'observation', 'source', 'split', 'first', 'records', 'key', 'muggers', 'xp_rate',
               'ticks', 'read', 'control_seed')


def free_outputs(outputs):
    """(settings, policies, cells, evidence) of sim.hour free outputs read as one evaluation: every manifest holds the
    same FREE_SHARED settings and tables and at least one record, each life is there once, one rotation period at most,
    each policy seed one policy and one set of pass seeds, each checkpoint and initialization under one seed, and no
    record whose two sampled passes share a seed; an output or a row of a changed world (sim.hour free --change) is
    refused: it is a description of the scorecard, never an evaluation that a plan reads. policies: {seed: the
    manifest's policy, with its pass seeds under
    'passes'}; cells: {(actor, policy seed or None, pass or None): {record: row}}; evidence: the parts' cut last lines,
    the lives files a manifest declares that are missing, and the unsupported lives."""
    from sim.hour import FREE as PLAY, POLICY_ACTORS, free_rows
    settings, policies, digests, cells, periods = None, {}, {}, {}, set()
    evidence = dict(cut=[], missing=[])
    for output in map(Path, outputs):
        manifest = read(output / 'manifest.json')
        if manifest is None:
            raise ValueError(f'{output} holds no sim.hour free manifest')
        if manifest.get('rows') != PLAY['rows']:
            raise ValueError(f"{output} holds rows of revision {manifest.get('rows')}: this code reads {PLAY['rows']}")
        if manifest.get('change') is not None:
            raise ValueError(f"{output} holds the changed world {manifest['change'].get('name')!r}: a description of the "
                             'scorecard, never an evaluation that a plan reads')
        shared = dict({k: manifest.get(k) for k in FREE_SHARED}, tables=manifest['tables']['sha256'])
        if settings is not None and shared != settings:
            raise ValueError(f'{output} is another evaluation: {sorted(k for k in shared if shared[k] != settings[k])} differ')
        settings = shared
        if not (type(settings['records']) is int and settings['records'] >= 1):
            raise ValueError(f"{output} declares {settings['records']!r} records: an evaluation reads one or more")
        if manifest['policy'] is not None:
            seed = manifest['policy']['seed']
            policy = dict(manifest['policy'], passes={p['name']: p['seed'] for p in manifest['passes']})
            known = policies.setdefault(seed, policy)
            if known['passes'] != policy['passes']:
                raise ValueError(f"seed {seed}'s outputs hold other pass seeds: {known['passes']} and {policy['passes']}")
            if known != policy:
                raise ValueError(f'two policies of seed {seed}')
            for actor in POLICY_ACTORS:
                digest = manifest['policy'][actor]['sha256']
                if digests.setdefault(digest, seed) != seed:
                    raise ValueError(f'one checkpoint ({digest[:12]}) under seeds {digests[digest]} and {seed}')
        periods |= {manifest['period']} - {None}
        if len(periods) > 1:
            raise ValueError(f'rotations of {sorted(periods)} ticks: one evaluation reads one rotation (Claim R)')
        for index in range(manifest['parts']):
            path = output / f'lives-{index}.jsonl'
            if not path.exists():
                evidence['missing'].append(f'{output.name}/{path.name}')
                continue
            rows, short = free_rows(path)
            evidence['cut'] += [f'{output.name}/{path.name}'] if short else []
            for row in rows:
                if row.get('change') is not None:
                    raise ValueError(f"{output}/{path.name}: a life of the changed world {row['change']!r}, never a row of "
                                     'an evaluation that a plan reads')
                cell = cells.setdefault((row['actor'], row['policy_seed'], row['pass']), {})
                if row['record'] in cell or not 0 <= row['record'] < settings['records']:
                    raise ValueError(f"{output}: {row['actor']} {row['pass']} record {row['record']} twice or outside the records")
                cell[row['record']] = row
    for (actor, seed, p), cell in cells.items():
        other = cells.get((actor, seed, 'sampled2'), {}) if p == 'sampled1' else {}
        same = sorted(r for r, row in cell.items() if r in other and other[r]['seed'] == row['seed'])
        if same:
            raise ValueError(f'{actor} of seed {seed}: records {same} play one seed in both sampled passes')
    evidence['unsupported'] = [dict(actor=a, policy_seed=s, pass_name=p, record=r['record'], site=r['unsupported']['file'],
                                    line=r['unsupported']['line'], message=r['unsupported']['message'])
                               for (a, s, p), cell in cells.items() for r in cell.values() if r['unsupported'] is not None]
    return dict(settings, period=next(iter(periods), None)), policies, cells, evidence


def free_frozen(plan):
    """A plan holds a value for every key and no None at any depth (in its run facts, starts and pass seeds too), one
    policy seed and one claim at least."""
    unset = lambda v: v is None or isinstance(v, dict) and any(map(unset, v.values())) or \
        isinstance(v, (list, tuple)) and any(map(unset, v))
    return not unset(plan) and plan['policy_seeds'] not in ((), []) and plan['claims'] not in ((), [])


def free_plan_world(plan):
    """The world a plan reads: its `world`, held to as it is. Any plan that names none reads sim/hour.py
    FREE['first_world'] (village-world-v1): in practice the pilot's, frozen before any other world existed, since the
    final plan holds the key and a frozen plan holds no None."""
    from sim.hour import FREE as PLAY
    return plan['world'] if 'world' in plan else PLAY['first_world']


def free_plan_engine(plan):
    """The engine facts (sim/hour.py FREE['engine_facts']) that a plan pins: those its run holds, or, when it holds the
    run facts alone, the Python reference's (FREE['reference_engine']: engine 'cpu', no pack), the form of the pilot's
    plan, frozen before the C engine trained free play."""
    from sim.hour import FREE as PLAY
    return {k: plan['run'][k] for k in PLAY['engine_facts'] if k in plan['run']} or dict(PLAY['reference_engine'])


def free_plan_shape(name, plan, facts, engine):
    """Refuse a frozen plan that no output could match for its form: a world of sim/combat.py VILLAGES (or none:
    free_plan_world); claims a tuple or list among FREE_CLAIMS, each
    once; policy seeds a tuple or list of whole numbers; pass seeds {sampled1, sampled2} of two different whole numbers;
    start {policy seed: digest}; run with exactly the run facts `facts`, or those and the engine facts `engine`, which
    name the C engine and its pack's sha256 (a plan of the Python reference holds the run facts alone); every value of
    a form that JSON holds."""
    from sim.combat import VILLAGES
    passes, start, bad = plan['pass_seeds'], plan['start'], []
    claims, seeds = plan['claims'], plan['policy_seeds']
    if free_plan_world(plan) not in VILLAGES:
        bad.append(f'world one of {sorted(VILLAGES)}, or no world (village-world-v1: free_plan_world)')
    if not (isinstance(claims, (tuple, list)) and set(claims) <= set(FREE_CLAIMS) and len(set(claims)) == len(claims)):
        bad.append(f'claims a tuple or list among {list(FREE_CLAIMS)}, each once')
    if not (isinstance(seeds, (tuple, list)) and all(type(s) is int for s in seeds) and len(set(seeds)) == len(seeds)):
        bad.append('policy_seeds a tuple or list of different whole numbers')
    if not (isinstance(passes, dict) and set(passes) == set(FREE['passes']) and all(type(v) is int for v in passes.values())
            and len(set(passes.values())) == len(passes)):
        bad.append(f'pass_seeds of {list(FREE["passes"])}, two different whole numbers')
    if not (isinstance(start, dict) and isinstance(seeds, (tuple, list)) and set(start) == set(seeds)
            and all(isinstance(v, str) for v in start.values())):
        bad.append('start {policy seed: its initialization\'s tensor digest} for each policy seed')
    if not isinstance(plan['run'], dict):
        bad.append(f'run {{fact: value}} over {list(facts)}')
    else:
        expected = set(facts) | (set(engine) if set(engine) & set(plan['run']) else set())
        if set(plan['run']) != expected:
            bad.append(f"run without the facts {sorted(expected - set(plan['run']))} and with "
                       f"{sorted(set(plan['run']) - expected)} beside them")
        elif set(engine) <= set(plan['run']) and not (plan['run']['engine'] == 'native' and isinstance(plan['run']['pack'], str)):
            bad.append("engine facts of the C engine ('native') and its pack's sha256: a plan of the Python reference holds "
                       'the run facts alone')
    for key, value in [*((k, plan[k]) for k in plan if k != 'run'),
                       *((f'run.{k}', v) for k, v in (plan['run'] if isinstance(plan['run'], dict) else {}).items())]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            bad.append(f'{key} of a form that JSON holds, not {value!r}')
    if bad:
        raise ValueError(f'plan {name} is malformed: it needs {"; ".join(bad)}')


def free_clean(code):
    """The outputs' source is one commit with no change (no changed or untracked file, no patch)."""
    code = code or {}
    return code.get('commit') is not None and not code.get('changed') and not code.get('untracked') and \
        code.get('patch') is None


def free_plan_run(policies, code):
    """Why no plan reads these policies, whatever its values: a run that has not completed; a checkpoint that is not
    its run's final (latest.zip, whose tensor digest the run summary records as policy_sha256_at_end); a summary that
    records no final digest; a start other than these three (no --init in any): the seed's own (environment seed = the
    seed, with no --weights); the frozen initialization's file at the seed's own environment seed (--weights: a run on
    another host trains from this Mac's file, since an initialization made there can differ in its last bits); or its
    one relaunch from the frozen initialization (environment seed = the seed + RELAUNCH, with --weights). For each, the
    start digest, compared with the plan's (free_plan_differs), shows the tensors were the frozen initialization's; a summary
    that records no training tables (the fake environment; a C-engine run from before its summary recorded the archive
    its pack was held equal to); a C-engine run whose pack does not record its own tables and pool as the inputs it was
    exported from (summary 'pack' inputs: what ties the plan's pack to the plan's tables and pool); a run whose
    summary records no source (an older run), whose source is not one commit with no change, or whose commit is not the
    commit of the outputs' source `code`."""
    runs = {s: p['run'] for s, p in sorted(policies.items())}
    unfinished = [s for s, r in runs.items() if r['status'] != 'completed']
    later = {s: r['checkpoint'] for s, r in runs.items() if r['checkpoint'] != 'latest.zip' or
             r.get('policy_sha256_at_end') not in (None, policies[s]['policy'].get('policy_sha256'))}
    undigested = [s for s, r in runs.items() if r.get('policy_sha256_at_end') is None]
    own = lambda s, r: r.get('env_seed') == s and r.get('init') is None  # its own start, or from the frozen file
    relaunch = lambda s, r: r.get('env_seed') == s + RELAUNCH and r.get('weights') is not None and r.get('init') is None
    started = {s: r.get('env_seed') for s, r in runs.items() if not (own(s, r) or relaunch(s, r))}
    untabled = {s: f"env {r.get('env')}, engine {r.get('engine')}" for s, r in runs.items()
                if (r.get('facts') or {}).get('tables') is None}
    exported = lambda r: {k: ((r.get('pack') or {}).get('inputs') or {}).get(k) for k in ('tables', 'pool')}
    unproved = {s: exported(r) for s, r in runs.items() if r.get('engine') == 'native' and s not in untabled and
                exported(r) != {k: (r.get('facts') or {}).get(k) for k in ('tables', 'pool')}}
    unsourced = [s for s, r in runs.items() if r.get('source') is None]
    unclean = {s: r['source'] for s, r in runs.items() if r.get('source') is not None and not free_clean(r['source'])}
    commit = (code or {}).get('commit')
    elsewhere = {s: r['source'].get('commit') for s, r in runs.items()
                 if r.get('source') is not None and s not in unclean and r['source'].get('commit') != commit}
    return ([f'a checkpoint of a run that has not completed (seeds {unfinished})'] if unfinished else []) + \
        ([f"a checkpoint that is not its run's final ({', '.join(f'seed {s}: {c}' for s, c in later.items())})"]
         if later else []) + \
        ([f'a run summary that records no final checkpoint digest (policy_sha256_at_end; seeds {undigested})']
         if undigested else []) + \
        ([f"a run started otherwise than at its own seed or as its one relaunch from the initialization at seed + "
          f"{RELAUNCH:,} ({', '.join(f'seed {s}: environment seed {e}' for s, e in started.items())})"] if started else []) + \
        ([f"a run summary that records no training tables, as the fake environment's and an older C-engine run's: a plan "
          f"pins the archive a run trained on ({', '.join(f'seed {s}: {e}' for s, e in untabled.items())})"]
         if untabled else []) + \
        ([f"a C-engine run whose pack does not record its tables and pool as the inputs it was exported from "
          f"({', '.join(f'seed {s}: the pack records {e}' for s, e in unproved.items())})"] if unproved else []) + \
        ([f'a run summary that records no source, as an older run\'s (seeds {unsourced})'] if unsourced else []) + \
        ([f"a run trained from a source that is not one commit with no change ({', '.join(f'seed {s}: {c}' for s, c in unclean.items())})"]
         if unclean else []) + \
        ([f"a run trained at another commit than the outputs' {commit} ({', '.join(f'seed {s}: {c}' for s, c in elsewhere.items())})"]
         if elsewhere else [])


def free_plan_differs(plan, settings, policies, period=True):
    """What of a frozen plan of its form the outputs differ in, compared in the manifests' JSON form: the settings (the
    world, free_plan_world's; the rotation's period only when `period`), the tables, the policy seeds, each seed's pass
    seeds, the world each seed's run trained in (sim/hour.py trained_world), the run facts, the engine facts
    (free_plan_engine: the plan's, or the Python reference's) and each seed's start digest."""
    from sim.hour import FREE as PLAY, engine_facts, trained_world
    pinned = free_plan_engine(plan)
    plan = dict(json.loads(json.dumps(plan)), world=free_plan_world(plan))
    differ = [k for k in FREE_PLAN_SETTINGS if (period or k != 'period') and settings[k] != plan[k]]
    differ += [] if settings['tables'] == plan['tables'] else ['tables']
    differ += [] if set(policies) <= set(plan['policy_seeds']) else ['policy_seeds']
    sampled = lambda passes: {p: s for p, s in passes.items() if p in FREE['passes']}
    differ += [] if all(sampled(p['passes']) == plan['pass_seeds'] for p in policies.values()) else ['pass_seeds']
    trained = lambda run: trained_world(run.get('batch_in_effect') or {}, (run.get('facts') or {}).get('tables'))  # noqa: E731
    elsewhere = [s for s, p in sorted(policies.items()) if trained(p['run']) != plan['world']]
    differ += [f'the run\'s world (seeds {elsewhere})'] if elsewhere else []
    facts = sorted({k for p in policies.values() for k, v in plan['run'].items()
                    if k not in PLAY['engine_facts'] and (p['run'].get('facts') or {}).get(k) != v} |
                   {k for p in policies.values() for k, v in pinned.items() if engine_facts(p['run'])[k] != v})
    differ += [f'run ({", ".join(facts)})'] if facts else []
    starts = [s for s, p in sorted(policies.items()) if p['run'].get('policy_sha256_at_start') != plan['start'].get(str(s))]
    return differ + ([f'start (seeds {starts})'] if starts else [])


def free_plan(settings, policies, controls=True):
    """(the name of the plan that the outputs match exactly, or None, and why they match no plan). A plan is matched when
    it is frozen (free_frozen) and of its form (free_plan_shape), at least one policy and one control output were read,
    no reason of free_plan_run holds, and the outputs differ from it in nothing (free_plan_differs). At an XP rate other
    than 1, no plan: the 25x read is a description only."""
    from sim.hour import FREE as PLAY
    if settings['xp_rate'] != 1:
        return None, [f"XP rate {settings['xp_rate']}: out of the training distribution"]
    why = (['no policy output: a plan reads one at least'] if not policies else []) + \
        (['no control output: a plan reads the controls of its records'] if not controls else []) + \
        free_plan_run(policies, settings['source'])
    blocked, matched = bool(why), []
    for name, plan in FREE['plans'].items():
        if not free_frozen(plan):
            why.append(f'plan {name} is not frozen')
            continue
        free_plan_shape(name, plan, PLAY['run_facts'], PLAY['engine_facts'])
        differ = free_plan_differs(plan, settings, policies, period=controls)
        if differ:
            why.append(f'not plan {name}: {", ".join(differ)} differ')
        elif not blocked:
            matched.append(name)
    if len(matched) > 1:
        raise ValueError(f'the outputs match the plans {matched}: one evaluation is read under one plan')
    return (matched[0], []) if matched else (None, why)


def free_plan_check(name, settings, policies, code):
    """Why plan `name` would not read an output of sim.hour free (its --plan), before any life is played: the plan is
    not frozen, the XP rate is not 1, the source is not one commit with no change, a reason of free_plan_run holds, or
    the output differs from it (the rotation's period only when the output plays the rotation). The verdict step stays
    the judge; this refuses early with its reasons. A plan of another form raises ValueError."""
    from sim.hour import FREE as PLAY
    if name not in FREE['plans']:
        raise ValueError(f'no plan {name}: the plans are {sorted(FREE["plans"])}')
    plan = FREE['plans'][name]
    if not free_frozen(plan):
        return [f'plan {name} is not frozen']
    free_plan_shape(name, plan, PLAY['run_facts'], PLAY['engine_facts'])
    differ = free_plan_differs(plan, settings, policies, period=settings['period'] is not None)
    return ([f"XP rate {settings['xp_rate']}: out of the training distribution"] if settings['xp_rate'] != 1 else []) + \
        ([f'the source is not one commit with no change: {code}'] if not free_clean(code) else []) + \
        free_plan_run(policies, code) + ([f'{", ".join(differ)} differ'] if differ else [])


def free_cell(cell, records, value=lambda r: r['gain_s'], died=lambda r: r['death'] is not None):
    """One actor's lives in one pass: complete when every record is there and none stopped unsupported; the mean of
    `value` (the integer level gain over S) and the deaths a life, over the lives that ended."""
    ended = [r for r in (cell or {}).values() if r['unsupported'] is None]
    mean = lambda v: round(sum(v) / len(v), 4) if v else None
    return dict(lives=len(cell or {}), records=records, complete=len(ended) == records, unsupported=len(cell or {}) - len(ended),
                mean_gain=mean([value(r) for r in ended]), deaths=sum(died(r) for r in ended),
                deaths_a_life=mean([died(r) for r in ended]))


def free_difference(policy, control, records, value=lambda r: r['gain_s']):
    """The mean paired difference policy - control of `value` over the records and its 95% percentile bootstrap interval
    (baseline/compare.py paired_bootstrap; FREE statistics), the pairs taken by record; None unless both are complete."""
    from baseline.compare import paired_bootstrap
    if not (free_cell(policy, records)['complete'] and free_cell(control, records)['complete']):
        return None
    diffs = [value(policy[r]) - value(control[r]) for r in range(records)]
    return paired_bootstrap(diffs, draws=FREE['statistics']['draws'], seed=FREE['statistics']['seed'])


def above_zero(difference):
    """A claim's test of one difference: its lower bound above zero (None, incomplete, when it has none)."""
    return dict(difference=difference, threshold=0, holds=None if difference is None else difference['low'] > 0)


def all_of(values):
    """True when all hold, False when one fails, else None (incomplete)."""
    return False if any(v is False for v in values) else True if all(v is True for v in values) else None


def free_claims(policy, controls, records):
    """Claims L, B and R and No rescue in one pass, with every number each one used."""
    outside = lambda own: lambda r: r['gain_s'] - sum(r['gain'].get(s, 0) for s in own)
    claims = dict(L=above_zero(free_difference(policy, controls.get(FREE['zero']), records)))
    refs = {}
    for ref, own in FREE['references'].items():
        total = above_zero(free_difference(policy, controls.get(ref), records))
        beyond = above_zero(free_difference(policy, controls.get(ref), records, outside(own)))
        refs[ref] = dict(own=list(own), total=total, outside=beyond, holds=all_of([total['holds'], beyond['holds']]))
    claims['B'] = dict(references=refs, holds=all_of([r['holds'] for r in refs.values()]))
    claims['R'] = above_zero(free_difference(policy, controls.get(FREE['timer']), records))
    rescue = {}
    for e in FREE['exploits']:
        test = above_zero(free_difference(policy, controls.get(e), records))
        rescue[e] = dict(test, matches=None if test['holds'] is None else not test['holds'])
    matched = [r['matches'] for r in rescue.values()]
    claims['no_rescue'] = dict(exploits=rescue, matches=True if True in matched else
                               False if all(m is False for m in matched) else None)
    return claims


def unread(claim):
    """A claim that no plan reads, its holds and matches renamed would_hold and would_match, so that no reader takes
    them for a result."""
    names = dict(holds='would_hold', matches='would_match')
    return {names.get(k, k): unread(v) if isinstance(v, dict) else v for k, v in claim.items()}


def tri(passes, key=lambda c: c['holds']):
    """A claim over a seed's two passes, never pooled: pass when it holds in each, fail when a valid pass fails."""
    return {True: 'pass', False: 'fail', None: 'incomplete'}[all_of([key(c) for c in passes])]


def free_readings(words):
    """The readings of a verdict's claim words: an exploit that matches, "the return is cheap
    in this world"; L passes and B fails, "learns, not shown beyond one activity"; L and B pass and R fails, "not shown
    better than a timer". Any other verdict has no reading beyond its words."""
    r, w = FREE['readings'], words.get
    return [r['no_rescue']] * (w('no_rescue') == 'match') + [r['B']] * (w('L') == 'pass' and w('B') == 'fail') + \
        [r['R']] * (w('L') == 'pass' and w('B') == 'pass' and w('R') == 'fail')


def free_reads(cell, ticks):
    """The unpaid reads of one actor's lives in one pass (no bar, never paid). a_life: the
    means a life of the levels gained by skill, production by family, kills by npc name, damage taken by source, food
    eaten, eats at full hitpoints, time by activity and switches between activities. totals, over all the lives: the
    deaths with their ticks and killers, the eats by the hitpoints before each, and the most switches in one life. An
    activity is an XP event's cause (sim/hour.py free_reads); the ticks from the previous XP event (or arm) to an event
    count for its activity, and those after the last to the life's end for 'none'. A switch is an XP event whose activity
    differs from the one before it."""
    lives = [r for r in (cell or {}).values() if r['unsupported'] is None]
    if not lives:
        return None
    n, sums = len(lives), lambda rows: dict(sum((Counter(r) for r in rows), Counter()))
    per = lambda counts: {k: round(v / n, 4) for k, v in sorted(counts.items())}
    times, switches = Counter(), []
    for r in lives:
        last = 0
        for activity, _, end, _ in r['activity']:
            times[activity] += end - last
            last = end
        times['none'] += r['ticks'] - last
        switches.append(max(0, len(r['activity']) - 1))
    deaths = [r['death'] for r in lives if r['death'] is not None]
    a_life = dict(levels=per(sums([r['gain'] for r in lives])), made=per(sums([r['made'] for r in lives])),
                  kills=per(sums([r['kills'] for r in lives])), damage=per(sums([r['damage'] for r in lives])),
                  eats=round(sum(len(r['eat_log']) for r in lives) / n, 4),
                  eats_full=round(sum(r['eats_full'] for r in lives) / n, 4), time=per(times),
                  switches=round(sum(switches) / n, 4))
    totals = dict(deaths=dict(count=len(deaths), ticks=sorted(d['tick'] for d in deaths),
                              killers=dict(Counter(d['killer'] for d in deaths))),
                  eats_by_hp=dict(sorted(Counter(e[1] for r in lives for e in r['eat_log']).items())),
                  switches_max=max(switches))
    return dict(lives=n, ticks=ticks, a_life=a_life, totals=totals)


def free_json(outputs, path):
    """free.json from the rows of sim.hour free outputs alone: for each
    policy seed and pass, never pooled, every actor's mean integer level gain and deaths a life, the paired differences
    policy - each other actor, Claims L, B and R and No rescue with every number each used, the 6,000-tick read and the
    unpaid reads; greedy and the read are descriptions with no claim. A verdict is written only when the outputs match a
    frozen plan (free_plan), from the plan's claims over the plan's seeds: a seed's claim passes when it holds in both
    sampled passes, fails when a valid pass fails, and is incomplete otherwise (a missing or unsupported life, or a
    declared seed with no output), and every claim of a seed is incomplete unless its WAIT, initialization (both sampled
    passes) and greedy lives are complete; over the seeds, a claim passes in every seed or fails in one. No rescue
    matches when an exploit script matches in one pass or the other of one seed. A pass's claims hold only the plan's;
    every other claim, and every claim of outputs that match no plan, is under 'described', with no word of its own
    and its holds and matches named would_hold and would_match. A seed's incomplete_controls names each control that
    is not complete: WAIT, its initialization and greedy (which make every word incomplete) and each read claim's own.
    Outputs that match no plan give the verdict kind 'description'; at an XP rate other than 1, the differences only
    and FREE's off_rate kind. Every
    free.json holds FREE's statement, both kinds of seed, the policy's deaths a life in each verdict, and whether WAIT,
    the initialization and greedy play are complete."""
    settings, policies, cells, evidence = free_outputs(outputs)
    n = settings['records']
    controls = {a: cell for (a, s, p), cell in cells.items() if s is None and p is None}
    plan, why = free_plan(settings, policies, controls=bool(controls))
    if plan is not None and not free_clean(settings['source']):
        raise ValueError(f"plan {plan} reads outputs of one commit with no change, not of {settings['source']}")
    claimed = settings['xp_rate'] == 1
    read = FREE['plans'][plan]['claims'] if plan else ()
    read_value, read_died = (lambda r: r['read']['gain_s']), (lambda r: r['read']['died'])
    describe = lambda cell: dict(free_cell(cell, n), read=free_cell(cell, n, read_value, read_died),
                                 reads=free_reads(cell, settings['ticks']))
    kind = 'verdict' if plan else 'description' if claimed else FREE['off_rate']
    sampled, complete = FREE['passes'], lambda cell: free_cell(cell, n)['complete']
    seeds = {}
    for seed in (FREE['plans'][plan]['policy_seeds'] if plan else sorted(policies)):
        if seed not in policies:  # a declared seed with no output
            seeds[seed] = dict(missing=True, verdict=dict(kind=kind, **{c: 'incomplete' for c in read}, readings=[],
                                                          deaths_a_life=None))
            continue
        passes = {}
        for p in (*sampled, 'greedy'):
            mine = cells.get(('policy', seed, p))
            others = dict(controls, **({} if p == 'greedy' else {'initial': cells.get(('initial', seed, p))}))
            passes[p] = dict(policy=describe(mine), initial=None if p == 'greedy' else describe(others['initial']),
                             differences={a: free_difference(mine, c, n) for a, c in sorted(others.items())},
                             read_differences={a: free_difference(mine, c, n, read_value) for a, c in sorted(others.items())})
            if p in sampled and claimed:  # the plan's claims; any other is described, with no claim of its own
                every = free_claims(mine, controls, n)
                passes[p].update({k: v for k, v in (('claims', {c: every[c] for c in read}),
                                                    ('described', {c: unread(every[c]) for c in every if c not in read}))
                                  if v})
        deaths = {p: passes[p]['policy']['deaths_a_life'] for p in passes}
        verdict = dict(kind=kind, deaths_a_life=deaths)
        if plan:
            claims = [passes[p]['claims'] for p in sampled]
            words = {c: tri([k[c] for k in claims]) for c in read if c != 'no_rescue'}
            if 'no_rescue' in read:
                matched = [k['no_rescue']['matches'] for k in claims]
                words['no_rescue'] = 'match' if True in matched else 'no match' if matched == [False, False] else 'incomplete'
            needed = [('wait', controls.get('wait')), *((f'initial {p}', cells.get(('initial', seed, p))) for p in sampled),
                      ('greedy', cells.get(('policy', seed, 'greedy')))]
            gaps = [name for name, cell in needed if not complete(cell)]
            words = {c: 'incomplete' for c in words} if gaps else words  # the controls of every record, every time
            own = [a for c in read for a in FREE_CONTROLS[c] if not complete(controls.get(a))]  # each claim's own
            verdict = dict(kind=kind, **words, readings=free_readings(words), incomplete_controls=gaps + own,
                           deaths_a_life=deaths)
        seeds[seed] = dict(checkpoint=policies[seed]['policy'], initial=policies[seed]['initial'],
                           run=policies[seed]['run'], pass_seeds=policies[seed]['passes'], passes=passes, verdict=verdict)
    verdict = dict(kind=kind, plan=plan, why=why, seeds=sorted(seeds),
                   deaths_a_life={s: v['verdict']['deaths_a_life'] for s, v in seeds.items()})
    if plan:
        each = [s['verdict'] for s in seeds.values()]
        words = {c: 'pass' if all(v[c] == 'pass' for v in each) else 'fail' if any(v[c] == 'fail' for v in each)
                 else 'incomplete' for c in read if c != 'no_rescue'}
        if 'no_rescue' in read:
            words['no_rescue'] = 'match' if any(v['no_rescue'] == 'match' for v in each) else 'no match' if \
                all(v['no_rescue'] == 'no match' for v in each) else 'incomplete'
        verdict.update(words, readings=free_readings(words))
    result = dict(task=settings['task'], statement=FREE['statement'], settings=settings, rules=FREE,
                  outputs=[str(o) for o in outputs], evidence=evidence,
                  complete=dict(wait=complete(controls.get('wait')),
                                initial={s: {p: complete(cells.get(('initial', s, p))) for p in sampled} for s in seeds},
                                greedy={s: complete(cells.get(('policy', s, 'greedy'))) for s in seeds}),
                  controls={a: describe(c) for a, c in sorted(controls.items())}, seeds=seeds, verdict=verdict)
    text = json.dumps(result, indent=1, default=str)
    Path(path).write_text(text + '\n', encoding='utf-8')
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='evaluate one checkpoint (or WAIT) and write gate.json')
    run.add_argument('--checkpoint', type=Path)
    run.add_argument('--policy', choices=('final', 'wait'), help='default: final with --checkpoint')
    run.add_argument('--skills', nargs='+', choices=SKILLS, default=list(SKILLS))
    run.add_argument('--records', choices=tuple(GATE['records']), default='development')
    run.add_argument('--final', action='store_true', help='the final records are final evaluation data: say so')
    run.add_argument('--workers', type=int, default=4, help='simulator evaluation processes at once')
    run.add_argument('--timeout', type=int, default=4 * 3600, help='seconds per evaluation process')
    run.add_argument('--native', action='store_true', help='also the native check (scripts/stack.py world service)')
    run.add_argument('--schedule', choices=('serial', 'overlap'), default='serial',
                     help='serial (phase 1): the native passes after the simulator parts; overlap: beside them, one '
                          'stack at a time, so --workers simulator processes and the native check run at once')
    run.add_argument('--output', type=Path, required=True)
    trials = commands.add_parser('native-trials', help='(inside the stack) one native pass')
    trials.add_argument('--output', type=Path, required=True)
    trials.add_argument('--policy', choices=('final', 'wait'), required=True)
    trials.add_argument('--checkpoint', type=Path)
    trials.add_argument('--sampled', action='store_true')
    trials.add_argument('--seed', type=int, required=True)
    trials.add_argument('--skills', nargs='+', choices=SKILLS, required=True)
    trials.add_argument('--records', choices=tuple(GATE['records']), required=True)
    again = commands.add_parser('rules', help='gate.json again from an existing output')
    again.add_argument('output', type=Path)
    later = commands.add_parser('native', help='the native check for an existing output, then gate.json again')
    later.add_argument('output', type=Path)
    four = commands.add_parser('final4', help='one final on the final4 records (FINAL4), written as final4.json')
    four.add_argument('--checkpoint', type=Path, required=True)
    four.add_argument('--policy-seed', type=int, required=True, choices=FINAL4['policy_seeds'],
                      help='the final\'s training seed (its trial seeds)')
    four.add_argument('--records', choices=('final4', 'development'), required=True)
    four.add_argument('--final', action='store_true', help='the final4 records are final evaluation data: say so')
    four.add_argument('--layouts', type=int, help='development records only: the first LAYOUTS (default 200)')
    four.add_argument('--workers', type=int, default=3, help='passes at once')
    four.add_argument('--timeout', type=int, default=4 * 3600, help='seconds per pass')
    four.add_argument('--output', type=Path, required=True)
    free = commands.add_parser('free', help='free-play-v1: the statistics and the verdict from sim.hour free rows')
    free.add_argument('outputs', type=Path, nargs='+', help='sim.hour free outputs of one record set (a policy seed each, '
                      'the controls once)')
    free.add_argument('--output', type=Path, required=True, help='free.json')
    for p in (run, later):
        p.add_argument('--bun', type=Path, help='the bun binary (native check)')
        p.add_argument('--engine', action='append', default=[], metavar='NAME=CHECKOUT',
                       help='engine checkouts (default pinned=vendor/rs-sdk, arena=logs/vendor-arena/rs-sdk)')
        p.add_argument('--native-python', type=Path, default=ROOT / '.venv' / 'bin' / 'python',
                       help='the native environment\'s interpreter')
    args = parser.parse_args()
    if args.command == 'free':
        try:
            result = free_json(args.outputs, args.output)
        except ValueError as error:
            raise SystemExit(f'[free] {error}')
        print(json.dumps(dict(verdict=result['verdict'], complete=result['complete'], evidence=result['evidence']),
                         indent=1))
        return
    if args.command == 'native-trials':
        exits = native_trials(args.output, args.policy, args.checkpoint, args.sampled, args.seed, args.skills, args.records)
        print(json.dumps(exits))
        raise SystemExit(0 if all(v['exit'] == 0 for v in exits.values()) else 1)
    if args.command == 'final4':
        if (args.records == 'final4') != args.final:
            parser.error('the final4 records need --final, and --final needs the final4 records')
        if args.records == 'final4' and args.layouts is not None:
            parser.error('the final4 records are played whole: --layouts is for the development records')
        if args.output.exists():
            parser.error(f'{args.output} exists: final4 never writes over an output')
        if not args.checkpoint.is_file():
            parser.error(f'no checkpoint at {args.checkpoint}')
        if args.records == 'final4':
            verify_final4_records()
        args.output.mkdir(parents=True, exist_ok=False)
        done = execute(final4_jobs(args.output, args.checkpoint, args.policy_seed, args.records, args.layouts),
                       args.workers, args.output / 'logs', args.timeout)
        result = final4_json(args.output, args.checkpoint, args.policy_seed, args.records, args.layouts, dict(simulator=done))
        print(json.dumps(dict(verdict=result['verdict'], rules={r['pass_name']: dict(value=r['value'], valid=r['valid'])
                                                                  for r in result['rules']}), indent=1))
        failed = [k for k, v in done.items() if v['exit'] != 0]
        if failed:
            raise SystemExit(f'processes failed: {failed} (logs in {args.output / "logs"})')
        return
    engines = dict(e.split('=', 1) for e in getattr(args, 'engine', [])) or \
        dict(pinned=ROOT / 'vendor' / 'rs-sdk', arena=ROOT / 'logs' / 'vendor-arena' / 'rs-sdk')
    if args.command in ('rules', 'native'):
        stamp = read(args.output / 'gate.json')
        stale = stale_keys(stamp.get('gate'), json.loads(json.dumps(GATE)))
        if stale:
            raise SystemExit(f'{args.output / "gate.json"} was scored with another gate than this {GATE["version"]}; '
                             f'these keys differ: {", ".join(stale)}. Score it again only with the code of its own commit')
        checkpoint, timing = stamp['checkpoint'] and Path(stamp['checkpoint']['path']), stamp['timing']
        if args.command == 'native':
            if args.bun is None:
                parser.error('the native check needs --bun')
            t1 = time.perf_counter()
            timing['native'] = native(args.output, stamp['policy'], checkpoint, stamp['skills'], stamp['records']['name'],
                                      args.bun, engines, args.native_python)
            timing['native_seconds'] = round(time.perf_counter() - t1, 1)
        result = gate_json(args.output, stamp['skills'], stamp['policy'], checkpoint, stamp['records']['name'],
                           args.command == 'native' or stamp['native_records'] is not None, timing)
        print(json.dumps(result['verdict'], indent=1))
        failed = [k for k, v in timing.get('native', {}).items() if v['exit'] != 0] if args.command == 'native' else []
        if failed:
            raise SystemExit(f'native processes failed: {failed} (logs in {args.output / "logs"})')
        return
    policy = args.policy or ('final' if args.checkpoint else None)
    if policy is None or (policy == 'final') != (args.checkpoint is not None):
        parser.error('give --checkpoint (policy final) or --policy wait')
    if (GATE['records'][args.records]['split'] == 'final') != args.final:
        parser.error('the final records need --final, and --final needs the final records')
    served = GATE['records'][args.records].get('serves', SKILLS)
    if not set(args.skills) <= set(served):
        parser.error(f'the {args.records} records serve {" ".join(served)} only: pass --skills {" ".join(served)}')
    if args.native and args.bun is None:
        parser.error('--native needs --bun')
    args.output.mkdir(parents=True, exist_ok=False)
    found, merges = jobs(args.output, args.skills, policy, args.checkpoint, args.records)
    checks = lambda: native(args.output, policy, args.checkpoint, args.skills, args.records, args.bun, engines,
                            args.native_python)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as beside:
        early = beside.submit(timed, checks) if args.native and args.schedule == 'overlap' else None
        timing = dict(simulator=execute([(name, argv) for name, argv, _ in found], args.workers, args.output / 'logs',
                                        args.timeout))
        timing['simulator_seconds'] = round(time.perf_counter() - t0, 1)
        timing['merges'] = execute([(f'merge-{k}', argv) for k, argv in enumerate(merges)], args.workers,
                                   args.output / 'logs', 600)
        if args.native:
            timing['native'], timing['native_seconds'] = timed(checks) if early is None else early.result()
    if args.schedule != 'serial':
        timing.update(schedule=args.schedule, total_seconds=round(time.perf_counter() - t0, 1))
    result = gate_json(args.output, args.skills, policy, args.checkpoint, args.records, args.native, timing)
    print(json.dumps(dict(verdict=result['verdict'], timing={k: v for k, v in timing.items() if k.endswith('seconds')}), indent=1))
    failed = [k for part in ('simulator', 'merges', 'native') for k, v in timing.get(part, {}).items() if v['exit'] != 0]
    if failed:
        raise SystemExit(f'processes failed: {failed} (logs in {args.output / "logs"})')


if __name__ == '__main__':
    main()
