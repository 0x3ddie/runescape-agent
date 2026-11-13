"""hour-v1, continuous play: one persistent character with a supplied goal plays one game hour, 6,000
native ticks from arm, in one simulated world. Nothing resets at the task horizon: the world, the inventory, XP and
levels, hitpoints and the agent's interface memory carry through the hour, and the recurrent state has one episode
start, at arm. Death ends the hour and its tick is recorded. A step the simulator cannot model (sim.engine.Unsupported)
stops the trial: it is reported with its evidence and never scored. A diagnostic evaluation, not a gate.

Starts are the goal's training starts (sim/vec.py CpuBatch._episode), one layout per record from HOUR's split: wood a
supply-world-v1 layout as generated (sim/search.py; the `ample` variant, so every grove stands) with draw_supply's grove
and tile; fish and mine a compact layout with task.draw_start; melee a compact layout with sustained combat's npcs and
a draw_combat start of HOUR['combat_group']. Every world leaves out HOUR['exclusions'] (random events, clue and loot
drops), as sustained combat and supply declare in training: the readiness roll at every 500th tick would otherwise stop
about 40% of hours, and the short tasks never reach it. Two sampled passes and greedy play are separate runs, as are the
XP rates (25x the training rate, 1x the game's; World xp_rate, natively NODE_XPRATE).

The policy's task clock (budget_vec and task_vec's decision count, which training resets each episode) follows
HOUR['clock']: 'window' restarts both at every multiple of the goal's trained task horizon (HOUR['windows']), so they
stay in the range training showed; 'hour' counts the one hour down. No development layout has a bank, so the "bank"
response to a full inventory cannot occur on them; it is counted all the same.

Reported per trial: XP and levels per skill at arm and at the end, certified items produced (sim/audit.py: the goal
family, every family and anything else produced), certified kills of the goal's enemies, death and its tick, distinct
tiles stood on, inventory-full events and the response to each (drop, bank, other or stall; HOUR['full_response_ticks']),
decisions and decisions per 1,000 ticks, refusals, interface loops (A-B-A alternations of interface actions) and stalls
(sim/search.py SEARCH['stall'] windows without a goal outcome: an item of the goal family produced or a certified kill).

    .venv-sim/bin/python -m sim.hour run --checkpoint RUN/latest.zip --policies final wait --trials 8 --xp-rate 25 \
        --sampled --seed 606001 --output OUT [--part I N]
    .venv-sim/bin/python -m sim.hour merge OUT

native-hour-v1 plays the phase 1 finals for one native
game hour each, with fixed weights: one freshly booted world per hour from a scripts/stack.py world service, the wood goal
at Barbarian Village on NATIVE's records, hour-v1's window clock, one audit of the whole hour and certified production per
1,200-tick window from the audited trace. One pass per stack (B0: --b0):

    NODE_RANDOM_EVENTS=false PYTHONPATH=.:rl .venv/bin/python scripts/stack.py --bun BUN --output OUT/stack \
        --timeout-seconds S --lockstep --xp-rate 25 --worlds 0 --engine pinned=vendor/rs-sdk --blocks 1 -- \
        .venv/bin/python -m sim.hour native --output OUT/PASS --pass PASS [--b0]
    .venv/bin/python -m sim.hour native-summary OUT/PASS ...
"""
import argparse
import bisect
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from baseline import config
from baseline.resource_task import GOAL_FAMILIES, GOALS, ResourceEpisode
from baseline.telemetry import sha256
from obs import MAX_DECISIONS
from sim.agent import A_ACT0, A_HELD0, HELD_CONTRACT, family
from sim.audit import kill_ticks
from sim.combat import COMBAT, ControlActor, PolicyActor, combat_npcs, draw_combat, kit
from sim.engine import RETALIATE_CONTRACT, SKILLS, Unsupported, World
from sim.search import SUPPLY, measures, supply_layouts, supply_sites, draw_supply
from sim.spec import HORIZON_TICKS
from sim.task import FAMILIES, WORLD_START_TICK, GoalEpisode, draw_start, object_families
from sim.world import ORIGIN, LayoutTables, generate, split_seeds

HOUR = dict(
    version='hour-v1',
    ticks=6000,                    # native ticks from arm: one game hour
    clock='window',                # the policy's task clock (module docstring): 'window' or 'hour'
    windows=dict(wood=SUPPLY['horizon'], fish=HORIZON_TICKS, mine=HORIZON_TICKS, melee=COMBAT['horizon']),
    xp_rates=(25, 1),              # each a separate run; 25x is the training rate, 1x reported beside
    exclusions=COMBAT['exclusions'],
    split='development', first=0, key=19, trials=8,
    final=dict(first=9800, key=151),  # phase 1's final records: --split final's default
    final2=dict(first=6400, key=173, goals=('wood',)),  # the native-click study's: --records final2
    seeds=(606001, 909001),        # the sampled passes' bases: trial seed = seed + 1000 x goal index + record
    combat_group='healthy',        # sim/combat.py COMBAT['groups']: the melee hour's start
    decision_limit=6000,           # every decision advances at least 2 ticks; a trial that reaches it is not scored
    full_response_ticks=100,       # a drop or deposit this soon after the inventory fills is the response to it
)
POLICIES = ('final', 'initial', 'wait', 'random')
SKILLS_SHOWN = ('attack', 'strength', 'defence', 'hitpoints', 'woodcutting', 'fishing', 'mining')
# A decision row: tick from arm, action, ACT or HELD option, (reward: none), x, z (local), (unused), ended option's
# reason, refused, the engine's note, hitpoints after, free inventory slots after. Columns 4, 5, 7 and 8 are
# sim/search.py's, so its measures() and stalls() read these rows.
ROW_FIELDS = ('tick', 'action', 'option', 'reward', 'x', 'z', 'unused', 'ended', 'refused', 'note', 'hp', 'free')


class WindowClock:
    """The 'window' task clock: the budget vector and task_vec's decision count restart every `window` ticks from arm,
    so each window shows what a fresh episode of the trained horizon showed. hour-v1 and the native hour share it."""

    def __init__(self, window):
        self.window = window

    def arm(self, decision):
        """At arm: the observation's decision count (setup steps count, as in a fresh episode)."""
        self.arm_decision, self.index, self.offset = decision, 0, 0

    def __call__(self, ticks, decision):
        """(budget_vec, task_vec[0]) of an observation `ticks` from arm whose decision count is `decision`."""
        k = ticks // self.window
        if k != self.index:  # a new window shows the clock a fresh episode showed at arm
            self.index, self.offset = k, decision - self.arm_decision
        budget = np.asarray([((k + 1) * self.window - ticks) / self.window, self.window / 10000], np.float32)
        return budget, np.float32(min(max((decision - self.offset) / MAX_DECISIONS, 0.0), 1.0))


class HourEpisode(GoalEpisode):
    """GoalEpisode over the hour: armed as the goal's training start, ended by death or HOUR['ticks'] ticks from arm,
    paying nothing (the outcomes are the audit's at the end), its task clock shown under `clock`."""

    def __init__(self, world, cfg, goal, clock=HOUR['clock']):
        if clock not in ('window', 'hour'):
            raise ValueError(f'unknown clock {clock!r}')
        window = HOUR['windows'][goal]
        super().__init__(world, cfg, goal, horizon=window, scale=1.0, combat=goal == 'melee')
        self.window, self.clock, self.horizon = window, clock, HOUR['ticks']
        self.window_clock = WindowClock(window)

    def held(self):
        return 0.0

    def _arm(self, raw):
        self.window_clock.arm(int(raw.get('decision', 0)))
        return super()._arm(raw)

    def project(self):
        result = super().project()  # budget_vec over the hour: the 'hour' clock
        if self.clock == 'window':
            tick, _ = self.budget()
            result['budget_vec'], result['task_vec'][0] = self.window_clock(tick - self.start_tick,
                                                                            int(self.raw.get('decision', 0)))
        return result


def records(goal, trials, tables, split=HOUR['split'], first=HOUR['first'], key=HOUR['key']):
    """The hour's starts of `goal` (module docstring): record k on the split's layout k from index `first`, drawn with
    world slot GOALS.index(goal) and episode k. Returns (layouts, records)."""
    slot = GOALS.index(goal)
    if goal == 'wood':
        layouts, _ = supply_layouts(split, trials, tables, first)
        sites = [supply_sites(layout, tables) for layout in layouts]
        return layouts, [draw_supply(key, slot, k, layouts, sites, layout=k, variant='ample') for k in range(trials)]
    layouts = [generate(seed, tables) for seed in split_seeds(split, trials, first)]
    out = []
    for k in range(trials):
        start = draw_start(key, slot, k, layouts, layout=k)
        if goal == 'melee':
            start['combat'] = draw_combat(key, slot, k, layouts[k], start, tables, group=HOUR['combat_group'])
        out.append(start)
    return layouts, out


def hour_episode(goal, layout, record, tables, xp_rate, clock=HOUR['clock'], cfg=None):
    """A started HourEpisode of a record in a fresh world on its layout at `xp_rate`."""
    lt = LayoutTables(tables, layout)
    if goal == 'wood':
        lt.CACHE = SUPPLY['cache']
    extra = dict(npcs=combat_npcs(layout, tables, record['combat']['muggers'])) if goal == 'melee' else {}
    world = World(lt, seed=record['world_seed'], start_tick=WORLD_START_TICK, exclusions=HOUR['exclusions'],
                  xp_rate=xp_rate, **extra)
    task = HourEpisode(world, cfg or config.TASK_CFG, goal, clock)
    if goal == 'melee':
        c = record['combat']
        task.start(record['tile'], record['weapon'], record['levels'], kit=kit(record['weapon'], c['food']), hp=c['hp'],
                   wield=c['worn'])
    else:
        task.start(record['tile'], record['weapon'], record['levels'])
    return task


def skills(player):
    """{skill: (base level, XP)} of every skill (XP is stored x10 in Player.stats)."""
    return {name: (int(player.base_levels[k]), int(player.stats[k]) // 10) for k, name in enumerate(SKILLS)}


class FullInventory:
    """Inventory-full events at decision boundaries: each time the inventory has no free slot after a decision when it
    had one before (or at arm), and what freed the first slot (the first removal or deposit from the inventory among the
    world's events since): its response is that kind when it came within HOUR['full_response_ticks'] ticks ('drop' or
    'bank', else 'other'), 'stall' when nothing freed it in time, 'open' when the hour ended first."""

    def __init__(self, task):
        self.task, self.events, self.current = task, [], None
        self.observe()

    def observe(self):
        task = self.task
        free, tick = task.agent.player.inv.freespace(), task.world.tick
        if free == 0 and self.current is None:
            self.current = dict(tick=tick - task.start_tick, decision=task.decisions, seen=len(task.world.events))
        elif free > 0 and self.current is not None:
            events = task.world.events[self.current['seen']:]
            # an equipment swap moves an item out of the inventory and the worn one back in the same tick: it frees nothing
            back = {t for t, kind, data in events if kind == 'transfer' and data['dst'] == 'inv'}
            cause = None
            for t, kind, data in events:
                if kind == 'remove':
                    cause = (t, 'drop' if data['why'] == 'drop' else data['why'])
                elif kind == 'transfer' and data['src'] == 'inv' and not (data['dst'] == 'worn' and t in back):
                    cause = (t, 'bank' if data['dst'] == 'bank' else 'other')
                if cause:
                    break
            self.close(*(cause if cause else (tick, 'other')))

    def close(self, at, cause):
        start, task = self.current['tick'], self.task
        delay = at - task.start_tick - start
        response = ('drop' if cause == 'drop' else 'bank' if cause == 'bank' else 'other') \
            if delay <= HOUR['full_response_ticks'] else 'stall'
        self.events.append(dict(tick=start, decision=self.current['decision'], freed_by=cause, freed_after=delay,
                                response=response))
        self.current = None

    def finish(self, end):
        """At the hour's end: an event still open is a stall if it lasted longer than the response window."""
        if self.current is not None:
            lasted = end - self.current['tick']
            self.events.append(dict(tick=self.current['tick'], decision=self.current['decision'], freed_by=None,
                                    freed_after=None, response='stall' if lasted > HOUR['full_response_ticks'] else 'open'))
            self.current = None
        return self.events


def trial(task, actor, seed, limit=HOUR['decision_limit']):
    """One hour of `task` (an armed HourEpisode) with `actor` (sim/combat.py PolicyActor or ControlActor): (decision rows,
    final info or None at the decision limit, distinct tiles, FullInventory). An Unsupported step propagates."""
    actor.reset(seed)
    w, p = task.world, task.agent.player
    tiles = {(p.level, p.x, p.z)}
    cycle = w.cycle  # the episode's watcher; its close puts the world's own cycle back

    def counted():
        cycle()
        tiles.add((task.agent.player.level, task.agent.player.x, task.agent.player.z))
    w.cycle = counted
    full, obs, rows = FullInventory(task), task.projected, []
    for _ in range(limit):
        tick, raw = w.tick, task.raw
        action = actor.act(obs, task)
        kind = family(action)
        option = raw['actSel'][action - A_ACT0]['option'] if kind == 'act' and action - A_ACT0 < len(raw['actSel']) else \
            raw['heldSel'][action - A_HELD0]['option'] if kind == 'held' and action - A_HELD0 < len(raw['heldSel']) else None
        obs, _, done, info = task.step(action)
        ended = task.raw['feedback'].get('ended') if not done else None
        rows.append([tick - task.start_tick, action, option, 0.0, p.x - ORIGIN[0], p.z - ORIGIN[1], False,
                     ended and ended['reason'], bool(ended and ended['refused']), task.raw.get('actionNote'), p.hp,
                     p.inv.freespace()])
        full.observe()
        if done:
            return rows, info, tiles, full
    return rows, None, tiles, full


def outcome_ticks(task):
    """Ticks from arm of the goal's outcomes: each item of the goal family produced, or each certified kill (melee)."""
    events = task.world.events[task.event_start:]
    end = task.terminal['tick'] if task.terminal else task.world.tick
    if task.goal == 'melee':
        return [t - task.start_tick for t in kill_ticks([e for e in events if e[0] < end], task.agent.name, task.enemies)]
    return [t - task.start_tick for t, kind, data in events if kind == 'produce' and t < end and data['obj'] in task.family
            for _ in range(data['count'])]


def score(task, rows, info, tiles, full, before, tables):
    """A finished trial's record (module docstring)."""
    report, p = info['report'], task.agent.player
    ticks = info['episode']['native_ticks']
    family = object_families(tables)
    by_family, other = Counter(), Counter()
    for obj, count in report['produced_items'].items():
        if family[obj] >= 0:
            by_family[FAMILIES[family[obj]]] += count
        else:
            other[tables.obj_types[obj]['debugname']] += count
    after = skills(p)
    produced = outcome_ticks(task)
    gaps = np.diff([0, *produced, ticks]) if ticks else np.asarray([0])
    found = measures(rows, produced, None, len(task.agent.visited))
    events = full.finish(ticks)
    return dict(
        ticks=ticks, died=not report['survived'], death_tick=None if report['survived'] else ticks,
        decisions=info['episode']['l'], decisions_per_1000_ticks=1000 * info['episode']['l'] / max(1, ticks),
        skills={name: dict(level=[before[name][0], after[name][0]], xp=[before[name][1], after[name][1]])
                for name in SKILLS if name in SKILLS_SHOWN or after[name] != before[name]},
        produced=report['produced'], retained=report['retained'], produced_by_family=dict(by_family),
        produced_other=dict(other), kills=report['kills']['lower'], enemies=sorted(task.enemies),
        distinct_tiles=len(tiles), cells=len(task.agent.visited),
        full_events=len(events), full_responses=dict(Counter(e['response'] for e in events)), full=events,
        refusals=found['refused'], interface_actions=found['interface_actions'],
        interface_loops=found['interface_alternations'], stalls=found['stalls'], stalled_ticks=found['stalled_ticks'],
        longest_without_outcome=int(gaps.max()), revisits=found['revisits'], path_tiles=found['path_tiles'],
        families=found['families'], ended=found['ended'], hp_end=int(p.hp))


def run(output, *, goals=GOALS, trials=HOUR['trials'], xp_rate=HOUR['xp_rates'][0], policies=('final',), checkpoint=None,
        initial=None, sampled=False, seed=HOUR['seeds'][0], clock=HOUR['clock'], split=HOUR['split'], first=HOUR['first'], key=HOUR['key'],
        part=(0, 1), tables=None, final=False):
    """Every policy's hour on records `part[0]`, `part[0] + part[1]`, ... of each goal: results-{part}.json (the header
    and one row per trial) and trials-{part}.jsonl.gz (each trial's decision rows too). Seeds are paired: every policy
    plays record k of goal g with seed + 1000 g + k. The final split is final evaluation data: it needs final=True (and
    a `first` no earlier final evaluation used)."""
    from sim.data import Tables
    if xp_rate not in HOUR['xp_rates'] or split == 'train':
        raise ValueError(f'hour-v1 runs at {HOUR["xp_rates"]}x XP on held-out layouts')
    if (split == 'final') != final:
        raise ValueError('the final split needs final=True (--final), and final=True needs the final split')
    tables = tables or Tables.load()
    paths = dict(final=checkpoint, initial=initial)
    missing = [p for p in policies if p in paths and paths[p] is None]
    if missing:
        raise ValueError(f'policies {missing} need their checkpoint')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index, count = part
    header = dict(version=HOUR['version'], retaliate=RETALIATE_CONTRACT, held=HELD_CONTRACT, hour=HOUR, goals=list(goals), trials=trials, xp_rate=xp_rate, clock=clock,
                  split=split, first=first, key=key, sampled=sampled, seed=seed, part=[index, count],
                  policies=list(policies), row_fields=ROW_FIELDS,
                  checkpoints={p: dict(path=str(paths[p]), sha256=sha256(paths[p])) for p in policies if p in paths})
    rows, timing, t_start = [], Counter(), time.perf_counter()
    with gzip.open(output / f'trials-{index}.jsonl.gz', 'wt') as stream:
        for goal in goals:
            layouts, starts = records(goal, trials, tables, split, first, key)
            for policy in policies:
                actor = PolicyActor(paths[policy], deterministic=not sampled) if policy in paths else ControlActor(policy)
                for k in range(index, trials, count):
                    t0 = time.perf_counter()
                    trial_seed = seed + 1000 * GOALS.index(goal) + k
                    row = dict(goal=goal, policy=policy, record=k, seed=trial_seed, layout_seed=layouts[k].seed,
                               start={key_: v for key_, v in starts[k].items() if key_ != 'combat'},
                               combat=starts[k].get('combat'))
                    task = hour_episode(goal, layouts[k], starts[k], tables, xp_rate, clock)
                    before = skills(task.agent.player)
                    try:
                        decisions, info, tiles, full = trial(task, actor, trial_seed)
                    except Unsupported as error:
                        row.update(unsupported=str(error), tick=task.world.tick - task.start_tick, decision=task.decisions)
                        decisions = None
                    else:
                        if info is None:
                            row.update(decision_limit=True)
                        else:
                            row.update(score(task, decisions, info, tiles, full, before, tables))
                    row['seconds'] = time.perf_counter() - t0
                    timing[policy] += row['seconds']
                    rows.append(row)
                    stream.write(json.dumps(dict(row, decisions=decisions), default=str) + '\n')
    result = dict(header, timing_seconds=dict(timing), wall_seconds=time.perf_counter() - t_start, trials_played=rows)
    (output / f'results-{index}.json').write_text(json.dumps(result, indent=1, default=str) + '\n', encoding='utf-8')
    return result


SUMMED = ('produced', 'retained', 'kills', 'distinct_tiles', 'full_events', 'refusals', 'interface_loops', 'stalls',
          'stalled_ticks', 'decisions', 'ticks')


def summarize(rows):
    """One policy and goal's trials: counts of scored, unsupported and decision-limit trials, deaths and their ticks,
    means of SUMMED, decisions per 1,000 ticks over every scored tick, production by family, levels gained per skill
    and the inventory-full responses."""
    scored = [r for r in rows if 'ticks' in r]
    mean = lambda values: float(np.mean(values)) if values else None
    out = dict(trials=len(rows), scored=len(scored), unsupported=sum('unsupported' in r for r in rows),
               decision_limit=sum(bool(r.get('decision_limit')) for r in rows), deaths=sum(r['died'] for r in scored),
               death_ticks=sorted(r['death_tick'] for r in scored if r['died']),
               valid=len(scored) == len(rows))
    out.update({f'mean_{k}': mean([r[k] for r in scored]) for k in SUMMED})
    ticks = sum(r['ticks'] for r in scored)
    out['decisions_per_1000_ticks'] = 1000 * sum(r['decisions'] for r in scored) / ticks if ticks else None
    out['mean_produced_by_family'] = {f: mean([r['produced_by_family'].get(f, 0) for r in scored]) for f in FAMILIES}
    out['mean_levels_gained'] = {s: mean([r['skills'][s]['level'][1] - r['skills'][s]['level'][0] for r in scored])
                                 for s in SKILLS_SHOWN}
    out['full_responses'] = dict(sum((Counter(r['full_responses']) for r in scored), Counter()))
    return out


def merge(output):
    """results.json from every results-*.json of `output`: refuses parts of different runs, a missing or repeated part,
    or a missing or repeated trial; cells[policy][goal] = summarize(...)."""
    output = Path(output)
    parts = [json.loads(p.read_text()) for p in sorted(output.glob('results-*.json'))]
    if not parts:
        raise ValueError(f'no results-*.json in {output}')
    same = ('version', 'retaliate', 'held', 'goals', 'trials', 'xp_rate', 'clock', 'split', 'first', 'key', 'sampled', 'seed',
            'policies', 'checkpoints')
    first = parts[0]
    value = lambda part, k: part.get('held', 'held-v1') if k == 'held' else part[k]  # a part with no `held` is held-v1
    if any(value(p, k) != value(first, k) for p in parts for k in same):
        raise ValueError('parts of different runs')
    count = first['part'][1]
    if sorted(p['part'][0] for p in parts) != list(range(count)) or any(p['part'][1] != count for p in parts):
        raise ValueError(f'parts {[p["part"] for p in parts]} are not one complete split into {count}')
    rows = [r for p in parts for r in p['trials_played']]
    keys = Counter((r['goal'], r['policy'], r['record']) for r in rows)
    expected = {(g, p, k) for g in first['goals'] for p in first['policies'] for k in range(first['trials'])}
    if set(keys) != expected or max(keys.values()) != 1:
        raise ValueError('missing or repeated trials')
    cells = {policy: {goal: summarize([r for r in rows if r['policy'] == policy and r['goal'] == goal])
                      for goal in first['goals']} for policy in first['policies']}
    result = dict({k: v for k, v in first.items() if k not in ('trials_played', 'part', 'timing_seconds', 'wall_seconds')},
                  parts=count, seconds=sum(p['wall_seconds'] for p in parts), cells=cells,
                  trials_played=sorted(rows, key=lambda r: (r['goal'], r['policy'], r['record'])))
    (output / 'results.json').write_text(json.dumps(result, indent=1, default=str) + '\n', encoding='utf-8')
    return result


# ---- native-hour-v1: the native hour with fixed weights ----

NATIVE = dict(
    version='native-hour-v1',
    ticks=HOUR['ticks'],
    goal='wood',
    report_window=1200,            # reporting windows; the policy's clock restarts every HOUR['windows'][goal] ticks
    decision_limit=6000,           # run_trial's cap: a native decision takes at least 2 ticks, so the hour ends first
    engine='pinned',
    site='barbarian', start=(3096, 3432),
    record_seed=20261031, records=8, b0_record=8,
    records_sha256='b74acaaf5f886f9864bd92807d77bee864ac3c8b4893799b58db2db3494e5a1d',
    passes=dict(sampled1=(True, 3101001), sampled2=(True, 3202001), greedy=(False, 3101001)),  # trial seed = base + record
    seeds=(101, 202, 303, 404, 505),
    policies=('S_wood', 'SH', 'initial'),
    b0=dict(events=0.75 * 20000, read_seconds=2.5, release_seconds=2.5),
    near=64,                       # tiles from a drop-in point or test village: such steps are reported apart
    acquire_timeout=300.0,
    statistics=dict(resamples=10000, seed=17017),
)
ROOT = Path(__file__).resolve().parents[1]
# A native decision row: tick from arm of the observation it was made from, action, option and target name of an ACT or
# HELD row, x, z, plane, the ended option's reason, refused, the engine's note, hitpoints, free inventory slots and the
# woodcutting base level after it, and the tick of the observation after it. Columns 0, 1, 4, 5, 7 and 8 are hour-v1's,
# so sim/search.py's measures() reads these rows too.
NATIVE_ROW_FIELDS = ('tick', 'action', 'option', 'target', 'x', 'z', 'plane', 'ended', 'refused', 'note', 'hp', 'free',
                     'woodcutting', 'seen')
INVENTORY_SLOTS = 28


class NativeHourEpisode(ResourceEpisode):
    """A built native episode (baseline.world.make_episode) over the hour: armed for NATIVE['ticks'] ticks, its task clock
    shown as hour-v1's window clock. Its own settlement is logged and never read; the audit gives the outcomes."""

    def __init__(self, episode):
        super().__init__(episode.transport, episode.control, episode.cfg, NATIVE['ticks'], episode.scale, episode.record,
                         episode.prepare, goal=episode.goal)
        self.window_clock, self.armed_clock = WindowClock(HOUR['windows'][episode.goal]), None

    def _budget(self):
        result = super()._budget()
        decision = int(self.raw.get('decision', 0))
        if self.clock is not self.armed_clock:  # the first observation of a new arm
            self.armed_clock = self.clock
            self.window_clock.arm(decision)
        result['budget_vec'], result['task_vec'][0] = self.window_clock(self.clock.last_tick - self.clock.start, decision)
        return result


class Timed:
    """The resource control with each operation's wall time: (operation, start, seconds)."""

    def __init__(self, control):
        self.control, self.calls = control, []

    def __call__(self, operation, **kwargs):
        start = time.monotonic()
        try:
            return self.control(operation, **kwargs)
        finally:
            self.calls.append((operation, start, time.monotonic() - start))


class NativeLog:
    """One native hour's evidence as it is recorded: decision rows (NATIVE_ROW_FIELDS), the arm observation's summary,
    the clock's latest tick, the first tick each inventory item was seen and the trace the audit read."""

    def __init__(self):
        self.rows, self.trace, self.start, self.tick, self.raw, self.arm, self.pending = [], None, None, None, None, None, None
        self.items = {}  # item id: tick from arm of the first observation that showed it in the inventory

    @staticmethod
    def observed(raw):
        p, ended = raw['player'], raw['feedback'].get('ended')
        skills = {s['name']: s for s in raw.get('skills', [])}
        return [p['x'], p['z'], p['level'], ended and ended['reason'], bool(ended and ended['refused']), raw.get('actionNote'),
                p['hp'], INVENTORY_SLOTS - len(raw['inventory']), skills['Woodcutting']['baseLevel']]

    def __call__(self, kind, data):
        if kind == 'resource_control':
            response = data['response']
            if data['operation'] == 'arm':
                self.start = response['audit']['startTick']
            elif data['operation'] == 'budget':
                self.tick = response['budget']['observedTick']
            elif data['operation'] == 'read':
                self.trace = response['audit']
        elif kind == 'episode_start':
            self.raw = data['observation']
            skills = {s['name']: s for s in self.raw.get('skills', [])}
            self.arm = dict(free=INVENTORY_SLOTS - len(self.raw['inventory']), items=sorted({i['id'] for i in self.raw['inventory']}),
                            woodcutting=[skills['Woodcutting']['baseLevel'], skills['Woodcutting']['xp']])
        elif self.start is None:
            return  # setup before arm: the declared start's style step records a request and an observation too
        elif kind == 'requested':
            action, raw = data['action'], self.raw
            entry = raw['actSel'][action - A_ACT0] if family(action) == 'act' and action - A_ACT0 < len(raw['actSel']) else \
                raw['heldSel'][action - A_HELD0] if family(action) == 'held' and action - A_HELD0 < len(raw['heldSel']) else {}
            self.pending = [self.tick - self.start, action, entry.get('option'), entry.get('name')]
        elif kind == 'observed':
            self.raw = data['observation']
            self.rows.append(self.pending + self.observed(self.raw))
        elif kind == 'task_transition' and data['executed_action'] is not None:
            seen = data['native_tick'] - self.start  # the observation's tick, from the clock read after the step
            self.rows[-1].append(seen)
            for item in self.raw['inventory']:
                self.items.setdefault(item['id'], seen)


def native_records(count=NATIVE['records'] + 1):
    """Part B's start records, in their frozen order: B1 plays the first NATIVE['records'], B0 the
    next. Raises ValueError if B1's differ from their frozen sha256."""
    earlier = config.earlier_native_draws()
    check = [{k: r[k] for k in ('set_stats', 'seed_items', 'reset_seed')} for r in config.check_records('entrana')]
    x, z = NATIVE['start']
    out = [dict(start=f"{NATIVE['site']}-hour-{k}", spawnTele=f'0,{x >> 6},{z >> 6},{x & 63},{z & 63}', **draw)
           for k, draw in enumerate(config.study_draws(NATIVE['record_seed'], count, earlier + check))]
    digest = hashlib.sha256(json.dumps(out[:NATIVE['records']], sort_keys=True).encode()).hexdigest()
    if digest != NATIVE['records_sha256'] or any(r in earlier + check for r in out):
        raise ValueError('the native hour records differ from their frozen draw')
    return out


def native_fixture(record, goal=NATIVE['goal']):
    """A declared-start goal fixture of one record over the hour, as baseline/config.py's start-set fixtures are."""
    return dict(id=f"native-hour-{record['start']}", kind='goal', horizon=NATIVE['ticks'], capacity='declared', goal=goal,
                final=False, **record)


def window_counts(trace, width=NATIVE['report_window'], windows=NATIVE['ticks'] // NATIVE['report_window']):
    """Certified production per item family and reporting window, from the audited trace's production events (native-op
    with a production source; the item is the stack's second-last value, as baseline/evaluate.py counts completions):
    {base id: [count in window 1, ...]}. The hour's last tick belongs to the last window."""
    out = {}
    for event in trace['events']:
        tail = event.get('stackTail') or []
        if event.get('kind') == 'native-op' and event.get('productionSource') is True and len(tail) >= 2:
            k = min((event['tick'] - trace['startTick']) // width, windows - 1)
            out.setdefault(tail[-2], [0] * windows)[k] += 1
    return out


def production_ticks(trace, families):
    """Ticks from arm of every production event of the item families `families`."""
    return [e['tick'] - trace['startTick'] for e in trace['events'] if e.get('kind') == 'native-op'
            and e.get('productionSource') is True and len(e.get('stackTail') or []) >= 2 and e['stackTail'][-2] in families]


def native_full(rows, arm_free, end, window=HOUR['full_response_ticks']):
    """Inventory-full events at decision boundaries (as FullInventory): each answered by the first decision after which a
    slot is free, a drop, a deposit ('bank') or 'other', or a 'stall' when none came within `window` ticks; 'open' when the
    hour ended first."""
    events, current = [], dict(tick=0, decision=0) if arm_free == 0 else None
    for k, row in enumerate(rows):
        free, seen, kind = row[11], row[13], family(row[1])
        if free == 0 and current is None:
            current = dict(tick=seen, decision=k + 1)
        elif free > 0 and current is not None:
            delay = seen - current['tick']
            response = dict(drop='drop', deposit='bank').get(kind, 'other') if delay <= window else 'stall'
            events.append(dict(current, freed_by=kind, freed_after=delay, response=response))
            current = None
    if current is not None:
        lasted = end - current['tick']
        events.append(dict(current, freed_by=None, freed_after=None, response='stall' if lasted > window else 'open'))
    return events


def far_points():
    """{name: (x, z)} of every drop-in point and test village (baseline/config.py DROP_IN, Seers', Catherby)."""
    out = {}
    for name, place in {**config.DROP_IN, **{v: config.VILLAGES[v] for v in ('seers', 'catherby')}}.items():
        _, mx, mz, lx, lz = map(int, place['spawnTele'].split(','))
        out[name] = (mx * 64 + lx, mz * 64 + lz)
    return out


def native_score(log, trial, timed, width=NATIVE['report_window']):
    """A complete native hour's record. Raises EvidenceError if the windows do not sum to the
    audit's production per family, so the hour is incomplete."""
    from baseline.evidence import EvidenceError
    report, trace, rows = trial['report'], log.trace, log.rows
    windows = NATIVE['ticks'] // width
    counts = window_counts(trace, width, windows)
    audited = {int(base): f['produced'] for base, f in report['families'].items() if f['produced']}
    if {b: sum(c) for b, c in counts.items()} != audited or sum(audited.values()) != report['produced']:
        raise EvidenceError('window counts differ from the audit\'s production')
    wood = GOAL_FAMILIES['wood']
    logs = [sum(c[w] for b, c in counts.items() if b in wood) for w in range(windows)]
    produced = production_ticks(trace, wood)
    ticks = report['delivered_ticks']
    window = lambda t: min(t // width, windows - 1)  # window_counts' rule
    # the woodcutting level at each window's end: the last observation at or before it (the arm observation at tick 0)
    seen = [(0, log.arm['woodcutting'][0])] + [(r[13], r[12]) for r in rows]
    level_at = lambda t: [level for tick, level in seen if tick <= t][-1]
    levels = [level_at(min((w + 1) * width, ticks)) for w in range(windows)]
    chops = [(r[0], r[3]) for r in rows if r[2] == 'Chop down']
    chop_ticks, trees = [c[0] for c in chops], Counter()
    for t in produced:  # the kind of the latest chop requested before the production tick
        k = bisect.bisect_left(chop_ticks, t)
        trees[chops[k - 1][1] if k else 'unattributed'] += 1
    near = far_points()
    close = Counter(name for r in rows for name, (x, z) in near.items() if max(abs(r[4] - x), abs(r[5] - z)) <= NATIVE['near'])
    declared = {t[k] for t in trace['targets'] for k in ('base', 'note') if t[k] is not None}
    foreign = {str(i): t for i, t in sorted(log.items.items()) if i not in declared and i not in log.arm['items']}
    by_window = [[r for r in rows if window(r[0]) == w] for w in range(windows)]
    found = [measures(wr, [t for t in produced if window(t) == w], None, 0) for w, wr in enumerate(by_window)]
    total = measures(rows, produced, None, 0)
    full = native_full(rows, log.arm['free'], ticks)
    calls = {}
    for operation, _, seconds in timed.calls:
        calls.setdefault(operation, []).append(seconds)
    (_, read, read_seconds), = [c for c in timed.calls if c[0] == 'read']
    (_, release, release_seconds), = [c for c in timed.calls if c[0] == 'release']
    xs = [max(abs(r[4] - NATIVE['start'][0]), abs(r[5] - NATIVE['start'][1])) for r in rows]
    return dict(
        ticks=ticks, decisions=trial['decisions'], decisions_per_1000_ticks=1000 * trial['decisions'] / max(1, ticks),
        survived=report['survived'], death_tick=None if report['survived'] else ticks,
        attacker_at_end=log.raw.get('attacker'), hp_end=rows[-1][10] if rows else None,
        produced=report['produced'], retained=[report['retained_lower'], report['retained_upper']],
        families={str(b): dict(f) for b, f in report['families'].items()}, kills=report.get('kills', {}).get('lower'),
        window_counts={str(b): c for b, c in counts.items()}, logs=sum(logs), logs_by_window=logs,
        primary=[logs[w] - logs[0] for w in range(1, windows)], logs_by_tree=dict(trees),
        woodcutting=dict(level=[log.arm['woodcutting'][0], levels[-1]], xp_start=log.arm['woodcutting'][1],
                         level_at_window_end=levels,
                         levels_gained_by_window=[levels[w] - (levels[w - 1] if w else log.arm['woodcutting'][0])
                                                  for w in range(windows)]),
        full_events=len(full), full_responses=dict(Counter(e['response'] for e in full)), full=full,
        refusals=total['refused'], stalls=total['stalls'], stalled_ticks=total['stalled_ticks'],
        interface_loops=total['interface_alternations'],
        by_window=dict(decisions=[len(wr) for wr in by_window], refusals=[f['refused'] for f in found],
                       stalls=[f['stalls'] for f in found], interface_loops=[f['interface_alternations'] for f in found]),
        ended=total['ended'], families_of_actions=total['families'], path_tiles=total['path_tiles'],
        farthest_from_start=max(xs, default=0), near_steps=dict(close), foreign_items=foreign,
        capture_events=len(trace['events']),
        control_seconds={op: dict(count=len(v), total=sum(v), max=max(v)) for op, v in calls.items()},
        # each control call includes the status check the service makes at most once a second
        read_seconds=read_seconds, release_seconds=release_seconds, audit_seconds=release - read - read_seconds)


def git_state():
    """The code commit and whether tracked files differ from it."""
    git = lambda *args: subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=True).stdout
    return dict(commit=git('rev-parse', 'HEAD').strip(), dirty=bool(git('status', '--porcelain', '--untracked-files=no').strip()))


# Phase 1's frozen initializations: the file of each training seed and its tensor digest (sim/train.py policy_digest).
PHASE1_INIT = dict(init='logs/phase1/init/s{seed}.zip',
                   init_policy_sha256={'101': 'ea33d35a3240971f6b703eb299d3a981b270dc271f4da93259977be5aa98492c',
                                       '202': '48b621f6ef212e51cb2f248cbb336daed5358d1c75e38154c953dcdf034f6f7c',
                                       '303': 'f65295f1081391c809b66a33cdb29c4f938515950e6fa1c7ea4248aa4fad4159',
                                       '404': 'e56e9635946fd8dc995e5264636bc608a017235d74cf7ed2333ebfab23c13ce3',
                                       '505': 'ffe8ebcaa0963bf91d353b45e261a97d201dbad1cbf15a3611f88229d6d0f89b'})


def load_policy(name, seed):
    """(model, facts) of a Part B policy of `seed`: a final ('S_wood', 'SH') checked against its gate.json sha256, an
    initialization ('initial') against PHASE1_INIT's frozen tensor digest; both must read goal_vec and the name table."""
    from sb3_contrib import RecurrentPPO
    from baseline.policy import check_names
    from sim.train import policy_digest
    if name == 'initial':
        path = ROOT / PHASE1_INIT['init'].format(seed=seed)
        frozen = None
    else:
        gate = json.loads((ROOT / f'logs/phase1/s{seed}/eval/gate-{name}-final/gate.json').read_text())['checkpoint']
        path, frozen = ROOT / gate['path'], gate['sha256']
    digest = sha256(path)
    if frozen is not None and digest != frozen:
        raise ValueError(f'{path} differs from its gate.json sha256')
    model = RecurrentPPO.load(str(path), device='cpu')
    check_names(model.policy)
    tensors = policy_digest(model.policy)
    if name == 'initial' and tensors != PHASE1_INIT['init_policy_sha256'][str(seed)]:
        raise ValueError(f'{path} is not seed {seed}\'s frozen initialization')
    if 'goal_vec' not in model.observation_space.spaces:
        raise ValueError(f'{path} does not read the goal')
    return model, dict(path=str(path.relative_to(ROOT)), sha256=digest, policy_sha256=tensors)


def native_schedule(b0=False, seeds=NATIVE['seeds'], policies=NATIVE['policies'], records=range(NATIVE['records'])):
    """[(policy, seed, record)] of a pass: B0 plays WAIT and S_wood seed 101 on the B0 record; B1 plays WAIT on every
    record, then each seed's policies record by record."""
    if b0:
        return [('wait', None, NATIVE['b0_record']), ('S_wood', 101, NATIVE['b0_record'])]
    return [('wait', None, k) for k in records] + [(p, s, k) for s in seeds for k in records for p in policies]


def native_hour(worlds, join, actor, fixture, trial_seed, path, engine=NATIVE['engine']):
    """One native hour in a freshly booted world: its record, 'complete' or
    'incomplete' with the cause, tick and decisions. A failed capture, release or audit leaves the hour incomplete; its
    capture is retired and its world released. No world, or one of another build, raises: the pass stops before the
    hour. A failed world release is reported as `service_error`: the pass stops after the hour."""
    from baseline.evaluate import run_trial
    from baseline.telemetry import EpisodeRecorder
    from baseline.world import JsonlEvidence, make_episode
    world = worlds.acquire(engine, timeout=NATIVE['acquire_timeout'])
    if world['tree'] != config.ENGINE_TREES[engine]:
        worlds.release(world['id'])
        raise RuntimeError(f"world {world['id']} runs another engine build than {engine}")
    hour = dict(world={k: world.get(k) for k in ('id', 'engine', 'tree', 'block', 'boot_id')}, evidence=path.name)
    log, evidence, transport, task, started = NativeLog(), JsonlEvidence(path), None, None, time.monotonic()

    def write(kind, **data):
        evidence(kind, **data)
        log(kind, data)
    record = EpisodeRecorder(write)
    try:
        transport, service = join(world, record)
        timed = Timed(service)
        record('scheduled_episode', index=0, fixture=fixture, trial=fixture['start'], seed=trial_seed)
        task = NativeHourEpisode(make_episode(transport, timed, fixture, record))
        trial = run_trial(task, fixture, actor, trial_seed, record, NATIVE['decision_limit'])
        hour.update(native_score(log, trial, timed), status='complete')
    except Exception as error:
        hour.update(status='incomplete', cause=f'{type(error).__name__}: {error}', decisions=len(log.rows),
                    tick=None if log.tick is None or log.start is None else log.tick - log.start)
    finally:
        try:
            if task is not None:
                task.close()  # retires a capture still bound, then leaves the world
            elif transport is not None:
                transport.close()
        finally:
            evidence.close()
            hour['seconds'] = time.monotonic() - started
            try:
                worlds.release(world['id'])
            except Exception as error:
                hour['service_error'] = f'{type(error).__name__}: {error}'
    if hour['status'] == 'complete':
        hour.update(seconds_per_decision=hour['seconds'] / max(1, hour['decisions']),
                    seconds_per_tick=hour['seconds'] / max(1, hour['ticks']))
    return hour


def native_run(output, name, schedule, worlds=None, join=None, policies=load_policy):
    """Play `schedule` (native_schedule) as pass `name` of NATIVE['passes'] inside a scripts/stack.py world service:
    OUT/results.json (header and one row per hour, rewritten after each) and each hour's raw evidence in OUT/hours/. A
    world that fails to boot or exits stops the pass; the hours left are reported as not played, never replaced."""
    from baseline.evaluate import Actor
    from baseline.world import FreshWorlds, join as join_world
    sampled, base = NATIVE['passes'][name]
    if os.environ.get('NODE_RANDOM_EVENTS') != 'false':
        raise ValueError('launch the stack with NODE_RANDOM_EVENTS=false (the native hour plays without random events)')
    worlds, join = worlds or FreshWorlds(), join or join_world
    output = Path(output)
    (output / 'hours').mkdir(parents=True, exist_ok=False)
    records = native_records()
    result = dict(version=NATIVE['version'], native=NATIVE, hour_windows=HOUR['windows'], row_fields=NATIVE_ROW_FIELDS,
                  pass_name=name, sampled=sampled, base=base, xp_rate=config.XP_RATE,
                  random_events=os.environ['NODE_RANDOM_EVENTS'], code=git_state(), engines=worlds.identity().get('engines'),
                  schedule=[list(s) for s in schedule], policies={}, status='running', hours=[])

    def save():
        (output / 'results.tmp').write_text(json.dumps(result, indent=1, default=str) + '\n', encoding='utf-8')
        os.replace(output / 'results.tmp', output / 'results.json')
    loaded, started = {}, time.monotonic()
    try:
        for index, (policy, seed, k) in enumerate(schedule):
            if policy == 'wait':
                actor = Actor('wait')
            else:
                if (policy, seed) not in loaded:
                    loaded = {key: v for key, v in loaded.items() if key[1] == seed}  # one seed's policies in memory
                    model, facts = policies(policy, seed)
                    loaded[policy, seed] = model
                    result['policies'].setdefault(str(seed), {})[policy] = facts
                actor = Actor('checkpoint', loaded[policy, seed], deterministic=not sampled)
            trial_seed = base + k
            row = dict(index=index, policy=policy, seed=seed, record=k, trial_seed=trial_seed,
                       start=records[k]['start'])
            path = output / 'hours' / f'{index:03d}-{policy}-{seed or "none"}-r{k}.jsonl.gz'
            rest = [dict(index=i, policy=p, seed=s, record=r, status='not_played')
                    for i, (p, s, r) in enumerate(schedule) if i > index]
            try:
                row.update(native_hour(worlds, join, actor, native_fixture(records[k]), trial_seed, path))
            except Exception as error:  # no world for this hour: the world service failed and the pass stops
                result['hours'] += [dict(row, status='not_played', cause=f'{type(error).__name__}: {error}'), *rest]
                result['status'] = 'stopped'
                return result
            result['hours'].append(row)
            print(f"[native-hour {name}] {index + 1}/{len(schedule)} {policy} s{seed} r{k}: {row['status']}"
                  + (f" logs {row['logs']} {row['logs_by_window']} wc {row['woodcutting']['level']} decisions "
                     f"{row['decisions']} {row['seconds']:.0f}s" if row['status'] == 'complete' else f" {row.get('cause')}"),
                  flush=True)
            if 'service_error' in row:
                result['hours'] += rest
                result['status'] = 'stopped'
                return result
            save()
        result['status'] = 'completed'
        return result
    finally:
        result['seconds'] = time.monotonic() - started
        save()


def bootstrap(values, resamples=NATIVE['statistics']['resamples'], seed=NATIVE['statistics']['seed']):
    """Mean and 95% percentile bootstrap interval of per-record values (records resampled)."""
    values = np.asarray(values, float)
    if not len(values):
        return None
    draws = np.random.default_rng(seed).integers(0, len(values), (resamples, len(values)))
    means = values[draws].mean(axis=1)
    return dict(n=len(values), mean=float(values.mean()), low=float(np.percentile(means, 2.5)),
                high=float(np.percentile(means, 97.5)))


def native_summary(outputs):
    """Per pass, seed and policy: hours by status, the primary per window (windows 2-5 minus 1,
    each hour its own control) with its bootstrap interval, logs, levels, deaths and the reported fields; and per seed and
    pass the paired difference in hour logs from WAIT on the same records."""
    out = {}
    for output in outputs:
        result = json.loads((Path(output) / 'results.json').read_text())
        hours = result['hours']
        cells = {}
        for h in hours:
            cells.setdefault(('wait' if h['policy'] == 'wait' else f"{h['policy']} s{h['seed']}"), []).append(h)
        wait = {h['record']: h['logs'] for h in hours if h['policy'] == 'wait' and h['status'] == 'complete'}
        summary = {}
        for cell, rows in cells.items():
            done = [h for h in rows if h['status'] == 'complete']
            mean = lambda key: float(np.mean([h[key] for h in done])) if done else None
            windows = len(done[0]['primary']) if done else 0
            summary[cell] = dict(
                hours=len(rows), complete=len(done), incomplete=[(h['record'], h.get('cause')) for h in rows
                                                                 if h['status'] == 'incomplete'],
                not_played=sum(h['status'] == 'not_played' for h in rows),
                primary=[bootstrap([h['primary'][w] for h in done]) for w in range(windows)],
                logs=bootstrap([h['logs'] for h in done]),
                logs_by_window=np.mean([h['logs_by_window'] for h in done], axis=0).tolist() if done else None,
                minus_wait=bootstrap([h['logs'] - wait[h['record']] for h in done if h['record'] in wait]),
                woodcutting_start=[h['woodcutting']['level'][0] for h in done],
                woodcutting_end=[h['woodcutting']['level'][1] for h in done],
                levels_gained_by_window=np.mean([h['woodcutting']['levels_gained_by_window'] for h in done],
                                                axis=0).tolist() if done else None,
                deaths=sum(not h['survived'] for h in done), death_ticks=sorted(h['death_tick'] for h in done if not h['survived']),
                logs_by_tree=dict(sum((Counter(h['logs_by_tree']) for h in done), Counter())),
                full_responses=dict(sum((Counter(h['full_responses']) for h in done), Counter())),
                **{f'mean_{k}': mean(k) for k in ('decisions', 'ticks', 'refusals', 'stalls', 'interface_loops',
                                                  'decisions_per_1000_ticks', 'seconds')},
                by_window={k: np.mean([h['by_window'][k] for h in done], axis=0).tolist() if done else None
                           for k in ('decisions', 'refusals', 'stalls', 'interface_loops')},
                near_steps=dict(sum((Counter(h['near_steps']) for h in done), Counter())),
                foreign_items=sum(bool(h['foreign_items']) for h in done),
                max_capture_events=max((h['capture_events'] for h in done), default=None),
                max_read_seconds=max((h['read_seconds'] for h in done), default=None),
                max_release_seconds=max((h['release_seconds'] for h in done), default=None))
        out[result['pass_name']] = dict(status=result['status'], seconds=result.get('seconds'), cells=summary)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    play = commands.add_parser('run', help='play hours of each goal and write one part')
    play.add_argument('--checkpoint', type=Path, help='the policy `final` plays')
    play.add_argument('--initial', type=Path, help='the policy `initial` plays (the run\'s saved initialization)')
    play.add_argument('--policies', nargs='+', default=['final'], choices=POLICIES)
    play.add_argument('--goals', nargs='+', default=list(GOALS), choices=GOALS)
    play.add_argument('--trials', type=int, default=HOUR['trials'], help='records per goal')
    play.add_argument('--xp-rate', type=int, default=HOUR['xp_rates'][0], choices=HOUR['xp_rates'])
    play.add_argument('--clock', default=HOUR['clock'], choices=('window', 'hour'))
    play.add_argument('--split', default=HOUR['split'], choices=('development', 'final'))
    play.add_argument('--final', action='store_true', help='the final split is final evaluation data: say so')
    play.add_argument('--records', default='final', choices=('final', 'final2'),
                      help="with --split final, the record set: phase 1's (HOUR['final']) or the native-click study's "
                           "wood records (HOUR['final2'])")
    play.add_argument('--first', type=int, help="the split's first seed index (default: HOUR's, or HOUR['final']'s with --split final)")
    play.add_argument('--key', type=int, help="the start key (default: HOUR's, or HOUR['final']'s with --split final)")
    play.add_argument('--sampled', action='store_true')
    play.add_argument('--seed', type=int, default=HOUR['seeds'][0], help='the sampled pass\'s base (HOUR seeds)')
    play.add_argument('--part', type=int, nargs=2, default=(0, 1), metavar=('I', 'N'))
    play.add_argument('--output', type=Path, required=True)
    join = commands.add_parser('merge', help='merge the parts of one run into results.json')
    join.add_argument('output', type=Path)
    native = commands.add_parser('native', help='Part B: one pass of native hours inside a scripts/stack.py world service')
    native.add_argument('--output', type=Path, required=True)
    native.add_argument('--pass', dest='pass_name', required=True, choices=list(NATIVE['passes']))
    native.add_argument('--b0', action='store_true', help='the B0 rehearsal: WAIT and S_wood seed 101 on the B0 record')
    native.add_argument('--seeds', type=int, nargs='+', default=list(NATIVE['seeds']), choices=NATIVE['seeds'])
    native.add_argument('--policies', nargs='+', default=list(NATIVE['policies']), choices=NATIVE['policies'])
    summary = commands.add_parser('native-summary', help="summarize native passes' results.json")
    summary.add_argument('outputs', type=Path, nargs='+')
    args = parser.parse_args()
    if args.command == 'merge':
        result = merge(args.output)
        print(json.dumps(result['cells'], indent=1))
        return
    if args.command == 'native':
        result = native_run(args.output, args.pass_name, native_schedule(args.b0, tuple(args.seeds), tuple(args.policies)))
        print(f"[native-hour {args.pass_name}] {result['status']}: {len(result['hours'])} hours in {result['seconds']:.0f} s")
        raise SystemExit(0 if result['status'] == 'completed' else 1)
    if args.command == 'native-summary':
        print(json.dumps(native_summary(args.outputs), indent=1))
        return
    if args.records == 'final2' and (args.split != 'final' or not set(args.goals) <= set(HOUR['final2']['goals'])):
        parser.error(f"the final2 records are final-split records of {' '.join(HOUR['final2']['goals'])} only")
    records = HOUR[args.records] if args.split == 'final' else HOUR
    result = run(args.output, goals=tuple(args.goals), trials=args.trials, xp_rate=args.xp_rate, policies=tuple(args.policies),
                 checkpoint=args.checkpoint, initial=args.initial, sampled=args.sampled, seed=args.seed, clock=args.clock,
                 split=args.split, first=records['first'] if args.first is None else args.first,
                 key=records['key'] if args.key is None else args.key, part=tuple(args.part), final=args.final)
    print(f'[hour] part {args.part[0]}/{args.part[1]}: {len(result["trials_played"])} trials in {result["wall_seconds"]:.1f} s')


if __name__ == '__main__':
    main()
