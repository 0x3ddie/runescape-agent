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

native-hour-v1 (the combined-skill hour, Part B, frozen 31 October 2026) keeps only its start records here
(native_records); its native replay is not in this tree.

free-play-v1's evaluation plays one record set of the named world (--world:
village-world-v1 Stage A or village-world-v2, sim/combat.py VILLAGES; no default) over that world's archive (sim/free.py
world_tables), record r the split's layout first + r of the world's layouts with the draw_village start (key, slot 0,
episode r) in a fresh world and one life of --ticks (sim/free.py Life, the soak's record), with a read of the same life
at --read-ticks (FreeRead). A policy that
sim.train train --task free-play-v1 saved (load_free_policy: obs-v6 only) plays sampled in two passes with their own
seeds and greedy; its saved initialization plays sampled in both passes; each control of sim/free.py plays once with the
control seed, shared by every pass and every policy seed. One JSONL row a life (free_life) and a manifest; parts run as
separate processes, one process a part (a lock on its lives file), and a part resumes where it stopped. The checkpoint
is its training run's latest.zip or one of its checkpoints/, beside the run's initial.zip and summary.json (policy_run):
the run's seed, status and batch in effect go in the manifest, and the run must have trained in the evaluation's world
over that world's archive (trained_world: the batch's recorded world; a summary written before the batch recorded one
is village-world-v1's only when it trained over that world's archive).
sim.gate free reads these outputs alone, and no other file of its checkout: the rows for the statistics, and the
manifests' settings and run records against its frozen plans for the verdict. The muggers, the XP rate, the record
range and key, the seeds and the rotation's period have no default: nothing is frozen. --plan NAME refuses an output that plan
NAME would not read before any life is played.

--change NAME plays a changed world of the scorecard (sim/changed.py; adopted as
descriptions): only the records whose base holds the change's fact, each built by sim/changed.py life from the base
record, with the base row's seeds (the same pass seeds, policy seed, control seed and record), so that the base
evaluation's row of the same actor, pass and record is the base half of each pair. The policy plays the sampled passes
of --change-passes (pass 1 alone by default), with no greedy pass and no initialization; the controls are those the
change's reads need (its candidate scripts, random legal play and WAIT: by default all of them, never another). Each row
carries the change's name. village-world-v2 without muggers only, and never with --plan: a changed world is a
description, and sim.gate free refuses its rows (sim/gate.py free_outputs). It is read beside the base
evaluation.

    .venv-sim/bin/python -m sim.hour free --output OUT --world village-world-v1 --first 4000 --records 64 --key 223 \
        --muggers 0 --xp-rate 1 \
        --checkpoint RUN/latest.zip --initial RUN/initial.zip --policy-seed 101 --seeds S1 S2 \
        --controls wait random wood fish mine melee chickens once eat rotation --period P --control-seed C --part I N
    .venv-sim/bin/python -m sim.hour free --change only-wood --output OUT --world village-world-v2 ... (as the base
        evaluation, without --plan and --controls)
"""
import argparse
from collections import Counter
import functools
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from baseline import config
from baseline.resource_task import GOALS
from baseline.telemetry import sha256
from obs import MAX_DECISIONS
from sim.agent import A_ACT0, A_HELD0, HELD_CONTRACT, family
from sim.audit import kill_ticks
from sim.combat import COMBAT, VILLAGE, VILLAGES, ControlActor, PolicyActor, combat_npcs, draw_combat, kit
from sim.engine import RETALIATE_CONTRACT, SKILLS, Unsupported, World
from sim.search import SUPPLY, measures, supply_layouts, supply_sites, draw_supply
from sim.spec import HORIZON_TICKS
from sim.task import FAMILIES, FREE_PLAY, WORLD_START_TICK, GoalEpisode, draw_start, free_play_cfg, object_families
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


# ---- native-hour-v1's start records (the combined-skill hour, Part B, frozen 31 October) ---------------------------
# sim/gate.py's final2 draw is disjoint from them. Part B's native replay (the commands native and native-summary, with
# its episode, log, scores and summary) is not in this tree.
NATIVE = dict(
    site='barbarian', start=(3096, 3432),
    record_seed=20261031, records=8,
    records_sha256='b74acaaf5f886f9864bd92807d77bee864ac3c8b4893799b58db2db3494e5a1d',
)


def native_records(count=NATIVE['records'] + 1):
    """Part B's start records, in their frozen order: B1's first NATIVE['records'], then B0's one.
    Raises ValueError if B1's differ from their frozen sha256."""
    earlier = config.earlier_native_draws()
    check = [{k: r[k] for k in ('set_stats', 'seed_items', 'reset_seed')} for r in config.check_records('entrana')]
    x, z = NATIVE['start']
    out = [dict(start=f"{NATIVE['site']}-hour-{k}", spawnTele=f'0,{x >> 6},{z >> 6},{x & 63},{z & 63}', **draw)
           for k, draw in enumerate(config.study_draws(NATIVE['record_seed'], count, earlier + check))]
    digest = hashlib.sha256(json.dumps(out[:NATIVE['records']], sort_keys=True).encode()).hexdigest()
    if digest != NATIVE['records_sha256'] or any(r in earlier + check for r in out):
        raise ValueError('the native hour records differ from their frozen draw')
    return out


# ---- free-play-v1's evaluation in the Python reference ---------------------------------------------------------------
FREE = dict(
    task=FREE_PLAY['version'],
    rows=2,                        # the row form's revision (free_life); 2: a policy life's seed is a 64-bit word
    ticks=36000,                   # a life: the trial's length
    read=6000,                     # the read beside it: the same life at 6,000 ticks from arm (FreeRead)
    passes=('sampled1', 'sampled2', 'greedy'),
    torch_threads=1,               # one intra-op thread a process, so a row does not depend on how many processes play
    # The training run's facts that sim.gate free's plans pin (policy_run copies them from summary.json): its decisions
    # and worlds, its learner settings as summary.json's 'learner' holds them, its batch's muggers and life-length
    # distribution, its layout pool's sha256 and the sha256 of the tables it trained on (summary 'tables': the Python
    # reference's archive, or on the C engine the archive its pack's recorded input was held equal to; None on the fake
    # environment and in a C-engine summary written before 'tables' held that archive), and whether it made the
    # --validate checks (summary 'validate'; None for a summary written before that key).
    run_facts=('decisions', 'worlds', 'n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gae_lambda', 'clip_range',
               'ent_coef', 'vf_coef', 'max_grad_norm', 'lstm_hidden_size', 'half_life_ticks', 'value_norm', 'muggers',
               'life', 'pool', 'tables', 'validate'),
    # The engine facts beside them (engine_facts): the engine the run collected on (summary 'engine': 'cpu', the Python
    # reference, or 'native', the C engine) and its pack's sha256 (None on the Python reference). A plan's run pins them
    # beside the run facts, or holds the run facts alone and pins the Python reference (reference_engine): the pilot's
    # plan, frozen before the C engine could train free play, is of that form.
    engine_facts=('engine', 'pack'),
    reference_engine=dict(engine='cpu', pack=None),
    # The world of a record written before the world was a recorded setting of a run and a plan. A plan that names none
    # (sim/gate.py free_plan_world; the pilot's) is this world's, since none other existed when such a plan was frozen.
    # A training summary whose batch in effect records no world (trained_world) is this world's only when it trained
    # over this world's archive: earlier code trained village-world-v2 pools without recording the world. A record
    # that names a world is held to it.
    first_world=VILLAGE['version'],
    # That archive, pinned (trained_world): the sha256 of sim/build.py DATA at the pilot's freeze, which the frozen pilot
    # plan pins as its tables (sim/gate.py FREE['plans']['pilot']), so that reading an old record never depends on this
    # checkout's files.
    first_world_archive='307b77a7a58327782bc25654127048968d12c9e88ef1a74c221ca60b3d6c5771',
)
POLICY_ACTORS = ('policy', 'initial')


@functools.lru_cache(maxsize=None)
def world_archive(world):
    """The sha256 of the archive `world` plays over (sim/free.py world_tables), hashed once a process; None when this
    checkout has no such file."""
    from sim import free
    path = free.world_tables(world)
    return sha256(path) if path.exists() else None


def trained_world(batch, tables):
    """The world a free-play training run played, from its batch in effect (summary.json free_play.batch_in_effect,
    sim/workers.py BATCH_SETTINGS) and the sha256 of the archive it trained on (`tables`: summary.json's tables, or a
    run record's facts; None in an older C-engine summary). A batch that records its `world` is held to it as it is
    (None included). A batch that records none was written before the batch recorded its world: it is
    FREE['first_world']'s only when it trained over that world's archive (FREE['first_world_archive'], a pinned value,
    not this checkout's file), or records no archive (an older C-engine run, which could play that world only), as the
    pilot's run of seed 77 did. Otherwise None: a record that names no world, which no evaluation and no plan reads."""
    if 'world' in batch:
        return batch['world']
    return FREE['first_world'] if tables is None or tables == FREE['first_world_archive'] else None


def fingerprint_beside(path):
    """The name table fingerprint written beside a published policy (scripts/export_policy.py: STEM.names.json beside
    STEM.zip), or None when there is none."""
    found = Path(path).with_name(Path(path).stem + '.names.json')
    return json.loads(found.read_text()) if found.exists() else None


def load_free_policy(path, contract=None):
    """(model, facts) of a free-play-v1 checkpoint (sim.train train --task free-play-v1: latest.zip, initial.zip or one of
    checkpoints/), for sampled and greedy play. The standard RecurrentPPO zip is loaded into the network its own saved
    settings build (obs-v6's table, PopArt's value head with its statistics or a plain one, both LSTMs), then checked as
    saved (baseline/policy.py check_names: obs-v6, the act, mask and held contract, the name table) and its observation
    space against obs-v6's table, before any step is played. A checkpoint of another observation version, one that reads
    a goal or one of another table is refused: its weights are never played under another contract. The policy is put
    in evaluation mode, as collection puts it (sim/learn.py). contract: the contract it must record (baseline/policy.py
    run_contract; item-v2's CONTRACT by default; a real-engine checkpoint's own version's, sim/watch.py load_actor),
    whose item contract names the observation version and table (sim/workers.py observation_table: obs-v7 under item-v4,
    free-play-v11's; obs-v6 otherwise). A published copy (scripts/export_policy.py) holds no name vectors and carries its
    table's fingerprint beside it (fingerprint_beside): a table rebuilt from other game files plays it when the
    fingerprint holds; its saved mark (name_vectors 'removed') has its tables filled from the table in use, which must
    all be zero. facts['names'] says how the table was taken (check_names) and where the vectors come from."""
    from sb3_contrib import RecurrentPPO
    from baseline.policy import CONTRACT, NameEmbedding, check_names, load_name_vectors
    from names import name_table
    from item_contract import ITEM_V2
    from sim.train import policy_digest
    from sim.workers import observation_table
    contract = contract or CONTRACT
    path, (observation, shapes) = Path(path), observation_table(contract.get('items', ITEM_V2))
    model = RecurrentPPO.load(str(path), device='cpu')
    model.policy.set_training_mode(False)
    names = check_names(model.policy, observation=observation, contract=contract, fingerprint=fingerprint_beside(path))
    if getattr(model, 'name_vectors', None) == 'removed':  # a public copy (scripts/export_policy.py)
        if any(bool(m.table.any()) for m in model.policy.modules() if isinstance(m, NameEmbedding)):
            raise ValueError(f'{path} is marked a public copy but holds name vectors')
        load_name_vectors(model.policy, name_table().vectors)
        names = dict(names, vectors='the table in use (a public copy holds none)')
    else:
        names = dict(names, vectors="the checkpoint's own")
    spaces = {k: tuple(s.shape) for k, s in model.observation_space.spaces.items()}
    if spaces != {k: tuple(v) for k, v in shapes.items()}:
        raise ValueError(f'{path} reads another observation table than {observation}\'s')
    return model, dict(path=str(path), sha256=sha256(path), policy_sha256=policy_digest(model.policy),
                       observation=observation, contract=model.policy.contract, value_norm=model.policy.value_norm,
                       free_play=getattr(model, 'free_play', None), half_life_ticks=getattr(model, 'half_life_ticks', None),
                       lstm=model.policy.lstm_actor.hidden_size, names=names)


class FreePolicyActor(PolicyActor):
    """A loaded free-play policy (load_free_policy) as sim/combat.py PolicyActor plays it: MaskedRecurrentPolicy.forward
    with both recurrent states, zeroed at the life's start (its one episode start), sampled with torch's generator seeded
    by the life's seed, or greedy. It refuses an observation whose arrays are not the policy's own, where PolicyActor
    would fill a missing one with zeros."""

    def __init__(self, model, deterministic):
        import torch
        self.torch, self.model, self.deterministic = torch, model, deterministic
        self.keys = set(model.observation_space.spaces)

    def act(self, observation, task=None, trace=False):
        if set(observation) != self.keys:
            raise ValueError(f'an observation with {sorted(set(observation) ^ self.keys)} unlike the policy\'s own arrays')
        return super().act(observation, task, trace)


def trial_seed(base, policy_seed, record):
    """A policy life's torch seed: one 64-bit word of SeedSequence([pass seed, policy seed, record]). Two lives share a seed
    only when two words collide (about 10^-10 for 40,960 lives); sim.gate free refuses a record whose sampled passes
    share one."""
    return int(np.random.SeedSequence([base, policy_seed, record]).generate_state(1, np.uint64)[0])


class FreeRead:
    """The life's levels in S `ticks` from arm, read at the end of the world cycle in which a life of `ticks` would end
    (sim/task.py _watch: a death, or the first cycle at or after its end). obs-v6 shows neither a life's length nor its
    time, so the read equals a life of `ticks` played by the same actor with the same seed; None when the life stopped
    at an unsupported site first."""

    def __init__(self, task, ticks):
        self.task, self.ticks, self.read = task, ticks, None
        world, cycle = task.world, task.world.cycle  # the episode's watcher; its close puts the world's own cycle back

        def watched():
            cycle()
            if self.read is None and (task.terminal is not None or world.tick >= task.start_tick + ticks):
                self.read = self.snapshot()
        world.cycle = watched

    def snapshot(self):
        from sim.audit import fractional_level
        t, p = self.task, self.task.agent.player
        gain = {name: p.base_levels[s] - t.arm_levels[s] for name, s in zip(t.skill_names, t.skills)}
        start, now = [fractional_level(t.arm_stats[s]) for s in t.skills], [fractional_level(p.stats[s]) for s in t.skills]
        return dict(ticks=self.ticks, tick=t.world.tick - t.start_tick,
                    died=t.terminal is not None and t.terminal['reason'] == 'death',
                    gain={k: v for k, v in gain.items() if v}, gain_s=sum(gain.values()),
                    fractional=sum(b - a for a, b in zip(start, now)))


def free_reads(task):
    """The unpaid reads of a played life from its events (none is paid, none has a bar):
    production by family, the agent's kills by npc name, each eat as [tick from arm, hitpoints before it, hitpoints it
    healed] (0 healed: an eat at full hitpoints, which writes no heal event), and the XP events in runs of one activity,
    [activity, first tick, last tick, events]. An XP event's activity is its cause's: 'fight:<npc name>' for a hit
    (chicken, goblin), else its skill (woodcutting, mining, fishing; cooking, prayer or firemaking outside S)."""
    from sim.engine import SKILLS
    w = task.world
    name = lambda npc_type: w.t.npc_types[npc_type]['name'].lower()
    npcs, events = {n.nid: name(n.base) for n in w.npcs}, w.events[task.event_start:]
    kills = Counter(name(d['type']) for _, kind, d in events if kind == 'npc_defeated' and d['source'] == task.agent.name)
    hp, eats, runs = task.arm_hp, [], []
    for t, kind, d in events:
        tick = t - task.start_tick
        if kind == 'agent_hp':
            if d['cause'] == 'eat' and eats and eats[-1][0] == tick and eats[-1][2] == 0:
                eats[-1][1:] = [d['before'], d['after'] - d['before']]
            hp = d['after']
        elif kind == 'remove' and d['why'] == 'eat':
            eats.append([tick, hp, 0])
        elif kind == 'xp':
            activity = f"fight:{npcs[d['npc']]}" if d.get('cause') == 'hit' else SKILLS[d['stat']]
            if runs and runs[-1][0] == activity:
                runs[-1][2:] = [tick, runs[-1][3] + 1]
            else:
                runs.append([activity, tick, tick, 1])
    return dict(made={f: int(c) for f, c in zip(FAMILIES, task.made) if c}, kills=dict(kills), eat_log=eats,
                eats_full=sum(e[2] == 0 for e in eats), activity=runs)


def free_jobs(records, policy, controls, change=None):
    """Every life of a run as (actor, pass, record), record by record: the policy in each pass, its initialization in each
    sampled pass, then each control, played once with the control seed and shared by every pass. `records`: a count, or
    with `change` (a manifest's change: its records and the policy's passes) none, the change's records alone and the
    policy in its passes only."""
    sampled = [p for p in FREE['passes'] if p != 'greedy']
    policy_jobs = [('policy', p) for p in change['passes']] if change else \
        [('policy', p) for p in FREE['passes']] + [('initial', p) for p in sampled]
    per = (policy_jobs if policy else []) + [(c, None) for c in controls]
    return [(name, p, r) for r in (change['records'] if change else range(records)) for name, p in per]


def free_life(job, run, tables, models):
    """The row of one life (the soak's row, sim/free.py Life.row, and what identifies the actor, its pass and seed, the
    6,000-tick read and the unpaid reads): `job` is (actor, pass, record), `run` the manifest's settings, `models` the
    loaded policies by actor. A policy plays with trial_seed(pass seed, policy seed, record), a control with the soak's
    actor seed [control seed, record]."""
    from sim import free
    name, pass_name, record = job
    t0 = time.perf_counter()
    change = (run.get('change') or {}).get('name')
    if change is None:
        life = free.Life(run['first'], run['key'], record, run['muggers'], run['xp_rate'], run['ticks'], tables, run['split'],
                         world=run['world'])
    else:  # the base record changed by sim/changed.py, its seeds the base row's
        from sim import changed
        life = changed.life(change, run['first'], run['key'], record, run['muggers'], run['xp_rate'], run['ticks'], tables,
                            run['split'], world=run['world'])
    if name in POLICY_ACTORS:
        p = next(p for p in run['passes'] if p['name'] == pass_name)
        actor, seed = FreePolicyActor(models[name], deterministic=not p['sampled']), trial_seed(p['seed'], run['policy']['seed'], record)
        who = dict(policy_seed=run['policy']['seed'], checkpoint=run['policy'][name]['sha256'], deterministic=not p['sampled'])
    else:
        actor, seed = free.make(name, run['period']), [run['control_seed'], record]
        who = dict(policy_seed=None, checkpoint=None, deterministic=None)
    marks = free.PhaseMarks(actor) if isinstance(actor, free.Rotation) else None
    reader = FreeRead(life.task, run['read'])
    row = life.row(name, run['period'], actor, free.play(life.task, actor, seed, watch=marks), marks, t0)
    row.update(rows=FREE['rows'], split=run['split'], first=run['first'], seed=seed, read=reader.read,
               **{'pass': pass_name}, **who, **free_reads(life.task), **({'change': change} if change else {}))
    return row


def free_rows(path):
    """(rows, whether a last line was cut) of a part's lives file: a line cut by a stopped process can only be the last."""
    lines = Path(path).read_text().splitlines(keepends=True) if Path(path).exists() else []
    rows = []
    for k, line in enumerate(lines):
        try:
            if not line.endswith('\n'):
                raise ValueError('no line end')
            rows.append(json.loads(line))
        except ValueError:
            if k != len(lines) - 1:
                raise ValueError(f'{path}: line {k + 1} is not a row')
            return rows, True
    return rows, False


def run_facts(run):
    """FREE's run facts of a training run's summary.json (None where the summary holds no such fact)."""
    learner, batch = run.get('learner') or {}, (run.get('free_play') or {}).get('batch_in_effect') or {}
    facts = dict(decisions=run.get('decisions'), worlds=run.get('worlds'), muggers=batch.get('muggers'),
                 life=batch.get('life'), pool=(run.get('pool') or {}).get('sha256'),
                 tables=(run.get('tables') or {}).get('sha256'), validate=run.get('validate'))
    return {k: facts[k] if k in facts else learner.get(k) for k in FREE['run_facts']}


def engine_facts(run):
    """FREE's engine facts of a training run's summary.json, or of the run record that policy_run writes from it: its
    engine and its pack's sha256 (None without a pack)."""
    return dict(engine=run.get('engine'), pack=(run.get('pack') or {}).get('sha256'))


def policy_run(checkpoint, initial, facts, policy_seed, muggers, xp_rate, unfinished=False, *, world):
    """The training run that the checkpoint and its own saved initialization come from (a policy is checked against
    its actual saved initialization), from the training command's summary.json beside the initialization: a
    free-play-v1 run of `policy_seed` that did not resume, whose tensors at the start are the initialization's, and whose
    folder holds the checkpoint as latest.zip or as one of its checkpoints/. When the summary records the final
    checkpoint's tensor digest (policy_sha256_at_end) or the periodic ones' (checkpoint_digests, by file name), the
    checkpoint's must be the recorded one. latest.zip needs a completed run; another checkpoint of a run that has not
    completed only with `unfinished`, and every reading of it is then a description (sim/gate.py free_plan). A run with
    no summary (one killed before sim/train.py wrote it) cannot be evaluated or described: `unfinished` serves a run
    that failed and left its summary. The run's batch in effect must be the evaluation's world: the world `world` (a
    name of sim/combat.py VILLAGES; the run's own, trained_world: a summary that names no world is refused), obs-v6, the
    muggers and the agent settings of free_play_cfg, and the XP rate, or 1 when the evaluation is the declared 25x read.
    The archive the run trained on (summary 'tables') must be the evaluation world's (world_archive), the one its lives
    play over; a summary that records none (an older C-engine run) is not compared. Returns the
    run's facts for the manifest: FREE's run facts, its start (environment seed, the --weights and --init records) and
    the source it trained from (summary 'source', baseline/telemetry.py source(); None for an older summary)."""
    folder = Path(initial).resolve().parent
    summary = folder / 'summary.json'
    if not summary.exists():
        raise ValueError(f'{initial} has no run summary beside it ({summary}): only a run that left its summary.json '
                         '(completed, or failed with one) is evaluated or described')
    run = json.loads(summary.read_text())
    if 'resume' in run:
        raise ValueError(f'the run {summary} resumed from a recovery state: a resumed run is not evaluated')
    if run.get('task') != FREE['task'] or run.get('seed') != policy_seed or \
            run.get('policy_sha256_at_start') != facts['initial']['policy_sha256']:
        raise ValueError(f'{initial} is not shown to be the initialization of a {FREE["task"]} run of seed {policy_seed}: '
                         f'its run summary {summary} must record that task and seed and its tensors at the start')
    where = Path(checkpoint).resolve()
    name = 'latest.zip' if where == (folder / 'latest.zip').resolve() else \
        f'checkpoints/{where.name}' if where.parent == (folder / 'checkpoints').resolve() else None
    if name is None:
        raise ValueError(f'{checkpoint} is not latest.zip or one of checkpoints/ of the run of {initial}')
    final, periodic = run.get('policy_sha256_at_end'), run.get('checkpoint_digests')
    if name == 'latest.zip':
        checked, recorded = final is not None, final
    else:
        checked, recorded = periodic is not None, (periodic or {}).get(where.name)
    if checked and recorded != facts['policy']['policy_sha256']:
        raise ValueError(f'{checkpoint} holds other tensors than its run summary {summary} records for {name}')
    status = run.get('status')
    if status == 'completed' and unfinished:
        raise ValueError(f'the run {summary} completed: unfinished names a run that has not')
    if status != 'completed' and (name == 'latest.zip' or not unfinished):
        raise ValueError(f'the run {summary} has status {status!r}: latest.zip needs a completed run, and another '
                         'checkpoint of a run that has not completed needs unfinished (a description only)')
    batch, tables = (run.get('free_play') or {}).get('batch_in_effect') or {}, (run.get('tables') or {}).get('sha256')
    if 'world' not in batch and trained_world(batch, tables) is None:
        raise ValueError(f"the run {summary} records no world and trained over another archive ({tables}) than "
                         f"{FREE['first_world']}'s: a record that names no world is not evaluated")
    seen = dict(batch, world=trained_world(batch, tables))
    wanted = dict(world=world, observation=FREE_PLAY['observation'], muggers=muggers,
                  cfg=json.loads(json.dumps(free_play_cfg())))
    differ = sorted(k for k, v in wanted.items() if seen.get(k) != v)
    trained = batch.get('xp_rate')
    differ += [] if trained == xp_rate or (trained, xp_rate) == (1, 25) else ['xp_rate']
    if differ:
        raise ValueError(f'the run {summary} trained in another world than this evaluation: {", ".join(differ)} differ')
    if tables is not None and tables != world_archive(world):
        raise ValueError(f"the run {summary} trained over another archive ({tables}) than {world}'s "
                         f'({world_archive(world)}), which its lives play over')
    return dict(summary=str(summary), checkpoint=name, seed=run['seed'], policy_sha256_at_start=run['policy_sha256_at_start'],
                policy_sha256_at_end=final, status=status, unfinished=unfinished, batch_in_effect=batch,
                facts=run_facts(run), env_seed=run.get('env_seed'), weights=run.get('weights'), init=run.get('init'),
                env=run.get('env'), engine=run.get('engine'), pack=run.get('pack'), source=run.get('source'))


class FreeStop(RuntimeError):
    """A life stopped at an unsupported site: its row holds the evidence and the part plays no further life."""


def agree(path, manifest):
    """Write a run's manifest, or check that the one there (another part's, or this part's before a resume) holds the same
    settings; written by a link from a part's own file, so two parts that start together never both write it."""
    text = json.dumps(manifest, indent=1, sort_keys=True, default=str) + '\n'
    mine = path.with_name(f'.{path.name}.{os.getpid()}')
    mine.write_text(text)
    try:
        os.link(mine, path)
    except FileExistsError:
        there, here = json.loads(path.read_text()), json.loads(text)
        differ = sorted(k for k in set(there) | set(here) if there.get(k) != here.get(k))
        if differ:
            raise ValueError(f'{path} holds another run\'s settings: {", ".join(differ)} differ')
    finally:
        mine.unlink()


def free_run(output, *, world, split, first, records, key, muggers, xp_rate, ticks=FREE['ticks'], read=FREE['read'],
             checkpoint=None, initial=None, policy_seed=None, seeds=None, controls=(), period=None, control_seed=None,
             unfinished=False, plan=None, part=(0, 1), final=False, tables=None, change=None, change_passes=('sampled1',)):
    """Part `part[0]` of `part[1]` of a free-play evaluation (module docstring) of `world` (a name of sim/combat.py
    VILLAGES; no default): every `part[1]`-th life of free_jobs from the part's index, each record a fresh world of its
    layouts and one life of `ticks` at `xp_rate`, over the world's archive (sim/free.py world_tables, whose path and
    sha256 the manifest and the plan check record; `tables`, a loaded archive, must be that file's), written to
    lives-{part}.jsonl one row a life as it ends, by one process at a time (an exclusive lock on that file). Resumable:
    the rows already in the
    part's file are kept and not played again (a cut last line is dropped), and the run's manifest.json must hold the
    same settings. Every setting is checked, the whole record range and the policy's run (policy_run) included, before
    anything is written. With `plan` (a name of sim/gate.py FREE['plans']), the output is refused before any life is
    played when that plan would not read it (sim/gate.py free_plan_check: a plan not frozen, other settings, a source
    that is not one commit with no change, a run that does not fit); sim.gate free stays the judge. An unsupported site
    writes its life's row with the evidence and raises FreeStop; a part that holds such a row refuses to go on. `change`:
    a changed world of sim/changed.py (module docstring), its records those of the range whose base holds the fact, the
    policy in `change_passes`; refused with `plan`, in another world, with muggers, or with a control its reads do not
    need (the controls default to those they need)."""
    import fcntl
    from baseline.telemetry import source
    from sim import free
    from sim.combat import check_village
    from sim.data import Tables
    index, count = part
    if world not in VILLAGES:
        raise ValueError(f'free play is evaluated in a world of {sorted(VILLAGES)}, named with no default: not {world!r}')
    if split not in ('development', 'final') or (split == 'final') != final:
        raise ValueError('free play is evaluated on development or final records, and the final split needs final=True')
    if type(first) is not int or first < 0 or type(records) is not int or records < 1:
        raise ValueError(f'records from a first index of 0 or more: first {first!r}, records {records!r}')
    split_seeds(split, records, first)  # refuses a range past the split's end, before anything is written
    if type(muggers) is not bool or xp_rate not in FREE_PLAY['xp_rates']:
        raise ValueError(f"name the muggers (True or False; no default) and an XP rate of {FREE_PLAY['xp_rates']}")
    if not (type(ticks) is int and type(read) is int and 0 < read < ticks) or not 0 <= index < count:
        raise ValueError(f'a read before the life\'s end ({read} of {ticks} ticks) and a part of the run')
    if change is not None:
        from sim import changed
        if plan is not None:
            raise ValueError('a changed world is a description of the scorecard, never a plan\'s evaluation: no --plan')
        if change not in changed.CHANGES:
            raise ValueError(f'change {change!r} is not one of {sorted(changed.CHANGES)}')
        if world != changed.WORLD or muggers is not False:
            raise ValueError(f'the changed worlds are {changed.WORLD}\'s without muggers (sim/changed.py): not {world!r}, '
                             f'muggers {muggers!r}')
        needs = (*changed.CHANGES[change].scripts, 'random', 'wait')
        controls = tuple(c for c in free.CONTROLS if c in needs) if not controls else tuple(controls)
        if set(controls) - set(needs):
            raise ValueError(f'{change} reads the controls {sorted(needs)}: not {sorted(set(controls) - set(needs))}')
        if not change_passes or len(set(change_passes)) != len(change_passes) or \
                set(change_passes) - {p for p in FREE['passes'] if p != 'greedy'}:
            raise ValueError(f'a changed world plays the policy in one or both sampled passes: not {list(change_passes)}')
    unknown = sorted(set(controls) - set(free.CONTROLS))
    if unknown or len(set(controls)) != len(controls) or (checkpoint is None and not controls):
        raise ValueError(f'controls among {free.CONTROLS}, each once, and a checkpoint or a control to play: {list(controls)}')
    if checkpoint is not None and (initial is None or policy_seed is None or seeds is None or len(seeds) != 2
                                   or seeds[0] == seeds[1]):
        raise ValueError('a policy plays with its own saved initialization, its seed and two sampled passes\' seeds')
    if checkpoint is None and unfinished:
        raise ValueError('unfinished names the run of a checkpoint: there is none')
    if control_seed is None:
        raise ValueError('the evaluation needs its control seed (random legal play\'s; the soak\'s actor seed), which every '
                         'output of one evaluation shares')
    if ('rotation' in controls) != (period is not None):
        raise ValueError('the rotation needs its period and only the rotation takes one: the study declares none')
    if period is not None:
        free.Rotation(free.ROTATION_ORDER, period)  # refuses a period that is not a positive whole number of ticks
    archive = free.world_tables(world)
    if not archive.exists():
        raise ValueError(f'{archive} is missing: {world} plays over it (sim/build.py says how each archive is made)')
    archive_sha256 = world_archive(world)
    models, facts = {}, {}
    if checkpoint is not None:
        missing = [str(p) for p in (checkpoint, initial) if not Path(p).exists()]
        if missing:
            summary = Path(initial).parent / 'summary.json'
            begun = summary.exists() and json.loads(summary.read_text()).get('status') == 'initialized'
            raise ValueError(f'{missing[0]} does not exist' + (': its run stopped at its start (status initialized, '
                                                              'sim.train --initial-only) and saved no checkpoint to play'
                                                              if begun else ''))
        for name, path in (('policy', checkpoint), ('initial', initial)):
            models[name], facts[name] = load_free_policy(path)
        facts['run'] = policy_run(checkpoint, initial, facts, policy_seed, muggers, xp_rate, unfinished, world=world)
    passes = [dict(name=name, seed=seeds[0] if name == 'greedy' else seeds[k], sampled=name != 'greedy')
              for k, name in enumerate(FREE['passes'])] if checkpoint is not None else []
    if plan is not None:
        from sim.gate import free_plan_check
        shown = dict(world=world, split=split, first=first, records=records, key=key, xp_rate=xp_rate, muggers=muggers,
                     ticks=ticks, read=read, period=period, control_seed=control_seed, tables=archive_sha256)
        policies = {} if checkpoint is None else {policy_seed: json.loads(json.dumps(
            dict(policy=facts['policy'], initial=facts['initial'], run=facts['run'],
                 passes={p['name']: p['seed'] for p in passes}), default=str))}
        reasons = free_plan_check(plan, shown, policies, source())
        if reasons:
            raise ValueError(f'plan {plan} would not read this output: {"; ".join(reasons)}')
    with np.load(archive) as data:  # the archive's own record of what it holds, beside the loaded one's
        held = json.loads(data['meta'].tobytes())
    if tables is None:
        tables = Tables.load(archive)
    elif tables.meta != held:
        raise ValueError(f'the tables given are not {archive.name}, the archive {world} plays over')
    check_village(VILLAGES[world], tables, muggers)
    if change is not None:
        held_records = changed.records(change, first, records, key, tables, split, world=world)
        if not held_records:
            raise ValueError(f'no record of {first} .. {first + records - 1} holds the fact of {change}')
        change = dict(name=change, records=held_records,
                      passes=[p for p in FREE['passes'] if p in change_passes] if checkpoint is not None else [])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run = dict(task=FREE['task'], rows=FREE['rows'], world=world, observation=FREE_PLAY['observation'],
               source=source(output), tables=dict(path=str(archive), sha256=archive_sha256), split=split, first=first,
               records=records, key=key, muggers=muggers, xp_rate=xp_rate, ticks=ticks, read=read, passes=passes,
               policy=dict(seed=policy_seed, **facts) if checkpoint is not None else None, controls=list(controls),
               period=period, control_seed=control_seed, plan=plan, parts=count, torch_threads=FREE['torch_threads'],
               **({'change': change} if change is not None else {}))
    agree(output / 'manifest.json', run)
    run = json.loads(json.dumps(run, default=str))
    path = output / f'lives-{index}.jsonl'
    lock = path.open('a')  # held while this process plays the part: one process a part
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError(f'{path} is being played by another process: one process a part')
    try:
        return play_part(path, run, models, tables, records, checkpoint is not None, controls, part)
    finally:
        lock.close()


def play_part(path, run, models, tables, records, policy, controls, part):
    """The lives of one part (free_run) that its lives file does not hold yet, played and written one row a life."""
    index, count = part
    rows, cut = free_rows(path)
    if cut:
        path.write_text(''.join(json.dumps(r, default=str) + '\n' for r in rows))
    stopped = [r for r in rows if r['unsupported'] is not None]
    if stopped:
        r = stopped[0]
        raise FreeStop(f"{path} stopped at an unsupported site ({r['actor']} {r['pass']} record {r['record']}: "
                       f"{r['unsupported']['message']}); a part with such a row plays no further life")
    mine = free_jobs(records, policy, controls, run.get('change'))[index::count]
    done = {(r['actor'], r['pass'], r['record']) for r in rows}
    if not done <= set(mine) or len(done) != len(rows):
        raise ValueError(f'{path} holds rows that are not this part\'s, or a life twice')
    import torch
    threads, t0 = torch.get_num_threads(), time.perf_counter()
    torch.set_num_threads(FREE['torch_threads'])
    played = 0
    try:
        with path.open('a') as stream:
            for job in mine:
                if job in done:
                    continue
                row = free_life(job, run, tables, models)
                stream.write(json.dumps(row, default=str) + '\n')
                stream.flush()
                played += 1
                if row['unsupported'] is not None:
                    raise FreeStop(f"{job}: unsupported site {row['unsupported']['file']}:{row['unsupported']['line']} "
                                   f"({row['unsupported']['message']}); its row in {path} holds the evidence")
    finally:
        torch.set_num_threads(threads)
    return dict(part=[index, count], lives=len(mine), played=played, kept=len(done), cut=cut,
                seconds=round(time.perf_counter() - t0, 1))


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
    from sim.free import CONTROLS
    free = commands.add_parser('free', help='free-play-v1: a policy, its initialization and the controls on one record set '
                               '(one part)')
    free.add_argument('--output', type=Path, required=True)
    free.add_argument('--world', required=True, choices=sorted(VILLAGES),
                      help='the world played, over its own archive; the policy must have trained in it (no default)')
    free.add_argument('--split', default='development', choices=('development', 'final'))
    free.add_argument('--final', action='store_true', help='the final split is final evaluation data: say so')
    free.add_argument('--first', type=int, required=True, help="the split's index of record 0's layout")
    free.add_argument('--records', type=int, required=True)
    free.add_argument('--key', type=int, required=True, help="the start draws' key (sim/combat.py draw_village)")
    free.add_argument('--muggers', type=int, choices=(0, 1), required=True, help='Stage A without (0) or with (1) muggers')
    free.add_argument('--xp-rate', type=int, choices=FREE_PLAY['xp_rates'], required=True)
    free.add_argument('--ticks', type=int, default=FREE['ticks'], help="a life's native ticks from arm")
    free.add_argument('--read-ticks', type=int, default=FREE['read'], help='the read of the same life beside it')
    free.add_argument('--checkpoint', type=Path, help='a free-play-v1 checkpoint (obs-v6)')
    free.add_argument('--initial', type=Path, help="that run's saved initialization (its initial.zip)")
    free.add_argument('--policy-seed', type=int, help="the policy's training seed (its lives' seeds)")
    free.add_argument('--seeds', type=int, nargs=2, metavar=('SAMPLED1', 'SAMPLED2'), help="the sampled passes' seeds")
    free.add_argument('--controls', nargs='+', default=[], choices=CONTROLS)
    free.add_argument('--period', type=int, help="the rotation's ticks a member (no default)")
    free.add_argument('--control-seed', type=int, required=True,
                      help="the controls' actor seed (random legal play's), which every output of one evaluation shares")
    free.add_argument('--plan', help="a plan of sim/gate.py FREE['plans']: refuse before any life is played when that "
                                     'plan would not read this output')
    free.add_argument('--unfinished-run', action='store_true',
                      help='the checkpoint is one of checkpoints/ of a run that failed and left its summary.json: a '
                           'description only')
    free.add_argument('--part', type=int, nargs=2, default=(0, 1), metavar=('I', 'N'))
    free.add_argument('--change', help='a changed world of sim/changed.py CHANGES (the scorecard; never with --plan)')
    free.add_argument('--change-passes', nargs='+', default=['sampled1'], choices=('sampled1', 'sampled2'),
                      help="with --change, the policy's sampled passes (default: pass 1 alone)")
    args = parser.parse_args()
    if args.command == 'free':
        try:
            result = free_run(args.output, world=args.world, split=args.split, first=args.first, records=args.records,
                              key=args.key,
                              muggers=bool(args.muggers), xp_rate=args.xp_rate, ticks=args.ticks, read=args.read_ticks,
                              checkpoint=args.checkpoint, initial=args.initial, policy_seed=args.policy_seed,
                              seeds=args.seeds, controls=tuple(args.controls), period=args.period,
                              control_seed=args.control_seed, unfinished=args.unfinished_run, plan=args.plan,
                              part=tuple(args.part), final=args.final, change=args.change,
                              change_passes=tuple(args.change_passes))
        except (ValueError, FreeStop) as error:
            raise SystemExit(f'[free] {error}')
        print(json.dumps(result))
        return
    if args.command == 'merge':
        result = merge(args.output)
        print(json.dumps(result['cells'], indent=1))
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
