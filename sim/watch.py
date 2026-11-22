"""native-watch-v1: a person watches a goal-free policy (free-play-v1, obs-v6) play the real engine. A description,
never a gate: no monitor is paid and none has a bar.

The world is a setting of every run, named with no default (--world: a name of sim/combat.py VILLAGES, village-world-v1
Stage A or village-world-v2, or of REAL_WORLDS, village-world-v3, the real engine's only): the run applies its rules
(world_rules: its exclusions, its start group and its loot outside the slice) and its archive (sim/free.py
world_tables, recorded with path and sha256), a policy actor's training run must have trained in it (sim/hour.py
policy_run, or the launch of a real-engine checkpoint, real_run), and run.json, the monitors and the proposal record
it. The sites are a setting apart: any four names of SITES, in any world (a policy of one world can be watched at
another world's sites). A run.json
that records no world was written before the world was a setting, when no other world existed: it is village-world-v1's
(run_world). `check` makes every check that needs no native world, the policy's world, the output folder and the engine
checkout included, before scripts/stack.py starts any; `run` makes them again before it joins a world. The random-events
and XP-rate checks read the stack's own environment and stay in `run`.

Four persistent worlds of one scripts/stack.py run (--worlds 4 --lockstep --xp-rate 1, never paced), one site a world,
each played by one thread until the world has spent WATCH['ticks'] armed native ticks. A life starts from the world's
next start record (Records), is armed for the ticks its world has left, and ends at a death or when those ticks are
spent; after a death the same world starts its next life from its next record. Ticks between lives (the reset and the
setup steps before arm) are not armed and are reported apart. A wall-time guard (WATCH['guard'] s from the first reset
of the run) stops every world: a run that the guard cuts is incomplete, not a result. A world whose transport, capture
control or setup fails stops, its life ends incomplete with its ticks to its last clock tick, and the run is
incomplete; a world whose thread is still running after the guard and its grace is 'hung'. A capture failure in a life's
read trace (capture_failure: the combat capture's, an overflow past its event bound included, or the resource capture's
other than a retire) makes that life incomplete and the run too; its world goes on with its next record.

Start records (Records): the simulator's free-play start distribution as far as a native reset can write it
(the world's start group, `danger` in both worlds, without muggers; sim/spec.py START_LEVELS and START_WEAPONS): base
levels, the weapon, 1-3 food items (shrimp or bread), the tools, the weapon worn in half of the lives (a setup Wield after
prepare_declared's style step, as sim/task.py _setup) and a reset seed that picks the start tile within 3 of the site's
tile (env-ts/core/world.ts jitterWorld). Hitpoints below the base level are drawn and recorded but not applied: a
native reset writes levels only (env-ts/engine/agent.ts reset), so every native life starts at full hitpoints. Each
world draws its own record stream from (key, world); the sites rotate over the worlds (site_order: world k plays site
(k + rotation) mod 4), so over four rotations every record stream meets every site.

Actors: WAIT and random legal play (sim/free.py Wait, Random), a free-play run's saved initialization (`initial`) and a
checkpoint of that run (`policy`), both loaded by sim/hour.py load_free_policy (obs-v6 only) and tied to their training
run by policy_run, sampled unless --greedy. A checkpoint of a real-engine run (sim.train --env real; one that has no
summary.json while incomplete) plays as `policy` with --real-checkpoint FILE: its checks are its own (load_free_policy:
obs-v6, the contract versions, the name table), its world rules the watch's --world, and the run record holds the file's
path, sha256 and tensor digest, the training run's launch.json with --launch (a real-engine free-play run at XP rate 1
in the watch's world; its start weights are recorded) and that it trained in the real engine (real_run); a
free-play-v3 checkpoint plays with the extra kit it saved (its tinderbox) and its monitors add every stat's level gains
(summarize). A life's seed is life_seed(seed, world, life). Policy calls hold one lock and each actor keeps its own torch generator state, so the
four threads never share a draw.

Two tools for longer lives (stage 1, no training). A profile (`--start-levels name=level,...`, parse_levels)
plays the same records with the named stats at those levels and every other draw as drawn (profiled), each life's
hitpoints at the arm the new hitpoints level less the record's own deficit; run.json, each watch_life row and the
monitors name it, and its folder is not named as a fixed-form watch ('...final'). Without it the records and the cfgs
are as before. The mining script (`--actor script-mine`, sim/mine_script.py; a check of feasibility, never a label)
plays the site rimmington_mine_west; a watch of one site (`--sites SITE`) plays it in all four worlds, each world with
its own record stream, so the policy can be watched at that site on the same records. Every life reads reached_rock
and ores_per_10k, and the summaries read the ores, mining and every stat's levels for each 10,000 ticks by start Mining
level (by_start_mining).

The world's declared exclusions (sim/engine.py EXCLUSIONS) have no native switch but random events
(NODE_RANDOM_EVENTS=false, required): clue and loot drops still fall. Its excluded objects (excluded_items) are the
treasure-trail objects and its loot outside the slice: a cabbage is one in village-world-v1 and is in village-world-v2's
declared world (its field's Pick), so only a village-world-v1 life ends on a cabbage, picked or dropped. A life is
flagged (`excluded_seen`) at its first observation, the arm's included, that shows one in its ground list (the six
nearest ground items within 15 tiles) or among its Take affordances (excluded_in_view); a drop farther down both lists
is not seen. A pickup of one (the
inventory holds it) ends that life as incomplete (retired); its world goes on with its next record. Flagged lives and
pickup lives are reported apart (run.json `apart`, the monitors' `apart` block); a pickup alone leaves the run complete.

Raw evidence: one JSONL file a world (world-K.jsonl.gz: every reset, observation, request, control reply and the read
trace with its combat evidence, each row with its life), and run.json. The monitors (sim.watch monitors) are computed
from those files after the run, one life at a time: production (logs, ores, fish), fish cooked and burnt, kills, eats
with hitpoints at each eat and eats below full hitpoints, damage taken, dealt and planned, deaths, distance, time by
activity, switches, bank trips, levels and XP by skill up to the terminal (the XP of the step the terminal falls in
apart), new items, drops of food and of tools read from the observations (watch_food_drops: an executed drop row whose
entry offers Eat, watch_tool_drops: one whose entry's name env-ts TOOL_PROTECT_RE names; each counted when that kind's
count in the pack fell: not the engine's life-row rule, which counts by the dropped slot), the capture's
event counts against their bounds and failures, and the controls' wall times; the
summary covers the clean lives, and the flagged, pickup
and incomplete lives each have their own block. Natively an activity is read from the observed XP between two
decisions (the native XP events are deferred), so its times are at decision resolution. `propose` refuses a rehearsal
that does not measure every world.

    PYTHONPATH=.:rl .venv/bin/python -m sim.watch check --output OUT/run --world WORLD --actor ACTOR \\
        [--run RUN --policy-seed S] --sites A B C D --rotation R --key K --seed S [--ticks T] [--guard G]  # before the stack
    NODE_RANDOM_EVENTS=false PYTHONPATH=.:rl .venv/bin/python scripts/stack.py --bun BUN --output OUT/stack \\
        --timeout-seconds 900 --lockstep --xp-rate 1 --worlds 4 -- \\
        .venv/bin/python -m sim.watch run --output OUT/run --world WORLD --actor ACTOR [--run RUN --policy-seed S] \\
        --sites A B C D --rotation R --key K --seed S [--ticks T] [--guard G] [--rehearsal]
    PYTHONPATH=.:rl .venv/bin/python -m sim.watch monitors OUT/run [...]
    PYTHONPATH=.:rl .venv/bin/python -m sim.watch propose OUT/run [...]      # after the rehearsal's runs
"""
import argparse
import bisect
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

from baseline import config
from baseline.evidence import EvidenceError
from baseline.resource_task import GOAL_FAMILIES, ResourceEpisode, need
from feedback_contract import episode_start
from sim.agent import A_DEPOSIT0, A_DROP0, A_HELD0, A_WITHDRAW0, TOOL_PROTECT_RE
from wide_contract import ITEM_V4, page_entry, page_row, project_free, project_v7
from sim.combat import COMBAT, VILLAGE, VILLAGES
from sim.spec import START_LEVELS, START_WEAPONS
from sim.task import FREE_PLAY, free_play_cfg
from obs import SKILL_ORDER
from item_contract import ITEM_V2
from sim.workers import FREE_PLAY_VERSIONS, REAL_WORLDS, VILLAGE_SITES, item_contract, refuse_removed

WATCH = dict(
    version='native-watch-v1', task=FREE_PLAY['version'], observation=FREE_PLAY['observation'],
    worlds=4,                       # four worlds side by side, one site a world
    ticks=36000,                    # armed ticks a world (six game hours); the rehearsal proposes the frozen count
    guard=600.0,                    # wall seconds from the run's first reset; a cut run is incomplete
    grace=60.0,                     # seconds after the guard for a world's thread to retire its capture and stop
    hour=6000,                      # native ticks of one game hour
    xp_rate=1,                      # native 1x (scripts/stack.py --xp-rate 1, NODE_XPRATE=1)
    engine='pinned',                # the vendor checkout's build (baseline/config.py ENGINE_TREES)
    max_lives=512,                  # a world that needs more start records stops (none should: a life takes ticks)
    event_bound=20000,              # env-ts maxEvents of both captures, a started hour of the window (eventLimit)
    actors=('wait', 'random', 'initial', 'policy', 'script-mine'),
)
SCRIPT_ACTOR = 'script-mine'  # sim/mine_script.py MineScript, imported only when it plays: a check of feasibility
# The world of a run.json that names none: written before the world was a setting (the 13 November rehearsal and first
# watch), when village-world-v1 was the only world (run_world).
FIRST_WORLD = VILLAGE['version']
TOOLS = tuple(name for name, _ in COMBAT['tools'])
COMBAT_SKILLS = ('attack', 'strength', 'defence', 'hitpoints')
PRODUCT = dict(wood='logs', mine='ores', fish='fish')
# Candidate sites, none chosen by default: name -> start tile (x, z), level 0, on the pinned build. From a
# read-only census of 12 November over the vendor map and the SDK collision export (8-neighbour walking steps, ordinary
# doors and gates passable): no native tile holds a plain tree, a copper or tin rock, a small-net spot, a chicken and a
# goblin within 60 steps; the comment gives the nearest of each in that order. compact-world-v1 holds each within 40
# steps of every start (sim/world.py WORLD goal_steps; median 27 for the farthest of the five). The chickens of every
# candidate but draynor_manor_south are behind a 'Door' loc, a kind no training layout has (its pens have a 'Gate'), and
# the rocks of every candidate but that one are the Rimmington mine, the Rimmington check site's (final data,
# baseline/config.py START_SETS).
SITES = dict(
    port_sarim_north=(3050, 3270),      # 1, 67, 67, 39, 59; two level-7 dark wizards 6-8 tiles from its net spots
    draynor_west=(3070, 3277),          # 3, 87, 47, 57, 39; the same dark wizards
    falador_farm_west=(3002, 3279),     # 1, 33, 104, 52, 62
    rimmington_coast=(2978, 3197),      # 5, 38, 26, 109, 17; 52 tiles from drop1 (a held-out test point)
    rimmington_mine_west=(2985, 3231),  # 5, 3, 56, 78, 18; 13 tiles from the Rimmington check start
    draynor_manor_south=(3092, 3326),   # 15, 98, 97, 98, 23; Barbarian Village's tin, Lumbridge's chickens through a Gate;
                                        # an npc named Tree that attacks, and dark wizards
    # village-world-v3's shore sites (the tiles chosen on 19 November), from a census of 19 November by the same method
    # over the pinned build's own boot collision: the walking steps to the small-net spot, then the nearest plain tree,
    # chicken and goblin. No start of either, at its radius (sim/workers.py REAL_WORLDS) and with the reset's jitter, is
    # within 15 tiles of an npc that attacks first.
    rimmington_shore_east=(3001, 3155),  # 4 (spot 2996,3158), 6, 148, 45
    rimmington_shore_west=(2991, 3169),  # 8 (spot 2986,3176; beside its position 2990,3169), 12, 136, 33
)
# A site's own start radius (the engine's places around its tile; sim/workers.py REAL_WORLDS): every other site takes
# the batch's (sim/real.py REAL radius 16 unless --batch-setting radius). With 16 at a shore site, 2 to 30% of the
# starts would have no spot in view.
SITE_RADIUS = {site: r for world in REAL_WORLDS.values() for site, r in world['site_radius'].items()}
# The Rimmington mine's rocks of copper, tin and iron: tile -> (ore, the loc type
# at boot, the Mining level the ore needs). The copper and tin tiles are baseline/config.py START_SETS['rimmington']'s
# list less its two clay rocks (2986,3239 and 2987,3240: clay, 5 XP an ore, is in no simulator world, sim/world.py
# ROCKS); the iron tiles are the mine's seven from the vendor map (server/content/maps/m46_50.jm2, loc types 2092 and
# 2093 = ironrock1, ironrock2), two of them 3 and 4 tiles from rimmington_mine_west and all within 17 of it, as the
# listed coppers are. A mined rock turns into an empty rock (plainrock) that keeps the name 'Rocks' and the option
# Mine, so a rock is live while its loc type is its type at boot (the affordance's `id`).
MINE_ROCKS = {
    (2982, 3234): ('iron', 2092, 15), (2981, 3233): ('iron', 2093, 15), (2986, 3235): ('tin', 2094, 1),
    (2984, 3237): ('tin', 2094, 1), (2971, 3237): ('iron', 2093, 15), (2977, 3245): ('copper', 2091, 1),
    (2969, 3240): ('iron', 2092, 15), (2969, 3242): ('iron', 2092, 15), (2976, 3247): ('copper', 2090, 1),
    (2977, 3247): ('copper', 2090, 1), (2968, 3239): ('iron', 2092, 15), (2978, 3248): ('copper', 2090, 1),
    (2979, 3248): ('copper', 2090, 1),
}
MINE_SITES = ('rimmington_mine_west',)  # the sites the mining script plays: a rock of MINE_ROCKS 3 steps away


# ---- the world -------------------------------------------------------------------------------------------------------
def village(world):
    """The definition of free-play world `world`: a world of sim/combat.py VILLAGES, or a real-engine world of
    sim/workers.py REAL_WORLDS over the rules of the world it is based on (village-world-v3: village-world-v2's
    exclusions, start group, loot outside the slice and archive), named with no default."""
    if world in VILLAGES:
        return VILLAGES[world]
    if world in REAL_WORLDS:
        return dict(VILLAGES[REAL_WORLDS[world]['based_on']], **REAL_WORLDS[world])
    raise ValueError(f'a watch applies the rules of a world of {sorted(VILLAGES) + sorted(REAL_WORLDS)}, named with no '
                     f'default: not {world!r}')


def world_sites(world):
    """The sites a real-engine batch of world `world` draws from when it names none, and the only ones it may name: the
    world's own (village-world-v3's eight), else VILLAGE_SITES. A watch names any of SITES in any world."""
    return tuple(village(world).get('sites', VILLAGE_SITES))


def world_rules(world):
    """The rules of free-play world `world` (village(): a name of sim/combat.py VILLAGES or REAL_WORLDS, given with no
    default) that a watch applies, as sim/combat.py village_life gives them to the simulator's World: its exclusions, its
    start group and its loot outside the slice (sim/engine.py LOOT_OUTSIDE where the world names no part of it)."""
    from sim.engine import LOOT_OUTSIDE
    village_spec = village(world)
    loot = village_spec.get('loot_outside')
    return dict(exclusions=list(village_spec['exclusions']), start_group=village_spec['start_group'],
                loot_outside=list(LOOT_OUTSIDE if loot is None else loot))


def run_world(run):
    """The world whose rules a watch run applied: its run.json's `world`, held to as it is; FIRST_WORLD for a run.json
    that records none (written before the world was a setting)."""
    return run['world'] if 'world' in run else FIRST_WORLD


# ---- start records ---------------------------------------------------------------------------------------------------
def tele(x, z):
    """baseline/config.py's spawnTele form of tile (x, z), level 0."""
    return f'0,{x >> 6},{z >> 6},{x & 63},{z & 63}'


class Records:
    """World `world`'s stream of start records under `key` in start group `group` (sim/combat.py COMBAT groups: the
    played world's, world_rules), drawn in order (module docstring). Each record: set_stats and seed_items in
    baseline/world.py's declared form (the tools, the food, then the weapon: sim/combat.py kit's order), the weapon, the
    food, whether the weapon starts worn, the drawn hitpoints (applied natively by free-play-v12 to v14 alone: life_cfg
    startHp) and the reset seed."""

    def __init__(self, key, world, group):
        if type(key) is not int or type(world) is not int or key < 0 or world < 0:
            raise ValueError(f'a record stream needs a key and a world index of 0 or more: {key!r}, {world!r}')
        self.rng, self.drawn = np.random.default_rng(np.random.SeedSequence([key, world])), 0
        self.spec = COMBAT['groups'][group]

    def next(self):
        rng, spec = self.rng, self.spec
        levels = {stat: int(rng.choice(choices)) for stat, choices in START_LEVELS.items()}
        weapon = START_WEAPONS[int(rng.integers(len(START_WEAPONS)))]
        food = [COMBAT['food'][int(rng.integers(len(COMBAT['food'])))] for _ in range(int(rng.integers(spec['food'][0], spec['food'][1] + 1)))]
        hp = max(1, min(levels['hitpoints'], levels['hitpoints'] - int(rng.integers(spec['hp'][0], spec['hp'][1] + 1))))
        worn = bool(rng.random() < COMBAT['worn'])
        record = dict(index=self.drawn, levels=levels, weapon=weapon, food=food, worn=worn, hp_drawn=hp,
                      reset_seed=int(rng.integers(1, 1_000_000)),
                      set_stats=','.join(f'{stat}:{level}' for stat, level in levels.items()),
                      seed_items=','.join(f'{name}:1' for name in (*TOOLS, *food, weapon)))
        self.drawn += 1
        return record


def parse_levels(text):
    """A profile of start levels (`--start-levels woodcutting=50,firemaking=52`):
    {stat: level} in rl/obs.py SKILL_ORDER's order, the stats named by the engine's names in lower case. Refuses an
    unknown or repeated name (a native reset skips a name it does not know: env-ts/engine/agent.ts setStats), a level
    that is not a whole number of 1 to 99, hitpoints under 10, and an empty profile."""
    names = [name.lower() for name in SKILL_ORDER]
    pairs = [pair.split('=') for pair in str(text).split(',')] if text else []
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError(f'start levels are name=level pairs joined by commas: {text!r}')
    levels = {}
    for name, level in pairs:
        name = name.strip().lower()
        if name not in names:
            raise ValueError(f'unknown stat {name!r}: the stats are {", ".join(names)}')
        if name in levels:
            raise ValueError(f'stat {name!r} named twice')
        if not level.strip().isdigit() or not 1 <= int(level) <= 99 or (name == 'hitpoints' and int(level) < 10):
            raise ValueError(f'{name} level {level.strip()!r}: a whole number of 1 to 99, hitpoints 10 or more')
        levels[name] = int(level)
    return {name: levels[name] for name in names if name in levels}


def profiled(record, levels):
    """`record` with the stats of `levels` (parse_levels) at those levels, or `record` itself when `levels` is None. Every
    other draw is the record's own: the tile (its reset seed), the weapon, the food, the worn draw and the levels of every
    stat not named (Records' stream is not touched). Its hitpoints at the arm (hp_drawn) are the new hitpoints level less
    the record's own deficit (its drawn level less its drawn hitpoints), never under 1. The record names the profile and
    keeps its drawn levels and hitpoints (drawn_levels, drawn_hp)."""
    if levels is None:
        return record
    new = {**record['levels'], **levels}  # drawn stats in START_LEVELS' order, then the profile's others
    deficit = record['levels']['hitpoints'] - record['hp_drawn']
    return dict(record, levels=new, hp_drawn=max(1, new['hitpoints'] - deficit), profile=dict(levels),
                drawn_levels=record['levels'], drawn_hp=record['hp_drawn'],
                set_stats=','.join(f'{stat}:{level}' for stat, level in new.items()))


def life_seed(seed, world, life):
    """The actor's seed for life `life` of world `world` in a run of seed `seed`: one 64-bit word."""
    return int(np.random.SeedSequence([seed, world, life]).generate_state(1, np.uint64)[0])


def site_order(sites, rotation):
    """The site each world plays: world k plays sites[(k + rotation) mod len(sites)], or every world the one site named
    (a single-site watch, such as the mining script's: each world keeps its own record stream)."""
    sites = list(sites)
    if len(sites) not in (1, WATCH['worlds']) or len(set(sites)) != len(sites) or type(rotation) is not int or rotation < 0:
        raise ValueError(f"one site or {WATCH['worlds']} distinct sites, and a rotation of 0 or more: {sites}, {rotation!r}")
    return [sites[(k + rotation) % len(sites)] for k in range(WATCH['worlds'])]


def kit_items(record, extra=()):
    """The declared start's items: the record's seed_items, then `extra` (item, count) pairs (free-play-v3's tinderbox;
    none for free-play-v1), in baseline/world.py's seedItems form."""
    return record['seed_items'] + ''.join(f',{name}:{count}' for name, count in extra)


def life_cfg(tile, record, extra=(), items=ITEM_V2, hp=None):
    """The native reset's settings of a life: free_play_cfg (TASK_CFG with limits that never cut a life of up to
    MAX_HORIZON ticks), the site's tile and the record's declared start (no starter kit, these items and levels), with
    `extra` items beside the record's (kit_items), under the item contract `items` (TASK_CFG's item-v2 unless a version
    names another: free-play-v10's item-v3, the page of kinds), and with `hp` the life's current hitpoints after its
    levels (startHp: free-play-v12's to v14's drawn hitpoints; no field otherwise, so the other cfgs are as before)."""
    return dict(free_play_cfg(), spawnTele=tele(*tile), kit='none', seedItems=kit_items(record, extra),
                setStats=record['set_stats'], **({} if items == ITEM_V2 else dict(contract=items)),
                **({} if hp is None else dict(startHp=int(hp))))


def setup(record, extra=()):
    """The setup before arm (ResourceEpisode's prepare): prepare_declared (the declared inventory, the extra items with
    it, and the levels, then the style step), then a Wield of the weapon when the record starts it worn, as sim/task.py
    _setup does."""
    from baseline.scripted import KIT_NAMES, ItemGate, prepare_declared

    def prepare(transport, obs, write):
        obs = prepare_declared(transport, obs, write, kit_items(record, extra), record['set_stats'])
        if record['worn']:
            name = KIT_NAMES[record['weapon']]
            slot = next((k for k, h in enumerate(obs['heldSel']) if h['name'] == name and h['option'] == 'Wield'), None)
            if slot is None:
                raise ValueError(f'no Wield offered for {name} at setup')
            gate = ItemGate(transport, write)
            gate.accept(obs)
            obs = gate.step(page_row(A_HELD0, slot))  # pack option `slot`'s row: A_HELD0 + slot, or act-v4's 96.. past 8
            write('setup', action=page_row(A_HELD0, slot), observation=obs)
        return obs
    return prepare


# ---- the clue and loot drops the simulator excludes -------------------------------------------------------------------
def excluded_items(tables, world):
    """{object id: exclusion} of every object that world `world` leaves out (sim/engine.py EXCLUSIONS): treasure-trail
    objects ('clue_drops') and its loot outside the slice ('loot_drops': world_rules), by the simulator's own rules."""
    from sim.engine import loot_outside, trail_categories
    trail, names = trail_categories(tables), tuple(world_rules(world)['loot_outside'])
    return {int(obj): 'clue_drops' if o.get('category') in trail else 'loot_drops'
            for obj, o in sorted(tables.obj_types.items()) if o.get('category') in trail or loot_outside(tables, obj, names)}


# ---- one world's lives -----------------------------------------------------------------------------------------------
class WatchEpisode(ResourceEpisode):
    """ResourceEpisode's native protocol (fresh incarnation each reset, the armed capture, the task clock) with obs-v6,
    or obs-v7 in an item-v4 life (free-play-v11): each observation is projected by its stamp (rl/wide_contract.py
    project_free), with no budget, goal or critic input. Nothing is paid and nothing is scored here: at the terminal the
    trace is read and released, and the monitors read it from the evidence. abandon() retires a capture before its
    terminal (a guard cut or an excluded pickup), reads and releases it. `trace` is the life's read trace (None before it
    is read).

    The ORACLE constructor checks act-v3's targets under the cfg it is given, which is free_play_cfg() (item-v2) for the
    episode of a world (play_world); each life then sets its own cfg (`cfg`, life_cfg), which its reset, its stamp checks
    and its records use. An item-v4 life's start (prepare_start) repeats the ORACLE method's steps with obs-v7's
    projection, since the ORACLE method's projection (rl/obs.py project) takes item-v2 and item-v3 alone; every other
    life calls the ORACLE method as it is."""
    trace = None

    def reset(self, seed):
        self.trace = None
        out = super().reset(seed)
        self._stamp()
        return out

    def prepare_start(self, seed):
        """baseline/resource_task.py ResourceEpisode.prepare_start (its digest: sim/tests/test_watch.py
        PREPARE_START_SOURCE), with project_v7 for an item-v4 life; the ORACLE method itself for every other."""
        if self.cfg.get('contract') != ITEM_V4:
            return super().prepare_start(seed)
        need(not self.failed and not self.closed and not self.active and not self.bound, 'invalid resource reset lifecycle')
        try:
            self.raw = self.transport.reset(seed=seed, cfg=deepcopy(self.cfg))
            self.record('reset', seed=seed, cfg=self.cfg, scoring=self.scoring, goal=self.goal, observation=self.raw)
            if self.prepare is not None:
                self.raw = self.prepare(self.transport, self.raw, self.record)
            need(not self.raw.get('done') and not self.raw.get('truncated') and not self.raw.get('optionBusy')
                 and not self.raw.get('modalOpen'), 'setup did not yield a live idle public observation')
            self.raw = episode_start(self.raw)
            self.record('episode_start', observation=self.raw)
            self._stamp()  # item-v4 stamped, before obs-v7's projection refuses anything else
            project_v7(self.raw)
            self.prepared = True
        except Exception:
            self.failed = True
            raise

    def _stamp(self):
        """Each observation stamps the item contract its life's reset named (TASK_CFG's item-v2, or the version's)."""
        need(self.raw.get('contractVersion') == self.cfg.get('contract'),
             f"an observation of the item contract {self.raw.get('contractVersion')!r}, the life's is "
             f"{self.cfg.get('contract')!r}")

    def _budget(self):
        self.clock.accept(self._control('budget', identity=self.clock.identity))
        self._stamp()
        self.projected = project_free(self.raw)
        return self.projected

    def step(self, action):
        need(not self.failed and not self.closed and self.active, 'invalid resource step lifecycle')
        try:
            need(type(action) is int and 0 <= action < len(self.projected['mask']) and self.projected['mask'][action],
                 'invalid/masked policy action')
            self.decisions += 1
            self._budget()  # inference time consumes native ticks too: no dispatch after the horizon
            executed = None
            if self.clock.last_status == 'active':
                self.record('requested', action=action)
                self.raw = self.transport.step(action)
                executed = action
                self.record('observed', action=action, observation=self.raw)
                self._budget()
            self.record('task_transition', requested_action=action, executed_action=executed,
                        native_tick=self.clock.last_tick, status=self.clock.last_status)
            if self.clock.last_status == 'active':
                need(not self.raw.get('done') and not self.raw.get('truncated'), 'transport ended before native task closure')
                return self.projected, False
            trace = self._control('read', identity=self.clock.identity)['audit']
            need(trace.get('terminal') and trace['terminal']['reason'] == self.clock.last_status
                 and trace['startTick'] == self.clock.start and trace['observedTick'] == self.clock.last_tick,
                 'terminal trace and task clock disagree')
            self.trace = trace
            need(self._control('release', identity=self.clock.identity)['audit'] == trace, 'released trace changed')
            self.bound = self.active = False
            return self.projected, True
        except Exception:
            self.failed = True
            raise

    def abandon(self, cause):
        need(self.active and self.bound and not self.failed, 'abandon needs an armed life')
        try:
            self.record('abandoned', cause=cause, native_tick=self.clock.last_tick)
            self._control('retire', identity=self.clock.identity)
            self.trace = trace = self._control('read', identity=self.clock.identity)['audit']
            need(self._control('release', identity=self.clock.identity)['audit'] == trace, 'released trace changed')
            self.bound = self.active = False
        except Exception:
            self.failed = True
            raise


RETIRED = 'capture retired before complete horizon'  # env-ts resource-capture.ts retire(): a life ended on purpose


def capture_failure(trace):
    """The capture failures of a life's read trace as one text, or None. The resource capture's failure counts unless it
    is the one a retire writes (a guard cut or a pickup ends the life on purpose; the events before it stand). The combat
    capture's counts always: its events stop at its first failure (env-ts combat-capture.ts live()), an overflow past
    its event bound included, so its kills, damage and fights are a prefix of the life's. A trace with no combat
    evidence counts too."""
    if trace is None:
        return 'no trace was read'
    out = [f"resource capture: {trace['failure']}"] if trace.get('failure') not in (None, RETIRED) else []
    combat = trace.get('combat')
    if not isinstance(combat, dict):
        out.append('combat capture: no combat evidence')
    elif combat.get('failure') is not None:
        out.append(f"combat capture: {combat['failure']}")
    return '; '.join(out) or None


def excluded_in_view(raw, excluded):
    """The excluded objects a native observation shows (sorted ids): those in its ground list, the six nearest ground
    items within 15 tiles (env-ts/engine/observe.ts N_GROUND_OBS), and those its act affordances offer to Take, among
    the 16 nearest targets it publishes. A drop farther down both lists is not seen until it is among them."""
    return sorted({g.get('id') for g in raw.get('ground') or [] if g.get('id') in excluded} |
                  {a.get('id') for a in raw.get('actSel') or [] if a.get('kind') == 'obj' and a.get('id') in excluded})


class Timed:
    """The resource control with each call's wall seconds, by operation (for the rehearsal's read and release times)."""

    def __init__(self, control):
        self.control, self.calls = control, []

    def __call__(self, operation, **kwargs):
        start = time.monotonic()
        try:
            return self.control(operation, **kwargs)
        finally:
            self.calls.append((operation, time.monotonic() - start))

    def take(self):
        out = {}
        for operation, seconds in self.calls:
            c = out.setdefault(operation, dict(count=0, total=0.0, max=0.0))
            c['count'], c['total'], c['max'] = c['count'] + 1, c['total'] + seconds, max(c['max'], seconds)
        self.calls = []
        return {k: dict(v, total=round(v['total'], 4), max=round(v['max'], 4)) for k, v in out.items()}


class Locked:
    """A torch actor (sim/hour.py FreePolicyActor) behind the run's one policy lock, with its own torch generator state:
    reset seeds it, and each act restores it first and keeps it after, so a life's draws depend on its seed and its
    observations only, whatever the other threads do."""

    def __init__(self, actor, lock):
        import torch
        self.torch, self.actor, self.lock, self.state = torch, actor, lock, None

    def reset(self, seed):
        with self.lock:
            self.actor.reset(seed)
            self.state = self.torch.get_rng_state()

    def act(self, observation, task=None):
        with self.lock:
            self.torch.set_rng_state(self.state)
            try:
                return self.actor.act(observation, task)
            finally:
                self.state = self.torch.get_rng_state()


def play_world(k, transport, service, site, tile, records, actor, seed, ticks, deadline, path, excluded, extra=(),
               items=ITEM_V2, start_hp=False, start_levels=None):
    """World k's lives (module docstring) until it has spent `ticks` armed ticks, the guard (`deadline`, monotonic) or a
    failure stops it. Every row goes to `path` (one JSONL file) with its life. Returns the world's result, whatever
    fails: its lives (record, seed, horizon, status, reason, cause, capture failure, excluded drop seen, ticks, decisions,
    wall seconds), its armed ticks (every life's, a failed one's to its last clock tick) and its status: complete (every
    tick spent), cut (the guard) or stopped (a failure, with its cause, or more than WATCH['max_lives'] lives).

    A life ends complete (its terminal: reason 'death' or 'horizon'), cut, or incomplete: an excluded pickup (reason
    'excluded pickup'; reported apart, its world goes on), a capture failure in its read trace (capture_failure, the
    terminal's reason kept; its world goes on) or a failure (reason 'failure'; the world stops). With `start_levels`
    (parse_levels) each record is played with those levels (profiled); its watch_life row holds the profiled record."""
    from baseline.world import JsonlEvidence
    result = dict(world=k, site=site, tile=list(tile), lives=[], status='running', armed_ticks=0)
    state, started, evidence, episode = dict(life=None), time.monotonic(), None, None

    def write(kind, **data):
        evidence(kind, life=state['life'], **data)
    try:
        evidence = JsonlEvidence(path)
        timed = Timed(service)
        if hasattr(service, 'record'):  # baseline/world.py ResourceService: its status rows go to the same file
            service.record = lambda kind, **data: write(kind, world=k, **data)
        episode = WatchEpisode(transport, timed, free_play_cfg(), 1, 1.0, write)
        while result['armed_ticks'] < ticks and result['status'] == 'running':
            if time.monotonic() >= deadline:
                result.update(status='cut', cause='the guard expired before the world spent its ticks')
                break
            if len(result['lives']) >= WATCH['max_lives']:
                result.update(status='stopped', cause=f"more than {WATCH['max_lives']} lives")
                break
            record, life = profiled(records.next(), start_levels), len(result['lives'])
            state['life'] = life
            row = dict(life=life, record=record['index'], seed=life_seed(seed, k, life), horizon=ticks - result['armed_ticks'],
                       status='running', reason=None, cause=None, capture_failure=None, excluded_seen=None,
                       wall=round(time.monotonic() - started, 3))
            result['lives'].append(row)
            write('watch_life', world=k, site=site, tile=list(tile), record=record, seed=row['seed'], horizon=row['horizon'])
            episode.horizon, episode.cfg, episode.prepare = row['horizon'], life_cfg(
                tile, record, extra, items, hp=record['hp_drawn'] if start_hp else None), \
                setup(record, extra)
            before, t0, failed = episode.clock, time.monotonic(), True

            def flag():  # the first observation of the life that shows an excluded drop
                seen = excluded_in_view(episode.raw, excluded) if row['excluded_seen'] is None else None
                if seen:
                    row['excluded_seen'] = episode.clock.last_tick - episode.clock.start
                    write('excluded_seen', tick=row['excluded_seen'], items=seen)
            try:
                obs = episode.reset(record['reset_seed'])
                actor.reset(row['seed'])
                flag()  # the arm observation, tick 0
                while True:
                    if time.monotonic() >= deadline:
                        episode.abandon('guard')
                        row.update(status='cut', reason='guard')
                        result.update(status='cut', cause='the guard expired')
                        break
                    obs, done = episode.step(int(actor.act(obs, episode)))
                    flag()
                    held = sorted({i['id'] for i in episode.raw.get('inventory') or []} & set(excluded))
                    if held:  # a pickup of an excluded drop ends the life as incomplete
                        if not done:
                            episode.abandon('excluded pickup')
                        row.update(status='incomplete', reason='excluded pickup', items=held)
                        break
                    if done:
                        row.update(status='complete', reason=episode.clock.last_status)
                        break
                row['capture_failure'] = capture_failure(episode.trace)
                if row['capture_failure'] and row['status'] == 'complete':
                    row['status'] = 'incomplete'  # its reason stays the terminal's: a death or the horizon
                failed = False
            except Exception as error:
                row.update(status='incomplete', reason='failure', cause=f'{type(error).__name__}: {error}')
                raise
            finally:
                armed = episode.clock is not None and episode.clock is not before  # this life's clock: it was armed
                row.update(ticks=episode.clock.last_tick - episode.clock.start if armed else 0,
                           decisions=episode.decisions if armed else 0, seconds=round(time.monotonic() - t0, 3))
                result['armed_ticks'] += row['ticks']
                try:
                    write('watch_life_end', **{key: v for key, v in row.items() if key != 'life'}, controls=timed.take())
                except Exception:
                    if not failed:
                        raise
        if result['status'] == 'running':
            result['status'] = 'complete'
    except Exception as error:
        result.update(status='stopped', cause=f'{type(error).__name__}: {error}')
        if evidence is not None:
            try:
                write('watch_failure', cause=result['cause'])
            except Exception:
                pass
    finally:
        try:
            if episode is not None:
                episode.close()  # retires a capture still bound, then leaves the world
            else:
                transport.close()
        except Exception as error:
            result['close_error'] = f'{type(error).__name__}: {error}'
        if evidence is not None:
            try:
                evidence.close()
            except Exception as error:
                result.update(status='stopped', cause=result.get('cause') or f'the evidence did not close: {error}')
        samples = getattr(service, 'samples', [])  # the world's status rows: None when none was read
        result.update(seconds=round(time.monotonic() - started, 3),
                      lockstep=all(s.get('lockstep') is True for s in samples) if samples else None)
    return result


# ---- the run ---------------------------------------------------------------------------------------------------------
def make_actor(name, models, greedy, lock):
    """A fresh actor of `name` for one world: WAIT, random legal play, the mining script (sim/mine_script.py) or a loaded
    policy (`initial` or `policy`)."""
    from sim import free
    if name == 'wait':
        return free.Wait()
    if name == 'random':
        return free.Random()
    if name == SCRIPT_ACTOR:
        from sim.mine_script import MineScript
        return MineScript()
    from sim.hour import FreePolicyActor
    return Locked(FreePolicyActor(models[name], deterministic=greedy), lock)


def env_ts(commit):
    """The last commit at or before `commit` that changed env-ts/ (the engine wrapper whose observations a policy reads),
    and the env-ts files changed in the working tree when `commit` is HEAD: a local git read, None where git cannot read
    the commit here."""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    git = lambda *args: subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=True).stdout
    try:
        last = git('log', '-1', '--format=%H', commit, '--', 'env-ts').strip() or None
        changed = sorted(line[3:] for line in git('status', '--porcelain', '--', 'env-ts').splitlines()) \
            if commit == 'HEAD' else None
    except (subprocess.CalledProcessError, OSError):
        return None
    return dict(commit=last, changed=changed) if commit == 'HEAD' else last


# The free-play versions a checkpoint names in its saved free_play_task (each defined with the trainer's task,
# sim/workers.py), and what each saves beside its name: free-play-v2 (the death rule of 18 November 2026) its
# death_price, free-play-v3 (every stat paid) and every later version its paid_skills and kit_extra (the tinderbox),
# free-play-v15 those and its inner_pay; a free-play-v1 checkpoint saves none of these. The watch pays nothing: it
# records what the checkpoint saved and plays a v3 or later checkpoint with its kit_extra. A checkpoint of a removed
# study arm (sim/workers.py REMOVED) is refused by name.
FREE_PLAY_V2_NAME, FREE_PLAY_V3_NAME, FREE_PLAY_V10_NAME = 'free-play-v2', 'free-play-v3', 'free-play-v10'
FREE_PLAY_V11_NAME, FREE_PLAY_V12_NAME = 'free-play-v11', 'free-play-v12'
# free-play-v14 (v12 with the start rule old-life-v1, sim/workers.py FREE_PLAY_V14) saves what v12 saves; a watch plays
# its records as a v12 checkpoint's, every life as drawn, or every life with the profile of --start-levels
FREE_PLAY_V14_NAME = 'free-play-v14'
# free-play-v15 (v14 with the pay rare-xp-v2) saves v14's and its inner_pay; a watch pays nothing and plays it as v14
FREE_PLAY_V15_NAME = 'free-play-v15'
VERSION_SAVES = {FREE_PLAY_V2_NAME: ('death_price',), FREE_PLAY_V3_NAME: ('paid_skills', 'kit_extra'),
                 FREE_PLAY_V10_NAME: ('paid_skills', 'kit_extra'), FREE_PLAY_V11_NAME: ('paid_skills', 'kit_extra'),
                 FREE_PLAY_V12_NAME: ('paid_skills', 'kit_extra'),
                 FREE_PLAY_V14_NAME: ('paid_skills', 'kit_extra'),
                 FREE_PLAY_V15_NAME: ('paid_skills', 'kit_extra', 'inner_pay')}
# the versions that paid every stat: their monitors add each gain
EVERY_STAT = (FREE_PLAY_V3_NAME, FREE_PLAY_V10_NAME, FREE_PLAY_V11_NAME, FREE_PLAY_V12_NAME, FREE_PLAY_V14_NAME,
              FREE_PLAY_V15_NAME)


def run_items(task):
    """The item contract a checkpoint of free-play version `task` plays under (sim/workers.py item_contract): item-v2 for
    none, free-play-v1 or an unknown name (which real_run refuses), item-v3 for free-play-v10, item-v4 for v11."""
    return item_contract(task) if task in FREE_PLAY_VERSIONS else ITEM_V2


def real_run(checkpoint, model, facts, launch, world):
    """facts['run'] of a checkpoint of a real-engine run (sim.train --env real), which has no summary.json
    while it is incomplete: the checks are the checkpoint's own (load_free_policy: obs-v6, the contract versions, the
    name table; a free-play checkpoint), and `launch`, the training run's launch.json when given, must be a free-play-v1
    run of the real engine at the trial's XP rate whose lives started in `world` (--batch-setting starts), and its start
    weights (--weights) and batch settings are recorded. The checkpoint holds no engine of its own: without `launch`,
    that it trained in the real engine is the --real-checkpoint declaration. env_ts: the env-ts commit the policy trained
    with (env_ts of the launch's source commit, with the env-ts files that commit's tree had changed) and the one this
    watch plays with (env_ts of HEAD). A checkpoint of a later version (VERSION_SAVES) plays under the same checks as
    a free-play-v1 one, its version and what it saved beside it recorded (a free-play-v1 checkpoint's record is as
    before, with none of these), and a launch must be of the checkpoint's own version. A free-play-v15 checkpoint plays
    as a v14 one, with no inner pay, which its record says; one of a removed study arm is refused by name."""
    from baseline.telemetry import sha256
    from sim.train import batch_settings
    if not facts['free_play']:
        raise ValueError(f'{checkpoint} is no free-play checkpoint')
    named = getattr(model, 'free_play_task', None)
    refuse_removed(named, f'the checkpoint {checkpoint}')
    saved = {k: getattr(model, k, None) for keys in VERSION_SAVES.values() for k in keys}
    wanted = VERSION_SAVES.get(named, ())
    if (named is not None and named not in VERSION_SAVES) or any((v is None) == (k in wanted) for k, v in saved.items()):
        raise ValueError(f'{checkpoint} names the task {named} and saves {saved}: a free-play-v1 checkpoint saves none of '
                         f'these, the others {VERSION_SAVES}')
    task = named or FREE_PLAY['version']
    out = dict(engine='real', checkpoint=str(checkpoint), decisions=int(model.num_timesteps), launch=None, start=None,
               **({} if named is None else dict(task=task, **{k: saved[k] for k in wanted})),
               **({} if 'inner_pay' not in wanted else dict(inner_pay_in_watch='none: a watch pays nothing; the policy '
                                                             'plays with its kit, as a free-play-v14 one')),
               env_ts=dict(trained=None, trained_changed=None, played=env_ts('HEAD')),
               source='the checkpoint (no summary.json: a real-engine run, declared by --real-checkpoint)')
    if launch is not None:
        record = json.loads(Path(launch).read_text())
        command = record.get('command') or []
        option = lambda name: command[command.index(name) + 1] if name in command[:-1] else None
        rest = command[command.index('--batch-setting') + 1:] if '--batch-setting' in command else []
        try:
            settings = batch_settings(rest[:next((i for i, c in enumerate(rest) if c.startswith('--')), len(rest))]) or {}
        except SystemExit as error:
            raise ValueError(f'{launch}: {error}')
        if option('--env') != 'real' or option('--task') != task or record.get('xp_rate') != WATCH['xp_rate'] \
                or settings.get('starts') != world:
            raise ValueError(f"{launch} is no {task} run (the checkpoint's version) of the real engine at XP rate "
                             f"{WATCH['xp_rate']} in {world}: --env {option('--env')}, --task {option('--task')}, XP rate "
                             f"{record.get('xp_rate')}, starts {settings.get('starts')}")
        trained = (record.get('source') or {}).get('commit')
        out['env_ts'].update(trained=None if trained is None else env_ts(trained), trained_changed=sorted(
            f for k in ('changed', 'untracked') for f in (record.get('source') or {}).get(k) or [] if f.startswith('env-ts/')))
        out.update(launch=dict(path=str(launch), sha256=sha256(launch), name=record.get('name'), command=command,
                               source=record.get('source')),
                   start=option('--weights'), settings=settings, source='the checkpoint and the training run\'s launch.json')
    return out


def load_actor(actor, run, policy_seed, checkpoint, world, real_checkpoint=None, launch=None):
    """(models, facts) of a policy actor: the run's saved initialization and its checkpoint (RUN/latest.zip unless
    `checkpoint`), each by load_free_policy (obs-v6 only) and tied to the run by sim/hour.py policy_run (a free-play-v1
    run of `policy_seed`, completed, that trained in `world`: no muggers, the trial's XP rate); or, `real_checkpoint`, one
    checkpoint of a real-engine run played as `policy` under its own checks (real_run, with the run's launch.json when
    given). Controls load nothing."""
    if actor in ('wait', 'random', SCRIPT_ACTOR):
        if run is not None or policy_seed is not None or checkpoint is not None or real_checkpoint is not None \
                or launch is not None:
            raise ValueError(f'{actor} plays no policy: name no run, seed, checkpoint or launch')
        return {}, {}
    from sim.hour import load_free_policy, policy_run
    if real_checkpoint is not None:
        if actor != 'policy' or run is not None or policy_seed is not None or checkpoint is not None:
            raise ValueError('a real-engine checkpoint plays as the policy actor, alone: no --run, --policy-seed or '
                             '--checkpoint beside it')
        from baseline.policy import run_contract
        import zipfile
        with zipfile.ZipFile(real_checkpoint) as archive:  # its version names its item contract, which it must hold
            named = json.loads(archive.read('data').decode()).get('free_play_task')
        model, facts = load_free_policy(real_checkpoint, contract=run_contract(run_items(named)))
        facts = dict(policy=facts)
        facts['run'] = real_run(Path(real_checkpoint), model, facts['policy'], launch, world)
        return {'policy': model}, facts
    if launch is not None:
        raise ValueError('--launch goes with --real-checkpoint')
    if run is None or policy_seed is None:
        raise ValueError(f'{actor} plays a free-play run\'s policy: name its run folder and seed')
    run = Path(run)
    checkpoint, initial = Path(checkpoint or run / 'latest.zip'), run / 'initial.zip'
    models, facts = {}, {}
    for name, path in (('policy', checkpoint), ('initial', initial)):
        models[name], facts[name] = load_free_policy(path)
        if getattr(models[name], 'free_play_task', None) is not None:  # a simulator run trains free-play-v1 alone
            raise ValueError(f"{path} is a {models[name].free_play_task} checkpoint: a simulator run's policy is "
                             f"{FREE_PLAY['version']}'s (a real-engine checkpoint plays by --real-checkpoint)")
    facts['run'] = policy_run(checkpoint, initial, facts, policy_seed, False, WATCH['xp_rate'], world=world)
    return {actor: models[actor]}, facts


def check_settings(*, output, world, actor, sites, rotation, key, seed, ticks, guard, run=None, policy_seed=None,
                   checkpoint=None, tables=None, real_checkpoint=None, launch=None, start_hp=None,
                   start_levels=None):
    """Every setting of a run that needs no native world, checked (`check` makes these before scripts/stack.py starts
    any; watch_run makes them again before it joins a world), and what the run plays from them: the world's rules
    (world_rules: a world named with no default), the site order, the actor's models and facts (load_actor: a policy's
    run must have trained in the world), the world's archive (sim/free.py world_tables, which must exist, recorded with
    path and sha256; `tables`, a test's stand-in, is used as given and recorded as None) and its excluded objects.
    Refuses an output that exists (a run writes a new folder), an actor outside WATCH's, a site outside the candidates,
    ticks outside 1..MAX_HORIZON, no guard, and a record key or run seed that is not a whole number of 0 or more. With
    `start_levels` (the text of --start-levels, parse_levels) the run plays a profile: its output folder's name must not
    end in 'final', the name of a fixed-form watch. The mining script plays MINE_SITES alone."""
    from baseline.resource_task import MAX_HORIZON
    from baseline.telemetry import sha256
    from sim.combat import check_village
    rules = world_rules(world)
    if Path(output).exists():
        raise ValueError(f'{output} exists: a run writes a new folder')
    if actor not in WATCH['actors']:
        raise ValueError(f"an actor of {WATCH['actors']}, not {actor!r}")
    order = site_order(sites, rotation)
    unknown = sorted(set(sites) - set(SITES))
    if unknown:
        raise ValueError(f'sites outside the candidates (SITES): {unknown}')
    if actor == SCRIPT_ACTOR and set(sites) - set(MINE_SITES):
        raise ValueError(f'{SCRIPT_ACTOR} plays {", ".join(MINE_SITES)} alone, a rock of its table in reach: not {sorted(sites)}')
    levels = None if start_levels is None else parse_levels(start_levels)
    if levels is not None and Path(output).name.lower().endswith('final'):
        raise ValueError(f"{Path(output).name}: a profile watch's folder is named as its own, not as a fixed-form watch "
                         "('...final')")
    if type(ticks) is not int or not 1 <= ticks <= MAX_HORIZON or not (isinstance(guard, (int, float)) and guard > 0):
        raise ValueError(f'ticks in 1..{MAX_HORIZON} and a positive guard: {ticks!r}, {guard!r}')
    for value in (key, seed):
        if type(value) is not int or value < 0:
            raise ValueError(f'the record key and the run seed are whole numbers of 0 or more (no default): {key!r}, {seed!r}')
    archive = None
    if tables is None:
        from sim.free import world_tables
        archive = world_tables(world)
        if not archive.exists():
            raise ValueError(f'{archive} is missing: {world} plays over it (sim/build.py says how each archive is made)')
    models, facts = load_actor(actor, run, policy_seed, checkpoint, world, real_checkpoint, launch)
    if archive is not None:
        from sim.data import Tables
        tables = Tables.load(archive)
    check_village(village(world), tables, False)  # an archive without the content the world's scripts read
    return dict(rules=rules, order=order, models=models, facts=facts, excluded=excluded_items(tables, world),
                tables=None if archive is None else dict(path=str(archive), sha256=sha256(archive)),
                start_hp=start_rule(actor, facts, start_hp), start_levels=levels)


def start_rule(actor, facts, given):
    """Whether each life starts at its record's drawn hitpoints (run.json hp_applied): a policy's checkpoint's own rule
    (free-play-v12 to v14: drawn), which `given` (--start-hp) may name again but never change; for a control actor
    (wait, random, the mining script) `given`, so that a control beside a v12 to v14 policy starts as it does; full
    hitpoints otherwise."""
    if given not in (None, 'drawn'):
        raise ValueError(f"--start-hp is 'drawn' or not given, not {given!r}")
    task = (facts.get('run') or {}).get('task')
    own = FREE_PLAY_VERSIONS.get(task, {}).get('start_hp') == 'drawn'
    if actor in ('initial', 'policy'):
        if given is not None and not own:
            raise ValueError(f"--start-hp drawn: the checkpoint's version {task or FREE_PLAY['version']} starts every "
                             'life at full hitpoints; a policy plays under its own rule')
        return own
    return given == 'drawn'


def checkout_tree():
    """The tree of the engine checkout that scripts/stack.py boots (baseline/evaluate.py engine_tree, a local git read),
    or the watch's refusal, not a traceback, when vendor/rs-sdk is no git checkout (git fails) or git cannot run."""
    import subprocess
    from baseline.evaluate import engine_tree
    try:
        return engine_tree()
    except (subprocess.CalledProcessError, OSError) as error:
        raise ValueError(f"no engine checkout at vendor/rs-sdk ({type(error).__name__}): the trial plays the "
                         f"{WATCH['engine']} build") from error


def check_engine(tree):
    """Refuse an engine checkout (baseline/evaluate.py engine_tree, a local git read of the vendor checkout that
    scripts/stack.py boots) other than the trial's build. `check` makes it before the stack; check_stack again."""
    if tree != config.ENGINE_TREES[WATCH['engine']]:
        raise ValueError(f"the engine checkout ({tree}) is not the {WATCH['engine']} build")


def check_stack(environ, tree, worlds):
    """Refuse a stack outside the trial's: random events on, another XP rate (both the stack's own environment),
    another engine build (check_engine) or another number of worlds."""
    if environ.get('NODE_RANDOM_EVENTS') != 'false':
        raise ValueError('launch the stack with NODE_RANDOM_EVENTS=false (the declared exclusion of random events)')
    if config.XP_RATE != WATCH['xp_rate']:
        raise ValueError(f"the trial plays at XP rate {WATCH['xp_rate']} (scripts/stack.py --xp-rate 1), not {config.XP_RATE}")
    check_engine(tree)
    if len(worlds) != WATCH['worlds']:
        raise ValueError(f"the trial plays {WATCH['worlds']} worlds (scripts/stack.py --worlds {WATCH['worlds']}), not {len(worlds)}")


def watch_run(output, *, world, actor, sites, rotation, key, seed, ticks=WATCH['ticks'], guard=WATCH['guard'], run=None,
              policy_seed=None, checkpoint=None, greedy=False, rehearsal=False, worlds=None, join=None, tables=None,
              environ=None, tree=None, real_checkpoint=None, launch=None, start_hp=None, start_levels=None):
    """One run of native-watch-v1 (module docstring) in the stack's worlds, under the rules of `world` (a name of
    sim/combat.py VILLAGES, no default): run.json (the manifest with the world, its rules and its archive, each world's
    lives and status) and world-K.jsonl.gz. Every setting is checked (check_settings, check_stack) before a world is
    joined. Returns the run's result."""
    from baseline.telemetry import source
    from baseline.world import join as join_world, worlds as stack_worlds
    plays = check_settings(output=output, world=world, actor=actor, sites=sites, rotation=rotation, key=key, seed=seed,
                           ticks=ticks, guard=guard, run=run, policy_seed=policy_seed, checkpoint=checkpoint, tables=tables,
                           real_checkpoint=real_checkpoint, launch=launch, start_hp=start_hp,
                           start_levels=start_levels)
    if worlds is None:
        try:
            worlds = stack_worlds()
        except RuntimeError as error:  # not inside scripts/stack.py
            raise ValueError(str(error))
    check_stack(os.environ if environ is None else environ, checkout_tree() if tree is None else tree, worlds)
    order, models, facts, excluded = plays['order'], plays['models'], plays['facts'], plays['excluded']
    start_hp = plays['start_hp']  # free-play-v12, v14, v15, or a control told so: each life at its drawn hitpoints
    kit = tuple(tuple(x) for x in (facts.get('run') or {}).get('kit_extra') or ())  # a free-play-v3 policy's own kit
    items = run_items((facts.get('run') or {}).get('task'))  # the item contract of the checkpoint's version
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(version=WATCH['version'], watch=WATCH, world=world, rules=plays['rules'], rehearsal=rehearsal,
                  actor=actor, greedy=greedy,
                  sampled=actor in ('initial', 'policy') and not greedy,
                  policy=facts or None, policy_seed=policy_seed,
                  sites={name: list(SITES[name]) for name in sites}, rotation=rotation, order=order, key=key, seed=seed,
                  ticks=ticks, guard=guard, xp_rate=config.XP_RATE,
                  random_events=(os.environ if environ is None else environ).get('NODE_RANDOM_EVENTS'),
                  engine=dict(name=WATCH['engine'], tree=config.ENGINE_TREES[WATCH['engine']]),
                  worlds=[dict(w) for w in worlds], hp_applied=start_hp, cfg=free_play_cfg(),
                  excluded=dict(count=len(excluded), ids=sorted(excluded)), tables=plays['tables'], source=source(output),
                  status='running')
    if plays['start_levels'] is not None:  # a profile watch (parse_levels): no key in a run that plays the records as drawn
        result['start_levels'] = plays['start_levels']
    if actor == SCRIPT_ACTOR:
        from sim.mine_script import MINE_SCRIPT
        result['script'] = dict(MINE_SCRIPT, rocks={f'{x},{z}': list(v) for (x, z), v in MINE_ROCKS.items()})

    def save():
        (output / 'run.tmp').write_text(json.dumps(result, indent=1, default=str) + '\n', encoding='utf-8')
        os.replace(output / 'run.tmp', output / 'run.json')
    save()
    join = join or join_world
    links, lock, results = [], threading.Lock(), [None] * len(worlds)
    try:
        for k, world in enumerate(worlds):  # before the guard starts: each world is booted and ready
            links.append(join(world, lambda kind, **data: None, k))
    except Exception as error:
        for transport, _ in links:
            transport.close()
        result.update(status='stopped', cause=f'a world could not be joined: {type(error).__name__}: {error}')
        save()
        return result
    if models:
        import torch
        result['torch_threads'], threads = 1, torch.get_num_threads()
        torch.set_num_threads(1)
    started = time.monotonic()
    deadline = started + guard

    def play(k):  # a world's result, whatever fails: 'hung' only for a thread still running after the grace
        transport, service = links[k]
        stopped = dict(world=k, site=order[k], tile=list(SITES[order[k]]), lives=[], status='stopped', armed_ticks=0)
        try:
            player = make_actor(actor, models, greedy, lock)
        except Exception as error:
            results[k] = dict(stopped, seconds=round(time.monotonic() - started, 3),
                              cause=f'the actor could not be made: {type(error).__name__}: {error}')
            try:
                transport.close()
            except Exception:
                pass
            return
        try:
            results[k] = play_world(k, transport, service, order[k], SITES[order[k]],
                                    Records(key, k, plays['rules']['start_group']), player, seed, ticks, deadline,
                                    output / f'world-{k}.jsonl.gz', excluded, kit, items, start_hp, plays['start_levels'])
        except Exception as error:  # play_world returns its failures; a last guard all the same
            results[k] = dict(stopped, seconds=round(time.monotonic() - started, 3), cause=f'{type(error).__name__}: {error}')
    workers = [threading.Thread(target=play, args=(k,), name=f'world-{k}', daemon=True) for k in range(len(worlds))]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=max(0.0, deadline + WATCH['grace'] - time.monotonic()))
    if models:
        torch.set_num_threads(threads)
    result['seconds'] = round(time.monotonic() - started, 3)
    result['results'] = [r if r is not None else dict(world=k, site=order[k], status='hung', lives=[],
                                                      cause=f"no end within the guard and {WATCH['grace']} s")
                         for k, r in enumerate(results)]
    # a life with an excluded drop in view, or ended by its pickup, is reported apart; a pickup alone
    # leaves the run complete
    apart = [dict(world=r['world'], life=life['life'], status=life['status'], reason=life['reason'],
                  excluded_seen=life.get('excluded_seen'), items=life.get('items'))
             for r in result['results'] for life in r['lives']
             if life['reason'] == 'excluded pickup' or life.get('excluded_seen') is not None]
    reasons = [f"world {r['world']} {r['status']}" + (f": {r['cause']}" if r.get('cause') else '')
               for r in result['results'] if r['status'] != 'complete']
    reasons += [f"world {r['world']} life {life['life']} {life['status']} ({life['reason']})"
                + (f": {life.get('cause') or life.get('capture_failure')}" if life.get('cause') or life.get('capture_failure') else '')
                for r in result['results'] for life in r['lives']
                if life['status'] != 'complete' and life['reason'] != 'excluded pickup']
    reasons += [f"world {r['world']} life {life['life']} capture failure: {life['capture_failure']}"
                for r in result['results'] for life in r['lives']
                if life.get('capture_failure') and life['reason'] == 'excluded pickup']
    reasons += [f"world {r['world']} was not lockstep" for r in result['results'] if r.get('lockstep') is False]
    result.update(status='incomplete' if reasons else 'complete', reasons=reasons, apart=apart)
    save()
    return result


# ---- the monitors, from the saved JSONL ------------------------------------------------------------------------------
def product(base):
    """The monitor's name of a produced item family: logs, ores, fish, or other (gems)."""
    return next((PRODUCT[goal] for goal, bases in GOAL_FAMILIES.items() if base in bases), 'other')


def skills(observation):
    """{skill (lower case): (base level, whole XP)} of a native observation."""
    return {s['name'].lower(): (int(s.get('baseLevel') or s.get('level', 1)), int(s.get('xp', 0))) for s in observation.get('skills', [])}


def counts(observation):
    """{item id: (name, count in the inventory)} of a native observation."""
    out = {}
    for item in observation.get('inventory') or []:
        name, n = out.get(item['id'], (item.get('name', ''), 0))
        out[item['id']] = (name, n + item['count'])
    return out


def life_reads(rows):
    """The monitors of one life from its rows in order (module docstring; none is paid, none has a bar), or None when
    the life was never armed. Ticks are from arm, and the life ends at its terminal (a death, its horizon, a cut or a
    pickup; for a failed life, its last clock tick).

    The reads stop at the terminal. A native step that the terminal falls in runs on to its own decision point (the
    simulator stops at the terminal cycle); `overrun` is how far past the terminal its observation lies. At the
    horizon its XP cannot be placed before or after the terminal: the levels, XP, activities and times are read up to
    the last observation at or before the terminal, that step's XP is reported apart (xp_last_step), and its stretch
    counts for no activity ('none'). At a death a dead agent gains nothing after its terminal, so that step's XP counts
    and its stretch, to the death, goes to its activity. Distance, bank opens and new items count only observations at
    or before the terminal, at a death too (its death script can move the agent), and so do fish cooked and burnt: a
    step in which a raw fish (GOAL_FAMILIES fish) left the pack burnt as many as the burnt items it brought ('Burnt
    ...') and, when the cooking XP rose, cooked the rest (a fish dropped, or used on nothing that cooks, is neither). An
    eat counts when it was asked before the terminal; eats_below_full are those at less than full hitpoints. A death is
    read from the trace's terminal (the life row's reason says the same).

    An activity is the cause of the XP that rose between two observations: 'fight:<npc>' for attack, strength, defence
    and hitpoints, by the npc the agent last hit in the combat evidence; else the skill. The ticks from the previous XP
    observation (or arm) to one count for its activity, those after the last for 'none', as sim/gate.py free_reads
    counts the simulator's XP events (natively at decision resolution).

    A capture failure (capture_failure) makes that capture's reads unusable: after a combat failure kills, damage
    taken, dealt and planned and the killer are None and a fight is 'fight' with no npc; after a resource failure (not
    a retire) production is None. damage_taken is the agent's hitpoints lost by attacker (the npc whose damaging attack
    came last): the losses the combat capture saw cycle by cycle, and for a death the hitpoints left before the death
    cycle, which the capture closes before it records (an eat in the cycle of a hit nets the two). The horizon's last
    cycle is not recorded either: when the final observation lies at the horizon, its hitpoints against the last
    recorded give that cycle's loss (end_cycle_hp); when it lies past it, that loss is not read (end_cycle_hp None;
    at most one cycle a horizon life). damage_planned is
    the npcs' attacks on the agent as each was rolled. new_items: each item the life held that it did not hold at arm
    and that is no product, with the first tick it was held and the most held at once (a death drop of a previous life
    taken up shows here). The excluded-drop flag (excluded_seen) reads what excluded_in_view reads. reached_rock: the first
    tick at or before the terminal at which the agent stands within one tile of a rock of MINE_ROCKS (None if never), so
    that a life that never came to a rock cannot read as "mining does not pay"; ores_per_10k: the ores made (production)
    for each 10,000 ticks of the life."""
    arm = next((k for k, r in enumerate(rows) if r['kind'] == 'resource_control' and r['operation'] == 'arm'), None)
    if arm is None:
        return None
    start = rows[arm]['response']['audit']['startTick']
    begin = next(r for r in rows if r['kind'] == 'watch_life')
    end = next((r for r in rows if r['kind'] == 'watch_life_end'), None)
    reset = next((r['observation'] for r in rows if r['kind'] == 'reset'), None)
    first = next(r['observation'] for r in rows[:arm] if r['kind'] == 'episode_start')
    observed = [first] + [r['observation'] for r in rows[arm:] if r['kind'] == 'observed']
    actions = [r['action'] for r in rows[arm:] if r['kind'] == 'requested']
    trace = next((r['response']['audit'] for r in rows if r['kind'] == 'resource_control' and r['operation'] == 'read'), None)
    ticks = end['ticks'] if end and end.get('ticks') is not None else observed[-1]['tick'] - start
    reason = end.get('reason') if end else None
    terminal = ((trace or {}).get('terminal') or {}).get('reason')
    died = terminal == 'death' or reason == 'death'
    final = observed[-1]
    last = final if died else [o for o in observed if o['tick'] - start <= ticks][-1]  # the arm observation is within
    failure = capture_failure(trace)
    combat_ok = trace is not None and isinstance(trace.get('combat'), dict) and trace['combat'].get('failure') is None
    resource_ok = trace is not None and trace.get('failure') in (None, RETIRED)
    names = {n['id']: n['name'].lower() for o in observed for n in o.get('npcs') or []}
    combat = (trace or {}).get('combat') or {}
    events = (combat.get('events') or []) if combat_ok else []
    lives = {e['life']: e['nid'] for e in events if e['kind'] == 'life'}
    npc = lambda nid: names.get(nid, f'npc {nid}')
    hits = [(e['tick'] - start, npc(lives.get(e['life']))) for e in events if e['kind'] == 'hit' and e['hitter'] == 'agent']
    rolled = [e for e in events if e['kind'] == 'agent-hit']
    attacks = [(e['tick'] - start, npc(e['nid']) if e.get('nid') is not None else 'unknown') for e in rolled]
    damaging = [a for a, e in zip(attacks, rolled) if e['damage'] > 0]
    keys = {id(rows): [r[0] for r in rows] for rows in (attacks, damaging)}

    def latest(rows, t):
        """The name of the last of `rows` ((tick, name), in tick order) at or before tick t, or None."""
        k = bisect.bisect_right(keys[id(rows)], t)
        return rows[k - 1][1] if k else None

    def attacker(t):
        """The npc whose damaging attack on the agent came last at or before tick t (any attack if none damaged)."""
        return latest(damaging, t) or latest(attacks, t) or 'other'
    kills = dealt = taken = planned = killer = end_cycle = None
    if combat_ok:
        kills, dealt, taken, planned = Counter(), Counter(), Counter(), Counter()
        hp = combat.get('agentHp', first['player']['hp'])  # the agent's hitpoints at arm, then after each recorded change
        for e in events:
            if e['kind'] == 'damage' and e['hitter'] == 'agent':
                name = npc(lives.get(e['life']))
                dealt[name] += e['hpBefore'] - e['hpAfter']
                kills[name] += e['hpBefore'] > 0 and e['hpAfter'] == 0
            elif e['kind'] == 'agent-hp':
                if e['hpAfter'] < e['hpBefore']:
                    taken[attacker(e['tick'] - start)] += e['hpBefore'] - e['hpAfter']
                hp = e['hpAfter']
            elif e['kind'] == 'agent-hit':
                planned[npc(e['nid']) if e.get('nid') is not None else 'unknown'] += e['damage']
        if died:
            killer = attacker(ticks)
            if hp > 0:  # the death cycle's loss: the capture closes on the lethal hit before it records it
                taken[killer] += hp
        elif terminal == 'horizon':  # the horizon cycle's loss, readable when the final observation lies at the horizon
            end_cycle = max(0, hp - final['player']['hp']) if final['tick'] - start == ticks else None
            if end_cycle:
                taken[attacker(ticks)] += end_cycle
        kills = {k: v for k, v in kills.items() if v}
    made = None
    if resource_ok:
        made = Counter()
        for e in trace.get('events') or []:
            tail = e.get('stackTail') or []
            if e.get('kind') == 'native-op' and e.get('productionSource') is True and len(tail) >= 2:
                made[product(tail[-2])] += 1
    audit = None
    if trace is not None:
        from baseline.native_value import audit_native_value
        try:
            report = audit_native_value(trace)
            audited = sum(f['produced'] for f in report['families'].values())
            counted = sum((made or {}).values())
            audit = 'match' if made is not None and audited == counted else f'differs: {audited} audited, {counted} counted'
        except (EvidenceError, KeyError, TypeError, ValueError) as error:
            audit = f'not audited: {type(error).__name__}: {error}'
    reached = next((o['tick'] - start for o in observed if o['tick'] - start <= ticks and any(
        max(abs(o['player']['x'] - x), abs(o['player']['z'] - z)) <= 1 for x, z in MINE_ROCKS)), None)
    eat_log, full, runs, distance, farthest, banks, new, cooked, burnt = [], 0, [], 0, 0, 0, {}, 0, 0
    food_drops = tool_drops = 0
    raw_fish = GOAL_FAMILIES['fish']
    hit_ticks = [h[0] for h in hits]
    x0, z0 = first['player']['x'], first['player']['z']
    targets = set().union(*GOAL_FAMILIES.values())
    held0 = set(counts(first))
    for k, (before, after) in enumerate(zip(observed, observed[1:])):
        t0, t1 = before['tick'] - start, after['tick'] - start
        action = actions[k] if k < len(actions) else None
        have, now = counts(before), counts(after)
        entry = None if action is None else page_entry(action)  # a page family's row: rows 29..83, act-v4's 96..127
        if entry is not None and entry[0] == A_HELD0 and entry[1] < len(before.get('heldSel') or []):
            held = before['heldSel'][entry[1]]
            if held['option'] == 'Eat' and now.get(held['id'], ('', 0))[1] < have.get(held['id'], ('', 0))[1]:
                eat_log.append([t0, before['player']['hp'], after['player']['hp'] - before['player']['hp']])
                full += before['player']['hp'] >= before['player']['maxHp']
        if entry is not None and entry[0] == A_DROP0 and entry[1] < len(before.get('invSel') or []):
            item = before['invSel'][entry[1]]
            if item.get('slot', -1) >= 0 and now.get(item['id'], ('', 0))[1] < have.get(item['id'], ('', 0))[1]:
                food_drops += 'Eat' in (item.get('options') or ())
                tool_drops += bool(TOOL_PROTECT_RE.search(item.get('name') or ''))
        if t1 <= ticks:  # the observation past the terminal shows the world after it (a death's script, a walk on)
            for item, (name, n) in now.items():
                if item not in targets and item not in held0:
                    seen = new.setdefault(name, dict(tick=t1, most=n))
                    seen['most'] = max(seen['most'], n)
            distance += max(abs(after['player']['x'] - before['player']['x']), abs(after['player']['z'] - before['player']['z']))
            farthest = max(farthest, abs(after['player']['x'] - x0), abs(after['player']['z'] - z0))
            banks += bool(after.get('bankOpen')) and not before.get('bankOpen')
            fell = sum(max(0, n - now.get(item, ('', 0))[1]) for item, (_, n) in have.items() if item in raw_fish)
            if fell:  # a raw fish left the pack: burnt when a burnt item came, cooked when the cooking XP rose
                charred = min(fell, sum(max(0, n - have.get(item, ('', 0))[1]) for item, (name, n) in now.items()
                                        if name.startswith('Burnt')))
                warmer = skills(after).get('cooking', (1, 0))[1] > skills(before).get('cooking', (1, 0))[1]
                cooked, burnt = cooked + (fell - charred if warmer else 0), burnt + charred
        elif not died:  # the step the horizon falls in: its XP is reported apart
            continue
        t1 = min(t1, ticks)  # a death step's XP came before the death
        was = skills(before)
        rose = [s for s, (_, xp) in skills(after).items() if xp > was.get(s, (0, xp))[1]]
        fought = bisect.bisect_right(hit_ticks, t1)
        for s in sorted(set(rose), key=lambda s: (s not in COMBAT_SKILLS, s)):
            activity = s if s not in COMBAT_SKILLS else 'fight' if not combat_ok else \
                f"fight:{hits[fought - 1][1] if fought else 'unknown'}"
            if runs and runs[-1][0] == activity:
                runs[-1][2:] = [t1, runs[-1][3] + 1]
            else:
                runs.append([activity, t1, t1, 1])
    times, end_tick = Counter(), 0
    for activity, _, stop, _ in runs:
        times[activity] += stop - end_tick
        end_tick = stop
    times['none'] += ticks - end_tick
    a, b, c = skills(first), skills(last), skills(final)
    S = FREE_PLAY['skills']
    gain = {s: b[s][0] - a[s][0] for s in a if s in b and b[s][0] != a[s][0]}
    executed = [r['executed_action'] for r in rows[arm:] if r['kind'] == 'task_transition' and r['executed_action'] is not None]
    seen = next((r for r in rows if r['kind'] == 'excluded_seen'), None)
    return dict(
        life=begin['life'], record=begin['record']['index'], seed=begin['seed'], horizon=begin['horizon'],
        status=end['status'] if end else 'incomplete', reason=reason, terminal=terminal, cause=end.get('cause') if end else None,
        capture_failure=failure if trace is not None else None, ticks=ticks,
        decisions=len(actions), overrun=final['tick'] - start - ticks,
        setup_ticks=start - reset['tick'] if reset and reset.get('tick') is not None else None,
        seconds=end.get('seconds') if end else None, hp0=first['player']['hp'], hp_drawn=begin['record']['hp_drawn'],
        levels0={s: a[s][0] for s in S if s in a}, levels1={s: b[s][0] for s in S if s in b}, gain=gain,
        gain_s=sum(gain.get(s, 0) for s in S), xp={s: b[s][1] - a[s][1] for s in a if s in b and b[s][1] != a[s][1]},
        xp_last_step={s: c[s][1] - b[s][1] for s in b if s in c and c[s][1] != b[s][1]},
        made=dict(made) if made is not None else None, kills=kills, reached_rock=reached,
        ores_per_10k=round(made['ores'] * 10000 / ticks, 2) if made is not None and ticks else None,
        damage_taken=dict(taken) if taken is not None else None, damage_dealt=dict(dealt) if dealt is not None else None,
        damage_planned=dict(planned) if planned is not None else None, end_cycle_hp=end_cycle,
        death=dict(tick=ticks, killer=killer) if died else None,
        eat_log=eat_log, eats=len(eat_log), eats_full=full, eats_below_full=len(eat_log) - full,
        watch_food_drops=food_drops, watch_tool_drops=tool_drops, fish_cooked=cooked, fish_burnt=burnt,
        activity=runs, time=dict(times), switches=max(0, len(runs) - 1), distance=distance, farthest=farthest,
        bank=dict(opens=banks, deposits=sum((e := page_entry(x)) is not None and e[0] == A_DEPOSIT0 for x in executed),
                  withdrawals=sum(A_WITHDRAW0 <= x < A_WITHDRAW0 + 8 for x in executed)),
        new_items=new, excluded_seen=seen['tick'] if seen else None, excluded_items=seen['items'] if seen else [],
        capture=dict(events=len((trace or {}).get('events') or []), combat=len(combat.get('events') or []),
                     bound=event_bound(begin['horizon']), failure=(trace or {}).get('failure'),
                     combat_failure=combat.get('failure')) if trace else None,
        audit=audit, controls=end.get('controls') if end else None)


def event_bound(horizon):
    """Both captures' event bound for a window of `horizon` ticks: env-ts/engine/resource-capture.ts eventLimit with
    its maxEvents (WATCH['event_bound'] a started game hour)."""
    return WATCH['event_bound'] * max(1, math.ceil(horizon / WATCH['hour']))


OBSERVED = ('tick', 'player', 'skills', 'inventory', 'invSel', 'heldSel', 'npcs', 'bankOpen')  # what life_reads reads of one


def world_lives(path):
    """(life, rows) of a world's JSONL file, one life at a time and in order (a world's lives are written one after the
    other), each row kept only as far as life_reads reads it: no status rows, no control replies but the arm's and the
    read trace, and of each observation OBSERVED only. A file cut by a stopped process ends at its last whole row."""
    from baseline.world import read_jsonl
    life, rows = None, []
    for row in read_jsonl(path):
        if row.get('life') != life:
            if life is not None and rows:
                yield life, rows
            life, rows = row.get('life'), []
        kind = row['kind']
        if kind == 'server_clock' or (kind == 'resource_control' and row.get('operation') not in ('arm', 'read')):
            continue
        if 'observation' in row:
            row = dict(row, observation={k: row['observation'].get(k) for k in OBSERVED})
        rows.append(row)
    if life is not None and rows:
        yield life, rows


def block(read):
    """The block a life's monitors go to: 'pickup' (ended by an excluded pickup), 'incomplete' (cut, a failure or a
    capture failure), 'flagged' (complete, with an excluded drop in view) or 'clean' (complete)."""
    if read['reason'] == 'excluded pickup':
        return 'pickup'
    if read['status'] != 'complete':
        return 'incomplete'
    return 'flagged' if read['excluded_seen'] is not None else 'clean'


def per_10k(lives, value):
    """The sum of `value` over `lives` for each 10,000 of their armed ticks (None with no ticks)."""
    ticks = sum(r['ticks'] for r in lives)
    return round(sum(value(r) for r in lives) * 10000 / ticks, 3) if ticks else None


def by_level(lives):
    """`lives` grouped by their start Mining level (levels0)."""
    groups = {}
    for r in lives:
        groups.setdefault(r['levels0'].get('mining'), []).append(r)
    return groups


def summarize(lives, every=False):
    """The totals over `lives` (production, kills, damage, eats, deaths, distance, time by activity, switches, bank
    trips, new items) and the means a life (ticks, levels). A read that a capture failure left unusable (None) is left
    out of its total, and `unusable` counts those lives. Level gains are never summed over lives. every (a watch of a
    free-play-v3 policy, which every stat paid): also `every_skill`, the mean level gain a life over every stat and for
    each stat that any life gained XP in, and the lives with XP in it. reached_rock: the lives that came within a tile of a
    rock of MINE_ROCKS; ores_per_10k: the ores for each 10,000 armed ticks of the lives whose production was read;
    by_start_mining: for each start Mining level, the lives, their ticks, the ores and the level gain of mining and of
    every stat for each 10,000 ticks."""
    n = len(lives)
    if not n:
        return dict(lives=0)
    total = lambda key: dict(sum((Counter(r[key]) for r in lives if r[key] is not None), Counter()))
    return dict(
        lives=n, deaths=sum(r['death'] is not None for r in lives),
        killers=dict(Counter(r['death']['killer'] for r in lives if r['death'] and r['death']['killer'])),
        armed_ticks=sum(r['ticks'] for r in lives), made=total('made'), kills=total('kills'),
        damage_taken=total('damage_taken'), damage_dealt=total('damage_dealt'), damage_planned=total('damage_planned'),
        unusable=dict(combat=sum(r['kills'] is None for r in lives), production=sum(r['made'] is None for r in lives),
                      end_cycle_hp=sum(r['terminal'] == 'horizon' and r['kills'] is not None and r['end_cycle_hp'] is None
                                       for r in lives)),
        eats=sum(r['eats'] for r in lives), eats_full=sum(r['eats_full'] for r in lives),
        eats_below_full=sum(r['eats_below_full'] for r in lives),
        watch_food_drops=sum(r['watch_food_drops'] for r in lives), watch_tool_drops=sum(r['watch_tool_drops'] for r in lives),
        fish_cooked=sum(r['fish_cooked'] for r in lives),
        fish_burnt=sum(r['fish_burnt'] for r in lives),
        eats_by_hp=dict(sorted(Counter(e[1] for r in lives for e in r['eat_log']).items())), time=total('time'),
        switches=sum(r['switches'] for r in lives), distance=sum(r['distance'] for r in lives), bank=total('bank'),
        new_items=dict(Counter(name for r in lives for name in r['new_items'])),
        xp_last_step=total('xp_last_step'),
        reached_rock=sum(r['reached_rock'] is not None for r in lives), ores_per_10k=per_10k(
            [r for r in lives if r['made'] is not None], lambda r: r['made'].get('ores', 0)),
        by_start_mining={level: dict(lives=len(group), ticks=sum(r['ticks'] for r in group),
                                     ores=per_10k([r for r in group if r['made'] is not None],
                                                  lambda r: r['made'].get('ores', 0)),
                                     mining=per_10k(group, lambda r: r['gain'].get('mining', 0)),
                                     every_stat=per_10k(group, lambda r: sum(r['gain'].values())))
                         for level, group in sorted(by_level(lives).items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
        a_life=dict(ticks=round(sum(r['ticks'] for r in lives) / n, 1), gain_s=round(sum(r['gain_s'] for r in lives) / n, 3),
                    levels={s: round(sum(r['gain'].get(s, 0) for r in lives) / n, 3) for s in FREE_PLAY['skills']}),
        **({} if not every else dict(every_skill=dict(
            gain_all=round(sum(sum(r['gain'].values()) for r in lives) / n, 3),
            levels={s: round(sum(r['gain'].get(s, 0) for r in lives) / n, 3)
                    for s in sorted({s for r in lives for s in r['xp']})},
            lives_with_xp={s: sum(s in r['xp'] for r in lives) for s in sorted({s for r in lives for s in r['xp']})}))))


def monitors(output):
    """monitors.json of one run folder (module docstring): every life's reads; each world's lives, ticks and rates; the
    summary over the clean lives (complete, no excluded drop in view) and, apart, the flagged lives, the lives an
    excluded pickup ended and the incomplete lives (summarize); and every capture failure. A run that is not complete is
    labelled with its reasons (or its cause): its monitors describe what was played and are not a result. A run whose
    run.json holds no results is read from the world files it left: one still 'running' (killed before its end, by the
    stack's deadline or a kill) is labelled unfinalized, and one that stopped before play (a world not joined) keeps
    its status and cause. `unusable` counts, among other reads, the horizon lives whose last cycle's hitpoints were not
    read (life_reads end_cycle_hp). `world`: the world whose rules the run applied (run_world)."""
    output = Path(output)
    run = json.loads((output / 'run.json').read_text())
    finalized = run.get('results') is not None
    running = not finalized and run.get('status') == 'running'
    paths = {r['world']: (r, output / f'world-{r["world"]}.jsonl.gz') for r in run['results']} if finalized else \
        {int(p.name.split('-')[1].split('.')[0]): (None, p) for p in sorted(output.glob('world-*.jsonl.gz'))}
    worlds, lives = [], []
    for k, (result, path) in sorted(paths.items()):
        reads = [] if not path.exists() else \
            [r for r in (life_reads(rows) for _, rows in world_lives(path)) if r is not None]
        site = result['site'] if result else None
        if result is None and path.exists():  # the site its first life names
            from baseline.world import read_jsonl
            site = next((row['site'] for row in read_jsonl(path) if row['kind'] == 'watch_life'), None)
        lives += [dict(world=k, site=site, **r) for r in reads]
        armed = sum(r['ticks'] for r in reads)
        seconds = (result or {}).get('seconds')
        worlds.append(dict(world=k, site=site, status=result['status'] if result else 'unfinalized' if running else run['status'],
                           cause=(result or {}).get('cause'), lives=len(reads), deaths=sum(r['death'] is not None for r in reads),
                           armed_ticks=armed, armed_ticks_run=(result or {}).get('armed_ticks'),
                           setup_ticks=sum(r['setup_ticks'] or 0 for r in reads), decisions=sum(r['decisions'] for r in reads),
                           seconds=seconds, ticks_per_second=round(armed / seconds, 2) if seconds else None,
                           decisions_per_second=round(sum(r['decisions'] for r in reads) / seconds, 2) if seconds else None,
                           lockstep=(result or {}).get('lockstep')))
    blocks = {name: [r for r in lives if block(r) == name] for name in ('clean', 'flagged', 'pickup', 'incomplete')}
    every = ((run.get('policy') or {}).get('run') or {}).get('task') in EVERY_STAT  # every stat paid: every gain
    status = 'unfinalized' if running else run['status']
    reasons = run.get('reasons') or ([run['cause']] if run.get('cause') else [])
    out = dict(version=WATCH['version'], run=str(output), world=run_world(run), status=status, reasons=reasons,
               **({} if 'start_levels' not in run else dict(start_levels=run['start_levels'])),
               **({} if 'script' not in run else dict(script=run['script'])),
               cause=run.get('cause'),
               rehearsal=run.get('rehearsal'), actor=run.get('actor'), greedy=run.get('greedy'), ticks=run.get('ticks'),
               guard=run.get('guard'), seconds=run.get('seconds'), order=run.get('order'), worlds=worlds,
               summary=summarize(blocks['clean'], every),
               apart=dict(flagged=summarize(blocks['flagged'], every), pickup=summarize(blocks['pickup'], every)),
               incomplete=summarize(blocks['incomplete'], every),
               capture_failures=[dict(world=r['world'], life=r['life'], cause=r['capture_failure'])
                                 for r in lives if r['capture_failure']],
               capture_max=dict(events=max((r['capture']['events'] for r in lives if r['capture']), default=None),
                                combat=max((r['capture']['combat'] for r in lives if r['capture']), default=None)),
               audits=dict(Counter((r['audit'] or 'none').split(':')[0] for r in lives)),
               lives=lives,
               note=None if status == 'complete' else
               'unfinalized: run.json holds no results; read from the world files it left, not a result' if running
               else f'{status}: a description of what was played, not a result')
    (output / 'monitors.json').write_text(json.dumps(out, indent=1, default=str) + '\n', encoding='utf-8')
    return out


def propose(outputs, guard=WATCH['guard'], hours=WATCH['ticks'] // WATCH['hour']):
    """The rehearsal's proposal of the frozen tick count: each world's armed ticks a wall second
    over the rehearsal runs (a world cut by the guard counts up to its cut), the slowest of them, the largest whole
    number of game hours that it plays under the guard, and that number capped at the trial's six hours. It refuses,
    naming each, a run that is no rehearsal, is not finalized or has another number of worlds than the trial's, and a
    world that hung, stopped, was not lockstep, or has no seconds or no armed ticks: no world of a rehearsal is left
    out. `worlds` names every world it measured. A world's seconds run from its thread's start to its end, so a cut
    world's include the retire, read and release of its capture and its close after the guard: its rate is a little
    low. A proposal is of one world (run_world), which it records: runs under the rules of more than one are refused. A
    proposal only: it freezes nothing."""
    rates, refused, played = [], [], {}
    for output in outputs:
        run = json.loads((Path(output) / 'run.json').read_text())
        if not run.get('rehearsal'):
            refused.append(f'{output}: not a rehearsal run')
            continue
        results = run.get('results')
        if results is None or run.get('status') not in ('complete', 'incomplete'):
            refused.append(f"{output}: not finalized (status {run.get('status')!r}, no results)" if results is None
                           else f"{output}: status {run.get('status')!r}")
            continue
        if len(results) != WATCH['worlds']:
            refused.append(f"{output}: {len(results)} worlds, not {WATCH['worlds']}")
        played.setdefault(run_world(run), []).append(str(output))
        for r in results:
            where = f"{output} world {r.get('world')} ({r.get('site')})"
            if r.get('status') not in ('complete', 'cut'):
                refused.append(f"{where}: {r.get('status')}" + (f": {r['cause']}" if r.get('cause') else ''))
            elif r.get('lockstep') is False:
                refused.append(f'{where}: not lockstep')
            elif not r.get('seconds') or not r.get('armed_ticks'):
                refused.append(f'{where}: no seconds or no armed ticks')
            else:
                rates.append(dict(run=str(output), actor=run['actor'], world=r['world'], site=r['site'], status=r['status'],
                                  armed_ticks=r['armed_ticks'], seconds=r['seconds'],
                                  ticks_per_second=round(r['armed_ticks'] / r['seconds'], 2)))
    if len(played) > 1:
        refused.append('runs under the rules of more than one world: ' + '; '.join(f'{w}: {", ".join(runs)}'
                                                                                   for w, runs in sorted(played.items())))
    if refused or not rates:
        raise ValueError('the rehearsal does not measure every world: ' + ('; '.join(refused) or 'no run given'))
    slowest = min(rates, key=lambda r: r['armed_ticks'] / r['seconds'])
    fits = math.floor(guard * slowest['armed_ticks'] / slowest['seconds'] / WATCH['hour'])
    return dict(world=next(iter(played)), guard=guard, worlds=rates, slowest=slowest, fits_hours=fits,
                proposed_hours=min(fits, hours), proposed_ticks=min(fits, hours) * WATCH['hour'], cap_hours=hours)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    settings = argparse.ArgumentParser(add_help=False)  # a run's settings, which check takes too
    settings.add_argument('--output', type=Path, required=True, help='the run folder, which must not exist')
    settings.add_argument('--world', required=True, choices=sorted(VILLAGES),
                          help="the world whose rules the run applies, over its archive; a policy's run trained in it (no default)")
    settings.add_argument('--actor', required=True, choices=WATCH['actors'])
    settings.add_argument('--run', type=Path, help="a free-play-v1 training run's folder (initial and policy)")
    settings.add_argument('--policy-seed', type=int, help="that run's training seed")
    settings.add_argument('--checkpoint', type=Path, help='policy: one of the run\'s checkpoints in place of RUN/latest.zip')
    settings.add_argument('--real-checkpoint', type=Path, help='policy: one checkpoint of a real-engine run (sim.train '
                          '--env real), played under its own checks, with no --run')
    settings.add_argument('--launch', type=Path, help='with --real-checkpoint: the training run\'s launch.json')
    settings.add_argument('--greedy', action='store_true', help='greedy play in place of sampled')
    settings.add_argument('--sites', nargs='+', required=True, metavar='SITE',
                          help=f'{WATCH["worlds"]} names of SITES, or one that every world plays')
    settings.add_argument('--rotation', type=int, required=True, help='world k plays site (k + rotation) mod 4')
    settings.add_argument('--key', type=int, required=True, help="the start records' key (no default)")
    settings.add_argument('--seed', type=int, required=True, help="the run's seed (each life's actor seed; no default)")
    settings.add_argument('--ticks', type=int, default=WATCH['ticks'], help='armed native ticks a world')
    settings.add_argument('--guard', type=float, default=WATCH['guard'], help='wall seconds from the first reset')
    settings.add_argument('--start-levels', help='a profile: these stats at these levels in every record, the rest of '
                          'each record as drawn (for example woodcutting=50,firemaking=52,hitpoints=34)')
    settings.add_argument('--start-hp', choices=('drawn',), help='each life at its record\'s drawn hitpoints: a control '
                          'actor beside a free-play-v12, v14 or v15 policy (a policy plays under its own version\'s rule)')
    play = commands.add_parser('run', parents=[settings],
                               help='one run of four worlds inside scripts/stack.py --worlds 4 --lockstep --xp-rate 1')
    play.add_argument('--rehearsal', action='store_true', help='label the run a rehearsal: it measures, it shows nothing')
    commands.add_parser('check', parents=[settings],
                        help="a run's checks that need no native world (the policy's world and the engine checkout "
                             'included): before scripts/stack.py')
    read = commands.add_parser('monitors', help="the monitors of run folders from their saved JSONL (monitors.json)")
    read.add_argument('outputs', type=Path, nargs='+')
    plan = commands.add_parser('propose', help="the rehearsal's proposal of the frozen tick count")
    plan.add_argument('outputs', type=Path, nargs='+')
    plan.add_argument('--guard', type=float, default=WATCH['guard'])
    args = parser.parse_args()
    if args.command in ('run', 'check'):
        chosen = dict(output=args.output, world=args.world, actor=args.actor, sites=args.sites, rotation=args.rotation, key=args.key,
                      seed=args.seed, ticks=args.ticks, guard=args.guard, run=args.run, policy_seed=args.policy_seed,
                      checkpoint=args.checkpoint, real_checkpoint=args.real_checkpoint, launch=args.launch,
                      start_hp=args.start_hp, start_levels=args.start_levels)
        try:
            if args.command == 'check':
                check_engine(checkout_tree())
                plays = check_settings(**chosen)
            else:
                result = watch_run(greedy=args.greedy, rehearsal=args.rehearsal, **chosen)
        except ValueError as error:
            raise SystemExit(f'[watch] {error}')
    if args.command == 'check':
        print(f"[watch] checked: {args.world} over {plays['tables']['path']} (sha256 {plays['tables']['sha256']}), "
              f"{args.actor}, sites {plays['order']}, {len(plays['excluded'])} excluded objects", flush=True)
        if args.real_checkpoint is not None:
            real = plays['facts']['run']
            print(f"[watch] real-engine checkpoint {real['checkpoint']} (sha256 {plays['facts']['policy']['sha256']}, "
                  f"{real['decisions']} decisions), start {real['start']}, launch {(real['launch'] or {}).get('path')}, "
                  f"env-ts trained {real['env_ts']['trained']}, played {real['env_ts']['played']}", flush=True)
        return
    if args.command == 'run':
        print(f"[watch] {result['status']}: " + '; '.join(
            f"world {r['world']} {r['site']} {r['status']} {len(r['lives'])} lives {r.get('armed_ticks')} ticks "
            f"{r.get('seconds')} s" for r in result.get('results', [])) + ''.join(f"\n  {x}" for x in result.get('reasons', [])),
            flush=True)
        raise SystemExit(0 if result['status'] == 'complete' else 1)
    if args.command == 'monitors':
        for output in args.outputs:
            out = monitors(output)
            print(json.dumps(dict(run=out['run'], status=out['status'], worlds=out['worlds'], summary=out['summary']),
                             indent=1, default=str))
        return
    print(json.dumps(propose(args.outputs, args.guard), indent=1))


if __name__ == '__main__':
    main()
