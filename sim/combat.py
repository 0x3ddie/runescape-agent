"""sustained-combat-v1, the melee goal of goal-activity-v3.

An episode is several fights in one world. The player's hitpoints, food, equipment and delays carry from one fight to
the next; nothing is restored after a kill. The eligible enemies (COMBAT['enemies']) can all damage the player:
goblins fight back when attacked, and muggers also attack a player who comes within their hunt range. Food in the
inventory and natural healing (1 hitpoint per 100 ticks) are the ways to recover; retreating and waiting are ordinary
actions. Death ends the episode and settles its return to zero; death and respawn recovery are outside this slice.

The world is a training layout (sim/world.py) with its npcs replaced: the fishing spots stay, every chicken spawn
becomes a goblin spawn (the layouts' start tiles reach one within WORLD['goal_steps'] steps, through a pen's gate where
there is one), and an episode of the `danger` group adds muggers near the goblins. Npcs change no collision, so the
layout's tables serve unchanged. wood, fish and mine keep goal-activity-v2's task, starts and horizon.

A start is drawn by `draw_combat` after task.draw_start's own draws (which it leaves unchanged): the episode's group,
its current hitpoints (below the base level in three groups), its food, whether the carried weapon starts worn (a
setup Wield, an ordinary action before arm) and its muggers. Groups (COMBAT['groups']) are the evaluation's declared
conditions; training draws them uniformly. Return: min(certified kills, target) / target if the player survives the
horizon, else 0 (sim/task.py pays each kill as it is certified and settles to the audit). Eating, waiting, damage and
healing pay nothing.

`reference_action` is a checked test instrument (feasibility and task-design measurements), never a source of labels.

village-world-v1 Stage A, free-play-v1's first world (VILLAGE, draw_village, village_life), is this world with the
layout's chickens added and muggers a declared setting. village-world-v2, its second (VILLAGE_V2), is the same on
sim/world.py VILLAGE_WORLD layouts, which add a field, door pens, a willow, a yew and a highwayman; it holds no muggers.
"""
import numpy as np

from sim import paths
from act_contract import CLICK_ROWS
from sim.agent import A_ACT0, A_CLOSE_IF, A_DIALOG_CONTINUE, A_HELD0, A_WAIT
from sim.engine import HITPOINTS, LOOT_OUTSIDE, NPC_HP, WEARPOS_RHAND, Rng
from sim.world import ORIGIN, PEN_GATE, VILLAGE_WORLD, flood, split_seeds, world_of

COMBAT = dict(
    version='sustained-combat-v1', task='goal-activity-v3',
    enemies=('goblin', 'mugger'),        # eligible kills; every life is credited once (sim/audit.py)
    target=5, horizon=1200,              # certified kills; native ticks from arm
    tools=(('bronze_axe', 1), ('net', 1), ('bronze_pickaxe', 1)),  # START_KIT without its food
    food=('shrimp', 'bread'),            # heal 3 and 4; each item drawn uniformly
    worn=0.5,                            # the drawn weapon starts worn with this probability
    mugger_camp=(6, 12),                 # Chebyshev tiles from the goblins' centre
    mugger_start=7,                      # and at least this far from the start tile
    exclusions=('random_events', 'clue_drops', 'loot_drops'),  # sim/engine.py EXCLUSIONS: declared, in training and evaluation
    # hitpoints at arm: (lowest, highest) below the base level (0 = base), or absolute (abs=True); food items
    groups=dict(healthy=dict(hp=(0, 2), food=(2, 4), muggers=0),
                hurt_fed=dict(hp=(2, 4), abs=True, food=(2, 4), muggers=0),
                hurt_unfed=dict(hp=(4, 7), abs=True, food=(0, 0), muggers=0),
                danger=dict(hp=(0, 3), food=(1, 3), muggers=1)),
)
GROUPS = tuple(COMBAT['groups'])
# Task versions that are goal-activity-v5 with declared bindings and nothing else changed. goal-activity-v11 (the
# native gap study's arm Layouts, 8 November 2026: mine only): its compact pool is compact-world-v1's train
# seeds 1,000-1,127 in order (sim.world pool --split train --count 128 --first 1000) and its goal set is mine alone.
# CpuBatch and NativeBatch refuse any other (check_bindings).
BOUND = {'goal-activity-v11': dict(pool=dict(world='compact-world-v1', split='train', first=1000, count=128),
                                   goals=('mine',))}
TASKS = ('goal-activity-v2', COMBAT['task'], 'goal-activity-v4', 'goal-activity-v5', *BOUND)


def task_spec(version):
    """A four-goal task version's per-goal (GOALS order) horizon in native ticks and certified target: v2 as the
    four-activity task, v3 with the melee goal replaced by this module's task, v4 as v3 with the wood goal replaced by
    sim/search.py's search-wood-v1 (its starts on the search pool, its own horizon), v5 as v3 with the wood goal replaced
    by sim/search.py's supply-wood-v1 (its paired starts on the supply pool, its horizon and production cap), v11 as v5
    with BOUND's `pool` and `goals`."""
    from baseline.resource_task import GOAL_TARGETS, GOALS
    from sim.spec import HORIZON_TICKS
    if version not in TASKS:
        raise ValueError(f'unknown task {version}')
    base = 'goal-activity-v5' if version in BOUND else version
    combat = base in (COMBAT['task'], 'goal-activity-v4', 'goal-activity-v5')
    supply = base == 'goal-activity-v5'
    search = base == 'goal-activity-v4' or supply  # a pool of 64-tile layouts for the wood slots
    if search:
        from sim.search import SEARCH, SUPPLY
        wood = SUPPLY if supply else SEARCH
    horizon = lambda g: COMBAT['horizon'] if combat and g == 'melee' else wood['horizon'] if search and g == 'wood' else HORIZON_TICKS
    target = lambda g: COMBAT['target'] if combat and g == 'melee' else (wood['cap'] if supply else wood['target']) \
        if search and g == 'wood' else GOAL_TARGETS[g]
    return dict(version=version, horizons=tuple(map(horizon, GOALS)), targets=tuple(map(target, GOALS)), combat=combat,
                search=search, supply=supply, **BOUND.get(version, {}))


def check_bindings(spec, layouts, goals):
    """Refuse what a task spec's bindings (BOUND) exclude: `goals` (sim/spec.py goal_set) must be the bound goal set,
    and `layouts`, the compact pool's (seed, declared world) in pool order (an iterable, read only for a bound pool),
    exactly the bound pool's. A version without bindings takes any pool and goal set."""
    if 'goals' in spec and tuple(goals) != spec['goals']:
        raise ValueError(f"{spec['version']} plays goals {list(spec['goals'])}, not {list(goals)}")
    if 'pool' in spec:
        p, layouts = spec['pool'], list(layouts)
        seeds = split_seeds(p['split'], p['count'], p['first'])
        if layouts != [(seed, p['world']) for seed in seeds]:
            raise ValueError(f"{spec['version']} plays {p['world']} {p['split']} seeds {seeds[0]}-{seeds[-1]} in order, not "
                             f"{len(layouts)} layouts of {sorted({world for _, world in layouts}, key=str)} (seeds "
                             f"{[seed for seed, _ in layouts][:4]}...)")
# Purposes of the combat draws, apart from sim/task.py's R_LAYOUT..R_WORLD (1-5).
R_GROUP, R_HP, R_FOOD_N, R_FOOD, R_WORN, R_MUGGER = range(16, 22)


def enemy_types(tables):
    """The eligible enemies' npc type ids."""
    ids = {v['debugname']: k for k, v in tables.npc_types.items()}
    return frozenset(ids[name] for name in COMBAT['enemies'])


def outside(layout):
    """Tiles of the layout's main walkable component with every gate closed (the start tiles' component)."""
    steps = paths.step_masks(layout.flags)
    reached = flood(steps, layout.start) >= 0
    return reached & layout.walkable


def camp(layout, tables):
    """The local tiles of the layout's chicken spawns (each becomes a goblin spawn)."""
    chickens = {k for k, v in tables.npc_types.items() if v['debugname'] in ('chicken', 'chicken_brown')}
    return [(int(x), int(z)) for _, ty, x, z in layout.npcs if int(ty) in chickens]


def draw_combat(key, slot, episode, layout, start, tables, group=None, region=None, place_muggers=True):
    """The combat part of episode `episode` of world slot `slot` (module docstring), after task.draw_start's `start` on
    `layout`: group (`group` replaces its draw), hitpoints at arm, food (debugnames in inventory order), whether the
    weapon starts worn, and mugger spawn tiles (local; none drawn without `place_muggers`, a world that holds no muggers:
    the other draws have purposes of their own, so they are the same either way). `region`: outside(layout), if the
    caller has it. Keyed draws (engine.Rng on (key, slot, episode, purpose)), as draw_start's."""
    h = Rng(key).prefix(slot, episode)
    draw = lambda *k: Rng.draw(h, *k)
    pick = lambda lo, hi, *k: lo + int(draw(*k) * (hi - lo + 1))
    group = GROUPS[int(draw(R_GROUP) * len(GROUPS))] if group is None else group
    spec = COMBAT['groups'][group]
    base = start['levels']['hitpoints']
    lo, hi = spec['hp']
    hp = pick(lo, hi, R_HP) if spec.get('abs') else base - pick(lo, hi, R_HP)
    hp = max(1, min(base, hp))
    food = [COMBAT['food'][int(draw(R_FOOD, i) * len(COMBAT['food']))] for i in range(pick(*spec['food'], R_FOOD_N))]
    muggers = []
    if spec['muggers'] and place_muggers:
        region = outside(layout) if region is None else region
        tiles = camp(layout, tables)
        cx, cz = np.mean(tiles, axis=0)
        sx, sz = start['tile'][0] - ORIGIN[0], start['tile'][1] - ORIGIN[1]
        lo, hi = COMBAT['mugger_camp']
        options = [(int(x), int(z)) for x, z in np.argwhere(region)
                   if lo <= max(abs(x - cx), abs(z - cz)) <= hi and max(abs(x - sx), abs(z - sz)) >= COMBAT['mugger_start']
                   and (int(x), int(z)) not in tiles]
        if len(options) < spec['muggers']:
            raise ValueError(f'layout {layout.seed}: {len(options)} tiles for {spec["muggers"]} muggers')
        for i in range(spec['muggers']):
            tile = options[int(draw(R_MUGGER, i) * len(options))]
            options.remove(tile)
            muggers.append(tile)
    return dict(group=group, hp=hp, food=food, worn=draw(R_WORN) < COMBAT['worn'], muggers=muggers)


def combat_npcs(layout, tables, muggers):
    """The episode's npc spawns (World's `npcs`): the layout's fishing spots, a goblin at every chicken spawn, then the
    muggers, with nids in that order."""
    ids = {v['debugname']: k for k, v in tables.npc_types.items()}
    chickens = {ids['chicken'], ids['chicken_brown']}
    rows = [(int(ty), int(x), int(z)) for _, ty, x, z in layout.npcs if int(ty) not in chickens]
    rows += [(ids['goblin'], x, z) for x, z in camp(layout, tables)]
    rows += [(ids['mugger'], x, z) for x, z in muggers]
    return [dict(nid=k, type=ty, x=ORIGIN[0] + x, z=ORIGIN[1] + z, level=0, size=tables.npc_types[ty]['size'])
            for k, (ty, x, z) in enumerate(rows)]


def kit(weapon, food):
    """The inventory at reset: the tools, the drawn food, then the weapon (carried; a worn start wields it in setup)."""
    return [*COMBAT['tools'], *((name, 1) for name in food), (weapon, 1)]


# ---- village-world-v1 Stage A, free-play-v1's world ------------------------------------------------------------------
# The melee task's world (combat_npcs: the fishing spots, a goblin at every chicken spawn, the muggers) with the layout's
# chickens added at their own spawns, on compact-world-v1 layouts. Muggers are a declared setting with no default:
# `muggers` True places them by the danger group's rule, False places none. The start (levels, tile, weapon,
# hitpoints, food, a worn weapon) is the same draw in both forms: task.draw_start, then draw_combat in `start_group`
# (proposal: the danger group's hitpoints, food and worn draws, the draws step A's melee world used), whose mugger tiles
# are kept only with muggers. The food comes only from the start kit (kit()); the exclusions are the melee task's.
VILLAGE = dict(version='village-world-v1', stage='A', layouts='compact-world-v1', start_group='danger',
               exclusions=COMBAT['exclusions'])
# village-world-v2 (accepted 13 November 2026): the same world on its own
# layouts (sim/world.py VILLAGE_WORLD), whose highwayman is a layout npc that combat_npcs passes through, so it spawns
# in every life of a layout that holds one. It holds no muggers (`muggers` must be False). The cabbage is in its declared
# world: `loot_outside` is LOOT_OUTSIDE without it; its fields' picks and the use scripts of the field items and of the
# highwayman's cape (sim/engine.py FIELD_USE) run over an archive with the pick rows (`content`: sim/build.py picks, the
# archive `tables`); the cabbage's Eat runs over any archive; no npc of this world drops a cabbage. The start draw, the
# kit and the other exclusions are VILLAGE's.
VILLAGE_V2 = dict(version='village-world-v2', layouts=VILLAGE_WORLD['version'], start_group='danger',
                  exclusions=COMBAT['exclusions'], muggers=False, loot_outside=tuple(n for n in LOOT_OUTSIDE if n != 'cabbage'),
                  content='picks', tables='draynor-nc2.npz')
VILLAGES = {v['version']: v for v in (VILLAGE, VILLAGE_V2)}


def village_of(layout):
    """The free-play world (VILLAGES) that plays the layout's declared world, or None."""
    name = world_of(layout)['version']
    return next((v for v in VILLAGES.values() if v['layouts'] == name), None)


def check_village(village, tables, muggers):
    """Refuse what a free-play world cannot play: muggers other than True or False, muggers where it holds none, and an
    archive without the content its scripts read."""
    if type(muggers) is not bool:
        raise ValueError('village-world-v1 Stage A places muggers or not: pass True or False')
    if muggers and village.get('muggers') is False:
        raise ValueError(f"{village['version']} holds no muggers: pass False")
    content = village.get('content')
    if content and content not in tables.meta.get('content', {}):
        raise ValueError(f"{village['version']} plays over an archive with the {content} rows ({village['tables']}, "
                         'sim/build.py), not this one')


def village_npcs(layout, tables, muggers):
    """The world's npc spawns (World's `npcs`): combat_npcs (in village-world-v2 the layout's highwayman with the fishing
    spots), then the layout's chickens at their spawns, with nids in that order."""
    ids = {v['debugname']: k for k, v in tables.npc_types.items()}
    rows = combat_npcs(layout, tables, muggers)
    chickens = [(int(ty), int(x), int(z)) for _, ty, x, z in layout.npcs if int(ty) in (ids['chicken'], ids['chicken_brown'])]
    return rows + [dict(nid=len(rows) + k, type=ty, x=ORIGIN[0] + x, z=ORIGIN[1] + z, level=0,
                        size=tables.npc_types[ty]['size']) for k, (ty, x, z) in enumerate(chickens)]


def draw_village(key, slot, episode, layouts, tables, muggers, layout=None, region=None):
    """A free-play start in the world that plays the drawn layout's (village_of: VILLAGE or VILLAGE_V2): task.draw_start's
    dict with the drawn hitpoints, food, worn weapon and mugger tiles under 'village' (no tiles without muggers; none drawn
    in a world that holds none). `muggers` must be True or False, and False in village-world-v2."""
    from sim.task import draw_start
    if type(muggers) is not bool:
        raise ValueError('village-world-v1 Stage A places muggers or not: pass True or False')
    start = draw_start(key, slot, episode, layouts, layout=layout)
    village = village_of(layouts[start['layout']])
    if village is None:
        raise ValueError(f"free play plays the layouts of {[v['layouts'] for v in VILLAGES.values()]}")
    if muggers and village.get('muggers') is False:
        raise ValueError(f"{village['version']} holds no muggers: pass False")
    c = draw_combat(key, slot, episode, layouts[start['layout']], start, tables, group=village['start_group'], region=region,
                    place_muggers=village.get('muggers') is not False)
    start['village'] = dict(c, muggers=c['muggers'] if muggers else [], placed=muggers)
    return start


def village_life(start, layout, lt, tables, xp_rate, length, cfg=None):
    """A started free-play life (sim/task.py FreePlayEpisode) of a draw_village start in a fresh world on `layout`, whose
    LayoutTables is `lt`, at `xp_rate`, `length` native ticks long, in the world that plays the layout's (village_of): its
    exclusions and its loot outside the slice."""
    from sim.engine import World
    from sim.task import WORLD_START_TICK, FreePlayEpisode
    v, village = start['village'], village_of(layout)
    if village is None:
        raise ValueError(f"free play plays the layouts of {[w['layouts'] for w in VILLAGES.values()]}")
    check_village(village, tables, v['placed'])
    world = World(lt, seed=start['world_seed'], start_tick=WORLD_START_TICK, npcs=village_npcs(layout, tables, v['muggers']),
                  exclusions=village['exclusions'], xp_rate=xp_rate, loot_outside=village.get('loot_outside'))
    task = FreePlayEpisode(world, length, xp_rate, cfg)
    task.start(start['tile'], start['weapon'], start['levels'], kit=kit(start['weapon'], v['food']), hp=v['hp'],
               wield=v['worn'])
    return task


# ---- the reference controller: a test instrument --------------------------------------------------------------
# Styles: `sustain` eats at eat_at hitpoints with food, and without food rests (retreats from an attacker, then waits)
# at rest_at until resume_at; `reckless` never eats or rests; `full` (heal to full first) eats or rests until the base
# level before every new fight. Each attacks the nearest goblin at full health, fights back an attacker, opens a
# closed pen gate first, and acts only through the published legal actions.
REFERENCE = dict(sustain=dict(eat_at=4, rest_at=3, resume_at=6), reckless=dict(eat_at=0, rest_at=0, resume_at=0),
                 full=dict(eat_at=None, rest_at=None, resume_at=None))
FOOD_NAMES = ('Shrimps', 'Bread')


class Reference:
    def __init__(self, style='sustain'):
        self.style, self.rule = style, REFERENCE[style]
        self.resting = False

    def reset(self, seed):
        self.resting = False

    def act(self, observation, task):
        return self.action(task)

    def action(self, task):
        a, w, p = task.agent, task.world, task.agent.player
        obs, mask = a.last_observation, a.last_observation['mask']
        base = p.base_levels[HITPOINTS]
        if mask[A_DIALOG_CONTINUE]:
            return A_DIALOG_CONTINUE
        if obs['dialogOpen'] or obs['modalOpen']:
            return A_CLOSE_IF
        held = lambda option, names: next((A_HELD0 + k for k, h in enumerate(obs['heldSel'])
                                           if h['option'] == option and h['name'] in names and mask[A_HELD0 + k]), None)
        if p.worn.items[WEARPOS_RHAND] is None:
            wield = held('Wield', ('Bronze dagger', 'Bronze sword'))
            if wield is not None:
                return wield
        attackers = [n for n in w.npcs if n.is_valid() and n.target is p]
        fighting = p.target is not None or attackers
        eat = held('Eat', FOOD_NAMES)
        if self.style == 'full':
            want = p.hp < base and not fighting
            if want and eat is not None:
                return eat
            if want:
                return A_WAIT
            if p.hp <= 2 and eat is not None:
                return eat
        else:
            if p.hp <= self.rule['eat_at'] and eat is not None:
                return eat
            if eat is None and p.hp <= self.rule['rest_at']:
                self.resting = True
            if self.resting and p.hp >= self.rule['resume_at']:
                self.resting = False
            if self.resting:
                if attackers:  # act-v3 has no RETREAT: a click away from them, as RETREAT walked
                    away = click_away(task, attackers)
                    return A_WAIT if obs.get('optionBusy') and obs.get('optionAction') in CLICK_ROWS else away
                return A_WAIT
        if obs.get('optionBusy') and (p.target is not None or not attackers):
            return A_WAIT
        targets = self.targets(task, attackers)
        for k, sel in enumerate(obs['actSel']):
            if mask[A_ACT0 + k] and (sel['kind'], sel['x'], sel['z'], sel['option']) in targets:
                return A_ACT0 + k
        if not targets:
            return A_WAIT
        from sim.run import click_toward
        return click_toward(task, {(t[1], t[2]) for t in targets})

    @staticmethod
    def targets(task, attackers):
        """(kind, x, z, option) of what to act on: an attacker first, a closed pen gate, then a goblin at full health."""
        w, p = task.world, task.agent.player
        if attackers:
            return {('npc', n.x, n.z, 'Attack') for n in attackers}
        gate = {('loc', loc.x, loc.z, 'Open') for loc in w.locs
                if loc.active and w.t.loc_types[loc.type]['debugname'] in PEN_GATE}
        if gate:
            return gate
        goblins = [n for n in w.npcs if n.is_valid() and w.t.npc_types[n.type]['debugname'] == 'goblin'
                   and n.target in (None, p)]
        engaged = [n for n in goblins if p.name in n.heroes]
        return {('npc', n.x, n.z, 'Attack') for n in engaged or goblins
                if engaged or n.levels[NPC_HP] == w.t.npc_types[n.type]['stats'][NPC_HP]}


# ---- episodes, evidence and evaluation ----------------------------------------------------------------------
def episode(record, layout, lt, tables, cfg=None):
    """A started combat episode (GoalEpisode) of a start record (draw_start's dict with its draw_combat under 'combat'),
    in a fresh world on `layout` (whose LayoutTables is `lt`)."""
    from baseline import config
    from sim.engine import World
    from sim.task import WORLD_START_TICK, GoalEpisode
    c = record['combat']
    world = World(lt, seed=record['world_seed'], start_tick=WORLD_START_TICK, npcs=combat_npcs(layout, tables, c['muggers']),
                  exclusions=COMBAT['exclusions'])
    task = GoalEpisode(world, cfg or config.TASK_CFG, 'melee', horizon=COMBAT['horizon'], scale=1.0, combat=True)
    task.start(record['tile'], record['weapon'], record['levels'], kit=kit(record['weapon'], c['food']), hp=c['hp'],
               wield=c['worn'])
    return task


def fights(task):
    """What happened in a finished combat episode, from its world's events inside the window (arm to the terminal
    cycle): one row per enemy life the agent fought (it hit the agent or the agent hit it) with who struck first,
    damage dealt and received, the agent's hitpoints around it and whether it is a certified kill; every hitpoint
    gain (eat, regen or a Hitpoints level-up) and whether a certified kill came after it; food eaten without a gain."""
    from sim.audit import kill_lives
    w = task.world
    window = [e for e in w.events[task.event_start:] if e[0] < task.terminal['tick']]
    certified = kill_lives(window, task.agent.name, task.enemies)
    kill_at = sorted(certified.values())
    life, rows, heals, eaten = {}, {}, [], []
    names = {n.nid: w.t.npc_types[n.base]['debugname'] for n in w.npcs}

    def row(nid, lf, tick):
        return rows.setdefault((nid, lf), dict(npc=nid, life=lf, type=names.get(nid), first=tick, last=tick, aggressor=None,
                                               dealt=0, hits=0, received=0, struck=0, hp_before=None, hp_after=None,
                                               defeated=None, certified=(nid, lf) in certified))
    hp = task.arm_hp
    for tick, kind, data in window:
        if kind == 'npc_damage' and data['source'] == task.agent.name:
            r = row(data['npc'], data['life'], tick)
            life[data['npc']] = data['life']
            r['aggressor'] = r['aggressor'] or 'agent'
            r['dealt'] += min(data['damage'], data['hp'])
            r['hits'] += 1
            r['last'] = tick
            r['hp_before'] = hp if r['hp_before'] is None else r['hp_before']
        elif kind == 'npc_defeated':
            if (data['npc'], data['life']) in rows:
                rows[(data['npc'], data['life'])]['defeated'] = tick
            life[data['npc']] = data['life'] + 1
        elif kind == 'agent_hp':
            if data['cause'] == 'damage':
                if data['source'] is not None:
                    r = row(data['source'], life.get(data['source'], 0), tick)
                    r['aggressor'] = r['aggressor'] or 'npc'
                    r['received'] += data['before'] - data['after']
                    r['struck'] += 1
                    r['last'] = tick
                    r['hp_before'] = data['before'] if r['hp_before'] is None else r['hp_before']
                    r['hp_after'] = data['after']
            else:
                heals.append(dict(tick=tick, cause=data['cause'], gained=data['after'] - data['before'], hp=data['before'],
                                  kill_after=any(k > tick for k in kill_at)))
            hp = data['after']
        elif kind == 'remove' and data['why'] == 'eat':
            eaten.append(dict(tick=tick, obj=w.t.obj_types[data['obj']]['debugname']))
    eat_ticks = {h['tick'] for h in heals if h['cause'] == 'eat'}
    out = sorted(rows.values(), key=lambda r: (r['first'], r['npc']))
    for r in out:
        r['eats'] = sum(r['first'] <= t <= r['last'] for t in eat_ticks)
        for k in ('first', 'last', 'defeated'):
            r[k] = None if r[k] is None else r[k] - task.start_tick
    for h in heals:
        h['tick'] -= task.start_tick
    for e in eaten:
        e['tick'] -= task.start_tick
    eats = [h for h in heals if h['cause'] == 'eat']
    return dict(fights=out, heals=heals, eaten=eaten, kill_ticks=[k - task.start_tick for k in kill_at],
                summary=dict(fights=len(out), certified=len(kill_at), aggressed=sum(r['aggressor'] == 'npc' for r in out),
                             dealt=sum(r['dealt'] for r in out), received=sum(r['received'] for r in out),
                             eats=len(eats), eat_gain=sum(h['gained'] for h in eats),
                             eats_without_gain=len(eaten) - len(eats),
                             regen_gain=sum(h['gained'] for h in heals if h['cause'] == 'regen'),
                             levelup_gain=sum(h['gained'] for h in heals if h['cause'] == 'levelup'),
                             eats_then_kill=sum(h['kill_after'] for h in eats),
                             regen_then_kill=sum(h['gained'] for h in heals if h['cause'] == 'regen' and h['kill_after']),
                             by_type={t: dict(fights=sum(r['type'] == t for r in out),
                                              kills=sum(r['type'] == t and r['certified'] for r in out),
                                              received=sum(r['received'] for r in out if r['type'] == t))
                                      for t in COMBAT['enemies']}))


class PolicyActor:
    """A saved policy's greedy actions through MaskedRecurrentPolicy.forward with both recurrent states (actor and
    critic), reset at each trial, so the action, its probabilities and the value come from the same call and memory.
    With `trace`, act() also returns the five likeliest actions with their probabilities and the value."""

    def __init__(self, path, deterministic=True):
        import torch
        from sb3_contrib import RecurrentPPO
        from baseline.policy import check_names
        self.torch = torch
        self.model = RecurrentPPO.load(path, device='cpu')
        check_names(self.model.policy)
        self.deterministic = deterministic

    def reset(self, seed):
        from sb3_contrib.common.recurrent.type_aliases import RNNStates
        policy, torch = self.model.policy, self.torch
        torch.manual_seed(seed)
        zeros = lambda lstm: (torch.zeros(lstm.num_layers, 1, lstm.hidden_size), torch.zeros(lstm.num_layers, 1, lstm.hidden_size))
        self.state = RNNStates(zeros(policy.lstm_actor), zeros(policy.lstm_critic))
        self.first = True

    def act(self, observation, task=None, trace=False):
        torch, policy = self.torch, self.model.policy
        observation = {**observation, **{k: np.zeros(space.shape, space.dtype) for k, space in
                                         self.model.observation_space.spaces.items() if k not in observation}}
        obs, _ = policy.obs_to_tensor(observation)
        with torch.no_grad():
            action, value, _, self.state = policy.forward(obs, self.state, torch.tensor([float(self.first)]),
                                                          deterministic=self.deterministic)
            probs = policy._distribution.distribution.probs[0] if trace else None
        self.first = False
        action = int(action.item())
        if not trace:
            return action
        top = torch.topk(probs, 5)
        return action, dict(value=round(float(value.item()), 4),
                            top=[[int(a), round(float(p), 4)] for a, p in zip(top.indices.tolist(), top.values.tolist())])



AWAY_GAIN = 4  # act-v2's retreat walked to a move endpoint at least this much farther from the threat


def click_away(task, attackers):
    """The combat reference's retreat under act-v3 (privileged): the legal click whose route ends farthest (Chebyshev)
    from the nearest attacker, at least AWAY_GAIN farther than the player stands (ties to the lower slot); else WAIT."""
    w, p, obs = task.world, task.agent.player, task.agent.last_observation
    tree = w.player_scene(p)
    away = lambda x, z: min(max(abs(x - n.x), abs(z - n.z)) for n in attackers)
    best, choice = away(p.x, p.z) + AWAY_GAIN - 1, A_WAIT
    for k, c in enumerate(obs['clicks']):
        if obs['mask'][CLICK_ROWS[k]]:
            route = paths.client_route(tree, p.x, p.z, (p.x + c['dx'], p.z + c['dz']), nearest=True)
            if route and away(*route[-1]) > best:
                best, choice = away(*route[-1]), CLICK_ROWS[k]
    return choice

class ControlActor:
    """WAIT, uniformly random legal actions, or a reference style ('reference:sustain' ...)."""

    def __init__(self, policy):
        self.policy = policy
        self.reference = Reference(policy.split(':', 1)[1]) if policy.startswith('reference:') else None

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        if self.reference:
            self.reference.reset(seed)

    def act(self, observation, task=None, trace=False):
        if self.policy == 'wait':
            action = A_WAIT
        elif self.policy == 'random':
            action = int(self.rng.choice(np.flatnonzero(observation['mask'])))
        else:
            action = self.reference.action(task)
        return (action, None) if trace else action


def records(split, per_group, key, first, tables, only=None):
    """The frozen evaluation records: per_group starts of each group, record k on the split's layout first + k (one
    layout each), with task.draw_start(key, 3, k) and draw_combat(..., group). Returns (layouts, records) as dicts by
    record index; `only` limits both to those indices (a part generates only its own layouts)."""
    from sim.task import draw_start
    from sim.world import generate, split_seeds
    seeds = split_seeds(split, per_group * len(GROUPS), first)
    wanted = range(len(seeds)) if only is None else only
    pool = {k: generate(seeds[k], tables) for k in wanted}
    out = {}
    for k in wanted:
        group = GROUPS[k // per_group]
        start = draw_start(key, 3, k, pool, layout=k)
        start['combat'] = draw_combat(key, 3, k, pool[k], start, tables, group=group)
        out[k] = start
    return pool, out


# A decision row of play(), in this order. The first nine are those of the 20 October campaign's records; act-v3 has
# one escape, the clicks, where act-v2 rows had the moves and RETREAT offered.
DECISION_FIELDS = ('tick', 'action', 'option', 'reward', 'hp', 'action_delay', 'eat_delay', 'ended', 'note', 'hp_before',
                   'food_before', 'food_after', 'style', 'clicks_offered')


def food_count(raw):
    return sum(i.get('count', 1) for i in raw['inventory'] if i['name'] in FOOD_NAMES)


def play(task, actor, seed, trace=False, limit=2400):
    """One trial: (decision rows, diagnostic rows or None, the final info). A row (DECISION_FIELDS): tick, action, the
    ACT or HELD option, reward, hitpoints after, action and eat delays left, the ended option's reason, the engine's note
    on what it dispatched; then hitpoints and food before and food after, the effective combat style after (com_mode as
    [proc,player_combat_stat] clamps it) and how many clicks the mask offered for escape when the action was chosen."""
    from sim.agent import family
    actor.reset(seed)
    obs, rows, diag, info = task.projected, [], [] if trace else None, None
    p = task.agent.player
    for _ in range(limit):
        tick, raw = task.world.tick, task.raw
        hp_before, food_before = p.hp, food_count(raw)
        offered = [int(np.sum(obs['mask'][list(CLICK_ROWS)]))]
        if trace:
            action, extra = actor.act(obs, task, trace=True)
            diag.append(dict(tick=tick - task.start_tick, legal=np.flatnonzero(obs['mask']).tolist(), hp=p.hp,
                             food=sum(1 for i in raw['inventory'] if i['name'] in FOOD_NAMES),
                             worn=[e['name'] for e in raw['equipment']],
                             npcs=[(n['name'], n['distance'], n['inCombat']) for n in raw['npcs'][:4]],
                             budget=round(float(obs['budget_vec'][0]), 4), **(extra or {})))
        else:
            action = actor.act(obs, task)
        meaning = family(action)
        option = raw['heldSel'][action - A_HELD0]['option'] if meaning == 'held' and action - A_HELD0 < len(raw['heldSel']) \
            else raw['actSel'][action - A_ACT0]['option'] if meaning == 'act' and action - A_ACT0 < len(raw['actSel']) else None
        obs, reward, done, info = task.step(action)
        ended = task.raw['feedback'].get('ended') if not done else None
        style = task.raw['style']
        rows.append([tick - task.start_tick, action, option, round(float(reward), 6), p.hp,
                     max(0, p.varps['action_delay'] - task.world.tick), max(0, p.varps['eat_delay'] - task.world.tick),
                     ended and ended['reason'], task.raw.get('actionNote'), hp_before, food_before, food_count(task.raw),
                     min(style['mode'], style['count'] - 1), *offered])
        if done:
            return rows, diag, info
    return rows, diag, None


def evaluate(output, *, checkpoint=None, initial=None, policies=('final', 'initial', 'random', 'wait'), split='development',
             per_group=50, key=11, first=0, part=(0, 1), diagnostics=2, seed=303001, tables=None, sampled=False, previous=None):
    """Every policy on the same frozen records (records()), each trial in a fresh world. part=(i, n): this process plays
    records i, i + n, ... (merge() joins the parts). For checkpoint policies (final, initial, previous: another
    checkpoint on the same records) the first `diagnostics` records of each group and every failed trial keep each
    decision's legal actions, observation summary, five likeliest actions and value. Writes results-{i}.json and
    trials-{i}.jsonl.gz."""
    import gzip
    import json
    import time
    from pathlib import Path
    from sim.data import Tables
    from sim.engine import Unsupported
    from sim.world import LayoutTables
    from baseline.telemetry import sha256
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = tables or Tables.load()
    t0 = time.monotonic()
    mine = range(part[0], per_group * len(GROUPS), part[1])
    pool, recs = records(split, per_group, key, first, tables, only=mine)
    paths = dict(final=checkpoint, initial=initial, previous=previous)
    rows, timing = [], {}
    stream = gzip.open(output / f'trials-{part[0]}.jsonl.gz', 'wt')
    for policy in policies:
        actor = PolicyActor(paths[policy], deterministic=not sampled) if policy in paths else ControlActor(policy)
        t1 = time.monotonic()
        for k in mine:
            rec = recs[k]
            group, index = rec['combat']['group'], k % per_group
            task = episode(rec, pool[k], LayoutTables(tables, pool[k]), tables)
            trace = policy in paths
            base = dict(policy=policy, record=k, group=group, layout=pool[k].seed)
            try:
                decisions, diag, info = play(task, actor, seed + k, trace=trace)
            except Unsupported as error:  # counted apart, never scored
                rows.append(dict(base, unsupported=str(error)))
                stream.write(json.dumps(dict(base, start=rec, unsupported=str(error)), default=str) + '\n')
                continue
            if info is None:
                rows.append(dict(base, decision_limit=True))
                continue
            paid = sum(r[3] for r in decisions)
            if abs(paid - info['episode']['r']) > 1e-6:
                raise RuntimeError('paid rewards disagree with the audited return')
            evidence = fights(task)
            row = dict(base, success=info['episode']['r'] == 1.0, episode_return=info['episode']['r'],
                       survived=info['report']['survived'], reason=info['report']['reason'], kills=info['outcomes'],
                       decisions=info['episode']['l'], ticks=info['episode']['native_ticks'],
                       requested=dict(eat=sum(r[2] == 'Eat' for r in decisions), attack=sum(r[2] == 'Attack' for r in decisions),
                                      click=sum(r[1] in CLICK_ROWS for r in decisions), wait=sum(r[1] == A_WAIT for r in decisions)),
                       **evidence['summary'])
            rows.append(row)
            keep = index < diagnostics or not row['success']  # a fixed sample of each group, and every failure
            stream.write(json.dumps(dict(base, start=rec, outcome=row, report=info['report'], decisions=decisions,
                                         fights=evidence['fights'], heals=evidence['heals'], eaten=evidence['eaten'],
                                         kill_ticks=evidence['kill_ticks'], diagnostic=diag if keep else None),
                                    default=str) + '\n')
        timing[policy] = time.monotonic() - t1
        print(f'[combat] part {part[0]}/{part[1]}: {policy} {timing[policy]:.0f}s', flush=True)
    stream.close()
    from sim.agent import HELD_CONTRACT, MASK_CONTRACT
    from sim.engine import RETALIATE_CONTRACT
    result = dict(version=COMBAT['version'], split=split, per_group=per_group, key=key, first=first, part=list(part), sampled=sampled,
                  mask=MASK_CONTRACT, retaliate=RETALIATE_CONTRACT, held=HELD_CONTRACT, decision_fields=list(DECISION_FIELDS),
                  diagnostics=diagnostics, seed=seed, groups=list(GROUPS), policies=list(policies),
                  checkpoints={n: dict(path=str(p), sha256=sha256(p)) for n, p in paths.items() if p and n in policies},
                  layout_seeds=[pool[k].seed for k in mine], timing_seconds=timing, wall_seconds=time.monotonic() - t0,
                  trials=rows)
    (output / f'results-{part[0]}.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return result


def merge(output):
    """Join an evaluation's parts (results-*.json) into results.json: the trials, then per policy and group the
    successes with a Wilson interval, deaths and the recovery evidence (eats with a gain, eats followed by a certified
    kill, regen gains followed by one, aggressor fights). Refuses missing or duplicate records."""
    import json
    from pathlib import Path
    from baseline.compare import wilson
    output = Path(output)
    parts = [json.loads(p.read_text()) for p in sorted(output.glob('results-*.json'))]
    if not parts:
        raise ValueError(f'no parts in {output}')
    first = parts[0]
    n = first['part'][1]
    if sorted(p['part'][0] for p in parts) != list(range(n)) or any(p['part'][1] != n for p in parts):
        raise ValueError('parts missing or of different splits')
    for key in ('version', 'split', 'per_group', 'key', 'first', 'policies', 'seed', 'checkpoints', 'sampled', 'mask',
                'retaliate', 'held'):  # a part with no `held` (held-v1) differs from a held-v2 part
        if any(p.get(key) != first.get(key) for p in parts):
            raise ValueError(f'parts differ in {key}')
    trials = sorted((t for p in parts for t in p['trials']), key=lambda t: (first['policies'].index(t['policy']), t['record']))
    total = first['per_group'] * len(GROUPS)
    for policy in first['policies']:
        got = sorted(t['record'] for t in trials if t['policy'] == policy)
        if got != list(range(total)):
            raise ValueError(f'{policy}: records missing or repeated')
    cells = {}
    for policy in first['policies']:
        for group in (*GROUPS, 'all'):
            mine = [t for t in trials if t['policy'] == policy and group in (t['group'], 'all')]
            scored = [t for t in mine if 'success' in t]
            ok = sum(t['success'] for t in scored)
            s = lambda k: sum(t.get(k, 0) for t in scored)
            cells.setdefault(policy, {})[group] = dict(
                trials=len(mine), scored=len(scored), unsupported=sum('unsupported' in t for t in mine),
                decision_limit=sum('decision_limit' in t for t in mine), successes=ok, interval=wilson(ok, len(scored)),
                deaths=sum(not t['survived'] for t in scored), mean_kills=round(s('kills') / max(1, len(scored)), 3),
                eats=s('eats'), eats_then_kill=s('eats_then_kill'), eat_requests=sum(t['requested']['eat'] for t in scored),
                eats_without_gain=s('eats_without_gain'), regen_gain=s('regen_gain'), regen_then_kill=s('regen_then_kill'),
                trials_eat_then_kill=sum(t.get('eats_then_kill', 0) > 0 for t in scored),
                trials_regen_then_kill=sum(t.get('regen_then_kill', 0) > 0 for t in scored),
                aggressed=s('aggressed'), received=s('received'), mugger_kills=sum(t['by_type']['mugger']['kills'] for t in scored),
                valid=len(scored) == len(mine))
    result = {k: v for k, v in first.items() if k not in ('part', 'trials', 'layout_seeds', 'timing_seconds', 'wall_seconds')}
    result.update(parts=n, timing_seconds={p['part'][0]: p['timing_seconds'] for p in parts}, cells=cells, trials=trials)
    (output / 'results.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return result


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    ev = sub.add_parser('evaluate', help='policies and controls on the frozen grouped combat records (one part)')
    ev.add_argument('--checkpoint', type=Path, help='policy `final`')
    ev.add_argument('--initial', type=Path, help='policy `initial`: the run\'s actual starting checkpoint')
    ev.add_argument('--previous', type=Path, help='policy `previous`: another checkpoint on the same records')
    ev.add_argument('--policies', nargs='+', default=['final', 'initial', 'random', 'wait'])
    ev.add_argument('--split', choices=('development', 'final'), default='development')
    ev.add_argument('--final', action='store_true', help='the final split is final evaluation data: say so')
    ev.add_argument('--per-group', type=int, default=50)
    ev.add_argument('--key', type=int, default=11)
    ev.add_argument('--first', type=int, default=0, help='the split\'s seed index of record 0\'s layout')
    ev.add_argument('--part', type=int, nargs=2, default=(0, 1), metavar=('I', 'N'))
    ev.add_argument('--diagnostics', type=int, default=2, help='records per group with full checkpoint diagnostics')
    ev.add_argument('--tables', type=Path, help='the simulator archive as given (its pins unchecked)')
    ev.add_argument('--sampled', action='store_true', help='sample the checkpoints\' actions')
    ev.add_argument('--seed', type=int, default=303001, help='the sampling-seed base: record k samples under seed + k')
    ev.add_argument('--output', type=Path, required=True)
    mg = sub.add_parser('merge', help='join the parts of an evaluation')
    mg.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'merge':
        result = merge(args.output)
        print(json.dumps({p: {g: (c['successes'], c['scored'], c['deaths']) for g, c in cells.items()}
                          for p, cells in result['cells'].items()}, indent=1))
        return
    if (args.split == 'final') != args.final:
        parser.error('the final split needs --final, and --final needs the final split')
    from sim.data import Tables
    evaluate(args.output, checkpoint=args.checkpoint, initial=args.initial, previous=args.previous,
             policies=tuple(args.policies), split=args.split,
             per_group=args.per_group, key=args.key, first=args.first, part=tuple(args.part), diagnostics=args.diagnostics,
             tables=Tables.load(args.tables, check_pins=False) if args.tables else None, sampled=args.sampled, seed=args.seed)


if __name__ == '__main__':
    main()
