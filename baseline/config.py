"""Task, world and learner settings shared by training and evaluation.

The short/long development fixtures match the September 2026 pilot so results
stay comparable with it; the rest add the start states that need recovery.
"""
import os

# A village is a bank spawn tile beside the booths; that is all the environment gets.
# The policy finds banks and trees among what it observes within 15 tiles (act-v3). Seven
# banks train: the four free-to-play ones and three members' ones with plain trees near
# a ground-floor booth. Al Kharid has no trees and trains fishing; the Fishing Guild's trees sit behind a
# door that needs Fishing 68, which the collision snapshot treats as open, so a chop
# there was re-issued 584 times. Both Varrock banks never train but were
# used to diagnose failures, so they are development villages. Seers' and Catherby
# are the frozen test set: report their results, never choose a design, checkpoint
# or hyperparameter from them.
VILLAGES = dict(
    draynor=dict(spawnTele='0,48,50,20,43'),
    falador=dict(spawnTele='0,46,52,3,40'),
    varrock=dict(spawnTele='0,49,53,49,48'),
    falador_east=dict(spawnTele='0,47,52,5,27'),
    edgeville=dict(spawnTele='0,48,54,22,35'),
    varrock_east=dict(spawnTele='0,50,53,53,28'),
    yanille=dict(spawnTele='0,40,48,53,19'),
    ardougne_north=dict(spawnTele='0,40,52,55,4'),
    ardougne_south=dict(spawnTele='0,41,51,31,19'),
    seers=dict(spawnTele='0,42,54,36,37'),
    catherby=dict(spawnTele='0,43,53,57,49'),
    al_kharid=dict(spawnTele='0,51,49,5,31'),
)
# Drop-in starts (October 9): the held-out test once the bank villages ran out. In the
# map files every ground-floor bank with a usable booth is already a training, development
# or test village, or the Fishing Guild. So the policy starts where it has never been, with
# the starter kit and usually no bank in sight. One rule chose the points from the map files
# before any policy ran there: level 0, x 2368-3455, z 2880-3519 (south of the wilderness),
# at least 64 tiles from every village above and the Fishing Guild, at least 100 tiles
# walkable within 20, at least 48 tiles apart, in a seeded order (mulberry32, 20261009).
# drop1-12 have a group of plain trees (3 within 6 tiles) the player can reach within 10
# tiles (no net spot qualified); bare1-4 have no plain tree or net spot within 30.
# Report them; never choose a design, checkpoint or setting from them.
DROP_IN = dict(
    drop1=dict(spawnTele='0,47,49,22,33'),  # 3030,3169
    drop2=dict(spawnTele='0,39,49,34,2'),  # 2530,3138
    drop3=dict(spawnTele='0,39,47,39,50'),  # 2535,3058
    drop4=dict(spawnTele='0,41,54,36,4'),  # 2660,3460
    drop5=dict(spawnTele='0,51,52,62,15'),  # 3326,3343
    drop6=dict(spawnTele='0,40,46,20,32'),  # 2580,2976
    drop7=dict(spawnTele='0,39,54,18,37'),  # 2514,3493
    drop8=dict(spawnTele='0,39,50,55,11'),  # 2551,3211
    drop9=dict(spawnTele='0,38,53,28,9'),  # 2460,3401
    drop10=dict(spawnTele='0,42,52,25,36'),  # 2713,3364
    drop11=dict(spawnTele='0,49,50,34,59'),  # 3170,3259
    drop12=dict(spawnTele='0,38,49,42,20'),  # 2474,3156
    bare1=dict(spawnTele='0,52,50,23,10'),  # 3351,3210
    bare2=dict(spawnTele='0,37,47,27,47'),  # 2395,3055
    bare3=dict(spawnTele='0,52,48,9,21'),  # 3337,3093
    bare4=dict(spawnTele='0,46,47,31,29'),  # 2975,3037
)
# Search starts (October 11): development points chosen from the map files before any policy
# ran there: the drop-in rule's box and spacing, at least 64 tiles from every village above, the
# Fishing Guild and every DROP_IN point; no plain tree or net spot within 32 tiles, but a group
# of plain trees 33-60 tiles away that a walk of at most 90 steps reaches (seed 20261011). The
# rule missed other trees: 11 of the 12 have dead trees or evergreens in view, so they
# test unfamiliar places, not search; only search3 has nothing to chop within 32 tiles.
# Unlike DROP_IN they may inform design; the start archive keeps away from them.
SEARCH = dict(
    search1=dict(spawnTele='0,50,49,5,38'),  # 3205,3174; nearest grouped tree 3157,3205 at 53 steps
    search2=dict(spawnTele='0,37,49,4,44'),  # 2372,3180; nearest grouped tree 2419,3139 at 47 steps
    search3=dict(spawnTele='0,52,51,2,12'),  # 3330,3276; nearest grouped tree 3294,3331 at 80 steps
    search4=dict(spawnTele='0,45,49,7,5'),  # 2887,3141; nearest grouped tree 2944,3148 at 60 steps
    search5=dict(spawnTele='0,46,54,27,11'),  # 2971,3467; nearest grouped tree 2990,3415 at 56 steps
    search6=dict(spawnTele='0,44,54,57,27'),  # 2873,3483; nearest grouped tree 2907,3485 at 33 steps
    search7=dict(spawnTele='0,38,51,16,17'),  # 2448,3281; nearest grouped tree 2484,3276 at 37 steps
    search8=dict(spawnTele='0,41,46,22,12'),  # 2646,2956; nearest grouped tree 2617,2997 at 40 steps
    search9=dict(spawnTele='0,41,49,30,19'),  # 2654,3155; nearest grouped tree 2631,3198 at 56 steps
    search10=dict(spawnTele='0,37,50,1,43'),  # 2369,3243; nearest grouped tree 2309,3194 at 63 steps
    search11=dict(spawnTele='0,47,54,11,17'),  # 3019,3473; nearest grouped tree 3029,3523 at 67 steps
    search12=dict(spawnTele='0,48,52,31,24'),  # 3103,3352; nearest grouped tree 3105,3313 at 44 steps
)
# Hidden starts (October 11, before any pilot result): nothing a player can chop and no fishing
# spot within 32 tiles, level-1 trees (Tree, Dead tree, Evergreen) in a group 33-60 tiles away,
# at least 64 tiles from every village, DROP_IN and SEARCH point and every cell the pilot's
# archive arm trained in (seed 20261012). Only these four passed the rule.
HIDDEN = dict(
    hidden1=dict(spawnTele='0,46,46,29,21'),  # 2973,2965; nearest grouped tree 2919,3002 at 65 steps
    hidden2=dict(spawnTele='0,43,50,22,1'),  # 2774,3201; nearest grouped tree 2828,3177 at 69 steps
    hidden3=dict(spawnTele='0,45,46,43,7'),  # 2923,2951; nearest grouped tree 2919,3002 at 51 steps
    hidden4=dict(spawnTele='0,45,47,30,50'),  # 2910,3058; nearest grouped tree 2919,3002 at 75 steps
)
# Teacher-only labels: where to gather in each labelled village, per skill. One rule
# chose the trees from the map: among plain trees within 25 tiles of the bank that stand
# with at least two others within 6 tiles, the fewest walking steps from a booth. Net
# spots move between a few tiles; the point is the middle of those seen from the bank,
# and only Draynor, Al Kharid and Catherby have one within 32 tiles of the bank. Only
# baseline.scripted.Teacher reads them; the policy never does, and the test villages
# have none, so a policy deployed anywhere uses no hand-set place.
GATHER_AREAS = dict(
    woodcutting=dict(
        draynor=(3099, 3247), falador=(2947, 3356), falador_east=(3002, 3356), edgeville=(3113, 3502),
        varrock=(3168, 3413), varrock_east=(3240, 3441),
        yanille=(2621, 3109), ardougne_north=(2626, 3333), ardougne_south=(2679, 3289)),
    fishing=dict(draynor=(3086, 3229), al_kharid=(3268, 3147)),
)
# Engine reset configuration: item-v2 inventory/bank pages, four gather targets (any
# loc or npc with a gathering option), the 24 clicks to walk by.
# Every reset grants the 18-item starter kit (bronze axe included) and 25 banked coins.
# Combat 23: Draynor's dark wizards (level 7) and imps attack combat 14 and below,
# and killed a fresh combat-3 account within ~120 decisions once npcs could see it.
TASK_CFG = dict(VILLAGES['draynor'], contract='item-v2', speedMs=48, maxDecisions=4000,
                maxEpisodeTicks=12000, actTargets='act-v3',
                setStats='attack:20,strength:20,defence:20,hitpoints:20',
                # A decision takes 2 ticks plus 0-1 at random, as a browser client's did.
                paceJitterTicks=1,
                # Ask the policy only when an option ends or something happens
                # (env-ts/core/decision-points.ts), not every 2-3 ticks.
                decisionPoints=True)
# A spawn away from the bank: the road 4 tiles from a Draynor jail guard spawn; a guard
# (level 26) killed an agent walking back. Lumbridge, where a dead player respawns, is
# ~130 tiles from Draynor, outside every place the client loads, so it waits for a map.
SPAWNS = dict(jail='0,48,50,48,34')

# Reward is the game price of the certified produced items kept, in coins, / REWARD_SCALE, paid
# as the value held changes and settled to that total at the end (baseline/resource_task.py);
# nothing else pays. 40 is ten logs.
REWARD_SCALE = 40

# World that scripts/stack.py boots.
STATUS_URL = 'http://127.0.0.1:43610/status'
SPEED_MS = 48
XP_RATE = int(os.environ.get('RL_XP_RATE', 25))  # set by scripts/stack.py --xp-rate; 25 unless evaluating at 1
AGENT_NAME = 'upstream1'

# Horizon is in native game ticks. Capacity is the state scripted setup leaves:
#   ample       only the axe in the inventory
#   starter     the starter kit stays, so only 10 slots are free
#   tool-banked everything, the tool included, is in the bank
#   respawn     starter kit, no setup; used with the jail spawn and far start states
#   both-tools  axe and small fishing net in the inventory, the rest of the kit banked
# skill (default woodcutting) names the tool setup keeps. kind picks the success criterion
# in baseline/evaluate.py. reset_seed 0 starts on the spawn tile; other seeds start on a
# tile within 3 that walks to it (env-ts/spawn_tiles.ts).
TRAIN_FIXTURES = [
    dict(id='train-short', kind='short', horizon=240, capacity='ample', reset_seed=0),
    dict(id='train-long', kind='long', horizon=3600, capacity='ample', reset_seed=0),
    dict(id='train-starter', kind='long', horizon=3600, capacity='starter', reset_seed=0),
    dict(id='train-axe-banked', kind='long', horizon=3600, capacity='tool-banked', reset_seed=0),
    dict(id='train-guard', kind='long', horizon=3600, capacity='respawn', spawn='jail', reset_seed=0),
    # Other villages, so a policy has to use what it sees rather than one place.
    *(dict(id=f'train-{name}-{village}'.replace('_', '-'), kind=kind, horizon=horizon, capacity=capacity,
           village=village, reset_seed=0)
      for village in ('falador', 'falador_east', 'edgeville', 'yanille', 'ardougne_north', 'ardougne_south')
      for name, kind, horizon, capacity in (('short', 'short', 240, 'ample'), ('long', 'long', 3600, 'ample'),
                                            ('starter', 'long', 3600, 'starter'),
                                            ('axe-banked', 'long', 3600, 'tool-banked'))),
    # Start 25-45 walking steps from the bank, picked by reset seed, with the starter kit
    # (10 free slots). Imitation waited beside trees 30 tiles from a bank and, full, with a
    # booth 40 steps away: places the teacher, gathering by its label, rarely went.
    *(dict(id=f'train-far-{village}'.replace('_', '-'), kind='long', horizon=3600, capacity='respawn',
           village=village, spawn_steps=(25, 45), reset_seed=1)
      for village in ('draynor', 'falador', 'falador_east', 'edgeville', 'yanille', 'ardougne_north', 'ardougne_south')),
    # Fishing. Draynor's trees are labelled too, so there the fixtures hold only the net:
    # with both tools the teacher chops (scripted.SKILLS).
    *(dict(id=f'train-fish-{name}-{village}'.replace('_', '-'), kind=kind, horizon=horizon, capacity=capacity,
           village=village, skill='fishing', reset_seed=0)
      for village, variants in (('draynor', ('short', 'long')), ('al_kharid', ('short', 'long', 'starter', 'tool-banked')))
      for name, kind, horizon, capacity in (('short', 'short', 240, 'ample'), ('long', 'long', 3600, 'ample'),
                                            ('starter', 'long', 3600, 'starter'),
                                            ('tool-banked', 'long', 3600, 'tool-banked'))
      if name in variants),
    dict(id='train-fish-far-al-kharid', kind='long', horizon=3600, capacity='respawn', village='al_kharid',
         skill='fishing', spawn_steps=(25, 45), reset_seed=1),
]
# Goal-conditioned short tasks (goal-v1, baseline/resource_task.py GOAL_SCORING): the same Draynor
# start, kit, bank and spawn for both goals; only the requested activity differs. Training
# alternates them in balanced random order (baseline/task_schedule.py). Never mixed with the
# value-scored fixtures above: the goal is an observation input those checkpoints lack.
GOAL_FIXTURES = [dict(id=f'train-goal-{goal}', kind='goal', horizon=240, capacity='both-tools', goal=goal, reset_seed=0)
                 for goal in ('wood', 'fish')]
EVAL_FIXTURES = dict(
    short=dict(id='dev-short', kind='short', horizon=270, capacity='ample', reset_seed=0),
    long=dict(id='dev-long', kind='long', horizon=3800, capacity='ample', reset_seed=0),
    # Both tools at Draynor: the teacher chops there, though fishing keeps more (the RL test).
    starter=dict(id='dev-starter', kind='long', horizon=3800, capacity='starter', reset_seed=0),
    axe_banked=dict(id='dev-axe-banked', kind='long', horizon=3800, capacity='tool-banked', reset_seed=0),
    guard=dict(id='dev-guard', kind='long', horizon=3800, capacity='respawn', spawn='jail', reset_seed=0),
    falador=dict(id='dev-falador', kind='long', horizon=3800, capacity='ample', village='falador', reset_seed=0),
    falador_east=dict(id='dev-falador-east', kind='long', horizon=3800, capacity='ample', village='falador_east',
                      reset_seed=0),
    edgeville=dict(id='dev-edgeville', kind='long', horizon=3800, capacity='ample', village='edgeville', reset_seed=0),
    yanille=dict(id='dev-yanille', kind='long', horizon=3800, capacity='ample', village='yanille',
                reset_seed=0),
    ardougne_north=dict(id='dev-ardougne-north', kind='long', horizon=3800, capacity='ample', village='ardougne_north',
                       reset_seed=0),
    ardougne_south=dict(id='dev-ardougne-south', kind='long', horizon=3800, capacity='ample', village='ardougne_south',
                       reset_seed=0),
    # Never trained; used to diagnose, so development only.
    varrock=dict(id='dev-varrock', kind='long', horizon=3800, capacity='ample', village='varrock', reset_seed=0),
    varrock_east=dict(id='dev-varrock-east', kind='long', horizon=3800, capacity='ample', village='varrock_east',
                      reset_seed=0),
    # Frozen test set (September 28): see VILLAGES.
    seers=dict(id='test-seers', kind='long', horizon=3800, capacity='ample', village='seers', reset_seed=0),
    catherby=dict(id='test-catherby', kind='long', horizon=3800, capacity='ample', village='catherby', reset_seed=0),
    # Fishing: both training villages, and Catherby's net spots as the frozen test.
    fish_draynor=dict(id='dev-fish-draynor', kind='long', horizon=3800, capacity='ample', skill='fishing',
                      reset_seed=0),
    fish_al_kharid=dict(id='dev-fish-al-kharid', kind='long', horizon=3800, capacity='ample', village='al_kharid',
                        skill='fishing', reset_seed=0),
    fish_catherby=dict(id='test-fish-catherby', kind='long', horizon=3800, capacity='ample', village='catherby',
                       skill='fishing', reset_seed=0),
    # Drop-in test (see DROP_IN): the starter kit, no bank interaction at setup.
    **{name: dict(id=f'test-{name}', kind='drop', horizon=3800, capacity='respawn', village=name, reset_seed=0)
       for name in DROP_IN},
    # Search and hidden starts (see SEARCH, HIDDEN): development, the starter kit, no bank setup.
    **{name: dict(id=f'dev-{name}', kind='drop', horizon=3800, capacity='respawn', village=name, reset_seed=0)
       for name in (*SEARCH, *HIDDEN)},
    # The oak task (its check was removed), development data: plain trees and the bank in view at
    # Edgeville and no oak in the first observation, but ordinary play shows oaks within about 300
    # ticks (several stands 10-40 tiles away, two across the wilderness line). Run each trial on a
    # fresh world.
    edgeville_oak=dict(id='dev-edgeville-oak', kind='long', horizon=3600, capacity='respawn', reset_seed=0,
                       spawnTele='0,48,54,35,50'),
    # The two Draynor short training starts themselves (240 ticks, one tool each, same spawn), for
    # fresh PPO's local skill test: familiar starts, not generalization.
    short_axe=dict(next(f for f in TRAIN_FIXTURES if f['id'] == 'train-short')),
    short_net=dict(next(f for f in TRAIN_FIXTURES if f['id'] == 'train-fish-short-draynor')),
    # The goal-v1 training starts themselves (familiar starts): goal checkpoints only.
    goal_wood=dict(GOAL_FIXTURES[0]),
    goal_fish=dict(GOAL_FIXTURES[1]),
)
# Native start sets of the four-goal task (decision of October 15). No
# pinned map site has a tree, net spot, chicken and level-1 rock within the 32-tile place view, so:
#   arena       Entrana with six rocks added by vendor-patches/arena (run on that engine build): all four goals
#   entrana     the unmodified island: wood, fish, melee
#   rimmington  the unmodified Rimmington mine: wood, mine
# All three are final evaluation data. Before the final evaluation only the reference feasibility check,
# WAIT and the weapon check run there (baseline.evaluate refuses them without --final); mechanics and
# refusal checks use development sites (baseline/control_check.py MECHANICS).
# A start record: the set's tile (reset seed 0) or a connected tile within 3 picked by the reset seed
# (env-ts/core/world.ts jitterWorld), START_KIT plus one START_WEAPONS weapon carried, nothing worn or banked
# (TaskCfg kit 'none'), and base levels from START_LEVELS (sim/spec.py), drawn once with numpy
# default_rng(20261015) and frozen here. The same records serve every goal of a set.
# For the reference player only (baseline/scripted.py GoalReference), a set lists each activity's area,
# waypoints from the start and the level-1 rocks; nothing of it reaches a policy.
ENTRANA = (2859, 3342)  # minimax walking steps to tree 5, oak 14, net spot 30, chicken 31 (baked collision)
ENTRANA_AREAS = dict(wood=(2862, 3348), fish=(2875, 3342), mine=(2856, 3342), melee=(2852, 3370))
# The chickens' pen has a double gate, closed at boot; a trial that opens it leaves it open for a while.
ENTRANA_GATES = dict(melee=((2854, 3370), (2854, 3371)))
ENTRANA_ROUTES = dict(melee=[(2857, 3360), (2856, 3366)])
# Arena rocks (vendor-patches/arena): copper 2860,3348 and 2860,3351; tin 2854,3337 and 2862,3351; iron
# 2852,3340 and 2851,3338. Each is 3-13 walking steps from every start; none changes a walking distance
# from a start to the nearest tree, oak, net spot or chicken, nor cuts off a tile. scripts/arena_planner.py adds the
# same six tiles to the arena clone's SDK planner map.
ARENA_ROCK_TILES = ((2860, 3348), (2860, 3351), (2854, 3337), (2862, 3351), (2852, 3340), (2851, 3338))
START_SETS = dict(
    arena=dict(goals=('wood', 'fish', 'mine', 'melee'), engine='arena', start=ENTRANA, areas=ENTRANA_AREAS,
               routes=ENTRANA_ROUTES, gates=ENTRANA_GATES, rocks=((2860, 3348), (2860, 3351), (2854, 3337), (2862, 3351))),
    entrana=dict(goals=('wood', 'fish', 'melee'), engine='pinned', start=ENTRANA, areas=ENTRANA_AREAS,
                 routes=ENTRANA_ROUTES, gates=ENTRANA_GATES, rocks=()),
    rimmington=dict(goals=('wood', 'mine'), engine='pinned', start=(2998, 3238),
                    areas=dict(wood=(2999, 3245), mine=(2984, 3237)), routes={},
                    rocks=((2976, 3247), (2977, 3247), (2978, 3248), (2979, 3248), (2977, 3245), (2984, 3237),
                           (2986, 3235), (2986, 3239), (2987, 3240))),
)
# The engine checkout a set needs (git tree of vendor/rs-sdk): the pinned series, or it plus the arena patch.
# The arena tree includes its SDK planner map (scripts/arena_planner.py); the arena patch alone does not.
ENGINE_TREES = dict(pinned='6d7092ef18f5796a3985573697d52efbd1cf090c', arena='6e50ebf12e5a06686edb9229f58459cd79d62454')
START_KIT_ITEMS = 'bronze_axe:1,net:1,bronze_pickaxe:1,shrimp:2,bread:1'  # sim/spec.py START_KIT
START_DRAWS = [  # (setStats, weapon) per start record
    ('woodcutting:16,fishing:5,mining:10,attack:6,strength:7,defence:5,hitpoints:10', 'bronze_dagger'),
    ('woodcutting:15,fishing:1,mining:10,attack:8,strength:9,defence:1,hitpoints:11', 'bronze_sword'),
    ('woodcutting:14,fishing:16,mining:1,attack:9,strength:10,defence:10,hitpoints:11', 'bronze_sword'),
    ('woodcutting:1,fishing:20,mining:10,attack:5,strength:6,defence:7,hitpoints:11', 'bronze_dagger'),
    ('woodcutting:5,fishing:10,mining:16,attack:6,strength:9,defence:8,hitpoints:11', 'bronze_dagger'),
    ('woodcutting:15,fishing:14,mining:20,attack:1,strength:5,defence:9,hitpoints:10', 'bronze_sword'),
    ('woodcutting:20,fishing:10,mining:14,attack:3,strength:2,defence:6,hitpoints:10', 'bronze_sword'),
    ('woodcutting:15,fishing:15,mining:20,attack:9,strength:4,defence:10,hitpoints:11', 'bronze_dagger'),
]


def start_records(name):
    x, z = START_SETS[name]['start']
    return [dict(start=f'{name}-{k}', spawnTele=f'0,{x >> 6},{z >> 6},{x & 63},{z & 63}', reset_seed=k,
                 set_stats=stats, seed_items=f'{START_KIT_ITEMS},{weapon}:1')
            for k, (stats, weapon) in enumerate(START_DRAWS)]


# One final fixture family per start set and supported goal: <set>_<goal>, 240 native ticks (sim/spec.py).
START_SET_FIXTURES = {f'{name}_{goal}': dict(id=f'final-{name}-{goal}', kind='goal', horizon=240, capacity='declared',
                                             goal=goal, start_set=name, final=True, starts=start_records(name))
                      for name, start_set in START_SETS.items() for goal in start_set['goals']}
EVAL_FIXTURES.update(START_SET_FIXTURES)

# The native transfer study. A study start record is drawn as the simulator draws a
# start (sim/spec.py): each listed base level uniformly and independently from START_LEVELS, one START_WEAPONS weapon
# carried, START_KIT; and a reset seed in 1..999,999 that picks the start tile within 3 (env-ts/core/world.ts
# jitterWorld). The goal is not an input to the draw. Evaluation records: STUDY_EVAL_RECORDS draws from
# default_rng(STUDY_EVAL_SEED), shared by every start set, goal, group and checkpoint (an intermediate evaluation
# takes the first STUDY_INTERMEDIATE_TRIALS). Training records: STUDY_TRAIN_RECORDS draws from
# default_rng(STUDY_TRAIN_SEED), none equal to an evaluation record; a training slot picks one per episode
# (baseline/train.py). Both lists are frozen by their hashes in the study manifest.
STUDY_EVAL_SEED, STUDY_EVAL_RECORDS, STUDY_INTERMEDIATE_TRIALS = 20261019, 200, 50
STUDY_TRAIN_SEED, STUDY_TRAIN_RECORDS = 20261020, 4096


def study_draws(seed, count, exclude=()):
    import numpy as np
    from sim.spec import START_KIT, START_LEVELS, START_WEAPONS
    rng, draws = np.random.default_rng(seed), []
    kit = ','.join(f'{name}:{n}' for name, n in START_KIT)
    while len(draws) < count:
        stats = ','.join(f'{skill}:{int(rng.choice(levels))}' for skill, levels in START_LEVELS.items())
        draw = dict(set_stats=stats, seed_items=f'{kit},{START_WEAPONS[int(rng.integers(len(START_WEAPONS)))]}:1',
                    reset_seed=int(rng.integers(1, 1_000_000)))
        if draw not in exclude:
            draws.append(draw)
    return draws


def study_records(name, kind, count=None):
    """The study's start records ('eval' or 'train') of start set `name`, in their frozen order; `count`: only the
    first ones (the draws are taken in order, so they are the same records)."""
    evaluation = study_draws(STUDY_EVAL_SEED, STUDY_EVAL_RECORDS)
    draws = evaluation[:count] if kind == 'eval' else \
        study_draws(STUDY_TRAIN_SEED, STUDY_TRAIN_RECORDS if count is None else count, evaluation)
    x, z = START_SETS[name]['start']
    return [dict(start=f'{name}-{kind}-{k}', spawnTele=f'0,{x >> 6},{z >> 6},{x & 63},{z & 63}', **draw)
            for k, draw in enumerate(draws)]


# Evaluation fixtures of the study: study_<set>_<goal>, the evaluation records in order, final data.
STUDY_FIXTURES = {f'study_{name}_{goal}': dict(id=f'study-{name}-{goal}', kind='goal', horizon=240, capacity='declared',
                                               goal=goal, start_set=name, final=True, starts=study_records(name, 'eval'))
                  for name, start_set in START_SETS.items() for goal in start_set['goals']}
EVAL_FIXTURES.update(STUDY_FIXTURES)
# The native demonstration of the sustained-search policies (a diagnostic, not a
# decision): the wood goal for 300 native ticks, the horizon those policies trained with, on the first records of the
# study's training draws (study_records(name, 'train'), never evaluation data) at the pinned start sets. They differ in
# supply: Entrana's nearest stand is two trees 5 steps away (an oak 5-8) and no tree is among the 12 nearest objects, so
# continued production needs an oak (woodcutting 15) or a walk; Rimmington's nearest stand is five trees 3 steps away
# among many more.
SUPPLY_DEMO_RECORDS = 16
SUPPLY_DEMO_FIXTURES = {
    f'supply_{name}_wood': dict(id=f'supply-demo-{name}-wood', kind='goal', horizon=300, capacity='declared', goal='wood',
                                start_set=name, final=False,
                                starts=[dict(record, start=record['start'].replace('-train-', '-demo-'))
                                        for record in study_records(name, 'train', SUPPLY_DEMO_RECORDS)])
    for name in ('entrana', 'rimmington')}
EVAL_FIXTURES.update(SUPPLY_DEMO_FIXTURES)
# The native check of every phase 1 skill gate: CHECK_RECORDS start records drawn as a study start is
# (study_draws, START_KIT plus the weapon carried), from default_rng(CHECK_SEED) and none equal to an earlier native
# record (earlier_native_draws: the study's evaluation and training draws, which hold the 24 October supply
# demonstration's, and START_DRAWS of the 17 October transfer), frozen here as (setStats, weapon, reset seed) and in
# CHECK_DIGESTS, the sha256 of each site's records in baseline/study.py records_digest's form; baseline/tests check both
# against the draw. As the study's records, the same draws serve every site and skill, each site with its own start
# tile; each skill plays them at its two sites (CHECK_SITES) for its task's horizon. Final data, never trained on. Melee
# has no native check: no faithful native sustained-combat fixture was found.
CHECK_SEED, CHECK_RECORDS = 20261026, 16
CHECK_SITES = dict(wood=('entrana', 'rimmington'), fish=('arena', 'entrana'), mine=('arena', 'rimmington'))
CHECK_HORIZONS = dict(wood=300, fish=240, mine=240)  # supply-wood-v1's and the short task's native ticks
CHECK_DRAWS = [
    ('woodcutting:15,fishing:16,mining:10,attack:7,strength:4,defence:10,hitpoints:10', 'bronze_dagger', 380722),
    ('woodcutting:1,fishing:1,mining:14,attack:10,strength:8,defence:4,hitpoints:12', 'bronze_sword', 762871),
    ('woodcutting:20,fishing:10,mining:14,attack:3,strength:6,defence:7,hitpoints:10', 'bronze_dagger', 569963),
    ('woodcutting:15,fishing:10,mining:20,attack:9,strength:7,defence:1,hitpoints:12', 'bronze_dagger', 965997),
    ('woodcutting:14,fishing:15,mining:10,attack:7,strength:8,defence:7,hitpoints:10', 'bronze_dagger', 358211),
    ('woodcutting:14,fishing:20,mining:15,attack:10,strength:5,defence:1,hitpoints:11', 'bronze_dagger', 183284),
    ('woodcutting:1,fishing:1,mining:20,attack:4,strength:3,defence:8,hitpoints:12', 'bronze_sword', 737595),
    ('woodcutting:14,fishing:1,mining:10,attack:5,strength:3,defence:10,hitpoints:12', 'bronze_sword', 273079),
    ('woodcutting:1,fishing:20,mining:15,attack:6,strength:2,defence:8,hitpoints:12', 'bronze_dagger', 188720),
    ('woodcutting:14,fishing:5,mining:15,attack:5,strength:5,defence:4,hitpoints:10', 'bronze_dagger', 596549),
    ('woodcutting:1,fishing:10,mining:16,attack:7,strength:2,defence:5,hitpoints:10', 'bronze_sword', 252074),
    ('woodcutting:15,fishing:14,mining:14,attack:5,strength:5,defence:7,hitpoints:12', 'bronze_dagger', 281412),
    ('woodcutting:16,fishing:14,mining:10,attack:4,strength:7,defence:4,hitpoints:11', 'bronze_sword', 214635),
    ('woodcutting:5,fishing:15,mining:5,attack:5,strength:4,defence:5,hitpoints:11', 'bronze_sword', 661698),
    ('woodcutting:14,fishing:16,mining:14,attack:8,strength:8,defence:8,hitpoints:12', 'bronze_sword', 262092),
    ('woodcutting:20,fishing:1,mining:15,attack:2,strength:9,defence:6,hitpoints:10', 'bronze_sword', 357934),
]
CHECK_DIGESTS = dict(arena='65d72e1a8054013487166a5cfb8be029a5a68195a2ecf533a5d872ae652d50a4',
                     entrana='9ae6cb681b952f5153c871e8b2646e52b1fdc297c1f47e5322b70fc62468f3e3',
                     rimmington='dffab944ceab6174f3783e1acc5db784472802dc2e781bee3ec38e5d20e6a076')


def earlier_native_draws():
    """Every native start draw used before the check records, as study_draws' dicts."""
    evaluation = study_draws(STUDY_EVAL_SEED, STUDY_EVAL_RECORDS)
    transfer = [{k: r[k] for k in ('set_stats', 'seed_items', 'reset_seed')} for r in start_records('entrana')]
    return evaluation + study_draws(STUDY_TRAIN_SEED, STUDY_TRAIN_RECORDS, evaluation) + transfer


def check_records(name):
    """The native check's start records at start set `name`, in their frozen order."""
    x, z = START_SETS[name]['start']
    return [dict(start=f'{name}-check-{k}', spawnTele=f'0,{x >> 6},{z >> 6},{x & 63},{z & 63}', set_stats=stats,
                 seed_items=f'{START_KIT_ITEMS},{weapon}:1', reset_seed=reset_seed)
            for k, (stats, weapon, reset_seed) in enumerate(CHECK_DRAWS)]


def verify_check_records():
    """The check records as frozen: equal to their draw, disjoint from every earlier native draw and hashing to
    CHECK_DIGESTS; raises ValueError otherwise. Returns the digests."""
    import hashlib
    import json
    earlier = earlier_native_draws()
    drawn = study_draws(CHECK_SEED, CHECK_RECORDS, earlier)
    frozen = [{k: r[k] for k in ('set_stats', 'seed_items', 'reset_seed')} for r in check_records('entrana')]
    digests = {site: hashlib.sha256(json.dumps(check_records(site), sort_keys=True).encode()).hexdigest() for site in CHECK_DIGESTS}
    if frozen != drawn or any(r in earlier for r in frozen) or digests != CHECK_DIGESTS:
        raise ValueError('the native check records differ from their draw or their frozen digests')
    return digests


CHECK_FIXTURES = {f'check_{site}_{goal}': dict(id=f'check-{site}-{goal}', kind='goal', horizon=CHECK_HORIZONS[goal],
                                               capacity='declared', goal=goal, start_set=site, final=True,
                                               starts=check_records(site))
                  for goal, sites in CHECK_SITES.items() for site in sites}
EVAL_FIXTURES.update(CHECK_FIXTURES)  # after every earlier fixture, which keep their trial seeds (baseline/evaluate.py)
# The same sites, skills and horizons on development records, for rehearsals of the gate command (sim/gate.py
# --records development): the first CHECK_RECORDS training draws of the paused study, which trained policies already
# played on 24 October (SUPPLY_DEMO_FIXTURES), never evaluation data. A rehearsal never plays the check records.
REHEARSAL_FIXTURES = {f'rehearse_{site}_{goal}': dict(
    id=f'rehearse-{site}-{goal}', kind='goal', horizon=CHECK_HORIZONS[goal], capacity='declared', goal=goal, start_set=site,
    final=False, starts=[dict(record, start=record['start'].replace('-train-', '-rehearse-'))
                         for record in study_records(site, 'train', CHECK_RECORDS)])
    for goal, sites in CHECK_SITES.items() for site in sites}
EVAL_FIXTURES.update(REHEARSAL_FIXTURES)
# A watched demo's start (scripts/stack.py --demo-spectator; never evaluation data):
# rehearse_entrana_wood's first record in the streets of Falador, chosen from the map files on 5 November so that a click
# policy walks before it chops. Nothing in the woodcutting configs' tree list within 18 tiles; the first reachable tree 18
# walking steps away with every Gate and Door shut (2953,3363); the nearest npc with a hunt mode 44 tiles away; at least 64
# tiles from every DROP_IN point, Seers, Catherby and Rimmington; the spectator's waiting spawn wakes none of its npc
# chunks. Reset seed 0 keeps the exact tile.
# Only a --demo process adds it to EVAL_FIXTURES (baseline/evaluate.py), so every other fixture list stays as it was.
DEMO_FIXTURES = dict(demo_far_wood=dict(
    id='demo-far-wood', kind='goal', horizon=CHECK_HORIZONS['wood'], capacity='declared', goal='wood', final=False,
    starts=[dict(REHEARSAL_FIXTURES['rehearse_entrana_wood']['starts'][0], start='demo-far-0', spawnTele='0,46,52,9,53',
                 reset_seed=0)]))
MAX_EVAL_DECISIONS = 2400

TRANSITIONS = 8192
SEED = 1234
# Parallel training agents, each alone in its own world (scripts/stack.py --worlds).
AGENTS = 4
LEARNER = dict(n_steps=1024, batch_size=256, n_epochs=4, learning_rate=.0003, gamma=1., gae_lambda=1.,
               clip_range=.2, ent_coef=.01, vf_coef=.5, max_grad_norm=.5,
               policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, shared_lstm=False, enable_critic_lstm=True,
                                  # The critic has its own extractor, which also sees critic_vec.
                                  share_features_extractor=False,
                                  features_extractor_kwargs=dict(features_dim=128),
                                  net_arch=dict(pi=[64], vf=[64])))
# Weight of KL(reference policy || current policy) in the PPO loss, as in VPT (the reference: --init's policy).
# It keeps fine-tuning from erasing what the warm start learned; 0.2 did not.
KL_COEF = 1.0
# PPO from a pretrained policy. With LEARNER's settings and a random critic, PPO
# overwrote the imitation policy within 30k steps. So: train
# only the critic for critic_warmup steps, then take small, KL-limited steps.
# Fitted on demonstrations alone, the critic explained ~0 of the variance of the
# sampled policy's returns after 16k steps at 3e-5, so it now warms up longer, at
# full rate (only the critic moves then).
# gae_lambda: with gamma 1 and reward paid as earned, lambda 1 would score each action by the
# whole remaining return alone; .95 lets nearby payments assign credit, through the critic.
FINETUNE = dict(learning_rate=3e-5, target_kl=0.01, ent_coef=0.001, critic_warmup=65536, critic_learning_rate=3e-4,
                gae_lambda=.95)
CHECKPOINT_EVERY = 65536

# Actions that move items to or from the bank: DEPOSIT_0..7 and WITHDRAW_0..7.
BANK_ACTIONS = frozenset(range(45, 61))
