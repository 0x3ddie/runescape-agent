# runescape-agent

A reinforcement learning (RL) agent that learns to play RuneScape on its own. Its only
reward is the levels it gains.

Not affiliated with or endorsed by Jagex Ltd. RuneScape, its game art and its names are
the property of Jagex Ltd.

![One agent, one life, mostly mining](docs/media/one-life.gif)

*One agent, one life of about 3.5 hours of game time, sped up about 230x. Nobody told it to
mine. [Why it mines so much](#why-does-the-agent-in-the-clip-mine-so-much) is explained below.*

## What is this?

I'm training an agent to play old RuneScape (the 2004 version) by itself. There are no
scripts telling it what to do and no human recordings to copy. It sees roughly what a
player standing there would see, it picks actions like "click that tree", and it gets a
reward when its skill levels go up. That's the only learning signal. Over millions of
actions it keeps the habits that earned reward and drops the ones that didn't. That's
reinforcement learning in a nutshell.

Basically the question is: if you drop a fresh agent into the game and only reward
"get better at stuff", what does it figure out by itself?

**So what has it figured out?** At first I trained one agent per skill, each with a fixed
goal like "cut 8 logs in 3 minutes". That worked: the trained agents reached their goal in
95% or more of tries in my simulator, against 0 to 13% before training. Then I took the
goals away. Now one agent plays whole lives and is rewarded only for the levels it gains.
By itself it learned to cut trees and fight, and in the simulator it gained more levels
than my best hand-written script. Once I rewarded every skill, gave it a tinderbox (the
item that lights fires) and let it explore more for a while, it also started burning its
logs and burying bones. But then it got stuck, and it never tried mining, even though it
carried a pickaxe the whole time.

**How did it start mining?** I added a small exploration bonus for experience points (XP)
in skills it has rarely trained lately. The bonus doesn't name any skill, and it shrinks as
that kind of XP becomes common. With the bonus, the agent tried mining. Then I switched the
bonus off, and it kept mining and got better at it. In the clip above it spends 70% of a
life mining, from mining level 16 to 44, without dying once. The bonus got it to try, and
the normal level reward made it stick.

**What doesn't work yet?** It doesn't fish or cook, and it almost never eats when it's
hurt. The mining result comes from two training runs, not a big study, and it only shows
up when a life starts at high levels. Deaths also went up, though the counts are small.

**Why a simulator?** The real game server is slow for this: one agent in one world on my
laptop makes about 36 decisions a second. So I rewrote the game's rules in C, which trains
at about 1,900 decisions a second on the same laptop and about 26,500 on a rented cloud
GPU (graphics processor).
Later I got the real server fast enough to train in too, by putting hundreds of agents in
the same game worlds. Both are on the [speed chart](#how-fast-it-trains) below.

## Terms used here

- **Agent** (or **policy**): the program that plays. It's a small neural network with a
  short-term memory (an LSTM, long short-term memory, a common type of recurrent network).
  It's trained with PPO (Proximal Policy Optimization), a standard RL training method.
- **Tick**: the game's clock. One tick is 0.6 seconds, so 6,000 ticks is an hour of game
  time.
- **Decision** (or action): one choice by one agent, like "click that tree" or "attack that
  chicken". The agent has 96 possible actions, and the game tells it which ones are
  possible right now.
- **Observation** (what it sees): things within 15 tiles, its inventory, the open menu and
  the game's text messages. No map, no wiki.
- **Skill, XP and level**: RuneScape has skills like Woodcutting, Mining and Attack. Doing
  a skill earns XP, and enough XP raises that skill's level.
- **Life** (an episode, in RL terms): from when an agent appears in the world until it dies
  or hits a time limit of 1 to 6 hours of game time. Every life starts with some food, an
  axe, a pickaxe and a fishing net, and sometimes a weapon.
- **Reward**: what the agent is trained to get more of. Here it's the levels gained in a
  life, counted in fractions (halfway to the next level counts as half a level). Every
  reward is checked against the game's own XP records.
- **Update**: the agent plays a batch of 65,536 decisions, then learns from them once.
  A training run is a few dozen updates.
- **Seed**: one training run, started from its own random numbers. Two seeds means the
  same recipe trained twice. If both agree, the result is less likely to be luck.
- **Simulator** vs **real game**: the simulator is my C rewrite of the game's rules,
  which is fast. The real game is the actual server code
  ([rs-sdk](https://github.com/MaxBittker/rs-sdk), an open-source copy of the 2004 game
  server). The agent is always tested in the real game.
- **Script**: a simple hand-written player I compare against, like "attack the nearest
  chicken forever". Scripts are only yardsticks. The agent never learns from them.

## How the reward shapes what it does

The reward is just "levels gained". The catch is that each level costs more XP than the
one before:

| Skill level | XP needed for the next level |
| --- | --- |
| 1 | 80 |
| 16 | 231 |
| 30 | 607 |
| 44 | 1,595 |
| 50 | 2,412 |

So the same item gives less reward the higher that skill is. One iron ore (35 XP) is
worth 0.15 levels at mining level 16 but only 0.02 at level 44. In practice the agent
drifts toward wherever levels are cheapest right now, and that explains most of what it
does:

- **Woodcutting**: trees are near every start and it carries an axe. At low levels this is
  the easiest reward around, so a fresh agent cuts wood in practically every life.
- **Firemaking**: once it had a tinderbox, its own logs became extra reward.
- **Fighting**: creatures are everywhere, and each fight trains two skills at once (a
  combat skill and Hitpoints).
- **Burying bones**: fights drop bones, and burying them trains Prayer, which starts at
  level 1, so it's cheap.
- **Fishing and cooking**: fishing spots are usually out of view (more than 15 tiles away),
  so it never stumbles on them. Even with a fishing spot in view at the start, only 1 of
  60 lives caught five fish or more. No fish means no cooking.
- **Eating**: health gives no reward. Food only matters because dying ends the life early,
  and that's a weak, delayed signal. It mostly eats at full health, if it eats at all.

### Why does the agent in the clip mine so much?

Three reasons:

1. **It started at high levels.** The clip's life started at the levels of a long life
   (woodcutting 50, firemaking 52, combat skills 30 to 37), where those skills are
   expensive, while its mining was only 16. One iron ore was worth about 15 times as much
   reward as one log.
2. **It started right next to a mine.**
3. **Training with the exploration bonus taught it that mining exists.** Before that
   bonus, policies started at the same high levels fought about 90% of the time and never
   mined once.

This edge doesn't last forever. Each mining level costs more than the last: a script that
only mines beats those fighting policies over the first 10,000 ticks or so, but not over
30,000.

### Can we turn the knobs to get more of something?

Yes. These are the knobs I've tried so far, and one I haven't:

| Knob | What it changes | Tried? | What happened |
| --- | --- | --- | --- |
| Start levels | which skills are cheap | yes | high starting levels: fights about 90% of the time |
| Start location | what's in view at the start | yes | a fishing spot in view: 1 of 60 lives caught 5+ fish |
| Which skills count | 7 skills, then every skill plus a tinderbox | yes | fires and bones showed up (with more exploration) |
| Exploration strength (an entropy bonus, which rewards keeping its action choices random) | how much it tries new actions | yes | 5x for a while helped fires and bones, not mining |
| Rarity bonus (a count-based exploration bonus) | extra reward for XP in skills it rarely trains | yes | brought mining in both seeds, and it stayed after removal |
| Death penalty (minus 50 levels when it dies) | how much dying hurts | yes | no change in deaths |
| Weights per skill (say, woodcutting counts double) | how much each skill is worth | no | not built yet |

**More woodcutting?** The surest knob is to make woodcutting levels cheap again: start lives
at low woodcutting, near trees. A fresh agent already cuts wood in practically every life. Weighting
woodcutting higher in the reward would also work, but that's me picking its hobby for it,
which is the opposite of the point.

**More exploring?** The two exploration knobs (entropy and rarity) both pushed it into new
skills. A bonus for visiting new places isn't built yet, and it's probably the next thing
to try for fishing. The risk with any bonus is that the agent games it. An earlier "first
time using this item on that thing" bonus got farmed: the agent earned five times more
bonus than real levels. So a bonus has to fade over time, and every result is judged on the
plain level reward alone.

## Try the trained agent

The final policy of the 21 November release (the miner from the clip) is on the
[GitHub release](https://github.com/0x3ddie/runescape-agent/releases) as four files:

| File | What it is |
| --- | --- |
| `runescape-agent-2026-11-21.zip` | the policy, 37,581,569 bytes (about 36 MB), sha256 `3c8508e2a27f2e0d79ce1e9dbd27d2c3ff7493c884dc4ecf122dba54a1e4b1c5` |
| `runescape-agent-2026-11-21.names.json` | a fingerprint of the name table the policy trained with |
| `runescape-agent-2026-11-21.launch.json` | the training settings that the demo and the watch read |
| `runescape-agent-2026-11-21.sha256` | checksums of the three files above |

The agent reads item and creature names as text embeddings (one vector per name), built
from the game's own files. I can't share those, so the zip contains none of them and no
file paths. You rebuild the name table from your own copy of the game, and the loader
fills it in. It accepts your table if it's identical to mine, or if it matches the
fingerprint (same names in the same order, and nearly the same numbers). Otherwise it
refuses and tells you why. I've only tested this on one Mac, so a table rebuilt on
another machine is untested. The copy can play, but it can't be used to start training.

1. Set up the game server and the two Python environments as in
   [docs/RUNNING.md](docs/RUNNING.md).
2. Rebuild the name table. `scripts/name_vectors.py` turns each name into a vector with
   all-MiniLM-L6-v2, a small open text-embedding model:

   ```sh
   bun scripts/cache_names.ts logs/names/cache-names.json
   .venv/bin/python scripts/name_vectors.py logs/names/cache-names.json logs/names/name-vectors.npz
   ```

3. Download the four files into one folder and check them. Keep the `.names.json` file
   next to the zip: the loader looks for it there.

   ```sh
   shasum -a 256 -c runescape-agent-2026-11-21.sha256
   ```

4. Play it with `sim.real demo` (one world you watch live in the browser game client) or
   `sim.watch`, passing
   `--real-checkpoint runescape-agent-2026-11-21.zip --launch runescape-agent-2026-11-21.launch.json`
   ([docs/RUNNING.md](docs/RUNNING.md#watching-a-demo)).

## Quick start

```sh
git clone https://github.com/0x3ddie/runescape-agent.git
cd runescape-agent
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r baseline/requirements.txt
.venv/bin/python -m pytest
```

That runs the learner and task tests with a small fake name table. Everything else needs
the game server and files built from its game data, which this repository doesn't
include. [docs/RUNNING.md](docs/RUNNING.md) says how to build them, run every test suite,
train and evaluate.

## Charts and numbers

### How fast it trains

![Decisions a second in each setup](docs/media/speed.svg)

The real game on a laptop is slow (gray bars: playing only, no learning). Putting many
agents into the same game worlds helps a lot, and training in the real game on a rented
cloud machine runs at about 580 decisions a second. The C simulator trains at about 1,900
on the laptop and about 26,500 on a cloud GPU, roughly 45 times the real game's training
speed. That gap is why most early work happened in the simulator. Now the real game is
fast enough to train in too, so those runs can't pick up any mismatch between the
simulator and the real game.

### Main milestones

![Four milestones, each with its own yardstick](docs/media/milestones.svg)

1. **One skill at a time** (October to 12 November). Each agent was told its goal and
   rewarded for the items it made and kept. Trained agents reached the goal in 95 to 99% of
   simulator trials.
2. **No goal, rewarded for levels** (17 November). The score during training, for five
   seeds. All five learn the same way, which suggests the recipe is stable.
3. **Against simple players** (17 November). After training, the agent gains 85 to 89
   levels a life across the five seeds. Random play gets 11, the best one-skill script 58,
   and my best hand-written script 76.
4. **The rarity bonus** (21 November, real game). The share of lives that mined five ores
   or more: almost none without the bonus, 14 to 15% with it, and 15% and 48% after it was
   switched off.

### How I measure it

- **Tests are fixed before training.** The pass marks, the test starts and the numbers
  I'll read are written down before a run starts, so I can't move the goalposts after.
- **The agent plays with its normal randomness.** It samples each action from its learned
  probabilities, the same way it plays during training, instead of always taking its top
  pick.
- **Unseen starting spots.** The one-skill tests start in places the agent never trained
  on.
- **Yardsticks.** Every score is shown next to random play and simple scripts on the same
  starts.
- **The real game.** Agents trained in the simulator are always checked in the real game.
  There's also a fixed test (the "watch"): the same four places and settings every time.
- **The game's records check the reward.** An audit compares every rewarded level with the
  game's own XP records, so the agent can't get reward for something that didn't happen.
- **Two or more seeds.** A result counts only when the seeds agree.

## Changelog

### 21 November 2026: goal-free play, and a skill it found by itself

The clip at the top: one agent, one life of 21,073 ticks, about 230x normal speed
([video](docs/media/one-life.mp4)). It started at a long life's levels and spent 69.5% of its
time mining, 15.9% fighting and 12.0% cutting wood, with fires and bones in between: mining
level 16 to 44, 683 ores, 105 logs, 57 fires and 24 bones buried, with no death.

What changed:

- The agent has no goal anymore. One policy plays whole lives and is rewarded for the
  levels it gains, across every skill in the game. It still only sees what a player
  standing there could see.
- Training moved into the real game server. Many copies of one policy share each game
  world. The world pauses whenever an agent needs a new action, and while the learner
  updates. The C simulator stays as its tested twin.
- The rarity bonus: a small extra reward for XP of a kind that's still rare for the agent.
  It falls as that kind of XP becomes common, and it doesn't name any skill.

What happened: the agent learned four skills by itself, then stopped there. In one seed's
lives near the end of training, 100% cut wood, 95% made fires, 69% fought and 37% buried
bones. It didn't take up mining even when its lives started at high levels, where a rock's
first ores give a lot of reward, or when I turned up exploration. With the rarity bonus,
lives started at those levels mined five ores or more in 15.1% and 13.6% of cases (two
seeds), against 0.7% and 0.0% without it. After 25 more updates with the bonus switched
off, mining stayed and grew, to 15.1% and 47.7%, and those lives gained more levels than
lives that never had the bonus.

The same agent also changes its focus depending on its situation, with no new goal and no
script. From a fresh start it cuts and burns wood and fights. Started at a long life's
levels, where those skills give little reward, it becomes a fighting specialist and spends
90 to 93% of its time fighting. A short phase with the rarity bonus (25 updates, then 25
more without it, about 45 minutes each on one T4 GPU) made it a mining specialist, as in
the clip.

Limits: two seeds; mining only shows up on the high-level starts (lives at normal starting
levels still rarely mine); deaths went up, on small counts (19 and 10 against 9 and 7 with
the bonus, 20 and 23 against 9 and 7 after it). The agent doesn't fish or cook, and it
almost never eats when hurt.

### 12 November 2026: one skill at a time

The first release trained one policy per skill (woodcutting, fishing, mining and melee
combat), each told its goal, in the C simulator, then tested it in the real game. Each clip
is one test in the real game.

| Wood | Fish |
| --- | --- |
| ![Wood trial](docs/media/wood.gif) | ![Fish trial](docs/media/fish.gif) |
| **Mine** | **Melee** |
| ![Mine trial](docs/media/mine.gif) | ![Melee trial](docs/media/melee.gif) |

Each skill has a fixed task with a pass mark set before training. These numbers come from
the saved test runs, with the agent sampling its actions, on starts it never trained on.
"Items" is the average per test, misses included. "Ticks" is the median tick at which a
test reached its goal, among the tests that did.

In the simulator (200 starts per seed and test pass, 800 for melee):

| Skill and goal | Policy | Items per test | Reached the goal | Median ticks to the goal |
| --- | --- | --- | --- | --- |
| Wood: 8 logs in 300 ticks | Untrained | 1.3 logs | 0% | - |
|  | Phase 1 | 11.7 logs | 95% | 202 |
|  | Clicks | 10.3 logs | 80% | 219 |
| Fish: 2 fish in 240 ticks | Untrained | 0.3 fish | 4% | 105 |
|  | Phase 1 | 3.0 fish | 99% | 89 |
| Mine: 2 ore in 240 ticks | Untrained | 0.7 ore | 13% | 144 |
|  | Phase 1 | 3.3 ore | 98% | 94 |
|  | Clicks (1 seed) | 2.7 ore | 87% | 115 |
| Melee: 5 kills in 1,200 ticks | Untrained (1 seed) | 0.9 kills | 2% | 1,011 |
|  | Phase 1 | 9.6 kills | 96% | 449 |
|  | Clicks (1 seed) | 7.7 kills | 90% | 609 |

In the real game (32 tests per seed and test pass, at two places per skill):

| Skill and goal | Policy | Items per test | Reached the goal |
| --- | --- | --- | --- |
| Wood: 8 logs in 300 ticks | Untrained | 0.8 logs | 0.3% |
|  | Phase 1 | 14.2 logs | 99% |
|  | Clicks | 13.5 logs | 94% |
| Fish: 2 fish in 240 ticks | Phase 1 | 1.9 fish | 65% |
| Mine: 2 ore in 240 ticks | Phase 1 | 2.3 ore | 79% |
|  | Clicks (1 seed) | 1.4 ore | 52% |

"Phase 1" is the first full training round: five seeds per skill, each tested twice.
"Clicks" is a later version of the controls, where every action is a click, like a
player's (the wood one is also five seeds). "Untrained" is the starting policy before any
training (for melee, the start of the click run). The wood test here uses a sparse map
where only one tree is left near the start. Every run uses 25 times the normal XP rate.
There's no fair real-game test for melee yet. The limits on all of this are in
[docs/NOTES.md](docs/NOTES.md#results-in-detail).

## Repository layout

- `sim/`: the simulators (a Python reference and its C port in `sim/native/`), the tasks
  and rewards, the reward audit, the learner, training in the simulator and in the real
  game (`sim/real.py`), the fixed watch (`sim/watch.py`) and the test commands.
- `baseline/`: the policy network, real-game testing and bookkeeping.
- `rl/`: the definitions of what the agent sees and can do (observations, actions, items
  and names).
- `env-ts/`: the code that runs inside the real game server (TypeScript, Bun), including
  the shared game worlds.
- `scripts/`: starting the game worlds, building the name table, exporting a policy, a map
  tool and a server speed check.
- `vendor-patches/`: our patches to the game server and how to build it.
- `docs/`: notes, the running guide, the clips and the charts.

## More docs

- [docs/NOTES.md](docs/NOTES.md): how I got here, from the first agent to goal-free play,
  and every result of the 12 November release with its limits.
- [docs/RUNNING.md](docs/RUNNING.md): building the inputs, the tests, training (in the
  simulator and in the real game), the watch and the demo.

## Credits and licence

- [rs-sdk](https://github.com/MaxBittker/rs-sdk) and the [Lost City](https://github.com/LostCityRS)
  engine it builds on (MIT). The patches change their source, and `sim/` and `env-ts/`
  port or call parts of it; see [NOTICE](NOTICE).
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and
  [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib), Python
  RL libraries (MIT).

This repository's own code is under the licence in [LICENSE](LICENSE). Game assets, art
and names are not covered: they are the property of Jagex Ltd.
