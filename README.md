# runescape-agent

A reinforcement learning (RL) agent that learns to play RuneScape on its own. Its only
reward is the levels it gains.

Not affiliated with or endorsed by Jagex Ltd. RuneScape, its game art and its names are
the property of Jagex Ltd.

![One agent, one life, mostly mining](docs/media/one-life.gif)

*One agent, one life of about 3.5 hours of game time, sped up about 230x. Nobody told it to
mine. [Why it mines so much](docs/NOTES.md#why-does-the-agent-in-the-clip-mine-so-much).*

Hi! Since July 2026 I've been building an agent that plays old RuneScape (the 2004 version)
by itself. **As of 21 November 2026, one agent of about 2.5 million parameters, rewarded only
for the levels it gains, has taught itself woodcutting, firemaking, fighting and burying
bones, and, with a small exploration bonus, mining.** No scripts, no human recordings and no
goals. All the code is here, and the trained agent is on the
[release page](https://github.com/0x3ddie/runescape-agent/releases) for you to try.

## What is RuneScape?

RuneScape is an online role-playing game. You walk around an open world and train 21 skills,
like Woodcutting, Mining, Fishing and Attack, each from level 1 to 99. Doing a skill earns
experience points (XP), and enough XP raises its level. There's no main quest you have to
finish. Players pick their own goals.

## Why RuneScape?

It's a great testbed for agents that have to figure out what to do on their own:

- **There's no goal.** The game never tells you what to do next.
- **Lives are long.** One life here lasts 1 to 6 hours of game time and thousands of
  decisions.
- **Skills need setups.** Firemaking needs logs and a tinderbox, mining needs a pickaxe and a
  walk to a rock. Nothing pays until all the pieces are there.
- **The agent only sees 15 tiles around it.** Anything farther away doesn't exist to it until it
  wanders close.
- **It's a real game server**, with creatures that attack you and plenty of quirks.

The game itself runs on [rs-sdk](https://github.com/MaxBittker/rs-sdk), an open-source copy
of the 2004 game server, so I can reset worlds, patch the server and read the game state
directly.

## Why RL, and why only outcomes?

I could have had the agent copy a player. I tried that first: it copied a hand-written player
at one village and learned that village, not woodcutting (0 logs at two villages it hadn't
seen). Copying humans would need a big dataset of human play, which I don't have.

RL fits better. The agent makes its own training data by playing, and it starts from nothing,
literally clicking at random. The only signal it gets is the game's own outcome: the levels it
gains. No rule ever picks a skill for it, and the game never acts for it. That makes progress
slower, but everything it does is something it found.

## Where it's at

What it does by itself:

- cuts trees and burns the logs, fights, and buries the bones;
- in the simulator, gains more levels than my best hand-written script (85 to 89 levels a life
  against 76, five seeds);
- shifts its focus with its situation: started at high levels it becomes a fighter, and after a
  short phase with a bonus for rare kinds of XP it becomes a miner.

What it doesn't do yet:

- fish or cook (fishing spots are usually out of its view);
- eat when it's hurt;
- the mining result is two training runs, and only on high-level starts.

## Read more

Everything else is in one long page, [docs/NOTES.md](docs/NOTES.md):

1. [RL in five minutes](docs/NOTES.md#1-rl-in-five-minutes): the basics and the terms used here.
2. [The game and the environment](docs/NOTES.md#2-the-game-and-the-environment): the world,
   what the agent sees and what it can do.
3. [Rewards](docs/NOTES.md#3-rewards): why levels, why it mines, the knobs I've tried and what
   didn't work.
4. [The training setup](docs/NOTES.md#4-the-training-setup): the network, PPO (Proximal Policy
   Optimization), the C simulator and training in the real game.
5. [How I measure it](docs/NOTES.md#5-how-i-measure-it): tests fixed in advance, seeds and
   yardsticks.
6. [Results](docs/NOTES.md#6-results): every milestone with charts and numbers.
7. [What's next](docs/NOTES.md#7-whats-next).

[docs/RUNNING.md](docs/RUNNING.md) is the hands-on guide: building everything, running the
tests, training, and [playing the published agent](docs/RUNNING.md#playing-the-published-weights).

## Quick start

```sh
git clone https://github.com/0x3ddie/runescape-agent.git
cd runescape-agent
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r baseline/requirements.txt
.venv/bin/python -m pytest
```

That runs the learner and task tests with a small fake name table. Everything else needs the
game server and files built from its game data, which this repository doesn't include.
[docs/RUNNING.md](docs/RUNNING.md) says how to build them.

## Changelog

- **21 November 2026: goal-free play, and a skill it found by itself.** No more goals: one agent
  plays whole lives and is rewarded for the levels it gains. Training moved into the real game
  server, with hundreds of agents sharing worlds. A small bonus for rare kinds of XP got it to
  try mining, and it kept mining after the bonus was switched off. The trained agent is on the
  release page. ([details](docs/NOTES.md#goal-free-play-12-to-21-november))
- **12 November 2026: one skill at a time.** One agent per skill (woodcutting, fishing, mining
  and melee combat), each told its goal, trained in the C simulator and tested in the real game.
  All four passed in the simulator; woodcutting also passed in the real game.
  ([details](docs/NOTES.md#phase-1-one-skill-at-a-time-october-to-12-november))

| Wood | Fish | Mine | Melee |
| --- | --- | --- | --- |
| ![Wood trial](docs/media/wood.gif) | ![Fish trial](docs/media/fish.gif) | ![Mine trial](docs/media/mine.gif) | ![Melee trial](docs/media/melee.gif) |

## Repository layout

- `sim/`: the simulators (a Python reference and its C port in `sim/native/`), the tasks and
  rewards, the reward audit, the learner, training in the real game (`sim/real.py`) and the
  fixed test (`sim/watch.py`).
- `baseline/`: the policy network, real-game testing and bookkeeping.
- `rl/`: what the agent sees and can do (observations, actions, items and names).
- `env-ts/`: the code that runs inside the real game server (TypeScript, Bun).
- `scripts/`: starting the game worlds, building the name table, exporting a policy and a few
  tools.
- `vendor-patches/`: my patches to the game server and how to build it.
- `docs/`: the notes, the running guide, the clips and the charts.

## Credits and licence

- [rs-sdk](https://github.com/MaxBittker/rs-sdk) and the [Lost City](https://github.com/LostCityRS)
  engine it builds on (MIT). The patches change their source, and `sim/` and `env-ts/` port or
  call parts of it; see [NOTICE](NOTICE).
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and
  [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib), Python RL
  libraries (MIT).

This repository's own code is under the licence in [LICENSE](LICENSE). Game assets, art and names
are not covered: they are the property of Jagex Ltd.
