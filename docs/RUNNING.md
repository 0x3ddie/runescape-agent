# Running it

How to set up, build the inputs, test, train and evaluate. Everything runs on your own
machine; no service account is needed. The commands run from the root of a clone of
[runescape-agent](https://github.com/0x3ddie/runescape-agent).

## What you need

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- [Bun](https://bun.sh) 1.3.14, for the TypeScript environment and the engine.
- A C compiler and zlib, for the C simulator and the path finder.
- macOS for `sim.build` as written (it copies the engine with macOS's `cp -c`). The rest,
  the C simulator included, also builds on Linux.
- An NVIDIA GPU only for training at full size. The tests and small runs work on a CPU.

## Python environments

Two environments with pinned versions. `.venv` runs the main suite and the native
engine's evaluation; `.venv-sim` runs the simulator.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r baseline/requirements.txt
uv venv --python 3.12 .venv-sim
uv pip install --python .venv-sim/bin/python -r requirements.txt -r baseline/requirements.txt jax==0.7.2
```

## The engine

Clone rs-sdk into `vendor/rs-sdk`, apply the patches and build it, as
[vendor-patches/README.md](../vendor-patches/README.md) says. The checkout holds the
game's data, so `vendor/` is ignored by git; never commit it. Start native worlds only
through `scripts/stack.py`, which owns and stops its processes.

## Inputs built from the game data

The simulator's tables, the name vectors, the layout pools and the C pack all come from
the game data, so they are not in this repository. You build them from your engine
checkout into the ignored `logs/` folder. The simulator commands below use this shell
function:

```sh
simpy() { PYTHONPATH=.:rl .venv-sim/bin/python -m "$@"; }
```

**1. Simulator tables.** `sim.build` runs `sim/extract.ts` inside a copy of the engine
(under `logs/sim/work`) and writes `logs/sim/data/draynor-v3.npz`:

```sh
simpy sim.build
```

Most tests and commands read `logs/sim/data/draynor-nc1.npz`, which is the same archive
without the firemaking and cooking rows. `sim.build` refuses to write that name, so make
it from the first:

```sh
.venv-sim/bin/python - <<'EOF'
import json, numpy as np
with np.load('logs/sim/data/draynor-v3.npz') as d:
    meta = json.loads(d['meta'].tobytes())
    arrays = {k: d[k] for k in d.files if k != 'meta'}
content = meta['content']
for key in ('firemaking', 'cooking'):
    del content[key]
for key in ('fm_level', 'fm_here', 'fm_tinderbox', 'fm_attempt', 'fm_catch', 'cook_cant'):
    del content['texts'][key]
np.savez('logs/sim/data/draynor-nc1.npz', meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
EOF
```

On our own archives this gives a file equal to our `draynor-nc1.npz`, metadata and arrays.
Each archive records hashes of what it was built from (the engine commit, the patches,
the packed data, `sim/extract.ts` and two env-ts files), and the loader refuses an archive
whose hashes differ from your checkout. So rebuild after any change to those files,
including the name in the patch headers.

**2. Name vectors.** These need `transformers` and the all-MiniLM-L6-v2 weights (about
90 MB), which are not in the requirements:

```sh
bun scripts/cache_names.ts logs/names/cache-names.json
.venv/bin/python scripts/name_vectors.py logs/names/cache-names.json logs/names/name-vectors.npz
```

**3. Layout pools.** The training worlds are generated from seeds:

```sh
simpy sim.world pool --split train --count 8 --derived --output logs/sim/world/train-8-nc1-derived.npz
simpy sim.search pool --split train --count 128 --output logs/search/pools/train-128.npz
simpy sim.search supply-pool --split train --count 128 --output logs/supply/pools/train-128.npz
```

**4. The C pack.** The C simulator reads one file per pool set. The tests expect this name:

```sh
simpy sim.native build
simpy sim.native export --pool logs/sim/world/train-8-nc1-derived.npz \
    --supply-pool logs/supply/pools/train-128.npz --tables logs/sim/data/draynor-nc1.npz \
    --output logs/native/train-8-nc1-a6c325a-ids3-item3.pack
```

A pack goes stale when the Python reference changes; export it again after such a change.

**5. Goal-free play.** The goal-free task plays a wider world, `village-world-v2`, which needs the
field plants' pick rows in its own archive, its own layout pool and its own pack (a plain pool, so
the pack has no route trees):

```sh
simpy sim.build --picks --source logs/sim/data/draynor-nc1.npz
simpy sim.world pool --world village-world-v2 --split train --count 128 --first 20000 \
    --output logs/sim/world/train-128-20000-v2.npz
simpy sim.native export --no-trees --pool logs/sim/world/train-128-20000-v2.npz \
    --supply-pool logs/supply/pools/train-128.npz --tables logs/sim/data/draynor-nc2.npz \
    --output logs/native/train-128-20000-v2-a6c325a-ids3-item3.pack
```

These are the commands that made our own inputs, with one difference: our `draynor-v3.npz`
was made from `draynor-nc1.npz` with `sim.build --content-only`, not by a fresh
extraction. We haven't run the whole chain from a clean clone. If a fresh extraction
differs from ours, a test that pins a recorded digest will fail.

## Tests

```sh
.venv/bin/python -m pytest                      # default: learner, task reward, train and evaluate entry points
.venv/bin/python -m pytest baseline/tests rl/tests
PYTHONPATH=.:rl .venv-sim/bin/python -m pytest -o addopts= sim/tests
.venv/bin/ruff check
.venv/bin/vulture
cd env-ts && bun test
```

A test that needs an input you have not built skips and says which file it wants. A skip
is not a pass. What each group needs:

| Tests | Need |
| --- | --- |
| Most of `sim/tests` | `draynor-nc1.npz` |
| `sim/tests/test_fire_cook.py` | `draynor-v3.npz` as well |
| `sim/tests/test_native_*.py`, `test_workers.py` | the pools, and for most the C pack |
| Four of the seven `rl/tests` files, two tests in `baseline/tests` and three in `sim/tests` | `rl/tests/fixtures/reset.json`, a reset observation recorded from the engine; it holds game names, so it is not public |
| Four files in `env-ts` (`reach`, `spawn_tiles`, `core/world`, `engine/resource-native-inventory`) | the engine checkout in `vendor/rs-sdk` |

Some tests read the git commit of the checkout, so run them in a git clone, not in a bare
copy of the files.

In a fresh git clone with no engine checkout and no built inputs (macOS, the pinned
environments, 12 November 2026) the suites gave:

| Suite | Result |
| --- | --- |
| `pytest` (default) | 69 passed, 2 skipped (the native capture) |
| `pytest baseline/tests rl/tests` | 276 passed, 6 skipped (the native capture) |
| `sim/tests` | 114 passed, 306 skipped (303 want the archive, pools or pack; 3 the native capture) |
| `ruff check`, `vulture` | clean |
| `bun test` in `env-ts/` | 221 passed, 6 skipped (the engine checkout); with it, 227 passed |

## Training

A small check on the CPU:

```sh
simpy sim.train train --env cpu --workers 4 --pool logs/sim/world/train-8-nc1-derived.npz \
    --worlds 32 --n-steps 32 --batch-size 256 --decisions 2048 --validate --output logs/sim/runs/check
```

The phase 1 recipe for one skill (here wood, training seed 101) on a GPU, with the C
simulator in 16 worker processes:

```sh
simpy sim.train train --env cpu --engine native --pack logs/native/train-8-nc1-a6c325a-ids3-item3.pack --workers 16 \
    --device cuda --task goal-activity-v5 --goals wood --seed 101 \
    --pool logs/sim/world/train-8-nc1-derived.npz --search-pool logs/supply/pools/train-128.npz \
    --worlds 1024 --n-steps 64 --batch-size 1024 --epochs 4 --decisions 4194304 \
    --validate --recovery logs/runs/wood-s101/recovery --output logs/runs/wood-s101
```

A run writes `initial.zip`, `checkpoints/`, `latest.zip`, metrics and `summary.json`. It
stops before any update whose rollout reached a mechanic the simulator does not model.
`--resume DIR --env-seed S` continues from a recovery folder.

## Evaluation

The skill gate plays one checkpoint (or the WAIT control) through the simulator tests in
two sampled passes and a greedy pass, and writes `gate.json` with every rule's result:

```sh
simpy sim.gate run --checkpoint logs/runs/wood-s101/latest.zip --skills wood --records development \
    --output logs/gate/wood-s101
```

`--records development` reads development records only. The final records are for a
final verdict; read them once (`--records final --final`). `--native --bun "$(command -v bun)"`
adds the native check, which boots fresh worlds of the real engine through
`scripts/stack.py`; the arena site needs a second engine build (see
[vendor-patches/README.md](../vendor-patches/README.md)).

Other commands, each with its usage in the module's first lines: `sim.run goals` (the
four-goal evaluation), `sim.hour run` (an hour of continuous play), `sim.search` (the
wood supply task's records), `sim.check` (the simulator against saved native evidence),
`sim.native gate` (the C port against the Python reference) and `baseline.evaluate`
(native trials).

## Goal-free play

These commands run the 21 November method. Real-engine runs need the engine (see
[The engine](#the-engine)) and `bun`; `scripts/stack.py` starts and stops the engine worlds.

**In the simulator**, the goal-free task on the C engine:

```sh
simpy sim.train train --task free-play-v1 --env cpu --engine native \
    --pack logs/native/train-128-20000-v2-a6c325a-ids3-item3.pack --workers 8 \
    --pool logs/sim/world/train-128-20000-v2.npz --worlds 1024 --n-steps 64 --decisions 1048576 --validate \
    --batch-setting muggers=false 'life=["uniform",6000,36000]' world='"village-world-v2"' --output logs/runs/free-sim
```

`--batch-setting` names `muggers`, `life` (`["uniform",LO,HI]` or `["fixed",T]`, in ticks) and
`world`. `--validate` checks each life's pay against its audited level gain.

**In the real engine**, many agents of one policy share each world. The world stops at every tick
where an agent needs a new action and waits while the learner trains. One update of two worlds
of 128 agents each, from simulator weights:

```sh
NODE_RANDOM_EVENTS=false RL_SHARED_AGENTS=128 PYTHONPATH=.:rl .venv/bin/python scripts/stack.py \
    --bun "$(command -v bun)" --output logs/runs/free-real/stack --timeout-seconds 1800 --lockstep \
    --xp-rate 1 --worlds 2 -- \
    .venv-sim/bin/python -m sim.train train --env real --task free-play-v12 --worlds 1024 --n-steps 64 \
    --batch-size 1024 --epochs 4 --decisions 65536 --seed 1 --validate --weights CHECKPOINT --half-life 6000 \
    --batch-setting 'starts="village-world-v2"' 'life=["uniform",6000,36000]' worlds=2 agents=128 \
    --output logs/runs/free-real/run
```

On a GPU add `--device cuda --collect-device cpu --cudnn-tf32 off`. The task versions in
`sim/workers.py` are, among others: `free-play-v3` (paid over every stat, with a tinderbox in
the start kit), `free-play-v12` (lives start at their drawn hitpoints), `free-play-v14` (every
second agent starts at a long life's levels) and `free-play-v15` (v14 with the rare-XP pay).
`--weights` loads the same version or a declared crossing, such as v12 into v15 and v15 into
v14 (the pay switched off). There is no cloud launcher in this repository; any machine that
runs the engine and the learner can run these.

**The fixed watch** plays one policy in four real-engine worlds at fixed sites and writes the
monitors (levels, deaths and time per skill). First check the settings, which starts nothing:

```sh
.venv/bin/python -m sim.watch check --output logs/watch/w1 --world village-world-v2 --actor policy \
    --real-checkpoint CHECKPOINT --launch LAUNCH \
    --sites port_sarim_north draynor_west falador_farm_west rimmington_coast --rotation 0 \
    --key 20261113 --seed 113 --ticks 30000 --guard 600
```

Then run the same settings as `sim.watch run` inside
`NODE_RANDOM_EVENTS=false PYTHONPATH=.:rl .venv/bin/python scripts/stack.py --bun "$(command -v bun)"
--output logs/watch/w1-stack --timeout-seconds 900 --lockstep --xp-rate 1 --worlds 4 -- ...`.
`--actor` can also be `wait`, `random` or `script-mine` (a scripted miner, a check of what is
possible, never a training label). `--start-levels woodcutting=50,firemaking=52,attack=31,strength=37,defence=30,hitpoints=34`
plays every life at a long life's levels.

### Watching a demo

A demo is one world of the real engine with N agents of one checkpoint, which you watch live in
the web client. It is never evaluation evidence. `LAUNCH` is the training run's `launch.json`:

```sh
NODE_RANDOM_EVENTS=false RL_SHARED_AGENTS=1 PYTHONPATH=.:rl .venv/bin/python scripts/stack.py \
    --bun "$(command -v bun)" --output logs/demo/one/stack --timeout-seconds 1800 --lockstep --xp-rate 1 \
    --worlds 1 --demo-spectator viewer --demo-pace-ms 100 -- \
    .venv-sim/bin/python -m sim.real demo --output logs/demo/one/play --real-checkpoint CHECKPOINT \
    --launch LAUNCH --starts village-world-v2 --agents 1 --sites rimmington_mine_west \
    --start-levels woodcutting=50,firemaking=52,attack=31,strength=37,defence=30,hitpoints=34 \
    --key 20261113 --seed 113 --seconds 2400
```

Open http://127.0.0.1:8888/ and log in with the spectator's name (any password) within ten
minutes. `PYTHONPATH=.:rl .venv-sim/bin/python -m sim.real view --follow 0` keeps the camera on
the agent. `--trace` also writes every observation, and `sim.real demo-reads logs/demo/one/play`
writes the watch's monitors of each life the demo closed. The engine's servers listen on every
network interface, so run a demo on a private network.

With the published weights, `CHECKPOINT` is `runescape-agent-2026-11-21.zip` and `LAUNCH` is
`runescape-agent-2026-11-21.launch.json` from the release page, with
`runescape-agent-2026-11-21.names.json` in the same folder; rebuild the name table first (see the
README). `scripts/export_policy.py CHECKPOINT OUT/STEM` makes the same kind of public copy of your own
real-engine checkpoint: `STEM.zip` with no name vectors and no paths, `STEM.names.json`, `STEM.launch.json`
and `STEM.sha256`. A training run refuses such a copy.
