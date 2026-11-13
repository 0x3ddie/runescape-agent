# Native engine patches

Ten project patches to [MaxBittker/rs-sdk](https://github.com/MaxBittker/rs-sdk), on
upstream commit `160ffcbb5` (content revision 274), plus the arena patch.

They add worker error reporting, player reset, login and browser configuration,
stale-walk cancellation, non-production tick speed control, dormant-NPC handling and
lockstep world advancement. Patches 2 and 4 add and remove temporary tracing; apply all
ten in order. Player reset is not a complete world reset.

## Game data and engine identity

Never commit or publish engine checkouts, packed game data, collision maps, extracted
simulator tables, name vectors or git bundles. This repository does not track
`vendor/rs-sdk` (`.gitignore` ignores `vendor/`); each user clones the engine there. Verify
any rebuilt or reused tree against `ENGINE_TREES` in
[baseline/config.py](../baseline/config.py):

- Pinned base: `6d7092ef18f5796a3985573697d52efbd1cf090c`
- Arena with planner map: `6e50ebf12e5a06686edb9229f58459cd79d62454`

## Build the engine

From the project root, where `vendor/rs-sdk` does not yet exist:

```sh
git clone https://github.com/MaxBittker/rs-sdk.git vendor/rs-sdk
git -C vendor/rs-sdk checkout -b rl-patches 160ffcbb5
(cd vendor/rs-sdk && git am ../../vendor-patches/00*.patch)
git -C vendor/rs-sdk rev-parse HEAD^{tree}
```

Stop unless the tree is the pinned base.

```sh
(cd vendor/rs-sdk && PUPPETEER_SKIP_DOWNLOAD=1 bun install --frozen-lockfile)
(cd vendor/rs-sdk/server/engine && bun install && bun run build)
```

## Arena build and planner map

The arena adds six copper, tin and iron rocks to Entrana; unmodified tests use the
pinned base. Keep it out of the main engine checkout: never apply its patch there, rebuild that
map in place, or switch a checkout's `vendor/rs-sdk` target during a run. `sim.gate` looks for the arena
engine at `logs/vendor-arena/rs-sdk` (ignored) by default; a clone of `vendor/rs-sdk` at
the arena tree can stand there. To build one, make a separate project checkout ARENA (a
second checkout of this repository with its own `logs/`, `.venv` and `vendor/rs-sdk`),
bootstrap its engine as above and, from ARENA's root:

```sh
(cd vendor/rs-sdk && git am ../../vendor-patches/arena/*.patch)
(cd vendor/rs-sdk/server/engine && bun run build)
```

The SDK planner's map must also hold the rocks. A stack's fixed world boots the
`vendor/rs-sdk` of the project whose `scripts/stack.py` runs, so run this export from this
project's root (the pinned engine), then from ARENA's root, one stack at a time. Each
writes `logs/arena-map-export/collision.json` under its own root; `scripts/stack.py`
refuses an existing `--output`, so that folder must be new in each:

```sh
.venv/bin/python scripts/stack.py --bun "$(command -v bun)" --lockstep \
    --timeout-seconds 240 --output logs/arena-map-export -- \
    curl --fail --silent --show-error --max-time 30 \
    --output logs/arena-map-export/collision.json \
    http://127.0.0.1:8888/api/exportCollision
```

From ARENA's root (never onto the pinned map), with PINNED the absolute path of this
project:

```sh
.venv/bin/python scripts/arena_planner.py \
    PINNED/logs/arena-map-export/collision.json logs/arena-map-export/collision.json \
    PINNED/vendor/rs-sdk/sdk/collision-data.json \
    vendor/rs-sdk/sdk/collision-data.json
git -C vendor/rs-sdk add sdk/collision-data.json
git -C vendor/rs-sdk commit -m "arena: add the six rock flags to the SDK planner map"
git -C vendor/rs-sdk rev-parse HEAD^{tree}
```

`arena_planner.py` refuses unless the exports differ exactly at the six rock tiles. The
final tree must be the arena value. Replacing the shipped map with a whole live export
changes unrelated collision and door data. The world service then boots that engine as
`--engine arena=ARENA/vendor/rs-sdk` (`scripts/stack.py`); `sim.gate` takes it with
`--engine pinned=vendor/rs-sdk` beside it, since any `--engine` replaces both of its
defaults. Arena and unmodified-site jobs use separate checkouts; never disable
`baseline.evaluate`'s tree check or modify the main engine to pass it.
