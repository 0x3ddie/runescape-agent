"""The arena's SDK planner map: the shipped sdk/collision-data.json plus the arena rocks' collision.

The planner (env-ts/reach.ts, sdk/pathfinding.ts) reads the SDK's baked collision, which the arena patch
(vendor-patches/arena) does not change. This writes the arena clone's copy from two exports of the
engine's own collision (GET /api/exportCollision on worlds booted by scripts/stack.py): the pinned build's
and the arena build's. It refuses unless the two exports differ exactly at the declared rock tiles
(baseline/config.py ARENA_ROCK_TILES) and only by the LOC flag, and unless the shipped map agrees with the
pinned export there; it then applies those tiles' arena flags to the shipped map, so the arena planner
differs from the one the unmodified start sets use by the six rocks alone. The file is Jagex-derived:
it stays in the clone, never in this repository.

    python scripts/arena_planner.py PINNED_EXPORT ARENA_EXPORT SHIPPED OUT
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.config import ARENA_ROCK_TILES  # noqa: E402

LOC = 0x100  # CollisionFlag.LOC


def tiles(data):
    return {(level, x, z): flags for level, x, z, flags in data['tiles']}


def main(pinned_path, arena_path, shipped_path, out_path):
    pinned, arena = (json.loads(Path(p).read_text()) for p in (pinned_path, arena_path))
    raw = Path(shipped_path).read_bytes()
    shipped = json.loads(raw)
    if json.dumps(shipped, separators=(',', ':')).encode() != raw:
        raise SystemExit('the shipped map does not re-serialize byte for byte; refusing to rewrite it')
    p, a, s = tiles(pinned), tiles(arena), tiles(shipped)
    changed = {k for k in p.keys() | a.keys() if p.get(k, 0) != a.get(k, 0)}
    declared = {(0, x, z) for x, z in ARENA_ROCK_TILES}
    if changed != declared or any(a[k] != p.get(k, 0) | LOC for k in declared):
        raise SystemExit(f'the exports differ at {sorted(changed)}, not only by LOC at the declared {sorted(declared)}')
    if any(s.get(k, 0) != p.get(k, 0) for k in declared) or pinned['zones'] != arena['zones'] or \
            sorted(map(tuple, pinned['doors'])) != sorted(map(tuple, arena['doors'])):
        raise SystemExit('the shipped map or the exports disagree outside the rock tiles that matter here')
    rows = shipped['tiles']
    index = {(level, x, z): i for i, (level, x, z, _) in enumerate(rows)}
    for key in sorted(declared):
        if key in index:
            rows[index[key]][3] = a[key]
        else:
            rows.append([*key, a[key]])
    Path(out_path).write_text(json.dumps(shipped, separators=(',', ':')))
    print(json.dumps(dict(rock_tiles={f'{x},{z}': hex(a[(0, x, z)]) for _, x, z in sorted(declared)},
                          tiles_before=len(s), tiles_after=len(rows), out=str(out_path))))


if __name__ == '__main__':
    main(*sys.argv[1:5])
