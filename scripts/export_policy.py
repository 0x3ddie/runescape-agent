"""A public copy of a real-engine policy checkpoint: the weights go out with a public release, and their users rebuild
the name table from their own game files.

    PYTHONPATH=.:rl .venv-sim/bin/python scripts/export_policy.py CHECKPOINT OUT/STEM [--launch LAUNCH]

writes OUT/STEM.zip, OUT/STEM.names.json, OUT/STEM.launch.json and OUT/STEM.sha256, and refuses an OUT/STEM that
exists. STEM.zip is the same
stable-baselines3 zip (sim.watch and sim.real demo load it; sim/hour.py load_free_policy) with three changes: its name
tables hold no vectors (every NameEmbedding table zero: the name vectors are Jagex-derived and never published), every
string of its metadata that holds an absolute path is blanked (for example real_leg.first, the run's container folder),
and its episode buffers are empty. STEM.names.json is the fingerprint of the name table the policy trained with
(baseline/policy.py name_fingerprint: rows, dimension, the sha256 of the kinds and names, and per-dimension sums of the
vectors, no vector's values), which must be the table in use (RL_NAME_TABLE, by its digest). STEM.launch.json is the
public form of the training run's launch.json (LAUNCH, by default the one in the run's folder, beside its `free`
folder), which sim.real demo and sim.watch read beside a real-engine checkpoint: its XP rate and a command with only
--env, --task, --half-life and the batch settings `starts` and `life` (public_launch); no run name, source commit,
seed, start weights, output folder, workspace, resources or costs (the demo and the watch read no seed: the demo takes
its own --key and --seed). STEM.sha256 holds the sha256 of the files in sha256sum's form. A loader of the copy fills its tables from the table in use when that
table's digest is the trained one, or when it holds the fingerprint (check_names); a training run refuses the copy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

PATH = re.compile(r"(^|[\s'\"(=:])/[A-Za-z0-9_.-]")  # an absolute path in a string


def blanked(value):
    """`value` with every string that holds an absolute path replaced by '' (dicts and lists walked); the paths found."""
    if isinstance(value, dict):
        out = {k: blanked(v) for k, v in value.items()}
        return {k: v[0] for k, v in out.items()}, [p for v in out.values() for p in v[1]]
    if isinstance(value, (list, tuple)):
        out = [blanked(v) for v in value]
        return type(value)(v[0] for v in out), [p for v in out for p in v[1]]
    if isinstance(value, str) and PATH.search(value):
        return '', [value]
    return value, []


PUBLIC_OPTIONS = ('--env', '--task', '--half-life')  # the training command's options a public launch keeps
PUBLIC_SETTINGS = ('starts', 'life')  # and its batch settings (sim.watch reads starts, sim.real demo life)


def public_launch(record, task):
    """The public form of a training run's launch.json `record` (module docstring), refused unless it is a real-engine
    run of free-play version `task`."""
    command = record.get('command') or []
    option = lambda name: command[command.index(name) + 1] if name in command[:-1] else None  # noqa: E731
    if option('--env') != 'real' or option('--task') != task:
        raise ValueError(f"a launch of an --env real run of {task}, not --env {option('--env')}, --task {option('--task')}")
    rest = command[command.index('--batch-setting') + 1:] if '--batch-setting' in command else []
    rest = rest[:next((i for i, c in enumerate(rest) if c.startswith('--')), len(rest))]
    settings = [c for c in rest if c.split('=', 1)[0] in PUBLIC_SETTINGS]
    if not any(c.startswith('starts=') for c in settings):
        raise ValueError('a launch whose batch settings name no starts (the world the watch and the demo check)')
    kept = [part for name in PUBLIC_OPTIONS if option(name) is not None for part in (name, option(name))]
    return dict(note='a public copy of the training run\'s launch.json: only what sim.watch and sim.real demo read '
                     '(scripts/export_policy.py)', name='', xp_rate=record.get('xp_rate'),
                command=['python', '-m', 'sim.train', 'train', *kept, '--batch-setting', *settings])


def export(checkpoint, stem, launch=None):
    """Write the public copy of `checkpoint` at `stem` (module docstring), with the public form of `launch` (by default
    the run's launch.json, when the checkpoint has one). Returns what it did."""
    import numpy as np
    import torch
    from sb3_contrib import RecurrentPPO
    from baseline.policy import check_names, load_name_vectors, name_fingerprint
    from names import name_table
    stem = Path(stem)
    files = [stem.with_name(stem.name + suffix) for suffix in ('.zip', '.names.json', '.launch.json', '.sha256')]
    found = Path(checkpoint).resolve().parents[1] / 'launch.json'
    launch = Path(launch) if launch is not None else found if found.exists() else None
    if any(f.exists() for f in files):
        raise ValueError(f'{stem}: a public copy is written fresh, and {[str(f) for f in files if f.exists()]} exist')
    model = RecurrentPPO.load(str(checkpoint), device='cpu')
    public = None if launch is None else public_launch(json.loads(launch.read_text()),
                                                       getattr(model, 'free_play_task', None) or 'free-play-v1')
    table = name_table()
    taken = check_names(model.policy, observation=None)
    if taken['path'] != 'digest' or getattr(model, 'name_vectors', None) is not None:
        raise ValueError(f'{checkpoint}: the table in use must be the one it trained with, by its digest, and the '
                         f'checkpoint must hold its own vectors (no public copy): {taken}')
    load_name_vectors(model.policy, np.zeros(table.vectors.shape, np.float32))
    model.name_vectors = 'removed'  # the mark its loader fills the tables by (sim/hour.py load_free_policy)
    paths = []
    for name in sorted(vars(model)):
        value = getattr(model, name)
        if isinstance(value, (dict, list, tuple, str)) and not name.startswith('_'):
            value, found = blanked(value)
            if found:
                setattr(model, name, value)
                paths += [(name, p) for p in found]
    for buffer in (model.ep_info_buffer, model.ep_success_buffer):  # the run's last episodes, where it kept them
        if buffer is not None:
            buffer.clear()
    stem.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(files[0]))
    files[1].write_text(json.dumps(name_fingerprint(table), indent=1) + '\n')
    if public is not None:
        files[2].write_text(json.dumps(public, indent=1) + '\n')
    written = [f for f in files[:3] if f.exists()]
    files[3].write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n' for f in written))
    copy = RecurrentPPO.load(str(files[0]), device='cpu')  # the copy as written: no vector, no path
    if any(bool(t.any()) for k, t in copy.policy.state_dict().items() if k.endswith('embedding.table')):
        raise RuntimeError(f'{files[0]} holds name vectors')
    with zipfile.ZipFile(files[0]) as archive:
        left = PATH.search(archive.read('data').decode())
    if left:
        raise RuntimeError(f'{files[0]} still holds a path in its metadata near {left.group(0)!r}')
    same = {k: torch.equal(v, model.policy.state_dict()[k]) for k, v in copy.policy.state_dict().items()}
    if not all(same.values()):
        raise RuntimeError(f'{files[0]} differs from the policy it copies in {[k for k, v in same.items() if not v]}')
    return dict(checkpoint=str(checkpoint), files=[str(f) for f in files if f.exists()], table=table.digest,
                launch=None if launch is None else str(launch),
                blanked=[dict(attribute=n, value=p) for n, p in paths], tables_zeroed=len(
                    [k for k in copy.policy.state_dict() if k.endswith('embedding.table')]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('checkpoint')
    parser.add_argument('stem', help='OUT/STEM: writes STEM.zip, STEM.names.json, STEM.launch.json and STEM.sha256 there')
    parser.add_argument('--launch', help="the training run's launch.json (default: the one in the run's folder)")
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.checkpoint, args.stem, args.launch), indent=1))
    except (ValueError, RuntimeError) as error:
        sys.exit(f'[export_policy] {error}')


if __name__ == '__main__':
    main()
