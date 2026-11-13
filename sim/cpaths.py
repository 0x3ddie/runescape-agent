"""Builds and loads sim/_paths.c, the C port of sim/paths.py's hot path.

`load()` returns the extension module, or None when SIM_PATHS=python asks for the Python reference or no C compiler
can build it (sim/paths.py then keeps the reference and says so once). The build uses this interpreter's own compiler
and link command (sysconfig) and goes to SIM_BUILD (default ~/.cache/rs-sim), one file per source and interpreter, so
spawned workers and later runs reuse it; concurrent builds each write their own file and rename it into place.
"""
import hashlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig

SOURCE = Path(__file__).with_name('_paths.c')


def build_dir():
    return Path(os.environ.get('SIM_BUILD') or Path.home() / '.cache' / 'rs-sim')


def library():
    """The built file's path for this source and interpreter."""
    tag = hashlib.sha256(SOURCE.read_bytes() + sys.version.encode() + sysconfig.get_platform().encode()).hexdigest()[:16]
    return build_dir() / f'_paths-{tag}{sysconfig.get_config_var("EXT_SUFFIX")}'


def build(out):
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(f'{out.name}.{os.getpid()}.tmp')
    link = shlex.split(sysconfig.get_config_var('LDSHARED') or 'cc -shared')
    command = [*link, '-O2', '-fPIC', '-I', sysconfig.get_paths()['include'], str(SOURCE), '-o', str(temp)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        temp.unlink(missing_ok=True)
        raise OSError(f'{" ".join(command)} failed: {done.stderr.strip()[-2000:]}')
    os.replace(temp, out)


def load():
    if os.environ.get('SIM_PATHS') == 'python':
        return None
    out = library()
    try:
        if not out.exists():
            build(out)
    except OSError as error:
        print(f'[sim] sim/_paths.c not built, the Python path finder is used: {error}', file=sys.stderr)
        return None
    loader = importlib.machinery.ExtensionFileLoader('sim._paths', str(out))
    spec = importlib.util.spec_from_file_location('sim._paths', out, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module
