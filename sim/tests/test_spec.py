"""sim/spec.py is obs-v5: the shapes rl/obs.py projects from a native observation, plus the task inputs."""
import json
from pathlib import Path

import numpy as np
import pytest

from baseline.resource_task import GOAL_VEC
from obs import project
from sim.spec import OBSERVATION_SHAPES, check_batch, dtype

CAPTURE = Path(__file__).resolve().parents[2] / 'rl' / 'tests' / 'fixtures' / 'reset.json'
if not CAPTURE.exists():
    pytest.skip('needs rl/tests/fixtures/reset.json, a native reset capture that is not in this repository', allow_module_level=True)
RESET = json.load(open(CAPTURE))


def test_the_batched_spec_is_the_native_projection_with_the_task_inputs():
    one = project(RESET)
    one.update(budget_vec=np.zeros(2, np.float32), critic_vec=np.zeros(1, np.float32),
               goal_vec=np.eye(len(GOAL_VEC), dtype=np.float32)[0])
    assert {k: v.shape for k, v in one.items()} == OBSERVATION_SHAPES
    assert all(v.dtype == dtype(k) for k, v in one.items())
    batch = {k: np.repeat(np.asarray(v, dtype(k))[None], 3, 0) for k, v in one.items()}
    check_batch(batch, 3)
    batch['mask'][1] = 0
    try:
        check_batch(batch, 3)
    except ValueError:
        return
    raise AssertionError('an all-masked world passed')
