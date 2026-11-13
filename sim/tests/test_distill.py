"""sim/distill.py on the CPU: the memory of a teacher's own episodes, read back through the teacher, reproduces its stored
action probabilities (recurrent history rebuilt from each episode's start); arm A's drift probe changes nothing in its
training; the distillation phase keeps the actor nearer the teacher on held-out histories than plain PPO does and never
changes the critic; its state survives a recovery. The teacher is a policy saved on the tests' synthetic name table
(conftest.py), playing the simulator's real tasks."""
from pathlib import Path

import pytest
import torch

from sim.learn import DeviceEnv, attach, build
from sim.tests.test_learn import N_STEPS, SETTINGS, WORLDS, fake

ROOT = Path(__file__).resolve().parents[2]
STRATA = ('wood', 'danger')  # a short gathering stratum and the long dangerous fights


@pytest.fixture
def teacher(tmp_path):
    path = tmp_path / 'teacher.zip'
    build(DeviceEnv(fake(), 3), seed=3, device='cpu', **SETTINGS).save(path)
    return path


@pytest.fixture
def memory(teacher, tmp_path):
    from sim import distill
    from sim.build import DATA
    from sim.data import Tables
    for path in (DATA, ROOT / 'logs/sim/world/train-8-nc1-derived.npz', ROOT / 'logs/search/pools/train-128.npz'):
        if not path.exists():
            pytest.skip(f'{path} is absent (docs/RUNNING.md says how to build it)')
    saved = distill.DISTILL['strata']
    distill.DISTILL['strata'] = {k: (*v[:3], 8) for k, v in saved.items()}
    try:
        path = tmp_path / 'memory.npz'
        distill.collect(teacher, path, Tables.load(), ROOT / 'logs/sim/world/train-8-nc1-derived.npz',
                        ROOT / 'logs/search/pools/train-128.npz', strata=STRATA)
    finally:
        distill.DISTILL['strata'] = saved
    return path


def from_teacher(teacher, seed=7, updates=0, **treatments):
    """A model built on the teacher's weights, attached, then trained `updates` updates. Building seeds the global
    generators, so each model is built and trained before the next is built."""
    from sim.train import load_weights
    model = build(DeviceEnv(fake(), seed), seed=seed, device='cpu', **SETTINGS)
    load_weights(model, teacher, 'cpu')
    attach(model, seed=seed, **treatments)
    if updates:
        model.learn(updates * WORLDS * N_STEPS)
    return model


def same_policy(a, b):
    sa, sb = a.policy.state_dict(), b.policy.state_dict()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_the_memory_reads_back_through_its_teacher(memory, teacher):
    from sim.distill import check
    for name, row in check(memory, teacher).items():
        assert row['episodes'] == 8 and row['max_abs_log_prob_difference'] < 1e-4 and abs(row['mean_kl']) < 1e-6, (name, row)


def test_the_drift_probe_leaves_arm_a_unchanged(memory, teacher, monkeypatch):
    from sim import distill
    monkeypatch.setitem(distill.DISTILL, 'probe_every', 1)
    plain = from_teacher(teacher, updates=2, telemetry=True)
    probed = from_teacher(teacher, updates=2, telemetry=True, memory=memory)
    same_policy(plain, probed)
    assert len(probed.distill.records) == 2 and probed.distill.records[0]['steps'] == 0
    assert set(probed.distill.records[0]['drift']) == set(STRATA)


def test_distillation_keeps_the_actor_nearer_the_teacher_and_never_moves_the_critic(memory, teacher, monkeypatch):
    from sim import distill
    monkeypatch.setitem(distill.DISTILL, 'probe_every', 1)
    monkeypatch.setitem(distill.DISTILL, 'steps', 16)
    plain = from_teacher(teacher, updates=3, telemetry=True, memory=memory)
    distilled = from_teacher(teacher, updates=3, telemetry=True, memory=memory, distill=True)
    drift = lambda m: sum(m.distill.probe(m.policy).values())
    assert drift(distilled) < drift(plain)
    assert distilled.distill.records[-1]['steps'] == 16 and distilled.distill.records[-1]['loss_mean'] > 0
    critic = {k: v.clone() for k, v in distilled.policy.state_dict().items() if 'vf' in k or 'value' in k or 'critic' in k}
    actor = {k: v.clone() for k, v in distilled.policy.state_dict().items() if k.startswith(('features_extractor', 'lstm_actor'))}
    modes = []  # cuDNN's LSTM backward needs training mode, so the phase runs in it
    handle = distilled.policy.lstm_actor.register_forward_hook(lambda module, inputs, output: modes.append(module.training))
    distilled.distill.run(distilled)  # the phase alone
    handle.remove()
    assert len(modes) >= 16 and all(modes)
    after = distilled.policy.state_dict()
    assert critic and all(torch.equal(v, after[k]) for k, v in critic.items())
    assert any(not torch.equal(v, after[k]) for k, v in actor.items())


def test_the_distillation_state_survives_a_recovery(memory, teacher, tmp_path, monkeypatch):
    from sim import distill
    from sim.train import train
    monkeypatch.setitem(distill.DISTILL, 'steps', 4)
    rollout = WORLDS * N_STEPS
    common = dict(seed=1, worlds=WORLDS, n_steps=N_STEPS, batch_size=12, n_epochs=2, checkpoint_every=rollout, verbose=False,
                  weights=teacher, distill_memory=memory, distill=True)
    first = train(tmp_path / 'a', decisions=2 * rollout, recovery=tmp_path / 'rec', **common)
    assert first['treatments']['arm'] == 'S' and first['treatments']['memory']['sha256']
    assert all(u['treatments']['distill']['steps'] == 4 for u in first['updates'])
    # the step count --validate checks is PPO's alone (the rehearsal's failure); the distillation phase records its own
    assert first['gradient_steps'] == 2 * 2 * -(-rollout // 12)
    summary = train(tmp_path / 'b', decisions=3 * rollout, recovery=tmp_path / 'rec2', resume=tmp_path / 'rec', env_seed=7,
                    **{k: v for k, v in common.items() if k != 'weights'})
    assert summary['resume']['treatments_restored']['tensors'] > 0
    assert summary['status'] == 'completed' and summary['updates'][-1]['treatments']['distill']['steps'] == 4


def test_arm_t_trains_only_its_strata_each_at_its_arm_s_weight(memory, teacher, monkeypatch):
    from sim import distill
    monkeypatch.setitem(distill.DISTILL, 'steps', 3)
    model = from_teacher(teacher, memory=memory, distill=True, distill_strata=['danger'])
    real, seen, sums = distill.kl_by_stratum, [], []

    def spy(policy, mem, episodes):
        per = real(policy, mem, episodes)
        seen.append({mem.names[s] for s in mem.stratum[episodes]})
        sums.append(sum(float(v) for v in per.values()))
        return per
    monkeypatch.setattr(distill, 'kl_by_stratum', spy)
    record = model.distill.run(model)
    assert record['steps'] == 3 and seen[:3] == [{'danger'}] * 3  # then the probe reads every stratum
    # the danger stratum alone, at the weight it has among both strata in arm S: its KL over 2
    assert record['loss_mean'] == pytest.approx(sum(sums[:3]) / 3 / 2, rel=1e-5)
    with pytest.raises(ValueError, match='not in the memory'):
        from_teacher(teacher, memory=memory, distill=True, distill_strata=['fishing'])
