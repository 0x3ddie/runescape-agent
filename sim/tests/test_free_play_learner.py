"""The goal-free study's learner: the per-tick discount in the
advantage, the bootstrap of a life cut by its drawn length from its last observation under that life's recurrent state, the
PopArt value target, the policy built from the batch's declared table, the shared buffer's last observation and the
training command without a goal, with the digests that tie each zip it saves to its run and the source it ran, on
sim/learn.py FakeEnv's free-play stand-in.

Declared before the comparisons ran: an advantage is compared with an independent float64 computation within float32
rounding (rtol 1e-5, atol 2e-5 at values of order one); a goal-free loop with neutral settings, and every path with the
settings off, is compared with SB3's bit for bit; SB3's own time-limit bootstrap is compared within 1e-5, since it values one
last observation a call where the device collector values all of a step's at once (other matrix shapes, other rounding).
"""
import ast
from functools import cache
import math
from pathlib import Path
import threading
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from gymnasium import spaces
from hypothesis import assume, given, settings, strategies as st
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from baseline import config
from baseline.policy import POPART, MaskedRecurrentPolicy, PopArt, check_names, load_name_vectors
from names import name_table
from sim.learn import (FAKE_FREE_PLAY, FAKE_FREE_PLAY_SHAPES, DeviceEnv, DeviceRolloutBuffer, FakeEnv, HostDeviceEnv,
                       HostVecEnv, build)
from sim.spec import OBSERVATION_SHAPES, TERMINAL_DEATH, TERMINAL_HORIZON, TERMINAL_RUNNING
from sim.task import FAMILIES, FREE_PLAY
from sim.vec import EPISODE_FACTS
from sim.workers import FREE_PLAY_TASK

ROOT = Path(__file__).resolve().parents[2]
WORLDS, N_STEPS, BATCH = 4, 16, 16


def free(**kwargs):
    """The free-play stand-in with short lives and frequent deaths, so a rollout holds cuts and deaths."""
    return FakeEnv(WORLDS, id_rows=name_table().spec[0], free_play=True, **{'death_rate': .05, 'life_ticks': (4, 12), **kwargs})


def free_model(env=None, seed=3, **kwargs):
    return build(DeviceEnv(env or free(), seed), seed=seed, device='cpu', n_steps=N_STEPS, batch_size=BATCH, n_epochs=1,
                 free_play=True, **kwargs)


def collect(model):
    """One rollout, no update."""
    _, callback = model._setup_learn(WORLDS * N_STEPS, None)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, N_STEPS)
    return model.rollout_buffer


# ---- the advantage -------------------------------------------------------------------------------------------------
def filled(rewards, values, last, terminal, ticks, cut, first, *, gamma=1., gae_lambda=1., half_life=None, free_play=True):
    """A device buffer holding this rollout ([n_steps, n_envs] arrays; first: the episode starts of step 0), its returns
    and advantages computed; and SB3's buffer of the same rollout."""
    n_steps, n_envs = rewards.shape
    space = spaces.Dict(dict(mask=spaces.Box(0, 1, (5,), np.int8)))
    hidden = (n_steps, 1, n_envs, 4)
    mine = DeviceRolloutBuffer(n_steps, space, spaces.Discrete(96), hidden, 'cpu', gae_lambda, gamma, n_envs,
                               half_life=half_life, free_play=free_play)
    sb3 = RecurrentDictRolloutBuffer(n_steps, space, spaces.Discrete(96), hidden, 'cpu', gae_lambda, gamma, n_envs)
    ended = terminal != TERMINAL_RUNNING
    starts = np.vstack([first[None], ended[:-1]])
    states = RNNStates(*(tuple(torch.zeros(1, n_envs, 4) for _ in range(2)) for _ in range(2)))
    mask = np.ones((n_envs, 5), np.int8)
    for t in range(n_steps):
        value, log_prob = torch.as_tensor(values[t][:, None]), torch.zeros(n_envs)
        mine.add(dict(mask=torch.as_tensor(mask)), torch.zeros(n_envs, dtype=torch.long), torch.as_tensor(rewards[t]),
                 torch.as_tensor(starts[t], dtype=torch.float32), value, log_prob, lstm_states=states,
                 infos=dict(terminal=torch.as_tensor(terminal[t], dtype=torch.int32),
                            decision_ticks=torch.as_tensor(ticks[t], dtype=torch.int32)),
                 cut_values=torch.as_tensor(cut[t]) if free_play else None)
        sb3.add(dict(mask=mask), np.zeros((n_envs, 1)), rewards[t], starts[t], value, log_prob, lstm_states=states)
    mine.compute_returns_and_advantage(torch.as_tensor(last[:, None]), torch.as_tensor(ended[-1]))
    sb3.compute_returns_and_advantage(torch.as_tensor(last[:, None]), ended[-1])
    return mine, sb3


def direct_advantage(rewards, values, last, terminal, ticks, cut, gae_lambda, half_life):
    """The definition, as a sum and in float64: A_t = sum over k >= t of (prod over t <= j < k of g_j lambda c_j) delta_k,
    delta_k = r_k + g_k (c_k V_k+1 + B_k) - V_k, with g_k = gamma_tick ** tau_k, gamma_tick = 0.5 ** (1 / half_life),
    c_k = 0 where a life ended at step k (cut or death) and B_k the bootstrap value of a cut life."""
    n_steps, n_envs = rewards.shape
    g = (0.5 ** (1 / half_life)) ** ticks.astype(np.float64)
    goes_on = (terminal == TERMINAL_RUNNING).astype(np.float64)
    following = np.vstack([values[1:], last[None]]).astype(np.float64)
    delta = rewards + g * (goes_on * following + cut) - values
    advantage = np.zeros((n_steps, n_envs))
    for t in range(n_steps):
        weight = np.ones(n_envs)
        for k in range(t, n_steps):
            advantage[t] += weight * delta[k]
            weight = weight * g[k] * gae_lambda * goes_on[k]
    return advantage


def random_rollout(data, n_steps, n_envs):
    terminal = np.array(data.draw(st.lists(st.lists(st.sampled_from((TERMINAL_RUNNING,) * 3 + (TERMINAL_HORIZON, TERMINAL_DEATH)),
                                                    min_size=n_envs, max_size=n_envs), min_size=n_steps, max_size=n_steps)))
    ticks = np.array(data.draw(st.lists(st.lists(st.integers(0, 12), min_size=n_envs, max_size=n_envs),
                                        min_size=n_steps, max_size=n_steps)))
    rng = np.random.default_rng(data.draw(st.integers(0, 2 ** 32 - 1)))
    rewards, values = (rng.normal(size=(n_steps, n_envs)).astype(np.float32) for _ in range(2))
    cut = np.where(terminal == TERMINAL_HORIZON, rng.normal(size=(n_steps, n_envs)), 0).astype(np.float32)
    return rewards, values, rng.normal(size=n_envs).astype(np.float32), terminal, ticks, cut, rng.random(n_envs) < .5


@settings(max_examples=80, deadline=None)
@given(st.integers(1, 10), st.integers(1, 4), st.sampled_from((0., .5, .95, 1.)), st.sampled_from((1., 7., 1000., 3000.)),
       st.data())
def test_the_tick_discounted_advantage_is_the_direct_sum_over_random_ticks_rewards_and_ends(n_steps, n_envs, gae_lambda,
                                                                                            half_life, data):
    rewards, values, last, terminal, ticks, cut, first = random_rollout(data, n_steps, n_envs)
    mine, _ = filled(rewards, values, last, terminal, ticks, cut, first, gae_lambda=gae_lambda, half_life=half_life)
    expected = direct_advantage(rewards, values, last, terminal, ticks, cut, gae_lambda, half_life)
    np.testing.assert_allclose(mine.advantages.numpy(), expected, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(mine.returns, expected + values, rtol=1e-5, atol=2e-5)


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 9), st.integers(1, 5), st.sampled_from(((1., 1.), (.99, .95), (1., .95))), st.booleans(), st.data())
def test_the_goal_free_loop_with_neutral_settings_is_sb3s_bit_for_bit(n_steps, n_envs, discount, ticked, data):
    """No cut value and either gamma, or a half-life over steps of no game time (a discount of exactly 1): the goal-free
    loop gives SB3's advantages and returns bit for bit, so it changes nothing that its settings do not."""
    gamma, gae_lambda = discount
    assume(not (ticked and gamma != 1))
    rewards, values, last, terminal, _, _, first = random_rollout(data, n_steps, n_envs)
    zero = np.zeros_like(rewards)
    mine, sb3 = filled(rewards, values, last, terminal, zero.astype(int), zero, first, gamma=gamma, gae_lambda=gae_lambda,
                       half_life=1000. if ticked else None)
    assert torch.equal(mine.advantages, torch.as_tensor(sb3.advantages))
    assert np.array_equal(mine.returns, sb3.returns)


def test_a_cut_life_bootstraps_and_a_death_does_not_on_a_case_with_a_known_answer():
    """Half-life 2 ticks and 2 ticks a step: each step's discount is exactly 1/2; lambda 1/2. World 0's life is cut at step
    1 with bootstrap value 4, world 1 dies there; both start a new life at step 2, whose last value is 8. By hand:
    world 0 A = (1/4, 1, 3) (delta_1 = 1 + 4/2 - 2), world 1 A = (-1/4, -1, 3) (delta_1 = 1 + 0 - 2). The goal tasks' loop
    (no free play, gamma = lambda = 1) ends both lives at step 1 alike: a done stays terminal."""
    rewards, values = np.ones((3, 2), np.float32), np.full((3, 2), 2, np.float32)
    terminal = np.array([[0, 0], [TERMINAL_HORIZON, TERMINAL_DEATH], [0, 0]])
    cut = np.array([[0, 0], [4, 0], [0, 0]], np.float32)
    last, first, ticks = np.full(2, 8, np.float32), np.ones(2, bool), np.full((3, 2), 2)
    mine, _ = filled(rewards, values, last, terminal, ticks, cut, first, gae_lambda=.5, half_life=2.)
    assert mine.advantages.T.tolist() == [[.25, 1., 3.], [-.25, -1., 3.]]
    assert mine.returns.T.tolist() == [[2.25, 3., 5.], [1.75, 1., 5.]]
    goal, _ = filled(rewards, values, last, terminal, ticks, cut, first, free_play=False)
    assert goal.cut_values is None and goal.advantages[1].tolist() == [-1., -1.]
    assert goal.advantages[0].tolist() == [0., 0.]   # delta_0 = 1 + 2 - 2, plus the done step's -1 at lambda 1


# ---- the bootstrap in the collector ---------------------------------------------------------------------------------
def test_the_bootstrap_values_the_last_observation_under_the_lifes_recurrent_state_not_the_reset_state():
    """Each cut life's stored value is the critic's value of its last observation from the state its last step produced
    (the next step's stored state, or the rollout's last), with no episode start; from the reset (zero) state it differs.
    Deaths and running worlds store zero."""
    model = free_model(half_life_ticks=10.)
    steps, step_device = [], model.env.step_device

    def recorded(actions):
        out = step_device(actions)
        steps.append(out[3]['last_obs'])
        return out
    model.env.step_device = recorded
    b = collect(model)
    terminal = b.infos['terminal']
    assert bool((terminal == TERMINAL_HORIZON).any()) and bool((terminal == TERMINAL_DEATH).any())
    assert not bool(b.cut_values[terminal != TERMINAL_HORIZON].any())
    checked = 0
    for t, ended in enumerate(steps):
        assert ended.rows.tolist() == (terminal[t] != TERMINAL_RUNNING).nonzero().squeeze(1).tolist()
        for row, world in enumerate(ended.rows.tolist()):
            if terminal[t, world] != TERMINAL_HORIZON:
                continue
            if t + 1 < N_STEPS:
                h, c = b.hidden_states_vf[t + 1, world], b.cell_states_vf[t + 1, world]
            else:
                h, c = (s[:, world] for s in model._last_lstm_states.vf)
            obs = {k: v[row:row + 1] for k, v in ended.obs.items()}
            with torch.no_grad():
                life = model.policy.predict_values(obs, (h[:, None], c[:, None]), torch.zeros(1)).item()
                reset = model.policy.predict_values(obs, (torch.zeros_like(h)[:, None],) * 2, torch.zeros(1)).item()
            assert b.cut_values[t, world].item() == pytest.approx(life, abs=1e-6) and abs(life - reset) > 1e-4
            checked += 1
    assert checked == int((terminal == TERMINAL_HORIZON).sum())


def test_the_free_play_bootstrap_is_sb3s_own_time_limit_bootstrap():
    """SB3-Contrib's RecurrentPPO on HostVecEnv, where a cut life is a time-limit truncation with its last observation
    (constant gamma 0.99, lambda 0.95), against the device collector on the same play: the same actions and values, SB3's
    rewards are the paid ones plus gamma times the device's cut values, at the cuts only, and the advantages and returns
    agree."""
    seed, learner = 3, dict(n_steps=N_STEPS, batch_size=BATCH, n_epochs=1, gamma=.99, gae_lambda=.95)
    kwargs = {**config.LEARNER, **learner, 'policy_kwargs': {**config.LEARNER['policy_kwargs'], 'names': name_table().spec,
                                                             'observation': FAKE_FREE_PLAY}}
    reference = RecurrentPPO(MaskedRecurrentPolicy, HostVecEnv(free(), seed), seed=seed, device='cpu', **kwargs)
    load_name_vectors(reference.policy, name_table().vectors)
    device = build(DeviceEnv(free(), seed), seed=seed, device='cpu', free_play=True, **learner)
    for model in (reference, device):
        torch.manual_seed(7)
        np.random.seed(7)
        _, callback = model._setup_learn(WORLDS * N_STEPS, None)
        model.collect_rollouts(model.env, callback, model.rollout_buffer, N_STEPS)
    sb3, mine = reference.rollout_buffer, device.rollout_buffer
    terminal = mine.infos['terminal'].numpy()
    assert (terminal == TERMINAL_HORIZON).any() and (terminal == TERMINAL_DEATH).any()
    assert np.array_equal(sb3.actions[..., 0], mine.actions[..., 0].numpy()) and np.array_equal(sb3.values, mine.values)
    bootstrap = sb3.rewards - mine.rewards.numpy()
    assert ((bootstrap != 0) == (terminal == TERMINAL_HORIZON)).all()
    np.testing.assert_allclose(bootstrap, .99 * mine.cut_values.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sb3.advantages, mine.advantages.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sb3.returns, mine.returns, rtol=1e-5, atol=1e-5)


# ---- the life's pay against its episode_return -----------------------------------------------------------------------
def batch_lives(pays, ends):
    """What the free-play batch gives for these pays ([steps, worlds] float64, never negative) and life ends: each reward
    the float32 of its pay, and at a life's end episode_return the float32 of the life's float64 sum of its pays."""
    rewards = pays.astype(np.float32)
    returns = np.zeros(pays.shape, np.float32)
    total = np.zeros(pays.shape[1])
    for t in range(len(pays)):
        total += pays[t]
        returns[t] = np.where(ends[t], total, 0).astype(np.float32)
        total = np.where(ends[t], 0, total)
    return rewards, returns


def life_pay_by_rollouts(rewards, ends, returns, cuts):
    """sim/train.py life_pay over rollouts split at `cuts`, the pay carried across them as validate_free_play carries it."""
    from sim.train import life_pay
    paid, worst, excess = torch.zeros(rewards.shape[1], dtype=torch.float64), 0., -math.inf
    for a, b in zip((0, *cuts), (*cuts, len(rewards))):
        paid, w, e = life_pay(paid, *(torch.as_tensor(x[a:b]) for x in (rewards, ends, returns)))
        worst, excess = max(worst, w), max(excess, e)
    return worst, excess


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 300), st.integers(1, 3), st.floats(1e-4, .5), st.integers(0, 2 ** 32 - 1), st.data())
def test_a_lifes_pay_matches_the_batchs_float64_return_over_any_rollout_split(steps, worlds, scale, seed, data):
    """Any pays, ends and rollout splits: the batch's own arithmetic passes life_pay, and a return moved by twice its
    tolerance fails it."""
    rng = np.random.default_rng(seed)
    pays = np.where(rng.random((steps, worlds)) < .3, rng.exponential(scale, (steps, worlds)), 0.)
    ends = rng.random((steps, worlds)) < data.draw(st.sampled_from((0., .02, .2)))
    cuts = sorted(set(data.draw(st.lists(st.integers(1, steps), max_size=4))) - {steps})
    rewards, returns = batch_lives(pays, ends)
    worst, excess = life_pay_by_rollouts(rewards, ends, returns, cuts)
    assert excess <= 0
    if ends.any():
        t, b = map(int, np.argwhere(ends)[0])
        moved = returns.copy()
        moved[t, b] += np.float32(2 * (1e-5 + 2 ** -23 * abs(float(returns[t, b]))))
        assert life_pay_by_rollouts(rewards, ends, moved, cuts)[1] > 0


def test_a_long_earning_life_passes_where_a_float32_sum_of_its_rewards_misses():
    """One 6,000-decision life with 1,500 small unequal pays of about 90 levels in all, as a control earns in a 36,000-tick
    life: the float32 running sum the check once made misses the batch's return by more than 1e-5; the float64 sum is
    within its tolerance, over rollouts of 128 decisions."""
    rng = np.random.default_rng(7)
    pays = np.zeros((6000, 2))
    pays[rng.choice(6000, 1500, replace=False), 0] = rng.exponential(.06, 1500)
    pays[:, 1] = np.where(rng.random(6000) < .1, rng.exponential(.01, 6000), 0)
    ends = np.zeros((6000, 2), bool)
    ends[-1] = True
    rewards, returns = batch_lives(pays, ends)
    float32_sum = np.float32(0)
    for r in rewards[:, 0]:
        float32_sum = np.float32(float32_sum + r)
    assert 80 < returns[-1, 0] < 100 and abs(float32_sum - returns[-1, 0]) > 1e-5
    worst, excess = life_pay_by_rollouts(rewards, ends, returns, range(128, 6000, 128))
    assert excess <= 0 and worst < 1e-5 + 2 ** -23 * 100


def test_a_life_of_a_few_hundred_levels_needs_the_tolerances_term_relative_to_its_return():
    """A life of about 294 levels in 1,280 pays, each a power of two just above itself, so that every float32 reward
    rounds down by almost half its last place and the float32 return rounds up a whole last place of 2**-15: the float64
    sum of the rewards is 2**-15 from the return. That is within 1e-5 + 2**-23 * |return|, and beyond 1e-5 alone and
    beyond 1e-5 + 2**-24 * |return|; real lives at the XP rate of 25 come within 1.35e-5."""
    rng = np.random.default_rng(11)
    pays = np.zeros((5120, 1))
    pays[rng.choice(5120, 1280, replace=False), 0] = 2. ** -rng.integers(1, 5, 1280) * (1 + .49 * 2 ** -23)
    ends = np.zeros((5120, 1), bool)
    ends[-1] = True
    rewards, returns = batch_lives(pays, ends)
    ret = float(returns[-1, 0])
    worst, excess = life_pay_by_rollouts(rewards, ends, returns, range(128, 5120, 128))
    assert 262 < ret < 344 and worst == 2 ** -15 and excess <= 0
    assert worst > 1e-5 + 2 ** -24 * ret


# ---- the normalized value target ------------------------------------------------------------------------------------
@settings(max_examples=60, deadline=None)
@given(st.floats(-100, 100), st.floats(1e-2, 1e2), st.sampled_from((.1, .5, 1.)), st.integers(0, 2 ** 32 - 1))
def test_popart_keeps_every_value_output_through_updates_of_its_statistics(shift, scale, beta, seed):
    """Two updates towards targets of any location and scale: the head's outputs stay within float32 rounding, and its
    statistics are the debiased exponential averages of the targets' moments (an independent float64 computation).
    The rounding grows with the statistics (the output is std * n + mean): the bound is a few float32 steps of
    1 + |output| + |mean| + std."""
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed % 2 ** 31)
    head = PopArt(torch.nn.Linear(8, 1), beta=beta, min_std=POPART['min_std'])
    latent = torch.randn(64, 8, generator=generator)
    weight, first, second = 0., 0., 0.
    for _ in range(2):
        targets = torch.as_tensor(shift + scale * rng.normal(size=(5, 7)), dtype=torch.float32)
        with torch.no_grad():
            before = head(latent)
        head.update(targets)
        with torch.no_grad():
            after = head(latent)
        bound = 2e-6 * (1 + float(before.abs().max()) + abs(head.mean.item()) + head.std.item())
        np.testing.assert_allclose(after.numpy(), before.numpy(), rtol=0, atol=bound)
        x = targets.double().numpy()
        weight, first, second = (1 - beta) * weight + beta, (1 - beta) * first + beta * x.mean(), \
            (1 - beta) * second + beta * (x * x).mean()
        mean = first / weight
        std = math.sqrt(max(second / weight - mean * mean, POPART['min_std'] ** 2))
        assert head.mean.item() == pytest.approx(mean, rel=1e-6, abs=1e-6)
        assert head.std.item() == pytest.approx(std, rel=1e-5)


def test_popart_floors_the_scale_of_constant_targets():
    head = PopArt(torch.nn.Linear(3, 1), **{k: v for k, v in POPART.items() if k != 'method'})
    head.update(torch.full((10,), 2.5))
    assert head.mean.item() == 2.5 and head.std.item() == pytest.approx(POPART['min_std'])


def test_the_value_loss_fits_the_normalized_target():
    """After an update has moved the statistics, a minibatch's value loss is mean(((G - mean) / std - n(h))^2) with
    n(h) = W h + b the head's normalized output, not the unnormalized squared error."""
    model = free_model(value_norm=POPART)
    model.learn(WORLDS * N_STEPS)
    head = model.policy.value_net
    assert isinstance(head, PopArt) and abs(head.std.item() - 1) > .1
    model.train = lambda: None  # collect the next rollout, no update
    model.learn(WORLDS * N_STEPS)
    b = model.rollout_buffer
    data = b.compact(*next(b.layout(BATCH)))
    latents = []
    hook = head.register_forward_hook(lambda module, inputs, output: latents.append(inputs[0].detach()))
    with torch.no_grad():
        values, _, _ = model.policy.evaluate_compact(data.observations, data.actions, data.lstm_states, data.keep, data.pos,
                                                     data.n_seq, data.max_length)
    hook.remove()
    normalized = torch.nn.functional.linear(latents[0], head.weight, head.bias).flatten()
    expected = torch.mean(((data.returns - head.mean) / head.std - normalized) ** 2)
    stats = model._minibatch_step(data, .2)  # its losses come before its own optimizer step
    assert stats[2].item() == pytest.approx(expected.item(), rel=1e-4)
    assert stats[2].item() != pytest.approx(torch.mean((data.returns - values) ** 2).item(), rel=.01)


def test_each_update_fits_popart_statistics_of_its_own_returns_set_before_its_first_step():
    """Two updates: at each update's first optimizer step the head's mean and scale are the debiased moments of the
    returns of every rollout so far, this one's included (an independent float64 computation), not of the stored values
    and not the statistics before this rollout."""
    model = free_model(value_norm=POPART)
    head, beta, floor = model.policy.value_net, POPART['beta'], POPART['min_std']
    fresh, seen = [False], []
    update = model.train

    def train():
        fresh[0] = True
        return update()

    def first_step(optimizer, args, kwargs):
        if fresh[0]:
            fresh[0] = False
            seen.append((model.rollout_buffer._returns.double().clone(), head.mean.item(), head.std.item()))
    model.train = train
    handle = model.policy.optimizer.register_step_pre_hook(first_step)
    model.learn(2 * WORLDS * N_STEPS)
    handle.remove()
    assert len(seen) == 2
    weight, first, second = 0., 0., 0.
    for returns, mean, std in seen:
        weight, first, second = (1 - beta) * weight + beta, (1 - beta) * first + beta * returns.mean().item(), \
            (1 - beta) * second + beta * (returns ** 2).mean().item()
        expected = first / weight
        assert mean == pytest.approx(expected, rel=1e-6, abs=1e-6)
        assert std == pytest.approx(math.sqrt(max(second / weight - expected ** 2, floor ** 2)), rel=1e-5)
    assert seen[0][1] != 0 and not torch.equal(seen[0][0], seen[1][0])


def test_a_popart_checkpoint_keeps_its_statistics_and_one_of_another_setting_or_version_is_refused(tmp_path):
    from sim.train import CheckFailed, train
    common = dict(seed=1, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1, verbose=False)
    free_play = dict(common, decisions=2 * WORLDS * N_STEPS, task=FREE_PLAY_TASK)
    run = train(tmp_path / 'a', validate=True, recovery=tmp_path / 'rec', **free_play)
    assert run['checks']['checkpoint']['values_equal'] and run['checks']['checkpoint']['probs_equal']
    assert run['free_play']['batch_in_effect'] == dict(observation=FAKE_FREE_PLAY, life=['uniform', 6, 30], gain=.25,
                                                       death_rate=.002)
    loaded = RecurrentPPO.load(tmp_path / 'a' / 'latest.zip', device='cpu').policy
    head = loaded.value_net
    assert isinstance(head, PopArt) and loaded.value_norm == POPART and head.moments[0].item() == pytest.approx(.19)
    saved = torch.load(tmp_path / 'rec' / 'state.pt', weights_only=False)['policy']
    assert all(torch.equal(saved[f'value_net.{k}'], getattr(head, k)) for k in ('mean', 'std', 'moments', 'weight', 'bias'))
    for start in ('weights', 'init'):  # refused before any tensor is loaded
        with pytest.raises(ValueError, match="value normalization .*'popart'.*, the run none"):
            train(tmp_path / f'{start}-none', value_norm='none', **{start: tmp_path / 'a' / 'latest.zip'}, **free_play)
    again = train(tmp_path / 'b', weights=tmp_path / 'a' / 'latest.zip', half_life_ticks=3000, **free_play)
    assert again['weights']['policy_equal'] and again['source_check']['max_difference'] == 0.0
    assert again['weights']['saved_settings_not_imported']['half_life_ticks'] == 1000. and again['free_play']['half_life_ticks'] == 3000.
    started = train(tmp_path / 'g', init=tmp_path / 'a' / 'latest.zip', half_life_ticks=3000, **free_play)  # an ablation's start
    assert started['init']['policy_equal'] and started['init']['saved_settings_not_imported'] == dict(free_play=True,
                                                                                                      half_life_ticks=1000.)
    goal = dict(common, decisions=WORLDS * N_STEPS)
    train(tmp_path / 'goal', **goal)
    with pytest.raises(ValueError, match='reads obs-v5 inputs, not fake-obs-v6'):
        train(tmp_path / 'c', weights=tmp_path / 'goal' / 'latest.zip', **free_play)
    with pytest.raises(ValueError, match='reads fake-obs-v6 inputs, not obs-v5'):
        train(tmp_path / 'd', weights=tmp_path / 'a' / 'latest.zip', **goal)
    with pytest.raises(CheckFailed, match='goal-free settings'):  # a resume under another half-life
        train(tmp_path / 'e', resume=tmp_path / 'rec', half_life_ticks=3000, **dict(free_play, decisions=3 * WORLDS * N_STEPS))
    other = DeviceEnv(FakeEnv(WORLDS, id_rows=name_table().spec[0], free_play=True, life_ticks=(6, 31)), 1)
    with pytest.raises(CheckFailed, match='goal-free settings'):  # a resume on a batch under other settings
        train(tmp_path / 'h', resume=tmp_path / 'rec', environment=other, **dict(free_play, decisions=3 * WORLDS * N_STEPS))
    resumed = train(tmp_path / 'f', resume=tmp_path / 'rec', **dict(free_play, decisions=3 * WORLDS * N_STEPS))
    assert resumed['status'] == 'completed'
    assert resumed['resume']['learner'] == {k: v for k, v in run['free_play'].items() if k != 'observation'}


# ---- the policy and the command -------------------------------------------------------------------------------------
def test_an_obs_v6_policy_is_built_from_its_declared_table_and_records_its_version():
    """The stand-in's table has no goal_vec (6) and no budget_vec (2) and a prog_vec 7 wider: neither extractor reads the
    two, and each first layer is one input narrower than the goal policy's. The version recorded is the table's, so a
    loader that expects obs-v5 refuses it."""
    model = free_model()
    goal = build(DeviceEnv(FakeEnv(WORLDS, id_rows=name_table().spec[0]), 3), seed=3, device='cpu', n_steps=N_STEPS,
                 batch_size=BATCH, n_epochs=1)
    for side in ('features_extractor', 'vf_features_extractor'):
        mine, theirs = getattr(model.policy, side), getattr(goal.policy, side)
        assert not {'goal_vec', 'budget_vec'} & set(mine.keys) and {'goal_vec', 'budget_vec'} <= set(theirs.keys)
        assert theirs.encoder[0].in_features - mine.encoder[0].in_features == 6 + 2 - 7
    assert model.observation_space['prog_vec'].shape == FAKE_FREE_PLAY_SHAPES['prog_vec'] == (OBSERVATION_SHAPES['prog_vec'][0] + 7,)
    assert model.policy.observation == FAKE_FREE_PLAY and model.policy.value_norm is None
    check_names(model.policy, observation=FAKE_FREE_PLAY)
    with pytest.raises(ValueError, match='reads fake-obs-v6 inputs'):
        check_names(model.policy)


def test_the_goal_path_saves_and_builds_what_it_did_before(tmp_path):
    """With the goal-free settings off, a model has SB3's Linear value head and no cut-value storage, and its checkpoint
    records none of them: no free_play, half_life_ticks or value_norm."""
    import json
    import zipfile
    model = build(DeviceEnv(FakeEnv(WORLDS, id_rows=name_table().spec[0]), 3), seed=3, device='cpu', n_steps=N_STEPS,
                  batch_size=BATCH, n_epochs=1)
    assert type(model.policy.value_net) is torch.nn.Linear and model.rollout_buffer.cut_values is None
    model.save(tmp_path / 'goal.zip')
    with zipfile.ZipFile(tmp_path / 'goal.zip') as saved:
        data = json.loads(saved.read('data').decode())
    assert not {'free_play', 'half_life_ticks'} & set(data) and 'value_norm' not in data['policy_kwargs']


def test_the_free_play_command_trains_end_to_end_on_the_fake_env_without_reading_a_goal(tmp_path, monkeypatch, capsys):
    """sim.train train --task free-play-v1 --env fake: FREE_PLAY_LEARNER's settings read back from the model, every rollout
    validated (each life's pay equals its episode_return; the stand-in's info goal is -1, which a per-goal count would
    refuse), lives counted with their cut values, the segments replay, and the checkpoint loads as the stand-in's version.
    The command refuses a goal set, the goal tasks' telemetry, a missing rollout length, the settings on a goal task, a run
    on the CPU workers that leaves muggers or the life length unnamed, a search pool, and under PopArt the replay probe and
    the stock update, which skip its statistics update."""
    import json
    import sys
    from sim.train import FREE_PLAY_LEARNER, main, train
    monkeypatch.setattr(sys, 'argv', ['sim.train', 'train', '--task', FREE_PLAY_TASK, '--env', 'fake', '--worlds', str(WORLDS),
                                      '--n-steps', str(N_STEPS), '--batch-size', str(BATCH), '--epochs', '1',
                                      '--decisions', str(2 * WORLDS * N_STEPS), '--validate', '--segments',
                                      '--segment-worlds', '2', '--segment-decisions', '4', '--half-life', '3000',
                                      '--output', str(tmp_path / 'run')])
    main()
    summary = json.loads((tmp_path / 'run' / 'summary.json').read_text())
    assert summary['status'] == 'completed' and summary['goals'] == [] and summary['per_goal'] is None
    assert summary['tables'] is None  # the stand-in reads no archive
    learner = summary['learner']
    assert (learner['gamma'], learner['gae_lambda'], learner['half_life_ticks']) == (1., FREE_PLAY_LEARNER['gae_lambda'], 3000.)
    assert learner['free_play'] and learner['value_norm'] == POPART and learner['observation'] == FAKE_FREE_PLAY
    assert summary['free_play']['gamma_tick'] == 0.5 ** (1 / 3000) and summary['changes_from_learner']['gae_lambda'] == \
        dict(learner=1., here=FREE_PLAY_LEARNER['gae_lambda'])
    checks = summary['checks']['rollouts']
    assert all(c['problems'] == [] and c['life_pay_max_error'] == 0 for c in checks)
    lives = summary['lives']
    assert lives['lives'] == sum(c['lives_ended'] for c in checks) and lives['cut'] == sum(c['cut_lives'] for c in checks) > 0
    assert lives['transitions'] == 2 * WORLDS * N_STEPS and summary['segments']['updates'] == 2
    assert all(r['evidence']['transitions'] == {FREE_PLAY_TASK: WORLDS * N_STEPS} for r in summary['updates'])
    check_names(RecurrentPPO.load(tmp_path / 'run' / 'latest.zip', device='cpu').policy, observation=FAKE_FREE_PLAY)
    capsys.readouterr()
    common = dict(decisions=WORLDS * N_STEPS, seed=1, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1,
                  verbose=False)
    for refused, match in ((dict(task=FREE_PLAY_TASK, goals=('wood',)), 'no goal'), (dict(task=FREE_PLAY_TASK, telemetry=True), 'per goal'),
                           (dict(half_life_ticks=1000), 'free-play-v1'), (dict(task=FREE_PLAY_TASK, gae_lambda=2.), 'lambda'),
                           (dict(task=FREE_PLAY_TASK, env='cpu', pool=tmp_path / 'pool.npz', batch=dict(muggers=False)),
                            r"names muggers, life, world .*\['life', 'world'\] missing"),
                           *((dict(task=FREE_PLAY_TASK, env='cpu', pool=tmp_path / 'pool.npz', **engine,
                                   batch=dict(muggers=False, life=['fixed', 600], world='village-world-v3')),
                              'village-world-v3 is a world of the real engine only')  # the Python reference, the C engine
                             for engine in (dict(), dict(engine='native', pack=tmp_path / 'pack.bin'))),
                           (dict(task=FREE_PLAY_TASK, search_pool=tmp_path / 'search.npz'), 'no search pool'),
                           (dict(task=FREE_PLAY_TASK, replay=(0, ('stock', 'fast'))), 'PopArt')):
        with pytest.raises(ValueError, match=match):
            train(tmp_path / 'refused', **common, **refused)
    from sim.learn import DeviceRecurrentPPO
    with monkeypatch.context() as patch:
        patch.setattr(DeviceRecurrentPPO, 'FAST_UPDATE', False)  # --stock-update
        with pytest.raises(ValueError, match='PopArt'):
            train(tmp_path / 'refused', task=FREE_PLAY_TASK, **common)
    monkeypatch.setattr(sys, 'argv', ['sim.train', 'train', '--task', FREE_PLAY_TASK, '--decisions', '64', '--output',
                                      str(tmp_path / 'no-steps')])
    with pytest.raises(SystemExit, match='--n-steps'):
        main()


def test_the_summary_ties_latest_and_each_periodic_checkpoint_to_its_run_by_the_policy_digest(tmp_path):
    """policy_sha256_at_end is the policy_digest of the reloaded latest.zip and checkpoint_digests that of each periodic
    checkpoint by its file name (sim/hour.py loads a zip on the CPU and digests its tensors with sim/train.py policy_digest),
    for a goal task (checkpoints from the callback, the first one before any update) and for free play with a recovery
    directory (checkpoints at complete updates); a run that fails after two checkpoints still names both, and no
    latest.zip."""
    import json
    from stable_baselines3.common.callbacks import BaseCallback
    from sim.train import policy_digest, train
    reloaded = lambda path: policy_digest(RecurrentPPO.load(path, device='cpu').policy)
    saved = lambda run: {p.name: reloaded(p) for p in (run / 'checkpoints').glob('*.zip')}
    common = dict(seed=1, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1, verbose=False,
                  checkpoint_every=WORLDS * N_STEPS, decisions=2 * WORLDS * N_STEPS)
    for name, kwargs in (('goal', {}), ('free', dict(task=FREE_PLAY_TASK, recovery=tmp_path / 'rec'))):
        run = train(tmp_path / name, **common, **kwargs)
        digests = saved(tmp_path / name)
        assert run['checkpoint_digests'] == digests and len(set(digests.values())) == 2  # two saves of other weights
        assert run['policy_sha256_at_end'] == reloaded(tmp_path / name / 'latest.zip') != run['policy_sha256_at_start']

    class Stop(BaseCallback):  # after the checkpoint callback: the run fails at the step of its second save
        def _on_step(self):
            if self.num_timesteps == 2 * WORLDS * N_STEPS:
                raise RuntimeError('stopped after a save')
            return True
    with pytest.raises(RuntimeError, match='stopped after a save'):
        train(tmp_path / 'failed', **dict(common, decisions=3 * WORLDS * N_STEPS), callbacks=[Stop()])
    failed = json.loads((tmp_path / 'failed' / 'summary.json').read_text())
    assert failed['status'] == 'failed' and failed['checkpoint_digests'] == saved(tmp_path / 'failed')
    assert len(failed['checkpoint_digests']) == 2 and 'policy_sha256_at_end' not in failed


def test_the_summary_records_the_source_of_the_run_and_whether_it_made_the_checks(tmp_path, monkeypatch):
    """summary 'source' is baseline/telemetry.py source() with the run folder: in a checkout that differs from its commit,
    the commit, the changed and untracked files and the patch, written beside the summary; outside a git checkout (a git
    archive), no commit and the error, and no patch. 'validate' says whether the --validate checks ran."""
    import functools
    import hashlib
    import json
    import subprocess
    import baseline.telemetry
    from sim.train import train
    repo, bundle = tmp_path / 'repo', tmp_path / 'bundle'
    git = lambda *args: subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True).stdout
    for path in (repo, bundle):
        path.mkdir()
    git('init', '-q')
    (repo / 'kept.py').write_text('a = 1\n')
    git('add', '.')
    git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'base')
    (repo / 'kept.py').write_text('a = 2\n')
    (repo / 'new.py').write_text('b = 3\n')
    common = dict(decisions=WORLDS * N_STEPS, seed=1, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1,
                  verbose=False, task=FREE_PLAY_TASK)
    original = baseline.telemetry.source
    monkeypatch.setattr(baseline.telemetry, 'source', functools.partial(original, root=repo))
    run = train(tmp_path / 'changed', validate=True, **common)
    patch = (tmp_path / 'changed' / 'source.patch').read_bytes()
    assert run['source'] == dict(commit=git('rev-parse', 'HEAD').strip(), changed=[' M kept.py'], untracked=['new.py'],
                                 patch=dict(file='source.patch', sha256=hashlib.sha256(patch).hexdigest(), bytes=len(patch)))
    assert json.loads((tmp_path / 'changed' / 'summary.json').read_text())['source'] == run['source'] and run['validate'] is True
    monkeypatch.setattr(baseline.telemetry, 'source', functools.partial(original, root=bundle))
    run = train(tmp_path / 'unpacked', **common)
    assert run['source'] == dict(commit=None, error='not a git checkout') and run['validate'] is False
    assert not (tmp_path / 'unpacked' / 'source.patch').exists()


def test_validate_refuses_a_life_far_from_its_integer_gain_and_a_step_without_a_game_tick():
    """validate_free_play passes a collected rollout, and refuses it with one ended life's integer gain over the paid
    skills |S| levels or more from its paid gain (|S|: track B's sim/task.py FREE_PLAY skills), for a life whose return
    is a whole number (exactly that far) and for one whose return has a fraction (the stand-in's integer gain is the
    whole part of its pay, so |S| below it is that far whatever the fraction), or with one supported step that took no
    game tick."""
    from sim.train import CheckFailed, validate_free_play
    skills = len(FREE_PLAY['skills'])
    model = free_model()
    b = collect(model)
    carried = lambda: dict(paid=torch.zeros(WORLDS, dtype=torch.float64), done=torch.ones(WORLDS))
    assert validate_free_play(model, carried())['lives_ended'] > 0
    ended = [tuple(row) for row in torch.nonzero(b.infos['terminal'] != TERMINAL_RUNNING).tolist()]
    whole = next(life for life in ended if b.infos['episode_return'][life] % 1 == 0)
    part = next(life for life in ended if b.infos['episode_return'][life] % 1 != 0)
    for (t, w), field, value, match in ((whole, 'outcomes', b.infos['outcomes'][whole] - skills, 'integer gain'),
                                        (part, 'outcomes', b.infos['outcomes'][part] - skills, 'integer gain'),
                                        (whole, 'decision_ticks', 0, 'no game tick')):
        kept = b.infos[field][t, w].clone()
        b.infos[field][t, w] = value
        with pytest.raises(CheckFailed, match=match):
            validate_free_play(model, carried())
        b.infos[field][t, w] = kept


def test_the_learner_and_the_training_command_import_no_evaluation_module():
    """The learner stays apart from evaluation: sim/learn.py, sim/train.py and sim/workers.py import none of
    sim/free.py (the controls), sim/hour.py and sim/gate.py, at module level or inside a function; and in a fresh
    interpreter, sim.train and every module the three import (a worker's among them) leave all three unimported."""
    import subprocess
    import sys
    evaluation, modules = {'sim.free', 'sim.hour', 'sim.gate'}, set()
    for name in ('learn', 'train', 'workers'):
        tree = ast.parse((ROOT / 'sim' / f'{name}.py').read_text())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {f'{n.module}.{a.name}' if n.module == 'sim' else n.module for n in ast.walk(tree)
                     if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not evaluation & imported, name
        modules |= imported
    code = ('import importlib, sys; sys.path[:0] = [".", "rl"]; import sim.train\n'
            f'for name in {sorted(modules)!r}:\n    importlib.import_module(name)\n'
            f'print(sorted({sorted(evaluation)!r} & sys.modules.keys()))')
    done = subprocess.run([sys.executable, '-I', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.split() == ['[]'], done.stderr[-2000:]


# ---- the shared buffer ----------------------------------------------------------------------------------------------
class HostBatch:
    """sim/vec.py CpuBatch's interface over FakeEnv's free-play stand-in, through numpy, for the workers' plumbing; it
    records each reply, holds the free-play batch's settings (sim/workers.py BATCH_SETTINGS) and declares `observation`.
    drop: leave info['last_obs'] out; spoil: a function that changes each step's (reward, done, info); refuse: a reason
    to refuse being built; uneven: the worker of slot 0 runs another XP rate."""
    observation, drop, spoil, refuse, uneven = FAKE_FREE_PLAY, False, None, None, False

    def __init__(self, tables, layouts, count, *, derived=None, first=0, task=None, search=None, **settings):
        if self.refuse:
            raise ValueError(self.refuse)
        self.env, self.evidence, self.settings, self.first = free_env(count), [], settings, first
        self.xp_rate = settings.get('xp_rate', 1) + (self.uneven and first == 0)
        self.muggers, self.life, self.cfg = settings.get('muggers', False), ('uniform', 4, 12), dict(maxDecisions=9)
        self.world = settings.get('world', 'village-world-v1')  # CpuBatch's: the pool's world
        self.replies = []

    def reset(self, key):
        state, obs = self.env.reset(jax.random.key(key))
        obs = numpy(obs)
        self.replies.append(obs)
        return state, obs

    def step(self, state, action):
        state, obs, reward, done, info = self.env.step(state, jnp.asarray(action))
        info = dict(info)
        ended = numpy(info.pop('last_obs'))
        info = numpy(info)
        count = len(info['goal'])  # the CpuBatch fields sim/spec.py does not name
        info.update(made=np.zeros((count, len(FAMILIES)), np.int32), kills=np.zeros(count, np.int32),
                    facts=np.zeros((count, len(EPISODE_FACTS)), np.int32))
        if not self.drop:
            info['last_obs'] = ended
        reward, done = np.asarray(reward), np.asarray(done)
        if self.spoil is not None:  # a plain function, not a method
            reward, done, info = type(self).spoil(reward, done, info)
        out = numpy(obs), reward, done, info
        self.replies.append(out)
        return (state, *out)


@cache
def free_env(count):  # one compiled stand-in per size: it holds no state between calls
    return FakeEnv(count, id_rows=name_table().spec[0], free_play=True, death_rate=.05, life_ticks=(4, 12))


def numpy(arrays):
    return {k: np.asarray(v) for k, v in arrays.items()}


class ThreadProcess(threading.Thread):
    """multiprocessing's Process as a thread, so the workers' batch can be a test double. When the thread ends, as when
    a process exits, its end of the pipe closes: a worker that dies without a reply gives the parent's recv an EOFError,
    not a wait without end."""

    def run(self):
        ends = [arg for arg in self._args if isinstance(arg, ChildEnd)]  # Thread.run drops its arguments
        try:
            super().run()
        finally:
            for end in ends:
                end.conn.close()

    def kill(self):
        pass


class ChildEnd:
    """A pipe's child end shared with a thread: the parent's close() after start() (a process holds its own copy) leaves
    it open for the thread."""

    def __init__(self, conn):
        self.conn = conn
        self.send, self.recv = conn.send, conn.recv

    def close(self):
        pass


def thread_pipe():
    import multiprocessing
    parent, child = multiprocessing.Pipe()
    return parent, ChildEnd(child)


@pytest.fixture
def threaded(monkeypatch):
    """sim/workers.py CpuWorkers with each worker a thread and its batch HostBatch (the workers' own code otherwise)."""
    import sim.vec
    import sim.workers
    import sim.world
    batches = []

    class Recorded(HostBatch):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            batches.append(self)
    monkeypatch.setattr(sim.workers.mp, 'get_context', lambda name: SimpleNamespace(Pipe=thread_pipe, Process=ThreadProcess))
    monkeypatch.setattr(sim.vec, 'CpuBatch', Recorded)
    monkeypatch.setattr(sim.world, 'load_pool', lambda path: (None, None))
    monkeypatch.setattr(sim.workers, 'load_tables', lambda path: None)
    return batches


def test_the_workers_carry_each_ended_lifes_last_observation_through_the_shared_buffer(threaded):
    """Two workers over 8 free-play worlds: every step's observations, rewards, ends, info and last_obs, read from the
    shared buffer, are what each worker's batch returned for its slice; last_obs is the ended rows' and zero elsewhere;
    the batch gets its settings and no goals or horizon. HostDeviceEnv copies only the ended rows to the learner."""
    from sim.workers import CpuWorkers
    workers = CpuWorkers(8, 2, 'pool', task=FREE_PLAY_TASK, goals=None, observation=FAKE_FREE_PLAY, shapes=FAKE_FREE_PLAY_SHAPES,
                         batch=dict(xp_rate=1))
    try:
        batches = sorted(threaded, key=lambda b: b.first)  # the threads start in any order
        assert [(b.first, b.settings) for b in batches] == [(0, dict(xp_rate=1)), (4, dict(xp_rate=1))] and workers.goals == ()
        assert workers.settings_in_effect == dict(world='village-world-v1', observation=FAKE_FREE_PLAY, xp_rate=1, muggers=False,
                                                  life=['uniform', 4, 12],
                                                  cfg=dict(maxDecisions=9))
        state, obs = workers.reset(5)
        rng, ended = np.random.default_rng(0), 0
        for step in range(30):
            legal = [np.flatnonzero(row) for row in obs['mask']]
            action = np.asarray([rng.choice(a) for a in legal], np.int32)
            state, obs, reward, done, info = workers.step(state, action)
            for k, batch in enumerate(batches):
                mine = slice(4 * k, 4 * k + 4)
                b_obs, b_reward, b_done, b_info = batch.replies[-1]
                assert all(np.array_equal(obs[n][mine], b_obs[n]) for n in FAKE_FREE_PLAY_SHAPES)
                assert np.array_equal(reward[mine], b_reward) and np.array_equal(done[mine], b_done)
                assert all(np.array_equal(info[n][mine], b_info[n]) for n in b_info if n != 'last_obs')
                assert all(np.array_equal(info['last_obs'][n][mine], b_info['last_obs'][n]) for n in FAKE_FREE_PLAY_SHAPES)
            stopped = info['terminal'] != TERMINAL_RUNNING
            assert info['last_obs']['mask'][stopped].any(1).all() and not info['last_obs']['mask'][~stopped].any()
            ended += int(stopped.sum())
        assert ended > 0
    finally:
        workers.close()
    host = HostDeviceEnv(CpuWorkers(8, 2, 'pool', task=FREE_PLAY_TASK, goals=None, observation=FAKE_FREE_PLAY,
                                    shapes=FAKE_FREE_PLAY_SHAPES), seed=5)
    try:
        assert host.last_obs and host.observation == FAKE_FREE_PLAY
        obs, seen = host.reset(), 0
        for _ in range(30):
            obs, _, _, info = host.step_device(obs['mask'].float().argmax(1))
            terminal = host.host.v['info/terminal']
            rows = np.flatnonzero(terminal != TERMINAL_RUNNING)
            assert info['last_obs'].rows.tolist() == rows.tolist()
            assert all(np.array_equal(v.numpy(), host.host.v[f'last_obs/{k}'][rows]) for k, v in info['last_obs'].obs.items())
            seen += len(rows)
        assert seen > 0
    finally:
        host.close()


def spoiled(field, change):
    """A spoil (HostBatch) that changes one part of a step's reply: 'reward', 'done', an info field or 'last_obs/NAME';
    change(value) gives the new value, or None to leave the part out."""
    def spoil(reward, done, info):
        parts = dict(info, reward=reward, done=done)
        holder, key = parts, field
        if '/' in field:
            outer, key = field.split('/')
            holder = parts[outer] = dict(parts[outer])
        value = change(holder[key])
        if value is None:
            del holder[key]
        else:
            holder[key] = value
        return parts.pop('reward'), parts.pop('done'), parts
    return spoil


def test_a_worker_refuses_a_batch_that_departs_from_the_declared_table_or_drops_last_obs(threaded, monkeypatch):
    """The workers refuse, with the reason and before anything is written: an observation table or version other than
    the learner's, a batch that cannot be built (a missing setting), workers whose batches run under different settings,
    a step without last_obs, without one of its fields or with one of another shape, a cut life's last observation that
    offers no legal action, an info field missing (it would keep the earlier step's value) or one the buffer lacks, and a
    reward, done flag or info field of another shape or dtype."""
    from sim.workers import CpuWorkers
    fake = dict(task=FREE_PLAY_TASK, goals=None, observation=FAKE_FREE_PLAY, shapes=FAKE_FREE_PLAY_SHAPES)
    with monkeypatch.context() as patch:
        patch.setattr(HostBatch, 'observation', 'obs-v5')
        workers = CpuWorkers(8, 2, 'pool', task=FREE_PLAY_TASK, goals=None, observation='obs-v5', shapes=OBSERVATION_SHAPES)
        try:
            with pytest.raises(RuntimeError, match='departs from the declared table'):
                workers.reset(5)
        finally:
            workers.close()
    with pytest.raises(RuntimeError, match='could not build its batch: ValueError: the batch declares observation '
                                           f'{FAKE_FREE_PLAY}, the learner records obs-v6'):
        CpuWorkers(8, 2, 'pool', **dict(fake, observation='obs-v6'))
    with monkeypatch.context() as patch:
        patch.setattr(HostBatch, 'refuse', 'pass muggers=True or False')
        with pytest.raises(RuntimeError, match='slots 0-3 could not build its batch: ValueError: pass muggers=True or False'):
            CpuWorkers(8, 2, 'pool', **fake)
    with monkeypatch.context() as patch:
        patch.setattr(HostBatch, 'uneven', True)
        with pytest.raises(RuntimeError, match='different settings'):
            CpuWorkers(8, 2, 'pool', **fake)
    cut = lambda mask: np.zeros_like(mask)  # every row's: the refusal comes at the first cut life
    for field, change, match in (
            ('last_obs', lambda value: None, 'last_obs.*missing'),
            ('last_obs/option_vec', lambda value: None, 'last_obs departs from the declared table'),
            ('last_obs/mask', lambda value: value[:, :-1], 'last_obs mask'),
            ('last_obs/mask', cut, 'cut by its length offers no legal action'),
            ('decision_ticks', lambda value: None, r"info departs from the buffer's fields: \['decision_ticks'\]"),
            ('xp_events', None, r"info departs from the buffer's fields: \['xp_events'\]"),
            ('terminal', lambda value: value.astype(np.float32), 'info/terminal: float32'),
            ('reward', lambda value: value[:1], r'reward: float32\(1,\)'),
            ('done', lambda value: value.astype(np.int8), 'done: int8')):
        with monkeypatch.context() as patch:
            if field == 'xp_events':
                patch.setattr(HostBatch, 'spoil', lambda reward, done, info: (reward, done, dict(info, xp_events=done)))
            else:
                patch.setattr(HostBatch, 'spoil', spoiled(field, change))
            workers = CpuWorkers(4, 1, 'pool', **fake)
            try:
                state, obs = workers.reset(5)
                with pytest.raises(RuntimeError, match=match):
                    for _ in range(40):  # a cut comes within a few steps (lives of 4 to 12 ticks)
                        state, obs, *_ = workers.step(state, obs['mask'].argmax(1).astype(np.int32))
            finally:
                workers.close()
    with pytest.raises(ValueError, match='observation version and table'):
        CpuWorkers(8, 2, 'pool', task=FREE_PLAY_TASK, goals=None)


def test_the_free_play_check_records_each_outputs_largest_scale_and_refuses_one_that_is_not_finite():
    """sim/train.py validate_free_play: a rollout's counts keep each output's largest scale (recurrent_max_scale),
    the log-probability's from the legal logits alone (not the mask's fill of -1e9). A legal logit of minus infinity
    keeps the distribution valid and the sampled actions' gaps finite but makes its rows' tolerance infinite: the
    rollout fails by its scale."""
    from sim.train import CheckFailed, validate_free_play
    model = free_model(value_norm=POPART)
    b = collect(model)
    carried = lambda: dict(paid=torch.zeros(WORLDS, dtype=torch.float64), done=torch.ones(WORLDS))
    counts = validate_free_play(model, carried())
    scales = counts['recurrent_max_scale']
    assert counts['recurrent_worst']['ratio'] <= 1 and 0 < scales['log probability'] < 100 and scales['value'] > 0
    never = torch.bincount(b.actions.reshape(-1).long(), minlength=model.policy.action_net.out_features) == 0
    with torch.no_grad():
        model.policy.action_net.bias[never] = -torch.inf  # actions never sampled: some legal, none a sampled row's
    with pytest.raises(CheckFailed, match="a recurrent output's scale is not finite"):
        validate_free_play(model, carried())


def test_an_entropy_coefficient_trains_in_the_loss_is_recorded_and_crosses_by_weights(tmp_path):
    """sim.train train --ent-coef (the free-play tasks): without it a run records no new field, and with config.LEARNER's
    0.01 given it trains bit for bit as without it; 0.05 trains other weights from the same draws (the loss uses it) and
    is recorded in the summary's learner, free_play and changes_from_learner and in every checkpoint. --weights from a
    0.01 checkpoint starts a 0.05 run, the checkpoint's recorded beside it. Values outside 0 to 0.2, a NaN, and the flag
    on a goal task are refused by name before any output."""
    import json
    import zipfile
    from sim.train import train
    common = dict(seed=1, worlds=WORLDS, n_steps=N_STEPS, batch_size=BATCH, n_epochs=1, verbose=False, task=FREE_PLAY_TASK,
                  decisions=2 * WORLDS * N_STEPS, checkpoint_every=WORLDS * N_STEPS)
    runs = {name: train(tmp_path / name, **common, **given)
            for name, given in (('default', {}), ('given', dict(ent_coef=0.01)), ('more', dict(ent_coef=0.05)))}
    default, given, more = runs['default'], runs['given'], runs['more']
    assert 'ent_coef' not in default['free_play'] and 'ent_coef' not in default['changes_from_learner']
    assert default['learner']['ent_coef'] == given['learner']['ent_coef'] == 0.01 and given['free_play']['ent_coef'] == 0.01
    assert given['policy_sha256_at_end'] == default['policy_sha256_at_end'] != more['policy_sha256_at_end']
    assert more['learner']['ent_coef'] == more['free_play']['ent_coef'] == 0.05
    assert more['changes_from_learner']['ent_coef'] == dict(learner=0.01, here=0.05)
    zips = [tmp_path / 'more' / 'latest.zip', *sorted((tmp_path / 'more' / 'checkpoints').glob('*.zip'))]
    assert len(zips) == 3
    for path in zips:
        with zipfile.ZipFile(path) as archive:
            assert json.loads(archive.read('data'))['ent_coef'] == 0.05
    cross = train(tmp_path / 'cross', weights=tmp_path / 'default' / 'latest.zip', ent_coef=0.05, **common)
    assert cross['weights']['saved_settings_not_imported']['ent_coef'] == 0.01 and cross['learner']['ent_coef'] == 0.05
    refusals = ((dict(ent_coef=-0.01), '--ent-coef -0.01: the entropy coefficient is from 0.0 to 0.2'),
                (dict(ent_coef=0.21), '--ent-coef 0.21'), (dict(ent_coef=math.nan), '--ent-coef nan'),
                (dict(ent_coef=0.05, task='goal-activity-v2'), "--ent-coef is the free-play tasks'"))
    for k, (change, reason) in enumerate(refusals):
        with pytest.raises(ValueError, match=reason):
            train(tmp_path / f'refused-{k}', **dict(common, **change))
        assert not (tmp_path / f'refused-{k}').exists()
