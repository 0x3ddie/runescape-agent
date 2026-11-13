"""Actor self-distillation from a bounded memory of the agent's own earlier episodes (the sustained
resource search study's arm B, run label S). Reinforcement learning with an auxiliary self-distillation loss: PPO keeps learning from fresh
rollouts; after each PPO update the actor alone takes DISTILL['steps'] optimizer steps on KL(teacher || learner) over
stored histories. It uses the agent's own sampled experience, never a scripted or external label.

The memory (`collect`, frozen before a campaign) holds complete episodes that the teacher (the run's initialization)
played with its own sampled actions on training starts, in DISTILL['strata']: visible wood, fish and mine
(goal-activity-v2 starts on the training pool, 240 ticks), hidden-tree search (search-wood-v1 on the search training
pool) and the four sustained-combat start groups (healthy, hurt with and without food, a mugger nearby), every outcome
kept. Each stored decision holds the observation the policy received and the teacher's probability of every action.
Every DISTILL['probe']-th episode of a stratum is held out: both arms measure the actor's drift from the teacher on it.

Recurrent history is rebuilt, not stored: the learner reads each sampled episode from its first decision with a zero
state (an episode start), so its LSTM sees the same history the teacher did. With the teacher's own weights it reproduces
the stored probabilities (`check`). The critic takes no part: no value target, no gradient. Old trajectories never enter
the PPO objective; the loss gives every stratum the same weight.

    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.distill memory --teacher CKPT --output MEMORY.npz
    PYTHONPATH=.:rl .venv-sim/bin/python -m sim.distill check --memory MEMORY.npz --policy CKPT
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

DISTILL = dict(
    version='distill-v1',
    # stratum: (task, goal, combat group or None), episodes. The teacher's own sampled episodes, every outcome kept.
    strata=dict(wood=('goal-activity-v2', 'wood', None, 48), fish=('goal-activity-v2', 'fish', None, 48),
                mine=('goal-activity-v2', 'mine', None, 48), search=('goal-activity-v4', 'wood', None, 48),
                healthy=('goal-activity-v3', 'melee', 'healthy', 32), hurt_fed=('goal-activity-v3', 'melee', 'hurt_fed', 32),
                hurt_unfed=('goal-activity-v3', 'melee', 'hurt_unfed', 32), danger=('goal-activity-v3', 'melee', 'danger', 32)),
    probe=8,          # every 8th episode of a stratum (index % 8 == 7) is held out: the drift probe, never a training row
    key=707001,       # the memory's keyed starts (training layouts; slot = the stratum's index, episode = its count)
    seed=808001,      # the teacher's sampling seed base (one torch seed per episode)
    steps=64,         # distillation optimizer steps after each PPO update (the run's Adam, learning rate and clipping)
    per_stratum=1,    # episodes of each stratum in one step (8 episodes a step)
    coef=1.0,         # the loss: coef x the strata's mean of their mean per-decision KL(teacher || learner)
    probe_every=8,    # updates between drift probes (and the last update)
)


# ---- the teacher's episodes -------------------------------------------------------------------------------------
def teacher_step(actor, obs):
    """sim.combat.PolicyActor.act's computation with sampled actions: (action, every action's probability, the
    observation as the policy received it)."""
    torch, policy = actor.torch, actor.model.policy
    obs = {**obs, **{k: np.zeros(s.shape, s.dtype) for k, s in actor.model.observation_space.spaces.items() if k not in obs}}
    tensor, _ = policy.obs_to_tensor(obs)
    with torch.no_grad():
        action, _, _, actor.state = policy.forward(tensor, actor.state, torch.tensor([float(actor.first)]), deterministic=False)
        probs = policy._distribution.distribution.probs[0].numpy().astype(np.float32)
    actor.first = False
    return int(action.item()), probs, {k: np.asarray(obs[k], actor.model.observation_space.spaces[k].dtype) for k in obs}


def stratum_tasks(name, k, tables, pools):
    """The k-th start of a stratum as a started task (fresh world): its keyed draw and episode builder."""
    from baseline import config
    from sim.combat import COMBAT, combat_npcs, draw_combat, kit
    from sim.engine import World
    from sim.search import draw_search, episode as search_episode
    from sim.task import WORLD_START_TICK, GoalEpisode, draw_start
    task_name, goal, group, _ = DISTILL['strata'][name]
    slot = list(DISTILL['strata']).index(name)
    if task_name == 'goal-activity-v4':
        layouts, tiles, lts = pools['search']
        start = draw_search(DISTILL['key'], slot, k, layouts, tiles)
        return search_episode(start, layouts[start['layout']], lts[start['layout']], tables), start
    layouts, lts = pools['gather']
    start = draw_start(DISTILL['key'], slot, k, layouts)
    lt = lts[start['layout']]
    if group is None:
        world = World(lt, seed=start['world_seed'], start_tick=WORLD_START_TICK)
        task = GoalEpisode(world, config.TASK_CFG, goal, horizon=240, scale=1.0)
        task.start(start['tile'], start['weapon'], start['levels'])
        return task, start
    c = start['combat'] = draw_combat(DISTILL['key'], slot, k, layouts[start['layout']], start, tables, group=group)
    world = World(lt, seed=start['world_seed'], start_tick=WORLD_START_TICK, npcs=combat_npcs(layouts[start['layout']], tables,
                                                                                               c['muggers']),
                  exclusions=COMBAT['exclusions'])
    task = GoalEpisode(world, config.TASK_CFG, 'melee', horizon=COMBAT['horizon'], scale=1.0, combat=True)
    task.start(start['tile'], start['weapon'], start['levels'], kit=kit(start['weapon'], c['food']), hp=c['hp'], wield=c['worn'])
    return task, start


def collect(teacher, output, tables, gather_pool, search_pool, strata=None):
    """The memory: every stratum's episodes played by the teacher (a checkpoint path) with sampled actions. Writes an npz
    of the stacked decisions (each observation array, `probs`, `action`, `episode`) and per-episode `stratum`, `first`,
    `length`, `probe`, `success`, `survived`, `return`; returns its manifest."""
    from sim.combat import PolicyActor
    from sim.engine import Unsupported
    from sim.search import SEARCH, search_tiles
    from sim.world import LayoutTables, load_pool
    from baseline.telemetry import sha256
    t0 = time.monotonic()
    layouts, derived = load_pool(gather_pool)
    search_layouts, _ = load_pool(search_pool)
    lts = [LayoutTables(tables, layout, None if derived is None else derived[k]) for k, layout in enumerate(layouts)]
    slts = [LayoutTables(tables, layout) for layout in search_layouts]
    for lt in slts:
        lt.CACHE = SEARCH['cache']
    pools = dict(gather=(layouts, lts), search=(search_layouts, [search_tiles(lay, tables) for lay in search_layouts], slts))
    actor = PolicyActor(str(teacher), deterministic=False)
    steps, episodes = {}, []
    names = list(strata or DISTILL['strata'])
    for s, name in enumerate(names):
        wanted = DISTILL['strata'][name][3]
        k = 0
        while sum(e['stratum'] == s for e in episodes) < wanted:
            task, start = stratum_tasks(name, k, tables, pools)
            actor.reset(DISTILL['seed'] + 1000 * s + k)
            obs, rows, info = task.projected, [], None
            try:
                for _ in range(4000):
                    action, probs, seen = teacher_step(actor, obs)
                    rows.append((seen, probs, action))
                    obs, _, done, info = task.step(action)
                    if done:
                        break
            except Unsupported:  # never stored: an unsupported step has no successor state
                k += 1
                continue
            first = sum(len(v) for v in steps.get('action', []))
            for key in rows[0][0]:
                steps.setdefault(key, []).append(np.stack([r[0][key] for r in rows]))
            steps.setdefault('probs', []).append(np.stack([r[1] for r in rows]))
            steps.setdefault('action', []).append(np.asarray([r[2] for r in rows], np.int16))
            n = sum(e['stratum'] == s for e in episodes)
            episodes.append(dict(stratum=s, first=first, length=len(rows), probe=n % DISTILL['probe'] == DISTILL['probe'] - 1,
                                 success=bool(info and info['episode']['r'] == 1.0), survived=bool(info and info['report']['survived']),
                                 ret=float(info['episode']['r']) if info else 0.0, start=k))
            k += 1
    arrays = {k: np.concatenate(v) for k, v in steps.items()}
    arrays['episode'] = np.concatenate([np.full(e['length'], i, np.int32) for i, e in enumerate(episodes)])
    for field in ('stratum', 'first', 'length', 'probe', 'success', 'survived'):
        arrays[f'ep_{field}'] = np.asarray([e[field] for e in episodes], np.int32 if field in ('stratum', 'first', 'length') else bool)
    arrays['ep_return'] = np.asarray([e['ret'] for e in episodes], np.float32)
    arrays['strata'] = np.asarray(names)
    np.savez_compressed(output, **arrays)
    manifest = dict(version=DISTILL['version'], teacher=dict(path=str(teacher), sha256=sha256(teacher)), strata=names,
                    decisions=int(len(arrays['action'])), episodes=len(episodes),
                    per_stratum={name: dict(episodes=int(sum(e['stratum'] == s for e in episodes)),
                                            decisions=int(sum(e['length'] for e in episodes if e['stratum'] == s)),
                                            successes=int(sum(e['success'] for e in episodes if e['stratum'] == s)),
                                            deaths=int(sum(not e['survived'] for e in episodes if e['stratum'] == s)),
                                            probe=int(sum(e['probe'] for e in episodes if e['stratum'] == s)),
                                            skipped_unsupported=int(max(e['start'] for e in episodes if e['stratum'] == s) + 1 -
                                                                    sum(e['stratum'] == s for e in episodes)))
                                for s, name in enumerate(names)},
                    settings={k: v for k, v in DISTILL.items() if k != 'strata'}, seconds=round(time.monotonic() - t0, 1),
                    sha256=sha256(output))
    Path(output).with_suffix('.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


# ---- the memory on the learner's device -------------------------------------------------------------------------
class Memory:
    """A collected memory on a device: its decisions as tensors, its episodes' bounds and strata."""

    def __init__(self, path, observation_space, device='cpu'):
        import torch
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        self.path, self.device = str(path), torch.device(device)
        self.obs = {k: torch.as_tensor(arrays[k]).to(self.device) for k in observation_space.spaces}
        self.probs = torch.as_tensor(arrays['probs']).to(self.device)
        self.first, self.length = arrays['ep_first'].astype(np.int64), arrays['ep_length'].astype(np.int64)
        self.stratum, self.probe = arrays['ep_stratum'], arrays['ep_probe'].astype(bool)
        self.names = [str(n) for n in arrays['strata']]
        self.train = {s: np.flatnonzero((self.stratum == s) & ~self.probe) for s in range(len(self.names))}
        self.held = {s: np.flatnonzero((self.stratum == s) & self.probe) for s in range(len(self.names))}

    def batch(self, episodes):
        """Padded sequences of the given episodes, sequence-major as the policy's minibatches: (observations, episode
        starts, valid rows, teacher probabilities, the stratum of each row, the number of sequences)."""
        import torch
        n, longest = len(episodes), int(self.length[episodes].max())
        offsets = np.arange(longest)
        valid = offsets[None, :] < self.length[episodes][:, None]
        rows = torch.as_tensor(np.where(valid, self.first[episodes][:, None] + offsets[None, :], 0).reshape(-1), device=self.device)
        keep = torch.as_tensor(valid.reshape(-1), device=self.device)

        def pad(store):
            value = store[rows]
            return value.masked_fill(~keep.reshape(-1, *[1] * (value.dim() - 1)), 0)
        starts = torch.zeros(n, longest, device=self.device)
        starts[:, 0] = 1.0
        strata = torch.as_tensor(np.repeat(self.stratum[episodes], longest), device=self.device)
        return {k: pad(v) for k, v in self.obs.items()}, starts.reshape(-1), keep, pad(self.probs), strata, n


def actor_log_probs(policy, obs, episode_starts, n_seq):
    """The actor's log-probabilities of every action on padded sequences that each begin at an episode start (zero
    recurrent state); padded rows give zeros. The critic is not run. Mirrors baseline.policy's _evaluate_real_rows."""
    import torch
    from stable_baselines3.common.policies import BaseModel
    lstm = policy.lstm_actor
    state = (torch.zeros(lstm.num_layers, n_seq, lstm.hidden_size, device=episode_starts.device),
             torch.zeros(lstm.num_layers, n_seq, lstm.hidden_size, device=episode_starts.device))
    with policy._masked(obs, padding=True, late=torch.zeros((), dtype=torch.bool, device=episode_starts.device)):
        rows = policy._active_real.nonzero().squeeze(1)
        n = policy._active_real.shape[0]
        full_mask, full_slots = policy._active_mask, policy._active_slots
        sub = {k: v[rows] for k, v in obs.items()}
        try:
            policy._active_mask, policy._active_slots = full_mask[rows], sub
            features = BaseModel.extract_features(policy, sub, policy.pi_features_extractor)
            padded = features.new_zeros((n, *features.shape[1:])).index_copy(0, rows, features)
            latent, _ = policy._process_sequence(padded, state, episode_starts, lstm)
            latent = policy.mlp_extractor.forward_actor(latent[rows])
            logits = policy._get_action_dist_from_latent(latent).distribution.logits
        finally:
            policy._active_mask, policy._active_slots = full_mask, full_slots
    return logits.new_zeros((n, logits.shape[1])).index_copy(0, rows, logits)


def kl_by_stratum(policy, memory, episodes):
    """{stratum index: mean per-decision KL(teacher || actor)} over the given episodes, as a tensor that keeps the graph."""
    import torch
    obs, starts, keep, teacher, strata, n = memory.batch(episodes)
    log_policy = actor_log_probs(policy, obs, starts, n)
    kl = (torch.special.xlogy(teacher, teacher) - teacher * log_policy).sum(dim=1)
    return {int(s): kl[keep & (strata == s)].mean() for s in np.unique(memory.stratum[episodes])}


class Distill:
    """The distillation phase (train=True: arm S, or arm T with `strata`) or only the drift probe (train=False, arm A),
    attached to a model (sim/learn.py attach) and run after each PPO update (DeviceRecurrentPPO.train). `strata`: the
    strata the phase trains (default every one). Each keeps the weight it has in arm S, 1 / (all strata): the loss is the
    trained strata's KL sum over the count of every stratum, and each step samples one episode of each trained stratum.
    The drift probe always reads every stratum."""

    def __init__(self, memory_path, model, seed, train, strata=None):
        import torch
        self.memory = Memory(memory_path, model.observation_space, model.device)
        unknown = set(strata or ()) - set(self.memory.names)
        if unknown:
            raise ValueError(f'strata not in the memory: {sorted(unknown)}')
        self.strata = list(self.memory.names) if strata is None else [n for n in self.memory.names if n in strata]
        self.train_phase, self.updates, self.records = train, 0, []
        self.sampler = np.random.default_rng([seed, 4])  # episode sampling; its state is part of the recovery state
        self.torch = torch

    def state_dict(self):
        import torch
        state = json.dumps(self.sampler.bit_generator.state)
        return dict(updates=self.updates, sampler=torch.as_tensor(np.frombuffer(state.encode(), np.uint8).copy()))

    def load_state_dict(self, state):
        self.updates = int(state['updates'])
        self.sampler.bit_generator.state = json.loads(bytes(state['sampler'].cpu().numpy().astype(np.uint8)).decode())

    def run(self, model):
        """After one PPO update: the distillation steps (arm S) and, every probe_every updates, the drift probe."""
        torch, policy, memory = self.torch, model.policy, self.memory
        t0, losses, steps = time.perf_counter(), [], 0
        if self.train_phase:
            # training mode, as PPO leaves it: cuDNN's LSTM backward refuses eval mode, and the policy has no
            # dropout or normalization layer, so the outputs are the same. The telemetry's hooks may keep these
            # activations, but it accumulates only on PPO's steps, whose own forwards replace them first.
            tracker, model.tracker = model.tracker, None  # telemetry counts PPO steps only (sim/learn.py Tracker)
            policy.set_training_mode(True)
            try:
                trained = [s for s in sorted(memory.train) if memory.names[s] in self.strata]
                for _ in range(DISTILL['steps']):
                    episodes = np.concatenate([self.sampler.choice(memory.train[s], DISTILL['per_stratum'], replace=False)
                                               for s in trained])
                    per = kl_by_stratum(policy, memory, episodes)
                    if len(trained) == len(memory.names):  # arm S, as it ran
                        loss = DISTILL['coef'] * torch.stack(list(per.values())).mean()
                    else:  # arm T: each trained stratum at arm S's weight, 1 / (all strata)
                        loss = DISTILL['coef'] * torch.stack(list(per.values())).sum() / len(memory.names)
                    policy.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
                    # the run's Adam without its step hooks: sim/train.py counts PPO's steps there
                    from sim.learn import step_hooks_suspended
                    with step_hooks_suspended(policy.optimizer):
                        policy.optimizer.step()
                    losses.append(loss.detach())
                    steps += 1
            finally:
                model.tracker = tracker
        record = dict(steps=steps, loss_mean=float(torch.stack(losses).mean()) if losses else None,
                      seconds=round(time.perf_counter() - t0, 3))
        self.updates += 1
        if self.updates % DISTILL['probe_every'] == 1 or DISTILL['probe_every'] == 1:
            record['drift'] = self.probe(policy)
        self.records.append(record)
        return record

    def probe(self, policy):
        """{stratum: mean KL(teacher || actor)} on the held-out episodes (no gradient)."""
        torch, memory = self.torch, self.memory
        held = np.concatenate([memory.held[s] for s in sorted(memory.held)])
        with torch.no_grad():
            per = kl_by_stratum(policy, memory, held)
        return {memory.names[s]: round(float(v), 6) for s, v in per.items()}


def check(memory_path, policy_path, device='cpu', episodes=None):
    """The memory read back through a policy (the teacher, normally): per stratum the largest absolute difference between
    its log-probabilities and the stored teacher's on legal actions, and the mean KL. With the teacher both are near
    zero, which shows the recurrent histories are rebuilt as the teacher saw them."""
    import torch
    from sb3_contrib import RecurrentPPO
    model = RecurrentPPO.load(policy_path, device=device)
    memory = Memory(memory_path, model.observation_space, device)
    policy = model.policy
    policy.set_training_mode(False)
    out = {}
    for s, name in enumerate(memory.names):
        chosen = np.flatnonzero(memory.stratum == s)[:episodes]
        worst, kls = 0.0, []
        for k in range(0, len(chosen), 8):
            batch = chosen[k:k + 8]
            obs, starts, keep, teacher, strata, n = memory.batch(batch)
            with torch.no_grad():
                log_policy = actor_log_probs(policy, obs, starts, n)
            legal = (teacher > 0) & keep[:, None]
            diff = (log_policy - torch.log(teacher.clamp(min=1e-30))).abs()[legal]
            worst = max(worst, float(diff.max()) if diff.numel() else 0.0)
            kls.append(((torch.special.xlogy(teacher, teacher) - teacher * log_policy).sum(1))[keep])
        out[name] = dict(episodes=int(len(chosen)), max_abs_log_prob_difference=worst, mean_kl=float(torch.cat(kls).mean()))
    return out


def main():
    from sim.data import Tables
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    mem = sub.add_parser('memory', help='collect the teacher\'s episodes (training starts)')
    mem.add_argument('--teacher', type=Path, required=True)
    mem.add_argument('--output', type=Path, required=True)
    mem.add_argument('--gather-pool', type=Path, default=Path('logs/sim/world/train-8-nc1-derived.npz'))
    mem.add_argument('--search-pool', type=Path, default=Path('logs/search/pools/train-128.npz'))
    mem.add_argument('--tables', type=Path)
    mem.add_argument('--strata', nargs='+', choices=list(DISTILL['strata']))
    chk = sub.add_parser('check', help='read the memory back through a policy')
    chk.add_argument('--memory', type=Path, required=True)
    chk.add_argument('--policy', type=Path, required=True)
    chk.add_argument('--episodes', type=int, help='the first episodes of each stratum only')
    args = parser.parse_args()
    if args.command == 'memory':
        tables = Tables.load(args.tables, check_pins=False) if args.tables else Tables.load()
        print(json.dumps(collect(args.teacher, args.output, tables, args.gather_pool, args.search_pool, args.strata), indent=1))
    else:
        print(json.dumps(check(args.memory, args.policy, episodes=args.episodes), indent=1))


if __name__ == '__main__':
    main()
