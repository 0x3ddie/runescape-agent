"""Game features and scoped masks; PPO and recurrent batching remain upstream."""
from contextlib import contextmanager

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from act_contract import A_ACT0, A_CLICK_HIGH, A_CLICK_LOW, A_HELD0, ACT_V3, N_CLICK, N_CLICK_LOW, N_HELD
from feedback_contract import FEEDBACK_V2
from item_contract import ITEM_V2
from names import name_table
from obs import A_DEPOSIT0, A_DROP0, A_USE0, A_WITHDRAW0, MASK, N_ACTIONS, N_INV_SEL, OBSERVATION
from wide_contract import (ACT_V4, ITEM_V4, MASK_V6, N_ACTIONS_V4, N_PAGE, OBSERVATION_V7, PAGE_FAMILIES, WIDE_SHAPES,
                           page_row)

OPTION_VOCAB = 4001  # hashed option text (act_contract.name_id), +1 for the 0 pad slot
# The action, mask and feedback versions a checkpoint records beside its observation version (native-click).
CONTRACT = dict(actions=ACT_V3, mask=MASK, feedback=FEEDBACK_V2)


def run_contract(items=ITEM_V2):
    """The contract a run records in its checkpoints and checks every checkpoint against: CONTRACT under item-v2 (a
    checkpoint that records no item contract is item-v2's), with the item contract beside it otherwise (`items`:
    item-v3, the page of kinds; item-v4, the page of 16 kinds, with act-v4's 128 rows and mask-v6). check_names compares
    the whole record, so none loads as another."""
    if items == ITEM_V4:
        return dict(CONTRACT, actions=ACT_V4, mask=MASK_V6, items=ITEM_V4)
    return CONTRACT if items == ITEM_V2 else dict(CONTRACT, items=items)


def action_rows(contract):
    """The policy's row count under a recorded contract: act-v4's 128, act-v3's 96 otherwise."""
    return N_ACTIONS_V4 if (contract or {}).get('actions') == ACT_V4 else N_ACTIONS


def identity(names, options, observations, name, width):
    """Name embedding, plus the embedding of its option (or the sum of its options). Slots
    without names (compass moves) have no identity (zeros of the embedding's width). names and options: an
    input's key -> its rows (lookups)."""
    if f'{name}_ids' not in observations:
        feats = observations[f'{name}_feats']
        return feats.new_zeros((*feats.shape[:-1], width), dtype=torch.float32)
    value = names(f'{name}_ids')
    if f'{name}_op_ids' in observations:
        option = options(f'{name}_op_ids')
        value = value + (option.sum(-2) if option.dim() > value.dim() else option)
    return value


def lookups(embedding, options, observations, name_keys, option_keys):
    """(names, options) of one call: an input's key -> its embedded rows. Each use looks its input up, or, with
    NameEmbedding.MERGED, each of the two embeddings looks up all of its inputs at once and the rows are split back
    (the same rows: a lookup copies them), each input once however often it is used."""
    name_keys, option_keys = ([k for k in dict.fromkeys(keys) if k in observations] for keys in (name_keys, option_keys))
    if not name_keys or not NameEmbedding.merges(observations[name_keys[0]]):
        return (lambda key: embedding(observations[key].long(), key)), (lambda key: options(observations[key].long()))

    def merged(module, keys):
        if not keys:
            return {}
        ids = [observations[k] for k in keys]
        rows = module(torch.cat([i.reshape(-1) for i in ids]).long()).split([i.numel() for i in ids])
        return {k: r.view(*i.shape, r.shape[-1]) for k, i, r in zip(keys, ids, rows)}
    return merged(embedding, name_keys).__getitem__, merged(options, option_keys).__getitem__


class NameEmbedding(nn.Module):
    """names-v1 identities: a frozen row of pretrained name vectors (rl/names.py) and a learned
    map to `width`, so a name never trained on still means something. The table is saved in
    the checkpoint; row 0 (no name, or one the cache lacks) maps to zero.
    Inside one policy call (MaskedRecurrentPolicy._masked) `shared` holds each input's torch.unique
    result by observation key, for every embedding of the policy: the actor's and critic's
    extractors and the slot heads read the same rows, and equal ids give equal results. Each
    unique synchronizes a GPU; a call makes one per input, not one per use (obs-v2: 10, not 23).
    FULL_TABLE: every row of the table is projected, once per embedding and call when `shared` is set, and the ids look
    up their rows. Each projected row is the same product as with unique; only the matrix sizes differ, so results are
    equal in exact arithmetic, not bitwise. None (the default) projects the whole table on CUDA and only the names
    present elsewhere: a CUDA graph allows no synchronization or data-dependent shape, and the eager calls (the rollout
    check, an eager update) must do the graphs' arithmetic, or a recomputed log-probability differs from the collected
    one by about 1e-5, the check's tolerance. On the CPU, unique keeps the earlier results bitwise and is cheaper.
    MERGED (lookups): in one call each embedding, this one and the option embedding beside it, looks up all its inputs at
    once instead of once per input and use (6 lookups a policy call, not 30). A lookup copies rows, so the outputs are
    bit for bit the per-input ones wherever the table is projected whole; the backward then sums each row's gradient in
    one pass, the same sum in another order. On CUDA each embedding backward is one kernel (one thread block at width
    16) or a sort per call: they took about a third of a graphed update's kernel time on the G4. None (the default)
    merges wherever the whole table is projected; True or False forces it. A class switch, never saved with a model."""
    shared = None
    FULL_TABLE = None
    MERGED = None

    def __init__(self, rows, dim, width):
        super().__init__()
        self.register_buffer('table', torch.zeros(rows, dim, dtype=torch.float16))
        self.project = nn.Linear(dim, width, bias=False)

    @classmethod
    def whole(cls, ids):
        """Whether a call on `ids` projects the whole table (FULL_TABLE)."""
        return bool(cls.FULL_TABLE or (cls.FULL_TABLE is None and ids.is_cuda))

    @classmethod
    def merges(cls, ids):
        """Whether a call on `ids` looks up each embedding's inputs at once (MERGED)."""
        return cls.whole(ids) if cls.MERGED is None else bool(cls.MERGED)

    def forward(self, ids, key=None):
        # Project only the names present, then look up rows: the same values, without a
        # 384-wide copy of every entity (full-batch imitation on 93 episodes ran an L4 out
        # of memory), and without projecting all 3,366 rows on every step (7 ms per call,
        # most of PPO's time). F.embedding, not indexing: its backward sorts ids, where
        # indexing's atomic adds collide on common names and ran imitation 8 times slower.
        if self.whole(ids):
            shared = self.shared if self.shared is not None else {}
            if self not in shared:
                shared[self] = self.project(self.table.float())
            return nn.functional.embedding(ids, shared[self])
        shared = self.shared if key is not None else None
        found = shared.get(key) if shared is not None else None
        if found is None or found[1].shape != ids.shape:
            found = torch.unique(ids, return_inverse=True)
            if shared is not None:
                shared[key] = found
        present, rows = found
        return nn.functional.embedding(rows, self.project(self.table[present].float()))


def option_embedding(width=16):
    """Learned embeddings of hashed option text (act_contract.name_id); options are not in
    the name table. Options never trained on stay at zero and add no identity."""
    embedding = nn.Embedding(OPTION_VOCAB, width, padding_idx=0)
    nn.init.zeros_(embedding.weight)
    return embedding


# The normalized value target of the goal-free study: PopArt (a running scale was the other choice). Its statistics are
# a debiased exponential average over updates, `beta` the newest rollout's weight; the scale is at least `min_std`. A
# declared setting, not frozen.
POPART = dict(method='popart', beta=0.1, min_std=0.01)


class PopArt(nn.Module):
    """A value head whose output is a normalized value n(h) = W h + b, scaled back: V(h) = std * n(h) + mean
    (van Hasselt et al. 2016, "Preserving Outputs Precisely, while Adaptively Rescaling Targets"). It holds the very weight
    and bias Parameters of the Linear head it replaces, so an optimizer made before keeps them; mean, std and the moments
    are buffers, saved with the policy. update() moves the statistics towards a batch of value targets and rescales W and
    b so that V(h) stays the same for every h (within float32 rounding); the learner then fits n(h) to the normalized
    target, (target - mean) / std, which is the unnormalized error divided by std."""

    def __init__(self, linear, beta, min_std):
        super().__init__()
        if not (0 < beta <= 1 and min_std > 0):
            raise ValueError(f'PopArt needs 0 < beta <= 1 and min_std > 0, not {beta}, {min_std}')
        self.weight, self.bias = linear.weight, linear.bias
        self.beta, self.min_std = beta, min_std
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))
        # the debiasing weight and the first and second moments, each an exponential average over updates (float64)
        self.register_buffer('moments', torch.zeros(3, dtype=torch.float64))

    def forward(self, latent):
        return nn.functional.linear(latent, self.weight, self.bias) * self.std + self.mean

    @torch.no_grad()
    def update(self, targets):
        """Move the statistics towards these value targets (every element counts once) and keep the outputs."""
        x = targets.detach().reshape(-1).double()
        batch = torch.stack([x.new_ones(()), x.mean(), (x * x).mean()])
        self.moments.mul_(1 - self.beta).add_(self.beta * batch)
        mean = self.moments[1] / self.moments[0]
        std = (self.moments[2] / self.moments[0] - mean * mean).clamp(min=self.min_std ** 2).sqrt()
        old_mean, old_std = self.mean.double(), self.std.double()
        self.weight.copy_(self.weight.double() * old_std / std)
        self.bias.copy_((self.bias.double() * old_std + old_mean - mean) / std)
        self.mean.copy_(mean)
        self.std.copy_(std)


def check_value_norm(policy, value_norm):
    """A checkpoint trains on only with the value normalization it was built with (MaskedRecurrentPolicy value_norm): its
    statistics are part of its value head. None is an unnormalized value target, every checkpoint before goal-free play."""
    found = getattr(policy, 'value_norm', None)
    if found != value_norm:
        raise ValueError(f'the checkpoint has value normalization {found or "none"}, the run {value_norm or "none"}')


def load_name_vectors(policy, vectors):
    for module in policy.modules():
        if isinstance(module, NameEmbedding):
            module.table.copy_(torch.as_tensor(vectors, dtype=torch.float16))


# A name table rebuilt from a user's own game files by the same text model and script (scripts/name_vectors.py) can
# differ from the one a policy trained with in a few float16 values from one machine to another: its digest differs
# while it means the same. check_names takes such a table only by the trained table's fingerprint (name_fingerprint),
# which a published policy carries beside it (scripts/export_policy.py): its rows and dimension, the sha256 of its kinds
# and names in order, and for each dimension the sum, the sum of squares and the row-weighted sum (each row's weight its
# place in the table, 1 / N to 1) of its vectors; no vector's values. NAMES_RTOL: float16's step is at most 2^-10 of a
# value's magnitude in its normal range (2^-24 absolute below it, NAMES_ATOL a row), so a table moved one step in every
# value moves a dimension's sum or row-weighted sum by at most 2^-10 of its sum of magnitudes (the same sum weighted),
# and its sum of squares by at most 2^-9 + 2^-20 of itself; 2^-8 admits about two such steps in every value. 1e-3 would
# refuse a table one step off in every value away from zero (its sums of squares move up to 1.95e-3), and a scale of 1%
# moves every sum of squares by 2.01%, five times 2^-8. The row-weighted sums see vectors moved between rows, which the
# plain sums do not.
NAMES_FINGERPRINT = 'names-fingerprint-v1'
NAMES_RTOL, NAMES_ATOL = 2 ** -8, 2 ** -24


def name_fingerprint(table):
    """The fingerprint (NAMES_FINGERPRINT) of name table `table` (rl/names.py NameTable): no vector's values."""
    import hashlib
    order = sorted(table.rows, key=table.rows.get)
    vectors, weights = _fingerprint_parts(table)
    return dict(version=NAMES_FINGERPRINT, digest=table.digest, rows=int(table.spec[0]), dim=int(table.spec[1]),
                names_sha256=hashlib.sha256(''.join(f'{kind}\t{name}\n' for kind, name in order).encode()).hexdigest(),
                sums=vectors.sum(0).tolist(), squares=(vectors ** 2).sum(0).tolist(),
                weighted=(weights * vectors).sum(0).tolist())


def _fingerprint_parts(table):
    """A table's vectors (its rows past row 0, float64 from their float16 values) and each row's weight, 1 / N to 1."""
    vectors = table.vectors[1:].astype(np.float64)
    return vectors, np.arange(1, len(vectors) + 1, dtype=np.float64)[:, None] / max(1, len(vectors))


def fingerprint_problem(fingerprint, table, trained):
    """Why name table `table` is not the one fingerprint `fingerprint` describes, the table of digest `trained` (a
    policy's names[2]), or None: another version or digest, rows or dimension, kinds or names, or a sum beyond its
    tolerance (NAMES_RTOL of its scale, plus NAMES_ATOL a row)."""
    if not isinstance(fingerprint, dict) or fingerprint.get('version') != NAMES_FINGERPRINT:
        return f'no {NAMES_FINGERPRINT} fingerprint'
    if fingerprint.get('digest') != trained:
        return f"a fingerprint of table {fingerprint.get('digest')}, the policy's is {trained}"
    mine = name_fingerprint(table)
    if (fingerprint.get('rows'), fingerprint.get('dim')) != (mine['rows'], mine['dim']):
        return f"{mine['rows']} rows of {mine['dim']}, the fingerprint's {fingerprint.get('rows')} of {fingerprint.get('dim')}"
    if fingerprint.get('names_sha256') != mine['names_sha256']:
        return 'its kinds and names (in order) are not the fingerprint\'s'
    vectors, weights = _fingerprint_parts(table)
    floor = NAMES_ATOL * len(vectors)
    for key, scale in (('sums', np.abs(vectors).sum(0)), ('weighted', (weights * np.abs(vectors)).sum(0)),
                       ('squares', (vectors ** 2).sum(0))):
        want = np.asarray(fingerprint.get(key), dtype=np.float64)
        if want.shape != scale.shape:
            return f'the fingerprint\'s {key} are not {len(scale)} numbers'
        gap = np.abs(np.asarray(mine[key]) - want)
        over = np.flatnonzero(gap > NAMES_RTOL * scale + floor)
        if len(over):
            d = int(over[0])
            return (f'the {key} of dimension {d} are {mine[key][d]!r}, the fingerprint\'s {float(want[d])!r}: beyond '
                    f'{NAMES_RTOL:g} of {scale[d]:.6g} (and {len(over) - 1} more dimensions)')
    return None


def check_names(policy, observation=OBSERVATION, contract=CONTRACT, fingerprint=None):
    """A policy reads names through the table it was trained with, the one projection reads, and
    inputs of the observation version it was trained on (rl/obs.py OBSERVATION) under the action, mask
    and feedback versions it was trained with (CONTRACT). A checkpoint that records another version, or
    none (every one saved before obs-v3), is refused; an obs-v3, mask-v3 one plays only after
    convert_checkpoint. observation=None skips both checks (historical tools). The table in use
    (RL_NAME_TABLE) must have the policy's recorded digest, or, given the trained table's `fingerprint`
    (name_fingerprint), hold it (fingerprint_problem). It changes nothing (a public copy's vectors are filled by its
    loader: sim/hour.py load_free_policy). Returns how the table was taken: path 'digest' or 'fingerprint', the table's
    digest, and for a fingerprint the trained digest and the tolerance."""
    names = getattr(policy, 'names', None)
    if not names:
        raise ValueError('the checkpoint predates names-v1')
    table = name_table()
    if names[2] == table.digest:
        taken = dict(path='digest', table=table.digest)
    elif fingerprint is None:
        raise ValueError('the checkpoint was trained with another name table than RL_NAME_TABLE')
    else:
        problem = fingerprint_problem(fingerprint, table, names[2])
        if problem is not None:
            raise ValueError(f'the checkpoint was trained with another name table than RL_NAME_TABLE, by its '
                             f'fingerprint: {problem}')
        taken = dict(path='fingerprint', table=table.digest, trained=names[2], rtol=NAMES_RTOL)
    if observation is not None and getattr(policy, 'observation', None) != observation:
        raise ValueError(f"the checkpoint reads {getattr(policy, 'observation', None) or 'obs-v1'} inputs, not "
                         f'{observation}; train a fresh model, or convert an obs-v3 one (convert_checkpoint)')
    if observation is not None and getattr(policy, 'contract', None) != contract:
        found = getattr(policy, 'contract', None) or 'act-v2, mask-v2 and feedback-v1 (none recorded)'
        raise ValueError(f'the checkpoint was trained under {found}, not {contract}; train a fresh model, or convert a '
                         'mask-v3 one (convert_checkpoint)')
    return taken


class GameFeatures(BaseFeaturesExtractor):
    # Nearby npcs and affordances are encoded as sets: one shared per-entity layer,
    # then max and mean over the entities present. Affordances, places and held
    # options also have action slots, which MaskedRecurrentPolicy scores one at a time.
    SETS = ('npc', 'act')
    # Places reach only their pointer: in the global view their doors, ladders and
    # crates told imitation which village it was in, and at Varrock west it waited on
    # the bank tile for 1,524 decisions beside a choppable tree. Held options are the
    # inventory page again, which the global view already has.
    IGNORED = ('mask', 'ref_probs', 'place_ids', 'place_op_ids', 'place_feats', 'held_ids', 'held_op_ids', 'held_feats')
    # Inputs for the critic's own extractor only (asymmetric actor-critic, as in
    # AlphaStar's value network): training uses them, the deployed actor never does.
    CRITIC_ONLY = ('critic_vec',)

    def __init__(self, observation_space, names, features_dim=256, critic=False):
        super().__init__(observation_space, features_dim)
        # The mask gates logits and ref_probs feeds the KL penalty; neither is a network input.
        sets = [name for name in self.SETS if {f'{name}_ids', f'{name}_feats'} <= set(observation_space.spaces)]
        set_keys = {f'{name}_{part}' for name in sets for part in ('ids', 'op_ids', 'feats')}
        ignored = (*self.IGNORED, *set_keys, *(() if critic else self.CRITIC_ONLY))
        self.keys = sorted(k for k in observation_space.spaces if k not in ignored)
        self.embedding = NameEmbedding(names[0], names[1], 16)
        self.options = option_embedding()
        self.sets = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(16 + observation_space[f'{name}_feats'].shape[-1], 64), nn.ReLU()) for name in sets})
        width = sum(int(np.prod(observation_space[k].shape)) *
                    (16 if k.endswith(('_ids', '_op')) else 1) for k in self.keys) + 128 * len(sets)
        self.encoder = nn.Sequential(nn.Linear(width, 512), nn.ReLU(),
                                     nn.Linear(512, features_dim), nn.ReLU())

    def forward(self, observations):
        names, options = lookups(self.embedding, self.options, observations,
                                 [k for k in self.keys if k.endswith('_ids')] + [f'{name}_ids' for name in self.sets],
                                 [k for k in self.keys if k.endswith('_op')] + [f'{name}_op_ids' for name in self.sets])
        features = []
        for key in self.keys:
            value = observations[key]
            if key.endswith('_ids'):
                value = names(key)
            elif key.endswith('_op'):  # a hashed option (feedback-v1's last action): not a name
                value = options(key)
            features.append(value.float().flatten(1))
        for name, layer in self.sets.items():
            feats = observations[f'{name}_feats'].float()
            # A thing whose name the table lacks has id 0 but is still there.
            present = ((observations[f'{name}_ids'] > 0) | (feats.abs().sum(-1) > 0)).unsqueeze(-1).float()
            entities = layer(torch.cat([identity(names, options, observations, name, self.embedding.project.out_features),
                                        feats], dim=-1)) * present
            count = present.sum(1).clamp(min=1)
            features += [entities.max(1).values, entities.sum(1) / count]
        return self.encoder(torch.cat(features, dim=1))


class MaskedRecurrentPolicy(RecurrentMultiInputActorCriticPolicy):
    """Apply the observation's legality mask in every upstream actor entrypoint.

    No copied rollout, GAE, recurrent-buffer or optimizer implementation. Mask
    context is synchronous and scoped; one policy object is not thread-reentrant.
    Recurrent PPO pads observation sequences with zero rows during optimization;
    only evaluate_actions accepts that sentinel. Upstream excludes padded losses. A call reads the
    device once for all its checks and finds each name input's rows once (NameEmbedding.shared).

    Two learner-speed paths, both equal to SB3's in exact arithmetic but not bitwise, so they are
    switches of the class, never saved with a checkpoint. Both are on by default since 18 October: on the A100 their
    replayed update stayed inside stock's own run-to-run difference, and the PPO update took 43 s instead of 111 s
    at 1,024 worlds. Off reproduces SB3 exactly:
    FUSED_SEQUENCES: one LSTM call over a minibatch's sequences instead of SB3's step-by-step loop. The
    buffer starts a sequence at every episode start, so a start can only be a sequence's first step; it
    resets that sequence's initial state. A start after the first step falls back to SB3's loop.
    VALID_ROWS: evaluate_actions checks every row first (a real row with no legal action raises, as
    always), then runs the feature extractors, heads, distribution and value on the real rows only. The
    padded rows, all zero and always after a sequence's real steps, enter the LSTMs as zero features and
    return zero values, log-probabilities and entropies, which the loss masks out.
    """
    FUSED_SEQUENCES = True
    VALID_ROWS = True

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('features_extractor_class', GameFeatures)
        self._active_mask = None
        self._active_slots = None
        self._active_real = None
        self._late_starts = None
        self._distribution = None
        # KL(reference || policy) penalty, off unless train.py sets it; see evaluate_actions.
        self.kl_weight = 0.0
        self.kl_log = []
        # (rows, dim, digest) of the names-v1 table, and the observation version (rl/obs.py), both
        # saved with the checkpoint's policy_kwargs; a checkpoint without the version predates obs-v2.
        self.names = kwargs.pop('names')
        self.observation = kwargs.pop('observation', None)
        self.contract = kwargs.pop('contract', None)
        # convert_checkpoint's record (method, source and its versions) stays in the model's saved policy_kwargs.
        kwargs.pop('conversion', None)
        # The value target's normalization (POPART), saved with the policy_kwargs; None (every checkpoint before goal-free
        # play) keeps SB3's Linear value head.
        self.value_norm = kwargs.pop('value_norm', None)
        super().__init__(*args, **kwargs)
        rows = action_rows(self.contract)
        if not isinstance(self.action_space, spaces.Discrete) or self.action_space.n != rows:
            raise ValueError(f'Game policy requires the Discrete({rows}) head of its contract')
        if self.value_norm is not None:
            settings = dict(self.value_norm)
            if settings.pop('method', None) != 'popart':
                raise ValueError(f'value_norm is PopArt (POPART) or None, not {self.value_norm}')
            # the head's own Parameters, which the optimizer already holds, under the same state-dict names
            self.value_net = PopArt(self.value_net, **settings)
        self._name_embeddings = [m for m in self.modules() if isinstance(m, NameEmbedding)]

    def make_features_extractor(self):
        # Built first for the actor; without share_features_extractor, again for the
        # critic. The critic's own extractor learns the return, not frozen imitation features.
        critic = self.features_extractor is not None
        kwargs = {'names': self.names, **self.features_extractor_kwargs}
        return self.features_extractor_class(self.observation_space, critic=critic, **kwargs)

    # Action rows scored one observed slot at a time (a pointer, as in AlphaStar): (scorer, slots, first row, first
    # slot, slot count or None for all). Drop, deposit and use all point at the inventory page; one click scorer
    # scores the click slots in their two row groups (act-v3). Under act-v4 (WIDE_POINTERS) the page families lie in
    # two groups as well, entries 8..15 on rows 96..127, each scored by its family's one scorer.
    POINTERS = (('act', 'act', A_ACT0, 0, None), ('click', 'click', A_CLICK_LOW, 0, N_CLICK_LOW),
                ('held', 'held', A_HELD0, 0, N_HELD), ('drop', 'sel', A_DROP0, 0, N_INV_SEL),
                ('deposit', 'sel', A_DEPOSIT0, 0, N_INV_SEL), ('withdraw', 'bank', A_WITHDRAW0, 0, None),
                ('use', 'sel', A_USE0, 0, N_INV_SEL), ('click', 'click', A_CLICK_HIGH, N_CLICK_LOW, N_CLICK - N_CLICK_LOW))
    WIDE_POINTERS = tuple((name, slots, page_row(base, N_INV_SEL), N_INV_SEL, N_INV_SEL) for (name, slots), (base, _) in
                          zip((('held', 'held'), ('drop', 'sel'), ('deposit', 'sel'), ('use', 'sel')), PAGE_FAMILIES))

    def _build_mlp_extractor(self):
        super()._build_mlp_extractor()
        # ACT_k is scored from slot k alone, with weights shared by all slots, so a
        # plain tree one tile away scores the same in any slot and any village. Fixed
        # rows had learned which mix of trees and npcs meant "chop", and waited beside
        # four trees in Varrock. Every other slot family is scored the same way.
        observed = set(self.observation_space.spaces)
        wide = self.WIDE_POINTERS if self.action_space.n == N_ACTIONS_V4 else ()
        self.pointers = [pointer for pointer in (*self.POINTERS, *wide) if f'{pointer[1]}_feats' in observed]
        self.slot_embedding = NameEmbedding(self.names[0], self.names[1], 16)
        self.slot_options = option_embedding()
        for name, slots in dict.fromkeys((name, slots) for name, slots, *_ in self.pointers):
            width = self.mlp_extractor.latent_dim_pi + 16 + self.observation_space[f'{slots}_feats'].shape[-1]
            setattr(self, f'{name}_scorer', nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1)))

    @contextmanager
    def _masked(self, obs, *, padding=False, late=None):
        """late: FUSED_SEQUENCES's device flag for an episode start after a sequence's first step; it joins
        the call's one host read, and _process_sequence takes the answer from there."""
        if self._active_mask is not None:
            raise RuntimeError('Concurrent/reentrant policy use is unsupported')
        mask = obs['mask']
        if mask.ndim != 2 or mask.shape[1] != self.action_space.n:
            raise ValueError('Invalid action mask')
        binary = torch.all((mask == 0) | (mask == 1))
        mask = mask.bool().clone()
        empty = ~mask.any(dim=1)
        if padding:
            # A padded row is zero in every input: one flag per row. Indexing each input by `empty`
            # synchronized a GPU twice per input (71 times a call, 34 inputs).
            filled = torch.stack([(value != 0).flatten(1).any(1) if value.ndim > 1 else value != 0
                                  for value in obs.values()]).any(0)
            real_empty = empty & filled
        else:
            real_empty = empty
        # Every check of the call in one host read (each read synchronizes a GPU): bit 0 a mask
        # value other than 0 and 1, bit 1 a real row with no legal action, bit 2 the late start.
        flags = [~binary, torch.any(real_empty)] + ([late] if late is not None else [])
        found = int(sum(flag.long() << bit for bit, flag in enumerate(flags)))
        if found & 1:
            raise ValueError('Invalid action mask')
        if found & 2:
            raise ValueError('All-masked real observation')
        mask[:, 0] |= empty
        self._active_real = ~empty  # the rows with a legal action: every row that is not padding
        self._active_mask = mask
        self._active_slots = obs
        self._late_starts = bool(found & 4) if late is not None else None
        shared = {}
        for embedding in self._name_embeddings:
            embedding.shared = shared
        try:
            yield
        finally:
            self._active_mask = None
            self._active_slots = None
            self._active_real = None
            self._late_starts = None
            for embedding in self._name_embeddings:
                embedding.shared = None

    def _process_sequence(self, features, lstm_states, episode_starts, lstm):
        """SB3's _process_sequence; with FUSED_SEQUENCES one LSTM call (class docstring)."""
        if not self.FUSED_SEQUENCES:
            return super()._process_sequence(features, lstm_states, episode_starts, lstm)
        n_seq = lstm_states[0].shape[1]
        sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        late = self._late_starts  # evaluate_actions's one read answered it; collection steps have one row
        if late is None:
            late = starts.shape[0] > 1 and bool(torch.any(starts[1:] != 0))
        if late:
            return super()._process_sequence(features, lstm_states, episode_starts, lstm)
        keep = (1.0 - starts[0]).view(1, n_seq, 1)
        output, lstm_states = lstm(sequence, (keep * lstm_states[0], keep * lstm_states[1]))
        return torch.flatten(output.transpose(0, 1), start_dim=0, end_dim=1), lstm_states

    def _evaluate_real_rows(self, obs, actions, lstm_states, episode_starts):
        """RecurrentActorCriticPolicy.evaluate_actions with every row-wise part on the real rows only; the mask
        context (already entered on the whole batch, which checked every row) is narrowed to those rows for
        the heads. Returns full-size values, log-probabilities and entropies, zero on the padded rows."""
        rows = self._active_real.nonzero().squeeze(1)
        n = self._active_real.shape[0]
        full_mask, full_slots = self._active_mask, self._active_slots
        sub = {k: v[rows] for k, v in obs.items()}

        def padded(x):
            return x.new_zeros((n, *x.shape[1:])).index_copy(0, rows, x)
        try:
            self._active_mask, self._active_slots = full_mask[rows], sub
            pi_features, vf_features = self.extract_features(sub)
            latent_pi, _ = self._process_sequence(padded(pi_features), lstm_states.pi, episode_starts, self.lstm_actor)
            latent_vf, _ = self._process_sequence(padded(vf_features), lstm_states.vf, episode_starts, self.lstm_critic)
            latent_pi = self.mlp_extractor.forward_actor(latent_pi[rows])
            latent_vf = self.mlp_extractor.forward_critic(latent_vf[rows])
            distribution = self._get_action_dist_from_latent(latent_pi)
            log_prob, entropy = distribution.log_prob(actions[rows]), distribution.entropy()
            values = self.value_net(latent_vf)
        finally:
            self._active_mask, self._active_slots = full_mask, full_slots
        return values, log_prob, entropy, rows, padded

    def evaluate_compact(self, obs, actions, lstm_states, keep, pos, n_seq, max_length):
        """evaluate_actions on a minibatch's real rows only, with no host read (sim/learn.py DeviceRecurrentPPO's
        FAST_UPDATE; obs: sim/learn.py CompactSamples rows): both speed paths' arithmetic, operation for operation, so
        on one device it equals evaluate_actions on the padded minibatch at its real rows. Its per-call checks are made
        once per update by the caller (check_rows). The LSTMs run on the padded n_seq x max_length grid (`pos`, each
        row's place in it), starting from `keep` times the stored states. The distribution's log-probability and entropy
        are torch Categorical's arithmetic without its argument checks. Returns values, log-probabilities, entropies [B]."""
        if not (self.share_features_extractor is False and self.lstm_critic is not None) or self.kl_weight:
            raise ValueError('evaluate_compact needs separate actor and critic extractors and LSTMs, and no KL penalty')
        if self._active_mask is not None:
            raise RuntimeError('Concurrent/reentrant policy use is unsupported')
        shared = {}
        for embedding in self._name_embeddings:
            embedding.shared = shared
        self._active_mask, self._active_slots = obs['mask'].bool(), obs
        try:
            pi_features, vf_features = self.extract_features(obs)
            latent_pi = self._grid_sequence(pi_features, lstm_states.pi, keep, pos, n_seq, max_length, self.lstm_actor)
            latent_vf = self._grid_sequence(vf_features, lstm_states.vf, keep, pos, n_seq, max_length, self.lstm_critic)
            latent_pi = self.mlp_extractor.forward_actor(latent_pi)
            latent_vf = self.mlp_extractor.forward_critic(latent_vf)
            logits = self._masked_logits(latent_pi)
            values = self.value_net(latent_vf)
        finally:
            self._active_mask = self._active_slots = None
            for embedding in self._name_embeddings:
                embedding.shared = None
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)  # torch.distributions.Categorical(logits=...)
        log_prob = logits.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        entropy = -(torch.clamp(logits, min=torch.finfo(logits.dtype).min) * logits.softmax(-1)).sum(-1)
        return values.flatten(), log_prob, entropy

    @staticmethod
    def _grid_sequence(features, states, keep, pos, n_seq, max_length, lstm):
        """The FUSED_SEQUENCES LSTM call of _evaluate_real_rows: the real rows' features scattered into the zero-padded
        grid, one LSTM call from keep times the stored states, the outputs at the real rows."""
        grid = features.new_zeros((n_seq * max_length, features.shape[1])).index_copy(0, pos, features)
        sequence = grid.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        keep = keep.view(1, n_seq, 1)
        output, _ = lstm(sequence, (keep * states[0], keep * states[1]))
        return torch.flatten(output.transpose(0, 1), start_dim=0, end_dim=1)[pos]

    @staticmethod
    def check_rows(mask):
        """evaluate_actions' per-call checks for every stored row at once (one host read): mask values 0 or 1 and a legal
        action in every row. The collector's own policy call made them on the same rows already."""
        mask = mask.reshape(-1, mask.shape[-1])
        found = int((~torch.all((mask == 0) | (mask == 1))).long() | ((~mask.bool().any(1)).any().long() << 1))
        if found & 1:
            raise ValueError('Invalid action mask')
        if found & 2:
            raise ValueError('All-masked real observation')

    def _masked_logits(self, latent_pi):
        """The masked logits of _get_action_dist_from_latent, before its distribution."""
        logits = self.action_net(latent_pi)
        obs = self._active_slots
        names, options = lookups(self.slot_embedding, self.slot_options, obs, [f'{s}_ids' for _, s, *_ in self.pointers],
                                 [f'{s}_op_ids' for _, s, *_ in self.pointers])
        for name, slots, start, first, count in self.pointers:
            feats = obs[f'{slots}_feats'].float()
            width = self.slot_embedding.project.out_features
            rows = torch.cat([latent_pi.unsqueeze(1).expand(-1, feats.shape[1], -1),
                              identity(names, options, obs, slots, width), feats], dim=-1)
            rows = rows[:, first:first + (feats.shape[1] if count is None else count)]
            logits = torch.cat([logits[:, :start], getattr(self, f'{name}_scorer')(rows).squeeze(-1),
                                logits[:, start + rows.shape[1]:]], dim=1)
        if logits.shape != self._active_mask.shape:
            raise ValueError('Mask/logit batch mismatch')
        return logits.masked_fill(~self._active_mask, -1e9)

    def _get_action_dist_from_latent(self, latent_pi):
        if self._active_mask is None:
            raise RuntimeError('Actor distribution requested outside mask context')
        self._distribution = self.action_dist.proba_distribution(action_logits=self._masked_logits(latent_pi))
        return self._distribution

    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        with self._masked(obs):
            return super().forward(obs, lstm_states, episode_starts, deterministic)

    def get_distribution(self, obs, lstm_states, episode_starts):
        with self._masked(obs):
            return super().get_distribution(obs, lstm_states, episode_starts)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        """Upstream PPO loss, plus kl_coef * KL(reference || policy) when enabled.

        Upstream PPO adds ent_coef * -entropy to its loss. Returning
        entropy - kl_weight * KL, with kl_weight = kl_coef / ent_coef, adds exactly
        kl_coef * KL without copying the training loop. So with the penalty on,
        SB3's logged entropy_loss includes it; train.py logs the KL separately.
        """
        rows = late = None
        if self.FUSED_SEQUENCES:  # both LSTMs see the same sequences: one late-start flag for the two
            starts = episode_starts.reshape((lstm_states.pi[0].shape[1], -1))
            late = torch.any(starts[:, 1:] != 0) if starts.shape[1] > 1 else None
        with self._masked(obs, padding=True, late=late):  # every row checked, padded or not
            if self.VALID_ROWS and self.share_features_extractor is False and self.lstm_critic is not None:
                values, log_prob, entropy, rows, padded = self._evaluate_real_rows(obs, actions, lstm_states, episode_starts)
            else:
                values, log_prob, entropy = super().evaluate_actions(obs, actions, lstm_states, episode_starts)
        reference = obs.get('ref_probs')
        if self.kl_weight and reference is not None:
            reference = reference if rows is None else reference[rows]
            log_policy = self._distribution.distribution.logits
            kl = (torch.special.xlogy(reference, reference) - reference * log_policy).sum(dim=1)
            real = reference.sum(dim=1) > 0  # padded rows carry no reference
            if torch.any(real):
                self.kl_log.append(float(kl[real].mean().detach()))
            entropy = entropy - self.kl_weight * kl
        if rows is not None:
            values, log_prob, entropy = padded(values), padded(log_prob), padded(entropy)
        return values, log_prob, entropy


# The versions convert_checkpoint takes and gives.
CONVERTIBLE = dict(observation='obs-v3', contract=dict(CONTRACT, mask='mask-v3'))


def goal_columns(extractor, space):
    """The columns of a GameFeatures extractor's first encoder layer that read goal_vec: (first, width)."""
    first = 0
    for key in extractor.keys:
        if key == 'goal_vec':
            return first, int(np.prod(space[key].shape))
        first += int(np.prod(space[key].shape)) * (16 if key.endswith(('_ids', '_op')) else 1)
    raise ValueError('the extractor does not read goal_vec')


def convert_checkpoint(source, output):
    """The one tested conversion: an obs-v3 checkpoint under mask-v3, whose goal_vec
    has goal-v2's four entries, to obs-v5 under mask-v5 with goal-v3's six. Every tensor is copied, and each feature
    extractor's first encoder layer gets zero weights for the two new goal entries, so for the four old goals every
    output equals the source's up to float rounding. The mask is no network input and mask-v5 only adds page turns while
    an item is selected: the source plays under it as a declared conversion, reported apart. Optimizer state is not
    carried. The result records `conversion` (method, source path and sha256, source versions). Returns the record."""
    from sb3_contrib import RecurrentPPO
    from baseline.resource_task import GOAL_VEC, GOALS
    from baseline.telemetry import sha256
    model = RecurrentPPO.load(source, device='cpu')
    policy, old = model.policy, model.observation_space
    check_names(policy, observation=CONVERTIBLE['observation'], contract=CONVERTIBLE['contract'])
    if 'goal_vec' not in old.spaces or old['goal_vec'].shape != (len(GOALS),):
        raise ValueError('the checkpoint does not read the four-entry goal_vec of goal-v2')
    goal = old['goal_vec']
    space = spaces.Dict({**old.spaces, 'goal_vec': spaces.Box(goal.low.min(), goal.high.max(), (len(GOAL_VEC),),
                                                             goal.dtype)})
    record = dict(method='goal-v2 to goal-v3: zero encoder weights for goal_vec entries 4 and 5; mask-v3 declared under '
                         'mask-v5', source=str(source), source_sha256=sha256(source),
                  source_observation=policy.observation, source_contract=policy.contract)
    kwargs = dict(model.policy_kwargs, observation=OBSERVATION, contract=CONTRACT, conversion=record)
    new = type(policy)(space, model.action_space, model.lr_schedule, use_sde=model.use_sde, **kwargs)
    state = policy.state_dict()
    for name, module in policy.named_modules(remove_duplicate=False):
        if isinstance(module, GameFeatures):
            first, width = goal_columns(module, old)
            key = f'{name}.encoder.0.weight'
            weight = state[key]
            state[key] = torch.cat([weight[:, :first + width], weight.new_zeros((weight.shape[0], len(GOAL_VEC) - width)),
                                    weight[:, first + width:]], dim=1)
    new.load_state_dict(state, strict=True)
    model.policy, model.observation_space, model.policy_kwargs = new, space, kwargs
    model.save(output)
    return record


def widen_state(state, policy):
    """The one tested crossing to the wide contract (rl/wide_contract.py): the state dict
    `state` of an obs-v6 policy of act-v3's 96 rows (free-play-v3 or v10), as the obs-v7, act-v4 policy `policy` holds
    it. Every tensor is copied; each GameFeatures extractor's first encoder layer gets zero weights for the page's new
    entries 8..15 (sel_ids, sel_feats; the pack options' inputs reach only the scorers, which are kept), and the action
    head zero weights and biases for rows 96..127, which the families' kept scorers score in any case. So on a pack of 8
    kinds or fewer every score of rows 0..95 and the value equal the source's up to float rounding, and so do the choice
    probabilities under the source's mask (rl/tests/test_wide_contract.py); under mask-v6 the play can differ, since
    mask-v6 has no drop rule. Returns the new state dict; the caller checks the source's own versions first."""
    if policy.observation != OBSERVATION_V7 or policy.action_space.n != N_ACTIONS_V4:
        raise ValueError(f'widen_state gives an {OBSERVATION_V7} policy of {N_ACTIONS_V4} rows, not '
                         f'{policy.observation} of {policy.action_space.n}')
    state, space = dict(state), policy.observation_space
    for name, module in policy.named_modules(remove_duplicate=False):
        if isinstance(module, GameFeatures):
            key = f'{name}.encoder.0.weight'
            weight, parts, first = state[key], [], 0
            for k in module.keys:
                width = int(np.prod(space[k].shape)) * (16 if k.endswith(('_ids', '_op')) else 1)
                old = width // N_PAGE * N_INV_SEL if k in WIDE_SHAPES else width  # entry-major: entries 0..7 first
                parts += [weight[:, first:first + old], weight.new_zeros((weight.shape[0], width - old))]
                first += old
            if weight.shape[1] - first != 128 * len(module.sets):  # then the sets' pooled features (forward), kept
                raise ValueError(f"the source's {name} does not read obs-v6's inputs ({weight.shape[1]} columns)")
            state[key] = torch.cat([*parts, weight[:, first:]], dim=1)
    for key in ('action_net.weight', 'action_net.bias'):
        head = state[key]
        if head.shape[0] != N_ACTIONS:
            raise ValueError(f"the source's action head has {head.shape[0]} rows, not act-v3's {N_ACTIONS}")
        state[key] = torch.cat([head, head.new_zeros((N_ACTIONS_V4 - N_ACTIONS, *head.shape[1:]))])
    return state

if __name__ == '__main__':
    import argparse
    import json
    # The module as baseline.policy, not __main__: a loaded checkpoint's extractors are baseline.policy.GameFeatures, and
    # this module's own classes, run as __main__, are other classes, so no goal column would be found.
    from baseline.policy import convert_checkpoint as convert
    parser = argparse.ArgumentParser(description='convert_checkpoint: an obs-v3, mask-v3 checkpoint to obs-v5, mask-v5')
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=1))
