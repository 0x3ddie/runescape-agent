"""The native study runner's frozen evaluation commands (baseline/study.py): the final checkpoint's two sampled passes
beside the greedy pass, on the same records."""
from pathlib import Path

from baseline import study


def test_a_sampled_pass_differs_from_the_greedy_command_only_by_its_mode_and_declared_seed():
    target = Path('out/study_arena_wood-000')
    greedy = study.evaluate_command('ckpt.zip', target, 'study_arena_wood', 0, 25)
    sampled = study.evaluate_command('ckpt.zip', target, 'study_arena_wood', 0, 25, sampled=True, seed=study.SAMPLED_SEEDS[1])
    assert '--sampled' not in greedy and greedy[greedy.index('--seed') + 1] == str(study.EVAL_SEED)
    assert '--sampled' in sampled and sampled[sampled.index('--seed') + 1] == '909001'
    strip = lambda c: [a for k, a in enumerate(c) if a not in ('--sampled', '--seed') and c[k - 1] != '--seed']
    assert strip(greedy) == strip(sampled)  # fixture, trials, first trial, --final and the world service: the same records
    assert len(study.SAMPLED_SEEDS) == 2 == len(set(study.SAMPLED_SEEDS)) and study.SAMPLED_SEEDS[0] == study.EVAL_SEED
    control = study.evaluate_command(None, target, 'study_arena_wood', 25, 25, policy='reference')
    assert '--policy' in control and '--checkpoint' not in control and control[control.index('--first-trial') + 1] == '25'
