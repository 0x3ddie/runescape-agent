"""feedback-v2 (env-ts/core/feedback.ts) -> fixed arrays.

feedback: what the last decision asked for (its meaning when chosen: kind, name, option, the
item in hand), how a running option ended, what changed since that decision, and the text
the game showed. It is evidence only; nothing here concludes why an action failed.

Text becomes hashed words and word pairs (feature hashing, Weinberger et al. 2009):
deterministic across processes (crc32, never Python's hash), negations kept as words. It
carries no pretrained meaning; that can replace it later.

text-v2 (rl/obs.py): a number below 100 is its own token, so "a Mining level of 15"
and "of 30" differ; a larger one is a token of its digit count (bounded vocabulary).
text-v1 made every number "<num>".

feedback-v2 (native-click, act-v3) is feedback-v1 with one `click` family in place of the place, retreat and move
families; a click's meaning is its offset, as its option text "dx,dz".
"""
import re
import zlib

import numpy as np

from act_contract import name_id

FEEDBACK_V2 = 'feedback-v2'
FAMILIES = ('wait', 'act', 'click', 'held', 'drop', 'deposit', 'withdraw', 'close', 'dialog', 'page', 'style', 'use')
END_REASONS = ('done', 'stalled', 'expired', 'failed', 'superseded')
TEXT_DIM = 64
CHOICE_DIM = 16
N_CHOICES = 4
# one-hot family + "no action yet" + running + end reason + refused + 11 changes
FEEDBACK_FEATS = len(FAMILIES) + 1 + 1 + len(END_REASONS) + 1 + 11
REFUSED = len(FAMILIES) + 1 + 1 + len(END_REASONS)  # fb_vec's refused flag (feedback-v1's was 21)
TEXT_FEATS = 2 * TEXT_DIM + 5
KINDS = ('loc', 'npc', 'obj')
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")


def number(digits):
    """text-v2: 0..99 exactly (levels, hitpoints, small counts); larger numbers by digit count."""
    digits = digits.lstrip('0') or '0'
    return digits if len(digits) <= 2 else f'<num{min(len(digits), 10)}>'


def words(text):
    """Lowercase words of game text; numbers as number() tokens. Formatting tags (<col=...>) go."""
    text = re.sub(r'<[^>]*>', ' ', str(text).lower())
    return [number(t) if t.isdigit() else t for t in _WORD.findall(text)]


def text_vector(texts, dim):
    """Signed hashed counts of words and adjacent word pairs, unit length (zero for no text)."""
    v = np.zeros(dim, dtype=np.float32)
    for text in texts:
        tokens = words(text)
        for feature in tokens + [f'{a} {b}' for a, b in zip(tokens, tokens[1:])]:
            h = zlib.crc32(feature.encode())
            v[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def project_feedback(obs, ident):
    """`ident(kind, name)` gives a thing's row in the name table (rl/names.py)."""
    fb = obs.get('feedback')
    _check(isinstance(fb, dict) and fb.get('version') == FEEDBACK_V2, 'Invalid feedback version')
    for key in ('ticks', 'dx', 'dz', 'xp', 'hp', 'gained', 'lost'):
        _check(type(fb.get(key)) is int, f'Invalid feedback {key}')
    for key in ('running', 'levelChanged', 'bankOpened', 'dialogOpened'):
        _check(type(fb.get(key)) is bool, f'Invalid feedback {key}')
    _check(fb['ticks'] >= 0 and fb['xp'] >= 0 and fb['gained'] >= 0 and fb['lost'] >= 0, 'Negative feedback change')
    action, ended, chat, dialog = fb.get('action'), fb.get('ended'), fb.get('chat'), fb.get('dialog')
    vec = np.zeros(FEEDBACK_FEATS, dtype=np.float32)
    ids = np.zeros(2, dtype=np.int32)  # the thing acted on, the item in hand
    option = np.zeros(1, dtype=np.int32)
    if action is None:
        vec[len(FAMILIES)] = 1.0
    else:
        _check(isinstance(action, dict) and action.get('family') in FAMILIES and type(action.get('action')) is int
               and all(isinstance(action.get(k), str) for k in ('kind', 'name', 'option', 'item')), 'Invalid feedback action')
        vec[FAMILIES.index(action['family'])] = 1.0
        if action['kind'] in KINDS and action['name']:
            ids[0] = ident(action['kind'], action['name'])
        if action['item']:
            ids[1] = ident('obj', action['item'])
        option[0] = name_id(action['option']) if action['option'] else 0
    k = len(FAMILIES) + 1
    vec[k] = fb['running']
    if ended is not None:
        _check(isinstance(ended, dict) and ended.get('reason') in END_REASONS and type(ended.get('refused')) is bool,
               'Invalid feedback end')
        vec[k + 1 + END_REASONS.index(ended['reason'])] = 1.0
        vec[REFUSED] = ended['refused']
    k += 2 + len(END_REASONS)
    vec[k:] = [min(fb['ticks'] / 50, 4.0), np.clip(fb['dx'] / 16, -2, 2), np.clip(fb['dz'] / 16, -2, 2),
               min(max(abs(fb['dx']), abs(fb['dz'])) / 16, 2.0), fb['levelChanged'], np.log1p(fb['xp']) / 10,
               np.clip(fb['hp'] / 10, -1, 1), np.log1p(fb['gained']) / 3, np.log1p(fb['lost']) / 3,
               fb['bankOpened'], fb['dialogOpened']]
    _check(isinstance(chat, list) and all(isinstance(c, dict) and isinstance(c.get('text'), str) and type(c.get('age')) is int
                                          and c['age'] >= 0 for c in chat), 'Invalid feedback chat')
    _check(isinstance(dialog, dict) and isinstance(dialog.get('text'), str) and isinstance(dialog.get('choices'), list)
           and all(isinstance(c, str) for c in dialog['choices']), 'Invalid feedback dialog')
    text = np.zeros(TEXT_FEATS, dtype=np.float32)
    text[:TEXT_DIM] = text_vector([c['text'] for c in chat], TEXT_DIM)
    text[TEXT_DIM:2 * TEXT_DIM] = text_vector([dialog['text']], TEXT_DIM)
    text[2 * TEXT_DIM:] = [bool(chat), min(min((c['age'] for c in chat), default=0) / 50, 4.0), min(len(chat), 4) / 4,
                           bool(dialog['text'].strip()), min(len(dialog['choices']), 5) / 5]
    # Each choice's own text, at the position of its DIALOG_CHOICE row.
    choices = np.zeros((N_CHOICES, CHOICE_DIM), dtype=np.float32)
    for i, choice in enumerate(dialog['choices'][:N_CHOICES]):
        choices[i] = text_vector([choice], CHOICE_DIM)
    return {'fb_vec': vec, 'fb_ids': ids, 'fb_op': option, 'text_vec': text, 'choice_vec': choices}


def quiet(running=False, dialog=None):
    """Feedback with no action yet and nothing changed: an episode's first observation."""
    return dict(version=FEEDBACK_V2, action=None, running=running, ended=None, ticks=0, dx=0, dz=0, levelChanged=False,
                xp=0, hp=0, gained=0, lost=0, bankOpened=False, dialogOpened=False, chat=[],
                dialog=dialog or dict(text='', choices=[]))


def episode_start(raw):
    """The first observation of an episode: setup's last action, changes and chat belong to no
    decision of this episode (setup runs as ordinary steps before the task is armed). An open
    dialog is still on screen, so its text stays."""
    return dict(raw, feedback=quiet(bool(raw['feedback']['running']), raw['feedback']['dialog']))
