"""What a run did, recorded so that every claim can be traced to the evidence for it.

Each measure says what it counts. Activity is not progress: a
requested action is not an effect, an effect (XP, a moved player) is not certified
production, and production is not reward, which is 0 after a death.

- manifest(): the code (with a patch that restores uncommitted source), game and settings
  that produced a run.
- effect(): what changed in the game between two decisions.
- EpisodeRecorder: adds the episode id to every row, writes one 'effect' row per decision
  and one 'episode_measures' row per finished episode, and offers each finished episode
  to a TraceSink, which keeps the full rows of selected training episodes.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading

import numpy as np

from baseline import config
from baseline.world import JsonlEvidence
from act_contract import CLICK_ROWS

ROOT = Path(__file__).resolve().parents[1]
GATHER_SKILLS = ('Woodcutting', 'Fishing', 'Mining')
CELL = 8  # tiles; decision_cells counts these
DISPLACED = 3  # tiles a click must shift the player to count as displaced
PROBLEM_ENDS = ('failed', 'stalled', 'expired')  # with refusals, the option ends that count as problems


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _git(*args, cwd=ROOT):
    try:
        return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source(directory=None, root=ROOT):
    """The source a run used: its commit and, when the tree differs from it, a patch of tracked
    changes and untracked files that are not ignored, written to directory/source.patch.
    `git apply source.patch` on a checkout of the commit restores those files exactly."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=root, capture_output=True, timeout=60)
    try:
        head = git('rev-parse', 'HEAD')
        if head.returncode:
            return dict(commit=None, error='not a git checkout')
        changed = git('status', '--porcelain', '--untracked-files=no').stdout.decode().splitlines()
        untracked = git('ls-files', '--others', '--exclude-standard').stdout.decode().splitlines()
        state = dict(commit=head.stdout.decode().strip(), changed=changed, untracked=untracked, patch=None)
        if (changed or untracked) and directory is not None:
            patch = git('diff', 'HEAD', '--binary').stdout + b''.join(
                git('diff', '--no-index', '--binary', '/dev/null', name).stdout for name in untracked)
            (Path(directory) / 'source.patch').write_bytes(patch)
            state['patch'] = dict(file='source.patch', sha256=hashlib.sha256(patch).hexdigest(), bytes=len(patch))
        return state
    except (OSError, subprocess.SubprocessError) as error:
        return dict(commit=None, error=str(error))


def manifest(directory=None, **extra):
    """The source (see source(); its patch goes in `directory`), the patched game, the world
    settings, the configuration, the name table and the libraries that produced a run."""
    from names import DEFAULT_TABLE
    vendor = ROOT / 'vendor' / 'rs-sdk'
    names = Path(os.environ.get('RL_NAME_TABLE') or DEFAULT_TABLE)
    return dict(
        created=datetime.now(timezone.utc).isoformat(timespec='seconds'), argv=sys.argv,
        code=source(directory),
        game=dict(vendor_commit=_git('rev-parse', 'HEAD', cwd=vendor),
                  vendor_changed=len((_git('status', '--porcelain', '--untracked-files=no', cwd=vendor) or '').splitlines()),
                  speed_ms=config.SPEED_MS, xp_rate=config.XP_RATE,
                  worlds=json.loads(os.environ['RL_WORLDS']) if os.environ.get('RL_WORLDS') else None),
        config=json.loads(json.dumps(dict(task=config.TASK_CFG, learner=config.LEARNER, finetune=config.FINETUNE,
                                          kl_coef=config.KL_COEF, reward_scale=config.REWARD_SCALE), default=str)),
        names=dict(path=str(names), sha256=sha256(names) if names.exists() else None),
        python=platform.python_version(),
        packages={p: version(p) for p in ('torch', 'stable-baselines3', 'sb3-contrib', 'gymnasium', 'numpy')},
        **extra)


def _counts(inventory):
    counts = Counter()
    for item in inventory:
        counts[f"{item.get('name')} ({item.get('id')})"] += item.get('count', 1)
    return counts


def _change(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(set(before) | set(after)) if after.get(k, 0) != before.get(k, 0)}


def effect(before, after):
    """What changed in the game from one decision's observation to the next: XP per skill,
    carried items by name (id) (a deposit is a loss), HP, position, and what the game reported:
    the action's meaning, what the engine did with it (`action_note`: dispatched, continued,
    not dispatched, or an error), whether an option is still running, which action's option
    ended, how and why (a WAIT can end an earlier move), chat lines and dialog text."""
    p, q = before.get('player', {}), after.get('player', {})
    row = dict(xp=_change({s['name']: s['xp'] for s in before.get('skills', [])}, {s['name']: s['xp'] for s in after.get('skills', [])}),
               inventory=_change(_counts(before.get('inventory', [])), _counts(after.get('inventory', []))),
               hp=q.get('hp', 0) - p.get('hp', 0), dx=q.get('x', 0) - p.get('x', 0), dz=q.get('z', 0) - p.get('z', 0),
               level_changed=q.get('level', 0) != p.get('level', 0))
    fb = after.get('feedback') or {}
    meaning = {k: v for k, v in (fb.get('action') or {}).items() if k != 'action' and v}
    if meaning:
        row['meaning'] = meaning
    row['action_note'] = after.get('actionNote') or ''
    if fb.get('running'):
        row['running'] = True
    if fb.get('ended'):
        row['ended'] = dict(action=fb['ended'].get('action'), reason=fb['ended'].get('reason'),
                            refused=bool(fb['ended'].get('refused')), note=after.get('lastOptionNote') or '')
    if fb.get('chat'):
        row['chat'] = [line['text'] for line in fb['chat']]
    if (fb.get('dialog') or {}).get('text'):
        row['dialog'] = fb['dialog']['text']
    return row


def gathering_xp(raw):
    return sum(s['xp'] for s in raw.get('skills', []) if s['name'] in GATHER_SKILLS)


def _cell(raw):
    p = raw.get('player', {})
    return p.get('level', 0), p.get('x', 0) // CELL, p.get('z', 0) // CELL


class EpisodeMeasures:
    """Movement and exploration in one episode, from the raw observation after each executed
    decision. A measure only: a field it cannot read counts as unchanged, never as an error."""
    def __init__(self, raw):
        self.cells = {_cell(raw)}
        self.start, self.xp0 = raw.get('tick', 0), gathering_xp(raw)
        self.moves = self.displaced = 0
        self.gathering_xp_tick, self.probability = None, []

    def update(self, action, raw, click_probability=None):
        fb = raw.get('feedback', {})
        self.cells.add(_cell(raw))
        if action in CLICK_ROWS:
            self.moves += 1
            self.displaced += max(abs(fb.get('dx', 0)), abs(fb.get('dz', 0))) >= DISPLACED
        if self.gathering_xp_tick is None and gathering_xp(raw) > self.xp0:
            self.gathering_xp_tick = raw.get('tick', 0) - self.start
        if click_probability is not None:
            self.probability.append(click_probability)

    def summary(self):
        return dict(decision_cells=len(self.cells), clicks_requested=self.moves,
                    clicks_displaced=self.displaced, gathering_xp_tick=self.gathering_xp_tick,
                    click_probability=float(np.mean(self.probability)) if self.probability else None,
                    click_probability_states=len(self.probability))


class EpisodeRecorder:
    """Wraps one agent's record(kind, **data). A 'scheduled_episode' row (its `index` is the
    episode id) starts an episode; every later row carries `episode`, and the rows it writes
    carry the starting row's `agent` too. Writes one 'effect' row
    per decision and one 'episode_measures' row per finished episode, keeps the latter as
    `last`, and offers the episode's rows to `traces`; an episode replaced or cut off before
    its terminal row (a failed restart, the end of a run, an error) is offered as interrupted.
    `notes` are added to the measures row."""
    def __init__(self, record, traces=None):
        self.record, self.traces = record, traces
        self.episode, self.last, self.started, self.finished, self.rows = None, None, False, False, []
        if traces is not None:
            traces.recorders.append(self)

    def flush(self):
        """Offer the current episode's rows if it never reached its terminal row."""
        if self.traces is not None and self.rows and not self.finished:
            self.traces.offer_interrupted(self.episode, self.rows)
        self.rows = []

    def __call__(self, kind, **data):
        if kind == 'scheduled_episode':
            self.flush()
            self.episode, self.fixture, self.rows, self.notes = data['index'], data['fixture'], [], {}
            self.ids = {k: data[k] for k in ('agent',) if k in data}
            self.measures = self.before = self.after = self.action = self.probability = None
            self.decision, self.ends, self.refused, self.problems = 0, Counter(), 0, 0
            self.started = self.finished = False
        if self.episode is not None:
            data = dict(data, episode=self.episode)
        self._write(kind, data)
        if self.episode is None:
            return
        if kind == 'episode_start':
            self.before, self.started = data['observation'], True
            self.measures = EpisodeMeasures(self.before)
        elif kind == 'observed':
            self.after = data['observation']
        elif kind == 'evaluation_decision':
            self.probability = data.get('click_probability')
        elif kind == 'task_transition' and self.measures is not None:
            self._decision(data)
        elif kind == 'task_terminal' and self.measures is not None:
            self._finish(data['report'], data['reward'])

    def _write(self, kind, data):
        self.record(kind, **data)
        if self.traces is not None and self.episode is not None:
            self.rows.append((kind, data))

    def _decision(self, transition):
        row = dict(decision=self.decision, action=transition['requested_action'], executed=transition['executed_action'],
                   status=transition['status'], native_tick=transition['native_tick'], reward=transition['reward'])
        if transition['executed_action'] is not None and self.after is not None:
            row.update(tick=self.after.get('tick'), **effect(self.before, self.after))
            if 'ended' in row:
                # A refused option can still end 'done' ("I can't reach that!"): count refusals apart.
                ended = row['ended']
                self.ends[ended['reason']] += 1
                self.refused += ended['refused']
                self.problems += ended['refused'] or ended['reason'] in PROBLEM_ENDS
            self.measures.update(transition['executed_action'], self.after, self.probability)
            self.before = self.after
        self.decision += 1
        self.after = self.probability = None
        self._write('effect', dict(row, **self.ids, episode=self.episode))

    def _finish(self, report, episode_return):
        # final_value is game price (0 after death); episode_return is the reward's own total, which
        # for a goal-v1 task is the normalized goal return, not comparable with coin value.
        row = dict(fixture_id=self.fixture.get('id'), fixture_kind=self.fixture.get('kind'), goal=self.fixture.get('goal'),
                   episode_return=episode_return, decisions=self.decision,
                   survived=report['survived'], items_produced=report['produced'],
                   value_held_at_end=report['retained_lower'], final_value=report['reward_eligible_lower'],
                   deposits=report['deposit_events'], production_after_deposit=report['production_after_deposit'],
                   provenance_exact=report['provenance_exact'], option_ends=dict(self.ends), options_refused=self.refused,
                   option_problems=self.problems,
                   **self.measures.summary(), **self.notes)
        self.last, self.finished = row, True
        self._write('episode_measures', dict(row, **self.ids, episode=self.episode))
        if self.traces is not None:
            self.traces.offer(self.episode, row, self.rows)


class TraceSink:
    """Full rows of selected training episodes (training otherwise keeps effects and measures).
    RULE: keep an episode that died; one whose option problems (ends that failed, stalled or
    expired, or were refused) number at least 20 and a quarter of its decisions; one that never
    reached its terminal row; and every 50th episode of each agent. At most `cap` episodes a run
    (disk: a full episode is about 10 MB before compression); the rest are counted."""
    def __init__(self, path, cap=200):
        self.log, self.cap, self.lock = JsonlEvidence(path), cap, threading.Lock()
        self.kept, self.dropped, self.recorders = Counter(), Counter(), []

    @staticmethod
    def reason(episode, measures):
        problems = measures['option_problems']
        if not measures['survived']:
            return 'death'
        if problems >= 20 and 4 * problems >= measures['decisions']:
            return 'option_problems'
        return 'sample' if episode % 50 == 0 else None

    def offer_interrupted(self, episode, rows):
        self._keep('interrupted', episode, rows)

    def offer(self, episode, measures, rows):
        reason = self.reason(episode, measures)
        if reason is not None:
            self._keep(reason, episode, rows)

    def _keep(self, reason, episode, rows):
        with self.lock:
            full = sum(self.kept.values()) >= self.cap
            (self.dropped if full else self.kept)[reason] += 1
        if not full:
            self.log('trace_kept', episode=episode, reason=reason, agent=rows[0][1].get('agent') if rows else None)
            for kind, data in rows:
                self.log(kind, **data)

    def state(self):
        with self.lock:
            return dict(kept=dict(self.kept), dropped=dict(self.dropped), cap=self.cap)

    def close(self):
        for recorder in self.recorders:
            recorder.flush()
        self.log.close()
