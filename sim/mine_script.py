"""A script of one skill, mining, as an actor of the real-engine watch (sim/watch.py, actor `script-mine`): what
mining pays in the real engine at the site rimmington_mine_west, on the same records as the policy at that site. A
check of feasibility: its actions are never labels and never training data, and no trainer, learner or distillation
module imports it (sim/tests/test_watch.py).

It reads the native observation (sim/watch.py WatchEpisode.raw: the act rows' list, the pack, the skills, the dialog
and option flags) and one privileged table, the mine's rocks (sim/watch.py MINE_ROCKS: tile, ore, loc type at boot,
level), and acts through the rows the mask allows. At each decision, the first that applies:

1. a dialog's continue when it is offered, else the close of an open dialog or modal (a level-up's message);
2. WAIT while an option runs (a walk to a rock and its mining are one option);
3. with a full pack (every slot held), the DROP of an ore (baseline/resource_task.py GOAL_FAMILIES mine) on the shown
   inventory page, else the page turn toward the nearest page that holds one;
4. the Mine row of the nearest live rock of the table whose ore its current Mining level allows: the first such row in
   the act rows' order (walking steps first). A rock is live while its loc type is its type at boot: a mined rock turns
   into an empty rock that keeps the name 'Rocks' and the option Mine;
5. else WAIT (every rock in reach mined out: they grow back).

It does nothing else: no fighting back, no eating, no walk of its own and no pickup. At rimmington_mine_west a rock is 3
tiles from the start, so it needs no walk to mine; whether an npc there attacks first is not checked here, and a death
shows in the monitors (deaths, killers). It plays item-v2's pages, as every control does.
"""
from act_contract import A_ACT0
from baseline.resource_task import GOAL_FAMILIES
from obs import N_INV
from sim.agent import A_CLOSE_IF, A_DIALOG_CONTINUE, A_DROP0, A_INV_NEXT, A_INV_PREV, A_WAIT, N_INV_SEL

MINE_SCRIPT = dict(version='mine-script-v1', actor='script-mine')
ORES = GOAL_FAMILIES['mine']


def mining_level(raw):
    """The current Mining level of a native observation (its skills; 1 if absent)."""
    return next((int(s.get('level', 1)) for s in raw.get('skills') or [] if s.get('name', '').lower() == 'mining'), 1)


def drop_ore(raw, mask):
    """At a full pack: the DROP of an ore on the shown page, else the page turn toward the nearest page with one (item-v2
    pages of N_INV_SEL slots), else WAIT."""
    for k, item in enumerate((raw.get('invSel') or [])[:N_INV_SEL]):
        if item.get('slot', -1) >= 0 and item.get('id') in ORES and mask[A_DROP0 + k]:
            return A_DROP0 + k
    page = raw.get('invPage', 0)
    pages = {i['slot'] // N_INV_SEL for i in raw.get('inventory') or [] if i.get('id') in ORES}
    if any(q > page for q in pages) and mask[A_INV_NEXT]:
        return A_INV_NEXT
    if any(q < page for q in pages) and mask[A_INV_PREV]:
        return A_INV_PREV
    return A_WAIT


def live_rock(affordance, level):
    """Whether an act row is the Mine of a live rock of the table whose ore `level` allows."""
    from sim.watch import MINE_ROCKS
    rock = MINE_ROCKS.get((affordance.get('x'), affordance.get('z')))
    return affordance.get('kind') == 'loc' and affordance.get('option') == 'Mine' and rock is not None \
        and affordance.get('id') == rock[1] and rock[2] <= level


class MineScript:
    """The mining script (module docstring): one actor of one world, deterministic (no draw)."""

    def reset(self, seed):
        pass

    def act(self, observation, episode):
        raw, mask = episode.raw, observation['mask']
        if mask[A_DIALOG_CONTINUE]:
            return A_DIALOG_CONTINUE
        if (raw.get('dialogOpen') or raw.get('modalOpen')) and mask[A_CLOSE_IF]:
            return A_CLOSE_IF
        if raw.get('optionBusy'):
            return A_WAIT
        if len({i['slot'] for i in raw.get('inventory') or []}) >= N_INV:
            return drop_ore(raw, mask)
        level = mining_level(raw)
        for k, affordance in enumerate(raw.get('actSel') or []):
            if mask[A_ACT0 + k] and live_rock(affordance, level):
                return A_ACT0 + k
        return A_WAIT
