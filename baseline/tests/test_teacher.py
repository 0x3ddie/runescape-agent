"""The teacher's labels define correct behavior for imitation, so check its decisions."""
from act_contract import CLICK_OFFSETS, CLICK_ROWS
from baseline.scripted import TRIP_TICKS, Teacher

EMPTY = dict(slot=-1, id=-1, count=0, name='')
BOOTH = (110, 100)  # a bank booth in the 15-tile loc list
GUARD = dict(name='Jail guard', combatLevel=26, options=['Talk-to', 'Attack'], dx=1, dz=0, distance=1)
HERE = (100, 100)  # the player stands at (100, 100)
A_ACT0, A_HELD0, A_DEPOSIT0, A_WITHDRAW0 = 1, 29, 45, 53
CLOSE, CONTINUE, INV_NEXT, BANK_NEXT = 61, 62, 68, 70


def click(dx, dz):
    """The row of the act-v3 click at that offset."""
    return CLICK_ROWS[CLICK_OFFSETS.index((dx, dz))]


def item(slot, name, id=1511, count=1, options=()):
    return dict(slot=slot, id=id, count=count, name=name, options=list(options))


def act(name, option, x, z=100, kind='loc'):
    d = max(abs(x - 100), abs(z - 100))
    return dict(kind=kind, id=1, name=name, option=option, x=x, z=z, level=0, dx=x - 100, dz=z - 100, distance=d,
                steps=d, count=1, combatLevel=0)


def obs(inventory=(), equipment=(), bank=False, bank_items=(), bank_page=0, bank_pages=1, inv_page=0,
        trees=((101, 100),), booth=True, extra=(), banks=(BOOTH,), clicks=True, legal=None, npcs=(), in_combat=False,
        hp=20):
    """An act-v3 observation, its mask built as env-ts/core/mask.ts would: the booth among the affordances when in
    reach (`booth`), the banks in the loc list, the 24 clicks (legal unless `clicks` is False)."""
    acts = [act('Tree', 'Chop down', x, z) for x, z in trees] + ([act('Bank booth', 'Use-quickly', 110)] if booth else [])
    acts = sorted(acts + list(extra), key=lambda a: (a['steps'], a['x'], a['z'], a['kind'], a['id'], a['option']))
    inventory = list(inventory)
    page = [next((i for i in inventory if i['slot'] == inv_page * 8 + k), EMPTY) for k in range(8)]
    held = [dict(slot=i['slot'], id=i['id'], name=i['name'], count=i['count'], option=o)
            for i in page if i['count'] for o in i.get('options', ())]
    bank_sel = list(bank_items) + [EMPTY] * (8 - len(bank_items))
    locs = [dict(name='Bank booth', options=['Use-quickly', 'Use'], dx=x - 100, dz=z - 100) for x, z in banks]
    slots = [dict(dx=dx, dz=dz, inside=True, open=True) for dx, dz in CLICK_OFFSETS]
    if legal is None:
        legal = {0}
        if bank:
            legal |= {CLOSE, *(A_DEPOSIT0 + k for k, i in enumerate(page) if i['count']),
                      *(A_WITHDRAW0 + k for k, i in enumerate(bank_sel) if i['count'])}
            legal |= {BANK_NEXT} if bank_page + 1 < bank_pages else set()
        else:
            legal |= {A_ACT0 + k for k in range(len(acts))} | {A_HELD0 + k for k in range(len(held))}
            legal |= set(CLICK_ROWS) if clicks else set()
        legal |= {INV_NEXT} if inv_page < 3 else set()
    mask = [int(a in legal) for a in range(96)]
    player = dict(x=100, z=100, combatLevel=23, inCombat=in_combat, hp=hp, maxHp=20)
    return dict(mask=mask, player=player, npcs=list(npcs), locs=locs, inventory=inventory, equipment=list(equipment),
                bankOpen=bank, invPage=inv_page, invSel=page, bankSel=bank_sel, bankPage=bank_page, bankPages=bank_pages,
                actSel=acts, heldSel=held, clicks=slots)


def act_row(o, name, x=None, option=None):
    return next(A_ACT0 + k for k, a in enumerate(o['actSel'])
                if a['name'] == name and x in (None, a['x']) and option in (None, a['option']))


AXE = item(0, 'Bronze axe', id=1351, options=['Wield'])
WORN = [dict(slot=3, id=1351, name='Bronze axe')]


def test_at_its_labelled_trees_it_chops_the_nearest_or_waits_for_regrowth():
    teacher = Teacher(dict(woodcutting=HERE))
    o = obs(equipment=WORN, inv_page=2)
    assert teacher.action(o) == act_row(o, 'Tree', 101)
    assert teacher.action(obs(equipment=WORN, trees=())) == 0


def test_away_from_its_labelled_trees_it_heads_there_through_what_the_policy_sees():
    # A visible tree nearer the label is chopped, not waited beside: waiting for a reason
    # the policy could not see taught imitation to wait beside choppable trees.
    o = obs(equipment=WORN, trees=((101, 100), (112, 100)))
    assert Teacher(dict(woodcutting=(130, 100))).action(o) == act_row(o, 'Tree', 112)
    # No target in sight: the click that closes most on the point (16 east), or the 9-tile one when it lies 10 away.
    o = obs(equipment=WORN, trees=())
    assert Teacher(dict(woodcutting=(130, 100))).action(o) == click(16, 0)
    assert Teacher(dict(woodcutting=(100, 110))).action(o) == click(0, 9)
    o = obs(equipment=WORN, trees=(), legal=(0, INV_NEXT))
    assert Teacher(dict(woodcutting=(130, 100))).action(o) == 0
    # Nothing it can click leads closer: it chops what is in reach instead of waiting.
    o = obs(equipment=WORN, trees=((101, 100),), clicks=False)
    assert Teacher(dict(woodcutting=(90, 100))).action(o) == act_row(o, 'Tree', 101)


def test_it_wields_a_carried_axe_once():
    assert Teacher(dict(woodcutting=HERE)).action(obs(inventory=[AXE])) == A_HELD0
    o = obs(inventory=[AXE], equipment=[dict(slot=3, id=1, name='Bronze sword')])
    assert Teacher(dict(woodcutting=HERE)).action(o) == act_row(o, 'Tree')


def test_without_an_axe_it_walks_to_a_booth_and_fetches_one_across_bank_pages():
    teacher = Teacher(dict(woodcutting=HERE))
    o = obs(booth=False)  # the booth 10 east is listed, not yet in reach
    assert teacher.action(o) == click(9, 0)
    o = obs()
    assert teacher.action(o) == act_row(o, 'Bank booth', option='Use-quickly')
    other = [item(k, 'Coins', id=995) for k in range(8)]
    assert teacher.action(obs(bank=True, bank_items=other, bank_pages=2)) == BANK_NEXT
    assert teacher.action(obs(bank=True, bank_items=[item(9, 'Bronze axe', 1351)], bank_page=1, bank_pages=2)) == A_WITHDRAW0


def test_it_banks_a_full_inventory_unless_a_trip_limit_says_too_little_time_remains():
    full = [AXE] + [item(k, 'Logs') for k in range(1, 28)]
    booth = act_row(obs(), 'Bank booth')
    # Logs count the same carried or banked: by default it banks with one tick left.
    assert Teacher(dict(woodcutting=HERE)).action(obs(inventory=full, equipment=WORN), remaining=1) == booth
    assert Teacher(dict(woodcutting=HERE), trip_ticks=200).action(obs(inventory=full, equipment=WORN), remaining=201) == booth
    assert Teacher(dict(woodcutting=HERE), trip_ticks=200).action(obs(inventory=full, equipment=WORN), remaining=200) == 0
    # No bank in sight: back towards the labelled trees, not a wait for the rest of the episode.
    far = dict(inventory=full, equipment=WORN, booth=False, banks=())
    assert Teacher(dict(woodcutting=(80, 100))).action(obs(**far), remaining=TRIP_TICKS + 1) == click(-16, 0)
    assert Teacher(dict(woodcutting=HERE)).action(obs(**far), remaining=TRIP_TICKS + 1) == 0


def test_at_half_hitpoints_it_rests_by_a_bank_instead_of_chopping():
    # Jail guards wore a teacher down from 20 hitpoints to 3 at their trees, then killed it.
    o = obs(equipment=WORN, hp=11)
    assert Teacher(dict(woodcutting=HERE)).action(o) == act_row(o, 'Tree')
    o = obs(equipment=WORN, hp=10)
    assert Teacher(dict(woodcutting=HERE)).action(o) == click(9, 0)  # the booth 10 tiles away
    assert Teacher(dict(woodcutting=HERE)).action(obs(equipment=WORN, hp=10, banks=((102, 100),))) == 0


def test_at_the_bank_it_deposits_everything_but_the_axe_then_closes():
    logs = [AXE, item(12, 'Logs')]
    assert Teacher(dict(woodcutting=HERE)).action(obs(inventory=logs, bank=True)) == INV_NEXT
    assert Teacher(dict(woodcutting=HERE)).action(obs(inventory=logs, bank=True, inv_page=1)) == A_DEPOSIT0 + 4
    assert Teacher(dict(woodcutting=HERE)).action(obs(inventory=[AXE], bank=True)) == CLOSE


def test_it_dismisses_a_level_up_that_blocks_a_running_chop():
    # A level-up opened mid-chop and blocked it; waiting on the busy option never ended.
    level_up = dict(obs(equipment=WORN, legal=(0, CONTINUE, CLOSE)), modalOpen=True, dialogOpen=True, optionBusy=True)
    assert Teacher(dict(woodcutting=HERE)).action(level_up) == CLOSE
    banking = dict(obs(equipment=WORN, legal=(0,)), modalOpen=True, optionBusy=True)
    assert Teacher(dict(woodcutting=HERE)).action(banking) == 0


def test_it_walks_away_from_a_stronger_attacker_and_chops_away_from_one():
    attacked = obs(equipment=WORN, npcs=[GUARD], in_combat=True)
    chop, away = act_row(attacked, 'Tree'), click(-16, 0)  # the click farthest from the guard a tile east
    assert Teacher(dict(woodcutting=HERE)).action(dict(attacked, optionBusy=True, optionAction=chop)) == away  # even mid-chop
    assert Teacher(dict(woodcutting=HERE)).action(dict(attacked, optionBusy=True, optionAction=away)) == 0  # keeps walking
    goblin = dict(GUARD, name='Goblin', combatLevel=2)
    assert Teacher(dict(woodcutting=HERE)).action(obs(equipment=WORN, npcs=[goblin], in_combat=True)) == chop
    # A weaker attacker too, once at half hitpoints, even while an option holds the player.
    hurt = obs(equipment=WORN, npcs=[goblin], in_combat=True, hp=10)
    assert Teacher(dict(woodcutting=HERE)).action(dict(hurt, optionBusy=True, optionAction=chop)) == away
    # Not attacked yet: skip the tree beside the guard for one 12 tiles away,
    o = obs(equipment=WORN, npcs=[GUARD], trees=((101, 100), (112, 100)))
    assert Teacher(dict(woodcutting=HERE)).action(o) == act_row(o, 'Tree', 112)
    # and when every tree is near it, chop the farthest rather than wait.
    o = obs(equipment=WORN, npcs=[GUARD], trees=((101, 100), (105, 100)))
    assert Teacher(dict(woodcutting=HERE)).action(o) == act_row(o, 'Tree', 105)


def test_the_tool_it_holds_picks_the_skill():
    """Draynor labels trees and a net spot: the net alone means fishing, an axe chopping."""
    spot = [act('Fishing spot', option, 102, kind='npc') for option in ('Net', 'Bait')]
    teacher = Teacher(dict(woodcutting=HERE, fishing=HERE))
    net = [item(0, 'Small fishing net', id=303)]
    fishing = obs(inventory=net, extra=spot)
    assert teacher.action(fishing) == act_row(fishing, 'Fishing spot', option='Net')
    cages = [act('Fishing spot', option, 102, kind='npc') for option in ('Cage', 'Harpoon')]
    assert teacher.action(obs(inventory=net, extra=cages)) == 0
    chopping = obs(inventory=net, equipment=WORN, extra=spot)
    assert teacher.action(chopping) == act_row(chopping, 'Tree')


def test_banking_keeps_every_gathering_tool():
    net, logs = item(1, 'Small fishing net', id=303), item(2, 'Logs')
    assert Teacher(dict(woodcutting=HERE)).action(obs(inventory=[AXE, net, logs], equipment=WORN, bank=True)) == A_DEPOSIT0 + 2
    assert Teacher(dict(woodcutting=HERE)).action(obs(inventory=[AXE, net], equipment=WORN, bank=True)) == CLOSE
