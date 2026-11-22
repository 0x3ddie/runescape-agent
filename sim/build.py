"""Build the simulator's static Draynor tables from the engine's own code (sim/extract.ts).

The extraction runs in a private copy: vendor/rs-sdk is cloned (APFS copy-on-write) with
env-ts/ and sim/extract.ts under logs/sim/work, so the shared vendor checkout and any
running world are never touched. It loads configs and the map as World.start() does but
starts no tick, player or port. The result is Jagex-derived and stays in ignored logs/:

    .venv-sim/bin/python -m sim.build            # writes logs/sim/data/draynor-v3.npz
    .venv-sim/bin/python -m sim.build --content-only --source logs/sim/data/draynor-nc1.npz   # the same from nc1's tables

The archive pins what it was made from (vendor commit, patch and cache hashes, engine
settings); sim/data.py refuses an archive whose pins differ from the checkout. How each
archive in logs/sim/data was made (its recorded sim/extract.ts pin is that commit's file):

- draynor-v1-51c14db.npz, the first simulator's, kept under this name: an earlier
  sim/extract.ts (15 October 2026) wrote it as draynor-v1.npz with the
  schema draynor-sim-tables-v1 (the other four have v3).
- draynor-v1.npz, phase 1's: `python -m sim.build` with the sim/extract.ts of 15 October
  2026, which stayed until 3 November.
- draynor-nc1.npz (DATA), act-v3's, with no compass-move table: `python -m sim.build` as
  of 3 November 2026, when sim.build wrote this name. Recorded runs
  and the default pack name this file, so nothing here writes it (`_written`, which knows
  the file only below the logs/ of the checkout that runs, a folder or a link).
- draynor-v2.npz, an intermediate archive of the native-gap diagnosis, not used by this tree.
- draynor-v3.npz (DATA_V3): the commands above. The first
  one (9 November 2026) is the second command's: draynor-nc1.npz's extracted tables with
  the content read again, which adds the firemaking and cooking rows.
- draynor-nc2.npz (DATA_NC2; village-world-v2, sim/combat.py VILLAGE_V2): draynor-nc1.npz
  as it is, tables and content, with only the field plants' pick rows and texts added
  (`add_picks`; no firemaking or cooking rows, which the C export refuses):

    .venv-sim/bin/python -m sim.build --picks --source logs/sim/data/draynor-nc1.npz --output logs/sim/data/draynor-nc2.npz
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from sim import rs2

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'logs' / 'sim' / 'work'
DATA = ROOT / 'logs' / 'sim' / 'data' / 'draynor-nc1.npz'  # native-click's: no compass moves
DATA_V3 = DATA.with_name('draynor-v3.npz')  # DATA's tables with the firemaking and cooking content
DATA_NC2 = DATA.with_name('draynor-nc2.npz')  # DATA with the field plants' pick content (village-world-v2)
# The settings the native training and evaluation worlds run with (scripts/stack.py).
ENGINE_ENV = dict(NODE_XPRATE='25', RL_NPC_RADIUS='64')
BINARIES = dict(collision=('collision.i32', np.int32), parents=('parents.u8', np.uint8), orders=('orders.u16', np.uint16),
                dists=('dists.u8', np.uint8), walks=('walks.i16', np.int16),
                entity_reach=('entity_reach.u16', np.uint16), los=('los.u8', np.uint8), snapshot=('snapshot.u8', np.uint8),
                sides=('sides.u8', np.uint8))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def pins():
    """What the tables depend on: engine source, patches, packed cache and maps, and settings."""
    vendor = ROOT / 'vendor' / 'rs-sdk'
    pack = vendor / 'server' / 'engine' / 'data' / 'pack'
    commit = subprocess.run(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    files = [pack / '.cache' / 'maps-server.zip', *sorted((pack / 'server').glob('*.dat')), *sorted((pack / 'server').glob('*.idx'))]
    return dict(vendor_commit=commit,
                patches={p.name: sha256(p) for p in sorted((ROOT / 'vendor-patches').glob('*.patch'))},
                pack={str(p.relative_to(pack)): sha256(p) for p in files if p.exists()},
                inputs={name: sha256(ROOT / name) for name in ('sim/extract.ts', 'env-ts/reach.ts', 'env-ts/core/affordances.ts')},
                content=content_digest(),
                engine_env=ENGINE_ENV)


def content_digest():
    """One digest over the content scripts, maps and category pack the tables are read from."""
    digest = hashlib.sha256()
    files = [*sorted((CONTENT / 'scripts').rglob('*')), *sorted((CONTENT / 'maps').rglob('*')), CONTENT / 'pack' / 'category.pack']
    for path in files:
        if path.is_file():
            digest.update(str(path.relative_to(CONTENT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


CONTENT = ROOT / 'vendor' / 'rs-sdk' / 'server' / 'content'
# Game text the slice's content scripts print, located by file, script label, call and position
# among that call's string literals and read from the vendor content when an archive is built. Some
# engine and interface lines (for example "I can't reach that!") are quoted in the code itself.
TEXTS = dict(
    wc_level=('skill_woodcutting/scripts/woodcut.rs2', '[label,attempt_cut_tree]', 'mes', 0),
    wc_full=('skill_woodcutting/scripts/woodcut.rs2', '[label,attempt_cut_tree]', 'mes', 1),
    wc_swing=('skill_woodcutting/scripts/woodcut.rs2', '[label,attempt_cut_tree]', 'mes', 2),
    wc_get=('skill_woodcutting/scripts/woodcut.rs2', '[label,get_logs]', 'mes', 0),
    wc_noaxe=('skill_woodcutting/scripts/woodcut.rs2', '[proc,woodcutting_axe_checker]', 'mes', 0),
    fish_full=('skill_fishing/scripts/fishing_spots/saltfish.rs2', '[label,attempt_fish_shrimp]', '~mesbox', 0),
    fish_cast=('skill_fishing/scripts/fishing_spots/saltfish.rs2', '[label,attempt_fish_shrimp]', 'mes', 0),
    fish_level5=('skill_fishing/scripts/fishing_spots/saltfish.rs2', '[label,attempt_fish_sardine]', '~mesbox', 0),
    fish_no_bait=('skill_fishing/scripts/fishing.rs2', '[proc,check_fish_equipment]', '~mesbox', 0),
    select_option=('interface_chat/scripts/chat.rs2', '[proc,p_choice2]', 'if_settext', 0),
    bank_no_space=('interface_bank/scripts/bank.rs2', '[proc,bank_withdraw_request]', 'mes', 1),
    bank_carry_all=('interface_bank/scripts/bank.rs2', '[proc,bank_withdraw_request]', 'mes', 2),
    bank_withdraw_many=('interface_bank/scripts/bank.rs2', '[proc,bank_withdraw_request]', 'mes', 3),
    bank_full_stack=('interface_bank/scripts/bank.rs2', '[proc,bank_deposit_request]', 'mes', 0),
    bank_full=('interface_bank/scripts/bank.rs2', '[proc,bank_deposit_request]', 'mes', 1),
    under_attack=('skill_combat/scripts/player/player_combat.rs2', '[proc,player_in_combat_check]', 'mes', 0),
    fighting_other=('skill_combat/scripts/player/player_combat.rs2', '[proc,player_in_combat_check]', 'mes', 2),
    weakened=('skill_combat/scripts/npc/npc_combat_magic.rs2', '[proc,npc_stat_change_effect]', 'mes', 0),
    pickup_no_space=('player/scripts/pickup.rs2', '[proc,pickup_obj_check_for_space]', 'mes', 0),
    mine_none=('skill_mining/scripts/mining.rs2', '[label,mining_firstswing]', 'mes', 1),
    mine_swing=('skill_mining/scripts/mining.rs2', '[label,mining_firstswing]', 'mes', 2),
    mine_full=('skill_mining/scripts/mining.rs2', '[label,mining_firstswing]', '~mesbox', 0),
    mine_level=('skill_mining/scripts/mining.rs2', '[label,mining_firstswing]', '~mesbox', 1),
    mine_nopick=('skill_mining/scripts/mining.rs2', '[label,mining_firstswing]', '~mesbox', 2),
    mine_gem=('skill_mining/scripts/mining.rs2', '[label,get_ore_normal]', 'mes', 0),
    mine_get=('skill_mining/scripts/mining.rs2', '[label,get_ore_normal]', 'mes', 1),
    prospect_start=('skill_mining/scripts/prospect.rs2', '[label,prospect_rock]', 'mes', 0),
    prospect_empty=('skill_mining/scripts/prospect.rs2', '[label,finish_prospect_normal]', 'mes', 0),
    prospect_ore=('skill_mining/scripts/prospect.rs2', '[label,finish_prospect_normal]', 'mes', 1),
    equip_space=('player/scripts/equip.rs2', '[proc,equip]', 'mes', 2),
    eat=('player/scripts/consumption/effects/scripts/consume_effects.rs2', '[proc,consume_effect_messages]', 'mes', 5),
    eat_heal=('player/scripts/consumption/effects/scripts/consume_effects.rs2', '[proc,consume_effect_messages]', 'mes', 6),
    bury_dig=('skill_prayer/scripts/bury_bone.rs2', '[label,bury_bones]', 'mes', 0),
    bury_done=('skill_prayer/scripts/bury_bone.rs2', '[label,bury_bones]', 'mes', 1),
    fm_level=('skill_firemaking/scripts/firemaking.rs2', '[label,light_logs_inv]', 'mes', 0),
    fm_here=('skill_firemaking/scripts/firemaking.rs2', '[label,light_logs_inv]', 'mes', 1),
    fm_tinderbox=('skill_firemaking/scripts/firemaking.rs2', '[label,light_logs_inv]', 'mes', 2),
    fm_attempt=('skill_firemaking/scripts/firemaking.rs2', '[label,light_logs_inv]', 'mes', 3),
    fm_catch=('skill_firemaking/scripts/firemaking.rs2', '[proc,firemaking_success]', 'mes', 0),
    cook_cant=('skill_cooking/scripts/cooking.rs2', '[label,attempt_cook_item]', '~mesbox', 2),
)
# Script expressions inside the texts above, as the simulator's fields.
FIELDS = ((r'<tostring\(\$\w+\)>', '{level}'), (r'<lowercase\(oc_name\(\$product\)\)>', '{product}'), (r'<oc_name\(\$log\)>', '{log}'),
          (r'<db_getfield\(\$data, mining_table:ore_name, 0\)>', '{ore}'), (r'<lowercase\(oc_name\(\$consumable\)\)>', '{food}'),
          (r'<~add_article\(oc_name\(db_getfield\(\$data, gem_cutting_table:cut_gem, 0\)\)\)>', '{gem}'))
BANKER = ('interface_bank/scripts/banker.rs2', '[label,talk_to_banker]')
# Climb up and climb down: p_arrivedelay, then p_telejump by switch_coord(loc_coord).
STAIRS = ('ladders+stairs/scripts/stairs.rs2', ('[oploc1,loc_1722]', '[oploc1,loc_1723]'))
# Death scripts call this; the simulator runs it itself (sim/engine.py npc_death).
NATIVE_PROCS = {'[proc,npc_death]'}
# The field plants of village-world-v2 (sim/world.py VILLAGE_WORLD field_kinds): each one's [oploc2,...] script, the proc
# it calls and the player-count scale, read for sim/engine.py pick, whose port holds only while each keeps this shape.
PICKABLES = ('general_use/scripts/pickables.rs2', ('cabbage', 'potato', 'onion'))
PICK_SHAPE = ('p_arrivedelay;', 'if(inv_freespace(inv) = 0) {', 'mes', 'p_teleport(loc_coord);', 'p_delay(0);', '} else {',
              '~pickup_loc_floor', 'mes', '}')
PICKUP_LOC_FLOOR = ('player/scripts/pickup.rs2', '[proc,pickup_loc_floor]',
                    ('anim(human_pickupfloor, 0);', 'p_teleport($coord);', 'p_delay(0);', 'sound_synth(pick, 0, 0);',
                     'if ($worldpop_dependant = true) {', '$respawn_ticks = ~scale_by_playercount($respawn_ticks);', '}',
                     'loc_del($respawn_ticks);', 'inv_add(inv, $replacement, 1);',
                     'wealth_event(^wealth_pickup, oc_debugname($replacement), 1, oc_cost($replacement));'))
SCALE_BY_PLAYERCOUNT = ('general/scripts/player_count.rs2', '[proc,scale_by_playercount]',
                        ('def_int $playercount = min(playercount, 2000);', 'return (scale(sub(4000, $playercount), 4000, $base));'))
# The raw food the compact world produces, and the cooking_generic columns its rows may
# hold: those attempt_cook_item reads for them, and the castle range's own chance, which the port never reaches.
RAW_FOOD = ('raw_shrimp', 'raw_anchovies', 'raw_chicken')
COOK_COLUMNS = {'table', 'uncooked', 'cooked', 'burnt', 'levelrequired', 'experience', 'successchance', 'successchance_cookomatic',
                'cookmessage_fire', 'cookmessage_range', 'successmessage', 'burnmessage'}


def label_body(file, label):
    """The lines of one RuneScript script, from its header to the next header."""
    lines = (CONTENT / 'scripts' / file).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(label))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('[')), len(lines))
    return '\n'.join(line for line in lines[start + 1:end] if not line.lstrip().startswith('//'))


def call_strings(body, call):
    """String literals of each `call(...)` in order, one list per call."""
    found = []
    for match in re.finditer(r'(?<![\w~.])' + re.escape(call) + r'\(', body):
        depth, i = 1, match.end()
        while depth and i < len(body):
            depth += {'(': 1, ')': -1}.get(body[i], 0) if body[i] != '"' else 0
            if body[i] == '"':
                i = body.index('"', i + 1)
            i += 1
        found.append(re.findall(r'"([^"]*)"', body[match.end():i - 1]))
    return [s for s in found if s]


def coord(text):
    """A RuneScript coord literal (level_mx_mz_lx_lz) as [x, z, level]."""
    level, mx, mz, lx, lz = (int(v) for v in text.split('_'))
    return [(mx << 6) + lx, (mz << 6) + lz, level]


def stairs():
    """The stair scripts' literal cases: script -> [[loc x, z, level, destination x, z, level]].
    A case that computes its destination (movecoord) is left out, so it stays unhandled."""
    table = {}
    for label in STAIRS[1]:
        body = label_body(STAIRS[0], label)
        statements = [line.strip() for line in body.splitlines() if line.strip()]
        if statements[:2] != ['p_arrivedelay;', 'switch_coord (loc_coord) {']:
            raise ValueError(f'{label} changed shape')
        cases = re.findall(r'case (\d+_\d+_\d+_\d+_\d+) : p_telejump\((\d+_\d+_\d+_\d+_\d+)\);', body)
        table[label] = [coord(a) + coord(b) for a, b in cases]
    return table


def struct_params(file):
    structs, name = {}, None
    for line in (CONTENT / 'scripts' / file).read_text().splitlines():
        if line.startswith('['):
            name = line.strip('[]')
            structs[name] = {}
        elif line.startswith('param=') and name:
            key, _, value = line[6:].partition(',')
            structs[name][key] = int(value) if re.fullmatch(r'-?\d+', value) else value
    return structs


def dbrows(file):
    """Rows of a .dbrow file: name -> {column: [values per data line]}, and its table under 'table'."""
    rows, row = {}, None
    for line in (CONTENT / 'scripts' / file).read_text().splitlines():
        if line.startswith('['):
            row = rows.setdefault(line.strip('[]'), {})
        elif line.startswith('table=') and row is not None:
            row['table'] = line[6:].strip()
        elif line.startswith('data=') and row is not None:
            key, _, value = line[5:].partition(',')
            row.setdefault(key, []).append([int(v) if re.fullmatch(r'-?\d+', v) else v for v in value.split(',')])
    return rows


def combat(meta):
    """Unarmed styles, the area's npc spells, and the death scripts of the area's npcs."""
    consts = rs2.constants(CONTENT / 'scripts')
    resolve = lambda v: consts[v[1:]] if isinstance(v, str) and v.startswith('^') else v
    unarmed = dbrows('skill_combat/configs/combat.dbrow')['weapon_unarmed_table']
    styles = dict(damagestyle=[resolve(v[0]) for v in unarmed['damagestyle']],
                  damagetype=[resolve(v[0]) for v in unarmed['damagetype']])
    spells = {}
    for row in dbrows('skill_combat/configs/magic/magic_combat_spells.dbrow').values():
        spell = resolve(row['spell'][0][0])
        spells[spell] = dict(maxhit=row['maxhit'][0][0] if 'maxhit' in row else None,
                             proj=row['spotanim_proj'][0][1:] if 'spotanim_proj' in row else None,
                             stat_change=row.get('stat_change', []), freeze_time=row.get('freeze_time', [[0]])[0][0])
    roots = sorted({t['triggers']['AI_QUEUE3'] for t in meta['npc_types'].values() if t['triggers'].get('AI_QUEUE3')})
    scripts = rs2.parse_scripts(rs2.script_index(CONTENT / 'scripts'), roots, NATIVE_PROCS)
    used = {n[1] for s in scripts.values() for n in _nodes(s['body']) if n[0] == 'const'}
    names = {name: consts[name] for name in ('water_strike', 'confuse', 'weaken', 'earth_strike', 'wind_strike')}
    return dict(styles=styles, spells=spells, scripts=scripts, constants={k: consts[k] for k in used if k in consts},
                spell_ids=names)


def _nodes(node):
    if isinstance(node, list):
        if node and isinstance(node[0], str):
            yield node
        for child in node:
            yield from _nodes(child)


def content(meta):
    """The slice's game data from the content scripts: texts, tree and fish rows, level-ups,
    combat styles, spells and npc death scripts."""
    texts = {}
    for key, (file, label, call, n) in TEXTS.items():
        text = call_strings(label_body(file, label), call)[n][0]
        for pattern, field in FIELDS:
            text = re.sub(pattern, field, text)
        texts[key] = text
    enum = (CONTENT / 'scripts' / 'general' / 'configs' / 'displaymessage.enum').read_text()
    texts['dm_default'] = re.search(r'val=\^dm_default,(.*)', enum).group(1)
    equipment = struct_params('skill_fishing/configs/fishing_equipment.struct')
    for name in ('net', 'fishing_rod'):
        texts[f'fish_fail_{name}'] = equipment[f'fish_equipment_{name}']['fish_equipment_failmessage']
    bait = struct_params('skill_fishing/configs/fishing.struct')
    fish = {name: bait[f'fishing_struct_{suffix}'] for name, suffix in (('raw_shrimp', 'shrimps'), ('raw_anchovies', 'anchovies'))}
    texts['fish_no_bait'] = texts['fish_no_bait'].replace('<struct_param($struct, baitmessage)>',
                                                          equipment['fish_equipment_fishing_rod']['baitmessage'])
    # woodcutting_trees rows, keyed by each tree loc's debugname
    trees, row = {}, None
    for line in (CONTENT / 'scripts' / 'skill_woodcutting' / 'configs' / 'trees.dbrow').read_text().splitlines():
        if line.startswith('['):
            row = dict(trees=[], successchance={})
        elif line.startswith('data=') and row is not None:
            key, _, value = line[5:].partition(',')
            if key == 'tree':
                row['trees'].append(value)
                for name in row['trees']:
                    trees[name] = row
            elif key == 'successchance':
                axe, low, high = value.split(',')
                row['successchance'][axe] = (int(low), int(high))
            else:
                row[key] = int(value) if re.fullmatch(r'-?\d+', value) else value
    trees = {name: {k: v for k, v in r.items() if k != 'trees'} for name, r in trees.items()}
    # level-ups: dbrow, unlock enums, unlock dialog texts
    levelup, row = {}, None
    for line in (CONTENT / 'scripts' / 'levelup' / 'configs' / 'levelup.dbrow').read_text().splitlines():
        if line.startswith('['):
            row = {}
        elif line.startswith('data=') and row is not None:
            key, _, value = line[5:].partition(',')
            row[key] = value.strip('"')
            if key == 'stat':
                levelup[value] = row
    unlocks, name = {}, None
    for line in (CONTENT / 'scripts' / 'levelup' / 'configs' / 'levelup_unlocks.enum').read_text().splitlines():
        if line.startswith('['):
            name = line.strip('[]')
            unlocks[name] = []
        elif line.startswith('val=') and name:
            unlocks[name].append(int(line[4:]))
    for stat, row in levelup.items():
        row['interface'] = row['interface']
        row['unlocks'] = unlocks.get(row.get('unlocks_levels'), [])
        texts_by_level = {}
        path = CONTENT / 'scripts' / 'levelup' / 'scripts' / f'levelup_unlocks_{stat}.rs2'
        if path.exists():
            for m in re.finditer(r'case (\d+) : ~(objbox|doubleobjbox|mesbox)\(([^\n]*)', path.read_text()):
                strings = re.findall(r'"([^"]*)"', m.group(3))
                if strings:
                    texts_by_level[m.group(1)] = (m.group(2), strings[0])
        row['unlock_texts'] = texts_by_level
    body = label_body(*BANKER)
    strip = lambda calls: [[re.sub(r'^<p,[a-z_]+>', '', s) for s in c] for c in calls]  # chat animation tags
    npc, player, choices = (strip(call_strings(body, c)) for c in ('~chatnpcrange', '~chatplayer', '~p_choice2'))
    banker = dict(greet=npc[0][0], branch=npc[1][0], answer_do=npc[2][0], answer_varrock=npc[3][0],
                  choice_bank=choices[0][0], choice_what=choices[0][1], choice_do=choices[1][0], choice_varrock=choices[1][1])
    if [p[0] for p in player] != [banker['choice_bank'], banker['choice_what'], banker['choice_do'], banker['choice_varrock']]:
        raise ValueError('banker dialog changed shape')
    categories = dict(line.split('=', 1)[::-1] for line in (CONTENT / 'pack' / 'category.pack').read_text().splitlines() if '=' in line)
    return dict(texts=texts, fish=fish, trees=trees, levelup=levelup, banker=banker, wrong_spot={},
                bank_teller_category=int(categories['bank_teller']), combat=combat(meta),
                categories={name: int(cid) for name, cid in categories.items()}, stairs=stairs(), compact=compact(),
                firemaking=firemaking(), cooking=cooking(meta))


def _table_rows(files, table):
    return [row for file in files for row in dbrows(file).values() if row.get('table') == table]


def zones(file, row):
    """A coord_pair_table row's boxes, each [x0, z0, level0, x1, z1, level1] with both corners inside (INZONE)."""
    return [coord(a) + coord(b) for a, b in dbrows(file)[row]['coord_pair']]


def firemaking():
    """What the light scripts read beside the type tables (sim/engine.py light_logs_inv): the bank zones, where no fire
    is lit, and the duel arena's fight zones, where a fire does not move the player (not ported)."""
    script = 'skill_firemaking/scripts/firemaking.rs2'
    if call_strings(label_body(script, '[label,light_logs_ground]'), 'mes') != call_strings(label_body(script, '[label,light_logs_inv]'), 'mes'):
        raise ValueError('the two light scripts print different texts')
    arena = 'minigames/game_duelarena/configs/duelarena.dbrow'
    return dict(bank_zones=zones('skill_firemaking/configs/bank_zones.dbrow', 'bank_zones'),
                duel_zones=zones(arena, 'duel_arena_fight_zones') + zones(arena, 'duel_arena_obstacle_fight_zones'))


def cooking(meta):
    """cooking_generic's rows for RAW_FOOD, each message as attempt_cook_item prints it for that food (the row's own, else
    the script's default with the cooked item's name); every raw item the table names (one with no row here is not
    ported); the items the script treats apart; and Tutorial Island's zones, where a fire runs another script."""
    objs = {t['debugname']: t for _, t in sorted(meta['obj_types'].items(), key=lambda kv: -int(kv[0]))}
    body = label_body('skill_cooking/scripts/cooking.rs2', '[label,attempt_cook_item]')
    defaults = re.findall(r'\$(succesmessage|burnmessage|cookmessage) = "([^"]*)";', body)
    if [name for name, _ in defaults] != ['succesmessage', 'burnmessage', 'cookmessage', 'cookmessage']:
        raise ValueError('attempt_cook_item changed shape')
    success, burn, stove, fire = (text for _, text in defaults)
    table = _table_rows(['skill_cooking/configs/cooking_source/cooking_generic.dbrow'], 'cooking_generic')
    rows = {}
    for row in table:
        raw, cooked = row['uncooked'][0][0], row['cooked'][0][0]
        if raw not in RAW_FOOD:
            continue

        def text(column, default):
            value = ','.join(str(v) for v in row[column][0]) if column in row else default
            return value.replace('<lowercase(oc_name($cooked_item))>', objs[cooked]['name'].lower())
        rows[raw] = dict(cooked=cooked, burnt=row['burnt'][0][0], level=row['levelrequired'][0][0], exp=row['experience'][0][0],
                         chance=row['successchance'][0], fire=text('cookmessage_fire', fire), range=text('cookmessage_range', stove),
                         success=text('successmessage', success), burn=text('burnmessage', burn))
        # what the port does not read: other columns, a message of "0" (none shown), the pairs 0,0 and 1,1 (no roll)
        # and a members item (refused in a free world)
        if set(row) - COOK_COLUMNS or '0' in rows[raw].values() or rows[raw]['chance'] in ([0, 0], [1, 1]) \
                or objs[raw]['members'] or objs[cooked]['members']:
            raise ValueError(f'the cooking row of {raw} holds what the port does not read')
    island = re.findall(r'inzone\((\d+_\d+_\d+_\d+_\d+), (\d+_\d+_\d+_\d+_\d+), \$coord\)',
                        label_body('tutorial/scripts/util.rs2', '[proc,in_tutorial_island]'))
    if set(rows) != set(RAW_FOOD) or len(island) != 2:
        raise ValueError('the cooking table or in_tutorial_island changed shape')
    return dict(rows=rows, uncooked=sorted({value[0] for row in table for value in row['uncooked']}),
                special=sorted(set(re.findall(r'\$uncooked_item = (\w+)', body))),
                tutorial_zones=[coord(a) + coord(b) for a, b in island])


def compact():
    """The compact world's activities beyond the slice: mining rows, pickaxes and gems, food, weapon
    styles and wield requirements (sim/engine.py ports the scripts that read them)."""
    consts = rs2.constants(CONTENT / 'scripts')
    resolve = lambda v: consts[v[1:]] if isinstance(v, str) and v.startswith('^') else v
    one = lambda row, key, default=None: row[key][0][0] if key in row else default
    rocks = {}
    for row in _table_rows(['skill_mining/configs/mine.dbrow'], 'mining_table'):
        if 'rock_output' in row:
            data = dict(ore=one(row, 'ore_name'), output=one(row, 'rock_output'), level=one(row, 'rock_level', 0),
                        exp=one(row, 'rock_exp', 0), chance=row.get('rock_successchance', [[0, 0]])[0],
                        respawn=one(row, 'rock_respawnrate', 0))
            rocks.update({rock: data for (rock,) in row['rock']})
    checker = label_body('skill_mining/scripts/pickaxe_checker.rs2', '[proc,pickaxe_checker]')
    gems = label_body('skill_mining/scripts/mining.rs2', '[proc,mining_gem_table]')
    normal = label_body('skill_mining/scripts/mining.rs2', '[label,get_ore_normal]')
    cut = {one(row, 'uncut_gem'): one(row, 'cut_gem')
           for row in _table_rows(['skill_crafting/configs/gem/gem.dbrow'], 'gem_cutting_table')}
    food = 'player/configs/consumption/'
    eat = {}
    for row in _table_rows([food + 'consume_normal.dbrow', food + 'consume_effects.dbrow'], 'consume_table'):
        effect = dict(stat_heal=row.get('stat_heal', []), stat_change=row.get('stat_change', []), healenergy='healenergy' in row)
        eat.update({item: effect for (item,) in row['consumable']})
    messages, items, texts = {}, [], {}  # message values are whole lines (they may hold commas)
    for line in (CONTENT / 'scripts' / food / 'consume_messages.dbrow').read_text().splitlines() + ['[']:
        if line.startswith('['):
            messages.update({item: texts for item in items})
            items, texts = [], {}
        elif line.startswith('data=consumable,'):
            items.append(line.split(',', 1)[1])
        elif line.startswith('data='):
            key, _, value = line[5:].partition(',')
            texts[key] = int(value) if key == 'message_delay' else value
    consume = (CONTENT / 'scripts' / 'player' / 'scripts' / 'consumption' / 'consume.rs2').read_text()
    eat_triggers = {f'[{m[0]}]': [m[1], None if m[2] == 'null' else int(m[2]), None if m[3] == 'null' else int(m[3])]
                    for m in re.findall(r'^\[(opheld\d,[^\]]+)\] @player_consume_item\((\w+), (\w+), (\w+)\);', consume, re.M)}
    styles = {name: dict(damagestyle=[resolve(v[0]) for v in row['damagestyle']], damagetype=[resolve(v[0]) for v in row['damagetype']])
              for name, row in dbrows('skill_combat/configs/combat.dbrow').items() if row.get('table') == 'combat_style_table'}
    switch = label_body('skill_combat/scripts/combat.rs2', '[proc,combat_get_weapon_style_data]')
    style_of = {cat.strip(): table for cats, table in re.findall(r'case ([\w, ]+) : return\((\w+)\);', switch)
                for cat in cats.split(',') if cat.strip() != 'default'}
    wield = {}
    for path in [*sorted((CONTENT / 'scripts' / 'levelrequire' / 'scripts').glob('*.rs2'))]:
        for script, label, args in re.findall(r'^\[(opheld2,\w+)\] @(\w+)\(([^)]*)\);', path.read_text(), re.M):
            wield[f'[{script}]'] = [label, [int(a) if re.fullmatch(r'\d+', a.strip()) else a.strip() for a in args.split(',')]]
    tutorial = label_body('tutorial/scripts/tutorial.rs2', '[label,tutorial_island_equip]')
    return dict(rocks=rocks, pickaxes=re.findall(r'oc_param\((\w+), levelrequire\)', checker),
                gem_chance=int(re.search(r'def_int \$chance = (\d+);', normal).group(1)),
                gem_roll=int(re.search(r'random\((\d+)\)', gems).group(1)),
                gems=[[int(n), g] for n, g in re.findall(r'if \(\$rand < (\d+)\) return \((\w+)\);', gems)], cut_gems=cut,
                eat=eat, eat_messages=messages, eat_triggers=eat_triggers, styles=styles, style_of=style_of, wield=wield,
                tutorial_equip=consts[re.search(r'%tutorial < \^(\w+)', tutorial).group(1)],
                tutorial_complete=consts['tutorial_complete'], lootdrop_duration=consts['lootdrop_duration'])


def picks():
    """(rows, texts) of the field plants (PICKABLES): per plant loc debugname, the item ~pickup_loc_floor adds, its
    respawn ticks and whether ~scale_by_playercount scales them; per plant the texts pick_full_<name> and pick_<name>, as
    sim/engine.py pick prints them. Refused unless each script, the proc and the scale keep the shape the port follows."""
    statements = lambda body: tuple(line.strip() for line in body.splitlines() if line.strip())
    for file, label, shape in (PICKUP_LOC_FLOOR, SCALE_BY_PLAYERCOUNT):
        if statements(label_body(file, label)) != shape:
            raise ValueError(f'{label} changed shape')
    file, names = PICKABLES
    rows, texts = {}, {}
    for name in names:
        label = f'[oploc2,{name}]'
        body = label_body(file, label)
        lines = statements(body)
        call = re.fullmatch(r'~pickup_loc_floor\(loc_coord, (\w+), (\d+), (true|false)\);', lines[6]) if len(lines) == 9 else None
        if call is None or any(not line.startswith(want) for line, want in zip(lines, PICK_SHAPE)):
            raise ValueError(f'{label} changed shape')
        (full,), (picked,) = call_strings(body, 'mes')
        rows[name] = dict(item=call.group(1), respawn=int(call.group(2)), scaled=call.group(3) == 'true')
        texts[f'pick_full_{name}'], texts[f'pick_{name}'] = full, picked
    return rows, texts


def add_picks(output=DATA_NC2, source=DATA):
    """Write `source`'s tables and content to `output` with the pick rows and texts added (picks()) and the source named
    in meta['derived']. A new file only: an existing archive is never written over. Refused when `source`'s pins differ
    from the checkout's or it holds rows of another archive script (firemaking, cooking), which the C export refuses."""
    from sim.engine import ARCHIVE_SCRIPTS
    output, source = _written(output), Path(source)
    if output.exists():
        raise ValueError(f'{output} exists: an archive is written once')
    with np.load(source) as data:
        arrays = {k: data[k] for k in data.files if k != 'meta'}
        meta = json.loads(data['meta'].tobytes())
    if meta['pins'] != pins():
        raise ValueError(f'{source} was built from other inputs than this checkout\'s')
    other = sorted(set(ARCHIVE_SCRIPTS) & set(meta['content']))
    if other:
        raise ValueError(f'{source} holds the {" and ".join(other)} rows')
    rows, texts = picks()
    if set(texts) & set(meta['content']['texts']):
        raise ValueError(f'{source} already holds pick texts')
    meta['content']['picks'] = rows
    meta['content']['texts'].update(texts)
    meta['derived'] = dict(source=source.name, sha256=sha256(source), added=['content.picks', *(f'content.texts.{k}' for k in sorted(texts))])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
    print(f'[build] wrote {output}: {source.name} with the pick rows of {sorted(rows)}')
    return output


def _written(output):
    """The file np.savez writes for `output` (it adds a missing .npz), refused when that is DATA: the same file, through a
    link too, or the same resolved name whatever its letter case, since a Mac volume takes such names for one file.
    DATA is this checkout's: another checkout with no logs/ entry does not know the canonical checkout's archive."""
    path = Path(output)
    path = path if str(path).endswith('.npz') else path.with_name(path.name + '.npz')
    recorded = DATA.resolve()
    if (path.exists() and recorded.exists() and os.path.samefile(path, recorded)) \
            or str(path.resolve()).casefold() == str(recorded).casefold():
        raise ValueError(f'{path} is the archive recorded runs name; write another file')
    return path


def build(output=DATA_V3, bun=None, timeout=3600):
    output = _written(output)
    bun = bun or shutil.which('bun') or str(Path.home() / '.bun' / 'bin' / 'bun')
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / 'vendor').mkdir(parents=True)
    # -c: clonefile, so the 900 MB copy shares blocks with the original until written.
    subprocess.run(['cp', '-cR', str((ROOT / 'vendor' / 'rs-sdk').resolve()), str(WORK / 'vendor' / 'rs-sdk')], check=True)
    shutil.copytree(ROOT / 'env-ts', WORK / 'env-ts')
    (WORK / 'sim').mkdir()
    shutil.copy2(ROOT / 'sim' / 'extract.ts', WORK / 'sim' / 'extract.ts')
    raw = WORK / 'tables'
    # The stairs' destinations, from which extract.ts finds the upper floors' supported tiles.
    (WORK / 'stairs.json').write_text(json.dumps(stairs()))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NODE_', 'RL_'))} | ENGINE_ENV
    subprocess.run([bun, 'run', str(WORK / 'sim' / 'extract.ts'), str(raw), str(WORK / 'stairs.json')],
                   cwd=WORK / 'vendor' / 'rs-sdk' / 'server' / 'engine', env=env, check=True, timeout=timeout)
    meta = json.loads((raw / 'tables.json').read_text())
    if meta['self_check']['mismatched']:
        raise RuntimeError(f"extraction self-check failed: {meta['self_check']}")
    arrays = {name: np.fromfile(raw / file, dtype=dtype) for name, (file, dtype) in BINARIES.items()}
    meta['content'] = content(meta)
    meta['pins'] = pins()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
    print(f"[build] wrote {output} ({output.stat().st_size / 1e6:.0f} MB): {len(meta['sources'])} supported tiles, "
          f"self-check {meta['self_check']}")
    return output


def refresh_content(output=DATA_V3, source=None):
    """Re-read the content scripts into an archive: `output` itself, or `source`'s tables written to `output`. The bun
    tables are kept, so every pin of theirs must equal the checkout's (another sim/extract.ts, vendor commit, patch or
    pack needs a build)."""
    output = _written(output)
    with np.load(source or output) as data:
        arrays = {k: data[k] for k in data.files if k != 'meta'}
        meta = json.loads(data['meta'].tobytes())
    current = pins()
    if any(meta['pins'].get(k) != v for k, v in current.items() if k != 'content'):
        raise ValueError(f'{source or output} was extracted from other inputs, so its tables cannot be kept: '
                         f'`python -m sim.build --output {output}` extracts them again from this checkout')
    meta['content'], meta['pins'] = content(meta), current
    np.savez(output, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
    print(f'[build] refreshed content in {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, help='the archive to write (DATA_NC2 with --picks, else DATA_V3)')
    parser.add_argument('--content-only', action='store_true', help='re-read game text and rows into the archive')
    parser.add_argument('--picks', action='store_true', help='--source with the field plants\' pick rows added (add_picks)')
    parser.add_argument('--source', type=Path, help='with --content-only or --picks: the archive whose extracted tables '
                        '--output takes')
    parser.add_argument('--bun')
    parser.add_argument('--timeout', type=int, default=3600, help='seconds the extraction may run')
    args = parser.parse_args()
    if args.picks:
        add_picks(args.output or DATA_NC2, args.source or DATA)
    elif args.content_only:
        refresh_content(args.output or DATA_V3, args.source)
    else:
        build(args.output or DATA_V3, args.bun, args.timeout)


if __name__ == '__main__':
    main()
