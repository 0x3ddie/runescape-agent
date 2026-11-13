"""Certified melee kills from the native combat capture (env-ts/engine/combat-capture.ts).

Source chain (pinned content, skill_combat): player_melee_attack adds its capped damage to the npc's
hero points and queues the damage (npc_queue 2); the npc applies it itself (npc_damage) and at zero
hitpoints runs its death script (npc_death, then npc_del); the chicken drop goes to npc_findhero, the
actor with the most hero points. The capture tags each queued damage with the actor that queued it.

One credit per enemy life, counted in the shared window (arm to the resource terminal), when:
  - the life is of a declared enemy type (resource_task.MELEE_ENEMIES);
  - at this agent's first hit on it, it had its base hitpoints and no other actor's hero points;
  - every hit, damage and hitpoint loss on it up to its defeat came from this agent's hits;
  - this agent's damage took it to zero hitpoints (the defeat), and this agent then held more hero
    points than any other actor, so npc_findhero names it.
The upper bound counts every enemy life defeated in the window that this agent damaged. Damage alone,
items (drops, pickups, transfers) and XP never count. A death of the agent settles every goal to zero
in goal_return; missing or inconsistent evidence raises, it never scores zero.
"""
from baseline.evidence import EvidenceError, integer
from baseline.native_trace import require

SCHEMA = 'native-combat-capture-v1'
ACTORS = ('agent', 'player', 'npc', 'world')


def heroes(value):
    require(isinstance(value, dict) and set(value) == {'agent', 'other'}, 'invalid hero points')
    return integer(value['agent'], 'agent hero points'), integer(value['other'], 'other hero points')


def audit_native_combat(trace, enemies):
    combat = trace.get('combat')
    require(isinstance(combat, dict) and combat.get('schema') == SCHEMA, 'missing native combat capture')
    require(combat.get('failure') is None and combat.get('open') is False, 'invalid or still open combat capture')
    require(combat.get('identity') == trace.get('identity') and combat.get('startTick') == trace.get('startTick')
            and combat.get('endTick') == trace.get('endTick'), 'combat and resource windows differ')
    require(combat.get('enemyTypes') == sorted(enemies), 'declared melee enemies differ')
    terminal = trace.get('terminal')
    require(isinstance(terminal, dict), 'missing terminal')
    start, close = integer(trace.get('startTick'), 'start tick'), integer(terminal.get('tick'), 'terminal tick')
    integer(combat.get('agentHp'), 'agent hitpoints at arm')
    events = combat.get('events')
    require(isinstance(events, list), 'missing combat receipts')
    lives, hits, last_tick = {}, {}, start
    for sequence, event in enumerate(events, 1):
        require(isinstance(event, dict) and event.get('sequence') == sequence, 'combat receipt gap/duplicate')
        tick = integer(event.get('tick'), 'combat tick')
        require(last_tick <= tick <= close, 'combat clock outside the window')
        last_tick = tick
        kind = event.get('kind')
        if kind == 'agent-hp':
            integer(event.get('hpBefore'), 'agent hitpoints'); integer(event.get('hpAfter'), 'agent hitpoints')
            continue
        if kind == 'agent-hit':  # an attack on the agent (its planned damage, zero included): evidence, never a kill
            integer(event.get('damage'), 'planned damage on the agent'); integer(event.get('hp'), 'agent hitpoints')
            continue
        key = event.get('life')
        require(isinstance(key, str) and key, 'missing enemy life')
        if kind == 'life':
            require(key not in lives, 'enemy life recorded twice')
            nid, type_ = integer(event.get('nid'), 'npc id'), integer(event.get('type'), 'npc type')
            hp, base = integer(event.get('hp'), 'npc hitpoints'), integer(event.get('baseHp'), 'npc base hitpoints', 1)
            previous = event.get('previous')
            if previous is not None:
                require(previous in lives and lives[previous]['removed'] and lives[previous]['nid'] == nid
                        and hp == base, 'respawn without a removed life')
            lives[key] = dict(nid=nid, type=type_, hp=hp, base=base, removed=False, first=None, foreign=False,
                              damaged=False, defeat=None)
            continue
        require(key in lives and not lives[key]['removed'], 'receipt for an unknown or removed life')
        life = lives[key]
        open_life = life['defeat'] is None
        if kind == 'hit':
            hitter = event.get('hitter')
            require(hitter in ACTORS, 'unknown hitter')
            damage = integer(event.get('damage'), 'queued damage'); integer(event.get('delay'), 'queue delay')
            require(event.get('hp') == life['hp'] and event.get('baseHp') == life['base'], 'hit hitpoints disagree')
            _, other = heroes(event.get('heroes'))
            hits[sequence] = dict(life=key, hitter=hitter, damage=damage, used=False)
            if hitter == 'agent' and life['first'] is None and open_life:
                life['first'] = life['hp'] == life['base'] and other == 0 and not life['foreign']
            elif hitter != 'agent' and open_life:
                life['foreign'] = True
        elif kind == 'damage':
            hitter = event.get('hitter')
            require(hitter in ACTORS, 'unknown hitter')
            amount = integer(event.get('amount'), 'damage'); integer(event.get('hitType'), 'hitmark')
            before, after = event.get('hpBefore'), event.get('hpAfter')
            require(before == life['hp'] and after == max(0, before - amount), 'damage hitpoints disagree')
            if event.get('hit') is not None:
                hit = hits.get(event['hit'])
                require(hit is not None and not hit['used'] and hit['life'] == key and hit['hitter'] == hitter
                        and hit['damage'] == amount, 'damage does not match its queued hit')
                hit['used'] = True
            if open_life:
                life['foreign'] |= hitter != 'agent'
                life['damaged'] |= hitter == 'agent' and after < before
                if before > 0 and after == 0:
                    agent, other = heroes(event.get('heroes'))
                    life['defeat'] = dict(tick=tick, hitter=hitter, hero=agent > other)
            life['hp'] = after
        elif kind == 'hp':
            before, after = event.get('hpBefore'), event.get('hpAfter')
            require(before == life['hp'] and type(after) is int and 0 <= after != before, 'hitpoint change disagrees')
            if open_life and after < before:
                life['foreign'] = True  # a loss no hit explains: not this agent's damage
            life['hp'] = after
        elif kind == 'removed':
            life['removed'] = True
        else:
            raise EvidenceError('unsupported combat receipt kind')
    enemy = [life for life in lives.values() if life['type'] in enemies]
    credited = [life for life in enemy if life['first'] and life['defeat'] and life['defeat']['hitter'] == 'agent'
                and life['defeat']['hero'] and not life['foreign']]
    upper = [life for life in enemy if life['defeat'] and life['damaged']]
    return dict(lower=len(credited), upper=len(upper), kill_ticks=sorted(life['defeat']['tick'] for life in credited),
                lives=len(lives), enemy_lives_hit=sum(life['first'] is not None for life in enemy),
                other_defeats=sum(1 for life in lives.values() if life['type'] not in enemies and life['defeat']
                                  and life['defeat']['hitter'] == 'agent'))

