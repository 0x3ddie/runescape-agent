import { expect, test } from 'bun:test';
import { installCombatCapture } from './combat-capture';

const HP = 3, DAMAGE_QUEUE = 118;
class FakeNpc {
    levels = [1, 1, 1, 3]; baseLevels = [1, 1, 1, 3]; queue: any[] = []; heroPoints: any[] = []; x = 5; z = 6; level = 0;
    constructor(public nid: number, public type: number) {}
    enqueueScript(queueId: number, delay = 0, arg = 0) { this.queue.push({ queueId, args: [], delay, lastInt: arg }); }
}
function fixture() {
    const control = { tick: 10, open: true };
    const agent = { name: 'agent', levels: [10] }, other = { name: 'other' };
    const handlers: Record<number, (state: any) => any> = {
        // npc_queue(2, damage, delay) as player_melee_attack issues it, the active npc being the target
        1: s => { s._activeNpc.heroPoints.push({ hash: s.self, points: s.arg }); return s._activeNpc.enqueueScript(DAMAGE_QUEUE, 2, s.arg); },
        // npc_damage(type, amount): applyDamage floors hitpoints at zero
        2: s => { const n = s._activeNpc; n.levels[HP] = Math.max(0, n.levels[HP] - s.intStack[s.isp - 1]); return 'native'; },
    };
    const runner = { init(_script: any, self: any, _target: any, args: any) {
        return { self, _activeNpc: self, intOperand: 0, args, intStack: [] as number[], isp: 0 }; } };
    const world = { addNpc(npc: any) { npc.levels[HP] = npc.baseLevels[HP]; }, removeNpc() {}, cycle() { control.tick++; } };
    const capture = installCombatCapture({ handlers, opcodes: { queue: 1, damage: 2 }, npcPrototype: FakeNpc.prototype,
        runner, world, damageTrigger: DAMAGE_QUEUE, hitpoints: HP, enemyTypes: [41, 1017], tick: () => control.tick,
        newestRequest: npc => npc.queue.at(-1),
        heroes: (npc, player) => ({ agent: npc.heroPoints.filter((h: any) => h.hash === player).reduce((a: number, h: any) => a + h.points, 0),
            other: Math.max(0, ...npc.heroPoints.filter((h: any) => h.hash !== player).map((h: any) => h.points)) }),
        actor: (self, player) => self === player ? 'agent' : self === other ? 'player' : self instanceof FakeNpc ? 'npc' : 'world',
        agentHp: player => player.levels[0], open: () => control.open });
    capture.bind(agent, { accountId: 'agent' }, 10, 20);
    const hit = (self: any, npc: FakeNpc, damage: number) => handlers[1]({ self, _activeNpc: npc, arg: damage });
    // the npc's queue runs its damage script, which starts in ScriptRunner.init with the entry's args
    const due = (npc: FakeNpc) => {
        for (const request of npc.queue.splice(0)) {
            const state = runner.init(null, npc, null, request.args);
            state.intStack = [1, request.lastInt]; state.isp = 2;
            expect(handlers[2](state)).toBe('native');
        }
    };
    const direct = (self: any, npc: FakeNpc, amount: number) => handlers[2]({ self, _activeNpc: npc, intOperand: 0, intStack: [1, amount], isp: 2 });
    return { control, agent, other, world, hit, due, direct, audit: () => capture.audit(agent), capture };
}
const kinds = (audit: any) => audit.events.map((e: any) => e.kind);

test('queued damage is attributed to the actor whose script queued it, through the npc queue', () => {
    const f = fixture(), chicken = new FakeNpc(7, 41);
    f.hit(f.agent, chicken, 1); f.hit(f.other, chicken, 1); f.due(chicken);
    f.hit(f.agent, chicken, 2); f.due(chicken);
    const audit = f.audit();
    expect(audit.failure).toBeNull();
    expect(kinds(audit)).toEqual(['life', 'hit', 'hit', 'damage', 'damage', 'hit', 'damage']);
    const [life, first, second, d1, d2, , last] = audit.events;
    expect(life).toMatchObject({ life: '7:0', type: 41, hp: 3, baseHp: 3 });
    expect(first).toMatchObject({ hitter: 'agent', hp: 3, baseHp: 3, damage: 1, heroes: { agent: 1, other: 0 } });
    expect(second).toMatchObject({ hitter: 'player', hp: 3, heroes: { agent: 1, other: 1 } });
    expect([d1.hitter, d1.hit, d1.hpBefore, d1.hpAfter]).toEqual(['agent', first.sequence, 3, 2]);
    expect([d2.hitter, d2.hit, d2.hpBefore, d2.hpAfter]).toEqual(['player', second.sequence, 2, 1]);
    expect([last.hitter, last.hpBefore, last.hpAfter, last.heroes]).toEqual(['agent', 1, 0, { agent: 3, other: 1 }]);
});

test('damage applied directly is the running script owner; other npcs and closed windows are not followed', () => {
    const f = fixture(), chicken = new FakeNpc(7, 1017), guard = new FakeNpc(8, 26);
    f.direct(f.other, chicken, 1);
    f.hit(f.other, guard, 1); f.due(guard);           // another actor fighting another kind of npc: ignored
    f.hit(f.agent, guard, 1); f.due(guard);           // the agent's own fights are always followed
    f.control.open = false;
    f.hit(f.agent, chicken, 1); f.due(chicken);       // after the terminal nothing is recorded
    const audit = f.audit();
    expect(kinds(audit)).toEqual(['life', 'damage', 'life', 'hit', 'damage']);
    expect(audit.events[1]).toMatchObject({ hitter: 'player', hit: null, hpBefore: 3, hpAfter: 2 });
    expect(audit.events[2]).toMatchObject({ life: '8:0', type: 26, hp: 2 });
    expect(audit.open).toBe(false);
});

test('regeneration, removal and respawn are recorded; each respawn is a new life', () => {
    const f = fixture(), chicken = new FakeNpc(7, 41);
    f.hit(f.agent, chicken, 1); f.due(chicken);
    chicken.levels[HP] = 3; f.world.cycle();          // regenerated during the tick
    f.hit(f.agent, chicken, 3); f.due(chicken);
    Object.assign(chicken, { lastMovement: 12, x: 3230, z: 3297 });   // its last step, for npc_arrivedelay
    f.world.removeNpc(chicken); f.world.addNpc(chicken, -1, false);
    f.agent.levels[0] = 9; f.world.cycle();
    const audit = f.audit();
    expect(kinds(audit)).toEqual(['life', 'hit', 'damage', 'hp', 'hit', 'damage', 'removed', 'life', 'agent-hp']);
    expect(audit.events[6]).toMatchObject({ life: '7:0', lastMovement: 12, x: 3230, z: 3297 });
    expect(audit.events[3]).toMatchObject({ life: '7:0', hpBefore: 2, hpAfter: 3, tick: 11 });
    expect(audit.events[7]).toMatchObject({ life: '7:1', previous: '7:0', hp: 3 });
    expect(audit.events[8]).toMatchObject({ hpBefore: 10, hpAfter: 9 });
    expect(audit.agentHp).toBe(10);
});

test('a respawn without removal or a missing queue entry is a sticky failure, never a silent gap', () => {
    const f = fixture(), chicken = new FakeNpc(7, 41);
    f.hit(f.agent, chicken, 1);
    f.world.addNpc(chicken, -1, false);
    expect(f.audit().failure).toBe('followed npc respawned without removal');
    const g = fixture(), other = new FakeNpc(9, 41);
    other.queue = { push() {}, at: () => undefined, splice: () => [] } as any;  // the entry is not where the capture reads it
    g.hit(g.agent, other, 1);
    expect(g.audit().failure).toBe('queued damage entry not found');
    expect(() => g.capture.release(g.agent)).toThrow('release a closed combat capture');
    g.control.open = false;
    expect(g.capture.release(g.agent).schema).toBe('native-combat-capture-v1');
});

test('every attack on the agent is recorded with its planned damage and attacker, a zero included', () => {
    const control = { tick: 10, open: true };
    class FakePlayer { levels = [7]; queue: any[] = []; enqueueScript(script: any, type: any, delay = 0, args: any[] = []) { this.queue.push({ script, type, delay, args }); } }
    const agent = new FakePlayer(), bystander = new FakePlayer(), goblin = new FakeNpc(9, 100);
    const DAMAGE_SCRIPT = 77, OTHER_SCRIPT = 78;
    const handlers: Record<number, (state: any) => any> = {
        1: s => s._activeNpc.enqueueScript(DAMAGE_QUEUE, 2, s.arg), 2: () => 'native',
        // queue(script, delay, arg) in a script whose active player is the target (npc_default_attack's playerhit_n_melee)
        3: s => s._activePlayer.enqueueScript({ id: s.script }, 'normal', 0, [s.arg]),
    };
    const capture = installCombatCapture({ handlers, opcodes: { queue: 1, damage: 2, playerQueue: 3 }, npcPrototype: FakeNpc.prototype,
        playerPrototype: FakePlayer.prototype, agentDamageScript: DAMAGE_SCRIPT,
        runner: { init: () => ({}) }, world: { addNpc() {}, removeNpc() {}, cycle() { control.tick++; } }, damageTrigger: DAMAGE_QUEUE,
        hitpoints: HP, enemyTypes: [41, 1017], tick: () => control.tick, newestRequest: npc => npc.queue.at(-1),
        heroes: () => ({ agent: 0, other: 0 }), actor: (self, player) => self === player ? 'agent' : self instanceof FakeNpc ? 'npc' : 'world',
        agentHp: player => player.levels[0], open: () => control.open });
    capture.bind(agent, { accountId: 'agent' }, 10, 20);
    handlers[3]({ self: goblin, _activePlayer: agent, script: DAMAGE_SCRIPT, arg: 1 });
    handlers[3]({ self: goblin, _activePlayer: agent, script: DAMAGE_SCRIPT, arg: 0 });      // a miss still counts
    handlers[3]({ self: goblin, _activePlayer: agent, script: OTHER_SCRIPT, arg: 1 });       // another queued script
    handlers[3]({ self: goblin, _activePlayer: bystander, script: DAMAGE_SCRIPT, arg: 1 });  // another player
    expect(agent.queue).toHaveLength(3);                                                      // the native queue still runs
    const audit = capture.audit(agent);
    expect(audit.failure).toBeNull();
    expect(audit.events.map((e: any) => [e.kind, e.damage, e.nid, e.type, e.hp])).toEqual(
        [['agent-hit', 1, 9, 100, 7], ['agent-hit', 0, 9, 100, 7]]);
});
