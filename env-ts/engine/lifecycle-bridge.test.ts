import { expect, test } from 'bun:test';
import { LifecycleBridge, type PrivateItems } from './lifecycle-bridge';
import type { LifecycleSnapshot } from './lifecycle';
import { monitorLifecycle, type LifecyclePlayer, type LifecycleScript } from './lifecycle';

function fixture() {
    const player = {};
    let state: LifecycleSnapshot = { phase: 'alive', deathSequence: 0, respawnSequence: 0,
        deathTick: null, respawnTick: null, completionSource: null, failure: null };
    let sequence = 0;
    const listeners = new Set<(e: any) => void>();
    const monitor = { snapshot: () => ({ ...state }), subscribe(fn: (e: any) => void) {
        listeners.add(fn); return () => { listeners.delete(fn); };
    } };
    const privateItems: PrivateItems = { inventory: [{ slot: 0, id: 1351, count: 1 }],
        worn: [], bank: [{ slot: 0, id: 995, count: 25 }] };
    const identity = { worldBootId: 'world1', accountId: 'alice', accountEpoch: 1, incarnationId: 'p1' };
    const emit = (kind: 'death' | 'respawn', tick: number, overrideSequence?: number) => {
        state = kind === 'death' ? { ...state, phase: 'dying', deathSequence: state.deathSequence + 1, deathTick: tick }
            : { ...state, phase: 'alive', respawnSequence: state.deathSequence, respawnTick: tick,
                completionSource: 'player_death_queue' };
        const event = { ...state, kind, eventSequence: overrideSequence ?? ++sequence };
        for (const listener of listeners) listener(event);
    };
    return { player, monitor, privateItems, identity, emit, listeners };
}

test('captures missed client zero-HP and synchronous private item transitions without policy leakage', () => {
    const f = fixture(), bridge = new LifecycleBridge();
    bridge.bind(f.player, { ...f.identity, hidden: 'do not expose' } as any, f.monitor, () => f.privateItems);
    f.emit('death', 10);
    f.privateItems.inventory = [];
    f.emit('respawn', 14);
    const policy = bridge.policySnapshot(f.player, 15);
    expect(policy.deathSequence).toBe(1);
    expect(policy.respawnSequence).toBe(1);
    expect(policy.version).toBe('continuous-v1');
    expect(JSON.stringify(policy)).not.toMatch(/bank|inventory|hidden|1351|995/);
    const audit = bridge.auditSnapshot(f.player, 15);
    expect(audit.events[0].privateItems!.inventory[0].id).toBe(1351);
    expect(audit.events[1].privateItems!.inventory).toEqual([]);
    expect(audit.events[0].privateItems!.bank).toEqual(audit.events[1].privateItems!.bank);
    audit.events[0].privateItems!.bank[0].count = 999;
    expect(bridge.auditSnapshot(f.player, 15).events[0].privateItems!.bank[0].count).toBe(25);
});

test('bounded ledger reports cursor loss, repeated reads do not create events', () => {
    const f = fixture(), bridge = new LifecycleBridge({ maxEventsPerAccount: 2 });
    bridge.bind(f.player, f.identity, f.monitor, () => f.privateItems);
    f.emit('death', 1); f.emit('respawn', 2); f.emit('death', 3);
    expect(bridge.auditSnapshot(f.player, 3).eventsLost).toBe(true);
    expect(bridge.auditSnapshot(f.player, 3, 1).eventsLost).toBe(false);
    expect(bridge.auditSnapshot(f.player, 3, 2).events.map(x => x.sequence)).toEqual([3]);
    expect(bridge.auditSnapshot(f.player, 3).latestSequence).toBe(3);
    expect(() => bridge.auditSnapshot(f.player, 3, 4)).toThrow();
});

test('account binding and reset are explicit, old listeners cannot contaminate a new epoch', () => {
    const f = fixture(), bridge = new LifecycleBridge();
    bridge.bind(f.player, f.identity, f.monitor, () => f.privateItems);
    expect(() => bridge.bind(f.player, f.identity, f.monitor, () => f.privateItems)).toThrow();
    expect(() => bridge.bind({}, f.identity, f.monitor, () => f.privateItems)).toThrow();
    f.emit('death', 1); f.emit('respawn', 2);
    expect(bridge.unbind(f.player, 2).events.length).toBe(2);
    expect(f.listeners.size).toBe(0);
    expect(() => bridge.policySnapshot(f.player, 2)).toThrow();
    expect(() => bridge.bind(f.player, { ...f.identity, accountEpoch: 2 }, f.monitor, () => f.privateItems)).toThrow();
    const fresh = fixture();
    bridge.bind(fresh.player, { ...f.identity, accountEpoch: 2, incarnationId: 'p2' }, fresh.monitor, () => fresh.privateItems);
    expect(bridge.auditSnapshot(fresh.player, 3).bindingPrivateItems.bank[0].count).toBe(25);
    expect(bridge.auditSnapshot(fresh.player, 3).events).toEqual([]);
});

test('late attachment cannot pretend it captured predeath evidence', () => {
    const f = fixture(), bridge = new LifecycleBridge();
    f.emit('death', 1);
    expect(() => bridge.bind(f.player, f.identity, f.monitor, () => f.privateItems)).toThrow();
});

test('capture failures latch failure without throwing into game execution', () => {
    const f = fixture(), bridge = new LifecycleBridge();
    let broken = false;
    bridge.bind(f.player, f.identity, f.monitor, () => { if (broken) throw Error('capture'); return f.privateItems; });
    broken = true;
    expect(() => f.emit('death', 1)).not.toThrow();
    expect(bridge.policySnapshot(f.player, 1).phase).toBe('failed');
    expect(bridge.auditSnapshot(f.player, 1).events[0].privateItems).toBeNull();
});

test('event gaps and clock regression fail visibly', () => {
    const f = fixture(), bridge = new LifecycleBridge();
    bridge.bind(f.player, f.identity, f.monitor, () => f.privateItems);
    f.emit('death', 3); f.emit('respawn', 4, 4);
    expect(bridge.policySnapshot(f.player, 4).failure).toContain('sequence');
    expect(() => bridge.policySnapshot(f.player, 3)).toThrow();
    const g = fixture(), other = new LifecycleBridge();
    other.bind(g.player, g.identity, g.monitor, () => g.privateItems);
    g.emit('death', 3); g.emit('respawn', 2);
    expect(other.policySnapshot(g.player, 4).failure).toContain('regressing');
});

test('bound account capacity and required identity fail closed', () => {
    const f = fixture(), bridge = new LifecycleBridge({ maxAccounts: 1 });
    for (const accountEpoch of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, '1', null])
        expect(() => bridge.bind(f.player, { ...f.identity, accountEpoch } as any, f.monitor, () => f.privateItems)).toThrow();
    bridge.bind(f.player, f.identity, f.monitor, () => f.privateItems);
    expect(() => bridge.bind({}, { ...f.identity, accountId: 'bob', incarnationId: 'p2' }, f.monitor, () => f.privateItems)).toThrow();
});

test('real monitor bridge captures death and completed root before any client observation', () => {
    const root = { id: 4, name: '[queue,player_death]' };
    const f = fixture();
    let tick = 4;
    const player: LifecyclePlayer = {
        levels: [10], applyDamage(n) { (this.levels as number[])[0] -= n; }, changeStat() {},
        runScript(script: LifecycleScript) {
            f.privateItems.inventory = [];
            (this.levels as number[])[0] = 10;
            script.execution = 1; script.opcount++; return 1;
        }
    };
    const monitor = monitorLifecycle(player, 0, () => tick, root);
    const bridge = new LifecycleBridge();
    bridge.bind(player, f.identity, monitor, () => f.privateItems);
    player.applyDamage(10, 0);
    tick = 8;
    player.runScript({ script: root, self: player, _activePlayer: player, execution: 0, fp: 0, opcount: 0 });
    const audit = bridge.auditSnapshot(player, tick);
    expect(audit.events.map(e => [e.kind, e.worldTick, e.privateItems!.inventory.length])).toEqual([
        ['death', 4, 1], ['respawn', 8, 0]
    ]);
    expect(audit.failure).toBeNull();
});
