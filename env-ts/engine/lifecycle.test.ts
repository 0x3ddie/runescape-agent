import { expect, test } from 'bun:test';
import { monitorDeath } from './death';
import { monitorLifecycle, type LifecyclePlayer, type LifecycleScript, type LifecycleEvent } from './lifecycle';

const HP = 3;
function fixture() {
    const root = { id: 12, name: '[queue,player_death]' };
    const proc = { id: 13, name: '[proc,player_death]' };
    let tick = 1;
    const player: LifecyclePlayer = {
        levels: Uint8Array.from([1, 1, 1, 10]),
        applyDamage(amount) { (this.levels as Uint8Array)[HP] = Math.max(0, this.levels[HP] - amount); },
        changeStat() {},
        runScript(s: LifecycleScript) {
            // Model actual ScriptRunner: same ScriptState is resumed, while the
            // current script and gosub frame pointer change during p_delay.
            s.opcount++;
            if (s.opcount < 3) { s.script = proc; s.fp = 1; s.execution = 2; return 2; }
            s.script = root; s.fp = 0; s.execution = 1;
            (this.levels as Uint8Array)[HP] = 10;
            return 1;
        }
    };
    const script = (): LifecycleScript => ({ script: root, self: player, _activePlayer: player, execution: 0, fp: 0, opcount: 0 });
    return { player, root, proc, script, tick: () => tick, advance: () => { tick++; } };
}

test('same suspended death queue acknowledges only after full root completion; events occur synchronously', () => {
    const f = fixture(), m = monitorLifecycle(f.player, HP, f.tick, f.root);
    const events: Array<{ event: Readonly<LifecycleEvent>; hp: number }> = [];
    m.subscribe(event => events.push({ event, hp: f.player.levels[HP] }));
    f.player.applyDamage(10, 0);
    const s = f.script();
    f.player.runScript(s); f.advance();
    expect(m.snapshot()).toMatchObject({ phase: 'dying', deathSequence: 1, respawnSequence: 0, deathTick: 1 });
    f.player.runScript(s); f.advance();
    expect(m.snapshot().respawnSequence).toBe(0);
    f.player.runScript(s);
    expect(m.snapshot()).toMatchObject({ phase: 'alive', deathSequence: 1, respawnSequence: 1, respawnTick: 3, completionSource: 'player_death_queue' });
    expect(events.map(e => [e.event.kind, e.event.eventSequence, e.hp])).toEqual([['death', 1, 0], ['respawn', 2, 10]]);
    // Both edges may occur between collector observations; counters retain them.
    f.player.applyDamage(10, 0);
    const next = f.script();
    for (let i = 0; i < 3; i++) f.player.runScript(next);
    expect(m.snapshot()).toMatchObject({ deathSequence: 2, respawnSequence: 2 });
});

test('HP healing and repeated reads cannot acknowledge or multiply one death', () => {
    const f = fixture(), m = monitorLifecycle(f.player, HP, f.tick, f.root);
    f.player.applyDamage(10, 0);
    (f.player.levels as Uint8Array)[HP] = 10;
    f.player.changeStat(HP);
    for (let i = 0; i < 3; i++) expect(m.snapshot()).toMatchObject({ phase: 'dying', deathSequence: 1, respawnSequence: 0 });
});

test('legacy latch stays latched while opt-in observer acknowledges natural respawn', () => {
    const f = fixture();
    const legacy = monitorDeath(f.player, HP, f.tick);
    const m = monitorLifecycle(f.player, HP, f.tick, f.root);
    f.player.applyDamage(10, 0);
    const s = f.script(); for (let i = 0; i < 3; i++) f.player.runScript(s);
    expect(m.snapshot().phase).toBe('alive'); expect(legacy.dead).toBe(true);
});

test('protected-access denial is not ABORTED, and later same root may execute', () => {
    const f = fixture(); const run = f.player.runScript; let denied = true;
    f.player.runScript = function (s) { if (denied) { denied = false; return -1; } return run.call(this, s); };
    const m = monitorLifecycle(f.player, HP, f.tick, f.root);
    f.player.applyDamage(10, 0); const s = f.script();
    expect(f.player.runScript(s)).toBe(-1);
    expect(m.snapshot().phase).toBe('dying');
    for (let i = 0; i < 3; i++) f.player.runScript(s);
    expect(m.snapshot().phase).toBe('alive');
});

test.each(['abort', 'throw', 'wrong-root', 'wrong-player', 'dead-finish'])(
    'death completion fails closed: %s', mode => {
        const f = fixture();
        f.player.runScript = function (s) {
            if (mode === 'throw') throw new Error('script failure');
            s.execution = mode === 'abort' ? -1 : 1;
            if (mode === 'wrong-root') s.script = f.proc;
            if (mode === 'wrong-player') s._activePlayer = {};
            if (mode !== 'dead-finish') (this.levels as Uint8Array)[HP] = 10;
            return s.execution;
        };
        const m = monitorLifecycle(f.player, HP, f.tick, f.root);
        f.player.applyDamage(10, 0);
        try { f.player.runScript(f.script()); } catch {}
        expect(m.snapshot()).toMatchObject({ phase: 'failed', deathSequence: 1, respawnSequence: 0 });
    }
);

test('lookalike ScriptFile object cannot acknowledge the installed root', () => {
    const f = fixture(), m = monitorLifecycle(f.player, HP, f.tick, f.root);
    f.player.applyDamage(10, 0);
    const s = f.script(); s.script = { ...f.root }; s.opcount = 2;
    f.player.runScript(s);
    expect(m.snapshot()).toMatchObject({ phase: 'dying', respawnSequence: 0 });
});

test('overlapping queue and missing lethal event are failures', () => {
    const a = fixture(), ma = monitorLifecycle(a.player, HP, a.tick, a.root);
    a.player.runScript(a.script()); expect(ma.snapshot().phase).toBe('failed');
    const b = fixture(), mb = monitorLifecycle(b.player, HP, b.tick, b.root);
    b.player.applyDamage(10, 0); b.player.runScript(b.script()); b.player.runScript(b.script());
    expect(mb.snapshot().failure).toBe('overlapping death queue');
});

test('listener failure is explicit, unsubscribe works, and snapshots cannot mutate state', () => {
    const f = fixture(), m = monitorLifecycle(f.player, HP, f.tick, f.root);
    let called = 0; const off = m.subscribe(() => { called++; }); off();
    m.subscribe(() => { throw new Error('evidence writer failed'); });
    f.player.applyDamage(10, 0);
    expect(called).toBe(0); expect(m.snapshot().failure).toBe('lifecycle event listener failed');
    expect(Object.isFrozen(m.snapshot())).toBe(true);
});

test('late installation and wrong root are rejected; installation is idempotent', () => {
    const f = fixture();
    expect(() => monitorLifecycle(f.player, HP, f.tick, f.proc)).toThrow('exact player_death queue');
    const m = monitorLifecycle(f.player, HP, f.tick, f.root);
    expect(monitorLifecycle(f.player, HP, f.tick, f.root)).toBe(m);
    expect(() => monitorLifecycle(f.player, 0, f.tick, f.root)).toThrow('HP index');
    const dead = fixture(); dead.player.applyDamage(10, 0);
    expect(() => monitorLifecycle(dead.player, HP, dead.tick, dead.root)).toThrow('living player');
});

test('regressing ticks fail and duplicate completed script cannot create a second acknowledgement', () => {
    const f = fixture(); let tick = 5;
    const m = monitorLifecycle(f.player, HP, () => tick, f.root);
    tick = 4;
    expect(m.snapshot().failure).toBe('invalid or regressing world tick');
    const other = fixture(), mo = monitorLifecycle(other.player, HP, other.tick, other.root);
    other.player.applyDamage(10, 0);
    const s = other.script(); for (let i = 0; i < 3; i++) other.player.runScript(s);
    other.player.runScript(s);
    expect(mo.snapshot()).toMatchObject({ phase: 'failed', respawnSequence: 1, failure: 'completed death queue executed again' });
});
