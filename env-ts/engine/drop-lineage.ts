/** Opt-in native opcode/World observer. Evaluator receipts only; no policy data.
 * Dependencies are injected so the same adapter runs with Player or NetworkPlayer
 * and offline tests can exercise native call ordering without booting a world.
 */
import type { LifecycleMonitor } from './lifecycle';
import type { LifecycleIdentity, PrivateItems } from './lifecycle-bridge';

type Native = any;
export interface DropLineageHost {
    world: { addObj: (...args: any[]) => any; removeObj: (...args: any[]) => any;
        changeObj: (...args: any[]) => any; revealObj: (...args: any[]) => any };
    handlers: Record<number, (state: Native) => any>;
    invDropAllOpcode: number;
    objTakeItemOpcode: number;
    deathLossProc: object;
    despawnLifecycle: number;
    tick(): number;
    enabled(): boolean;
    production(): boolean;
    playerCount(): number;
    isPlayer(player: unknown): boolean;
    isObj(obj: unknown): boolean;
    /** Check exact current native player/account identity, and socket for browser. */
    validatePlayer(player: Native, identity: LifecycleIdentity): void;
    /** Explicit mechanics fixture, living and idle; no placement/item mutation. */
    verifyArm(player: Native): void;
    privateItems(player: Native): PrivateItems;
    groundAt(level: number, x: number, z: number): Iterable<Native>;
    targetIsNonstackable(): boolean;
    settings(): { objDespawnScale: number; objRevealTicks: number };
}
interface Binding {
    player: Native; identity: LifecycleIdentity; monitor: LifecycleMonitor;
    armed: boolean; retired: boolean; failure: string | null; lastTick: number;
    baseline: Native; events: Native[]; sequence: number; object: Tracked | null;
}
interface Tracked { obj: Native; id: string; binding: Binding; turns: number; lastTurnTick: number | null;
    remaining: number; removed: boolean; removal: string | null }
interface Context { kind: 'death-drop' | 'pickup' | 'expiry'; binding: Binding; state?: Native; object?: Tracked }
const copy = <T>(value: T): T => structuredClone(value);
const installations = new WeakSet<object>();
const TARGET = 1351;

export function installDropLineage(host: DropLineageHost, options: { maxEvents?: number; maxAccounts?: number } = {}) {
    const maxEvents = options.maxEvents ?? 256, maxAccounts = options.maxAccounts ?? 16;
    if (![maxEvents, maxAccounts].every(n => Number.isSafeInteger(n) && n > 0) || maxEvents > 4096 || maxAccounts > 64)
        throw new Error('invalid drop lineage bounds');
    if (!host.enabled() || host.production() || host.playerCount() !== 0 || installations.has(host.world) ||
        !host.targetIsNonstackable() || !host.handlers[host.invDropAllOpcode] || !host.handlers[host.objTakeItemOpcode])
        throw new Error('drop lineage requires one opt-in nonproduction installation and nonstackable axe');
    const bindings = new Map<object, Binding>();
    const tracked = new WeakMap<object, Tracked>();
    let context: Context | null = null;
    const fail = (b: Binding, error: unknown) => { b.failure ??= String((error as Error)?.message ?? error); };
    const safe = (b: Binding, fn: () => void) => { try { fn(); } catch (error) { fail(b, error); } };
    const require = (ok: unknown, message: string) => { if (!ok) throw new Error(message); };
    const gate = (b: Binding) => {
        require(!b.retired && host.enabled() && !host.production() && host.playerCount() === 1,
            'drop lineage requires active owned one-player experiment');
        host.validatePlayer(b.player, b.identity);
        const tick = host.tick();
        require(Number.isSafeInteger(tick) && tick >= b.lastTick && tick >= 0, 'lineage tick regression');
        b.lastTick = tick;
        return tick;
    };
    const snapshot = (obj: Native) => ({ type: obj.type, count: obj.count, x: obj.x, z: obj.z, level: obj.level,
        lifecycle: obj.lifecycle, lifecycleTick: obj.lifecycleTick, lastLifecycleTick: obj.lastLifecycleTick,
        receiver64: String(obj.receiver64), reveal: obj.reveal, isActive: obj.isActive });
    const capture = (b: Binding, obj: Native) => { let value: Native = null; safe(b, () => { value = snapshot(obj); }); return value; };
    const inventory = (b: Binding) => copy(host.privateItems(b.player));
    const count = (items: PrivateItems) => [...items.inventory, ...items.worn].reduce((n, i) => n + (i.id === TARGET ? i.count : 0), 0);
    const record = (b: Binding, kind: string, data: Native) => {
        const tick = gate(b);
        require(b.events.length < maxEvents, 'drop lineage ledger full');
        b.events.push(copy({ sequence: ++b.sequence, kind, tick, ...data }));
    };
    const watched = (obj: Native) => [...bindings.values()].filter(b => b.armed && !b.retired && obj?.type === TARGET &&
        obj.level === b.baseline.level && obj.x === b.baseline.x && obj.z === b.baseline.z);
    const objBinding = (obj: Native) => tracked.get(obj)?.binding;
    function withContext<T>(next: Context | null, run: () => T): T {
        const previous = context; context = next;
        try { return run(); } finally { context = previous; }
    }
    function watchTurns(t: Tracked) {
        const original = t.obj.turn;
        require(typeof original === 'function', 'native Obj.turn missing');
        t.obj.turn = function (...args: any[]) {
            const before = capture(t.binding, t.obj);
            safe(t.binding, () => {
                gate(t.binding);
                require(this === t.obj && !t.removed, 'tracked turn identity/removal mismatch');
                require(t.lastTurnTick === null || host.tick() === t.lastTurnTick + 1, 'tracked object skipped/repeated turn');
                require(before.lifecycleTick === t.remaining, 'unobserved object lifetime change');
                if (!t.turns) record(t.binding, 'first-turn', { objectId: t.id, before,
                    expectedExpiryTick: host.tick() + before.lifecycleTick - 1 });
            });
            const call = () => original.apply(this, args);
            let result: any;
            try { result = before?.lifecycleTick === 1 ? withContext({ kind: 'expiry', binding: t.binding, object: t }, call) : call(); }
            catch (error) { fail(t.binding, 'native tracked turn threw'); throw error; }
            safe(t.binding, () => {
                t.turns++; t.lastTurnTick = host.tick(); t.remaining = t.obj.lifecycleTick;
                if (before?.lifecycleTick === 1) record(t.binding, 'expiry-turn', { objectId: t.id, before, after: snapshot(t.obj) });
            });
            return result;
        };
    }
    const originalAdd = host.world.addObj;
    host.world.addObj = function (obj: Native, receiver64: bigint, duration: number, ...rest: any[]) {
        const ctx = context?.kind === 'death-drop' ? context : null;
        const affected = obj?.type === TARGET ? (ctx ? [ctx.binding] : watched(obj)) : [];
        const before = affected.length ? capture(affected[0], obj) : null;
        let result: any;
        try { result = originalAdd.call(this, obj, receiver64, duration, ...rest); }
        catch (error) { affected.forEach(b => fail(b, 'native target add threw')); throw error; }
        for (const b of affected) safe(b, () => {
            gate(b);
            require(ctx && ctx.binding === b, 'unattributed axe appeared on watched tile');
            require(host.isObj(obj) && host.isPlayer(b.player) && !b.object && !tracked.has(obj), 'axe native identity duplicated');
            require(obj.type === TARGET && obj.count === 1 && obj.lifecycle === host.despawnLifecycle && obj.isActive === true,
                'axe add was not one active despawn object');
            require(obj.level === b.baseline.level && obj.x === b.baseline.x && obj.z === b.baseline.z,
                'death drop moved outside armed tile');
            require([...host.groundAt(obj.level, obj.x, obj.z)].filter(o => o.type === TARGET && o.isActive).length === 1 &&
                [...host.groundAt(obj.level, obj.x, obj.z)].includes(obj), 'axe was merged or not inserted');
            require(Number.isSafeInteger(obj.lifecycleTick) && obj.lifecycleTick > 0, 'axe effective lifetime missing');
            const life = b.monitor.snapshot();
            require(life.phase === 'dying' && life.deathSequence === 1 && life.respawnSequence === 0, 'drop not first pending death');
            const t: Tracked = { obj, binding: b, id: `${b.identity.incarnationId}:axe:1`, turns: 0,
                lastTurnTick: null, remaining: obj.lifecycleTick, removed: false, removal: null };
            b.object = t; tracked.set(obj, t); watchTurns(t);
            record(b, 'created', { objectId: t.id, deathSequence: life.deathSequence, deathTick: life.deathTick,
                requestedReceiver64: String(receiver64), requestedDuration: duration, before,
                after: snapshot(obj), settings: host.settings(), privateItems: inventory(b) });
        });
        return result;
    };
    const originalRemove = host.world.removeObj;
    host.world.removeObj = function (obj: Native, duration: number, ...rest: any[]) {
        const t = tracked.get(obj), before = t ? capture(t.binding, obj) : null;
        let result: any;
        try { result = originalRemove.call(this, obj, duration, ...rest); }
        catch (error) { if (t) fail(t.binding, 'native target remove threw'); throw error; }
        if (t) safe(t.binding, () => {
            const kind = context?.object === t ? context.kind : 'unknown';
            const removed = before!.isActive === true && obj.isActive === false;
            record(t.binding, 'removed', { objectId: t.id, cause: kind, requestedDuration: duration, removed,
                before, after: snapshot(obj), privateItems: inventory(t.binding) });
            require(removed && !t.removed, 'target removal duplicate or ineffective');
            t.removed = true; t.removal = kind;
            require(kind === 'pickup' || kind === 'expiry', 'unattributed target removal/eviction');
        });
        return result;
    };
    for (const name of ['changeObj', 'revealObj'] as const) {
        const original = host.world[name];
        host.world[name] = function (obj: Native, ...args: any[]) {
            const b = objBinding(obj), before = b ? capture(b, obj) : null;
            let result: any;
            try { result = original.call(this, obj, ...args); }
            catch (error) { if (b) fail(b, `native ${name} threw`); throw error; }
            if (b) safe(b, () => {
                record(b, name, { objectId: tracked.get(obj)!.id, before, after: snapshot(obj) });
                require(name !== 'changeObj', 'nonstackable axe changed count');
            });
            return result;
        };
    }
    const originalDrop = host.handlers[host.invDropAllOpcode];
    host.handlers[host.invDropAllOpcode] = function (state: Native) {
        const b = bindings.get(state?._activePlayer);
        let before: PrivateItems | null = null, ctx: Context | null = null;
        if (b?.armed && !b.retired) safe(b, () => {
            gate(b);
            require(state.intOperand === 0 && state.self === b.player && host.isPlayer(b.player) && state.script === host.deathLossProc &&
                b.monitor.isPendingDeathScript?.(state) === true, 'drop is not exact owned pending death script');
            before = inventory(b); ctx = { kind: 'death-drop', binding: b, state };
        });
        let result: any;
        try { result = withContext(ctx, () => originalDrop.call(this, state)); }
        catch (error) { if (b?.armed) fail(b, 'native death drop handler threw'); throw error; }
        if (b?.armed && before) safe(b, () => {
            const after = inventory(b);
            record(b, 'drop-opcode', { before, after, deathSequence: b.monitor.snapshot().deathSequence });
            require(count(before!) - count(after) === (count(before!) ? 1 : 0), 'unexpected axe inventory loss');
            if (count(before!)) require(!!b.object, 'lost axe has no tracked creation');
        });
        return result;
    };
    const originalTake = host.handlers[host.objTakeItemOpcode];
    host.handlers[host.objTakeItemOpcode] = function (state: Native) {
        const obj = state?._activeObj ?? state?.activeObj, t = tracked.get(obj);
        let before: PrivateItems | null = null;
        if (t) safe(t.binding, () => {
            gate(t.binding);
            require(state.intOperand === 0 && state._activePlayer === t.binding.player && state.self === t.binding.player &&
                host.isPlayer(state._activePlayer) && host.isObj(obj) && !t.removed && obj.isActive,
                'pickup changed native player/object identity');
            require(t.binding.monitor.snapshot().phase === 'alive', 'pickup before completed respawn');
            before = inventory(t.binding);
        });
        let result: any;
        try { result = withContext(t ? { kind: 'pickup', binding: t.binding, object: t, state } : null,
            () => originalTake.call(this, state)); }
        catch (error) { if (t) fail(t.binding, 'native pickup handler threw'); throw error; }
        if (t && before) safe(t.binding, () => {
            const after = inventory(t.binding);
            record(t.binding, 'pickup-opcode', { objectId: t.id, before, after, object: snapshot(obj) });
            require(t.removed && t.removal === 'pickup' && !obj.isActive && count(after) - count(before!) === 1,
                'pickup did not conserve ground axe into inventory');
        });
        return result;
    };
    installations.add(host.world);
    return {
        bind(player: Native, identity: LifecycleIdentity, monitor: LifecycleMonitor) {
            require(host.isPlayer(player) && !bindings.has(player) && bindings.size < maxAccounts, 'invalid/full lineage binding');
            require(typeof monitor.isPendingDeathScript === 'function', 'pending death script predicate required');
            require(['worldBootId','accountId','incarnationId'].every(k => typeof (identity as any)[k] === 'string' && (identity as any)[k]) &&
                Number.isSafeInteger(identity.accountEpoch) && identity.accountEpoch > 0, 'invalid lineage identity');
            const b: Binding = { player, identity: copy(identity), monitor, armed: false, retired: false,
                failure: null, lastTick: -1, baseline: null, events: [], sequence: 0, object: null };
            gate(b); bindings.set(player, b);
        },
        arm(player: Native) {
            const b = bindings.get(player); require(b, 'unbound lineage player');
            gate(b!); require(!b!.armed && !b!.failure, 'lineage already armed or failed');
            host.verifyArm(player);
            const life = b!.monitor.snapshot();
            require(life.phase === 'alive' && life.deathSequence === 0 && life.respawnSequence === 0, 'arm requires fresh alive lifecycle');
            const ground = [...host.groundAt(player.level, player.x, player.z)].filter(o => o.type === TARGET);
            require(!ground.length, 'axe already exists on fixture tile');
            const items = inventory(b!); require(count(items) === 1 && items.inventory.some(i => i.id === TARGET && i.count === 1),
                'fixture must carry exactly one inventory axe');
            b!.baseline = { level: player.level, x: player.x, z: player.z, tick: host.tick(), targetType: TARGET,
                ground: ground.map(snapshot), privateItems: items };
            b!.armed = true; record(b!, 'armed', { baseline: b!.baseline });
            return copy(b!.baseline);
        },
        audit(player: Native) {
            const b = bindings.get(player); require(b, 'unbound lineage player');
            safe(b!, () => gate(b!));
            return copy({ schema: 1, ...b!.identity, observedTick: b!.lastTick, armed: b!.armed,
                failure: b!.failure, eventsLost: false, latestSequence: b!.sequence, baseline: b!.baseline,
                events: b!.events, object: b!.object ? { objectId: b!.object.id, ...snapshot(b!.object.obj),
                    turns: b!.object.turns, removed: b!.object.removed, removal: b!.object.removal } : null });
        },
        retire(player: Native) {
            const b = bindings.get(player); require(b && !b.retired, 'lineage retirement mismatch');
            const receipt = this.audit(player); b!.retired = true; return receipt;
        }
    };
}
