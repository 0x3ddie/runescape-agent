/** Account-scoped, evaluator-only lifecycle bridge. No network listener or game mutation.
 * Bind before damage, unbind explicitly before reset/relogin, and archive its receipt.
 * A server snapshot does NOT prove Chromium consumed post-respawn packets. Callers
 * must separately require a correlated fresh client update and reject disconnects.
 */
import type { LifecycleMonitor, LifecycleSnapshot } from './lifecycle';

export interface LifecycleIdentity {
    worldBootId: string;
    accountId: string;
    accountEpoch: number;
    incarnationId: string;
}
export interface PrivateItem { slot: number; id: number; count: number }
export interface PrivateItems {
    inventory: PrivateItem[];
    worn: PrivateItem[];
    bank: PrivateItem[];
}
interface EventSnapshot extends LifecycleSnapshot { eventSequence: number; kind: 'death' | 'respawn' }
type SubscribableMonitor = LifecycleMonitor & { subscribe(listener: (event: EventSnapshot) => void): () => void };
interface Receipt {
    sequence: number;
    monitorEventSequence: number;
    kind: 'death' | 'respawn';
    worldTick: number;
    lifecycle: LifecycleSnapshot;
    privateItems: PrivateItems | null;
}
interface Binding {
    identity: LifecycleIdentity;
    monitor: SubscribableMonitor;
    capture: () => PrivateItems;
    initialItems: PrivateItems;
    events: Receipt[];
    sequence: number;
    lastMonitorSequence: number | null;
    observedTick: number;
    failure: string | null;
    unsubscribe: () => void;
}
const copy = <T>(value: T): T => structuredClone(value);
function items(value: PrivateItems): PrivateItems {
    const result = copy(value);
    for (const key of ['inventory', 'worn', 'bank'] as const) {
        if (!Array.isArray(result[key])) throw new Error('missing private item snapshot');
        const seen = new Set<number>();
        for (const item of result[key]) {
            if (![item.slot, item.id, item.count].every(Number.isSafeInteger) ||
                item.slot < 0 || item.id < 0 || item.count <= 0 || seen.has(item.slot))
                throw new Error('invalid private item snapshot');
            seen.add(item.slot);
        }
        result[key].sort((a, b) => a.slot - b.slot);
    }
    return result;
}
function lifecycle(snapshot: LifecycleSnapshot): LifecycleSnapshot {
    // Explicit projection: future private monitor fields must not become policy inputs.
    return { phase: snapshot.phase, deathSequence: snapshot.deathSequence,
        respawnSequence: snapshot.respawnSequence, deathTick: snapshot.deathTick,
        respawnTick: snapshot.respawnTick, completionSource: snapshot.completionSource,
        failure: snapshot.failure };
}

export class LifecycleBridge {
    private readonly bindings = new Map<object, Binding>();
    private readonly maxEvents: number;
    private readonly maxAccounts: number;
    constructor(options: { maxEventsPerAccount?: number; maxAccounts?: number } = {}) {
        this.maxEvents = options.maxEventsPerAccount ?? 64;
        this.maxAccounts = options.maxAccounts ?? 64;
        if (![this.maxEvents, this.maxAccounts].every(n => Number.isSafeInteger(n) && n > 0))
            throw new Error('invalid lifecycle bridge bounds');
    }
    bind(player: object, identity: LifecycleIdentity, monitor: SubscribableMonitor,
        capturePrivateItems: () => PrivateItems): void {
        if (this.bindings.has(player)) throw new Error('player already bound; explicit unbind required');
        if (this.bindings.size >= this.maxAccounts) throw new Error('lifecycle account capacity exceeded');
        for (const key of ['worldBootId', 'accountId', 'incarnationId'] as const)
            if (typeof identity[key] !== 'string' || !identity[key]) throw new Error('explicit lifecycle identity required');
        if (!Number.isSafeInteger(identity.accountEpoch) || identity.accountEpoch < 1)
            throw new Error('accountEpoch must be a positive safe integer');
        for (const bound of this.bindings.values()) {
            if (bound.identity.worldBootId === identity.worldBootId &&
                (bound.identity.accountId === identity.accountId || bound.identity.incarnationId === identity.incarnationId))
                throw new Error('account or incarnation already bound');
        }
        const initial = monitor.snapshot();
        if (initial.phase !== 'alive' || initial.deathSequence !== 0 || initial.respawnSequence !== 0)
            throw new Error('bind requires fresh alive monitor with zero counters; midlife attachment unsupported');
        const binding: Binding = { identity: { worldBootId: identity.worldBootId, accountId: identity.accountId,
            accountEpoch: identity.accountEpoch, incarnationId: identity.incarnationId }, monitor, capture: capturePrivateItems,
            initialItems: items(capturePrivateItems()), events: [], sequence: 0,
            lastMonitorSequence: initial.deathSequence + initial.respawnSequence,
            observedTick: -1, failure: null, unsubscribe: () => {} };
        this.bindings.set(player, binding);
        try {
            binding.unsubscribe = monitor.subscribe(event => {
                if (this.bindings.get(player) !== binding) return;
                if (binding.lastMonitorSequence !== null && event.eventSequence !== binding.lastMonitorSequence + 1)
                    binding.failure ??= 'monitor event sequence gap or duplicate';
                binding.lastMonitorSequence = event.eventSequence;
                const worldTick = event.kind === 'death' ? event.deathTick : event.respawnTick;
                if (worldTick === null || !Number.isSafeInteger(worldTick) ||
                    worldTick < Math.max(binding.observedTick, binding.events.at(-1)?.worldTick ?? -1))
                    binding.failure ??= 'invalid or regressing event tick';
                let privateItems: PrivateItems | null = null;
                try { privateItems = items(binding.capture()); }
                catch { binding.failure ??= 'private item capture failed'; }
                binding.events.push({ sequence: ++binding.sequence, monitorEventSequence: event.eventSequence,
                    kind: event.kind, worldTick: worldTick ?? -1, lifecycle: lifecycle(event), privateItems });
                if (binding.events.length > this.maxEvents) binding.events.shift();
            });
        } catch (error) { this.bindings.delete(player); throw error; }
    }
    private read(player: object, observedTick: number) {
        const binding = this.bindings.get(player);
        if (!binding) throw new Error('unbound player');
        if (!Number.isSafeInteger(observedTick) || observedTick < 0 || observedTick < binding.observedTick)
            throw new Error('invalid or regressing observation tick');
        binding.observedTick = observedTick;
        const current = lifecycle(binding.monitor.snapshot());
        if ((current.deathTick ?? 0) > observedTick || (current.respawnTick ?? 0) > observedTick)
            binding.failure ??= 'observation tick precedes lifecycle event';
        const envelope = { version: 'continuous-v1' as const, ...binding.identity, observedTick,
            ...current, phase: binding.failure ? 'failed' as const : current.phase,
            failure: binding.failure ?? current.failure };
        return { binding, envelope };
    }
    policySnapshot(player: object, observedTick: number) {
        return copy(this.read(player, observedTick).envelope);
    }
    auditSnapshot(player: object, observedTick: number, afterSequence = 0) {
        if (!Number.isSafeInteger(afterSequence) || afterSequence < 0) throw new Error('invalid event cursor');
        const { binding, envelope } = this.read(player, observedTick);
        if (afterSequence > binding.sequence) throw new Error('event cursor ahead of ledger');
        const oldestSequence = binding.events[0]?.sequence ?? binding.sequence + 1;
        return copy({ ...envelope, oldestSequence, latestSequence: binding.sequence,
            eventsLost: afterSequence < oldestSequence - 1,
            // Bound before admission/reset setup; NOT the returned reset baseline.
            // Collectors archive currentPrivateItems after reset separately.
            bindingPrivateItems: binding.initialItems,
            events: binding.events.filter(event => event.sequence > afterSequence) });
    }
    unbind(player: object, observedTick: number) {
        const receipt = this.auditSnapshot(player, observedTick);
        const binding = this.bindings.get(player)!;
        this.bindings.delete(player);
        binding.unsubscribe();
        return receipt;
    }
}
