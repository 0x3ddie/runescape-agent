/** Explicit experimental lifecycle contract; no game mutations or reward logic. */
import type { LifecycleSnapshot } from '../engine/lifecycle';

export interface ContinuityOptions { maxRespawnTicks: number }
export interface ContinuityIdentity {
    worldBootId: string;
    accountId: string;
    accountEpoch: number;
    incarnationId: string;
}
export function continuityOptions(cfg: { lifecycle?: unknown; contract?: unknown; maxRespawnTicks?: unknown }, gate?: string): ContinuityOptions | null {
    if (cfg.lifecycle === undefined || cfg.lifecycle === 'episodic-v1') {
        if (cfg.maxRespawnTicks !== undefined) throw new Error('maxRespawnTicks requires continuous-v1');
        return null;
    }
    if (cfg.lifecycle !== 'continuous-v1') throw new Error('unsupported lifecycle contract');
    if (cfg.contract !== 'item-v2') throw new Error('continuous-v1 requires item-v2');
    if (gate !== '1') throw new Error('continuous-v1 requires RL_CONTINUITY_EXPERIMENT=1');
    const maxRespawnTicks = cfg.maxRespawnTicks === undefined ? 250 : cfg.maxRespawnTicks;
    if (!Number.isSafeInteger(maxRespawnTicks) || (maxRespawnTicks as number) <= 0 || (maxRespawnTicks as number) > 10000)
        throw new Error('maxRespawnTicks must be an integer in 1..10000');
    return { maxRespawnTicks: maxRespawnTicks as number };
}

export function checkContinuity(snapshot: LifecycleSnapshot, observedTick: number, limit: number): void {
    if (!Number.isSafeInteger(observedTick) || observedTick < 0) throw new Error('invalid continuity observation tick');
    if (snapshot.phase === 'failed' || snapshot.failure) throw new Error(`continuity failed: ${snapshot.failure}`);
    const { deathSequence: deaths, respawnSequence: respawns, deathTick, respawnTick } = snapshot;
    if (!Number.isSafeInteger(deaths) || !Number.isSafeInteger(respawns) || respawns < 0 || deaths < respawns || deaths - respawns > 1)
        throw new Error('invalid lifecycle counters');
    for (const value of [deathTick, respawnTick]) {
        if (value !== null && (!Number.isSafeInteger(value) || value < 0 || value > observedTick))
            throw new Error('invalid lifecycle event tick');
    }
    if ((deaths === 0) !== (deathTick === null) || (respawns === 0) !== (respawnTick === null))
        throw new Error('missing lifecycle event tick');
    if (snapshot.phase === 'dying') {
        if (deaths !== respawns + 1 || deathTick === null) throw new Error('invalid dying lifecycle');
        if (observedTick - deathTick > limit) throw new Error('natural respawn deadline exceeded');
    } else if (snapshot.phase === 'alive') {
        if (deaths !== respawns) throw new Error('alive without matched respawn');
        if (deaths > 0 && (respawnTick === null || deathTick === null || respawnTick < deathTick || respawnTick - deathTick > limit))
            throw new Error('natural respawn completed late or out of order');
    } else throw new Error('unknown lifecycle phase');
    if (snapshot.completionSource !== (respawns > 0 ? 'player_death_queue' : null))
        throw new Error('invalid respawn acknowledgement source');
}

export function continuityEnvelope(identity: ContinuityIdentity, snapshot: LifecycleSnapshot, observedTick: number, limit: number) {
    if (![identity.worldBootId, identity.accountId, identity.incarnationId].every(v => typeof v === 'string' && v.length > 0)
        || !Number.isSafeInteger(identity.accountEpoch) || identity.accountEpoch < 1)
        throw new Error('invalid continuity identity');
    checkContinuity(snapshot, observedTick, limit);
    return { version: 'continuous-v1' as const, ...identity, observedTick, ...snapshot };
}
