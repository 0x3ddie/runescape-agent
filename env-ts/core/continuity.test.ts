import { expect, test } from 'bun:test';
import { checkContinuity, continuityEnvelope, continuityOptions } from './continuity';
import type { LifecycleSnapshot } from '../engine/lifecycle';

const alive: LifecycleSnapshot = { phase: 'alive', deathSequence: 0, respawnSequence: 0,
    deathTick: null, respawnTick: null, completionSource: null, failure: null };
const dying: LifecycleSnapshot = { ...alive, phase: 'dying', deathSequence: 1, deathTick: 10 };
const recovered: LifecycleSnapshot = { ...dying, phase: 'alive', respawnSequence: 1,
    respawnTick: 20, completionSource: 'player_death_queue' };

test('legacy stays opt-out and continuous mode requires both version and experiment gate', () => {
    expect(continuityOptions({}, undefined)).toBeNull();
    expect(continuityOptions({ lifecycle: 'episodic-v1' }, undefined)).toBeNull();
    expect(continuityOptions({ contract: 'item-v2' }, '1')).toBeNull();
    expect(() => continuityOptions({ lifecycle: 'continuous-v1' }, '1')).toThrow('item-v2');
    expect(() => continuityOptions({ lifecycle: 'continuous-v1', contract: 'item-v2' })).toThrow('RL_CONTINUITY_EXPERIMENT');
    expect(continuityOptions({ lifecycle: 'continuous-v1', contract: 'item-v2' }, '1')).toEqual({ maxRespawnTicks: 250 });
});

test.each([null, false, '1', 0, -1, 1.5, Infinity, 10001, Number.MAX_SAFE_INTEGER + 1])('invalid respawn bound %s is rejected', bound => {
    expect(() => continuityOptions({ lifecycle: 'continuous-v1', contract: 'item-v2', maxRespawnTicks: bound }, '1')).toThrow();
});

test('unknown lifecycle and orphan bounds rejected before configuration mutation', () => {
    expect(() => continuityOptions({ lifecycle: 'continuous-v2', contract: 'item-v2' }, '1')).toThrow('unsupported');
    expect(() => continuityOptions({ maxRespawnTicks: 250 }, '1')).toThrow('requires continuous-v1');
});

test('both pending death and completed-late respawn obey same inclusive tick deadline', () => {
    expect(() => checkContinuity(dying, 260, 250)).not.toThrow();
    expect(() => checkContinuity(dying, 261, 250)).toThrow('deadline');
    expect(() => checkContinuity({ ...recovered, respawnTick: 260 }, 270, 250)).not.toThrow();
    expect(() => checkContinuity({ ...recovered, respawnTick: 261 }, 270, 250)).toThrow('late');
});

test('HP-only alive status cannot replace root completion acknowledgement', () => {
    expect(() => checkContinuity({ ...dying, phase: 'alive' }, 20, 250)).toThrow('matched respawn');
    expect(() => checkContinuity({ ...recovered, completionSource: null }, 20, 250)).toThrow('acknowledgement');
    expect(() => checkContinuity({ ...dying, phase: 'failed', failure: 'aborted' }, 20, 250)).toThrow('aborted');
});

test('death+respawn between learner observations retains counters and exact account identity', () => {
    const identity = { worldBootId: 'boot', accountId: 'bot', accountEpoch: 1, incarnationId: 'player1' };
    const before = continuityEnvelope(identity, alive, 5, 250);
    const after = continuityEnvelope(identity, recovered, 22, 250);
    expect(after.deathSequence - before.deathSequence).toBe(1);
    expect(after.respawnSequence - before.respawnSequence).toBe(1);
    expect(after).toMatchObject({ ...identity, phase: 'alive', version: 'continuous-v1', observedTick: 22 });
    expect('bank' in after).toBe(false);
    expect(() => continuityEnvelope({ ...identity, accountEpoch: 0 }, alive, 5, 250)).toThrow('identity');
});

test('invalid ticks, chronology and lifecycle counters fail collection', () => {
    expect(() => checkContinuity(recovered, 19, 250)).toThrow('event tick');
    expect(() => checkContinuity({ ...recovered, respawnTick: 9 }, 20, 250)).toThrow('order');
    expect(() => checkContinuity({ ...dying, deathSequence: 3 }, 20, 250)).toThrow('counters');
    expect(() => checkContinuity({ ...alive, deathTick: 0 }, 20, 250)).toThrow('event tick');
});
