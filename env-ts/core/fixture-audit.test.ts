import { expect, test } from 'bun:test';
import { assertFixtureIdle, type FixtureIdleState } from './fixture-audit';

const idle = (): FixtureIdleState => ({ hp: 10, isDead: false, loggingOut: false,
    activeScript: null, target: null, hasWaypoints: false, bankOpen: false, delayed: false, protect: false });

test('only explicitly alive and idle state permits fixture mutation', () => {
    expect(() => assertFixtureIdle(idle())).not.toThrow();
    for (const hp of [0, -1, NaN, Infinity]) expect(() => assertFixtureIdle({ ...idle(), hp })).toThrow();
    for (const flag of ['isDead', 'loggingOut', 'hasWaypoints', 'bankOpen', 'delayed', 'protect']) {
        expect(() => assertFixtureIdle({ ...idle(), [flag]: true })).toThrow();
        expect(() => assertFixtureIdle({ ...idle(), [flag]: undefined } as any)).toThrow();
    }
    for (const field of ['activeScript', 'target']) {
        expect(() => assertFixtureIdle({ ...idle(), [field]: {} })).toThrow();
        expect(() => assertFixtureIdle({ ...idle(), [field]: undefined })).toThrow();
    }
});
