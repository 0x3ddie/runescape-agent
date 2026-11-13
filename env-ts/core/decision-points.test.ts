import { test, expect } from 'bun:test';
import { DECISION_WAIT_TICKS, keepWaiting } from './decision-points';

const running = { busy: true, dead: false, hp: 20, hpAtDecision: 20, dialog: false, waited: 0 };

test('a running option keeps waiting until it ends or something happens', () => {
    expect(keepWaiting(running)).toBe(true);
    expect(keepWaiting({ ...running, busy: false })).toBe(false);           // the option ended
    expect(keepWaiting({ ...running, hp: 19 })).toBe(false);                // hit
    expect(keepWaiting({ ...running, dialog: true })).toBe(false);          // level-up or bank dialog
    expect(keepWaiting({ ...running, dead: true })).toBe(false);
    expect(keepWaiting({ ...running, waited: DECISION_WAIT_TICKS })).toBe(false); // observe regularly
    expect(keepWaiting({ ...running, hp: 21 })).toBe(true);                 // regenerating is not news
});
