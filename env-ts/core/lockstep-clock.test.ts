import { test, expect } from 'bun:test';
import { SharedClock } from './lockstep-clock';

const settle = () => new Promise(r => setTimeout(r, 0));

test('the world ticks only when every agent waits, and each wait sees exactly one tick', async () => {
    let ticks = 0;
    const clock = new SharedClock(() => ticks++, 2);
    const a = {}, b = {};
    clock.join(a);
    clock.join(b);
    const seen: string[] = [];
    // a waits three ticks while b, deciding between steps, holds the world, then waits one tick at a time
    const aWaits = (async () => { for (let i = 0; i < 3; i++) { await clock.tick(a); seen.push(`a${ticks}`); } })();
    await settle();
    expect(ticks).toBe(0);
    for (let i = 0; i < 3; i++) {
        await clock.tick(b);
        seen.push(`b${ticks}`);
        await settle();
    }
    await aWaits;
    expect(ticks).toBe(3);
    expect(seen.filter(s => s.startsWith('a'))).toEqual(['a1', 'a2', 'a3']);
    expect(seen.filter(s => s.startsWith('b'))).toEqual(['b1', 'b2', 'b3']);
});

test('an agent that leaves no longer holds the world, and its own wait ends without a tick', async () => {
    let ticks = 0;
    const clock = new SharedClock(() => ticks++, 3);
    const a = {}, b = {}, c = {};
    [a, b, c].forEach(m => clock.join(m));
    const aDone = clock.tick(a), bDone = clock.tick(b);
    await settle();
    expect(ticks).toBe(0); // c holds it
    clock.leave(b);        // b's own wait ends; a still waits for c
    await bDone;
    expect(ticks).toBe(0);
    clock.leave(c);
    await aDone;
    expect(ticks).toBe(1);
    expect(clock.size).toBe(1);
});

test('a full world refuses a member, and only a member waits, once at a time', async () => {
    const clock = new SharedClock(() => {}, 2);
    const a = {};
    clock.join(a);
    clock.join({});
    expect(() => clock.join({})).toThrow('serves 2 agents');
    await expect(clock.tick({})).rejects.toThrow('only a member');
    void clock.tick(a);
    await expect(clock.tick(a)).rejects.toThrow('once at a time');
    expect(() => new SharedClock(() => {}, 0)).toThrow();
});

// A wall clock the test moves by hand: `later` keeps the run and its due time; fire(ms) moves the time and runs what is due.
function handTimers() {
    let now = 0;
    let due: { at: number; run: () => void }[] = [];
    return {
        timers: { now: () => now, later: (run: () => void, ms: number) => { due.push({ at: now + ms, run }); } },
        set: (ms: number) => { now = ms; },
        fire(ms: number) {
            now += ms;
            const ready = due.filter(d => d.at <= now);
            due = due.filter(d => d.at > now);
            for (const d of ready) d.run();
        },
        get count() { return due.length; }
    };
}

test('a paced clock ticks at most once every paceMs after its last tick, and only while every member waits', async () => {
    let ticks = 0;
    const hand = handTimers();
    const clock = new SharedClock(() => ticks++, 3, 100, hand.timers);
    const a = {}, b = {};
    clock.join(a);
    clock.join(b);
    const first = Promise.all([clock.tick(a), clock.tick(b)]);
    expect(ticks).toBe(1); // the first tick has no earlier one to wait for
    await first;
    const second = Promise.all([clock.tick(a), clock.tick(b)]);
    expect([ticks, clock.pending, hand.count]).toEqual([1, true, 1]); // set for 100 ms after the first
    hand.fire(99);
    expect(ticks).toBe(1);
    hand.fire(1);
    expect([ticks, clock.pending]).toEqual([2, false]);
    await second;
    hand.set(450); // waits that come later than the pace: the tick comes at once
    const third = Promise.all([clock.tick(a), clock.tick(b)]);
    expect([ticks, clock.pending]).toEqual([3, false]);
    await third;
    // a member that joins while a tick is set holds it: the tick comes when that member waits too
    const fourth = Promise.all([clock.tick(a), clock.tick(b)]);
    expect(clock.pending).toBe(true);
    const c = {};
    clock.join(c);
    hand.fire(100);
    expect([ticks, clock.pending]).toEqual([3, false]);
    const cWaits = clock.tick(c);
    expect(ticks).toBe(4);
    await Promise.all([fourth, cWaits]);
    // a member that leaves while a tick is set: its own wait ends, and the others' tick comes at its time
    const fifth = Promise.all([clock.tick(a), clock.tick(b)]);
    const cGone = clock.tick(c);
    expect(clock.pending).toBe(true);
    clock.leave(c);
    await cGone;
    expect(ticks).toBe(4);
    hand.fire(100);
    await fifth;
    expect(ticks).toBe(5);
    expect(() => new SharedClock(() => {}, 1, -1)).toThrow('pace');
    expect(() => new SharedClock(() => {}, 1, 4.5)).toThrow('pace');
});

test('an unpaced clock never sets a tick for later', async () => {
    let ticks = 0;
    const hand = handTimers();
    const clock = new SharedClock(() => ticks++, 1, 0, hand.timers);
    const a = {};
    clock.join(a);
    for (let i = 0; i < 3; i++) {
        const waits = clock.tick(a);
        expect(clock.pending).toBe(false);
        await waits;
    }
    expect([ticks, hand.count]).toEqual([3, 0]);
});

test('a tick made outside a paced clock (a demo world\'s idle ticks) counts: the first clock tick waits paceMs after it', async () => {
    let ticks = 0;
    const hand = handTimers();
    const clock = new SharedClock(() => ticks++, 1, 100, hand.timers);
    const a = {};
    clock.join(a);
    hand.set(1000);
    clock.markTick(); // the idle loop's last tick, at 1000 ms
    hand.set(1040);
    const waits = clock.tick(a);
    expect([ticks, clock.pending]).toEqual([0, true]);
    hand.fire(59);
    expect(ticks).toBe(0);
    hand.fire(1);
    await waits;
    expect(ticks).toBe(1);
    const unpaced = new SharedClock(() => ticks++, 1, 0, hand.timers);
    unpaced.join(a);
    unpaced.markTick();
    await unpaced.tick(a);
    expect(ticks).toBe(2); // an unpaced clock keeps no time
});
