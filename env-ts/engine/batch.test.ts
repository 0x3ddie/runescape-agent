import { test, expect } from 'bun:test';
import { SharedClock } from '../core/lockstep-clock';
import { existsSync } from 'node:fs';

// batch.ts needs the engine checkout in vendor/rs-sdk (vendor-patches/README.md); without it this file's tests skip.
const engine = existsSync(new URL('../../vendor/rs-sdk', import.meta.url));
const { Controller, FACT_FIELDS, FatalError, ProtocolError, SharedWorld, foodCount, heldKinds, offersEat, places,
    tookFrom } = engine ? await import('./batch') : ({} as typeof import('./batch'));
const { installXpLedger } = engine ? await import('./xp-ledger') : ({} as typeof import('./xp-ledger'));
import { kindPage } from '../core/items';

const MASK = new Array(96).fill(1);

// A stand-in agent on the real shared clock: a reset takes 2 ticks, a step `stepTicks` ticks.
// Each step grants its slot 10 XP in stat 0 through the players' addXp, which the real ledger follows.
class FakePlayer { stats = [0]; addXp(stat: number, xp: number) { this.stats[stat] += xp; } }

function world(capacity: number, stepTicks: (slot: number) => number, paceMs = 0) {
    let tick = 0;
    const agents: any[] = [], inWorldAtTicks: boolean[][] = [];
    const clock = new SharedClock(() => { tick++; inWorldAtTicks.push(agents.map(a => !a.left)); }, capacity, paceMs);
    const host = {
        capacity, clock, tick: () => tick, loginTicks: 4, facts: {},
        actions: { style: 72, held0: 29, count: 96 },
        ledger: installXpLedger({ playerPrototype: FakePlayer.prototype, tick: () => tick, stats: 1 }),
        makeAgent(slot: number) {
            const agent: any = {
                player: new FakePlayer(), dead: false, left: false,
                async waitWorldTicks(n: number) { for (let i = 0; i < n; i++) await clock.tick(agent); },
                async reset() { agent.player = new FakePlayer(); await agent.waitWorldTicks(2); return { mask: MASK, decision: 0 }; },
                async step(action: number) {
                    await agent.waitWorldTicks(stepTicks(slot));
                    agent.player.addXp(0, 10);
                    return { mask: MASK, action, done: action === 0 && agent.dead };
                },
                leave() { agent.left = true; },
                pay: () => ({ xp: [...agent.player.stats], base: [] }), facts: () => ({}),
                armed: [] as number[], armHurt() { agent.armed.push(tick); }, problem: null as string | null,
                declaredStartProblem: () => agent.problem, wieldSlot: () => 0
            };
            agents[slot] = agent;
            clock.join(agent);
            return agent;
        }
    };
    const shared = new SharedWorld(host);
    const ask = (req: any) => shared.handle(req);
    const reset = (slot: number) => ({ op: 'reset', slot, seed: 1, cfg: {} });
    const step = (slot: number, action = 5) => ({ op: 'step', slot, action });
    // the slots of a reply in slot order (a reply promises no order)
    const due = (reply: any) => reply.results.map((r: any) => [r.slot, r.kind, r.tick, r.ticks]).sort((a: any, b: any) => a[0] - b[0]);
    return { shared, ask, reset, step, due, now: () => tick, agents, inWorldAtTicks };
}

test.skipIf(!engine)('a reply holds every agent due at the first stop; an agent inside its item never blocks the clock', async () => {
    const { ask, reset, step, due } = world(4, slot => [2, 5, 3][slot]);
    await expect(ask({ cmd: 'batch', items: [] })).rejects.toThrow('hello first');
    await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1' });
    // first joins: 4 login ticks and a 2-tick reset each
    expect(due(await ask({ cmd: 'batch', items: [reset(0), reset(1)] }))).toEqual([[0, 'reset', 6, 6], [1, 'reset', 6, 6]]);
    expect(due(await ask({ cmd: 'batch', items: [step(0), step(1)] }))).toEqual([[0, 'step', 8, 2]]);
    await expect(ask({ cmd: 'batch', items: [step(0), step(1)] })).rejects.toThrow('inside its item');
    await expect(ask({ cmd: 'batch', items: [] })).rejects.toThrow('due slots without an item: 0');
    // slot 2 joins while slot 1 is inside its step: slot 1 still ends after its own 5 ticks, at 11
    expect(due(await ask({ cmd: 'batch', items: [step(0), reset(2)] }))).toEqual([[0, 'step', 10, 2]]);
    expect(due(await ask({ cmd: 'batch', items: [step(0)] }))).toEqual([[1, 'step', 11, 5]]);
    expect(due(await ask({ cmd: 'batch', items: [step(1)] }))).toEqual([[0, 'step', 12, 2]]);
    // slot 0 resets between two stops; slot 2's join (from tick 8) ends at 14 and slot 0's reset at 14
    const r = await ask({ cmd: 'batch', items: [reset(0)] });
    expect(due(r)).toEqual([[0, 'reset', 14, 2], [2, 'reset', 14, 6]]);
    const of = (slot: number) => r.results.find((x: any) => x.slot === slot);
    // the life it closed: started at its reset's tick 6; the stand-in grants 10 XP at the end of each step (8, 10, 12)
    expect(of(0).ledger).toEqual({ start: 6, end: 12, grants: [[0, 10, 8], [0, 10, 10], [0, 10, 12]], total: [30] });
    expect(of(0).pay.ledger).toEqual([0]); // a new life's running total
    expect(of(2).ledger).toBeNull(); // a first join closes no life
});

test.skipIf(!engine)('refused requests start nothing, item errors come back due, and a leave returns the closed life', async () => {
    const { ask, reset, step, due, now } = world(2, () => 2);
    await expect(ask({ cmd: 'hello', protocol: 'other' })).rejects.toThrow('speaks shared-lockstep-v1');
    await expect(ask({ cmd: 'hello', protocol: 'shared-lockstep-v1', answerLimitMs: 10 })).rejects.toThrow('answerLimitMs');
    expect((await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1' })).answerLimitMs).toBe(600_000);
    const refused = async (items: any[], code: string) => {
        const before = now();
        await ask({ cmd: 'batch', items }).then(() => { throw new Error('accepted'); },
            (e: InstanceType<typeof ProtocolError>) => expect(e.code).toBe(code));
        expect(now()).toBe(before);
    };
    await refused([step(0)], 'not-joined');
    await refused([reset(5)], 'unknown-slot');
    await refused([reset(0), reset(0)], 'duplicate-slot');
    await refused([reset(0), reset(1), { op: 'reset', slot: 2, cfg: {} }], 'unknown-slot');
    await ask({ cmd: 'batch', items: [reset(0), reset(1)] });
    const bad = await ask({ cmd: 'batch', items: [step(0, 96), { op: 'leave', slot: 1 }] });
    expect(bad.results.map((r: any) => [r.slot, r.kind, r.code ?? null]).sort()).toEqual([[0, 'error', 'illegal-action'], [1, 'left', null]]);
    expect(bad.results.find((r: any) => r.slot === 1).ledger).not.toBeNull();
    expect(due(await ask({ cmd: 'batch', items: [step(0)] }))).toEqual([[0, 'step', 8, 2]]);
    expect((await ask({ cmd: 'ping' })).type).toBe('pong');
});

test.skipIf(!engine)('a step on a life that is over is refused as an item, and a reset gives it a new life', async () => {
    const { shared, ask, reset, step } = world(1, () => 2);
    await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1' });
    await ask({ cmd: 'batch', items: [reset(0)] });
    const agent: any = (shared as any).slots.get(0).agent;
    agent.dead = true;
    const died = await ask({ cmd: 'batch', items: [step(0, 0)] });
    expect([died.results[0].done, died.results[0].dead]).toEqual([true, true]);
    expect((await ask({ cmd: 'batch', items: [step(0)] })).results[0].code).toBe('life-over');
    agent.dead = false;
    expect((await ask({ cmd: 'batch', items: [reset(0)] })).results[0].kind).toBe('reset');
    expect(shared.held()).toBe(true);
});

test.skipIf(!engine)('a turn with no tick and nothing over is fatal, not a hang', async () => {
    const { shared, ask, reset } = world(1, () => 2);
    await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1' });
    await ask({ cmd: 'batch', items: [reset(0)] });
    const agent: any = (shared as any).slots.get(0).agent;
    agent.step = () => new Promise(() => {}); // waits on something other than the clock
    await expect(ask({ cmd: 'batch', items: [{ op: 'step', slot: 0, action: 5 }] })).rejects.toBeInstanceOf(FatalError);
});

test.skipIf(!engine)('the ledger counts the XP the engine applied, after the rate and the cap, for followed players only', () => {
    let tick = 7;
    class P { stats = [0, 1_999_999_990]; addXp(stat: number, xp: number) { this.stats[stat] = Math.min(this.stats[stat] + xp * 25, 2_000_000_000); } }
    const ledger = installXpLedger({ playerPrototype: P.prototype, tick: () => tick, stats: 2 });
    const a = new P(), b = new P();
    ledger.follow(a);
    a.addXp(0, 40);
    tick = 9;
    a.addXp(1, 10); // capped: 10 of 250 applied
    b.addXp(0, 40);
    expect(ledger.close(a)).toEqual({ start: 7, end: 9, grants: [[0, 1000, 7], [1, 10, 9]], total: [1000, 10] });
    expect(ledger.close(a)).toBeNull();
    expect(ledger.close(b)).toBeNull();
});

test.skipIf(!engine)('places deal tiles round-robin within the radius of each centre, every usable tile once before any twice', () => {
    const usable = (x: number, z: number) => (x + z) % 2 === 0;
    const tiles = places(6, 3, [[100, 100], [500, 500]], 1, usable);
    expect(tiles.every(t => usable(t.x, t.z))).toBe(true);
    tiles.forEach((t, k) => expect(Math.max(Math.abs(t.x - [100, 500][k % 2]), Math.abs(t.z - [100, 500][k % 2]))).toBeLessThanOrEqual(1));
    const first = tiles.filter((_, k) => k % 2 === 0).map(t => `${t.x},${t.z}`);
    expect(new Set(first).size).toBe(3); // a centre has 5 usable tiles in its 3x3: no repeat among 3
    expect(places(6, 3, [[100, 100], [500, 500]], 1, usable)).toEqual(tiles);
    expect(() => places(1, 1, [[1, 0]], 0, usable)).toThrow('no usable tile');
});

test.skipIf(!engine)('every step result carries the running ledger total, which the paid XP must equal', async () => {
    const { ask, reset, step } = world(1, () => 2);
    await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1' });
    const start = (await ask({ cmd: 'batch', items: [reset(0)] })).results[0].pay;
    for (let k = 1; k <= 3; k++) {
        const pay = (await ask({ cmd: 'batch', items: [step(0)] })).results[0].pay;
        expect(pay.ledger).toEqual([10 * k]);
        expect(pay.xp[0] - start.xp[0]).toBe(pay.ledger[0]);
    }
});

test.skipIf(!engine)('a leave takes the player out of the world before the clock can tick without it', async () => {
    const { ask, reset, step, agents, inWorldAtTicks } = world(2, slot => [2, 5][slot]);
    await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1' });
    await ask({ cmd: 'batch', items: [reset(0), reset(1)] });
    await ask({ cmd: 'batch', items: [step(0), step(1)] }); // slot 0 back at tick 8, slot 1 inside its step
    const before = inWorldAtTicks.length;
    const r = await ask({ cmd: 'batch', items: [{ op: 'leave', slot: 0 }] });
    expect(r.results.map((x: any) => [x.slot, x.kind]).sort()).toEqual([[0, 'left'], [1, 'step']]);
    expect(agents[0].left).toBe(true);
    // every tick after the leave ran with slot 0's player gone
    expect(inWorldAtTicks.slice(before).every(ticked => ticked[0] === false)).toBe(true);
    expect(inWorldAtTicks.length - before).toBe(3); // slot 1's step ends at 11
});

test.skipIf(!engine)('an item that throws in the engine comes back at once as an internal error, not a later stall', async () => {
    const { shared, ask, reset } = world(1, () => 2);
    await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1' });
    await ask({ cmd: 'batch', items: [reset(0)] });
    (shared as any).slots.get(0).agent.facts = () => { throw new Error('broken facts'); };
    const r = (await ask({ cmd: 'batch', items: [{ op: 'step', slot: 0, action: 5 }] })).results[0];
    expect([r.kind, r.code, r.message]).toEqual(['error', 'internal', 'broken facts']);
});

test.skipIf(!engine)('a bad request line still starts the answer limit, so silence after it stops the world', async () => {
    const { shared, ask, reset } = world(1, () => 2);
    await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1', answerLimitMs: 5000 });
    await ask({ cmd: 'batch', items: [reset(0)] }); // slot 0 is due: the world is held
    const sent: any[] = [], fatal: string[] = [], timers: Array<{ run: () => void; ms: number }> = [];
    const controller = new Controller(shared, { send: m => sent.push(m), end() {}, fatal: code => fatal.push(code),
        timers: { set: (run, ms) => { timers.push({ run, ms }); return timers.length; }, clear() {} } });
    await controller.line('{not json');
    expect(sent.at(-1)).toMatchObject({ type: 'error', code: 'bad-request' });
    expect(timers.map(t => t.ms)).toEqual([5000]);
    timers[0].run();
    expect(fatal).toEqual(['answer-limit']);
});

test.skipIf(!engine)('a paced demo world gives the replies of an unpaced one, a tick at most every paceMs, and no stall', async () => {
    const play = async (paceMs: number) => {
        const { ask, reset, step, due, now } = world(3, slot => [2, 5, 3][slot], paceMs);
        await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1' });
        const replies = [due(await ask({ cmd: 'batch', items: [reset(0), reset(1), reset(2)] }))];
        let pending = replies[0].map((r: any) => r[0]);
        while (now() < 24) {
            const reply = await ask({ cmd: 'batch', items: pending.map((slot: number) => step(slot)) });
            replies.push(due(reply));
            pending = reply.results.map((r: any) => r.slot);
        }
        return { replies, ticks: now() };
    };
    const plain = await play(0);
    const started = performance.now();
    const paced = await play(15);
    const elapsed = performance.now() - started;
    expect(paced).toEqual(plain);
    expect(elapsed).toBeGreaterThanOrEqual(15 * (paced.ticks - 1) - 5); // the first tick waits for none before it
});

test.skipIf(!engine)('in a paced world a leave that completes the waiting set sets the tick for its time, and the leaver is not in it', async () => {
    const { ask, reset, step, due, now, inWorldAtTicks } = world(3, slot => [2, 6, 6][slot], 15);
    await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1' });
    await ask({ cmd: 'batch', items: [reset(0), reset(1), reset(2)] });
    expect(due(await ask({ cmd: 'batch', items: [step(0), step(1), step(2)] }))).toEqual([[0, 'step', 8, 2]]);
    // slot 0 leaves while 1 and 2 wait inside their steps: the clock then holds a paced tick, which is no stall
    const started = performance.now(), before = now();
    const reply = await ask({ cmd: 'batch', items: [{ op: 'leave', slot: 0 }] });
    expect(due(reply)).toEqual([[0, 'left', 8, 0], [1, 'step', 12, 6], [2, 'step', 12, 6]]);
    expect(performance.now() - started).toBeGreaterThanOrEqual(15 * (now() - before) - 5);
    expect(inWorldAtTicks.slice(before).every(inWorld => inWorld[0] === false)).toBe(true);
});

test.skipIf(!engine)('facts.held is the distinct item ids of the pack and the worn items, ascending, empty slots left out', () => {
    // a kind held twice in the pack and once worn counts once; empty slots (null, undefined) are no kind
    const pack = [{ id: 1351 }, null, { id: 2309 }, { id: 2309 }, undefined, { id: 303 }];
    const worn = [null, null, null, { id: 1205 }, { id: 303 }];
    expect(heldKinds([pack, worn])).toEqual([303, 1205, 1351, 2309]);
    expect(heldKinds([[], [null]])).toEqual([]);
    expect(FACT_FIELDS).toContain('held');  // the hello names it (engine-env.ts factFields)
});

test.skipIf(!engine)('facts.food counts the pack items whose own options include Eat; a Drop or an Eat took an item by eats\' rule', () => {
    // the game's options, not names: bread and shrimps offer Eat; raw shrimps, logs and a drink do not; a stack counts
    // by its count; empty slots count nothing
    const iop: Record<number, (string | null)[]> = { 2309: ['Eat', null, null, null, 'Drop'], 315: ['Eat'],
        317: [null, null, null, null, 'Drop'], 1511: [], 1993: ['Drink'] };
    const pack = [{ id: 2309, count: 1 }, null, { id: 317, count: 1 }, { id: 315, count: 3 }, undefined,
        { id: 1511, count: 1 }, { id: 1993, count: 1 }, { id: 2309, count: 1 }];
    expect(foodCount(pack, id => iop[id])).toBe(5);
    expect(foodCount([null, { id: 9999, count: 1 }], id => iop[id])).toBe(0);  // an item with no options
    expect([offersEat(['Eat']), offersEat(['eat']), offersEat([null, 'Drop']), offersEat(undefined)])
        .toEqual([true, false, false, false]);
    const shrimps = { id: 315, count: 3 };
    expect([tookFrom(shrimps, null), tookFrom(shrimps, { id: 315, count: 2 }), tookFrom(shrimps, { id: 2309, count: 3 }),
            tookFrom(shrimps, { id: 315, count: 3 })]).toEqual([true, true, true, false]);
    // the hello names every field in the agent's order (engine-env.ts factFields)
    expect([...FACT_FIELDS]).toEqual(['hp', 'weapon', 'eats', 'death', 'held', 'food', 'food_drops', 'tool_drops', 'hurt']);
});

test.skipIf(!engine)('under item-v3 the eat and food-drop facts read the kind entry\'s own slot (the kind\'s first)', () => {
    // bread in slots 2 and 6: the page's bread entry is slot 2 with that slot's count; an Eat or a Drop of the entry
    // takes the item of slot 2, which tookFrom (the drop rule; the Eat rule is the same test) sees
    const page = kindPage([{ slot: 2, id: 2309, count: 1, name: 'Bread' }, { slot: 6, id: 2309, count: 1, name: 'Bread' }], 0, 28);
    const bread = page[0];
    expect([bread.slot, bread.count, bread.total]).toEqual([2, 1, 2]);
    expect(tookFrom(bread, null)).toBe(true);   // slot 2 now empty: the other bread in slot 6 is no matter
    expect(tookFrom(bread, { id: 2309, count: 1 })).toBe(false);   // nothing taken from slot 2
});

test.skipIf(!engine)('a reset arms the life\'s hurt count once, when its item ends after its setup; a failed reset arms nothing', async () => {
    const { ask, agents } = world(2, () => 3);
    await ask({ cmd: 'hello', protocol: 'shared-lockstep-v1' });
    const setup = (slot: number) => ({ op: 'reset', slot, seed: 1, cfg: {}, setup: { wield: 'bronze_sword' } });
    const r = await ask({ cmd: 'batch', items: [setup(0)] });
    // 4 login ticks, a 2-tick reset, then the style step and the Wield (3 ticks each): the count starts at tick 12
    expect(r.results[0]).toMatchObject({ kind: 'reset', tick: 12 });
    expect(agents[0].armed).toEqual([12]);
    agents[0].problem = 'declared inventory differs';
    const failed = await ask({ cmd: 'batch', items: [setup(0)] });
    expect(failed.results[0]).toMatchObject({ kind: 'error', code: 'declared-start' });
    expect(agents[0].armed).toEqual([12]);
});
