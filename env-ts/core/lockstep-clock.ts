// env-ts/core/lockstep-clock.ts: the clock of a shared lockstep world (engine/batch.ts shared-lockstep-v1).
//
// The world advances one tick when every agent in it waits for one, so an agent whose result has come back (its
// controller deciding) holds the world for all, as the one agent of a lockstep world holds it. Each agent still counts
// its own ticks: an item that waits n ticks resolves after n advances, whatever the others ask. A first join's login
// waits on this clock too (engine/batch.ts), so a join at any time leaves every other agent's count right.
//
// A paced clock (a demo world: scripts/stack.py --demo-spectator NAME --demo-pace-ms MS with RL_SHARED_AGENTS) ticks
// at most once every paceMs of wall time, so a watching client sees the world at a viewable pace: when every member
// waits sooner than that after the last tick, the tick is set for its time (`pending`) and comes then if every member
// still waits. Only the wall time between ticks changes; which ticks each agent sees, and in what order, does not.

/** The wall clock a paced clock reads and sets its next tick on (a test passes its own). */
export interface PaceTimers { now(): number; later(run: () => void, ms: number): void }

const WALL: PaceTimers = { now: () => performance.now(), later: (run, ms) => { setTimeout(run, ms); } };

export class SharedClock {
    private readonly members = new Set<object>();
    private waiting: { member: object; resolve: () => void }[] = [];
    private lastTickAt = -Infinity;
    private scheduled = false;
    advanced = 0;

    constructor(private readonly advance: () => void, readonly capacity: number, readonly paceMs = 0,
                private readonly timers: PaceTimers = WALL) {
        if (!Number.isInteger(capacity) || capacity < 1) throw new Error('a shared clock serves at least one agent');
        if (!Number.isInteger(paceMs) || paceMs < 0) throw new Error('a clock\'s pace is a whole number of ms, 0 for none');
    }

    get size(): number {
        return this.members.size;
    }

    get waitingCount(): number {
        return this.waiting.length;
    }

    /** A tick the world made outside this clock (a demo world's idle ticks, core/spectator.ts idleTicks): a paced
     * clock's next tick comes no sooner than paceMs after it. */
    markTick(): void {
        if (this.paceMs > 0) this.lastTickAt = this.timers.now();
    }

    /** A paced tick is set to come: the world will advance with no further wait or result (engine/batch.ts). */
    get pending(): boolean {
        return this.scheduled;
    }

    join(member: object): void {
        if (this.members.size >= this.capacity) throw new Error(`a lockstep world serves ${this.capacity} agents`);
        this.members.add(member);
    }

    /** A member that leaves no longer holds the world; its own wait, if any, ends without a tick. */
    leave(member: object): void {
        if (!this.members.delete(member)) return;
        const own = this.waiting.filter(w => w.member === member);
        this.waiting = this.waiting.filter(w => w.member !== member);
        for (const w of own) w.resolve();
        this.release();
    }

    /** Resolves once the world has advanced one tick with every member waiting. */
    tick(member: object): Promise<void> {
        if (!this.members.has(member)) return Promise.reject(new Error('only a member waits for a tick'));
        if (this.waiting.some(w => w.member === member)) return Promise.reject(new Error('a member waits once at a time'));
        return new Promise(resolve => {
            this.waiting.push({ member, resolve });
            this.release();
        });
    }

    private release(): void {
        if (this.scheduled || this.waiting.length === 0 || this.waiting.length < this.members.size) return;
        if (this.paceMs > 0) {
            const wait = this.lastTickAt + this.paceMs - this.timers.now();
            if (wait > 0) {
                this.scheduled = true;
                this.timers.later(() => { this.scheduled = false; this.release(); }, wait);
                return;
            }
            this.lastTickAt = this.timers.now();
        }
        this.advance();
        this.advanced++;
        const ready = this.waiting;
        this.waiting = [];
        for (const w of ready) w.resolve();
    }
}
