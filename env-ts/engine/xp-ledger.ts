// env-ts/engine/xp-ledger.ts: the independent XP count of shared-lockstep-v1 (engine/batch.ts): every XP grant of a
// followed player, taken inside Player.addXp, the engine's only path that adds XP (script stat_advance; setLevel writes
// XP only at a reset, before a life starts). Each grant is the amount actually applied (the stat's XP after the call
// less before it: the XP rate and the 200M cap included), in tenths of an XP point, at the tick of the cycle that
// granted it. It reads nothing from the observation, so the learner can check the pay it took from step replies.

export interface LedgerHost {
    /** Player.prototype, whose addXp(stat, xp, allowMulti) adds XP */
    playerPrototype: { addXp(stat: number, xp: number, allowMulti?: boolean): unknown };
    tick(): number;
    stats: number;
}

export interface LifeLedger {
    start: number;
    end: number;
    grants: Array<[number, number, number]>;
    total: number[];
}

export function installXpLedger(host: LedgerHost) {
    const lives = new WeakMap<object, { start: number; grants: Array<[number, number, number]>; total: number[] }>();
    const add = host.playerPrototype.addXp;
    host.playerPrototype.addXp = function (this: { stats: ArrayLike<number> }, stat: number, ...rest: any[]) {
        const life = lives.get(this);
        if (!life) return (add as any).call(this, stat, ...rest);
        const before = this.stats[stat];
        try {
            return (add as any).call(this, stat, ...rest);
        } finally {
            const applied = this.stats[stat] - before;
            if (applied !== 0) {
                life.grants.push([stat, applied, host.tick()]);
                life.total[stat] += applied;
            }
        }
    };
    return {
        /** Follow `player` from now: its life starts at this tick. */
        follow(player: object): void {
            lives.set(player, { start: host.tick(), grants: [], total: new Array(host.stats).fill(0) });
        },
        /** The followed life's XP by stat so far (a copy, about 20 ns), or null when `player` is not followed. */
        running(player: object): number[] | null {
            return lives.get(player)?.total.slice() ?? null;
        },
        /** The followed life of `player`, ended now and no longer followed; null when it was not followed. */
        close(player: object): LifeLedger | null {
            const life = lives.get(player);
            if (!life) return null;
            lives.delete(player);
            return { start: life.start, end: host.tick(), grants: life.grants, total: life.total };
        }
    };
}
