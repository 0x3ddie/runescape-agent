// env-ts/engine/hurt.ts: facts.hurt (built for the cost of time hurt health-cost-v1, since removed; the life rows'
// hurt_ticks read it): the time an armed life spends hurt, tick by tick. At each world tick of an armed life whose
// player is alive at the tick's end, base - min(hp, base) is added to the count of that tick's base (the hitpoints'
// base level then): the missing hitpoints, and hitpoints above the base (a boost) count 0. Integers only, so the sum is
// exact: the life's full-deficit ticks (the missing share of hitpoints, summed over the ticks) are the sum over the
// bases of n / base. Nothing counts before the arm (the end of the batch's reset item, after its setup) or after the
// life's end.

export class HurtTicks {
    private readonly byBase = new Map<number, number>();
    private armed = false;

    /** The life starts: its count starts at 0. */
    arm(): void {
        this.byBase.clear();
        this.armed = true;
    }

    /** The life is over (a reset begins, or the agent leaves): the count is [] and no tick counts until the next arm. */
    end(): void {
        this.byBase.clear();
        this.armed = false;
    }

    /** One world tick's end: the player's current hitpoints `hp` of base `base`, and whether its life goes on. */
    tick(hp: number, base: number, alive: boolean): void {
        if (!this.armed || !alive) return;
        if (!Number.isInteger(hp) || !Number.isInteger(base) || base < 1)
            throw new Error(`facts.hurt: hitpoints ${hp} of base ${base}`);
        this.byBase.set(base, (this.byBase.get(base) ?? 0) + base - Math.min(Math.max(hp, 0), base));
    }

    /** [[base, missing hitpoint ticks at that base], ...], bases ascending: the armed life's count so far. */
    facts(): number[][] {
        return [...this.byBase].sort((a, b) => a[0] - b[0]).map(([base, n]) => [base, n]);
    }
}
