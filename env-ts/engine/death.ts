/** Event capture for the engine Player, which has no isDead property.
 * Current vendor damage writes HP in applyDamage; script stat reductions call
 * changeStat immediately after writing levels. Observe both synchronously so a
 * heal/respawn before the next environment observation cannot erase a death.
 */
export interface HealthSource {
    levels: ArrayLike<number>;
    applyDamage(damage: number, type: number): void;
    changeStat(stat: number): void;
}

export interface DeathMonitor {
    readonly dead: boolean;
    readonly deathTick: number | null;
}

const monitors = new WeakMap<HealthSource, DeathMonitor>();

/** onDeath runs once, inside the damage or stat change that ends the life (its attacker still targets the player). */
export function monitorDeath(player: HealthSource, hpIndex: number, tick: () => number,
                             onDeath?: () => void): DeathMonitor {
    const existing = monitors.get(player);
    if (existing) return existing;
    let deathTick: number | null = null;
    const sample = () => {
        if (deathTick === null && player.levels[hpIndex] <= 0) {
            deathTick = tick();
            onDeath?.();
        }
    };
    const damage = player.applyDamage;
    player.applyDamage = function (amount, type) {
        sample();
        try { return damage.call(this, amount, type); }
        finally { sample(); }
    };
    const change = player.changeStat;
    player.changeStat = function (stat) {
        sample();
        try { return change.call(this, stat); }
        finally { sample(); }
    };
    const monitor: DeathMonitor = {
        get dead() { sample(); return deathTick !== null; },
        get deathTick() { sample(); return deathTick; }
    };
    monitors.set(player, monitor);
    return monitor;
}
