import { expect, test } from 'bun:test';
import { monitorDeath, type HealthSource } from './death';

const HP = 3;
function player(): HealthSource {
    return {
        levels: Uint8Array.from([1, 1, 1, 10]),
        applyDamage(amount) {
            (this.levels as Uint8Array)[HP] = Math.max(0, this.levels[HP] - amount);
        },
        changeStat() {}
    };
}

test('source has no isDead: lethal damage remains latched after healing before observation', () => {
    const p = player();
    let tick = 12;
    const monitor = monitorDeath(p, HP, () => tick);
    expect('isDead' in p).toBe(false);
    p.applyDamage(10, 1);
    (p.levels as Uint8Array)[HP] = 10;
    tick = 17;
    expect(monitor.dead).toBe(true);
    expect(monitor.deathTick).toBe(12);
});

test('stat reductions latch before a change handler can heal the player', () => {
    const p = player();
    p.changeStat = function () { (this.levels as Uint8Array)[HP] = 10; };
    const monitor = monitorDeath(p, HP, () => 9);
    (p.levels as Uint8Array)[HP] = 0;
    p.changeStat(HP);
    expect(p.levels[HP]).toBe(10);
    expect(monitor.dead).toBe(true);
    expect(monitor.deathTick).toBe(9);
});

test('nonlethal damage and unrelated stat changes do not latch death', () => {
    const p = player();
    const monitor = monitorDeath(p, HP, () => 8);
    p.applyDamage(2, 1);
    p.changeStat(0);
    expect(p.levels[HP]).toBe(8);
    expect(monitor.dead).toBe(false);
    expect(monitor.deathTick).toBeNull();
    expect(monitorDeath(p, HP, () => 9)).toBe(monitor);
});

test('a replacement player has an independent latch and old callbacks cannot contaminate it', () => {
    const old = player();
    const previous = monitorDeath(old, HP, () => 1);
    old.applyDamage(10, 1);
    const fresh = player();
    const current = monitorDeath(fresh, HP, () => 2);
    old.applyDamage(1, 1);
    expect(previous.dead).toBe(true);
    expect(current.dead).toBe(false);
    expect(current.deathTick).toBeNull();
});

test('direct observed zero and zero before damage are retained even when a handler heals', () => {
    const p = player();
    const monitor = monitorDeath(p, HP, () => 3);
    (p.levels as Uint8Array)[HP] = 0;
    expect(monitor.dead).toBe(true);
    (p.levels as Uint8Array)[HP] = 10;
    expect(monitor.dead).toBe(true);
    const q = player();
    q.applyDamage = function () { (this.levels as Uint8Array)[HP] = 10; };
    const other = monitorDeath(q, HP, () => 4);
    (q.levels as Uint8Array)[HP] = 0;
    q.applyDamage(0, 0);
    expect(other.dead).toBe(true);
});
