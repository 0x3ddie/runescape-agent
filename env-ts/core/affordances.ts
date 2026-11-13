/**
 * act-v3: what a player could click, published as slots the policy points at.
 *
 * An affordance is one option of one nearby thing: a loc, an npc or a ground item
 * ("Tree: Chop down", "Banker: Bank", "Fishing spot: Net", "Goblin: Attack", "Logs:
 * Take"). The game's own option lists decide what is offered, so no row of the action
 * table belongs to a skill; the policy learns which affordance pays. Held items offer
 * their inventory options the same way ("Bronze axe: Wield", "Bread: Eat").
 *
 * Ordering is spatial only, never by level, experience or value: walking steps, then
 * position. Every kind of affordance is offered once before a second instance of one
 * already listed, so twelve doors and men near a bank cannot push the only tree out
 * (at Draynor nearer npcs and crates once pushed the bank booth out of the places).
 */
import { A_ACT0, A_HELD0, A_USE0, N_ACT, N_HELD, N_INV_SEL } from './actions';
import type { Item } from './items';

export const ACT_CONTRACT = 'act-v3';
/** The walk map's radius (env-ts/reach.ts walkMap) the affordances' steps come from: act-v2's PLACE_WALK_TILES. */
export const ACT_WALK_TILES = 48;
/** Chebyshev radius of the client's loc and ground-item scan (engine/observe.ts). */
export const ACT_SCAN_TILES = 15;
/** Same-kind things this close to an affordance are counted with it (a stand of trees). */
export const ACT_CLUSTER_TILES = 5;
/** A ground item's option: the game draws "Take" from the item's ground op 3. */
export const TAKE = 'Take';
/** The option every affordance offers while an item is picked for use. */
export const USE = 'Use';

export type ActKind = 'loc' | 'npc' | 'obj';
export type ActCandidate = { kind: ActKind; id: number; name: string; options: string[]; x: number; z: number;
    level: number; combatLevel?: number };
export type Affordance = { kind: ActKind; id: number; name: string; option: string; x: number; z: number;
    level: number; dx: number; dz: number; distance: number; steps: number; count: number; combatLevel: number };
export type HeldAffordance = { slot: number; id: number; name: string; count: number; option: string };

export function checkActContract(cfg: { contract?: string; actTargets?: string }): void {
    if (cfg.actTargets !== ACT_CONTRACT || cfg.contract !== 'item-v2')
        throw new Error('the task requires item-v2 and act-v3 affordances');
}

const chebyshev = (a: { x: number; z: number }, b: { x: number; z: number }) =>
    Math.max(Math.abs(a.x - b.x), Math.abs(a.z - b.z));

export const affordanceKey = (a: { kind: string; id: number; x: number; z: number; level: number; option: string }) =>
    `${a.kind}:${a.id}:${a.x}:${a.z}:${a.level}:${a.option.toLowerCase()}`;

/**
 * `steps(candidate)` is the walking distance to stand where the option can be used,
 * or -1 when the player cannot get there (env-ts/reach.ts: a loc from a side with no
 * wall between, an npc or item from beside it). `refused` leaves out an affordance
 * the game turned away a moment ago.
 */
export function selectAffordances<T extends ActCandidate>(candidates: T[], player: { x: number; z: number; level: number },
                                                        steps: (c: T) => number, refused?: (a: Affordance) => boolean):
                                                        { affordance: Affordance; candidate: T }[] {
    const all: Affordance[] = [];
    const source = new Map<Affordance, T>();
    const seen = new Set<string>();
    for (const c of candidates) {
        if (![c.id, c.x, c.z, c.level].every(Number.isInteger) || c.id < 0) throw new Error('invalid affordance candidate');
        if (c.level !== player.level || !c.name || chebyshev(c, player) > ACT_SCAN_TILES) continue;
        const options = [...new Set(c.options.filter(o => typeof o === 'string' && o.length > 0))];
        if (!options.length) continue;
        const walk = steps(c);
        if (walk < 0) continue;
        for (const option of options) {
            // Only public fields: the candidate may hold engine objects.
            const a: Affordance = { kind: c.kind, id: c.id, name: c.name, option, x: c.x, z: c.z, level: c.level,
                dx: c.x - player.x, dz: c.z - player.z, distance: chebyshev(c, player), steps: walk, count: 1,
                combatLevel: c.combatLevel ?? 0 };
            const key = affordanceKey(a);
            if (seen.has(key) || refused?.(a)) continue;
            seen.add(key);
            all.push(a);
            source.set(a, c);
        }
    }
    const group = (a: Affordance) => `${a.kind}:${a.name.toLowerCase()}:${a.option.toLowerCase()}`;
    for (const a of all) a.count = all.filter(b => group(b) === group(a) && chebyshev(a, b) <= ACT_CLUSTER_TILES).length;
    const order = (a: Affordance, b: Affordance) => a.steps - b.steps || a.x - b.x || a.z - b.z ||
        (a.kind < b.kind ? -1 : a.kind > b.kind ? 1 : 0) || a.id - b.id ||
        (a.option < b.option ? -1 : a.option > b.option ? 1 : 0);
    all.sort(order);
    const covered = new Set<string>(), chosen = new Set<Affordance>();
    for (const a of all) {
        if (chosen.size === N_ACT) break;
        if (!covered.has(group(a))) { covered.add(group(a)); chosen.add(a); }
    }
    for (const a of all) if (chosen.size < N_ACT) chosen.add(a);
    return [...chosen].sort(order).map(a => ({ affordance: a, candidate: source.get(a)! }));
}

/**
 * Held affordances: each option of each item on the visible inventory page, in slot
 * then option order. Drop has its own rows (DROP_k), which keep the tool protection.
 */
export function selectHeld(page: Item[], using?: { slot: number } | null): HeldAffordance[] {
    const held: HeldAffordance[] = [];
    for (const item of page) {
        if (!(item.count > 0) || item.slot === using?.slot) continue;
        // With an item picked for use, every other held item is a target for it.
        for (const option of new Set<string>(using ? [USE] : item.options ?? [])) {
            if (held.length === N_HELD) return held;
            if (!option || /^drop$/i.test(option)) continue;
            held.push({ slot: item.slot, id: item.id, name: item.name, count: item.count, option });
        }
    }
    return held;
}

export function actAction(action: number): boolean {
    return Number.isInteger(action) && action >= A_ACT0 && action < A_ACT0 + N_ACT;
}

export function heldAction(action: number): boolean {
    return Number.isInteger(action) && action >= A_HELD0 && action < A_HELD0 + N_HELD;
}

/**
 * Publish actSel and heldSel; returns the candidate behind each actSel slot. With an
 * item picked for use (obs.using), every nearby loc and npc offers "Use" instead of its
 * own options: the game, not this list, decides what an item does to it.
 */
export function publishAffordances<T extends ActCandidate>(obs: Record<string, any>, candidates: T[],
                                                         steps: (c: T) => number, refused?: (a: Affordance) => boolean): T[] {
    const sources = obs.using ? candidates.filter(c => c.kind !== 'obj') : candidates;
    const offered = obs.using ? sources.map(c => ({ ...c, options: [USE] })) : sources;
    const chosen = selectAffordances(offered, obs.player, steps, refused);
    obs.actTargetVersion = ACT_CONTRACT;
    obs.actSel = chosen.map(c => c.affordance);
    obs.heldSel = selectHeld(obs.invSel, obs.using);
    return chosen.map(c => sources[offered.indexOf(c.candidate)]);
}

export function useAction(action: number): boolean {
    return Number.isInteger(action) && action >= A_USE0 && action < A_USE0 + N_INV_SEL;
}
