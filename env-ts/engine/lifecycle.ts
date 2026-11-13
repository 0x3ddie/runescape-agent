/** Opt-in observer of the installed Player.runScript death-queue lifecycle.
 * Does not heal, reset, dispatch actions, or change the legacy monitorDeath latch.
 * Install before the account can take damage; supply the exact ScriptProvider
 * object for [queue,player_death], not the similarly named death proc.
 */
import type { HealthSource } from './death';

export interface LifecycleScript {
    script: { name: string; id: number };
    self: unknown;
    _activePlayer: unknown;
    execution: number;
    fp: number;
    opcount: number;
}
export interface LifecyclePlayer extends HealthSource {
    runScript(script: any, protect?: boolean, force?: boolean): number;
}
export interface LifecycleSnapshot {
    phase: 'alive' | 'dying' | 'failed';
    deathSequence: number;
    respawnSequence: number;
    deathTick: number | null;
    respawnTick: number | null;
    completionSource: 'player_death_queue' | null;
    failure: string | null;
}
export interface LifecycleMonitor {
    snapshot(): Readonly<LifecycleSnapshot>;
    subscribe(listener: (event: Readonly<LifecycleEvent>) => void): () => void;
    /** Evaluator-only identity predicate; never included in lifecycle snapshots. */
    isPendingDeathScript?(script: object): boolean;
}
export interface LifecycleEvent extends LifecycleSnapshot {
    eventSequence: number;
    kind: 'death' | 'respawn';
}

// ScriptState constants in the installed vendor. runScript also returns -1 for
// denied protected access WITHOUT changing script.execution to ABORTED.
const ABORTED = -1, FINISHED = 1;
const installed = new WeakMap<LifecyclePlayer, { root: object; hpIndex: number; monitor: LifecycleMonitor }>();

export function monitorLifecycle(
    player: LifecyclePlayer,
    hpIndex: number,
    tick: () => number,
    deathQueueScript: { name: string; id: number }
): LifecycleMonitor {
    if (deathQueueScript?.name !== '[queue,player_death]' || !Number.isInteger(deathQueueScript.id))
        throw new Error('lifecycle requires exact player_death queue ScriptFile');
    if (!Number.isInteger(hpIndex) || hpIndex < 0 || hpIndex >= player.levels.length)
        throw new Error('invalid lifecycle HP index');
    const previous = installed.get(player);
    if (previous) {
        if (previous.root !== deathQueueScript || previous.hpIndex !== hpIndex)
            throw new Error('lifecycle already installed with another root or HP index');
        return previous.monitor;
    }
    if (!(player.levels[hpIndex] > 0)) throw new Error('install lifecycle before lethal damage, on a living player');
    const state: LifecycleSnapshot = {
        phase: 'alive', deathSequence: 0, respawnSequence: 0,
        deathTick: null, respawnTick: null, completionSource: null, failure: null
    };
    let pending: LifecycleScript | null = null;
    let eventSequence = 0;
    const listeners = new Set<(event: Readonly<LifecycleEvent>) => void>();
    const retired = new WeakSet<object>();
    let lastTick = -1;
    const fail = (reason: string) => {
        state.phase = 'failed';
        state.failure ??= reason;
    };
    const emit = (kind: LifecycleEvent['kind']) => {
        const event = Object.freeze({ ...state, kind, eventSequence: ++eventSequence });
        for (const listener of listeners) {
            try { listener(event); }
            catch { fail('lifecycle event listener failed'); }
        }
    };
    const now = (): number => {
        const value = tick();
        if (!Number.isInteger(value) || value < 0 || value < lastTick) {
            fail('invalid or regressing world tick');
            return lastTick;
        }
        lastTick = value;
        return value;
    };
    const sample = () => {
        const at = now();
        if (state.phase === 'alive' && player.levels[hpIndex] <= 0) {
            state.phase = 'dying';
            state.deathSequence++;
            state.deathTick = at;
            emit('death');
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
    const run = player.runScript;
    player.runScript = function (script: LifecycleScript, protect = false, force = false): number {
        sample();
        const rootEntry = script.script === deathQueueScript;
        const tracked = pending === script;
        if ((rootEntry || tracked) && state.phase !== 'failed') {
            if (this !== player || script.self !== player || script._activePlayer !== player) fail('death queue player identity changed');
            else if (retired.has(script)) fail('completed death queue executed again');
            else if (rootEntry && pending && !tracked) fail('overlapping death queue');
            else if (state.phase !== 'dying') fail('death queue without observed lethal event');
            else if (!pending) {
                if (script.fp !== 0 || script.opcount !== 0) fail('death queue first observed after execution began');
                else pending = script;
            }
        }
        let result: number;
        try { result = run.call(this, script, protect, force); }
        catch (error) {
            if (pending === script) fail('death queue threw');
            throw error;
        } finally { sample(); }
        if (pending !== script || state.phase === 'failed') return result;
        if (script.self !== player || script._activePlayer !== player) {
            fail('death queue player identity changed');
        } else if (script.execution === ABORTED) {
            fail('death queue aborted');
        } else if (result === FINISHED) {
            // A suspended proc changes script.script. Only the SAME ScriptState
            // returning all the way to its original queue is an acknowledgement.
            // Tail-jumped duel/special branches are deliberately unsupported.
            if (script.execution !== FINISHED || script.script !== deathQueueScript || script.fp !== 0)
                fail('death queue finished outside original root');
            else if (!(player.levels[hpIndex] > 0)) fail('death queue finished without positive HP');
            else {
                state.respawnSequence = state.deathSequence;
                state.respawnTick = now();
                if (state.phase !== 'failed') {
                    state.phase = 'alive';
                    state.completionSource = 'player_death_queue';
                    retired.add(script);
                    pending = null;
                    emit('respawn');
                }
            }
        }
        return result;
    };
    const monitor: LifecycleMonitor = {
        snapshot() { sample(); return Object.freeze({ ...state }); },
        isPendingDeathScript(script) { return state.phase === 'dying' && pending === script; },
        subscribe(listener) { listeners.add(listener); return () => { listeners.delete(listener); }; }
    };
    installed.set(player, { root: deathQueueScript, hpIndex, monitor });
    sample();
    return monitor;
}
