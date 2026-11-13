/** Test-fixture safety checks. No policy inputs. */
export interface FixtureIdleState {
    hp: number;
    isDead: boolean;
    loggingOut: boolean;
    activeScript: unknown;
    target: unknown;
    hasWaypoints: boolean;
    bankOpen: boolean;
    delayed: boolean;
    protect: boolean;
}

export function assertFixtureIdle(state: FixtureIdleState): void {
    if (!state || !Number.isFinite(state.hp) || state.hp <= 0 || state.isDead !== false)
        throw new Error('fixture requires a living player with positive HP');
    for (const flag of ['loggingOut', 'hasWaypoints', 'bankOpen', 'delayed', 'protect'] as const) {
        if (state[flag] !== false) throw new Error(`fixture unsafe: ${flag}`);
    }
    if (state.activeScript !== null) throw new Error('fixture unsafe: activeScript');
    if (state.target !== null) throw new Error('fixture unsafe: target');
}
