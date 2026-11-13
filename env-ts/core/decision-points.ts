/**
 * decisionPoints (engine only): while an option runs and nothing happens, the
 * environment keeps waiting in paced chunks instead of asking the policy. About 90%
 * of decisions were "wait while the option runs"; they diluted credit, and a sampled
 * action there broke the option it interrupted. Ticks are charged exactly as before.
 */
export const DECISION_WAIT_TICKS = 100; // the policy still observes at least this often

export interface WaitState {
    busy: boolean;       // an option is still running
    dead: boolean;
    hp: number;
    hpAtDecision: number; // a hit hands control back at once
    dialog: boolean;      // a dialog or modal needs an answer
    waited: number;       // ticks already waited past the decision's own chunk
}

export function keepWaiting(s: WaitState): boolean {
    return s.busy && !s.dead && !s.dialog && s.hp >= s.hpAtDecision && s.waited < DECISION_WAIT_TICKS;
}
