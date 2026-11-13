// env-ts/core/budgets.ts: task configuration and episode/option budgets.
//
// These numbers define what an episode IS.

export const K_TICKS = 2; // decision cadence: one decision per 2 game ticks
export const MAX_DECISIONS = 1500; // truncate on LEARNER steps (protocol: done=true, truncated=true)
// Truncate on GAME TIME too. One decision can span a whole walk, so
// a decision budget alone no longer bounds how long an episode runs; 20k ticks is
// ~400s of game time at speedMs 20, comfortably enough to tour several sites.
export const MAX_EPISODE_TICKS = 20000;

export interface TaskCfg {
    contract?: 'item-v2'; // required by reset: physical inventory and bank pages (core/items.ts)
    actTargets?: 'act-v3'; // required by reset: what a player could click, and the clicks (core/affordances.ts, core/clicks.ts)
    lifecycle?: 'episodic-v1' | 'continuous-v1'; // experimental account continuity
    maxRespawnTicks?: number; // authoritative wait bound; failure is not a reset
    allowedActions?: number[]; // whitelist into mask (relevant for early curriculum stages)
    spawnTele?: string; // e.g. "0,49,50,59,20": teleported via ::tele after fresh reset
    spawnSteps?: [number, number]; // engine only: start lo..hi walking steps from spawnTele, picked by seed
    decisionPoints?: boolean; // engine only: ask the policy only when an option ends or something happens
    seedCoins?: number; // ::give coins N after fresh reset (toll-gate stages)
    seedItems?: string; // "name:count,name:count" via ::give (spawn curriculum)
    kit?: 'starter' | 'none'; // engine only: 'none' grants no starter kit and banks nothing; seedItems is the inventory
    setStats?: string; // "mining:30,fishing:25" via ::setstat (start-state curriculum)
    persistLevels?: boolean; // skip ::minme so levels carry across episodes
    maxDecisions?: number; // truncate after this many LEARNER steps (default MAX_DECISIONS)
    maxEpisodeTicks?: number; // truncate after this much GAME TIME (default MAX_EPISODE_TICKS)
    speedMs?: number; // world ms per tick; tickBudgetMs converts game-tick budgets
    paceJitterTicks?: number; // engine only: 0..N extra ticks per decision, seeded (see engine/agent.ts)
}

// Option budgets are declared in GAME TICKS, not milliseconds. They describe game
// events ("this walk has not moved for N ticks", "this interaction has had N
// ticks to finish"), so a change to the world speed must not change their meaning.
// They were wall-clock constants tuned at 20ms/tick; at 8ms/tick the same 1200 ms
// silently became 150 ticks of patience instead of 60, and at 5ms it would be 240.
// tickBudgetMs() converts at the CURRENT world speed, so the game-time semantics
// hold at any speed.
export const OPTION_HARD_TIMEOUT_TICKS = 1500; // was 30000 ms at 20 ms/tick
export const NAV_STALL_TICKS = 60; // was 1200 ms at 20 ms/tick
// An ACT option that changes nothing this long (no step, experience, item or interface) has
// stalled: a banker behind Falador's counter held one for the 1,500-tick hard timeout.
export const ACT_STALL_TICKS = 100;
export const RESET_VERIFY_TICKS = 1250; // was 25000 ms
export const RESET_ALIVE_TICKS = 1000; // was 20000 ms
export const RESET_STILL_TICKS = 2250; // was 45000 ms; quiescence is a game-time budget

// Spawn jitter is +/-3, so anything past this means the teleport did not land.
export const SPAWN_TOLERANCE_TILES = 4;

/** Convert a game-tick budget to wall-clock ms at the current world speed. */
export function tickBudgetMs(ticks: number, speedMs?: number): number {
    return ticks * (speedMs ?? 20);
}

export interface EpisodeCounters {
    /** steps where the agent HAD a choice (bounded by maxDecisions) */
    charged: number;
    /** game ticks this episode (bounded by maxEpisodeTicks) */
    episodeTicks: number;
}

// Two independent budgets: an episode of many cheap decisions and an episode of a
// few very long options both need a stop, and they are not the same quantity.
//
// maxDecisions charges only steps that offered a real choice. An interface option
// once held the mask at WAIT-only for ~8-20 decisions while it ran, so the count of
// forced steps was set by how often the policy banked, not by anything it decided. Measured on v21f: ~21 bank
// interactions x ~18 forced steps = ~374 of a 400 budget, leaving ~26 real
// choices per episode: far too few for the fish->bank->trees composition, and it
// looked exactly like a policy that would not learn. episodeTicks still bounds
// game time, so an episode cannot run away.
export function isTruncated(counters: EpisodeCounters, cfg: TaskCfg): boolean {
    return counters.charged >= (cfg.maxDecisions ?? MAX_DECISIONS) || counters.episodeTicks >= (cfg.maxEpisodeTicks ?? MAX_EPISODE_TICKS);
}
