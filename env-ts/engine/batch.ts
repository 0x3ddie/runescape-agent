// env-ts/engine/batch.ts: shared-lockstep-v1: many agents in one lockstep world, all stepped by one controller.
// This block is the protocol's source of truth for the learner; the code below implements it (engine-env.ts serves it).
//
// MODE. A world speaks shared-lockstep-v1 when its engine starts with RL_LOCKSTEP=1 (scripts/stack.py --lockstep) and
// RL_SHARED_AGENTS=N (1..2046: its capacity, the engine's player slots 1..2046). It replaces the prototype switch
// RL_LOCKSTEP_AGENTS. Without RL_SHARED_AGENTS a world is today's one-agent world, unchanged. In this mode the engine
// saves no player: no autosave, no save at a leave (no data/players/<profile> folder for the boot).
//
// TRANSPORT. TCP 127.0.0.1:RL_ENV_PORT. Requests are JSON lines; each engine message is 'PROTO|' + JSON + '\n'. One
// controller connection a world at a time (a second is refused with error 'controller-present'); its requests are
// handled in order, one reply each. A reply carries about 9 KB an observation.
//
// 1. ready (engine, on connect): {type:'ready', protocol:'shared-lockstep-v1', capacity, worldId}. No agent yet.
// 2. hello {cmd:'hello', protocol:'shared-lockstep-v1', answerLimitMs?}
//    -> {type:'hello', protocol, capacity, worldId, tick, xpRate, randomEvents, kTicks, skills, factFields, answerLimitMs}
//    xpRate: the engine's own XP rate (Environment.node.xpRate); randomEvents: whether random events are on in this
//    engine (its resolved setting Environment.NODE_RANDOM_EVENTS, which content reads through map_randomevents;
//    default off). Both are the engine's values, not the caller's environment (engineSettings).
//    skills: the engine's 21 stat names in stat order (the order of every pay and ledger array; indices 18 and 19
//    are unused in this content revision). factFields: the fields of every result's facts (FACT_FIELDS; the engine's
//    agent names them, so a learner that needs `hurt` refuses a world without it). Required first; answerLimitMs in
//    1,000..3,600,000, default 600,000
//    (ten minutes; ANSWER LIMIT).
// 3. places {cmd:'places', count, seed, centres?: [[x, z], ...], radius?}
//    -> {type:'places', tiles: [{x, z}, ...]}: start tiles on level 0, each walkable with a connected jitter tile
//    (spawn_tiles.ts). With centres: within `radius` tiles (the larger of |dx| and |dz|) of a centre, dealt round-robin
//    over the centres; without: the mainland test grid. The same seed gives the same tiles. No tick passes.
// 4. batch {cmd:'batch', items: [ITEM, ...]} -> {type:'batch', tick, results: [RESULT, ...]}
//    ITEM (at most one a slot; slot: 0..capacity-1, the learner's own index; the engine names the player w<world>s<slot>):
//      {op:'reset', slot, seed, cfg, setup}  start a life: a slot's first join, a new life after a death or after the
//                                            learner ends one (its drawn length); any time, no world restart
//      {op:'step', slot, action}             one decision: action 0..95, legal in the slot's last mask
//      {op:'leave', slot}                    the agent leaves the world (no save); the slot may join again later
//    THE STOP RULE. A slot is due when its last item's result has come back. A request carries items for exactly the
//    slots that are due (every slot the last reply returned, and any slot not in the world that joins now), and for no
//    slot inside an item; else the whole request is refused. The engine starts every item, then ticks the world while
//    every agent in it waits inside its item, and replies at the first tick where at least one agent's item is over;
//    the reply holds every agent whose item is over at that tick (every 'leave' at once). An agent inside its item is
//    not due and never blocks the clock; it keeps counting its own ticks. Agents reach their decision points at
//    different ticks, so a reply holds any subset of the world's agents, and the learner answers each when it is due.
//    Between a reply and the next request the world stands still (no tick), for as long as the learner takes.
//    RESULT: {slot, kind: 'reset' | 'step' | 'left' | 'error', tick, ticks, obs, done, dead, pay, facts, ledger,
//             code, message}
//      tick    World.currentTick at the result: the world's game tick, the same for every agent of a reply.
//      ticks   world ticks this item used: a step's are its decision's (K_TICKS, pace jitter, decision-point waits);
//              a reset's are its login (4, first join only), its settle (2 + 2) and its setup steps. Not in a life.
//      obs     the agent's observation, exactly today's reset or step reply (env-ts/engine/observe.ts buildObs, mask,
//              feedback, timing). The setup's two steps count in its decision and charged counters (and the agent's
//              episode ticks), as they did in today's watch: a life's first step is decision 3 after a Wield, 2 without.
//      done    the life is over: a death, or one of the cfg's own limits. The slot's next item is reset or leave.
//      dead    the life ended in a death (facts.death has its tick).
//      pay     {xp: [21], base: [21], ledger: [21] | null}: the engine's own XP of every stat (Player.stats, tenths of
//              an XP point) and the base levels, read from the player at `tick`, not from the observation; the learner
//              pays from these. ledger: the life's running ledger total, XP by stat granted since the life started
//              (the count its closing `ledger` ends with; zeros in a reset result), so the learner can check its pay
//              at every reply; null when no life is followed (a left or error result). It costs a copy of 21 numbers
//              a result (about 20 ns).
//      facts   {hp: [current, base], weapon: {id, name} | null (the worn weapon; name: the engine's debug name),
//              eats (Eats since the life started that took the item), death: {tick, killers: [names of the npcs within
//              16 tiles whose target was the agent when its hitpoints reached 0, which can include one that did not hit
//              it; empty when none was]} | null, held: the item kinds the player holds, its pack's and its worn items'
//              distinct item ids in ascending order (heldKinds; read from the player's inventories at `tick`, not from
//              the observation; no pay counts them), food: the items in the pack whose own options
//              (the item's iop) include Eat, each stack by its count (foodCount, read at `tick`), food_drops: the life's
//              Drops that took an item that offers Eat from its slot (by eats' rule: the slot then holds no item,
//              another kind or fewer). (facts.used, the uses the engine carried out, which only the removed use pay
//              first-use-v1 read, was removed after its study closed.) tool_drops:
//              the life's Drops that took an item whose name TOOL_PROTECT_RE matches (the rule of mask-v5, which mask-v6
//              does not have), by food_drops' rule; a read that nothing pays. hurt: the life's missing hitpoints tick
//              by tick (engine/hurt.ts HurtTicks): [[base, n], ...], bases ascending, n the sum over the life's ticks at
//              that base of base - min(hp, base), counted at the end of each world tick from the life's start (the
//              reset result) while the player is alive; integers, so the life's full-deficit ticks, sum n / base, are
//              exact; [] at a reset's result, a failed reset's and a leave's (HurtTicks.end at the reset's start and at
//              the leave). No pay charges by it today (health-cost-v1, which did, was removed); the life rows read it.}.
//      ledger  on reset, left and failed-reset results only: the life this item closed (null at a first join):
//              {start, end, grants: [[stat, xp10, tick], ...], total: [21]}: every XP grant to that life's player,
//              taken inside Player.addXp (the engine's only path that adds XP: script stat_advance), the amount
//              actually applied (after the XP rate and the 200M cap), at the tick of the cycle that granted it.
//              Independent of pay and obs: the learner compares the sum of its pay with `total`.
//    reset: cfg is today's reset cfg (env-ts/core/budgets.ts TaskCfg; free play: contract 'item-v2', or 'item-v3' for
//      the page of kinds (core/items.ts; every observation stamps it), actTargets
//      'act-v3', kit 'none', seedItems, setStats, spawnTele, paceJitterTicks, decisionPoints, maxDecisions,
//      maxEpisodeTicks; startHp, the start record's drawn hitpoints, free-play-v12's and v13's alone: the current
//      hitpoints after setStats, 1 to the base, lowered as the engine's stat_sub lowers a level). setup {wield: NAME | null} runs today's setup (baseline/scripted.py prepare_declared, then
//      sim/watch.py setup's Wield) through the ordinary dispatch before the result: the declared start is checked
//      (inventory exactly seedItems, nothing worn, base levels exactly setStats), then the style step (A_STYLE0),
//      then a Wield of NAME when given. The life starts at the result's tick; obs is the setup's last observation.
//    error results (kind 'error', code, message, obs null; the slot is due again): after 'illegal-action' the life
//      goes on (nothing was dispatched, no tick passed: the slot takes a legal step from its last observation, a reset
//      or a leave); after every other error the slot needs a reset or a leave. Codes: reset 'spawn-not-landed',
//      'declared-start', 'setup', 'left-world', 'reset' (any other reset failure); step 'life-over', 'illegal-action',
//      'left-world'; either 'internal' (the item threw in the engine: its error at once, never a later stall).
//    A RESET AND THE CLOCK. A reset is an item like a step: it runs inside one request, at any time, between any two
//      stops, with no world restart. It needs world ticks of its own (the replacement player's login, the teleport's
//      landing, the setup's two steps): about 8-12, plus 4 at a slot's first join. The world moves those ticks only as
//      it moves any tick, while every agent waits inside its item, so no other agent gets a tick it did not ask for:
//      an agent whose step ends during them comes back due at its own tick, and the reset goes on after it is
//      answered. The ticks are the shared world's time: other agents' lives go on during them; they are in no life.
// 5. ping {cmd:'ping'} -> {type:'pong', tick}: a keep-alive; no tick passes, the answer limit starts again.
//    profile {cmd:'profile', reset?} (RL_TICK_PROFILE=1 only). close {cmd:'close'} -> {type:'closing'}: every agent
//    leaves (no save) and the connection ends; the world stays up for the next controller. A connection that closes
//    does the same. Both discard the ledgers of the lives they end: a learner that wants them sends leave items first.
//
// ERRORS. A refused request: {type:'error', code, message}; nothing started, no tick. Codes: 'hello-first', 'protocol',
//   'bad-request', 'unknown-slot', 'duplicate-slot', 'busy', 'due-without-item', 'not-joined', 'capacity',
//   'controller-present', 'internal' (an unexpected engine error; the request may have started items).
// FATAL. Any exit of the engine process is fatal to its world, whatever its code: treat a closed connection with no
//   'closing' reply as the end of the world. Code 3: the engine's own fatal, its line {type:'fatal', code, message}
//   sent first (scripts/stack.py then stops the run). Code 1: a world tick threw; the vendor cycle removes every player
//   and exits at once, so the 'player-removed' line it starts may not arrive. Any other code or a signal: a crash.
//   The engine's own fatal codes:
//   'answer-limit': no request came within answerLimitMs of a reply that left slots due (the world was held);
//   'player-removed': the engine removed an agent's player by itself (an idle logout, or a world tick that threw and
//   removed every player); 'stall': a guard, no tick and no result in a turn (cannot happen under the stop rule).
//   No silent hang: a world that cannot go on says why and stops.
// ANSWER LIMIT. Counted from each batch reply to the next request of any kind on the connection (a ping counts), only
//   while some slot is due. During a learner update no request may come for minutes and the world stands still: the
//   learner declares a limit longer than its update (or sends a ping now and then); the default is ten minutes.
// LIMITS. capacity <= 2046; one item a slot a request; the world ticks only while every agent in it waits, so one slow
//   answer holds all agents; the login of a first join runs on the shared clock, so a join at any time leaves every
//   other agent's tick count right.
// DEMO. A shared world may also be a demo world (scripts/stack.py --demo-spectator NAME --demo-pace-ms MS,
//   core/spectator.ts): the protocol and its replies are the same, and its clock is paced (core/lockstep-clock.ts):
//   a tick comes at most once every MS of wall time, so the world's wall time between ticks changes and nothing else.
//   Until the controller connects the world ticks at that pace with no agent, so that the spectator can log in. Its
//   records are demos, never evidence (sim/real.py demo). No protocol version changes: no request, reply or rule does.
// NOT HERE. No native resource or combat capture (kills, loot, certified items: sim/watch.py's captures follow one actor
//   a world) and no reward: the engine reports pay, the ledger and facts; the learner pays. The world around an agent
//   is shared: other agents' drops, opened doors, killed npcs and depleted resources; respawns shorten with the
//   player count (World.scaleByPlayerCount).
//
// WHAT CHANGED FROM THE PROTOTYPE: RL_SHARED_AGENTS replaces RL_LOCKSTEP_AGENTS; no agent is admitted
// at connect and `join {count}` goes (a slot joins at its first reset, its login on the shared clock); agents are
// learner slots, not engine names; batch items gain setup, leave and the result fields; a request that leaves a due
// slot without an item is refused (the prototype's stall check stays as a guard); `places` gains centres and radius;
// `ping` is new. The stop rule is the prototype's.
import { connectedJitterTiles } from '../spawn_tiles';
import { isTileWalkable } from '../../vendor/rs-sdk/sdk/pathfinding';
import type { SharedClock } from '../core/lockstep-clock';
import type { LifeLedger } from './xp-ledger';
import { A_HELD0, N_INV_SEL, pageRow } from '../core/actions';
import { ITEM_WIDE } from '../core/items';

export const PROTOCOL = 'shared-lockstep-v1';
/** The fields of every result's facts, in the order the engine's agent writes them (hello's factFields). */
export const FACT_FIELDS = ['hp', 'weapon', 'eats', 'death', 'held', 'food', 'food_drops', 'tool_drops', 'hurt'] as const;
export const DEFAULT_ANSWER_LIMIT_MS = 600_000;
const ANSWER_LIMIT_RANGE = [1_000, 3_600_000];

/** A refused request: nothing started, no tick. */
export class ProtocolError extends Error {
    constructor(readonly code: string, message: string) { super(message); }
}

/** A world that cannot go on: engine-env sends it and exits. */
export class FatalError extends Error {
    constructor(readonly code: string, message: string) { super(message); }
}

/** What the shared world needs of an agent (engine/agent.ts EngineAgent). */
export interface SharedAgent {
    readonly player: object;
    readonly dead: boolean;
    reset(seed: number, cfg: any): Promise<Record<string, any>>;
    step(action: number): Promise<Record<string, any>>;
    waitWorldTicks(n: number): Promise<void>;
    leave(): void;
    pay(): { xp: number[]; base: number[] };
    facts(): Record<string, any>;
    /** the life starts: facts.hurt counts from 0 (engine/hurt.ts) */
    armHurt?(): void;
    declaredStartProblem(cfg: any): string | null;
    wieldSlot(obs: Record<string, any>, name: string): number;
}

export interface WorldHost<A extends SharedAgent> {
    capacity: number;
    clock: SharedClock;
    tick(): number;
    /** a new agent for `slot`, its player admitted and the clock joined */
    makeAgent(slot: number): A;
    ledger: { follow(player: object): void; running(player: object): number[] | null; close(player: object): LifeLedger | null };
    /** the world's facts in the hello reply (worldId, xpRate, kTicks, skills) */
    facts: Record<string, unknown>;
    actions: { style: number; held0: number; count: number };
    loginTicks: number;
    places?: (count: number, seed: number, centres?: Array<[number, number]>, radius?: number) => Array<{ x: number; z: number }>;
    nextTurn?: () => Promise<void>;
}

interface Slot<A> { agent: A; joining: boolean; busy: boolean; due: boolean; done: boolean; obs: Record<string, any> | null }

export type Result = Record<string, any>;

export class SharedWorld<A extends SharedAgent> {
    private readonly slots = new Map<number, Slot<A>>();
    private ready: Result[] = [];
    private greeted = false;
    answerLimitMs = DEFAULT_ANSWER_LIMIT_MS;

    constructor(private readonly host: WorldHost<A>) {}

    /** Some slot is due: the world is held until the next request (the answer limit runs). */
    held(): boolean {
        for (const s of this.slots.values()) if (s.due) return true;
        return false;
    }

    /** `player` is a live slot's player (engine-env: a removal the engine made by itself is fatal). */
    owns(player: object): boolean {
        for (const s of this.slots.values()) if (s.agent.player === player) return true;
        return false;
    }

    async handle(req: any): Promise<Record<string, any>> {
        if (req?.cmd === 'hello') return this.hello(req);
        if (!this.greeted) throw new ProtocolError('hello-first', 'send hello first');
        switch (req.cmd) {
            case 'places': {
                const { count, seed = 1, centres, radius } = req;
                if (!Number.isInteger(count) || count < 1 || (centres !== undefined && (!Array.isArray(centres) ||
                        !centres.every((c: any) => Array.isArray(c) && c.length === 2 && c.every(Number.isInteger)) ||
                        !Number.isInteger(radius) || radius < 0)))
                    throw new ProtocolError('bad-request', 'places takes a count, a seed, and centres with a radius');
                try {
                    return { type: 'places', tiles: (this.host.places ?? places)(count, seed, centres, radius) };
                } catch (error) {
                    throw new ProtocolError('bad-request', String((error as Error).message));
                }
            }
            case 'batch': return this.batch(req.items);
            case 'ping': return { type: 'pong', tick: this.host.tick() };
            case 'close':
                this.leaveAll();
                return { type: 'closing' };
            default: throw new ProtocolError('bad-request', `unknown cmd ${req.cmd}`);
        }
    }

    /** Every agent leaves (close, or the controller's connection ended). */
    leaveAll(): void {
        for (const slot of [...this.slots.keys()]) this.remove(slot);
        this.ready = [];
        this.greeted = false;
    }

    private hello(req: any): Record<string, any> {
        if (req.protocol !== PROTOCOL) throw new ProtocolError('protocol', `this world speaks ${PROTOCOL}`);
        const limit = req.answerLimitMs ?? DEFAULT_ANSWER_LIMIT_MS;
        if (!Number.isInteger(limit) || limit < ANSWER_LIMIT_RANGE[0] || limit > ANSWER_LIMIT_RANGE[1])
            throw new ProtocolError('bad-request', `answerLimitMs in ${ANSWER_LIMIT_RANGE.join('..')}`);
        this.answerLimitMs = limit;
        this.greeted = true;
        return { type: 'hello', protocol: PROTOCOL, capacity: this.host.capacity, tick: this.host.tick(),
            answerLimitMs: limit, ...this.host.facts };
    }

    private async batch(items: any): Promise<Record<string, any>> {
        if (!Array.isArray(items)) throw new ProtocolError('bad-request', 'batch items are a list');
        const seen = new Set<number>();
        let joins = 0;
        for (const item of items) {
            const slot = item?.slot;
            if (!Number.isInteger(slot) || slot < 0 || slot >= this.host.capacity)
                throw new ProtocolError('unknown-slot', `slot ${JSON.stringify(slot)} is not in 0..${this.host.capacity - 1}`);
            if (seen.has(slot)) throw new ProtocolError('duplicate-slot', `slot ${slot} has two items`);
            seen.add(slot);
            if (!['reset', 'step', 'leave'].includes(item.op)) throw new ProtocolError('bad-request', `op ${item.op}`);
            if (item.op === 'step' && !Number.isInteger(item.action)) throw new ProtocolError('bad-request', 'a step names its action');
            if (item.op === 'reset' && (typeof item.cfg !== 'object' || item.cfg === null))
                throw new ProtocolError('bad-request', 'a reset carries its cfg');
            const s = this.slots.get(slot);
            if (s?.busy) throw new ProtocolError('busy', `slot ${slot} is inside its item`);
            if (!s && item.op !== 'reset') throw new ProtocolError('not-joined', `slot ${slot} is not in the world`);
            if (!s) joins++;
        }
        if (this.slots.size + joins > this.host.capacity)
            throw new ProtocolError('capacity', `the world holds ${this.host.capacity} agents`);
        const missing = [...this.slots].filter(([slot, s]) => s.due && !seen.has(slot)).map(([slot]) => slot);
        if (missing.length) throw new ProtocolError('due-without-item', `due slots without an item: ${missing.join(', ')}`);
        const left: Result[] = [];
        // Every joining agent is in the clock before any item starts: an item's first wait may complete the set of
        // waiting agents and tick the world, which must count every agent of this request.
        for (const item of items) {
            if (!this.slots.has(item.slot))
                this.slots.set(item.slot, { agent: this.host.makeAgent(item.slot), joining: true, busy: false, due: false,
                    done: true, obs: null });
        }
        for (const item of items) {
            if (item.op === 'leave') {
                left.push(this.remove(item.slot));
                continue;
            }
            const slot = item.slot, own = this.slots.get(slot)!;
            own.busy = true;
            own.due = false;
            const t0 = this.host.tick();
            (item.op === 'reset' ? this.reset(slot, own, item) : this.step(slot, own, item)).catch(error => {
                own.done = true; // an item that threw (not one of reset's or step's own errors): its error at once
                const tick = this.host.tick();
                return { slot, kind: 'error', tick, ticks: tick - t0, obs: null, done: true, dead: false, pay: null,
                    facts: null, code: 'internal', message: String((error as Error)?.message ?? error) };
            }).then(result => {
                if (this.slots.get(slot) !== own) return; // it left meanwhile (close)
                own.busy = false;
                own.due = true;
                this.ready.push(result);
            });
        }
        // The ticks run as a chain of promise jobs that ends when some agent's item is over and the clock waits for it;
        // a later turn of the event loop sees the end. A turn with no tick and nothing over cannot happen under the stop
        // rule: it would hang, so it is fatal. A paced clock's tick set for later (a demo world, DEMO) is no hang: the
        // turns then wait a millisecond each, not spin.
        const busy = () => { for (const s of this.slots.values()) if (s.busy) return true; return false; };
        const clock = this.host.clock;
        const turn = this.host.nextTurn ?? (() => new Promise<void>(resolve => clock.pending ? setTimeout(resolve, 1) : setImmediate(resolve)));
        for (let last = this.host.tick(); ;) {
            await turn();
            if (this.ready.length > 0 || !busy()) break;
            if (this.host.tick() === last && !clock.pending) throw new FatalError('stall', 'no agent waits for a tick and none is due');
            last = this.host.tick();
        }
        const results = [...left, ...this.ready];
        this.ready = [];
        return { type: 'batch', tick: this.host.tick(), results };
    }

    private result(slot: number, s: Slot<A>, kind: string, t0: number, obs: Record<string, any> | null,
                   extra: Record<string, any> = {}): Result {
        const tick = this.host.tick();
        return { slot, kind, tick, ticks: tick - t0, obs, done: s.done, dead: s.agent.dead,
            pay: { ...s.agent.pay(), ledger: this.host.ledger.running(s.agent.player) }, facts: s.agent.facts(), ...extra };
    }

    private async reset(slot: number, s: Slot<A>, item: any): Promise<Result> {
        const t0 = this.host.tick();
        const closing = s.joining ? null : this.host.ledger.close(s.agent.player);
        const fail = (code: string, message: string) => {
            s.done = true;
            return this.result(slot, s, 'error', t0, null, { code, message, ledger: closing });
        };
        try {
            if (s.joining) {
                await s.agent.waitWorldTicks(this.host.loginTicks);
                s.joining = false;
            }
            let obs = await s.agent.reset(item.seed ?? 0, item.cfg);
            if (item.setup) {
                const problem = s.agent.declaredStartProblem(item.cfg);
                if (problem) return fail('declared-start', problem);
                obs = await s.agent.step(this.host.actions.style);
                if (item.setup.wield) {
                    const held = s.agent.wieldSlot(obs, item.setup.wield);
                    if (held < 0) return fail('setup', `no Wield offered for ${item.setup.wield}`);
                    // the held row of entry `held`: act-v4's second group for entries 8..15 of item-v4's page
                    obs = await s.agent.step(obs.contractVersion === ITEM_WIDE && held >= N_INV_SEL
                        ? pageRow(A_HELD0, held) : this.host.actions.held0 + held);
                }
                if (obs.done || s.agent.dead) return fail('setup', 'the life ended in its setup');
            }
            s.obs = obs;
            s.done = false;
            s.agent.armHurt?.();
            this.host.ledger.follow(s.agent.player);
            return this.result(slot, s, 'reset', t0, obs, { ledger: closing });
        } catch (error) {
            const message = String((error as Error)?.message ?? error);
            return fail(/did not land|no connected jitter/.test(message) ? 'spawn-not-landed'
                : /left the world|died/.test(message) ? 'left-world' : 'reset', message);
        }
    }

    private async step(slot: number, s: Slot<A>, item: any): Promise<Result> {
        const t0 = this.host.tick();
        if (s.done || !s.obs) return this.result(slot, s, 'error', t0, null, { code: 'life-over',
            message: `slot ${slot} has no live life: reset it` });
        const action = item.action;
        // the slot's own mask bounds its rows: act-v3's 96, or act-v4's 128 under item-v4
        if (action < 0 || action >= (s.obs.mask?.length ?? this.host.actions.count) || !s.obs.mask?.[action])
            return this.result(slot, s, 'error', t0, null, { code: 'illegal-action',
                message: `action ${action} is not legal in slot ${slot}'s last mask` });
        try {
            const obs = await s.agent.step(action);
            s.obs = obs;
            s.done = !!(obs.done || obs.truncated || s.agent.dead);
            return this.result(slot, s, 'step', t0, obs);
        } catch (error) {
            s.done = true;
            return this.result(slot, s, 'error', t0, null, { code: 'left-world', message: String((error as Error)?.message ?? error) });
        }
    }

    /** The slot's agent leaves (no save): its result, with the ledger of the life it closes. */
    private remove(slot: number): Result {
        const s = this.slots.get(slot)!;
        const t0 = this.host.tick();
        const closing = s.joining ? null : this.host.ledger.close(s.agent.player);
        s.done = true;
        const out = this.result(slot, s, 'left', t0, null, { ledger: closing });
        this.slots.delete(slot);
        // The player leaves the world first: leaving the clock can tick the world at once (every other agent may be
        // waiting), and the departing player must not be in that tick.
        s.agent.leave();
        this.host.clock.leave(s.agent);
        return out;
    }
}

export interface ControllerIO {
    send(msg: object): void;
    /** end the connection (after close) */
    end(): void;
    /** a world that cannot go on (engine-env: the fatal line, then exit 3) */
    fatal(code: string, message: string): void;
    /** a reply for a request the world does not serve (engine-env: profile), or null */
    extra?(req: any): Record<string, any> | null;
    /** each request's command and wall milliseconds (the tick profile) */
    done?(cmd: string, ms: number): void;
    timers?: { set(run: () => void, ms: number): unknown; clear(handle: unknown): void };
}

/** The controller connection's requests, one line each, and its answer limit (engine-env serves it). */
export class Controller {
    private timer: unknown = null;
    private readonly timers;

    constructor(private readonly world: SharedWorld<any>, private readonly io: ControllerIO) {
        this.timers = io.timers ?? { set: (run: () => void, ms: number) => setTimeout(run, ms),
            clear: (handle: unknown) => clearTimeout(handle as ReturnType<typeof setTimeout>) };
    }

    async line(text: string): Promise<void> {
        if (this.timer !== null) this.timers.clear(this.timer);
        this.timer = null;
        const t0 = performance.now();
        let req: any = null;
        try {
            // parsed inside the try whose finally starts the limit again: a bad line followed by silence still stops
            try {
                req = JSON.parse(text);
            } catch (error) {
                throw new ProtocolError('bad-request', `bad json: ${(error as Error).message}`);
            }
            const extra = this.io.extra?.(req);
            if (extra) {
                this.io.send(extra);
                return;
            }
            this.io.send(await this.world.handle(req));
            if (req?.cmd === 'close') this.io.end();
        } catch (error) {
            if (error instanceof FatalError) {
                this.io.fatal(error.code, error.message);
                return;
            }
            this.io.send({ type: 'error', code: error instanceof ProtocolError ? error.code : 'internal',
                message: String((error as Error)?.message ?? error) });
        } finally {
            this.io.done?.(String(req?.cmd), performance.now() - t0);
            // the world is held while some slot is due: no request within the limit stops it
            if (this.world.held()) this.timer = this.timers.set(() => this.io.fatal('answer-limit',
                `no request within ${this.world.answerLimitMs} ms while slots were due`), this.world.answerLimitMs);
        }
    }

    /** The connection ended: no limit, every agent leaves. */
    drop(): void {
        if (this.timer !== null) this.timers.clear(this.timer);
        this.timer = null;
        this.world.leaveAll();
    }
}

/** facts.held: the distinct item ids of `inventories` (each a list of slots, empty ones null), in ascending order. */
export function heldKinds(inventories: Iterable<Iterable<{ id: number } | null | undefined>>): number[] {
    const kinds = new Set<number>();
    for (const slots of inventories) for (const item of slots) if (item) kinds.add(item.id);
    return [...kinds].sort((a, b) => a - b);
}

/** Whether an item's own options (its iop, as the game defines it) include Eat. */
export function offersEat(iop: ArrayLike<string | null | undefined> | null | undefined): boolean {
    return !!iop && Array.from(iop).includes('Eat');
}

/** facts.food: the items of `slots` (a pack, empty slots null) that offer Eat (`iop(id)`: the item's own options), each
 * stack by its count. */
export function foodCount(slots: Iterable<{ id: number; count: number } | null | undefined>,
                          iop: (id: number) => ArrayLike<string | null | undefined> | null | undefined): number {
    let n = 0;
    for (const item of slots) if (item && offersEat(iop(item.id))) n += item.count;
    return n;
}

/** An Eat's or a Drop's effect, by the slot it acted on (`before`, from the observation the policy chose from): after
 * the decision the slot holds no item, another kind or fewer of it. */
export function tookFrom(before: { id: number; count: number }, now: { id: number; count: number } | null | undefined): boolean {
    return !now || now.id !== before.id || now.count < before.count;
}

/** The engine's own settings that hello reports: its XP rate and whether its random events are on. */
export function engineSettings(environment: { node: { xpRate: number }; NODE_RANDOM_EVENTS: boolean }):
        { xpRate: number; randomEvents: boolean } {
    return { xpRate: environment.node.xpRate, randomEvents: environment.NODE_RANDOM_EVENTS === true };
}

function shuffled<T>(items: T[], seed: number): T[] {
    let state = (seed >>> 0) || 1;
    for (let i = items.length - 1; i > 0; i--) {
        state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0;
        const j = state % (i + 1);
        [items[i], items[j]] = [items[j], items[i]];
    }
    return items;
}

const usable = (x: number, z: number) => isTileWalkable(0, x, z) && connectedJitterTiles(x, z).length > 0;

/**
 * `count` start tiles on level 0, each walkable with a connected jitter tile (spawn_tiles.ts). With centres: within
 * `radius` tiles (the larger of |dx| and |dz|) of a centre, dealt round-robin over the centres, every tile of a centre
 * once before any twice. Without: the mainland test grid (a 16-tile grid of x 2560-3392, z 3072-3520). Seeded.
 */
export function places(count: number, seed: number, centres?: Array<[number, number]>, radius = 0,
                       usableTile: (x: number, z: number) => boolean = usable): Array<{ x: number; z: number }> {
    const out: Array<{ x: number; z: number }> = [];
    if (!centres) {
        const grid: Array<{ x: number; z: number }> = [];
        for (let x = 2560; x < 3392; x += 16) for (let z = 3072; z < 3520; z += 16) grid.push({ x, z });
        for (const tile of shuffled(grid, seed)) {
            if (out.length >= count) break;
            if (usableTile(tile.x, tile.z)) out.push(tile);
        }
        if (out.length < count) throw new Error(`only ${out.length} places in the grid, not ${count}`);
        return out;
    }
    if (!centres.length) throw new Error('places needs at least one centre');
    const pools = centres.map(([cx, cz], k) => {
        const tiles: Array<{ x: number; z: number }> = [];
        for (let x = cx - radius; x <= cx + radius; x++) for (let z = cz - radius; z <= cz + radius; z++) tiles.push({ x, z });
        return { order: shuffled(tiles, seed + k), next: 0, good: [] as Array<{ x: number; z: number }>, used: 0 };
    });
    for (let k = 0; out.length < count; k++) {
        const pool = pools[k % pools.length];
        let tile: { x: number; z: number } | undefined;
        while (pool.next < pool.order.length && !tile) {
            const t = pool.order[pool.next++];
            if (usableTile(t.x, t.z)) { tile = t; pool.good.push(t); }
        }
        if (!tile) { // every tile of the centre is dealt once: deal them again in the same order
            if (!pool.good.length) throw new Error(`no usable tile within ${radius} of ${centres[k % pools.length]}`);
            tile = pool.good[pool.used++ % pool.good.length];
        }
        out.push(tile);
    }
    return out;
}
