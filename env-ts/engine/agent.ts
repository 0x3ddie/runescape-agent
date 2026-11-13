// env-ts/engine/agent.ts: one RL agent as a clientless Player inside the engine.
//
// This is the EFFECTOR half of the environment. The mask, the action table and the
// episode budgets come from env-ts/core/. Actions are direct calls on a `Player`,
// and reset writes fields. (A Chromium transport existed until October 9, 2026; some
// comments below explain choices made to match it.)
//
// THREE RULES, each learned from a probe failure (CHANGELOG 2026-08-16 15:10):
//   1. Path BEFORE you interact. The web client routes itself and sends waypoints
//      alongside the op, so setInteraction alone leaves the player standing still
//      out of range and nothing happens.
//   2. Rescan for a live target every time. A chopped tree's Loc becomes invalid
//      and setInteraction refuses it.
//   3. Never clear modal state before reading it, or you destroy the dialog you
//      were about to observe.

import { randomUUID } from 'node:crypto';
import { ACT_WALK_TILES, affordanceKey, actAction, checkActContract, heldAction, publishAffordances, useAction,
    type Affordance } from '../core/affordances';
import { ENTITY, ROUTE_RECORD, TILE, clickSlot, clickSlots, clientFlags, locArrival, sceneBase, tryMove, type Arrival,
    type RouteRecord } from '../core/clicks';
import { walkMap } from '../reach';
import { TextCapture, describeAction, feedback, type ActionMeaning, type EndReason, type OptionEnd, type Snapshot } from '../core/feedback';
import { continuityOptions, continuityEnvelope, checkContinuity, type ContinuityOptions } from '../core/continuity';
import { monitorLifecycle, type LifecycleMonitor } from './lifecycle';
import { LifecycleBridge } from './lifecycle-bridge';
import InvType from '../../vendor/rs-sdk/server/engine/src/cache/config/InvType.js';
import Component from '../../vendor/rs-sdk/server/engine/src/cache/config/Component.js';
import LocType from '../../vendor/rs-sdk/server/engine/src/cache/config/LocType.js';
import { replacePlayerForReset } from './reset-isolation';
import { ItemPages, checkItemContract, matchItem, type Item } from '../core/items';
import ObjType from '../../vendor/rs-sdk/server/engine/src/cache/config/ObjType.js';
import CategoryType from '../../vendor/rs-sdk/server/engine/src/cache/config/CategoryType.js';
import DbRowType from '../../vendor/rs-sdk/server/engine/src/cache/config/DbRowType.js';
import DbTableType from '../../vendor/rs-sdk/server/engine/src/cache/config/DbTableType.js';
import World from '../../vendor/rs-sdk/server/engine/src/engine/World.js';
import * as rsbuf from '../../vendor/rs-sdk/server/engine/src/network/rsbuf/index.js';
import { findPath } from '../../vendor/rs-sdk/server/engine/src/engine/GameMap.js';
import routefinder from '../../vendor/rs-sdk/server/engine/src/engine/routefinder/index.js';
import { Interaction } from '../../vendor/rs-sdk/server/engine/src/engine/entity/Interaction.js';
import { AllowRepath } from '../../vendor/rs-sdk/server/engine/src/engine/entity/AllowRepath.js';
import { CoordGrid } from '../../vendor/rs-sdk/server/engine/src/engine/CoordGrid.js';
import Player from '../../vendor/rs-sdk/server/engine/src/engine/entity/Player.js';
import { PlayerLoading } from '../../vendor/rs-sdk/server/engine/src/engine/entity/PlayerLoading.js';
import { PlayerStat } from '../../vendor/rs-sdk/server/engine/src/engine/entity/PlayerStat.js';
import ScriptProvider from '../../vendor/rs-sdk/server/engine/src/engine/script/ScriptProvider.js';
import ScriptRunner from '../../vendor/rs-sdk/server/engine/src/engine/script/ScriptRunner.js';
import ScriptState from '../../vendor/rs-sdk/server/engine/src/engine/script/ScriptState.js';
import ServerTriggerType from '../../vendor/rs-sdk/server/engine/src/engine/script/ServerTriggerType.js';
import VarPlayerType from '../../vendor/rs-sdk/server/engine/src/cache/config/VarPlayerType.js';
import Environment from '../../vendor/rs-sdk/server/engine/src/util/Environment.js';
import Packet from '../../vendor/rs-sdk/server/engine/src/io/Packet.js';
import IfSetText from '../../vendor/rs-sdk/server/engine/src/network/game/server/model/IfSetText.js';
import MessageGame from '../../vendor/rs-sdk/server/engine/src/network/game/server/model/MessageGame.js';

import {
    A_ACT0,
    A_CLOSE_IF,
    A_DEPOSIT0,
    A_DIALOG_CHOICE0,
    A_DIALOG_CONTINUE,
    A_DROP0,
    A_HELD0,
    A_STYLE0,
    A_USE0,
    A_WITHDRAW0,
    N_ACTIONS,
    N_INV_SEL,
    optionLabelFor
} from '../core/actions';
import { monitorDeath, type DeathMonitor } from './death';
import { ACT_STALL_TICKS, K_TICKS, NAV_STALL_TICKS, OPTION_HARD_TIMEOUT_TICKS, type TaskCfg, isTruncated, SPAWN_TOLERANCE_TILES } from '../core/budgets';
import { MASK_CONTRACT, buildMask } from '../core/mask';
import { keepWaiting } from '../core/decision-points';
import { farSpawn, jitterWorld, worldFromTele } from '../core/world';
import { findLongPath } from '../../vendor/rs-sdk/sdk/pathfinding';
import { requireRecoveryPlacementProfile } from '../core/recovery-placement';
import { demoSettings } from '../core/spectator';
import { actCandidates, buildObs, opIndexOf, scanNpcs, scanWorld, SKILL_NAMES, type EngineCandidate,
    type ScanResult } from './observe';

/** The 18-item starter kit `::~rlreset` grants (vendor-patches/0003 cheat_rlreset.rs2). */
const STARTER_KIT: Array<[string, number]> = [
    ['bronze_axe', 1],
    ['tinderbox', 1],
    ['net', 1],
    ['shrimp', 1],
    ['bucket_empty', 1],
    ['pot_empty', 1],
    ['bread', 1],
    ['bronze_pickaxe', 1],
    ['bronze_dagger', 1],
    ['bronze_sword', 1],
    ['wooden_shield', 1],
    ['shortbow', 1],
    ['bronze_arrow', 25],
    ['airrune', 25],
    ['mindrune', 15],
    ['waterrune', 6],
    ['earthrune', 4],
    ['bodyrune', 2]
];

const CONTINUITY_WORLD_BOOT_ID = process.env.RL_DEMO_BOOT_ID || randomUUID();
// RL_TICK_TRACE (sim.check chase, a fidelity check): each step's observation lists the player and the npcs within
// TICK_TRACE_TILES after every tick of the decision. Evidence for that check, never a policy input.
const TICK_TRACE = !!process.env.RL_TICK_TRACE;
const TICK_TRACE_TILES = 20;
// Milliseconds a lockstep demo world waits after each tick (scripts/stack.py --demo-pace-ms); 0 elsewhere.
const DEMO_PACE_MS = demoSettings(process.env)?.paceMs ?? 0;
// Where a reset without spawnTele starts: Lumbridge, beside the castle.
const LUMBRIDGE_TELE = '0,50,50,22,22';

// ^wearpos_rhand: the worn slot whose weapon picks the combat styles.
const WEARPOS_RHAND = 3;

// an inventory option follows OpHeldHandler and OpHeldUHandler, refused while delayed,
// then clearPendingAction, so an Eat, Wield or Drop ends a running activity. Each observation names it (heldVersion,
// sim/agent.py HELD_CONTRACT); an observation without the field is held-v1, and results of the two are never pooled.
export const HELD_CONTRACT = 'held-v2';

const SKILL_BY_NAME = new Map<string, number>(SKILL_NAMES.map((n, i) => [n.toLowerCase(), i]));

const chebyshev = (ax: number, az: number, bx: number, bz: number) => Math.max(Math.abs(ax - bx), Math.abs(az - bz));
// An affordance whose use ended with nothing changed (no experience, the same inventory,
// no bank or shop opened: an empty rock, too low a level, no tool, a conversation) is left
// out of the slots this long; a refilled rock is another loc anyway. Without it a policy
// with no net tried to net a fishing spot over 600 times.
const REFUSED_TICKS = 100;
// An exact option name, for the raw op lists.
const exactly = (option: string) => new RegExp(`^${option.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'i');

interface PendingOption {
    /** ACT only: the affordance, the state at its start, and the state last seen change */
    act?: { key: string; xp: number; items: string; lastXp: number; lastItems: string;
            loc?: { type: number; x: number; z: number; level: number } };
    label: string;
    /** the action that started it, published as optionAction */
    action?: number;
    startTick: number;
    lastPos: string;
    lastProgressTick: number;
    /** completion predicate, evaluated each step */
    done: () => boolean;
}

export class EngineAgent {
    readonly name: string;
    player: Player;
    private cfg: TaskCfg = {};
    private decision = 0;
    private charged = 0;
    private episodeTicks = 0;
    private lastMaskSum = 0;
    private option: PendingOption | null = null;
    private lastOptionNote = '';
    /** feedback-v2 (core/feedback.ts): game text as shown and how the last option ended. */
    private readonly text = new TextCapture();
    private optionEnd: OptionEnd | null = null;
    /** route-v1 (core/clicks.ts): this decision's click or ACT, and the latest one; evidence, never a policy input */
    private decisionRoute: RouteRecord | null = null;
    private latestRoute: RouteRecord | null = null;
    private tickTrace: unknown[] = [];
    /** this reset wrote levels without [changestat], and no style request has recomputed the combat stats since */
    private styleStale = false;
    private closed = false;
    private death!: DeathMonitor;
    private continuity: ContinuityOptions | null = null;
    private lifecycle: LifecycleMonitor | null = null;
    private continuityFailure: string | null = null;
    private accountEpoch = 0;
    private incarnationId = '';
    private lifecycleBridge = new LifecycleBridge({ maxAccounts: 1, maxEventsPerAccount: 64 });
    private lifecycleBoundPlayer: Player | null = null;
    private unsubscribeLifecycle: (() => void) | null = null;
    private itemPages = new ItemPages();
    private readonly refusedTargets = new Map<string, number>();
    private lastObservation: Record<string, any> | null = null;
    /** the scanned thing behind each published actSel slot */
    private actTargets: EngineCandidate[] = [];
    /** the held item picked with USE_k, as the client's "Use" cursor holds it */
    private using: { slot: number; id: number; name: string } | null = null;

    // A connected client keeps its connection fresh while its player stands still, but a
    // clientless agent refreshed it only during steps. A controller that paused longer than
    // the world's 500-tick idle limit (an episode-end audit, a learner update) had its
    // player logged out between steps, and the next status check failed.
    private readonly heartbeat: ReturnType<typeof setInterval>;

    constructor(name: string) {
        this.name = name;
        this.player = this.admit();
        this.heartbeat = setInterval(() => { if (!this.closed && this.isAdmitted()) this.keepAlive(); }, 1000);
        (this.heartbeat as any).unref?.();
    }

    // ---- lifecycle ---------------------------------------------------------

    /**
     * A clientless Player. `PlayerLoading.load` returns a fresh level-1 player
     * with 10 hitpoints when the save is under 2 bytes (PlayerLoading.ts:39),
     * before any magic/version/CRC check, so an empty Packet is the supported
     * new-account path. No save file is needed.
     */
    private admit(): Player {
        const player = PlayerLoading.load(this.name, new Packet(new Uint8Array(0)), null);
        player.staffModLevel = 4;
        // A clientless player drops every outgoing message (Player.write returns without a
        // client), so read them on the way: chat lines and interface text, as a client shows them.
        const write = player.write.bind(player);
        player.write = (message: Parameters<typeof write>[0]) => {
            if (player === this.player) {
                if (message instanceof MessageGame) this.text.onChat(message.msg, World.currentTick);
                else if (message instanceof IfSetText) this.text.onText(message.component, message.text, World.currentTick);
            }
            write(message);
        };
        // Initialize before admission/login can open the character-design modal.
        // Closing that modal queues tutorial_designed_character, which would
        // otherwise overwrite reset's tutorial_complete with 1 on the next tick.
        const tutorial = VarPlayerType.getByName('tutorial');
        if (!tutorial) throw new Error('admit: tutorial varp unavailable');
        player.setVar(tutorial.id, 1000);
        this.death = monitorDeath(player, PlayerStat.HITPOINTS, () => World.currentTick);
        this.incarnationId = randomUUID();
        this.lifecycle = null;
        this.continuityFailure = null;
        if (this.continuity) {
            const root = ScriptProvider.getByName('[queue,player_death]');
            if (!root) throw new Error('continuous-v1: death queue unavailable');
            this.lifecycle = monitorLifecycle(player, PlayerStat.HITPOINTS, () => World.currentTick, root);
            const limit = this.continuity.maxRespawnTicks;
            this.unsubscribeLifecycle = this.lifecycle.subscribe(event => {
                if (event.kind === 'death') this.option = null;
                try { checkContinuity(event, World.currentTick, limit); }
                catch (error) {
                    this.continuityFailure = String((error as Error)?.message ?? error);
                    throw error;
                }
            });
            this.lifecycleBridge.bind(player, {
                worldBootId: CONTINUITY_WORLD_BOOT_ID, accountId: this.name,
                accountEpoch: this.accountEpoch, incarnationId: this.incarnationId
            }, this.lifecycle, () => this.privateItems(player));
            this.lifecycleBoundPlayer = player;
        }
        this.keepAlive(player);
        World.newPlayers.add(player);
        return player;
    }

    /**
     * A clientless Player never refreshes `lastConnected`, because only
     * `NetworkPlayer.decodeIn` does that (NetworkPlayer.ts:63). Without this the
     * world requests an idle logout 500 ticks after admission, 4 seconds at 125
     * ticks/s (World.ts:798, constant World.ts:153).
     */
    private keepAlive(player: Player = this.player): void {
        player.lastConnected = World.currentTick;
        player.lastResponse = World.currentTick;
        if (player === this.player) this.observeNpcs();
    }

    private paceTicks: () => number = () => 0;
    private observedTick = -1;
    private observedFrom = { x: 0, z: 0, level: 0 };

    /**
     * NPCs hunt only players a connected client is watching (World.ts: hunt needs
     * rsbuf.getNpcObservers > 0), and an agent without a client watches nothing, so
     * aggressive npcs ignored it: dark wizards stood beside it for thousands of
     * decisions and never attacked, then killed the same policy within 30 ticks in
     * Chromium. Do the npc-info bookkeeping a client does, once per tick, and drop
     * the bytes, so npcs notice the agent as they would any player.
     */
    private observeNpcs(): void {
        if (World.currentTick === this.observedTick || !this.isAdmitted()) return;
        const p = this.player, from = this.observedFrom;
        rsbuf.npcInfo(0, p.slot, Math.abs(p.x - from.x), Math.abs(p.z - from.z), p.level !== from.level);
        this.observedTick = World.currentTick;
        this.observedFrom = { x: p.x, z: p.z, level: p.level };
    }

    isAdmitted(): boolean {
        return World.getPlayerByUsername(this.name) === this.player && !this.player.loggingOut;
    }

    /** Wait n engine cycles, keeping the agent alive while we do. */
    private async waitTicks(n: number): Promise<void> {
        if (World.lockstep) {
            // The world advances only here, one tick at a time, with the client
            // bookkeeping between ticks that a connected client does in real time.
            // A demo world (core/spectator.ts) also waits after each tick, so its
            // watching client receives the tick and sees it at a viewable pace.
            for (let i = 0; i < n; i++) {
                World.advance(1);
                this.keepAlive();
                if (TICK_TRACE) this.traceTick();
                if (DEMO_PACE_MS) await new Promise(r => setTimeout(r, DEMO_PACE_MS));
            }
            return;
        }
        const start = World.currentTick;
        const deadline = Date.now() + 30_000;
        while (World.currentTick - start < n && Date.now() < deadline) {
            this.keepAlive();
            await new Promise(r => setTimeout(r, 1));
        }
        this.keepAlive();
        if (this.continuity && World.currentTick - start < n)
            throw new Error('continuous-v1: world did not advance before wall-clock deadline');
    }

    /** RL_TICK_TRACE: [tick, x, z, hitpoints, xp sum, [[nid, x, z], ...]] after this tick. */
    private traceTick(): void {
        const p = this.player;
        const npcs = [...scanNpcs(p, TICK_TRACE_TILES)].map(n => [n.nid, n.x, n.z]);
        this.tickTrace.push([World.currentTick, p.x, p.z, p.levels[PlayerStat.HITPOINTS] ?? 0, this.totalXp(), npcs]);
    }

    private continuousState() {
        if (!this.continuity || !this.lifecycle) throw new Error('continuous lifecycle unavailable');
        if (!this.isAdmitted()) throw new Error('continuous-v1: account left world');
        if (this.continuityFailure) throw new Error(`continuous-v1: ${this.continuityFailure}`);
        const snapshot = this.lifecycleBridge.policySnapshot(this.player, World.currentTick);
        return continuityEnvelope({ worldBootId: CONTINUITY_WORLD_BOOT_ID, accountId: this.name,
            accountEpoch: this.accountEpoch, incarnationId: this.incarnationId },
            snapshot, World.currentTick, this.continuity.maxRespawnTicks);
    }

    /** Evaluator-only. Never merge this result into policy observations. */
    continuityAudit(afterSequence = 0) {
        if (!this.lifecycleBoundPlayer) throw new Error('continuous lifecycle audit unavailable');
        return { ...this.lifecycleBridge.auditSnapshot(this.lifecycleBoundPlayer, World.currentTick, afterSequence),
            currentPrivateItems: this.privateItems(this.lifecycleBoundPlayer) };
    }

    /** Private evaluator setup counters; unavailable to ordinary collectors. */
    recoveryPlacementState() {
        requireRecoveryPlacementProfile(process.env);
        if (!this.lifecycleBoundPlayer) throw new Error('recovery placement counters unavailable');
        return { optionBusy:this.option !== null, decision:this.decision,charged:this.charged,episodeTicks:this.episodeTicks };
    }

    private unbindLifecycle(): void {
        this.unsubscribeLifecycle?.();
        this.unsubscribeLifecycle = null;
        if (this.lifecycleBoundPlayer) {
            this.lifecycleBridge.unbind(this.lifecycleBoundPlayer, World.currentTick);
            this.lifecycleBoundPlayer = null;
        }
    }

    private privateItems(player: Player) {
        const bank = InvType.getId('bank');
        if (bank < 0) throw new Error('continuous-v1: bank inventory unavailable');
        return { inventory: this.inventoryItems(InvType.INV, player),
            worn: this.inventoryItems(InvType.WORN, player), bank: this.inventoryItems(bank, player) };
    }

    // ---- reset ------------------------------------------------------------

    async reset(seed: number, cfg: TaskCfg = {}): Promise<Record<string, any>> {
        checkItemContract(cfg); // reject an unknown contract before changing the account
        checkActContract(cfg);
        const nextContinuity = continuityOptions(cfg as TaskCfg & { lifecycle?: unknown; maxRespawnTicks?: unknown }, process.env.RL_CONTINUITY_EXPERIMENT);
        this.unbindLifecycle(); // caller must archive continuityAudit before explicit hard reset
        this.continuity = nextContinuity;
        this.accountEpoch++;
        this.cfg = cfg;
        // A browser decision takes 2-3 ticks against the engine's fixed 2, and an
        // imitation policy that only saw the fixed pace mistimed the end of a
        // Chromium episode. Randomize the pace per episode. Seeding from the reset
        // seed and epoch gave every fresh world the same "random" pattern (97% of
        // steps alike across demonstrations), which imitation learned; world
        // timing isn't reproducible anyway, so draw a fresh seed.
        const jitter = cfg.paceJitterTicks ?? 0;
        let state = (Math.random() * 0x100000000) >>> 0 || 1;
        this.paceTicks = () => {
            state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0;
            return state % (jitter + 1);
        };
        this.itemPages.reset();
        this.using = null;
        this.latestRoute = this.decisionRoute = null;
        this.refusedTargets.clear();
        this.lastObservation = null;
        this.decision = 0;
        this.charged = 0;
        this.episodeTicks = 0;
        this.lastMaskSum = 0;
        this.option = null;
        this.lastOptionNote = '';
        this.optionEnd = null;

        // Alive time-limit endings can also leave native scripts queued (for
        // example a level notification). Every explicit reset needs a new Player;
        // clearing the current modal does not clear the engine script queue.
        {
            const previous = this.player;
            const retained = cfg.persistLevels
                ? { stats: previous.stats.slice(), baseLevels: previous.baseLevels.slice() }
                : null;
            this.player = replacePlayerForReset(previous, World.newPlayers,
                player => World.removePlayerWithoutSave(player), () => this.admit());
            if (retained) {
                this.player.stats.set(retained.stats);
                this.player.baseLevels.set(retained.baseLevels);
                this.player.levels.set(retained.baseLevels);
                this.player.combatLevel = this.player.getCombatLevel();
            }
            await this.waitTicks(2);
        }

        const p = this.player as any;

        // Explicit hard episode setup writes the replacement Player's starting
        // state directly, then verifies the landed position after native ticks.
        // Natural continuous death/respawn does not enter this reset method.
        p.clearInteraction?.();
        this.player.clearWaypoints();
        this.player.closeModal();
        this.player.activeScript = null;
        this.player.resumeButtons = [];

        // Match browser ::~rlreset: this benchmark starts after Tutorial Island.
        // A fresh Player's default tutorial=0 otherwise blocks Bronze axe/dagger
        // wielding even after teleporting to Lumbridge (tutorial_island_equip).
        const tutorial = VarPlayerType.getByName('tutorial');
        if (!tutorial) throw new Error('reset: tutorial varp unavailable');
        this.player.setVar(tutorial.id, 1000); // ^tutorial_complete, quest.constant

        if (!cfg.persistLevels) {
            // ::minme: every stat to level 1, hitpoints to 10
            for (let i = 0; i < SKILL_NAMES.length; i++) this.player.setLevel(i, 1);
            this.player.setLevel(PlayerStat.HITPOINTS, 10);
        }

        // setLevel runs no [changestat]: the combat stats keep the login's level-1 values until
        // a style request recomputes them (obs.style, core/mask.ts).
        this.styleStale = !!cfg.persistLevels || !!cfg.setStats;
        // ::setstat curriculum, e.g. "attack:40,strength:40"
        if (cfg.setStats) {
            for (const pair of cfg.setStats.split(',')) {
                const [name, lvl] = pair.split(':');
                const stat = SKILL_BY_NAME.get((name ?? '').trim().toLowerCase());
                const level = parseInt(lvl ?? '', 10);
                if (stat !== undefined && Number.isFinite(level)) this.player.setLevel(stat, level);
            }
        }

        // ::~rlreset inventory state
        this.player.invClear(InvType.INV);
        this.player.invClear(InvType.WORN);
        const bank = InvType.getId('bank');
        const coins = ObjType.getId('coins');
        if (bank < 0 || coins < 0) throw new Error('item-v2 reset: bank/coins unavailable');
        this.player.invClear(bank);
        if (cfg.kit !== undefined && cfg.kit !== 'starter' && cfg.kit !== 'none') throw new Error('reset: unknown kit');
        if (cfg.kit !== 'none') {
            this.player.invAdd(bank, coins, 25);
            for (const [objName, count] of STARTER_KIT) {
                const id = ObjType.getId(objName);
                if (id !== -1) this.player.invAdd(InvType.INV, id, count);
            }
        }

        // ::give
        if (cfg.seedCoins) {
            const coins = ObjType.getId('coins');
            if (coins !== -1) this.player.invAdd(InvType.INV, coins, cfg.seedCoins);
        }
        if (cfg.seedItems) {
            for (const pair of cfg.seedItems.split(',')) {
                const [name, count] = pair.split(':');
                const id = ObjType.getId((name ?? '').trim());
                if (id !== -1) this.player.invAdd(InvType.INV, id, Math.max(1, parseInt(count ?? '1', 10) || 1));
            }
        }

        // ::tele, with the SAME seeded connected-tile jitter the browser uses, so
        // paired runs spawn on identical tiles.
        const base = worldFromTele(cfg.spawnTele ?? LUMBRIDGE_TELE);
        const j = cfg.spawnSteps
            ? farSpawn(base, cfg.spawnSteps, seed, walkMap(0, base.x, base.z, cfg.spawnSteps[1]).steps,
                       tile => this.walksBack(tile, base))
            : jitterWorld(base.x, base.z, seed);
        this.player.teleJump(j.x, j.z, 0);

        await this.waitTicks(2);

        // A teleport that did not land is a corrupt episode, not a slow one. The
        // browser needed 3 verified attempts; a direct field write either worked
        // or the tile was wrong, so one check is enough.
        const landed = chebyshev(p.x, p.z, j.x, j.z) <= SPAWN_TOLERANCE_TILES;
        if (!landed) {
            throw new Error(`reset: spawn did not land (want ${j.x},${j.z} got ${p.x},${p.z})`);
        }

        if (this.death.dead || !this.isAdmitted()) {
            throw new Error('reset: player died or left the world during setup');
        }
        if (this.player.getVar(tutorial.id) !== 1000) {
            throw new Error('reset: post-tutorial state was overwritten during setup');
        }

        // Teleport and setup text belong to no decision.
        this.text.clear();
        const scan = scanWorld(this.player);
        const obs = buildObs(this.player, scan, { decision: 0, done: false, truncated: false, isDead: false });
        this.projectItems(obs, scan);
        if (this.continuity) {
            obs.lifecycle = this.continuousState();
            if (obs.lifecycle.phase !== 'alive' || obs.lifecycle.deathSequence !== 0)
                throw new Error('continuous-v1: death during explicit reset');
        }
        obs.mask = this.mask(obs);
        obs.feedback = feedback(null, null, this.snapshot(), false, null, [], this.dialogText());
        obs.seed = seed;
        this.lastMaskSum = (obs.mask as number[]).reduce((a: number, b: number) => a + b, 0);
        return obs;
    }

    // ---- option tracking --------------------------------------------------

    private bankIsOpen(): boolean {
        const id = Component.getId('bank_main');
        return id >= 0 && this.player.modalMain === id;
    }

    private inventoryItems(invType: number, player = this.player): Item[] {
        const inv = player.getInventory(invType);
        const items: Item[] = [];
        if (inv) for (let slot = 0; slot < inv.capacity; slot++) {
            const it = inv.get(slot);
            if (it) items.push({ slot, id: it.id, count: it.count, name: ObjType.get(it.id).name ?? '' });
        }
        return items;
    }

    private projectItems(obs: Record<string, any>, scan: ScanResult): void {
        obs.bankOpen = this.bankIsOpen();
        this.itemPages.project(obs, obs.bankOpen ? this.inventoryItems(InvType.getId('bank')) : []);
        // A picked item that has left its slot (eaten, burnt, banked) is no longer in hand.
        const picked = this.using && this.inventoryItems(InvType.INV).find(i => i.slot === this.using!.slot);
        if (this.using && (picked?.id !== this.using.id || obs.bankOpen)) this.using = null;
        obs.using = this.using ? { ...this.using } : null;
        // Only what the player can walk up to: the fewer steps of a side with no wall between
        // and a tile beside it (a bank booth is used across its counter; the wall-free side
        // of Falador's booths is 19 steps round the back). A loc that cannot be used from
        // where the walk ends (a tree behind a fence) is refused after one attempt.
        const p = this.player;
        const walk = walkMap(p.level, p.x, p.z, ACT_WALK_TILES);
        const refused = (a: Affordance) => (this.refusedTargets.get(affordanceKey(a)) ?? -1) > World.currentTick;
        const steps = (c: EngineCandidate) => {
            const [side, beside] = [c.kind === 'loc' ? walk.operable(c.x, c.z) : -1, walk.approach(c.x, c.z)];
            return side < 0 ? beside : beside < 0 ? side : Math.min(side, beside);
        };
        this.actTargets = publishAffordances(obs, actCandidates(scan), steps, refused);
        // act-v3's clicks: natively every target lies in the declared world; `open` reads the live collision.
        obs.clicks = clickSlots(p.x, p.z, (x, z) => routefinder.collisionFlags.get(x, z, p.level));
        const attacker = scan.npcs.find(n => (n.npc as any).target === p);
        obs.style = this.styleFacts();
        // Evidence for checks and logs, not a policy input (rl/obs.py does not read it).
        obs.attacker = attacker ? { name: attacker.name, dx: attacker.npc.x - p.x, dz: attacker.npc.z - p.z } : null;
        this.lastObservation = obs;
    }

    private itemTransfer(index: number, withdraw: boolean): string {
        if (!this.bankIsOpen()) return 'bank: closed';
        const selected: Item | undefined = this.lastObservation?.[withdraw ? 'bankSel' : 'invSel']?.[index];
        const invType = withdraw ? InvType.getId('bank') : InvType.INV;
        const current = this.inventoryItems(invType).find(it => it.slot === selected?.slot);
        if (!matchItem(current, selected)) return 'bank: stale or empty selection';
        const label = withdraw ? 'bank_withdraw' : 'bank_deposit';
        const script = ScriptProvider.getByName(`[label,${label}]`);
        if (!script) throw new Error(`missing game banking script: ${label}`);
        // Execute the actual game's capacity, unbankable-item, and note conversion rules.
        // Item copies with the same ID are fungible; requested quantity is selected-stack count.
        this.player.executeScript(ScriptRunner.init(script, this.player, null,
            [selected!.slot, withdraw ? 1 : selected!.count]), true);
        return `${label}: slot ${selected!.slot} id ${selected!.id} requested ${withdraw ? 1 : selected!.count}`;
    }

    private posKey(): string {
        const p = this.player as any;
        return `${p.x},${p.z}`;
    }

    /** The running option ended now; the next observation's feedback says how. */
    private endOption(reason: EndReason, refused: boolean): void {
        if (this.option) this.optionEnd = { action: this.option.action ?? -1, reason, refused, reported: false };
    }

    private snapshot(): Snapshot {
        const p = this.player;
        const items = new Map<number, number>();
        for (const inv of [InvType.INV, InvType.WORN])
            for (const i of this.inventoryItems(inv)) items.set(i.id, (items.get(i.id) ?? 0) + i.count);
        return { tick: World.currentTick, x: p.x, z: p.z, level: p.level, hp: p.levels[PlayerStat.HITPOINTS] ?? 0,
                 xp: this.totalXp(), items, bankOpen: this.bankIsOpen(),
                 dialogOpen: p.modalChat !== -1 || p.activeScript?.execution === ScriptState.PAUSEBUTTON };
    }

    /** The open chat interface's text, and each choice button's own text. */
    private dialogText(): { text: string; choices: string[] } {
        const chat = this.player.modalChat;
        const text = chat === -1 ? '' : this.text.interfaceText(com => Component.get(com)?.rootLayer === chat);
        const suspended = this.player.activeScript?.execution === ScriptState.PAUSEBUTTON;
        return { text, choices: suspended ? this.player.resumeButtons.map(com => this.text.textOf(com)) : [] };
    }

    /**
     * Unlike the browser environment these watchdogs use GAME TICKS, not wall
     * clock. The browser had to use wall clock because its tick counter came from
     * client state pushes and a stuck player generated almost none; the absence
     * of ticks was the symptom, so it could not also be the clock. Inside the
     * engine `World.currentTick` always advances, so the principled unit works.
     */
    private optionBusy(): boolean {
        if (!this.option) return false;
        const now = World.currentTick;
        const pos = this.posKey();
        if (pos !== this.option.lastPos) {
            this.option.lastPos = pos;
            this.option.lastProgressTick = now;
        }
        const act = this.option.act;
        if (act) {
            const [xp, items] = [this.totalXp(), this.itemState()];
            if (xp !== act.lastXp || items !== act.lastItems || this.player.modalMain !== -1) {
                act.lastXp = xp; act.lastItems = items; this.option.lastProgressTick = now;
            }
        }
        if (this.option.done()) {
            const act = this.option.act;
            // A message ("You need a net") is no progress; an opened bank or shop is, and so is a
            // loc the act changed (an opened door is replaced by its open form).
            const refused = !!act && this.totalXp() === act.xp && this.itemState() === act.items && this.player.modalMain === -1
                && (!act.loc || this.locPresent(act.loc));
            if (refused) this.refusedTargets.set(act!.key, now + REFUSED_TICKS);
            this.lastOptionNote = `${this.option.label} done`;
            this.endOption('done', refused);
            this.option = null;
            return false;
        }
        const stalled = this.option.label.startsWith('nav') ? now - this.option.lastProgressTick > NAV_STALL_TICKS
            : !!act && now - this.option.lastProgressTick > ACT_STALL_TICKS;
        if (stalled && act) this.refusedTargets.set(act.key, now + REFUSED_TICKS);
        const expired = now - this.option.startTick > OPTION_HARD_TIMEOUT_TICKS;
        if (stalled || expired) {
            this.lastOptionNote = `${this.option.label} ${stalled ? 'stalled-cleared' : 'stale-cleared'}`;
            this.endOption(stalled ? 'stalled' : 'expired', stalled && !!act);
            this.option = null;
            return false;
        }
        return true;
    }

    /** Whether NAV could walk from `from` to `to`: the snapshot's route, in segments the live map allows. */
    private walksBack(from: { x: number; z: number }, to: { x: number; z: number }): boolean {
        const route = findLongPath(0, from.x, from.z, to.x, to.z, 500, []);
        const end = route[route.length - 1];
        if (!end || chebyshev(end.x, end.z, to.x, to.z) > 1) return false;
        let at = from;
        for (let i = 15; ; i += 15) {
            const next = route[Math.min(i, route.length - 1)];
            if (chebyshev(at.x, at.z, next.x, next.z) > 1) {
                const path = findPath(0, at.x, at.z, next.x, next.z);
                const last = path[path.length - 1];
                if (last === undefined || chebyshev((last >>> 14) & 16383, last & 16383, next.x, next.z) > 1) return false;
            }
            if (i >= route.length - 1) return true;
            at = next;
        }
    }

    private begin(label: string, done: () => boolean): void {
        const now = World.currentTick;
        this.option = { label, startTick: now, lastPos: this.posKey(), lastProgressTick: now, done };
    }

    /** What the combat-style rows can change (core/mask.ts): com_mode, the style count of the wielded weapon's
     * combat_style_table row ([proc,combat_get_weapon_style_data], as [proc,player_combat_stat] counts it) and
     * styleStale. Not a policy input (rl/obs.py does not read it). */
    private styleFacts(): { mode: number; count: number; stale: boolean } {
        const varp = VarPlayerType.getByName('com_mode');
        const proc = ScriptProvider.getByName('[proc,combat_get_weapon_style_data]');
        const table = DbTableType.getByName('combat_style_table');
        const column = table ? table.columnNames.indexOf('damagestyle') : -1;
        if (!varp || !proc || !table || column < 0) throw new Error('combat style data unavailable');
        const weapon = this.player.getInventory(InvType.WORN)?.get(WEARPOS_RHAND)?.id ?? -1;
        const lookup = ScriptRunner.init(proc, this.player, null, [weapon]);
        if (ScriptRunner.execute(lookup) !== ScriptState.FINISHED) throw new Error('combat style lookup failed');
        const row = DbRowType.get(lookup.popInt());
        const count = row.tableId === table.id ? row.columnValues[column].length / table.types[column].length : 0;
        if (!Number.isInteger(count) || count < 1) throw new Error(`weapon ${weapon}: no combat styles`);
        return { mode: this.player.getVar(varp.id) as number, count, stale: this.styleStale };
    }

    private mask(obs: Record<string, any>): number[] {
        obs.maskVersion = MASK_CONTRACT;
        obs.heldVersion = HELD_CONTRACT;
        if (this.continuity && obs.lifecycle?.phase === 'dying') {
            const mask = new Array(N_ACTIONS).fill(0); mask[0] = 1; return mask;
        }
        return buildMask(obs, {
            allowedActions: this.cfg.allowedActions
        });
    }

    // ---- effector ---------------------------------------------------------

    /**
     * Rule 1, as the 2004 client keeps it (act-v3): route first with the client's own tryMove
     * (core/clicks.ts) over the collision the client holds of its scene, from the player's server tile (a click issued
     * mid-walk is measured from there, a declared aid), then send what the client sends.
     */
    private clientRoute(dest: { x: number; z: number }, arrival: Arrival, nearest: boolean) {
        const p = this.player as any;
        return tryMove((x, z) => clientFlags(routefinder.collisionFlags.get(x, z, p.level)), sceneBase(p.originX, p.originZ),
            p.x, p.z, dest.x, dest.z, nearest, arrival);
    }

    /** MoveClickHandler, run off: clearPendingAction, then the turn points as the walk; a route that is only the
     * player's own tile clears the walk and leaves no repath (so no chase follows it); then the walk trigger. */
    private sendRoute(waypoints: Array<{ x: number; z: number }>): void {
        const p = this.player as any;
        p.clearPendingAction();
        if (waypoints.length === 1 && waypoints[0].x === p.x && waypoints[0].z === p.z) {
            p.queueWaypoints([]);
            p.setAllowRepath(AllowRepath.NONE);
        } else {
            p.queueWaypoints(waypoints.map(w => CoordGrid.packCoord(p.level, w.x, w.z)));
        }
        p.processWalktrigger();
    }

    /** route-v1 for this decision: what the click or ACT asked for and what was sent. */
    private record(kind: RouteRecord['kind'], action: number, target: { x: number; z: number },
                   route: ReturnType<EngineAgent['clientRoute']>, extra: Partial<RouteRecord>): void {
        const p = this.player as any;
        this.decisionRoute = this.latestRoute = { version: ROUTE_RECORD, kind, action, tick: World.currentTick,
            source: { x: p.x, z: p.z, level: p.level }, offset: null, target: { x: target.x, z: target.z }, thing: null,
            via: route ? route.via : 'none', end: route ? route.waypoints[route.waypoints.length - 1] : null,
            refused: null, waypoints: [], ...extra };
    }

    /** The latest route-v1 record (spectator and checks only; never a policy input), or null since reset. */
    get lastRoute(): RouteRecord | null {
        return this.latestRoute;
    }

    /**
     * CLICK_k: a minimap click at the slot's offset, tryMove with its 3 x 3 fallback, then MoveClickHandler. Refused
     * while the player is delayed, and when no route is found nothing is sent: an earlier walk continues and the
     * decision records the refusal. The option ends when the walk ends, or stalls after NAV_STALL_TICKS.
     */
    private click(action: number): string {
        const p = this.player as any;
        const slot = this.lastObservation?.clicks?.[clickSlot(action)!];
        if (!slot) return 'click: slot empty';
        const target = { x: p.x + slot.dx, z: p.z + slot.dz }, offset = { dx: slot.dx, dz: slot.dz };
        const route = p.delayed ? null : this.clientRoute(target, TILE, true);
        if (!route) {
            const refused = p.delayed ? 'delayed' : 'no route';
            this.record('click', action, target, null, { offset, refused });
            this.optionEnd = { action, reason: 'failed', refused: true, reported: false };
            return `click ${slot.dx},${slot.dz}: ${p.delayed ? 'refused while delayed' : 'no route'}`;
        }
        this.sendRoute(route.waypoints);
        this.record('click', action, target, route, { offset, waypoints: route.waypoints });
        this.begin(optionLabelFor(action)!, () => !this.player.hasWaypoints());
        const end = route.waypoints[route.waypoints.length - 1];
        return `click ${slot.dx},${slot.dz} to ${end.x},${end.z}`;
    }

    /**
     * Completion predicate for an ACT option.
     *
     * `!player.target` alone is WRONG and cost a measured bug: the engine clears
     * the target the moment an interaction ends OR before it has begun, so an
     * option issued from 10 tiles away reported "done" while the player was still
     * walking. The policy then re-issued it next step, the new walk reset the walk
     * queue, and the agent oscillated without ever reaching the tree: 80
     * consecutive CHOP actions produced 0 woodcutting xp. This is the same
     * self-interruption that gave v21f 54 CHOP actions and no experience.
     *
     * An option is finished only when experience landed, or when the player is
     * genuinely idle: no interaction target AND no queued steps, after a short
     * grace period for the engine to pick the interaction up. With decisionPoints
     * it runs until the player stops, as a player's click does: a net keeps fishing
     * until the spot moves or the inventory fills (978 decisions in a fishing
     * episode when every catch ended it).
     */
    private actDone(xpBefore: number, startTick: number): () => boolean {
        const IDLE_GRACE_TICKS = 4;
        return () => {
            if (!this.cfg.decisionPoints && this.totalXp() > xpBefore) return true;
            const idle = !this.player.target && !this.player.hasWaypoints();
            return idle && World.currentTick - startTick > IDLE_GRACE_TICKS;
        };
    }

    /** Experience over all skills: any skill's progress shows an option changed something. */
    /** Whether a loc of this type still stands on this tile (an opened door is replaced). */
    private locPresent(loc: { type: number; x: number; z: number; level: number }): boolean {
        for (const l of World.gameMap.getZone(loc.x, loc.z, loc.level).getAllLocsSafe())
            if (l.type === loc.type && l.x === loc.x && l.z === loc.z) return true;
        return false;
    }

    private totalXp(): number {
        return this.player.stats.reduce((a: number, b: number) => a + b, 0);
    }

    /** What the player holds, to tell whether an option changed it. */
    private itemState(): string {
        return this.inventoryItems(InvType.INV).map(i => `${i.slot}:${i.id}:${i.count}`).join(',');
    }

    /**
     * A loc op by the client's rule: tryMove to the loc (its rotated size and forceapproach, or its wall shape; no
     * fallback), a route sent as MoveClickHandler takes it, then OpLocHandler or OpLocUHandler: refused while the
     * player is delayed; clearPendingAction; the interaction. With no route only the op is sent: an earlier walk
     * continues and the op resolves when it ends ("I can't reach that!" if out of reach). Rule 2: caller rescans.
     */
    private interactLoc(action: number, target: { loc: any; rawOps: (string | null)[]; name: string }, re: RegExp | null,
                        option: string, label: string, using?: { slot: number; id: number }): string {
        // The op index must come from the RAW option array, holes included: the
        // trigger is APLOC1 + (op - 1), so a compacted list silently fires the
        // wrong op. See the note on rawOpsOf in observe.ts.
        const opIdx = re ? opIndexOf(target.rawOps, re) : -1;
        if (!using && opIdx === -1) return `${label}: no matching op`;
        const p = this.player as any, loc = target.loc, type = LocType.get(loc.type);
        const thing = { kind: 'loc' as const, id: loc.type, name: target.name, option };
        if (p.delayed) {
            this.record('act', action, loc, null, { thing, refused: 'delayed' });
            return `${label}: refused while delayed`;
        }
        const route = this.clientRoute(loc, locArrival(loc.shape, loc.angle, type.width, type.length, type.forceapproach), false);
        if (route) this.sendRoute(route.waypoints);
        this.record('act', action, loc, route, { thing, refused: route ? null : 'no route', waypoints: route?.waypoints ?? [] });
        p.clearPendingAction();
        if (using) { p.lastUseItem = using.id; p.lastUseSlot = using.slot; }
        const ok = this.player.setInteraction(Interaction.ENGINE, loc, using ? ServerTriggerType.APLOCU : ServerTriggerType.APLOC1 + opIdx);
        p.opcalled = true;
        this.begin(label, this.actDone(this.totalXp(), World.currentTick));
        return ok ? `${label} ${using ? `use ${using.id} ` : ''}issued` : `${label}: setInteraction refused`;
    }

    /**
     * An npc op by the client's rule: tryMove beside its tile (1 x 1, no fallback), then OpNpcHandler or OpNpcUHandler:
     * refused while the player or the npc is delayed or while the npc is not in the player's npc list; clearPendingAction;
     * the interaction. The server's own naive chase follows a moving npc (Player.pathToPathingTarget).
     */
    private interactNpc(action: number, target: { npc: any; rawOps: (string | null)[]; name: string }, re: RegExp | null,
                        option: string, label: string, using?: { slot: number; id: number }): string {
        const opIdx = re ? opIndexOf(target.rawOps, re) : -1;
        if (!using && opIdx === -1) return `${label}: no matching op`;
        const p = this.player as any, npc = target.npc;
        const thing = { kind: 'npc' as const, id: npc.type, name: target.name, option };
        const refused = p.delayed ? 'delayed' : npc.delayed ? 'npc delayed' : !rsbuf.hasNpc(p.slot, npc.nid) ? 'npc not in view' : null;
        if (refused) {
            this.record('act', action, npc, null, { thing, refused });
            return `${label}: refused (${refused})`;
        }
        const route = this.clientRoute(npc, ENTITY, false);
        if (route) this.sendRoute(route.waypoints);
        this.record('act', action, npc, route, { thing, refused: route ? null : 'no route', waypoints: route?.waypoints ?? [] });
        p.clearPendingAction();
        if (using) { p.lastUseItem = using.id; p.lastUseSlot = using.slot; }
        const ok = this.player.setInteraction(Interaction.ENGINE, npc, using ? ServerTriggerType.APNPCU : ServerTriggerType.APNPC1 + opIdx);
        p.opcalled = true;
        this.begin(label, this.actDone(this.totalXp(), World.currentTick));
        return ok ? `${label} ${using ? `use ${using.id} ` : ''}issued` : `${label}: setInteraction refused`;
    }

    /**
     * Run an inventory option as OpHeldHandler.ts:16-71 does (held-v2): refused while delayed; lastItem; clearPendingAction
     * (the backpack's layer is never the main modal), which ends a running chop, mine, net, attack or light; the script.
     */
    private opHeld(slot: number, re: RegExp): string {
        const inv = this.player.getInventory(InvType.INV);
        const item = inv?.get(slot);
        if (!item) return 'opheld: empty slot';
        if (this.player.delayed) return 'opheld: refused while delayed';
        const objType = ObjType.get(item.id) as unknown as { id: number; category: number; iop?: (string | null)[]; debugname?: string };
        const iop = objType.iop ?? [];
        const opIdx = iop.findIndex(o => !!o && re.test(o));
        if (opIdx === -1) return `opheld: no matching iop on ${objType.debugname}`;
        this.player.lastItem = item.id;
        this.player.lastSlot = slot;
        this.player.clearPendingAction();
        const script = ScriptProvider.getByTrigger(ServerTriggerType.OPHELD1 + opIdx, objType.id, objType.category);
        if (!script) return `opheld: no script for op ${opIdx + 1}`;
        this.player.executeScript(ScriptRunner.init(script, this.player), true);
        return `opheld ${iop[opIdx]} slot ${slot}`;
    }

    /**
     * DROP. `iop` rarely names "Drop" (the client renders it from op 5), so this
     * goes straight to the OPHELD5 trigger and falls back to removing the item. OpHeldHandler's op 5 (held-v2):
     * refused while delayed, and clearPendingAction before the trigger.
     */
    private dropSlot(slot: number): string {
        const inv = this.player.getInventory(InvType.INV);
        const item = inv?.get(slot);
        if (!item) return 'drop: empty slot';
        if (this.player.delayed) return 'drop: refused while delayed';
        const objType = ObjType.get(item.id) as unknown as { id: number; category: number };
        this.player.lastItem = item.id;
        this.player.lastSlot = slot;
        this.player.clearPendingAction();
        const script = ScriptProvider.getByTrigger(ServerTriggerType.OPHELD5, objType.id, objType.category);
        if (script) {
            this.player.executeScript(ScriptRunner.init(script, this.player), true);
            return `drop slot ${slot}`;
        }
        this.player.invDelSlot(InvType.INV, slot);
        return `drop slot ${slot} (direct)`;
    }

    /** Take a ground item by the client's rule: tryMove onto its tile, else beside it, then OpObjHandler: refused while
     * the player is delayed; clearPendingAction; the interaction on its Take op. */
    private pickUp(action: number, target: { x: number; z: number; id: number; name: string }): string {
        const p = this.player as any;
        const obj = World.getObj(target.x, target.z, p.level, target.id, this.player.hash64);
        if (!obj) return `pick: ${target.name} is gone`;
        const ops = (ObjType.get(target.id) as unknown as { op?: (string | null)[] })?.op ?? [];
        let opIdx = ops.findIndex(o => !!o && /take/i.test(o));
        if (opIdx === -1) opIdx = 2; // 'Take' is conventionally op 3 (OPOBJ3)
        const thing = { kind: 'obj' as const, id: target.id, name: target.name, option: 'Take' };
        if (p.delayed) {
            this.record('act', action, target, null, { thing, refused: 'delayed' });
            return `pick ${target.name}: refused while delayed`;
        }
        const onto = this.clientRoute(target, TILE, false), route = onto ?? this.clientRoute(target, ENTITY, false);
        if (route) this.sendRoute(route.waypoints);
        this.record('act', action, target, route, { thing, refused: route ? null : 'no route', waypoints: route?.waypoints ?? [],
            via: onto ? onto.via : route ? 'adjacent' : 'none' });
        p.clearPendingAction();
        const ok = this.player.setInteraction(Interaction.ENGINE, obj, ServerTriggerType.APOBJ1 + opIdx);
        p.opcalled = true;
        this.begin('pick_item', () => !this.player.target);
        return ok ? `pick ${target.name}` : `pick ${target.name}: setInteraction refused`;
    }

    /** Resume a suspended dialog with no client packet. See CHANGELOG 15:10 §1. */
    private resumeDialog(choice: number | null): string {
        const script = this.player.activeScript;
        if (!script || script.execution !== ScriptState.PAUSEBUTTON) return 'dialog: nothing suspended';
        const buttons = this.player.resumeButtons.slice();
        if (choice !== null && buttons.length > 0) {
            // A choice must be answered on a registered button: ResumePauseButton
            // refuses while resumeButtons is non-empty (ResumePauseButtonHandler.ts:16).
            const comId = buttons[Math.min(choice, buttons.length - 1)];
            this.player.lastCom = comId;
            this.player.resumeButtons = [];
            this.player.executeScript(script, true, true);
            return `dialog choice ${choice} -> com ${comId}`;
        }
        if (buttons.length > 0) return 'dialog: choice required';
        this.player.executeScript(script, true, true);
        return 'dialog continue';
    }

    /**
     * ACT_k: use the published option on the thing behind actSel[k], found again in this
     * step's scan (rule 2), never another one: no automatic retargeting.
     */
    private act(action: number, scan: ScanResult): string {
        const selected: Affordance | undefined = this.lastObservation?.actSel?.[action - A_ACT0];
        const source = this.actTargets[action - A_ACT0];
        if (!selected || !source) return 'act: slot empty';
        const label = optionLabelFor(action)!, verb = exactly(selected.option);
        const what = `${selected.option} ${selected.kind}=${selected.id}@${selected.x},${selected.z},${selected.level}`;
        let note: string;
        const using = this.using;
        this.using = null; // one use, as a click spends the client's "Use" cursor
        if (source.kind === 'loc') {
            const loc = scan.locs.find(l => l.loc.type === selected.id && l.loc.x === selected.x && l.loc.z === selected.z);
            if (!loc) return `act: ${what} gone`;
            note = this.interactLoc(action, loc, using ? null : verb, using ? 'Use' : selected.option, label, using ?? undefined);
        } else if (source.kind === 'npc') {
            // The same npc, wherever it has walked to since.
            const npc = scan.npcs.find(n => n.npc === source.entry.npc);
            if (!npc || !npc.npc.isValid()) return `act: ${what} gone`;
            note = this.interactNpc(action, npc, using ? null : verb, using ? 'Use' : selected.option, label, using ?? undefined);
        } else {
            const obj = scan.objs.find(o => o.id === selected.id && o.x === selected.x && o.z === selected.z);
            if (!obj) return `act: ${what} gone`;
            note = this.pickUp(action, obj);
        }
        if (this.option) {
            const [xp, items] = [this.totalXp(), this.itemState()];
            const loc = source.kind === 'loc' ? { type: selected.id, x: selected.x, z: selected.z, level: selected.level } : undefined;
            this.option.act = { key: affordanceKey(selected), xp, items, lastXp: xp, lastItems: items, loc };
        }
        return `${note} ${what}`;
    }

    /** Use a picked item on another held item as OpHeldUHandler does: refused while delayed (held-v2), then its script
     * lookup, swaps included. */
    private useOnHeld(slot: number, using: { slot: number; id: number }): string {
        const item = this.player.getInventory(InvType.INV)?.get(slot);
        if (!item) return 'use: empty slot';
        const p = this.player;
        if (p.delayed) return 'use: refused while delayed';
        const objType = ObjType.get(item.id) as any, useType = ObjType.get(using.id) as any;
        p.lastItem = item.id; p.lastSlot = slot; p.lastUseItem = using.id; p.lastUseSlot = using.slot;
        const swap = () => {
            [p.lastItem, p.lastUseItem] = [p.lastUseItem, p.lastItem];
            [p.lastSlot, p.lastUseSlot] = [p.lastUseSlot, p.lastSlot];
        };
        let script = ScriptProvider.getByTriggerSpecific(ServerTriggerType.OPHELDU, objType.id, -1);
        if (!script) { script = ScriptProvider.getByTriggerSpecific(ServerTriggerType.OPHELDU, useType.id, -1); swap(); }
        if (!script && objType.category !== -1)
            script = ScriptProvider.getByTriggerSpecific(ServerTriggerType.OPHELDU, -1, CategoryType.get(objType.category).id);
        if (!script && useType.category !== -1) {
            script = ScriptProvider.getByTriggerSpecific(ServerTriggerType.OPHELDU, -1, CategoryType.get(useType.category).id);
            swap();
        }
        p.clearPendingAction();
        if (!script) {
            // OpHeldUHandler.ts:117-126: its debug line in a debug world (the default), then its literal.
            if (Environment.node.debug) p.messageGame(`No trigger for [opheldu,${objType.debugname}]`);
            p.messageGame('Nothing interesting happens.');
            return 'use: nothing interesting happens';
        }
        p.executeScript(ScriptRunner.init(script, p), true);
        return `use ${using.id} on ${item.id}`;
    }

    private dispatch(action: number, scan: ScanResult): string {
        const p = this.player as any;

        // Only actions legal in the published observation reach the game.
        if (!this.lastObservation?.mask?.[action]) return 'action not legal in published observation';
        if (this.itemPages.turn(action)) return 'item page changed';
        if (action >= A_WITHDRAW0 && action < A_WITHDRAW0 + N_INV_SEL) return this.itemTransfer(action - A_WITHDRAW0, true);
        if (action >= A_DEPOSIT0 && action < A_DEPOSIT0 + N_INV_SEL) return this.itemTransfer(action - A_DEPOSIT0, false);
        if (action >= A_DROP0 && action < A_DROP0 + N_INV_SEL) {
            const selected = this.lastObservation?.invSel?.[action - A_DROP0];
            const current = this.inventoryItems(InvType.INV).find(i => i.slot === selected?.slot);
            return matchItem(current, selected) ? this.dropSlot(selected.slot) : 'drop: stale selection';
        }

        if (action === 0) return 'wait';

        if (useAction(action)) {
            const selected = this.lastObservation?.invSel?.[action - A_USE0];
            const current = this.inventoryItems(InvType.INV).find(i => i.slot === selected?.slot);
            if (!selected || !matchItem(current, selected)) return 'use: stale selection';
            this.using = { slot: selected.slot, id: selected.id, name: selected.name };
            return `use: picked ${selected.name}`;
        }
        if (this.using && action === A_CLOSE_IF) {
            this.using = null;
            return 'use: put back';
        }
        if (this.using && heldAction(action)) {
            const target = this.lastObservation?.heldSel?.[action - A_HELD0];
            const using = this.using;
            this.using = null;
            return target ? this.useOnHeld(target.slot, using) : 'use: slot empty';
        }

        if (clickSlot(action) !== null) return this.click(action);

        if (actAction(action)) return this.act(action, scan);

        if (heldAction(action)) {
            const selected = this.lastObservation?.heldSel?.[action - A_HELD0];
            const current = this.inventoryItems(InvType.INV).find(i => i.slot === selected?.slot);
            return selected && matchItem(current, selected) ? this.opHeld(selected.slot, exactly(selected.option))
                : 'held: stale selection';
        }

        if (action === A_DIALOG_CONTINUE) return this.resumeDialog(null);
        if (action >= A_DIALOG_CHOICE0 && action < A_DIALOG_CHOICE0 + 4) return this.resumeDialog(action - A_DIALOG_CHOICE0);

        if (action >= A_STYLE0 && action < A_STYLE0 + 4) {
            // varp 43 = com_mode (content/pack/varp.pack). set_attackstyle also
            // runs ~player_combat_stat so %damagestyle updates this tick.
            const style = action - A_STYLE0;
            const script = ScriptProvider.getByName('[proc,set_attackstyle]');
            if (script) {
                // Protected: it runs only without protection or a delay (Player.runScript), and then recomputes.
                if (!this.player.protect && !this.player.delayed) this.styleStale = false;
                this.player.executeScript(ScriptRunner.init(script, this.player, null, [style]), true);
                return `style ${style}`;
            }
            const varp = VarPlayerType.getByName('com_mode');
            this.player.setVar(varp ? varp.id : 43, style);
            return `style ${style} (varp)`;
        }

        if (action === A_CLOSE_IF) {
            // Rule 3 applies to the READER, not here: closing is the point.
            if (this.player.activeScript?.execution === ScriptState.PAUSEBUTTON) return this.resumeDialog(this.player.resumeButtons.length ? 0 : null);
            this.player.closeModal();
            return 'close_if';
        }

        return `unimplemented action ${action}`;
    }

    // ---- step -------------------------------------------------------------

    async step(action: number): Promise<Record<string, any>> {
        const tStep0 = Date.now();
        this.decision++;
        if (this.lastMaskSum > 1) this.charged++;
        const ticks = K_TICKS + this.paceTicks();
        this.episodeTicks += ticks;

        let actionNote = 'noop';
        // What the policy chose, read from the observation it chose from: the slot may hold
        // another thing in the next one.
        const meaning: ActionMeaning = describeAction(this.lastObservation, action);
        const before = this.snapshot();
        const lifecycleAtStart = this.continuity ? this.continuousState() : null;
        const deadAtStart = lifecycleAtStart ? lifecycleAtStart.phase === 'dying' : this.death.dead || !this.isAdmitted();
        const busy = !deadAtStart && this.optionBusy();
        const running = this.option;
        this.decisionRoute = null;
        const tDispatch0 = Date.now();
        try {
            // WAIT never supersedes: it means "let whatever is running continue".
            if (deadAtStart) {
                this.option = null;
                actionNote = this.continuity ? 'dying: awaiting natural respawn; action not dispatched' : 'terminal: action not dispatched';
            } else if (busy && (action === 0 || optionLabelFor(action) === this.option?.label)) {
                actionNote = `continue:${this.option?.label}`;
            } else {
                if (busy) {
                    this.lastOptionNote = `${this.option?.label} superseded by action ${action}`;
                    this.endOption('superseded', false);
                    this.option = null;
                }
                actionNote = this.dispatch(action, scanWorld(this.player));
            }
        } catch (e) {
            actionNote = `error: ${(e as Error)?.message ?? e}`;
        }
        if (this.option && this.option !== running) this.option.action = action;
        const tDispatchMs = Date.now() - tDispatch0;

        const tTick0 = Date.now();
        if (this.continuity || !deadAtStart) await this.waitTicks(ticks);
        if (this.cfg.decisionPoints && !this.continuity && !deadAtStart) {
            // Keep waiting while the option runs and nothing happens, in the same paced
            // chunks a WAIT would take, so ticks are charged exactly as before.
            const hpAtDecision = this.player.levels[PlayerStat.HITPOINTS] ?? 0;
            let waited = 0;
            while (keepWaiting({ busy: this.optionBusy(), dead: this.death.dead || !this.isAdmitted(),
                    hp: this.player.levels[PlayerStat.HITPOINTS] ?? 0, hpAtDecision, waited,
                    dialog: this.player.modalState !== 0 || this.player.modalChat !== -1
                        || this.player.activeScript?.execution === ScriptState.PAUSEBUTTON })
                    && !isTruncated({ charged: this.charged, episodeTicks: this.episodeTicks }, this.cfg)) {
                const chunk = K_TICKS + this.paceTicks();
                this.episodeTicks += chunk;
                waited += chunk;
                await this.waitTicks(chunk);
            }
        }
        const tTickMs = Date.now() - tTick0;

        const tObs0 = Date.now();
        const lifecycleNow = this.continuity ? this.continuousState() : null;
        const dead = lifecycleNow ? lifecycleNow.phase === 'dying' : this.death.dead || !this.isAdmitted();
        if (dead || (lifecycleNow && lifecycleAtStart && lifecycleNow.deathSequence !== lifecycleAtStart.deathSequence)) this.option = null;
        const truncated = (this.continuity || !dead) && isTruncated({ charged: this.charged, episodeTicks: this.episodeTicks }, this.cfg);
        const scan = scanWorld(this.player);
        const obs = buildObs(this.player, scan, { decision: this.decision, done: lifecycleNow ? truncated : dead || truncated, truncated, isDead: dead });
        // Settle completion now. Reading `option !== null`
        // reported an option busy for one step after it finished, so a level-up
        // arrived mid-option here and after it in Chromium, where imitation then
        // waited on the dialog for the rest of the episode. Before listing places,
        // so a place a walk just found no route to is already gone.
        const busyAfter = !dead && this.optionBusy();
        this.projectItems(obs, scan);
        if (lifecycleNow) obs.lifecycle = lifecycleNow;
        obs.deathTick = lifecycleNow ? lifecycleNow.deathTick : this.death.deathTick;
        obs.action = action;
        obs.actionNote = actionNote;
        obs.optionBusy = busyAfter;
        obs.optionAction = this.option?.action ?? -1;
        obs.lastOptionNote = this.lastOptionNote;
        // Evidence for checks, logs and the spectator, not a policy input (rl/obs.py does not read it).
        obs.route = this.decisionRoute;
        if (TICK_TRACE) { obs.tickTrace = this.tickTrace; this.tickTrace = []; }
        obs.mask = this.mask(obs);
        obs.feedback = feedback(meaning, before, this.snapshot(), busyAfter, this.optionEnd,
                                this.text.publish(World.currentTick), this.dialogText());
        obs.charged = this.charged;
        this.lastMaskSum = (obs.mask as number[]).reduce((a: number, b: number) => a + b, 0);
        const tObsMs = Date.now() - tObs0;

        obs.timing = { dispatch: tDispatchMs, tick: tTickMs, obs: tObsMs, total: Date.now() - tStep0 };
        return obs;
    }

    close(): void {
        if (this.closed) return;
        this.closed = true;
        clearInterval(this.heartbeat);
        try {
            this.unbindLifecycle();
        } catch {
            /* the world may already have dropped it */
        } finally {
            this.player.loggingOut = true;
        }
    }
}
