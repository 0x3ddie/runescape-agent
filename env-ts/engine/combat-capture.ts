/** Passive raw melee evidence: every hit, damage, defeat, removal and respawn of an enemy life.
 * This is NOT a kill counter; baseline/native_combat.py audits the events. Native methods always
 * execute even if observation fails; a sticky failure makes the trace unusable.
 *
 * Source chain (pinned content, skill_combat): player_melee_attack adds the capped damage as hero
 * points (npc_heropoints) and queues it on the npc (npc_queue(2, $damage, 0)). The npc's ai_queue2
 * applies it (npc_default_damage: npc_damage), and at 0 hitpoints queues ai_queue3, the death
 * script (npc_death: npc_del; the drop goes to npc_findhero, the most hero points). A queued damage
 * runs later as the npc's own script, so the hitter is the entity whose script queued it: the
 * queue entry's argument list is tagged at npc_queue and followed into ScriptRunner.init.
 * Hitpoint changes no npc_damage explains (regeneration, drains) are recorded at reconciliation.
 */
import { eventLimit } from './resource-capture';
type Native = any;
export type Actor = 'agent' | 'player' | 'npc' | 'world';
export interface CombatCaptureHost {
    handlers: Record<number, (state: Native) => any>;
    /** NPC_QUEUE and NPC_DAMAGE; with playerQueue (QUEUE), attacks on the agent are recorded too (agentDamageScript) */
    opcodes: { queue: number; damage: number; playerQueue?: number };
    /** Player.prototype: enqueueScript(script, type, delay, args), which QUEUE calls */
    playerPrototype?: Native;
    /** Script id of [queue,combat_damage_player]: an npc's attack queues it on its target with the planned damage */
    agentDamageScript?: number;
    /** Npc.prototype: enqueueScript(trigger, delay, arg), called only by NPC_QUEUE */
    npcPrototype: Native;
    /** ScriptRunner: a queued npc script starts in init(script, self, target, request.args) */
    runner: Native;
    /** World: addNpc(npc, duration, firstSpawn), removeNpc(npc, duration), cycle() */
    world: Native;
    /** Trigger id of ai_queue2, the damage queue */
    damageTrigger: number;
    hitpoints: number;
    /** Declared enemy npc types: their lives are followed whoever hits them. */
    enemyTypes: number[];
    tick(): number;
    /** The entry enqueueScript just appended, read without moving the queue cursor. */
    newestRequest(npc: Native): Native;
    heroes(npc: Native, player: Native): { agent: number; other: number };
    actor(self: Native, player: Native): Actor;
    agentHp(player: Native): number;
    /** The resource window this capture shares is open (arm to its terminal). */
    open(player: Native): boolean;
}
type Life = { key: string; npc: Native; seq: number; type: number; hp: number; removed: boolean };
type Binding = { player: Native; identity: Native; start: number; end: number; limit: number; sequence: number; events: Native[];
    failure: string | null; lives: Map<Native, Life>; seqs: Map<Native, number>; agentHp: number; armedHp: number };
type Tag = { binding: Binding; life: Life; hitter: Actor; hit: number };
const require = (ok: unknown, message: string) => { if (!ok) throw new Error(message); };

export function installCombatCapture(host: CombatCaptureHost, maxEvents = 20000) {
    require(['enqueueScript'].every(k => typeof host.npcPrototype[k] === 'function') &&
        typeof host.runner.init === 'function' && ['addNpc', 'removeNpc', 'cycle'].every(k => typeof host.world[k] === 'function') &&
        [host.opcodes.queue, host.opcodes.damage].every(op => typeof host.handlers[op] === 'function'),
        'missing native combat surfaces');
    const enemies = new Set(host.enemyTypes);
    let binding: Binding | null = null;
    let queuing: Native = null;
    const tags = new WeakMap<object, Tag>(), started = new WeakMap<object, Tag>();
    const live = () => binding && !binding.failure && host.open(binding.player) ? binding : null;
    const safe = (b: Binding, fn: () => void) => { try { fn(); } catch (error) { b.failure ??= String((error as Error)?.message ?? error); } };
    function record(b: Binding, kind: string, data: Native) {
        require(b.events.length < b.limit, 'combat capture overflow');
        const tick = host.tick();
        require(Number.isSafeInteger(tick) && tick >= b.start && tick <= b.end, 'combat event outside the window');
        b.events.push(structuredClone({ sequence: ++b.sequence, tick, kind, ...data }));
        return b.sequence;
    }
    function hpOf(npc: Native) {
        const hp = npc.levels?.[host.hitpoints], base = npc.baseLevels?.[host.hitpoints];
        require(Number.isSafeInteger(hp) && hp >= 0 && Number.isSafeInteger(base) && base > 0, 'invalid npc hitpoints');
        return { hp, base };
    }
    /** The npc's current life, recorded when first followed. */
    function lifeOf(b: Binding, npc: Native, spawned: string | null = null): Life {
        let life = b.lives.get(npc);
        if (life && !spawned) return life;
        const seq = (b.seqs.get(npc) ?? -1) + 1;
        b.seqs.set(npc, seq);
        const { hp, base } = hpOf(npc);
        require(Number.isSafeInteger(npc.nid) && Number.isSafeInteger(npc.type), 'invalid npc identity');
        life = { key: `${npc.nid}:${seq}`, npc, seq, type: npc.type, hp, removed: false };
        b.lives.set(npc, life);
        record(b, 'life', { life: life.key, nid: npc.nid, type: npc.type, hp, baseHp: base, x: npc.x, z: npc.z,
            level: npc.level, previous: spawned });
        return life;
    }
    function reconcile(b: Binding, life: Life) {
        if (life.removed) return;
        require(life.npc.type === life.type, 'enemy type changed within a life');
        const { hp } = hpOf(life.npc);
        if (hp !== life.hp) { record(b, 'hp', { life: life.key, hpBefore: life.hp, hpAfter: hp }); life.hp = hp; }
    }
    const followed = (b: Binding, npc: Native, hitter: Actor) => b.lives.has(npc) || enemies.has(npc.type) || hitter === 'agent';

    const queue = host.handlers[host.opcodes.queue];
    host.handlers[host.opcodes.queue] = function (state: Native) {
        const previous = queuing;
        queuing = state;
        try { return queue.call(this, state); } finally { queuing = previous; }
    };
    const enqueue = host.npcPrototype.enqueueScript;
    host.npcPrototype.enqueueScript = function (trigger: number, delay = 0, arg = 0) {
        const result = enqueue.call(this, trigger, delay, arg);
        const b = live();
        if (b && trigger === host.damageTrigger) safe(b, () => {
            const hitter = host.actor(queuing?.self ?? null, b.player);
            if (!followed(b, this, hitter)) return;
            const request = host.newestRequest(this);
            require(request && Array.isArray(request.args) && request.lastInt === arg && request.delay === delay,
                'queued damage entry not found');
            const life = lifeOf(b, this);
            reconcile(b, life);
            const { base } = hpOf(this);
            const hit = record(b, 'hit', { life: life.key, hitter, damage: arg, delay, hp: life.hp, baseHp: base,
                heroes: host.heroes(this, b.player) });
            tags.set(request.args, { binding: b, life, hitter, hit });
        });
        return result;
    };
    const init = host.runner.init;
    host.runner.init = function (...args: Native[]) {
        const state = init.apply(this, args);
        const tag = args[3] && tags.get(args[3]);
        if (tag) started.set(state, tag);
        return state;
    };
    const damage = host.handlers[host.opcodes.damage];
    // Every attack on the agent, a zero included (sim/check.py's npc-side rules): an npc's script queues
    // combat_damage_player on its target with the planned damage; hitpoints alone show only the damaging ones.
    const playerQueue = host.opcodes.playerQueue;
    if (playerQueue !== undefined) {
        require(typeof host.handlers[playerQueue] === 'function' && typeof host.playerPrototype?.enqueueScript === 'function' &&
            Number.isSafeInteger(host.agentDamageScript), 'missing native player queue surfaces');
        let playerQueuing: Native = null;
        const queued = host.handlers[playerQueue];
        host.handlers[playerQueue] = function (state: Native) {
            const previous = playerQueuing;
            playerQueuing = state;
            try { return queued.call(this, state); } finally { playerQueuing = previous; }
        };
        const enqueuePlayer = host.playerPrototype.enqueueScript;
        host.playerPrototype.enqueueScript = function (script: Native, type: Native, delay = 0, args: Native[] = []) {
            const result = enqueuePlayer.call(this, script, type, delay, args);
            const b = live();
            if (b && this === b.player && script?.id === host.agentDamageScript) safe(b, () => {
                const self = playerQueuing?.self ?? null;
                const npc = self && host.actor(self, b.player) === 'npc' ? self : null;
                record(b, 'agent-hit', { damage: args[0], delay, nid: npc?.nid ?? null, type: npc?.type ?? null,
                    hp: host.agentHp(b.player) });
            });
            return result;
        };
    }
    host.handlers[host.opcodes.damage] = function (state: Native) {
        const b = live();
        let before: Native = null;
        if (b) safe(b, () => {
            const npc = state.intOperand === 0 ? state._activeNpc : state._activeNpc2;
            const tag = started.get(state);
            const hitter = tag?.binding === b ? tag.hitter : host.actor(state.self ?? null, b.player);
            if (!npc || !followed(b, npc, hitter)) return;
            const life = lifeOf(b, npc);
            reconcile(b, life);
            const stack = state.intStack, isp = state.isp;
            before = { b, life, hitter, hit: tag?.binding === b && tag.life === life ? tag.hit : null,
                amount: stack[isp - 1], hitType: stack[isp - 2] };
        });
        const result = damage.call(this, state);
        if (before && !before.b.failure) safe(before.b, () => {
            const { life } = before, { hp } = hpOf(life.npc);
            record(before.b, 'damage', { life: life.key, hitter: before.hitter, hit: before.hit, amount: before.amount,
                hitType: before.hitType, hpBefore: life.hp, hpAfter: hp,
                ...(hp === 0 && life.hp > 0 ? { heroes: host.heroes(life.npc, before.b.player) } : {}) });
            life.hp = hp;
        });
        return result;
    };
    const remove = host.world.removeNpc;
    host.world.removeNpc = function (npc: Native, ...rest: Native[]) {
        const b = live(), life = b?.lives.get(npc);
        // lastMovement (PathingEntity: the tick after the npc's last step) decides npc_death's npc_arrivedelay,
        // so it explains each defeat-to-removal interval (sim/check.py E9).
        if (b && life && !life.removed) safe(b, () => {
            reconcile(b, life);
            record(b, 'removed', { life: life.key, lastMovement: npc.lastMovement, x: npc.x, z: npc.z });
            life.removed = true;
        });
        return remove.call(this, npc, ...rest);
    };
    const add = host.world.addNpc;
    host.world.addNpc = function (npc: Native, ...rest: Native[]) {
        const result = add.call(this, npc, ...rest);
        const b = live(), life = b?.lives.get(npc);
        if (b && life) safe(b, () => {
            require(life.removed, 'followed npc respawned without removal');
            lifeOf(b, npc, life.key);
        });
        return result;
    };
    const cycle = host.world.cycle;
    host.world.cycle = function (...args: Native[]) {
        const result = cycle.apply(this, args);
        const b = live();
        if (b) safe(b, () => {
            for (const life of b.lives.values()) reconcile(b, life);
            const hp = host.agentHp(b.player);
            if (hp !== b.agentHp) { record(b, 'agent-hp', { hpBefore: b.agentHp, hpAfter: hp }); b.agentHp = hp; }
        });
        return result;
    };
    return {
        bind(player: Native, identity: Native, start: number, end: number) {
            require(binding === null && Number.isSafeInteger(start) && Number.isSafeInteger(end) && end > start,
                'combat capture allows one actor and a positive window');
            binding = { player, identity: structuredClone(identity), start, end, limit: eventLimit(maxEvents, end - start),
                sequence: 0, events: [], failure: null,
                lives: new Map(), seqs: new Map(), agentHp: host.agentHp(player), armedHp: host.agentHp(player) };
        },
        audit(player: Native) {
            require(binding && binding.player === player, 'combat actor not bound');
            const b = binding!;
            return structuredClone({ schema: 'native-combat-capture-v1', identity: b.identity, startTick: b.start,
                endTick: b.end, enemyTypes: [...enemies].sort((x, y) => x - y), agentHp: b.armedHp,
                events: b.events, failure: b.failure, open: host.open(player) });
        },
        release(player: Native) {
            require(binding && binding.player === player && !host.open(player), 'release a closed combat capture');
            const receipt = this.audit(player);
            binding = null;
            return receipt;
        }
    };
}
