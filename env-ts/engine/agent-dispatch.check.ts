// env-ts/engine/agent-dispatch.check.ts: act-v3's click and ACT dispatch in the native engine agent, and its
// inventory options under held-v2.
//
// EngineAgent's own effector code (click, sendRoute, interactLoc, interactNpc, pickUp, opHeld, dropSlot, useOnHeld;
// engine/agent.ts) run over a stub of the game engine: every vendor module agent.ts reaches is mocked here, so the
// check needs no vendor tree and no world. The route is the real core/clicks.ts tryMove over the stub's collision;
// the stub records what the agent asks of the server's Player. Run by agent-dispatch.test.ts in its own `bun test`
// process, since a module mock lasts for the whole process and the other engine tests need the real modules.
import { expect, mock, test } from 'bun:test';
import { readFileSync } from 'node:fs';

const ENGINE = '../../vendor/rs-sdk/server/engine/src';
const stub = (path: string, exports: Record<string, unknown>) => mock.module(path, () => exports);

const world = { currentTick: 1000, objs: new Map<string, unknown>(),
    getObj: (x: number, z: number) => world.objs.get(`${x},${z}`) ?? null };
const flags = new Map<string, number>();   // the server's collision flags by tile; open where unset
const inView = new Set<number>();          // the npcs in the player's npc list (rsbuf)
const NONE = 'AllowRepath.NONE';
const pack = (level: number, x: number, z: number) => ((level & 0x3) << 28) | ((x & 0x3fff) << 14) | (z & 0x3fff);
const locTypes: Record<number, { width: number; length: number; forceapproach: number }> = {
    1276: { width: 1, length: 1, forceapproach: 0 } };

stub(`${ENGINE}/engine/World.js`, { default: world });
stub(`${ENGINE}/engine/routefinder/index.js`, { default: { collisionFlags: { get: (x: number, z: number) =>
    flags.get(`${x},${z}`) ?? 0 } } });
stub(`${ENGINE}/network/rsbuf/index.js`, { hasNpc: (_slot: number, nid: number) => inView.has(nid), npcInfo: () => {} });
stub(`${ENGINE}/engine/CoordGrid.js`, { CoordGrid: { packCoord: pack } });
stub(`${ENGINE}/engine/entity/AllowRepath.js`, { AllowRepath: { NONE } });
stub(`${ENGINE}/engine/entity/Interaction.js`, { Interaction: { ENGINE: 'Interaction.ENGINE' } });
stub(`${ENGINE}/engine/script/ServerTriggerType.js`, { default: { APLOC1: 100, APLOCU: 110, APNPC1: 200, APNPCU: 210,
    APOBJ1: 300, OPHELD1: 400, OPHELD5: 404, OPHELDU: 410 } });
stub(`${ENGINE}/cache/config/LocType.js`, { default: { get: (id: number) => locTypes[id] } });
// Held items: shrimps (Eat, op 1) and logs (no iop), each with a script for every held trigger but OPHELDU's.
const objTypes: Record<number, { id: number; category: number; iop: (string | null)[]; debugname: string; name: string }> = {
    315: { id: 315, category: -1, iop: ['Eat', null, null, null, 'Drop'], debugname: 'shrimp', name: 'Shrimps' },
    1511: { id: 1511, category: -1, iop: [null, null, null, null, 'Drop'], debugname: 'logs', name: 'Logs' },
    1277: { id: 1277, category: -1, iop: [null, 'Wield', null, null, 'Drop'], debugname: 'bronze_sword', name: 'Bronze sword' },
    // nine eatable kinds f0..f8, so that item-v4's page and pack options reach entries 8 and over
    ...Object.fromEntries(Array.from({ length: 9 }, (_, k) => [3000 + k, { id: 3000 + k, category: -1,
        iop: ['Eat', null, null, null, 'Drop'], debugname: `f${k}`, name: `f${k}` }])) };
stub(`${ENGINE}/cache/config/ObjType.js`, { default: { get: (id: number) => objTypes[id] ?? { op: [null, null, 'Take'] },
    getId: (name: string) => Object.values(objTypes).find(o => o.debugname === name)?.id ?? -1 } });
stub(`${ENGINE}/cache/config/InvType.js`, { default: { INV: 93 } });
stub(`${ENGINE}/cache/config/CategoryType.js`, { default: { get: (id: number) => ({ id }) } });
stub(`${ENGINE}/engine/script/ScriptProvider.js`, { default: {
    getByTrigger: (trigger: number, id: number) => trigger >= 400 && trigger < 410 ? { name: `${trigger}:${id}` } : null,
    getByTriggerSpecific: () => null, getByName: (name: string) => ({ name }) } });
stub(`${ENGINE}/engine/script/ScriptRunner.js`, { default: { init: (script: { name: string }) => ({ script: script.name }) } });
stub(`${ENGINE}/util/Environment.js`, { default: { node: { debug: true } } });   // the engine's default world config
for (const path of ['cache/config/Component.js', 'cache/config/DbRowType.js',
    'cache/config/DbTableType.js', 'cache/config/NpcType.js', 'cache/config/VarPlayerType.js',
    'engine/entity/Loc.js', 'engine/entity/Npc.js', 'engine/entity/Player.js',
    'engine/script/ScriptState.js', 'io/Packet.js',
    'network/game/server/model/IfSetText.js', 'network/game/server/model/MessageGame.js'])
    stub(`${ENGINE}/${path}`, { default: {} });
stub(`${ENGINE}/engine/entity/PlayerStat.js`, { PlayerStat: { HITPOINTS: 3 } });
stub(`${ENGINE}/engine/entity/PlayerLoading.js`, { PlayerLoading: {} });
stub(`${ENGINE}/engine/GameMap.js`, { findPath: () => { throw new Error('act-v3 never asks the server path finder'); } });
stub('../../vendor/rs-sdk/sdk/pathfinding', { findLongPath: () => [], isTileWalkable: () => true,
    initPathfinding: () => {} });
stub('../../vendor/rs-sdk/server/vendor/rsmod-pathfinder', { CollisionFlag: { WALL_NORTH: 0x2, WALL_EAST: 0x8,
    WALL_SOUTH: 0x20, WALL_WEST: 0x80 }, CollisionType: { NORMAL: 0 }, canTravel: () => true, isFlagged: () => false });

const { EngineAgent } = await import('./agent');
const { ItemPages, itemPage, kindPage } = await import('../core/items');
const { selectHeldWide } = await import('../core/wide');
const { pageRow } = await import('../core/actions');
const { selectHeld } = await import('../core/affordances');
const { A_DEPOSIT0, A_DROP0, A_HELD0, A_USE0 } = await import('../core/actions');
const { CLICK_ROWS } = await import('../core/clicks');

type Call = [string, ...unknown[]];
function fixture(x = 3200, z = 3200, pack?: Record<number, { id: number; count: number }>) {
    flags.clear(); inView.clear(); world.objs.clear();
    const calls: Call[] = [];
    const player = { x, z, level: 0, originX: x, originZ: z, slot: 1, delayed: false, stats: [0],
        waypoints: [] as number[], allowRepath: 'BEFOREDEST', target: null as unknown, trigger: -1, opcalled: false,
        lastItem: -1, lastSlot: -1, lastUseItem: -1, lastUseSlot: -1,
        clearPendingAction() { calls.push(['clearPendingAction']); this.target = null; },
        executeScript(state: { script: string }, protect: boolean) { calls.push(['executeScript', state.script, protect]); },
        messageGame(text: string) { calls.push(['messageGame', text]); },
        queueWaypoints(w: number[]) { calls.push(['queueWaypoints', [...w]]); this.waypoints = [...w]; },
        setAllowRepath(v: string) { calls.push(['setAllowRepath', v]); this.allowRepath = v; },
        processWalktrigger() { calls.push(['processWalktrigger']); },
        hasWaypoints() { return this.waypoints.length > 0; },
        setInteraction(kind: string, target: unknown, trigger: number) {
            calls.push(['setInteraction', kind, trigger]); this.target = target; this.trigger = trigger;
            this.targetOp = trigger; return true; }, targetOp: -1,
        getInventory: (inv: number) => inv === 93 ? { capacity: 28, get: (slot: number) => held[slot] ?? null } : null };
    const held: Record<number, { id: number; count: number }> = pack ?? { 0: { id: 315, count: 1 }, 1: { id: 1511, count: 1 } };
    const agent: any = Object.create(EngineAgent.prototype);
    Object.assign(agent, { player, cfg: {}, option: null, optionEnd: null, decisionRoute: null, latestRoute: null,
        decision: 7, itemPages: new ItemPages(), using: null,
        lastObservation: { clicks: Array.from({ length: 24 }, (_, k) => ({ dx: OFFSETS[k][0], dz: OFFSETS[k][1],
            inside: true, open: true })) } });
    return { agent, player, calls, names: () => calls.map(c => c[0]) };
}
const OFFSETS: Array<[number, number]> = JSON.parse(readFileSync(new URL('../core/clicks.fixture.json', import.meta.url),
    'utf8')).clicks.offsets;
/** Every tile around (x, z) blocked: no step leaves it. */
const enclose = (x: number, z: number) => {
    for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) if (dx || dz) flags.set(`${x + dx},${z + dz}`, 0x100);
};

test('a click while the player is delayed sends nothing, keeps the earlier walk and ends failed', () => {
    const { agent, player, calls } = fixture();
    player.waypoints = [pack(0, 3210, 3200)];
    player.delayed = true;
    agent.click(CLICK_ROWS[0]);
    expect(calls).toEqual([]);
    expect(player.waypoints).toEqual([pack(0, 3210, 3200)]);
    expect(agent.decisionRoute).toMatchObject({ kind: 'click', refused: 'delayed', via: 'none', waypoints: [],
        offset: { dx: 0, dz: 4 }, target: { x: 3200, z: 3204 }, source: { x: 3200, z: 3200, level: 0 } });
    expect(agent.optionEnd).toEqual({ action: CLICK_ROWS[0], reason: 'failed', refused: true, reported: false });
    expect(agent.option).toBeNull();
});

test('a click with no route sends nothing and keeps the earlier walk', () => {
    const { agent, player, calls } = fixture();
    enclose(3200, 3200);
    player.waypoints = [pack(0, 3201, 3201)];
    agent.click(CLICK_ROWS[16]);
    expect(calls).toEqual([]);
    expect(player.waypoints).toEqual([pack(0, 3201, 3201)]);
    expect(agent.decisionRoute).toMatchObject({ refused: 'no route', via: 'none', end: null, waypoints: [] });
    expect(agent.optionEnd).toMatchObject({ reason: 'failed', refused: true });
});

test('a routed click clears the pending action, queues its turn points, then runs the walk trigger', () => {
    const { agent, player, calls, names } = fixture();
    agent.click(CLICK_ROWS[2]);   // 4 east, open ground: one turn point, the target
    expect(names()).toEqual(['clearPendingAction', 'queueWaypoints', 'processWalktrigger']);
    expect(calls[1]).toEqual(['queueWaypoints', [pack(0, 3204, 3200)]]);
    expect(agent.decisionRoute).toMatchObject({ via: 'target', refused: null, end: { x: 3204, z: 3200 },
        waypoints: [{ x: 3204, z: 3200 }] });
    expect(agent.option.label).toBe('nav_click_2');
    expect(agent.option.done()).toBe(false);
    player.waypoints = [];
    expect(agent.option.done()).toBe(true);   // the option ends when the waypoint queue empties
});

test('a click onto a blocked tile walks to the nearest tile of its 3 x 3', () => {
    const { agent, calls } = fixture();
    // 9 north (slot 8) is a loc. Its 3 x 3's nearest tiles, 3199..3201 at z 3208, are all 8 steps away: the first in
    // x-then-z order is taken, reached by 7 steps north and one north-west
    flags.set('3200,3209', 0x100);
    agent.click(CLICK_ROWS[8]);
    expect(calls[1]).toEqual(['queueWaypoints', [pack(0, 3200, 3207), pack(0, 3199, 3208)]]);
    expect(agent.decisionRoute).toMatchObject({ via: 'fallback', end: { x: 3199, z: 3208 } });
});

test('at most the 25 turn points nearest the player are queued, as the Python port sends them', () => {
    const cases = JSON.parse(readFileSync(new URL('../core/clicks.fixture.json', import.meta.url), 'utf8'));
    const q = cases.queries.find((c: any) => c.scene === 'maze-long' && !c.nearest && c.arrival.every((v: number) => !v));
    const { agent, calls } = fixture(q.source[0], q.source[1]);
    Object.assign(agent.player, { originX: cases.base[0] + 48, originZ: cases.base[1] + 48 });   // the scene at base
    for (const [x, z, f] of cases.scenes['maze-long']) flags.set(`${cases.base[0] + x},${cases.base[1] + z}`, f);
    world.objs.set(`${q.dest[0]},${q.dest[1]}`, { id: 1511 });
    agent.pickUp(1, { x: q.dest[0], z: q.dest[1], id: 1511, name: 'Logs' });
    expect(q.route.waypoints.length).toBe(25);
    expect(calls[1]).toEqual(['queueWaypoints', q.route.waypoints.map(([x, z]: number[]) => pack(0, x, z))]);
    expect(agent.decisionRoute).toMatchObject({ via: 'target', waypoints: q.route.waypoints.map(([x, z]: number[]) =>
        ({ x, z })) });
});

test('an npc op beside the player sends only its own tile: the walk cleared, no repath, then the op', () => {
    const { agent, player, calls, names } = fixture();
    player.waypoints = [pack(0, 3210, 3200)];
    const npc = { nid: 7, type: 41, x: 3201, z: 3200, delayed: false };
    inView.add(7);
    agent.interactNpc(1, { npc, rawOps: [null, 'Attack'], name: 'Chicken' }, /^attack$/i, 'Attack', 'act_0');
    expect(names()).toEqual(['clearPendingAction', 'queueWaypoints', 'setAllowRepath', 'processWalktrigger',
        'clearPendingAction', 'setInteraction']);
    expect(calls[1]).toEqual(['queueWaypoints', []]);
    expect(player.allowRepath).toBe(NONE);
    expect(calls[5]).toEqual(['setInteraction', 'Interaction.ENGINE', 201]);
    expect(agent.decisionRoute).toMatchObject({ kind: 'act', via: 'reach', refused: null, waypoints: [{ x: 3200, z: 3200 }],
        thing: { kind: 'npc', id: 41, name: 'Chicken', option: 'Attack' } });
});

test('an npc op is refused before any route while the npc is out of the player\'s list, delayed, or the player is', () => {
    for (const [inList, npcDelayed, playerDelayed, refused] of [[false, false, false, 'npc not in view'],
        [true, true, false, 'npc delayed'], [true, false, true, 'delayed']] as Array<[boolean, boolean, boolean, string]>) {
        const { agent, player, calls } = fixture();
        player.waypoints = [pack(0, 3210, 3200)];
        player.delayed = playerDelayed;
        if (inList) inView.add(7);
        const npc = { nid: 7, type: 41, x: 3205, z: 3200, delayed: npcDelayed };
        agent.interactNpc(1, { npc, rawOps: [null, 'Attack'], name: 'Chicken' }, /^attack$/i, 'Attack', 'act_0');
        expect(calls).toEqual([]);
        expect(player.waypoints).toEqual([pack(0, 3210, 3200)]);
        expect(player.target).toBeNull();
        expect(agent.decisionRoute).toMatchObject({ refused, via: 'none', waypoints: [] });
        expect(agent.option).toBeNull();
    }
});

test('a loc op with no route sends only the op: the earlier walk continues', () => {
    const { agent, player, names } = fixture();
    player.waypoints = [pack(0, 3200, 3210)];
    for (const [dx, dz] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) flags.set(`${3205 + dx},${3200 + dz}`, 0x100);
    const loc = { type: 1276, x: 3205, z: 3200, level: 0, shape: 10, angle: 0 };
    agent.interactLoc(1, { loc, rawOps: ['Chop down'], name: 'Tree' }, /^chop down$/i, 'Chop down', 'act_0');
    expect(names()).toEqual(['clearPendingAction', 'setInteraction']);
    expect(player.waypoints).toEqual([pack(0, 3200, 3210)]);
    expect(player.target).toBe(loc);
    expect(agent.decisionRoute).toMatchObject({ refused: 'no route', via: 'none', waypoints: [] });
    expect(agent.option.label).toBe('act_0');
});

test('a ground item on a tile with no route is taken from beside it', () => {
    const { agent, calls } = fixture();
    flags.set('3203,3200', 0x100);   // the item lies on a blocked tile 3 east
    world.objs.set('3203,3200', { id: 1511 });
    agent.pickUp(1, { x: 3203, z: 3200, id: 1511, name: 'Logs' });
    expect(calls[1]).toEqual(['queueWaypoints', [pack(0, 3202, 3200)]]);
    expect(agent.decisionRoute).toMatchObject({ via: 'adjacent', end: { x: 3202, z: 3200 }, refused: null });
    expect(calls.at(-1)).toEqual(['setInteraction', 'Interaction.ENGINE', 302]);
});

test('an Eat or a Drop clears the pending action before its script, which ends a running chop (held-v2)', () => {
    for (const run of [(agent: any) => agent.opHeld(0, /^eat$/i), (agent: any) => agent.dropSlot(1)]) {
        const { agent, player, calls } = fixture();
        const tree = { type: 1276, x: 3201, z: 3200 };
        player.target = tree;   // a chop running: woodcut.rs2 sets this interaction again every cycle
        const note = run(agent);
        expect(calls.map(c => c[0])).toEqual(['clearPendingAction', 'executeScript']);
        expect(calls[1][2]).toBe(true);   // run protected, as OpHeldHandler's executeScript
        expect(player.target).toBeNull();
        expect(note).toMatch(/^(opheld Eat slot 0|drop slot 1)$/);
        expect([player.lastItem, player.lastSlot]).toEqual(note.startsWith('opheld') ? [315, 0] : [1511, 1]);
    }
});

test('while the player is delayed an inventory option or a use on a held item does nothing', () => {
    const { agent, player, calls } = fixture();
    const tree = { type: 1276, x: 3201, z: 3200 };
    Object.assign(player, { target: tree, delayed: true });
    expect(agent.opHeld(0, /^eat$/i)).toBe('opheld: refused while delayed');
    expect(agent.dropSlot(1)).toBe('drop: refused while delayed');
    expect(agent.useOnHeld(1, { slot: 0, id: 315 })).toBe('use: refused while delayed');
    expect(calls).toEqual([]);
    expect(player.target).toBe(tree);
    expect([player.lastItem, player.lastUseItem]).toEqual([-1, -1]);
});

test('a use on a held item with no script clears the pending action, then sends the handler\'s two lines', () => {
    const { agent, player, calls } = fixture();
    player.target = { type: 1276, x: 3201, z: 3200 };
    expect(agent.useOnHeld(0, { slot: 1, id: 1511 })).toBe('use: nothing interesting happens');
    expect(calls).toEqual([['clearPendingAction'], ['messageGame', 'No trigger for [opheldu,shrimp]'],
        ['messageGame', 'Nothing interesting happens.']]);
    expect(player.target).toBeNull();
    expect([player.lastItem, player.lastUseItem]).toEqual([1511, 315]);   // OpHeldUHandler's second lookup swaps them
});

test('a use on a loc sets its interaction with the use trigger and the picked item', () => {
    const { agent, player } = fixture();
    const loc = { type: 1276, x: 3201, z: 3200, level: 0, shape: 10, angle: 0 };
    agent.interactLoc(1, { loc, rawOps: ['Chop down'], name: 'Tree' }, null, 'Use', 'act_0', { slot: 1, id: 1511 });
    expect([player.target, player.targetOp, player.lastUseItem]).toEqual([loc, 110, 1511]);   // APLOCU
});

test('under item-v3 each row acts on its kind\'s first slot: drop one, pick, use on another kind, eat, deposit the kind', () => {
    // the pack: shrimps in slots 0 and 5, logs in slots 1 to 3, a sword in slot 9
    const pack = { 0: { id: 315, count: 1 }, 1: { id: 1511, count: 1 }, 2: { id: 1511, count: 1 }, 3: { id: 1511, count: 1 },
        5: { id: 315, count: 1 }, 9: { id: 1277, count: 1 } };
    const { agent, player, calls } = fixture(3200, 3200, pack);
    agent.bankIsOpen = () => true;
    const look = () => {  // the observation the next row is chosen from (projectItems' page and selectHeld)
        const items = agent.inventoryItems(93).map((i: any) => ({ ...i, options: objTypes[i.id].iop.filter(Boolean) }));
        const invSel = (agent.itemPages.contract === 'item-v3' ? kindPage : itemPage)(items, 0, 28);
        agent.lastObservation = { ...agent.lastObservation, mask: new Array(96).fill(1), invSel,
            heldSel: selectHeld(invSel, agent.using) };
        return agent.lastObservation;
    };
    agent.itemPages.reset('item-v3');
    expect(look().invSel.map((e: any) => [e.slot, e.id, e.count, e.total]).slice(0, 4)).toEqual(
        [[0, 315, 1, 2], [1, 1511, 1, 3], [9, 1277, 1, 1], [-1, -1, 0, 0]]);
    expect(agent.dispatch(A_DROP0 + 1, null)).toBe('drop slot 1');   // the one log of the kind's first slot
    expect(calls.at(-1)).toEqual(['executeScript', '404:1511', true]);
    expect(agent.dispatch(A_USE0 + 1, null)).toBe('use: picked Logs');   // a pick takes that log
    expect(agent.using).toEqual({ slot: 1, id: 1511, name: 'Logs' });
    const targets = look().heldSel;   // the page's other kinds, each once: the logs are none
    expect(targets.map((h: any) => [h.slot, h.option])).toEqual([[0, 'Use'], [9, 'Use']]);
    expect(agent.dispatch(A_HELD0, null)).toBe('use: nothing interesting happens');   // logs on the shrimps
    expect([player.lastItem, player.lastUseItem]).toEqual([1511, 315]);   // the logs on the shrimps (swapped)
    expect(look().heldSel.map((h: any) => [h.slot, h.option])).toEqual([[0, 'Eat'], [9, 'Wield']]);
    expect(agent.dispatch(A_HELD0, null)).toBe('opheld Eat slot 0');   // an Eat plays on the kind's first slot
    expect(agent.wieldSlot(agent.lastObservation, 'bronze_sword')).toBe(1);   // the setup's Wield finds its kind
    expect(agent.dispatch(A_DEPOSIT0 + 1, null)).toBe('bank_deposit: slot 1 id 1511 requested 3');   // the whole kind
    agent.itemPages.reset();   // item-v2: the physical page, a deposit of the slot's own count
    expect(look().invSel.map((e: any) => e.slot).slice(0, 4)).toEqual([0, 1, 2, 3]);
    expect(agent.dispatch(A_DEPOSIT0 + 1, null)).toBe('bank_deposit: slot 1 id 1511 requested 1');
    expect(agent.dispatch(A_DROP0 + 2, null)).toBe('drop slot 2');
});

test('under item-v4 the rows of entries 8..15 act on their entry: drop, pick, a pack option, a use and a deposit', () => {
    // f0..f8 in slots 0..8 (each offers Eat), logs in slots 9 and 11, the sword in slot 10: 11 kinds on one page of 16
    const pack: Record<number, { id: number; count: number }> = Object.fromEntries(Array.from({ length: 9 }, (_, k) =>
        [k, { id: 3000 + k, count: 1 }]));
    Object.assign(pack, { 9: { id: 1511, count: 1 }, 10: { id: 1277, count: 1 }, 11: { id: 1511, count: 1 } });
    const { agent, player, calls } = fixture(3200, 3200, pack);
    agent.bankIsOpen = () => true;
    agent.itemPages.reset('item-v4');
    const look = () => {
        const items = agent.inventoryItems(93).map((i: any) => ({ ...i, options: objTypes[i.id].iop.filter(Boolean) }));
        const invSel = kindPage(items, 0, 28, 16);
        agent.lastObservation = { ...agent.lastObservation, mask: new Array(128).fill(1), invSel,
            heldSel: selectHeldWide(invSel, agent.using) };
        return agent.lastObservation;
    };
    const page = look();
    expect(page.invSel.map((e: any) => [e.slot, e.id, e.total]).slice(8, 12)).toEqual(
        [[8, 3008, 1], [9, 1511, 2], [10, 1277, 1], [-1, -1, 0]]);
    expect(page.heldSel.map((h: any) => [h.slot, h.option])).toEqual([...Array.from({ length: 9 }, (_, k) => [k, 'Eat']),
        [10, 'Wield']]);
    expect(agent.dispatch(pageRow(A_HELD0, 8), null)).toBe('opheld Eat slot 8');   // row 96: the ninth pack option
    expect(agent.dispatch(pageRow(A_HELD0, 9), null)).toBe('opheld Wield slot 10');  // row 97
    expect(agent.dispatch(pageRow(A_DROP0, 9), null)).toBe('drop slot 9');   // row 105: the logs' first slot
    expect(calls.at(-1)).toEqual(['executeScript', '404:1511', true]);
    expect(agent.dispatch(pageRow(A_USE0, 9), null)).toBe('use: picked Logs');   // row 121
    expect(agent.using).toEqual({ slot: 9, id: 1511, name: 'Logs' });
    const targets = look().heldSel;   // every other kind's Use: f0..f8 and the sword
    expect(targets.map((h: any) => h.slot)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]);
    expect(agent.dispatch(pageRow(A_HELD0, 8), null)).toBe('use: nothing interesting happens');   // row 96: logs on f8
    expect([player.lastItem, player.lastUseItem]).toEqual([1511, 3008]);   // the logs on f8 (swapped)
    look();
    expect(agent.dispatch(pageRow(A_DEPOSIT0, 9), null)).toBe('bank_deposit: slot 9 id 1511 requested 2');  // row 113
    expect(agent.dispatch(pageRow(A_DROP0, 12), null)).toBe('drop: stale selection');   // row 108: an empty entry
});
