/** Private evaluator setup only. This is not an action or learned recovery. */
export const RECOVERY_PLACEMENT_PROFILE = 'near-respawn-isolated-v1';
// A measured member of death.rs2's map_findsquare(0_50_50_21_18, 0, 2).
export const RECOVERY_PLACEMENT_TILE = { x: 3223, z: 3219, level: 0 };
export class RecoveryPlacementRejected extends Error {
    readonly receipt: any;
    constructor(stage: string, before: any, checks: Record<string, boolean>, message: string) {
        super(message);
        this.receipt = structuredClone({ type:'recovery_place_rejected',profile:RECOVERY_PLACEMENT_PROFILE,
            stage,mutationAttempted:false,checks,failedChecks:Object.keys(checks).filter(k=>!checks[k]),before,message });
    }
}

/** A rejected private placement must never enter the browser freshness path. */
export async function completePlacementReply(receipt: any, certify: () => Promise<any>) {
    return receipt.type === 'recovery_place_rejected' ? structuredClone(receipt) : await certify();
}

/** Detached descriptors only: no native references or unbounded diagnostic output. */
export function placementTargets(npcs: Iterable<{nid:number;type:number;x:number;z:number;level:number;target:unknown}>, player: object) {
    const targets: {nid:number;type:number;x:number;z:number;level:number;targetIsActor:true}[] = [];
    let count=0;
    for (const npc of npcs) if (npc.target === player) {
        count++;
        if (targets.length < 16) targets.push({nid:npc.nid,type:npc.type,x:npc.x,z:npc.z,level:npc.level,targetIsActor:true});
    }
    return {count,overflow:count>targets.length,targets};
}
export function validatePlacementSnapshot(s: any): void {
    const integer = (v: unknown) => Number.isSafeInteger(v) && (v as number) >= 0;
    const fail = () => { throw new Error('incomplete or unsafe placement snapshot'); };
    if (!s || !['engine','browser'].includes(s.transport) ||
        !['tick','x','z','level','hp','maxHp','aggressors'].every(k => integer(s[k])) ||
        s.idle !== true || ['queuedWork','optionBusy','armed'].some(k => s[k] !== false)) fail();
    for (const [key, capacity] of [['inventory',28],['worn',14],['bank',240]] as const) {
        const rows = s[key];
        if (!Array.isArray(rows) || rows.some(r => !r || !integer(r.slot) || r.slot >= capacity ||
            !integer(r.id) || !integer(r.count) || r.count === 0) || new Set(rows.map(r=>r.slot)).size !== rows.length) fail();
    }
    for (const key of ['rawStats','levels','baseLevels'])
        if (!Array.isArray(s[key]) || s[key].length !== 21 || !s[key].every(integer)) fail();
    const a=s.lifecycle;
    if (!a || ['worldBootId','accountId','incarnationId'].some(k=>typeof a[k] !== 'string' || !a[k]) ||
        !integer(a.accountEpoch) || a.accountEpoch < 1 || a.phase !== 'alive' || a.deathSequence !== 0 ||
        a.respawnSequence !== 0 || a.failure !== null || a.eventsLost !== false || a.latestSequence !== 0 ||
        !Array.isArray(a.events) || a.events.length || a.deathTick !== null || a.respawnTick !== null) fail();
    if (s.transport === 'browser' && (typeof a.serverSessionId !== 'string' || !a.serverSessionId)) fail();
    if (s.transport === 'engine' && !['decision','charged','episodeTicks'].every(k=>integer(s[k]))) fail();
}

export function confirmPlacementSnapshot(expected: any, actual: any): void {
    validatePlacementSnapshot(expected); validatePlacementSnapshot(actual);
    const {tick:oldTick,...before}=expected, {tick:newTick,...after}=actual;
    if (newTick < oldTick || JSON.stringify(before) !== JSON.stringify(after))
        throw new Error('placement confirmation changed native state');
}
export function requireRecoveryPlacementProfile(env: Record<string, string | undefined>) {
    if (env.RL_RECOVERY_FIXTURE_PROFILE !== RECOVERY_PLACEMENT_PROFILE || env.RL_TEST_HOOKS !== '1' ||
        env.RL_CONTINUITY_EXPERIMENT !== '1' || env.RL_DROP_LINEAGE_EXPERIMENT !== '1' || !env.RL_DEMO_BOOT_ID)
        throw new Error('recovery placement profile unavailable');
}
export function createRecoveryPlacement() {
    const spent = new WeakSet<object>();
    return (player: object, host: {
        admission(): { profile?: string; flags: boolean; owned: boolean; players: number; production: boolean };
        snapshot(): any; place(tile: typeof RECOVERY_PLACEMENT_TILE): void;
    }) => {
        const gate = host.admission();
        if (gate.profile !== RECOVERY_PLACEMENT_PROFILE || !gate.flags || !gate.owned ||
            gate.players !== 1 || gate.production) throw new Error('isolated recovery placement disabled');
        if (spent.has(player)) throw new Error('recovery placement already spent');
        const before = structuredClone(host.snapshot());
        const checks = { schemaValid:true,idle:before?.idle === true,noNpcTargets:before?.aggressors === 0,
            noQueuedWork:before?.queuedWork === false,noOption:before?.optionBusy === false,
            hpTen:before?.hp === 10,maxHpTen:before?.maxHp === 10,bankX:before?.x === 3092,
            bankZ:before?.z === 3242,levelZero:before?.level === 0,alive:before?.lifecycle?.phase === 'alive',
            noDeaths:before?.lifecycle?.deathSequence === 0,noRespawns:before?.lifecycle?.respawnSequence === 0,
            unarmed:before?.armed === false,noLifecycleFailure:before?.lifecycle?.failure === null };
        try { validatePlacementSnapshot(before); }
        catch (error) {
            checks.schemaValid=false;
            throw new RecoveryPlacementRejected('snapshot-schema',before,checks,(error as Error).message);
        }
        if (Object.values(checks).some(ok=>!ok))
            throw new RecoveryPlacementRejected('untouched-bank',before,checks,'recovery placement requires untouched idle bank preparation');
        spent.add(player); // Even a throwing mutation or failed postcheck is not retryable.
        host.place(RECOVERY_PLACEMENT_TILE);
        const after = structuredClone(host.snapshot());
        validatePlacementSnapshot(after);
        const conserved = (s: any) => {
            const { x, z, level, tick, ...rest } = s;
            return rest;
        };
        if (JSON.stringify(conserved(before)) !== JSON.stringify(conserved(after)) ||
            !Number.isSafeInteger(after.tick) || after.tick < before.tick ||
            Object.entries(RECOVERY_PLACEMENT_TILE).some(([k, v]) => after[k] !== v))
            throw new Error('recovery placement conservation failed');
        return { type: 'recovery_place', profile: RECOVERY_PLACEMENT_PROFILE, before, after,
            scope: 'Privileged isolated evaluator preparation; not policy navigation or survival.' };
    };
}
