import { expect, test } from 'bun:test';
import { createRecoveryPlacement, RECOVERY_PLACEMENT_PROFILE } from './recovery-placement';
function fixture() {
    const player = {}, apply = createRecoveryPlacement();
    const gate = { profile: RECOVERY_PLACEMENT_PROFILE, flags: true, owned: true, players: 1, production: false };
    const state: any = { x:3092,z:3242,level:0,tick:100,idle:true,aggressors:0,queuedWork:false,optionBusy:false,
        transport:'engine',hp:10,maxHp:10,armed:false,decision:48,episodeTicks:96,charged:48,inventory:[{slot:0,id:1351,count:1}],worn:[],bank:[{slot:0,id:995,count:25}],
        rawStats:Array(21).fill(0),levels:Array(21).fill(1),baseLevels:Array(21).fill(1),lifecycle:{worldBootId:'boot',accountId:'a',accountEpoch:1,incarnationId:'p',phase:'alive',deathSequence:0,respawnSequence:0,failure:null,eventsLost:false,events:[],latestSequence:0,deathTick:null,respawnTick:null} };
    let calls=0;
    const host = { admission:()=>gate,snapshot:()=>state,place:(tile:any)=>{ calls++;Object.assign(state,tile); } };
    return {state,gate,host,run:()=>apply(player,host),calls:()=>calls};
}
test('one placement preserves items XP identity and counters with fixed destination',()=>{
    const f=fixture(),r=f.run();expect(r.after).toMatchObject({x:3223,z:3219,decision:48,episodeTicks:96,charged:48});
    expect(()=>f.run()).toThrow('spent');expect(f.calls()).toBe(1);
});
test.each([{profile:'other'},{flags:false},{owned:false},{players:2},{production:true}])('admission rejects %j',change=>{
    const f=fixture();Object.assign(f.gate,change);expect(()=>f.run()).toThrow();expect(f.calls()).toBe(0);
});
test.each([{idle:false},{aggressors:1},{queuedWork:true},{optionBusy:true},{hp:9},{x:3100},{armed:true},
    {lifecycle:{phase:'alive',deathSequence:1,respawnSequence:1}}])('unsafe preparation rejects %j',change=>{
    const f=fixture();Object.assign(f.state,change);expect(()=>f.run()).toThrow();expect(f.calls()).toBe(0);
});
test.each(['rawStats','inventory','bank','decision','episodeTicks','charged','levels','lifecycle'])('post mutation %s fails permanently',key=>{
    const f=fixture(),place=f.host.place;f.host.place=tile=>{place(tile);f.state[key]='changed';};
    expect(()=>f.run()).toThrow();expect(()=>f.run()).toThrow('spent');expect(f.calls()).toBe(1);
});
test('throwing mutation consumes one shot before entry',()=>{
    const f=fixture();f.host.place=()=>{throw new Error('native failure');};expect(()=>f.run()).toThrow('native failure');expect(()=>f.run()).toThrow('spent');
});

test('counter getter shared guard rejects every missing flag or ownership/profile',async()=>{
    const {requireRecoveryPlacementProfile}=await import('./recovery-placement');
    const env={RL_RECOVERY_FIXTURE_PROFILE:RECOVERY_PLACEMENT_PROFILE,RL_TEST_HOOKS:'1',RL_CONTINUITY_EXPERIMENT:'1',RL_DROP_LINEAGE_EXPERIMENT:'1',RL_DEMO_BOOT_ID:'owned'};
    expect(()=>requireRecoveryPlacementProfile(env)).not.toThrow();
    for(const key of Object.keys(env))expect(()=>requireRecoveryPlacementProfile({...env,[key]:undefined})).toThrow();
});

test.each(['inventory','worn','bank','rawStats','levels','baseLevels','decision','charged','episodeTicks','tick','queuedWork','optionBusy','armed'])('missing both-side %s is rejected before mutation',key=>{
    const f=fixture();delete f.state[key];expect(()=>f.run()).toThrow();expect(f.calls()).toBe(0);
});

test('post-decode native confirmation rejects bank XP session and lost-event changes',async()=>{
    const {confirmPlacementSnapshot}=await import('./recovery-placement');
    const f=fixture(),r=f.run();
    for(const mutate of [(s:any)=>s.bank[0].count++, (s:any)=>s.rawStats[0]++,
        (s:any)=>s.lifecycle.incarnationId='other',(s:any)=>s.lifecycle.eventsLost=true]) {
        const changed=structuredClone(r.after);mutate(changed);expect(()=>confirmPlacementSnapshot(r.after,changed)).toThrow();
    }
});

test('rejection captures the exact detached evaluated state and named failures with no mutation',async()=>{
    const {RecoveryPlacementRejected}=await import('./recovery-placement');
    const f=fixture();f.state.aggressors=1;f.state.hp=9;
    let reads=0;f.host.snapshot=()=>{reads++;return f.state;};
    try {f.run();throw new Error('expected rejection');} catch(error) {
        expect(error).toBeInstanceOf(RecoveryPlacementRejected);
        const r=(error as InstanceType<typeof RecoveryPlacementRejected>).receipt;
        expect(r).toMatchObject({type:'recovery_place_rejected',stage:'untouched-bank',mutationAttempted:false,failedChecks:['noNpcTargets','hpTen']});
        f.state.hp=10;expect(r.before.hp).toBe(9);expect(reads).toBe(1);expect(f.calls()).toBe(0);
    }
});

test('schema rejection is diagnostic but authentication and snapshot identity errors remain opaque',async()=>{
    const {RecoveryPlacementRejected}=await import('./recovery-placement');
    const f=fixture();delete f.state.rawStats;
    try {f.run();} catch(error) {expect(error).toBeInstanceOf(RecoveryPlacementRejected);expect((error as any).receipt.stage).toBe('snapshot-schema');}
    f.gate.owned=false;
    try {f.run();} catch(error) {expect(error).not.toBeInstanceOf(RecoveryPlacementRejected);expect((error as any).receipt).toBeUndefined();}
    f.gate.owned=true;f.host.snapshot=()=>{throw new Error('identity mismatch');};
    try {f.run();} catch(error) {expect(error).not.toBeInstanceOf(RecoveryPlacementRejected);expect((error as Error).message).toBe('identity mismatch');}
    expect(f.calls()).toBe(0);
});

test('private rejection bypasses every freshness callback; success still certifies once',async()=>{
    const {completePlacementReply}=await import('./recovery-placement');let calls=0;
    const r={type:'recovery_place_rejected',before:{hp:9}};
    const output=await completePlacementReply(r,async()=>{calls++;throw new Error('must not wait');});
    expect(output).toEqual(r);expect(calls).toBe(0);r.before.hp=10;expect(output.before.hp).toBe(9);
    expect(await completePlacementReply({type:'recovery_place'},async()=>{calls++;return 'certified';})).toBe('certified');expect(calls).toBe(1);
});

test('target descriptors are bounded detached references with exact total/overflow',async()=>{
    const {placementTargets}=await import('./recovery-placement');const player={};
    const npcs=Array.from({length:20},(_,nid)=>({nid,type:7,x:3090+nid,z:3242,level:0,target:player}));
    const r=placementTargets([...npcs,{nid:100,type:8,x:0,z:0,level:0,target:{}}],player);
    expect(r.count).toBe(20);expect(r.overflow).toBe(true);expect(r.targets).toHaveLength(16);
    npcs[0].x=0;expect(r.targets[0]).toEqual({nid:0,type:7,x:3090,z:3242,level:0,targetIsActor:true});
    expect(JSON.stringify(r)).not.toContain('"target":');expect(placementTargets([],player)).toEqual({count:0,overflow:false,targets:[]});
});
