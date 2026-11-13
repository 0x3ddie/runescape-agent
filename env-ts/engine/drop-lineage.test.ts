import { expect, test } from 'bun:test';
import { installDropLineage, type DropLineageHost } from './drop-lineage';
import { monitorLifecycle } from './lifecycle';

function fixture(options: { maxEvents?: number } = {}) {
    const control = { tick: 10, players: 0, enabled: true, production: false, captureThrows: false,
        valid: true, addCalls: 0, takeCalls: 0, dropCalls: 0, takeAdds: true, insertAdd: true,
        failNativeTake: null as Error | null };
    class NativePlayer { level=0; x=3222; z=3218; hash64=123n; inv=[{slot:0,id:1351,count:1},{slot:1,id:1277,count:1}]; }
    const player = new NativePlayer();
    const ground = new Set<NativeObj>();
    class NativeObj {
        type=1351; count=1; level=0; x=3222; z=3218; lifecycle=1; lifecycleTick=-1; lastLifecycleTick=-1;
        receiver64=-1n; reveal=-1; isActive=false;
        turn() { this.lifecycleTick--; if(this.lifecycleTick===0) world.removeObj(this,0); return 'turned'; }
    }
    const world = {
        addObj(obj: NativeObj, receiver: bigint, duration: number) {
            control.addCalls++; if(!control.insertAdd)return 'added';
            ground.add(obj); obj.isActive=true; obj.receiver64=receiver;
            obj.lifecycleTick=duration*2; obj.lastLifecycleTick=control.tick; return 'added';
        },
        removeObj(obj: NativeObj, _duration: number) { ground.delete(obj);obj.isActive=false;obj.lifecycleTick=-1;return 'removed'; },
        changeObj(obj: NativeObj, count: number) {obj.count=count;return 'changed';},
        revealObj(obj: NativeObj) {obj.receiver64=-1n;obj.reveal=-1;return 'revealed';}
    };
    const proc = {};
    const life = { phase:'alive' as 'alive'|'dying',deathSequence:0,respawnSequence:0,deathTick:null as number|null,
        respawnTick:null as number|null,completionSource:null as 'player_death_queue'|null,failure:null };
    let pending: any = null;
    const monitor = { snapshot:()=>({...life}),subscribe:()=>()=>{},isPendingDeathScript:(s:object)=>pending===s && life.phase==='dying' };
    const handlers = {
        1(state:any) {
            control.dropCalls++;
            const axe=state._activePlayer.inv.find((i:any)=>i.id===1351);
            if(axe) {
                state._activePlayer.inv=state._activePlayer.inv.filter((i:any)=>i!==axe);
                const obj=new NativeObj();world.addObj(obj,-1n,4);state.dropped=obj;
            }
            return 'dropped';
        },
        2(state:any) {
            control.takeCalls++;
            if(control.failNativeTake)throw control.failNativeTake;
            if(control.takeAdds)state._activePlayer.inv.push({slot:2,id:1351,count:1});
            world.removeObj(state._activeObj,0);return 'taken';
        }
    };
    const host: DropLineageHost = {
        world,handlers,invDropAllOpcode:1,objTakeItemOpcode:2,deathLossProc:proc,despawnLifecycle:1,
        tick:()=>control.tick,enabled:()=>control.enabled,production:()=>control.production,playerCount:()=>control.players,
        isPlayer:p=>p instanceof NativePlayer,isObj:o=>o instanceof NativeObj,
        validatePlayer(p){if(!control.valid||p!==player)throw new Error('native account replaced');},verifyArm(){},
        privateItems(p){if(control.captureThrows)throw new Error('capture failed');
            return structuredClone({inventory:p.inv,worn:[],bank:[{slot:0,id:995,count:25}]});},
        groundAt:()=>ground,targetIsNonstackable:()=>true,settings:()=>({objDespawnScale:200,objRevealTicks:100})
    };
    const observer=installDropLineage(host,options);control.players=1;
    const identity={worldBootId:'ownedboot',accountId:'actor',accountEpoch:1,incarnationId:'p1'};
    observer.bind(player,identity,monitor);
    const arm=()=>observer.arm(player);
    const drop=(override={})=>{
        Object.assign(life,{phase:'dying',deathSequence:1,deathTick:control.tick});
        pending={self:player,_activePlayer:player,script:proc,intOperand:0,...override};
        const result=handlers[1](pending);return {obj:pending.dropped as NativeObj,result};
    };
    const alive=()=>Object.assign(life,{phase:'alive',respawnSequence:1,respawnTick:control.tick,completionSource:'player_death_queue'});
    const take=(obj:NativeObj,override={})=>handlers[2]({self:player,_activePlayer:player,_activeObj:obj,intOperand:0,...override});
    return {control,NativeObj,NativePlayer,player,world,host,observer,arm,drop,alive,take,life,ground};
}

test('exact native axe: empty baseline, public death drop, actual scaled lifetime and conserved native pickup',()=>{
    const f=fixture();expect(f.arm().ground).toEqual([]);
    const {obj,result}=f.drop();expect(result).toBe('dropped');
    f.alive();expect(f.take(obj)).toBe('taken');
    const audit=f.observer.audit(f.player);
    expect(audit.failure).toBeNull();expect(audit.object).toMatchObject({type:1351,removed:true,removal:'pickup'});
    const created=audit.events.find((e:any)=>e.kind==='created');
    expect(created).toMatchObject({deathSequence:1,requestedDuration:4,requestedReceiver64:'-1',
        after:{lifecycleTick:8,receiver64:'-1',count:1,isActive:true}});
    expect(audit.events.map((e:any)=>e.kind)).toEqual(['armed','created','drop-opcode','removed','pickup-opcode']);
    expect(audit.events.at(-1).after.inventory.filter((i:any)=>i.id===1351)).toHaveLength(1);
    audit.events.length=0;expect(f.observer.audit(f.player).events.length).toBe(5);
});

test('no-pickup control expires same Obj after measured countdown; removal is not credited as pickup',()=>{
    const f=fixture();f.arm();const {obj}=f.drop();f.alive();
    for(let n=0;n<8;n++){expect(obj.turn()).toBe('turned');f.control.tick++;}
    const audit=f.observer.audit(f.player);
    expect(audit.failure).toBeNull();expect(audit.object.removal).toBe('expiry');
    expect(audit.events.find((e:any)=>e.kind==='first-turn').expectedExpiryTick).toBe(17);
    expect(audit.events.find((e:any)=>e.kind==='removed')).toMatchObject({tick:17,cause:'expiry'});
    expect(audit.events.some((e:any)=>e.kind==='pickup-opcode')).toBe(false);
});

test('unarmed objects and unrelated item types preserve native calls without tracking',()=>{
    const f=fixture();const obj=new f.NativeObj();expect(f.world.addObj(obj,-1n,4)).toBe('added');
    expect(f.observer.audit(f.player).events).toEqual([]);
    f.world.removeObj(obj,0);f.arm();const other=new f.NativeObj();other.type=995;
    expect(f.world.addObj(other,-1n,4)).toBe('added');expect(f.observer.audit(f.player).failure).toBeNull();
});

test('pre-existing axe baseline fails instead of silently deleting or adopting it',()=>{
    const f=fixture();const obj=new f.NativeObj();f.world.addObj(obj,-1n,4);
    expect(()=>f.arm()).toThrow('already exists');expect(f.ground.has(obj)).toBe(true);
});

test.each(['script','self','intOperand'])('wrong native death context %s fails audit while native opcode still executes',field=>{
    const f=fixture();f.arm();const {result}=f.drop({[field]:field==='intOperand'?1:{}});
    expect(result).toBe('dropped');expect(f.control.dropCalls).toBe(1);expect(f.control.addCalls).toBe(1);
    expect(f.observer.audit(f.player).failure).toContain('exact owned pending');
});

test('unrelated identical-looking axe is never adopted as the tracked death object',()=>{
    const f=fixture();f.arm();const {obj}=f.drop();f.alive();const fake=new f.NativeObj();
    expect(f.world.addObj(fake,-1n,4)).toBe('added');
    expect(f.observer.audit(f.player).failure).toContain('unattributed');
    expect(f.observer.audit(f.player).object.isActive).toBe(true);expect(f.ground.has(obj)).toBe(true);
});

test('unknown removal and count mutation fail rather than claiming pickup',()=>{
    for(const operation of ['remove','change']){
        const f=fixture();f.arm();const {obj}=f.drop();
        if(operation==='remove')expect(f.world.removeObj(obj,0)).toBe('removed');
        else expect(f.world.changeObj(obj,2)).toBe('changed');
        expect(f.observer.audit(f.player).failure).not.toBeNull();
    }
});

test('native pickup by a different player cannot acquire the original account lineage',()=>{
    const f=fixture();f.arm();const {obj}=f.drop();f.alive();const other=new f.NativePlayer();
    expect(f.take(obj,{_activePlayer:other,self:other})).toBe('taken');
    expect(f.observer.audit(f.player).failure).toContain('identity');
});

test('native ground removal without actual inventory gain is not successful recovery',()=>{
    const f=fixture();f.arm();const {obj}=f.drop();f.alive();f.control.takeAdds=false;
    expect(f.take(obj)).toBe('taken');expect(obj.isActive).toBe(false);
    expect(f.observer.audit(f.player).failure).toContain('conserve');
});

test('native add returning without inserting its exact object cannot fabricate a drop receipt',()=>{
    const f=fixture();f.arm();f.control.insertAdd=false;
    expect(f.drop().result).toBe('dropped');expect(f.control.addCalls).toBe(1);
    const audit=f.observer.audit(f.player);expect(audit.failure).not.toBeNull();expect(audit.object).toBeNull();
});

test('observer capture failure does not suppress native pickup or replace native return',()=>{
    const f=fixture();f.arm();const {obj}=f.drop();f.alive();f.control.captureThrows=true;
    expect(f.take(obj)).toBe('taken');expect(f.control.takeCalls).toBe(1);
    expect(f.observer.audit(f.player).failure).toContain('capture failed');expect(obj.isActive).toBe(false);
});

test('observer failure never replaces the original thrown native exception',()=>{
    const f=fixture();f.arm();const {obj}=f.drop();f.alive();f.control.captureThrows=true;
    const original=new Error('original exception');f.control.failNativeTake=original;
    let thrown;try{f.take(obj);}catch(e){thrown=e;}
    expect(thrown).toBe(original);expect(f.control.takeCalls).toBe(1);
});

test('bounded ledger never evicts old evidence, and overflow is a sticky failed audit',()=>{
    const f=fixture({maxEvents:2});f.arm();const {obj}=f.drop();
    const first=f.observer.audit(f.player);expect(first.events).toHaveLength(2);expect(first.failure).toContain('ledger full');
    f.alive();expect(f.take(obj)).toBe('taken');expect(f.observer.audit(f.player).events).toEqual(first.events);
});

test.each(['enabled','production','players','valid'])('changed runtime gate %s fails audit without suppressing native behavior',key=>{
    const f=fixture();f.arm();const {obj}=f.drop();f.alive();
    (f.control as any)[key]=key==='production'?true:key==='players'?2:false;
    expect(f.take(obj)).toBe('taken');expect(f.observer.audit(f.player).failure).not.toBeNull();
});

test('lifetime reset or missed tick invalidates expiry evidence without changing native turn',()=>{
    for(const mode of ['remaining','tick']){
        const f=fixture();f.arm();const {obj}=f.drop();expect(obj.turn()).toBe('turned');
        f.control.tick+=mode==='tick'?2:1;if(mode==='remaining')obj.lifecycleTick=99;
        expect(obj.turn()).toBe('turned');expect(f.observer.audit(f.player).failure).not.toBeNull();
    }
});

test('installation and arm bounds refuse ordinary/shared/stackable or reused instrumentation',()=>{
    const f=fixture();expect(()=>installDropLineage(f.host)).toThrow();
    f.arm();expect(()=>f.arm()).toThrow('already armed');
    expect(()=>installDropLineage(f.host,{maxEvents:0})).toThrow('bounds');
});

test('private pending-script predicate is exact and absent from public lifecycle snapshot',()=>{
    const root={name:'[queue,player_death]',id:1};
    const player={levels:[10],applyDamage(n:number){this.levels[0]-=n;},changeStat(){},
        runScript(s:any){s.execution=2;return 2;}};
    const monitor=monitorLifecycle(player,0,()=>1,root);
    const s={script:root,self:player,_activePlayer:player,execution:0,fp:0,opcount:0};
    player.applyDamage(10);expect(monitor.isPendingDeathScript!(s)).toBe(false);
    player.runScript(s);expect(monitor.isPendingDeathScript!(s)).toBe(true);
    expect(monitor.isPendingDeathScript!({...s})).toBe(false);
    expect('isPendingDeathScript' in monitor.snapshot()).toBe(false);
});
