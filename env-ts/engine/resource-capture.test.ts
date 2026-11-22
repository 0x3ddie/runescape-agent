import { expect, test } from 'bun:test';
import { HOUR_TICKS, eventLimit, installResourceCapture, readExistingResourceGround } from './resource-capture';
import { monitorLifecycle } from './lifecycle';

const identity={worldBootId:'boot',accountId:'actor',accountEpoch:1,incarnationId:'instance'};
function fixture(maxEvents=100,withLifecycle=false,withSlotSwap=false,targets=new Set([1511,1512]),useScope?:(name:string)=>any){
    const control={tick:10,players:0,enabled:true,production:false,valid:true,tool:true};
    class Inv {
        capacity=28;
        items:any[]=[];
        set(slot:number,item:any){this.items[slot]=item;}
        add(id:number,count:number){this.set(this.items.length,{id,count});return count;}
        remove(id:number,count:number){
            const slot=this.items.findIndex(i=>i?.id===id);if(slot<0)return 0;
            const n=Math.min(count,this.items[slot].count);this.items[slot].count-=n;
            if(!this.items[slot].count)this.delete(slot);return n;
        }
        delete(slot:number){this.set(slot,null);}
        removeAll(){this.items=[];}
    }
    const player={inventory:new Inv(),bank:new Inv(),levels:[10],
        applyDamage(amount:number){this.levels[0]-=amount;},changeStat(_stat:number){},runScript(){return 0;}},
        script={name:'[label,get_logs]',id:51},deathScript={name:'[queue,player_death]',id:99};
    const external=new Inv(),inventoryMap=new Map([[93,player.inventory],[95,player.bank],[96,external]]);
    const ground=new Set<any>();
    const world={
        cycle(){control.tick++;return 'cycle-result';},
        addObj(obj:any){
            const existing=[...ground].find(o=>o.type===1512 && o.type===obj.type);
            if(existing)world.changeObj(existing,existing.count+obj.count);
            else{ground.add(obj);obj.isActive=true;}
            return 'add-result';
        },
        removeObj(obj:any){ground.delete(obj);obj.isActive=false;return 'remove-result';},
        changeObj(obj:any,count:number){obj.count=count;return 'change-result';}
    };
    const handlers:Record<number,(state:any)=>any>={
        1:s=>{const [id,n]=s.intStack.splice(0);s.isp=0;return player.inventory.add(id,n);},
        2:()=>{const n=player.inventory.remove(1511,1);player.bank.add(1511,n);return n;},
        3:s=>{player.inventory.remove(1511,1);return world.addObj(s.obj);},
        4:s=>{player.inventory.add(s.obj.type,s.obj.count);return world.removeObj(s.obj);},
        5:()=>{player.inventory.add(1511,1);throw new Error('native failure');},
        6:()=>player.inventory.removeAll(),
        7:()=>42,
        8:()=>player.inventory.set(0,{...player.inventory.items[0]}),
        9:()=>player.inventory.add(999,1),
        4322:s=>{
            const [from,to,fromSlot,toSlot]=s.intStack.slice(s.isp-4,s.isp);s.isp-=4;
            const a=inventoryMap.get(from)!,b=inventoryMap.get(to)!;
            const first=a.items[fromSlot],second=b.items[toSlot];
            b.set(toSlot,first??null);a.set(fromSlot,second??null);
            return 'native-swap-result';
        }
    };
    const capture=installResourceCapture({handlers,inventoryPrototype:Inv.prototype,world,
        productionScript:script,addOpcode:1,tick:()=>control.tick,enabled:()=>control.enabled,
        production:()=>control.production,actorCount:()=>control.players,
        inventories:p=>p,isTarget:id=>targets.has(id),useScope,
        groundAt:()=>ground,
        lifecycle:withLifecycle?p=>monitorLifecycle(p,0,()=>control.tick,deathScript):undefined,
        toolAvailable:()=>control.tool,
        slotSwap:withSlotSwap?{opcode:4322,inventoryId:93,bankId:95,
            activePlayer:s=>s._activePlayer,container:(_p,id)=>inventoryMap.get(id)}:undefined,
        validate(p,id){if(!control.valid || p!==player || id.accountId!=='actor')throw new Error('identity changed');}
    },maxEvents);
    control.players=1;
    const arm=(horizon=2)=>capture.bind(player,identity,horizon);
    const run=(op:number,override:any={})=>handlers[op]({_activePlayer:player,script,pc:5,intStack:[1511,1],isp:2,...override});
    const audit=()=>capture.audit(player);
    const obj={type:1511,count:1,x:1,z:2,level:0,lifecycle:1,isActive:false,receiver64:1n};
    return {control,player,script,world,handlers,capture,arm,run,audit,obj,ground,external,inventoryMap};
}

test('owned slot receipt authenticates containers and preserves native stack/return semantics',()=>{
    const f=fixture(100,false,true);f.arm();f.run(1);f.run(2);
    const state={_activePlayer:f.player,script:f.script,pc:5,intStack:[777,95,95,0,4],isp:5};
    expect(f.handlers[4322](state)).toBe('native-swap-result');expect(state.isp).toBe(1);
    const trace=f.audit();expect(trace.failure).toBeNull();
    expect(trace.slotSwapEvidenceVersion).toBe(1);
    expect(trace.ownedContainers).toEqual({inventory:{id:93,capacity:28},bank:{id:95,capacity:28}});
    expect(trace.events.slice(0,2).every((e:any)=>e.ownedSlotSwap===null)).toBe(true);
    expect(trace.events.at(-1).ownedSlotSwap).toEqual({schema:'owned-slot-swap-v1',
        from:{location:'bank',inventoryId:95,slot:0},to:{location:'bank',inventoryId:95,slot:4}});
    expect(trace.events.at(-1).stackTail).toEqual([777,95,95,0,4]);
    expect(trace.last.bank).toEqual([{slot:4,id:1511,count:1}]);
});

test('external inventory swap executes but receives no owned-transfer exemption',()=>{
    const f=fixture(100,false,true);f.arm();f.run(1);
    f.run(4322,{intStack:[93,96,0,0],isp:4});
    const trace=f.audit();expect(trace.failure).toBeNull();
    expect(trace.events.at(-1).ownedSlotSwap).toBeNull();
    expect(trace.last.inventory).toEqual([]);expect(f.external.items[0]).toEqual({id:1511,count:1});
});

test('cross-owned-container and equal-looking slot swaps retain a receipt',()=>{
    const f=fixture(100,false,true);f.arm();f.run(1);f.run(1);
    f.run(4322,{intStack:[93,93,0,1],isp:4});
    expect(f.audit().events.at(-1).ownedSlotSwap.from.location).toBe('inventory');
    f.run(4322,{intStack:[93,95,0,2],isp:4});
    expect(f.audit().events.at(-1).ownedSlotSwap.to).toEqual({location:'bank',inventoryId:95,slot:2});
    expect(f.audit().failure).toBeNull();
});

test('native container remapping invalidates evidence without blocking the opcode',()=>{
    const f=fixture(100,false,true);f.arm();f.run(1);f.inventoryMap.set(95,f.external);
    expect(f.run(4322,{intStack:[93,93,0,1],isp:4})).toBe('native-swap-result');
    expect(f.player.inventory.items[1]).toEqual({id:1511,count:1});
    expect(f.audit().failure).toContain('mapping changed');
});

test('forged slot operands invalidate evidence while native handler still executes',()=>{
    const f=fixture(100,false,true);f.arm();f.run(1);
    expect(f.run(4322,{intStack:[93,93,0,99],isp:4})).toBe('native-swap-result');
    expect(f.audit().failure).toContain('slot out of bounds');
    expect(f.player.inventory.items[99]).toEqual({id:1511,count:1});
});

test('exact script identity tags production; source name alone cannot',()=>{
    const f=fixture();f.arm();expect(f.run(1)).toBe(1);
    f.run(1,{script:{...f.script}});
    const trace=f.audit();expect(trace.failure).toBeNull();
    expect(trace.events.map((e:any)=>e.productionSource)).toEqual([true,false]);
    expect(trace.events[0].stackTail).toEqual([1511,1]);
    expect(trace.events[0].before.inventory).toEqual([]);
    expect(trace.events[0].after.inventory).toEqual([{slot:0,id:1511,count:1}]);
});
test('transfer groups removal and addition; no manufactured production',()=>{
    const f=fixture();f.arm();f.run(1);expect(f.run(2)).toBe(1);
    const e=f.audit().events[1];expect(e.productionSource).toBe(false);
    expect(e.before.inventory[0].count).toBe(1);expect(e.after.inventory).toEqual([]);
    expect(e.after.bank[0].count).toBe(1);expect(f.audit().events.length).toBe(2);
});
test('ground identity survives drop/reacquisition and expiry outside scripts',()=>{
    const f=fixture();f.arm();f.run(1);
    expect(f.run(3,{obj:f.obj})).toBe('add-result');
    expect(f.run(4,{obj:f.obj})).toBe('remove-result');
    const trace=f.audit();
    expect(trace.events[1].ground[0].after.captureObjectId).toBe(trace.events[2].ground[0].before.captureObjectId);
    f.world.removeObj(f.obj);expect(f.audit().events.at(-1).kind).toBe('native-ground');
});
test('actual native cycle boundaries close exact horizon and freeze evidence',()=>{
    const f=fixture();f.arm();f.run(1);
    expect(f.world.cycle()).toBe('cycle-result');expect(f.audit().completed).toBe(false);
    f.world.cycle();const before=f.audit();expect(before.completed).toBe(true);
    f.run(1);f.world.cycle();expect(f.audit()).toEqual(before);
    const copy=f.audit();copy.events[0].after.inventory[0].count=999;
    expect(f.audit().events[0].after.inventory[0].count).toBe(1);
});
test('direct unobserved mutation invalidates even when count is later restored',()=>{
    const f=fixture();f.arm();f.player.inventory.add(1511,1);f.player.inventory.remove(1511,1);
    expect(f.audit().failure).toContain('outside native opcode');
    f.world.cycle();f.world.cycle();expect(f.audit().completed).toBe(false);
});
test('raw array mutation detected by next checkpoint',()=>{
    const f=fixture();f.arm();f.player.inventory.items.push({id:1511,count:5});
    expect(f.audit().failure).toContain('unobserved resource mutation');
});
test('same-count slot substitution is not hidden by aggregate counts',()=>{
    const f=fixture();f.arm();f.run(1);f.player.inventory.items=[null,{id:1511,count:1}];
    expect(f.audit().failure).toContain('unobserved resource mutation');
});
test('native exceptions and return values are preserved',()=>{
    const f=fixture();f.arm();expect(()=>f.run(5)).toThrow('native failure');
    expect(f.audit().failure).toContain('native resource opcode threw');expect(f.run(7)).toBe(42);
});
test('overflow is sticky and never cancels native mutation',()=>{
    const f=fixture(1);f.arm();f.run(1);expect(f.run(1)).toBe(1);
    expect(f.player.inventory.items.length).toBe(2);expect(f.audit().failure).toContain('overflow');
});
test('the event bound is maxEvents a started game hour of the horizon: an hour keeps it, a long life gets an hour\'s room each',()=>{
    for(const [horizon,room] of [[2,2],[6000,2],[6001,4],[36000,12]] as const){
        const f=fixture(2);f.arm(horizon);
        for(let k=0;k<room;k++)f.run(1);
        expect(f.audit().failure).toBeNull();expect(f.audit().events.length).toBe(room);
        f.run(1);expect(f.audit().failure).toContain('overflow');expect(f.audit().events.length).toBe(room);
    }
    expect(eventLimit(20000,36000)).toBe(120000);expect(eventLimit(20000,1200)).toBe(20000);expect(HOUR_TICKS).toBe(6000);
});
for(const fault of ['identity','enablement','density','clock','container'])test(`reject ${fault} change`,()=>{
    const f=fixture();f.arm();
    if(fault==='identity')f.control.valid=false;
    if(fault==='enablement')f.control.enabled=false;
    if(fault==='density')f.control.players=2;
    if(fault==='clock')f.control.tick=9;
    if(fault==='container')f.player.bank={items:[]} as any;
    expect(f.audit().failure).not.toBeNull();
});
test('premature retirement is not a completed horizon',()=>{
    const f=fixture();f.arm();f.capture.retire(f.player);
    expect(f.audit().completed).toBe(false);expect(f.audit().failure).toContain('before complete horizon');
});
test('preloaded ordinary logs and certificates reject the baseline',()=>{
    for(const id of [1511,1512]){
        const f=fixture();f.player.bank.add(id,1);expect(()=>f.arm()).toThrow('zero target items');
    }
});
test('missed native cycle is an evidence failure, not plausible elapsed time',()=>{
    const f=fixture();f.arm();f.control.tick++;f.world.cycle();
    expect(f.audit().failure).toContain('missing native cycle boundary');
    expect(f.audit().completed).toBe(false);
});
test('release returns evidence and requires explicit closure before another episode',()=>{
    const f=fixture();f.arm(1);expect(()=>f.capture.release(f.player)).toThrow('closed capture');
    f.world.cycle();const receipt=f.capture.release(f.player);expect(receipt.completed).toBe(true);
    f.arm(1);f.world.cycle();expect(f.audit().startTick).toBe(11);
    expect(receipt.startTick).toBe(10);
});
test('ground stack merging records actual destination instead of inventing insertion',()=>{
    const f=fixture();f.arm();const existing={...f.obj,type:1512,count:3,isActive:true};
    f.ground.add(existing);const incoming={...f.obj,type:1512,count:2};
    f.world.addObj(incoming);const event=f.audit().events.at(-1);
    expect(event.tileBefore[0].count).toBe(3);expect(event.tileAfter[0].count).toBe(5);
    expect(event.tileBefore[0].captureObjectId).toBe(event.tileAfter[0].captureObjectId);
    expect(event.after.isActive).toBe(false);expect(event.tileAfter.length).toBe(1);
});
test('ground inspection never creates an absent native zone',()=>{
    const zones=new Map(),map={zonemap:{zones},getZone(){throw new Error('mutating lookup used');}};
    expect([...readExistingResourceGround(map,100,4)]).toEqual([]);expect(zones.size).toBe(0);
    zones.set(100,{getObjsUnsafe:(coord:number)=>[{coord}]});
    expect([...readExistingResourceGround(map,100,4)]).toEqual([{coord:4}]);
    expect(()=>readExistingResourceGround({},100,4)).toThrow('layout changed');
});
test('nested ground merge records entry/exit order instead of pretending two transfers',()=>{
    const f=fixture();f.arm();
    const existing={...f.obj,type:1512,count:3,isActive:true};f.ground.add(existing);
    const incoming={...f.obj,type:1512,count:2};f.world.addObj(incoming);
    const rows=f.audit().events;
    expect(rows.length).toBe(2);
    const [child,parent]=rows;
    expect(child.kind).toBe('native-ground');expect(parent.kind).toBe('native-ground');
    expect([parent.callId,child.callId,child.endId,parent.endId]).toEqual([1,2,3,4]);
    expect(child.parentCallId).toBe(parent.callId);expect(child.rootCallId).toBe(parent.callId);
    expect(parent.parentCallId).toBeNull();expect(parent.rootCallId).toBe(parent.callId);
    expect(child.tileBefore[0].count).toBe(3);expect(child.tileAfter[0].count).toBe(5);
    expect(parent.tileBefore[0].count).toBe(3);expect(parent.tileAfter[0].count).toBe(5);
});
test('same-count target writes remain visible while unrelated item writes are omitted',()=>{
    const f=fixture();f.arm();f.run(1);f.run(8);
    const trace=f.audit();expect(trace.flowEvidenceVersion).toBe(1);
    expect(trace.events.length).toBe(2);
    expect(trace.events[1].resourceMutation).toBe(true);
    expect(trace.events[1].before).toEqual(trace.events[1].after);
    f.run(9);expect(f.audit().events.length).toBe(2);
});
test('native death closes the first life and healing cannot erase it',()=>{
    const f=fixture(100,true);f.arm(5);f.run(1);f.run(2);f.run(1);
    f.player.applyDamage(10);
    const receipt=f.audit();
    expect(receipt.completed).toBe(false);expect(receipt.evidenceComplete).toBe(true);
    expect(receipt.lifecycleObserved).toBe(true);expect(receipt.terminal.reason).toBe('death');
    expect(receipt.terminal.lifecycle.deathSequence).toBe(1);
    expect(receipt.terminal.holdings.bank[0].count).toBe(1);
    f.player.levels[0]=10;f.player.changeStat(0);f.run(1);f.world.cycle();
    expect(f.audit()).toEqual(receipt);
});
test('death before the last boundary fails; death after closure leaves the finished window alone',()=>{
    const dying=fixture(100,true);dying.arm(2);dying.world.cycle();dying.player.applyDamage(10);
    expect(dying.audit().terminal.tick).toBe(11);expect(dying.audit().terminal.reason).toBe('death');
    const alive=fixture(100,true);alive.arm(1);alive.control.tool=false;alive.world.cycle();
    const receipt=alive.audit();expect(receipt.terminal.reason).toBe('horizon');
    expect(receipt.terminal.toolAvailable).toBe(false);expect(receipt.terminal.lifecycle.deathSequence).toBe(0);
    alive.player.applyDamage(10);expect(alive.audit()).toEqual(receipt);
});
test('death during an unrecorded resource mutation invalidates ambiguous evidence',()=>{
    const f=fixture(100,true);f.arm();f.player.inventory.items.push({id:1511,count:1});
    f.player.applyDamage(10);expect(f.audit().evidenceComplete).toBe(false);
    expect(f.audit().failure).toContain('unrecorded resource mutation');
});

test('public budget follows native ticks and freezes at the terminal boundary without exposing holdings',()=>{
    const f=fixture(100,true);f.arm(2);f.run(1);
    const a=f.capture.budget(f.player);
    expect(a).toEqual({schema:'resource-budget-v1',startTick:10,endTick:12,observedTick:10,horizon:2,remainingTicks:2,status:'active'});
    f.world.cycle();expect(f.capture.budget(f.player).remainingTicks).toBe(1);
    f.world.cycle();const receipt=f.audit();const closed=f.capture.budget(f.player);
    expect(closed.status).toBe('horizon');expect(closed.remainingTicks).toBe(0);
    f.world.cycle();expect(f.capture.budget(f.player)).toEqual(closed);expect(f.audit()).toEqual(receipt);
    expect(closed).not.toHaveProperty('holdings');expect(closed).not.toHaveProperty('events');
});
test('public clock reports death before horizon and refuses retirement/faulted evidence',()=>{
    const f=fixture(100,true);f.arm(5);f.world.cycle();f.player.applyDamage(10);
    const b=f.capture.budget(f.player);expect(b.status).toBe('death');expect(b.remainingTicks).toBe(4);
    f.world.cycle();expect(f.capture.budget(f.player)).toEqual(b);
    const retired=fixture();retired.arm();retired.capture.retire(retired.player);
    expect(()=>retired.capture.budget(retired.player)).toThrow();
    const bad=fixture();bad.arm();bad.player.inventory.items.push({id:1511,count:1});
    expect(()=>bad.capture.budget(bad.player)).toThrow('evidence failed');
});

test('a target scope chosen at arm switches production scripts and is declared in the trace',()=>{
    const targets=new Set([1511,1512]),ore={name:'[label,get_ore_normal]',id:52};
    const declared=[{base:1511,note:1512,value:4},{base:436,note:437,value:3}];
    const f=fixture(100,false,false,targets,name=>{
        if(name!=='gathering-v1')throw new Error('unknown scope');
        targets.add(436);targets.add(437);
        return {targets:declared,isProductionScript:(s:unknown)=>s===f.script || s===ore};
    });
    f.capture.bind(f.player,identity,2,'gathering-v1');
    f.run(1,{script:ore,intStack:[436,1]});
    const trace=f.audit();
    expect(trace.targetScope).toBe('gathering-v1');expect(trace.targets).toEqual(declared);
    expect(trace.events.find((e:any)=>e.kind==='native-op').productionSource).toBe(true);
    expect(trace.last.inventory).toEqual([{slot:0,id:436,count:1}]);
    // The default scope never declares families, and a host without scopes refuses one.
    const logs=fixture();logs.arm();expect('targetScope' in logs.audit()).toBe(false);
    const plain=fixture();expect(()=>plain.capture.bind(plain.player,identity,2,'gathering-v1')).toThrow('unsupported resource target scope');
});
