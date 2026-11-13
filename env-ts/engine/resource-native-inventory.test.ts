/** Real vendor Inventory with tiny synthetic configs; no world or asset boot. */
import { expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

// The real Inventory needs the engine checkout in vendor/rs-sdk (vendor-patches/README.md); without it this file's tests skip.
const engine = existsSync(new URL('../../vendor/rs-sdk', import.meta.url));
const vendor = '../../vendor/rs-sdk/server/engine/src';
const { Inventory } = engine ? await import(`${vendor}/engine/Inventory`) : ({} as any);
const ObjType = engine ? (await import(`${vendor}/cache/config/ObjType`)).default : ({} as any);
const InvType = engine ? (await import(`${vendor}/cache/config/InvType`)).default : ({} as any);
import { installResourceCapture } from './resource-capture';

test.skipIf(!engine)('real Inventory supports partial bank withdrawals, full stacks and slot mutations',()=>{
    const oldObj=ObjType.configs[1511], oldInv=(InvType as any).configs[1];
    ObjType.configs[1511]=new ObjType(1511);
    (InvType as any).configs[1]=new InvType(1);
    const player={inventory:new Inventory(1,3,Inventory.NORMAL_STACK),bank:new Inventory(1,2,Inventory.ALWAYS_STACK)};
    let tick=0,players=0;
    const script={id:51,name:'[label,get_logs]'};
    const world={cycle(){tick++;},addObj(){},removeObj(){},changeObj(){}};
    const methods=['add','remove','set','delete','removeAll'];
    const original=Object.fromEntries(methods.map(k=>[k,(Inventory.prototype as any)[k]]));
    const handlers:Record<number,()=>any>={
        1:()=>player.inventory.add(1511,1),
        2:()=>{const n=player.inventory.remove(1511,3);return player.bank.add(1511,n);},
        3:()=>{const n=player.bank.remove(1511,2);return player.inventory.add(1511,n);},
        4:()=>player.inventory.delete(0)
    };
    try{
        const capture=installResourceCapture({handlers,inventoryPrototype:Inventory.prototype,world,
            productionScript:script,addOpcode:1,tick:()=>tick,enabled:()=>true,production:()=>false,
            actorCount:()=>players,inventories:p=>p,validate(){},isTarget:id=>id===1511,groundAt:()=>[]});
        players=1;capture.bind(player,{worldBootId:'boot',accountId:'actor',accountEpoch:1,incarnationId:'p'},1);
        const run=(n:number)=>(handlers[n] as any)({_activePlayer:player,script,pc:0,intStack:[],isp:0});
        expect([run(1),run(1),run(1),run(1)]).toEqual([1,1,1,0]);
        expect(run(2)).toBe(3);expect(run(3)).toBe(2);run(4);world.cycle();
        const trace=capture.audit(player);
        expect(trace.failure).toBeNull();expect(trace.completed).toBe(true);
        expect(trace.last.inventory).toEqual([{slot:1,id:1511,count:1}]);
        expect(trace.last.bank).toEqual([{slot:0,id:1511,count:1}]);
        expect(trace.events.filter((e:any)=>e.productionSource).length).toBe(3);
    }finally{
        for(const [k,v] of Object.entries(original))(Inventory.prototype as any)[k]=v;
        if(oldObj===undefined)delete ObjType.configs[1511];else ObjType.configs[1511]=oldObj;
        if(oldInv===undefined)delete (InvType as any).configs[1];else (InvType as any).configs[1]=oldInv;
    }
});
