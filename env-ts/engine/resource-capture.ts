/** Passive raw resource evidence. This is NOT a reward calculator or lot allocator.
 * Exact ScriptFile identity tags production; bank stacks retain no invented IDs.
 * Native methods always execute even if observation fails. A sticky fault makes
 * the trace unusable. Consumers must additionally verify lifecycle and transport.
 */
import type { LifecycleMonitor } from './lifecycle';
type Native = any;
export type ResourceIdentity = { worldBootId: string; accountId: string; accountEpoch: number; incarnationId: string };
export interface ResourceCaptureHost {
    handlers: Record<number, (state: Native) => any>;
    inventoryPrototype: Native;
    world: Native;
    productionScript: object;
    addOpcode: number;
    /**
     * Switch to a named scope of certified item families (logs is the default:
     * productionScript and isTarget as given). Returns the families the trace
     * declares and which scripts produce them.
     */
    useScope?(name: string): { targets: Native[]; isProductionScript(script: unknown): boolean };
    tick(): number;
    enabled(): boolean;
    production(): boolean;
    /** players that are not a declared demo spectator (population-v2) */
    actorCount(): number;
    inventories(player: Native): { inventory: Native; bank: Native };
    validate(player: Native, identity: ResourceIdentity): void;
    isTarget(id: number): boolean;
    groundAt(level:number,x:number,z:number): Iterable<Native>;
    lifecycle?(player:Native): LifecycleMonitor;
    toolAvailable?(player:Native): boolean;
    slotSwap?: {
        opcode:number; inventoryId:number; bankId:number;
        activePlayer(state:Native):Native;
        container(player:Native,id:number):Native;
    };
}
type Snapshot = { inventory: {slot:number; id:number; count:number}[]; bank: {slot:number; id:number; count:number}[] };
type Binding = { player: Native; identity: ResourceIdentity; scope: string; targets: Native[] | null;
    initial: Snapshot; last: Snapshot;
    start: number; end: number; observedTick: number; boundaryTick: number; sequence: number; events: Native[];
    failure: string | null; closed: boolean; completed: boolean;
    monitor: LifecycleMonitor|null; unsubscribe: (()=>void)|null; terminal: Native; ownedContainers:Native };
type Context = { opcode: number; scriptId: number; scriptName: string; pc: number;
    productionSource: boolean; stackTail: Native[]; touched: Map<Binding, Snapshot>; resourceWrites:Set<Binding>; ground: Native[];
    slotSwap: {binding:Binding; receipt:Native}|null };
const installations = new WeakSet<object>();
const clone = <T>(v: T): T => structuredClone(v);
const same = (a: unknown,b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const require = (ok: unknown, message: string) => { if (!ok) throw new Error(message); };

/** Source-pinned private read: GameMap.getZone creates absent zones. */
export function readExistingResourceGround(gameMap:Native,zoneIndex:number,localCoord:number):Iterable<Native>{
    const zones=gameMap?.zonemap?.zones;
    require(zones instanceof Map,'native zone map layout changed');
    const zone=zones.get(zoneIndex);
    return zone ? zone.getObjsUnsafe(localCoord) : [];
}

export function installResourceCapture(host: ResourceCaptureHost, maxEvents=20000) {
    require(host.enabled() && !host.production() && host.actorCount() === 0 && !installations.has(host.world),
        'resource capture requires opt-in nonproduction installation before login');
    require(Number.isSafeInteger(maxEvents) && maxEvents > 0 && maxEvents <= 100000, 'invalid capture bound');
    const methods = ['add','remove','set','delete','removeAll'];
    require(methods.every(k=>typeof host.inventoryPrototype[k] === 'function') &&
        ['addObj','removeObj','changeObj','cycle'].every(k=>typeof host.world[k] === 'function') &&
        typeof host.handlers[host.addOpcode] === 'function' && !!host.productionScript, 'missing native capture surfaces');
    const bindings = new Map<Native,Binding>();
    if(host.slotSwap)require(Number.isSafeInteger(host.slotSwap.opcode) &&
        typeof host.handlers[host.slotSwap.opcode]==='function' &&
        [host.slotSwap.inventoryId,host.slotSwap.bankId].every(id=>Number.isSafeInteger(id)&&id>=0) &&
        host.slotSwap.inventoryId!==host.slotSwap.bankId,'invalid native slot-swap configuration');
    const containers = new WeakMap<object,Binding>();
    const objectIds = new WeakMap<object,number>();
    let nextObject=0, context: Context|null=null;
    let produces=(script:unknown)=>script===host.productionScript;
    let groundOrder=0;
    const groundCalls:{callId:number;rootCallId:number}[]=[];
    const failure = (b:Binding,error:unknown) => { b.failure ??= String((error as Error)?.message ?? error); };
    const safe = (b:Binding,fn:()=>void) => { try{fn();}catch(error){failure(b,error);} };
    const active = () => [...bindings.values()].filter(b=>!b.closed);
    function ownedContainers(player:Native){
        if(!host.slotSwap)return null;
        const invs=host.inventories(player), result:Native={};
        for(const location of ['inventory','bank'] as const){
            const id=location==='inventory'?host.slotSwap.inventoryId:host.slotSwap.bankId;
            const inv=invs[location];
            require(host.slotSwap.container(player,id)===inv && Number.isSafeInteger(inv?.capacity) && inv.capacity>0,
                'native owned-container mapping changed');
            result[location]={id,capacity:inv.capacity};
        }
        return result;
    }
    function snapshot(b:Binding):Snapshot {
        const invs=host.inventories(b.player), result={} as Snapshot;
        for(const location of ['inventory','bank'] as const){
            const inv=invs[location];
            require(inv && containers.get(inv) === b, 'native inventory identity changed');
            require(Array.isArray(inv.items), 'native inventory items missing');
            result[location]=inv.items.flatMap((item:Native,slot:number)=>{
                if(!item || !host.isTarget(item.id))return [];
                require(Number.isSafeInteger(item.count) && item.count>0, 'invalid native resource count');
                return [{slot,id:item.id,count:item.count}];
            });
        }
        return result;
    }
    function gate(b:Binding){
        require(host.enabled() && !host.production() && host.actorCount()===1, 'resource capture ownership/enablement changed');
        host.validate(b.player,b.identity);
        require(same(ownedContainers(b.player),b.ownedContainers),'owned container identity/capacity changed');
        const tick=host.tick();
        require(Number.isSafeInteger(tick) && tick>=b.observedTick, 'native resource clock regressed');
        b.observedTick=tick;
    }
    function record(b:Binding,kind:string,data:Native){
        gate(b);
        require(b.events.length<maxEvents, 'resource capture overflow');
        b.events.push(clone({sequence:++b.sequence,tick:host.tick(),kind,...data}));
    }
    function reconcile(b:Binding){
        gate(b);
        if(b.monitor){
            const life=b.monitor.snapshot();
            require(!life.failure && life.phase!=='failed','native lifecycle monitor failed');
            if(!b.closed)require(life.phase==='alive' && life.deathSequence===0 && life.respawnSequence===0,
                'unreported lifecycle transition');
        }
        require(same(snapshot(b),b.last), 'unobserved resource mutation');
    }
    function close(b:Binding,reason:'horizon'|'death',lifecycle:Native){
        const tools=host.toolAvailable?.(b.player) ?? null;
        require(tools===null || typeof tools==='boolean','invalid native tool availability');
        b.terminal=clone({reason,tick:host.tick(),holdings:b.last,toolAvailable:tools,lifecycle});
        record(b,'terminal',b.terminal);
        b.closed=true;b.completed=reason==='horizon';
        b.unsubscribe?.();b.unsubscribe=null;
    }
    function touch(b:Binding,ctx:Context){
        if(ctx.touched.has(b))return;
        reconcile(b);
        ctx.touched.set(b,clone(b.last));
    }
    function finishContext(ctx:Context){
        for(const [b,before] of ctx.touched){
            if(b.closed)continue;
            safe(b,()=>{
                const after=snapshot(b);
                const resourceMutation=!same(before,after) || ctx.resourceWrites.has(b);
                if(resourceMutation || ctx.ground.length){
                    record(b,'native-op',{opcode:ctx.opcode,scriptId:ctx.scriptId,scriptName:ctx.scriptName,
                        pc:ctx.pc,productionSource:ctx.productionSource,stackTail:ctx.stackTail,
                        before,after,resourceMutation,ground:ctx.ground,
                        ...(host.slotSwap?{ownedSlotSwap:ctx.slotSwap?.binding===b?ctx.slotSwap.receipt:null}:{})});
                }
                b.last=after;
            });
        }
    }
    function groundSnapshot(obj:Native){
        if(!objectIds.has(obj))objectIds.set(obj,++nextObject);
        return {captureObjectId:objectIds.get(obj),id:obj.type,count:obj.count,level:obj.level,x:obj.x,z:obj.z,
            lifecycle:obj.lifecycle,isActive:obj.isActive,receiver64:String(obj.receiver64)};
    }
    function groundTile(obj:Native){
        const result=[];
        for(const other of host.groundAt(obj.level,obj.x,obj.z)){
            if(!host.isTarget(other.type))continue;
            require(result.length<2048,'resource ground tile capture overflow');
            result.push(groundSnapshot(other));
        }
        return result.sort((a,b)=>a.captureObjectId!-b.captureObjectId!);
    }

    // Context wrapping does no inventory scanning for ordinary arithmetic opcodes.
    for(const [key,original] of Object.entries(host.handlers)){
        const opcode=Number(key);
        host.handlers[opcode]=function(state:Native){
            const owner=bindings.get(state?._activePlayer) ?? bindings.get(state?._activePlayer2);
            if(!owner || owner.closed)return original.call(this,state);
            const previous=context;
            const ctx:Context={opcode,scriptId:state.script?.id,scriptName:state.script?.name,pc:state.pc,
                productionSource:opcode===host.addOpcode && produces(state.script),
                stackTail:[],touched:new Map(),resourceWrites:new Set(),ground:[],slotSwap:null};
            // Capture stack metadata without popping or altering the native state.
            try { ctx.stackTail=state.intStack.slice(Math.max(0,state.isp-6),state.isp); }
            catch(error){for(const b of active())failure(b,error);}
            if(host.slotSwap && opcode===host.slotSwap.opcode){
                try{
                    const player=host.slotSwap.activePlayer(state), b=bindings.get(player);
                    if(b && !b.closed){
                        const args=ctx.stackTail.slice(-4);
                        require(args.length===4 && args.every(x=>Number.isSafeInteger(x)&&x>=0),'invalid native slot operands');
                        const meta=ownedContainers(player), locations=['inventory','bank'] as const;
                        const from=locations.find(k=>meta[k].id===args[0]),to=locations.find(k=>meta[k].id===args[1]);
                        // External inventories are deliberately not certified.
                        if(from && to){
                            require(args[2]<meta[from].capacity && args[3]<meta[to].capacity,'native slot out of bounds');
                            ctx.slotSwap={binding:b,receipt:{schema:'owned-slot-swap-v1',
                                from:{location:from,inventoryId:args[0],slot:args[2]},
                                to:{location:to,inventoryId:args[1],slot:args[3]}}};
                        }
                    }
                }catch(error){failure(owner,error);}
            }
            context=ctx;
            try{return original.call(this,state);}
            catch(error){for(const b of ctx.touched.keys())failure(b,'native resource opcode threw');throw error;}
            finally{finishContext(ctx);context=previous;}
        };
    }
    for(const method of methods){
        const original=host.inventoryPrototype[method];
        host.inventoryPrototype[method]=function(...args:Native[]){
            const b=containers.get(this);
            if(b && !b.closed){
                safe(b,()=>{
                    require(context!==null,'inventory mutation outside native opcode');
                    touch(b,context!);
                    // Equal-count replacement is invisible in a quantity delta.
                    // Record actual target slot writes, but not a failed add
                    // attempt or an unrelated food/tool mutation.
                    const targeted=(method==='set' && (host.isTarget(args[1]?.id) || host.isTarget(this.items[args[0]]?.id))) ||
                        (method==='delete' && host.isTarget(this.items[args[0]]?.id)) ||
                        (method==='removeAll' && this.items.some((item:Native)=>item && host.isTarget(item.id)));
                    if(targeted)context!.resourceWrites.add(b);
                });
            }
            try{return original.apply(this,args);}
            catch(error){if(b && !b.closed)failure(b,'native inventory mutation threw');throw error;}
        };
    }
    for(const method of ['addObj','removeObj','changeObj']){
        const original=host.world[method];
        host.world[method]=function(obj:Native,...args:Native[]){
            const observers=active();
            if(!obj || !host.isTarget(obj.type) || !observers.length)return original.call(this,obj,...args);
            let before:Native=null,tileBefore:Native=null;
            const parent=groundCalls.at(-1),callId=++groundOrder;
            const rootCallId=parent?.rootCallId ?? callId;
            for(const b of observers)safe(b,()=>{gate(b);if(context)touch(b,context);
                before=groundSnapshot(obj);tileBefore=groundTile(obj);});
            let result:Native;
            groundCalls.push({callId,rootCallId});
            try{result=original.call(this,obj,...args);}
            catch(error){for(const b of observers)failure(b,'native ground mutation threw');throw error;}
            finally{groundCalls.pop();}
            const endId=++groundOrder;
            for(const b of observers)safe(b,()=>{
                // Nested changeObj merges and cross-tile eviction removeObj calls
                // overlap the outer snapshot. Entry/exit order lets replay use
                // their actual net effects without counting a merge twice.
                const event={method,callId,endId,rootCallId,parentCallId:parent?.callId ?? null,
                    before,after:groundSnapshot(obj),tileBefore,tileAfter:groundTile(obj)};
                if(context)context.ground.push(event);
                else record(b,'native-ground',event);
            });
            return result;
        };
    }
    const cycle=host.world.cycle;
    host.world.cycle=function(...args:Native[]){
        const observed=active(), before=host.tick();
        for(const b of observed)safe(b,()=>{
            require(before===b.boundaryTick,'missing native cycle boundary');
            reconcile(b);
        });
        let result:Native;
        try{result=cycle.apply(this,args);}
        catch(error){for(const b of observed)failure(b,'native cycle threw');throw error;}
        for(const b of observed)safe(b,()=>{
            if(b.closed)return;
            require(host.tick()===before || host.tick()===before+1,'native cycle tick jump');
            reconcile(b);
            if(b.closed)return;
            if(host.tick()!==before){
                record(b,'tick-boundary',{holdings:b.last});
                b.boundaryTick=host.tick();
                require(host.tick()<=b.end,'resource horizon overshot');
                if(host.tick()===b.end)close(b,'horizon',b.monitor?.snapshot() ?? null);
            }
        });
        return result;
    };
    installations.add(host.world);
    return {
        bind(player:Native,identity:ResourceIdentity,horizon:number,scope='logs'){
            require(bindings.size===0 && Number.isSafeInteger(horizon) && horizon>0, 'capture allows one actor and positive horizon');
            require(identity && ['worldBootId','accountId','incarnationId'].every(k=>typeof (identity as Native)[k]==='string' && (identity as Native)[k]) &&
                Number.isSafeInteger(identity.accountEpoch) && identity.accountEpoch>0,'explicit capture identity required');
            const start=host.tick();
            require(Number.isSafeInteger(start) && start>=0 && Number.isSafeInteger(start+horizon),'invalid native start/horizon');
            const invs=host.inventories(player);
            require(invs.inventory && invs.bank && invs.inventory!==invs.bank,'distinct native inventories required');
            require(scope==='logs' || typeof host.useScope==='function','unsupported resource target scope');
            const chosen=host.useScope?.(scope) ?? null;
            produces=scope==='logs' ? (script:unknown)=>script===host.productionScript : chosen!.isProductionScript;
            const b:Binding={player,identity:clone(identity),scope,targets:scope==='logs'?null:clone(chosen!.targets),
                initial:null!,last:null!,start,end:start+horizon,
                observedTick:start,boundaryTick:start,sequence:0,events:[],failure:null,closed:false,completed:false,
                monitor:null,unsubscribe:null,terminal:null,ownedContainers:ownedContainers(player)};
            containers.set(invs.inventory,b);containers.set(invs.bank,b);
            try{
                gate(b);b.initial=snapshot(b);b.last=clone(b.initial);
                require(b.initial.inventory.length===0 && b.initial.bank.length===0,'resource baseline requires zero target items');
                b.monitor=host.lifecycle?.(player) ?? null;
                if(b.monitor){
                    const initial=b.monitor.snapshot();
                    require(initial.phase==='alive' && !initial.failure && initial.deathSequence===0 && initial.respawnSequence===0,
                        'resource capture requires fresh alive lifecycle');
                    b.unsubscribe=b.monitor.subscribe(event=>{
                        if(b.closed)return;
                        safe(b,()=>{
                            require(event.kind==='death' && event.eventSequence===1 && event.deathSequence===1 &&
                                event.respawnSequence===0 && event.deathTick===host.tick() && event.phase==='dying' && !event.failure,
                                'invalid first-life death event');
                            gate(b);
                            require(host.tick()<b.end,'death outside open resource horizon');
                            require(same(snapshot(b),b.last),'death overlaps unrecorded resource mutation');
                            close(b,'death',event);
                        });
                    });
                }
                bindings.set(player,b);
            }catch(error){b.unsubscribe?.();containers.delete(invs.inventory);containers.delete(invs.bank);throw error;}
        },
        audit(player:Native){
            const b=bindings.get(player);require(b,'resource actor not bound');
            if(!b!.closed)safe(b!,()=>reconcile(b!));
            return clone({schema:'native-resource-capture-v2',flowEvidenceVersion:1,
                ...(host.slotSwap?{slotSwapEvidenceVersion:1,ownedContainers:b!.ownedContainers}:{}),
                // A logs trace is unchanged; a wider scope declares its item families.
                ...(b!.targets?{targetScope:b!.scope,targets:b!.targets}:{}),
                identity:b!.identity,startTick:b!.start,endTick:b!.end,
                observedTick:b!.observedTick,initial:b!.initial,last:b!.last,events:b!.events,
                failure:b!.failure,completed:b!.completed && !b!.failure,closed:b!.closed,
                evidenceComplete:b!.closed && !!b!.terminal && !b!.failure,
                lifecycleObserved:!!b!.monitor,terminal:b!.terminal});
        },
        /** The window is open: bound, before its terminal (combat evidence shares it). */
        open(player:Native){
            const b=bindings.get(player);
            return !!b && !b.closed;
        },
        budget(player:Native){
            const b=bindings.get(player);require(b,'resource actor not bound');
            // Do not clone the growing trace for every policy decision. Closed
            // receipts retain their terminal tick even if the world keeps running.
            require(host.enabled() && !host.production() && host.actorCount()===1,'resource budget ownership changed');
            host.validate(b!.player,b!.identity);
            if(!b!.closed)safe(b!,()=>reconcile(b!));
            require(!b!.failure,'resource budget evidence failed');
            require(!b!.closed || !!b!.terminal,'resource budget retired without terminal');
            const tick=b!.closed?b!.terminal.tick:b!.observedTick;
            require(Number.isSafeInteger(tick) && b!.start<=tick && tick<=b!.end,'invalid resource budget clock');
            return {schema:'resource-budget-v1',startTick:b!.start,endTick:b!.end,observedTick:tick,
                horizon:b!.end-b!.start,remainingTicks:b!.end-tick,
                status:b!.closed?b!.terminal.reason:'active'};
        },
        retire(player:Native){
            const b=bindings.get(player);require(b,'resource actor not bound');
            if(!b!.closed){failure(b!,'capture retired before complete horizon');b!.closed=true;}
            b!.unsubscribe?.();b!.unsubscribe=null;
        },
        release(player:Native){
            const b=bindings.get(player);require(b && b.closed,'archive a closed capture before releasing it');
            const receipt=this.audit(player);
            const invs=host.inventories(player);
            for(const inv of [invs.inventory,invs.bank])if(inv && containers.get(inv)===b)containers.delete(inv);
            bindings.delete(player);
            return receipt;
        }
    };
}
