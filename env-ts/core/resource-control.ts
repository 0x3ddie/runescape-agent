/** Evaluator protocol only. It cannot choose actions or mutate game items. */
import type { ResourceIdentity } from '../engine/resource-capture';
type Native=any;
/** The longest capture an arm may ask for, in native ticks: baseline/resource_task.py MAX_HORIZON (one bound for the
 * goal tasks and free-play-v1's long lives; native-watch-v1 arms a life with the ticks its world has left). */
export const MAX_RESOURCE_HORIZON=150_000;
export interface ResourceControlHost<P>{
    bootId:string;
    enabled():boolean;
    production():boolean;
    /** players that are not a declared demo spectator (population-v2) */
    actorCount():number;
    findPlayer(account:string):P|null;
    identity(player:P):ResourceIdentity;
    /** `targets` picks the certified item families: 'logs' (default) or a wider scope the host declares. */
    arm(player:P,identity:ResourceIdentity,horizon:number,targets?:string):Native;
    read(player:P):Native;
    budget(player:P):Native;
    retire(player:P):Native;
    release(player:P):Native;
}
export function resourceControl<P>(host:ResourceControlHost<P>){
    return (request:Native)=>{
        if(!host.enabled() || host.production())throw new Error('resource control disabled');
        if(!request || typeof request!=='object' || Array.isArray(request))throw new Error('object resource request required');
        const operation=request.operation;
        if(!['identity','arm','read','budget','retire','release'].includes(operation))throw new Error('unknown resource operation');
        const keys=['operation','bootId','accountName',...(operation==='identity'?[]:['identity']),
            ...(operation==='arm'?['horizon',...(Object.hasOwn(request,'targets')?['targets']:[])]:[])];
        if(Object.keys(request).length!==keys.length || keys.some(k=>!Object.hasOwn(request,k)))throw new Error('unexpected resource fields');
        if(!host.bootId || request.bootId!==host.bootId)throw new Error('owned resource boot mismatch');
        if(typeof request.accountName!=='string' || !/^[a-z0-9]{1,12}$/.test(request.accountName))throw new Error('invalid resource account');
        if(host.actorCount()!==1)throw new Error('resource control requires exactly one actor');
        const player=host.findPlayer(request.accountName);
        if(!player)throw new Error('resource actor absent');
        const identity=host.identity(player);
        if(operation==='identity')return {type:'resource_identity',identity};
        const wanted=request.identity;
        if(!wanted || typeof wanted!=='object' || Array.isArray(wanted) || Object.keys(wanted).length!==4 ||
            Object.keys(identity).some(k=>wanted[k]!==identity[k as keyof ResourceIdentity]))throw new Error('resource incarnation mismatch');
        if(operation==='budget')return {type:'resource_budget',identity,budget:host.budget(player)};
        let audit:Native;
        if(operation==='arm'){
            if(!Number.isSafeInteger(request.horizon) || request.horizon<1 || request.horizon>MAX_RESOURCE_HORIZON)throw new Error('invalid resource horizon');
            if(Object.hasOwn(request,'targets') && (typeof request.targets!=='string' || !/^[a-z0-9-]{1,32}$/.test(request.targets)))
                throw new Error('invalid resource target scope');
            audit=host.arm(player,identity,request.horizon,request.targets);
        }else audit=host[operation as 'read'|'retire'|'release'](player);
        return {type:'resource_capture',identity,audit};
    };
}

export function resourceControlHttp(control:(request:Native)=>Native){
    return async(request:Request)=>{
        const reply=(data:Native,status=200)=>Response.json(data,{status,headers:{'Cache-Control':'no-store'}});
        if(new URL(request.url).pathname!=='/resource-capture')return reply({error:'not found'},404);
        if(request.headers.has('origin'))return reply({error:'Origin access refused'},403);
        try{
            if(request.method!=='POST' || new URL(request.url).search ||
                request.headers.get('content-type')?.split(';')[0].trim()!=='application/json')throw new Error('exact JSON POST required');
            const reader=request.body?.getReader();if(!reader)throw new Error('missing resource body');
            let length=0;const parts:Uint8Array[]=[];
            try{for(;;){const {done,value}=await reader.read();if(done)break;length+=value.length;
                if(length>4096){await reader.cancel();throw new Error('resource body too large');}parts.push(value);}}
            finally{reader.releaseLock();}
            const bytes=new Uint8Array(length);let offset=0;for(const part of parts){bytes.set(part,offset);offset+=part.length;}
            return reply(control(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes))));
        }catch(error){return reply({error:String((error as Error)?.message??error)},409);}
    };
}
