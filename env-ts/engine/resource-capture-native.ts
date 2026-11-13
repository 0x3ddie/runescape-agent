/** Shared native adapter for engine Player and Chromium NetworkPlayer.
 * Explicit opt-in only. No policy fields, actions, item mutations or reward.
 */
import Player from '../../vendor/rs-sdk/server/engine/src/engine/entity/Player.js';
import { NetworkPlayer } from '../../vendor/rs-sdk/server/engine/src/engine/entity/NetworkPlayer.js';
import { Inventory } from '../../vendor/rs-sdk/server/engine/src/engine/Inventory.js';
import { CoordGrid } from '../../vendor/rs-sdk/server/engine/src/engine/CoordGrid.js';
import ZoneMap from '../../vendor/rs-sdk/server/engine/src/engine/zone/ZoneMap.js';
import InvType from '../../vendor/rs-sdk/server/engine/src/cache/config/InvType.js';
import ObjType from '../../vendor/rs-sdk/server/engine/src/cache/config/ObjType.js';
import DbTableType from '../../vendor/rs-sdk/server/engine/src/cache/config/DbTableType.js';
import DbRowType from '../../vendor/rs-sdk/server/engine/src/cache/config/DbRowType.js';
import ParamType from '../../vendor/rs-sdk/server/engine/src/cache/config/ParamType.js';
import ScriptRunner from '../../vendor/rs-sdk/server/engine/src/engine/script/ScriptRunner.js';
import ScriptProvider from '../../vendor/rs-sdk/server/engine/src/engine/script/ScriptProvider.js';
import { ScriptOpcode } from '../../vendor/rs-sdk/server/engine/src/engine/script/ScriptOpcode.js';
import World from '../../vendor/rs-sdk/server/engine/src/engine/World.js';
import Environment from '../../vendor/rs-sdk/server/engine/src/util/Environment.js';
import { PlayerStat } from '../../vendor/rs-sdk/server/engine/src/engine/entity/PlayerStat.js';
import Npc from '../../vendor/rs-sdk/server/engine/src/engine/entity/Npc.js';
import { NpcStatMap } from '../../vendor/rs-sdk/server/engine/src/engine/entity/NpcStat.js';
import NpcType from '../../vendor/rs-sdk/server/engine/src/cache/config/NpcType.js';
import ServerTriggerType from '../../vendor/rs-sdk/server/engine/src/engine/script/ServerTriggerType.js';
import { monitorLifecycle } from './lifecycle';
import { resourceControl, resourceControlHttp } from '../core/resource-control';
import { installResourceCapture, readExistingResourceGround, type ResourceIdentity } from './resource-capture';
import { installCombatCapture } from './combat-capture';
import { actorCount } from './spectator-native';

let capture: ReturnType<typeof installResourceCapture>|null=null;
let combat: ReturnType<typeof installCombatCapture>|null=null;
let boot: string|null=null;
const owners=new WeakMap<Player,{identity:ResourceIdentity; socket:unknown; session:unknown}>();
const identities=new WeakMap<Player,{identity:ResourceIdentity; socket:unknown; session:unknown}>();
const epochs=new Map<string,number>();
/** Every item a column of a game table names (a woodcutting product, a rock's ore...). */
function tableItems(table:string,column:string):number[]{
    const type=DbTableType.getByName(table);
    const index=type?.columnNames.indexOf(column) ?? -1;
    if(!type || index<0)throw new Error(`game table ${table}:${column} unavailable`);
    return DbRowType.getInTable(type.id).map(row=>row.columnValues[index]?.[0]).filter((id):id is number=>Number.isSafeInteger(id) && (id as number)>=0);
}

/**
 * gathering-v1: everything woodcutting, mining and fishing produce, read from the game's
 * own data, each with its noted form and the game's price.
 * Fish are the items with a fishing_struct: raw fish only, so the starter kit's cooked
 * shrimps are not a target.
 */
const GATHERING_SCRIPTS=['[label,get_logs]','[label,get_ore_normal]','[label,get_ore_fast]',
    '[label,get_ore_essence]','[label,get_ore_gem_rock]','[proc,fish_roll]','[proc,fish_roll_loc]'];
function paramItems(param:string):number[]{
    const id=ParamType.getId(param);
    if(id<0)throw new Error(`item parameter ${param} unavailable`);
    return ObjType.configs.filter(o=>o?.params?.has(id)).map(o=>o.id);
}
function gatheringScope(){
    const scripts=new Set(GATHERING_SCRIPTS.map(name=>{
        const script=ScriptProvider.getByName(name);
        if(!script)throw new Error(`production script ${name} unavailable`);
        return script;
    }));
    const bases=[...new Set([...tableItems('woodcutting_trees','product'),...tableItems('mining_table','rock_output'),
        ...tableItems('gem_cutting_table','uncut_gem'),...paramItems('fishing_struct')])].sort((a,b)=>a-b);
    const targets=bases.map(base=>{
        const type=ObjType.get(base);
        if(!type || type.stackable)throw new Error(`gathered item ${base} unsupported`);
        return {base,note:type.certlink>=0?type.certlink:null,value:type.cost};
    });
    return {scripts,targets,ids:new Set(targets.flatMap(t=>t.note===null?[t.base]:[t.base,t.note]))};
}

const enabled=()=>process.env.RL_TEST_HOOKS==='1' && process.env.RL_RESOURCE_CAPTURE_EXPERIMENT==='1' &&
    !!process.env.RL_DEMO_BOOT_ID && (boot===null || process.env.RL_DEMO_BOOT_ID===boot);

export function installNativeResourceCapture(){
    if(!enabled())return;
    if(capture)throw new Error('native resource capture already installed');
    const script=ScriptProvider.getByName('[label,get_logs]');
    const logs=ObjType.getByName('logs');
    if(!script || !logs || logs.id!==1511 || logs.stackable || logs.certlink<0)
        throw new Error('native ordinary-log production/configuration unavailable');
    const bank=InvType.getId('bank');
    if(bank<0 || InvType.INV<0 || [bank,InvType.INV].some(id=>InvType.get(id)?.scope===InvType.SCOPE_SHARED))
        throw new Error('native private inventory/bank unavailable');
    const deathRoot=ScriptProvider.getByName('[queue,player_death]');
    const axes=['bronze_axe','iron_axe','steel_axe','black_axe','mithril_axe','adamant_axe','rune_axe'].map(name=>ObjType.getId(name));
    const pickaxes=['bronze_pickaxe','iron_pickaxe','steel_pickaxe','mithril_pickaxe','adamant_pickaxe','rune_pickaxe'].map(name=>ObjType.getId(name));
    if(!deathRoot || [...axes,...pickaxes].some(id=>id<0))throw new Error('resource lifecycle/tool definitions unavailable');
    const logsScope={ids:new Set([logs.id,logs.certlink]),tools:axes};
    // Built on first use, so a logs episode never depends on the wider tables.
    let gathering:(ReturnType<typeof gatheringScope> & {tools:number[]})|null=null;
    let scope:{ids:Set<number>; tools:number[]}=logsScope;
    boot=process.env.RL_DEMO_BOOT_ID!;
    capture=installResourceCapture({
        handlers:ScriptRunner.HANDLERS,inventoryPrototype:Inventory.prototype,world:World,
        slotSwap:{opcode:ScriptOpcode.INV_MOVETOSLOT,inventoryId:InvType.INV,bankId:bank,
            activePlayer:state=>state.activePlayer,container:(player,id)=>player.invs.get(id)},
        productionScript:script,addOpcode:ScriptOpcode.INV_ADD,tick:()=>World.currentTick,
        enabled,production:()=>Environment.node.production,actorCount,
        // getInventory() creates missing containers; preparation must materialize
        // them before arm so this observer can remain passive.
        inventories:p=>({inventory:p.invs.get(InvType.INV),bank:p.invs.get(bank)}),
        isTarget:id=>scope.ids.has(id),
        useScope(name){
            if(name==='logs'){scope=logsScope;return {targets:[],isProductionScript:s=>s===script};}
            if(name!=='gathering-v1')throw new Error(`unknown resource target scope ${name}`);
            const chosen=gathering ??= {...gatheringScope(),tools:[...axes,...pickaxes]};
            scope=chosen;
            return {targets:chosen.targets,isProductionScript:s=>chosen.scripts.has(s as any)};
        },
        groundAt:(level,x,z)=>readExistingResourceGround(World.gameMap,ZoneMap.zoneIndex(x,z,level),CoordGrid.packZoneCoord(x,z)),
        lifecycle:p=>monitorLifecycle(p,PlayerStat.HITPOINTS,()=>World.currentTick,deathRoot),
        toolAvailable:p=>[p.invs.get(InvType.INV),p.invs.get(InvType.WORN)].some(inv=>
            inv?.items.some((item:any)=>item && scope.tools.includes(item.id) && item.count>0)),
        validate(p:Player,identity){
            const owner=owners.get(p);
            if(!owner || JSON.stringify(identity)!==JSON.stringify(owner.identity) ||
                identity.worldBootId!==boot || World.getPlayerByUsername(p.username)!==p)
                throw new Error('native resource actor/account changed');
            if(p instanceof NetworkPlayer && (p.client!==owner.socket || p.session!==owner.session ||
                p.client.state===-1 || p.client.player!==p || p.loggingOut))
                throw new Error('native resource browser session changed');
        }
    });
    // The melee enemies of goal-v2 (baseline/resource_task.py MELEE_ENEMIES): chicken and chicken_brown.
    const enemies=['chicken','chicken_brown'].map(name=>NpcType.getId(name));
    const hitpoints=NpcStatMap.get('HITPOINTS');
    if(enemies.join()!=='41,1017' || hitpoints===undefined)throw new Error('melee enemy definitions changed');
    const agentDamage=ScriptProvider.getByName('[queue,combat_damage_player]')?.id;
    if(agentDamage===undefined)throw new Error('combat_damage_player script unavailable');
    const resources=capture;
    combat=installCombatCapture({handlers:ScriptRunner.HANDLERS,
        opcodes:{queue:ScriptOpcode.NPC_QUEUE,damage:ScriptOpcode.NPC_DAMAGE,playerQueue:ScriptOpcode.QUEUE},
        playerPrototype:Player.prototype,agentDamageScript:agentDamage,
        npcPrototype:Npc.prototype,runner:ScriptRunner,world:World,damageTrigger:ServerTriggerType.AI_QUEUE2,
        hitpoints,enemyTypes:enemies,tick:()=>World.currentTick,
        // Source-pinned private read (LinkList.sentinel): tail() would move the cursor processQueue iterates with.
        newestRequest:npc=>{const last=npc.queue?.sentinel?.prev;return last===npc.queue?.sentinel?null:last;},
        heroes(npc,player){
            let agent=0,other=0;
            for(const hero of npc.heroPoints as {hash64:bigint;points:number}[]){
                if(!hero || hero.hash64===-1n)continue;
                if(hero.hash64===player.hash64)agent+=hero.points;else other=Math.max(other,hero.points);
            }
            return {agent,other};
        },
        actor:(self,player)=>self===player?'agent':self instanceof Player?'player':self instanceof Npc?'npc':'world',
        agentHp:player=>player.levels[PlayerStat.HITPOINTS],
        open:player=>resources.open(player)});
}

/** The resource audit with the combat evidence of the same window (baseline/native_combat.py). */
function withCombat(player:Player,audit:any){
    return {...audit,combat:combat!.audit(player)};
}

export function armNativeResourceCapture(player:Player,identity:ResourceIdentity,horizon:number,targets='logs'){
    if(!capture || !enabled() || owners.has(player))throw new Error('resource capture unavailable/already bound');
    if(!(player instanceof Player) || identity.accountId!==player.username)
        throw new Error('resource actor name mismatch');
    owners.set(player,{identity:structuredClone(identity),socket:player instanceof NetworkPlayer?player.client:null,
        session:player instanceof NetworkPlayer?player.session:null});
    try{capture.bind(player,identity,horizon,targets);}catch(error){owners.delete(player);throw error;}
    const armed=capture.audit(player);
    combat!.bind(player,identity,armed.startTick,armed.endTick);
    return withCombat(player,armed);
}

export function readNativeResourceCapture(player:Player){
    if(!capture || !enabled())throw new Error('resource capture unavailable');
    return withCombat(player,capture.audit(player));
}

export function nativeResourceBudget(player:Player){
    if(!capture || !enabled())throw new Error('resource capture unavailable');
    return capture.budget(player);
}

export function retireNativeResourceCapture(player:Player){
    if(!capture || !enabled())throw new Error('resource capture unavailable');
    capture.retire(player);
    return withCombat(player,capture.audit(player));
}

export function releaseNativeResourceCapture(player:Player){
    if(!capture || !enabled())throw new Error('resource capture unavailable');
    const receipt=capture.release(player);
    owners.delete(player);
    return {...receipt,combat:combat!.release(player)};
}

/** Server-generated generation identity; callers cannot choose/reset its epoch. */
export function nativeResourceIdentity(player:Player):ResourceIdentity{
    if(!capture || !enabled() || World.getPlayerByUsername(player.username)!==player || player.loggingOut)
        throw new Error('resource identity requires current native actor');
    let entry=identities.get(player);
    if(!entry){
        const epoch=(epochs.get(player.username)??0)+1;
        if(!Number.isSafeInteger(epoch))throw new Error('resource epoch overflow');
        entry={identity:{worldBootId:boot!,accountId:player.username,accountEpoch:epoch,incarnationId:crypto.randomUUID()},
            socket:player instanceof NetworkPlayer?player.client:null,session:player instanceof NetworkPlayer?player.session:null};
        identities.set(player,entry);epochs.set(player.username,epoch);
    }
    if(player instanceof NetworkPlayer && (player.client!==entry.socket || player.session!==entry.session ||
        player.client.state===-1 || player.client.player!==player))throw new Error('resource identity session changed');
    return structuredClone(entry.identity);
}

export function nativeResourceControl(){
    return resourceControl<Player>({bootId:process.env.RL_DEMO_BOOT_ID??'',enabled:()=>!!capture&&enabled(),
        production:()=>Environment.node.production,actorCount,
        findPlayer:name=>World.getPlayerByUsername(name)??null,identity:nativeResourceIdentity,
        arm:armNativeResourceCapture,read:readNativeResourceCapture,budget:nativeResourceBudget,
        retire:retireNativeResourceCapture,release:releaseNativeResourceCapture});
}
export function nativeResourceHttp(){return resourceControlHttp(nativeResourceControl());}
