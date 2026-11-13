import {expect,test} from 'bun:test';
import {resourceControl,resourceControlHttp} from './resource-control';

function fixture(){
    const player={},identity={worldBootId:'boot',accountId:'actor',accountEpoch:1,incarnationId:'p1'};
    const state={enabled:true,production:false,count:1,calls:[] as string[]};
    const host={bootId:'boot',enabled:()=>state.enabled,production:()=>state.production,actorCount:()=>state.count,
        findPlayer:(name:string)=>name==='actor'?player:null,identity:()=>identity,
        arm:(_p:any,_i:any,h:number)=>{state.calls.push(`arm:${h}`);return {armed:true};},
        read:()=>{state.calls.push('read');return {raw:true};},
        budget:()=>{state.calls.push('budget');return {schema:'resource-budget-v1',horizon:100,remainingTicks:50};},
        retire:()=>{state.calls.push('retire');return {closed:true};},
        release:()=>{state.calls.push('release');return {archived:true};}};
    const control=resourceControl(host),base={operation:'identity',bootId:'boot',accountName:'actor'};
    return {state,control,base,identity,http:resourceControlHttp(control)};
}
test('identity and exact arm/read/archive protocol work without choosing actions',()=>{
    const f=fixture();expect(f.control(f.base).identity).toEqual(f.identity);
    f.control({...f.base,operation:'arm',identity:{...f.identity},horizon:100});
    for(const operation of ['read','retire','release'])f.control({...f.base,operation,identity:f.identity});
    expect(f.state.calls).toEqual(['arm:100','read','retire','release']);
});
test('budget route returns only clock fields with the same incarnation guard',()=>{
    const f=fixture();
    const result=f.control({...f.base,operation:'budget',identity:f.identity});
    expect(result.type).toBe('resource_budget');expect(result).not.toHaveProperty('audit');
    expect(result.budget.remainingTicks).toBe(50);expect(f.state.calls).toEqual(['budget']);
    expect(()=>f.control({...f.base,operation:'budget',identity:{...f.identity,accountEpoch:2}})).toThrow();
});
for(const override of [{bootId:'other'},{accountName:'other'},{horizon:0},{horizon:10001},
    {horizon:1.5},{operation:'teleport'},{extra:1},{identity:{worldBootId:'boot',accountId:'actor',accountEpoch:1,incarnationId:'old'}}]){
    test(`reject invalid resource control ${JSON.stringify(override)}`,()=>{
        const f=fixture();expect(()=>f.control({...f.base,operation:'arm',identity:f.identity,horizon:100,...override})).toThrow();
        expect(f.state.calls).toEqual([]);
    });
}
for(const change of [{enabled:false},{production:true},{count:0},{count:2}])test(`owned resource guard ${JSON.stringify(change)}`,()=>{
    const f=fixture();Object.assign(f.state,change);expect(()=>f.control(f.base)).toThrow();expect(f.state.calls).toEqual([]);
});
test('HTTP endpoint rejects browser Origin, extra query, oversized and malformed bodies',async()=>{
    const f=fixture();
    for(const [body,extra,url] of [[JSON.stringify(f.base),{origin:'http://localhost'},''],
        [JSON.stringify(f.base),{},'?extra=1'],[' '.repeat(4097),{},''],['{',{},'']] as const){
        const response=await f.http(new Request(`http://127.0.0.1/resource-capture${url}`,{
            method:'POST',headers:{'Content-Type':'application/json',...extra},body}));
        expect(response.status).toBeGreaterThanOrEqual(400);
    }
    const response=await f.http(new Request('http://127.0.0.1/resource-capture',{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(f.base)}));
    expect(response.status).toBe(200);expect((await response.json()).identity).toEqual(f.identity);
});
