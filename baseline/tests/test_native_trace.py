from copy import deepcopy
import pytest
from baseline.native_trace import audit_native_trace
from baseline.evidence import EvidenceError


def trace(death=False,external=False):
    initial={'inventory':[],'bank':[]};events=[];previous=initial
    def op(after,production=False):
        nonlocal previous
        events.append(dict(kind='native-op',tick=10,sequence=len(events)+1,before=previous,after=after,
            opcode=4302 if production else 4320,scriptId=51,scriptName='[label,get_logs]' if production else '[proc,bank_deposit_request]',
            pc=5,productionSource=production,stackTail=[1,1511,1],ground=[]))
        previous=after
    item={'slot':0,'id':1511,'count':1}
    op({'inventory':[item],'bank':[]},True)
    op({'inventory':[],'bank':[item]})
    op({'inventory':[item],'bank':[item]},True)
    if external:op({'inventory':[dict(item,count=2)],'bank':[item]})
    life=dict(phase='dying' if death else 'alive',deathSequence=int(death),respawnSequence=0,
        deathTick=10 if death else None,respawnTick=None,completionSource=None,failure=None)
    if not death:
        for tick in (11,12):events.append(dict(kind='tick-boundary',tick=tick,sequence=len(events)+1,holdings=previous))
    terminal=dict(reason='death' if death else 'horizon',tick=10 if death else 12,holdings=previous,toolAvailable=True,lifecycle=life)
    events.append(dict(kind='terminal',sequence=len(events)+1,**terminal))
    return dict(schema='native-resource-capture-v2',identity=dict(worldBootId='boot',accountId='a',accountEpoch=1,incarnationId='p'),
        initial=initial,last=previous,startTick=10,endTick=12,observedTick=terminal['tick'],events=events,
        failure=None,closed=True,evidenceComplete=True,lifecycleObserved=True,completed=not death,terminal=terminal)


def test_native_chain_replay_has_exact_bounds_without_bank_bonus():
    r=audit_native_trace(trace())
    assert r['produced']==r['retained_lower']==r['retained_upper']==2
    assert r['deposit_events']==r['production_after_deposit']==1
    assert r['survived'] and r['provenance_exact']


def test_death_cannot_be_hidden_by_positive_resource_holdings():
    r=audit_native_trace(trace(death=True))
    assert r['produced']==2 and r['retained_lower']==2
    assert not r['survived'] and r['reward_eligible_lower']==0


def test_external_items_do_not_count_as_production():
    r=audit_native_trace(trace(external=True))
    assert r['total_holdings']==3 and r['produced']==r['retained_lower']==r['retained_upper']==2


@pytest.mark.parametrize('fault',['gap','state','source','opcode','args','clock','tail','lifecycle','fault','preload','incomplete'])
def test_reject_untrustworthy_trace(fault):
    t=deepcopy(trace())
    if fault=='gap':t['events'][1]['sequence']=3
    if fault=='state':t['events'][1]['before']={'inventory':[],'bank':[]}
    if fault=='source':t['events'][0]['scriptName']='[proc,gift]'
    if fault=='opcode':t['events'][0]['opcode']=4303
    if fault=='args':t['events'][0]['stackTail']=[1,1511,2]
    if fault=='clock':t['events'][3]['tick']=12
    if fault=='tail':t['events'].pop()
    if fault=='lifecycle':t['lifecycleObserved']=False
    if fault=='fault':t['failure']='overflow'
    if fault=='preload':t['initial']={'inventory':[{'slot':0,'id':1511,'count':1}],'bank':[]}
    if fault=='incomplete':t['closed']=False
    with pytest.raises(EvidenceError):audit_native_trace(t)


def test_uncertain_losses_produce_bounds_instead_of_invented_lineage():
    t=trace(external=True);prior=t['last'];after={'inventory':[{'slot':0,'id':1511,'count':1}],'bank':[]}
    event=dict(kind='native-op',tick=10,before=prior,after=after,opcode=4303,scriptId=90,scriptName='[proc,loss]',
        pc=1,productionSource=False,stackTail=[],ground=[])
    t['events'].insert(4,event)
    for seq,row in enumerate(t['events'],1):
        row['sequence']=seq
        if row['kind'] in ('tick-boundary','terminal'):row['holdings']=after
    t['last']=after;t['terminal']['holdings']=after
    r=audit_native_trace(t)
    assert (r['retained_lower'],r['retained_upper'])==(0,1) and not r['provenance_exact']
