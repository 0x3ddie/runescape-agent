from copy import deepcopy
import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO

from baseline.adapter import GameVecEnv
from baseline.policy import MaskedRecurrentPolicy
from obs import MASK, OBSERVATION
from baseline.resource_task import ResourceEpisode,ResourceBackend,VALUE_SCORING
from baseline.evidence import EvidenceError
from act_contract import CLICK_OFFSETS
from feedback_contract import quiet
from baseline.tests.test_native_flow import Trace,drop,item,pickup,trace
from baseline.config import VILLAGES
from names import name_table

CFG=dict(VILLAGES['draynor'],contract='item-v2',actTargets='act-v3')
# One coin a log keeps the certified counts readable as rewards: 1 log = .1 at scale 10.
LOGS=[dict(base=1511,note=1512,value=1)]
ARMED=('schema','flowEvidenceVersion','slotSwapEvidenceVersion','ownedContainers','targetScope','targets',
       'identity','startTick','endTick','failure','lifecycleObserved')


def raw():
    empty=dict(slot=-1,id=-1,count=0,name='');mask=[0]*96;mask[0]=1
    return dict(contractVersion='item-v2',actTargetVersion='act-v3',actSel=[],heldSel=[],
                feedback=quiet(),clicks=[dict(dx=dx,dz=dz,inside=False,open=True) for dx,dz in CLICK_OFFSETS],
                invPage=0,bankPage=0,bankPages=1,bankOpen=False,invSel=[dict(empty) for _ in range(8)],
                bankSel=[dict(empty) for _ in range(8)],player=dict(x=100,z=100,level=0),
                inventory=[],mask=mask,maskVersion=MASK,done=False,truncated=False,
                skills=[dict(name='Woodcutting',xp=99999,level=99)])


def capture(scenario,epoch,death):
    """banked: two logs kept, one deposited. own: a log dropped and picked up again, exact.
    mixed: a log and an outside one share a stack and one is lost, so 0 or 1 log is kept."""
    t=Trace();t.produce()
    if scenario=='banked':
        t.op(inventory=[],bank=[item()]);t.produce()
    elif scenario=='own' or (scenario=='alternate' and epoch%2):
        pickup(t,drop(t))
    else:
        t.external(slot=1)
        t.op(inventory=[],bank=[item(count=2)])
        t.op(inventory=[item()],bank=[item()])
        drop(t)
    result=trace(t,death)
    result.update(targetScope='gathering-v1',targets=deepcopy(LOGS))
    return result


class World:
    def __init__(self,scenario='banked',death=False):
        self.scenario=scenario;self.death=death
        self.epoch=0;self.calls=0;self.closed=False;self.released=0;self.retired=0
        self.state='active';self.tick=10;self.records=[]
    def reset(self,seed,cfg):
        self.epoch+=1;self.calls=0;self.seed=seed;self.state='active';self.tick=10
        self.receipt=capture(self.scenario,self.epoch,self.death)
        self.receipt['identity'].update(accountEpoch=self.epoch,incarnationId=str(self.epoch))
        return raw()
    def step(self,action):
        self.calls+=1
        if self.death:self.state='death'
        else:
            self.tick+=1
            if self.tick==12:self.state='horizon'
        return raw()
    def close(self):self.closed=True
    def control(self,operation,**kwargs):
        identity=self.receipt['identity']
        if operation=='identity':return dict(type='resource_identity',identity=deepcopy(identity))
        assert kwargs['identity']==identity
        if operation=='arm':
            assert kwargs['targets']=='gathering-v1'
            audit={k:deepcopy(self.receipt[k]) for k in ARMED}
            return dict(type='resource_capture',identity=deepcopy(identity),audit=dict(audit,closed=False))
        if operation=='budget':return dict(type='resource_budget',identity=deepcopy(identity),budget=dict(
            schema='resource-budget-v1',startTick=10,endTick=12,observedTick=self.tick,horizon=2,
            remainingTicks=12-self.tick,status=self.state))
        if operation=='release':self.released+=1
        if operation=='retire':self.retired+=1
        return dict(type='resource_capture',identity=deepcopy(identity),audit=deepcopy(self.receipt))
    def record(self,kind,**data):self.records.append(dict(kind=kind,**deepcopy(data)))


def episode(world):
    return ResourceEpisode(world,world.control,CFG,2,10,world.record)


def small_model(env,seed,**kwargs):
    return RecurrentPPO(MaskedRecurrentPolicy,env,n_steps=4,batch_size=4,n_epochs=1,gamma=1,seed=seed,device='cpu',**kwargs,
                        policy_kwargs=dict(lstm_hidden_size=16,names=name_table().spec, observation=OBSERVATION,
                                           features_extractor_kwargs=dict(features_dim=16),net_arch=dict(pi=[16],vf=[16])))


def test_only_terminal_retained_production_pays_and_time_feature_uses_native_ticks():
    w=World();e=episode(w);initial=e.reset(0)
    np.testing.assert_allclose(initial['budget_vec'],[1,.0002])
    obs,r,done,info=e.step(0)
    assert r==0 and not done and info=={}
    np.testing.assert_allclose(obs['budget_vec'],[.5,.0002])
    obs,r,done,info=e.step(0)
    assert r==.2 and done and w.released==1
    assert info['resource_report']['deposit_events']==1
    assert info['resource_report']['retained_lower']==2
    assert obs['budget_vec'][0]==0
    with pytest.raises(EvidenceError):e.step(0)


@pytest.mark.parametrize('death,total',[(False,.2),(True,0.)])
def test_rewards_paid_as_held_value_changes_sum_to_the_certified_total(death,total):
    w=World(death=death);e=episode(w);e.reset(0);held=iter((3,1))
    def step(action):
        w.calls+=1
        if death:w.state='death' if w.calls==2 else 'active'  # the fake trace dies at tick 10
        else:
            w.tick+=1
            if w.tick==12:w.state='horizon'
        return dict(raw(),inventory=[dict(slot=k,id=1511,count=1,name='Logs') for k in range(next(held))])
    w.step=step
    _,first,done,_=e.step(0)
    assert first==pytest.approx(.3) and not done  # three logs carried
    _,last,done,info=e.step(0)
    assert done and info['reward_bounds'][0]==total and first+last==pytest.approx(total)


def test_native_death_zeroes_positive_holdings_and_stops_first_life():
    w=World(death=True);e=episode(w);e.reset(0)
    _,r,done,info=e.step(0)
    assert done and r==0 and info['resource_report']['retained_lower']==2
    assert not info['resource_report']['survived']


def test_valid_ambiguous_terminal_pays_its_lower_bound_and_resets():
    w=World('mixed');e=episode(w);initial=e.reset(0)
    obs,reward,done,_=e.step(0)
    assert reward==0 and not done
    obs,reward,done,info=e.step(0)
    assert reward==0 and done and info['reward_bounds']==[0,.1]
    assert info['scoring']==VALUE_SCORING and not info['resource_report']['provenance_exact']
    assert not e.failed and w.released==1
    assert set(obs)==set(initial) and 'reward_bounds' not in obs and 'resource_report' not in obs
    terminal=next(r for r in w.records if r['kind']=='task_terminal')
    assert terminal['reward_upper']==.1 and terminal['report']['retained_upper']==1
    e.reset(1)
    assert w.epoch==2


def test_ambiguous_death_preserves_diagnostics_and_zeroes_both_reward_bounds():
    w=World('mixed',death=True);e=episode(w);e.reset(0)
    _,reward,done,info=e.step(0)
    assert done and reward==0 and info['reward_bounds']==[0,0]
    assert info['resource_report']['retained_upper']==1 and not info['resource_report']['survived']


def test_horizon_during_inference_does_not_dispatch_late_action():
    w=World();e=episode(w);e.reset(0);w.tick=12;w.state='horizon'
    _,r,done,_=e.step(0)
    assert done and r==.2 and w.calls==0
    assert next(r for r in w.records if r['kind']=='task_transition')['executed_action'] is None


@pytest.mark.parametrize('fault',['identity','regression','remaining','horizon','status'])
def test_clock_disagreement_aborts_and_poisoned_episode_cannot_resume(fault):
    w=World();e=episode(w);e.reset(0);original=w.control
    def corrupt(op,**kwargs):
        reply=original(op,**kwargs)
        if op=='budget':
            if fault=='identity':reply['identity']['incarnationId']='other'
            else:reply['budget'].update({'regression':{'observedTick':9},'remaining':{'remainingTicks':0},
                'horizon':{'horizon':3},'status':{'status':'horizon'}}[fault])
        return reply
    e.control=corrupt
    with pytest.raises(EvidenceError):e.step(0)
    assert e.failed and w.calls==0
    with pytest.raises(EvidenceError):e.reset(1)
    e.close();assert w.closed and w.retired==1


def test_lost_arm_reply_is_retired_and_transport_closed():
    w=World();e=episode(w);original=w.control
    def lost(op,**kwargs):
        result=original(op,**kwargs)
        if op=='arm':raise TimeoutError('lost arm response')
        return result
    e.control=lost
    with pytest.raises(TimeoutError):e.reset(0)
    e.close();assert w.retired==1 and w.closed


def test_arm_without_flow_coverage_is_refused_before_collection():
    w=World();e=episode(w);original=w.control
    def uncovered(op,**kwargs):
        reply=original(op,**kwargs)
        if op=='arm':del reply['audit']['flowEvidenceVersion']
        return reply
    e.control=uncovered
    with pytest.raises(EvidenceError,match='coverage at arm'):e.reset(0)
    assert w.calls==0 and e.failed
    e.close();assert w.retired==1


def test_a_task_without_act_v3_is_refused_before_world_reset():
    w=World()
    with pytest.raises(ValueError,match='act-v3'):
        ResourceEpisode(w,w.control,dict(CFG,actTargets='gather-v1'),2,10,w.record)
    assert w.epoch==0


def test_invalid_trace_and_transport_timeout_are_not_zero_reward_episodes():
    w=World();e=episode(w);e.reset(0);w.receipt['failure']='missing tick'
    e.step(0)
    with pytest.raises(EvidenceError):e.step(0)
    assert e.failed and w.released==0 and not any(r['kind']=='task_terminal' for r in w.records)
    e.close();assert w.retired==1 and w.closed
    w=World();e=episode(w);e.reset(0)
    w.step=lambda action:dict(raw(),done=True,truncated=True)
    with pytest.raises(EvidenceError,match='before native'):e.step(0)


def test_auto_reset_preserves_task_terminal_state_and_disables_timeout_bootstrap():
    w=World();backend=ResourceBackend(episode(w));first=backend.reset(7)
    adapter=GameVecEnv(backend,{k:v[0] for k,v in first.items()})
    # Adapter normally owns reset; inject its validated initial state for this boundary check.
    adapter._obs=adapter._validate(first)
    adapter.step_async(np.array([0]));adapter.step_wait()
    adapter.step_async(np.array([0]));obs,r,dones,infos=adapter.step_wait()
    assert dones.tolist()==[True] and w.seed==8 and w.epoch==2
    assert infos[0]['TimeLimit.truncated'] is False
    assert infos[0]['terminal_observation']['budget_vec'][0]==0
    assert obs['budget_vec'][0,0]==1
    assert infos[0]['episode']['native_ticks']==2


def test_upstream_collector_trains_through_terminals_and_keeps_ambiguous_episodes(tmp_path):
    torch.set_num_threads(1)
    sample=episode(World('own')).reset(0)
    w=World('alternate');env=GameVecEnv(ResourceBackend(episode(w)),sample)
    model=small_model(env,19)
    before={k:v.clone() for k,v in model.policy.state_dict().items()}
    model.learn(total_timesteps=8)
    terminals=[r for r in w.records if r['kind']=='task_terminal']
    assert model.num_timesteps==8 and w.released==4
    assert [r['report']['provenance_exact'] for r in terminals]==[True,False,True,False]
    assert [r['reward'] for r in terminals]==[.1,0.,.1,0.]
    assert any(not torch.equal(before[k],v) for k,v in model.policy.state_dict().items())
    model.save(tmp_path/'resource-policy')
    assert RecurrentPPO.load(tmp_path/'resource-policy',device='cpu').gamma==1
    env.close();assert w.closed


def test_same_incarnation_after_reset_is_rejected_even_if_epoch_increases():
    w=World();e=episode(w);e.reset(0);e.step(0);e.step(0)
    original=w.reset
    def reuse(seed,cfg):
        obs=original(seed,cfg);w.receipt['identity']['incarnationId']='1';return obs
    w.reset=reuse
    with pytest.raises(EvidenceError,match='fresh incarnation'):e.reset(1)


def test_closed_clock_cannot_advance_or_reopen():
    w=World();e=episode(w);e.reset(0);e.step(0);e.step(0)
    reply=w.control('budget',identity=e.clock.identity)
    reply['budget'].update(status='active',observedTick=11,remainingTicks=1)
    with pytest.raises(EvidenceError):e.clock.accept(reply)


def test_critic_input_counts_logs_banked_this_episode():
    e=episode(World());logs=lambda n:[dict(slot=k,id=1511,count=1,name='Logs') for k in range(n)]
    e.prices={1511:1,1512:1};e.carried=5
    # deposit 3, close, chop 2 more, reopen, withdraw 1, close
    for inventory,bank_open,banked in ((logs(5),True,0),(logs(2),True,3),(logs(2),False,3),(logs(4),False,3),
                                       (logs(4),True,3),(logs(5),True,2),(logs(5),False,2)):
        e.raw=dict(inventory=inventory,bankOpen=bank_open);e._track_bank()
        assert e.banked==banked  # deposits and withdrawals count only through an open bank
