from copy import deepcopy
import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO

from baseline.adapter import GameVecEnv
from baseline.resource_task import ResourceEpisode
from baseline.task_schedule import ScheduledResourceBackend
from baseline.tests.test_resource_task import CFG, World, raw, small_model


def backend(world):
    def make(fixture):
        return ResourceEpisode(world,world.control,CFG,2,10,world.record)
    return ScheduledResourceBackend([dict(id='first',reset_seed=5),dict(id='second',reset_seed=9)],make,world.record,world.close)


def test_schedule_preserves_final_observation_and_registered_seeds():
    w=World('alternate');b=backend(w);b.reset(seed=1234)
    assert w.seed==5
    b.step(np.asarray([0]));obs,reward,done,truncated,infos=b.step(np.asarray([0]))
    assert done.tolist()==[True] and truncated.tolist()==[False] and w.seed==9
    assert infos[0]['final_info']['reward_bounds']==[.1,.1]
    assert infos[0]['final_observation']['budget_vec'][0]==0 and obs['budget_vec'][0,0]==1
    b.step(np.asarray([0]));_,reward,done,_,infos=b.step(np.asarray([0]))
    assert done[0] and reward[0]==0 and infos[0]['final_info']['reward_bounds']==[0,.1]
    assert w.seed==5 and w.epoch==3
    with pytest.raises(Exception,match='discard'):b.reset()
    b.close(); assert w.retired==1 and w.closed
    tails=[r for r in w.records if r['kind']=='collection_tail']
    assert len(tails)==1 and tails[0]['scored'] is False


def test_actual_upstream_collector_optimizes_schedule_and_preserves_unscored_tail(tmp_path):
    torch.set_num_threads(1)
    w=World('alternate');b=backend(w)
    from obs import project
    sample=project(raw())
    sample.update(budget_vec=np.zeros(2,np.float32),critic_vec=np.zeros(1,np.float32))
    env=GameVecEnv(b,sample)
    model=small_model(env,17,gae_lambda=1.)
    before=deepcopy(model.policy.state_dict());model.learn(total_timesteps=8)
    assert model.num_timesteps==8 and w.released==4 and w.epoch==5
    assert any(not torch.equal(before[k],v) for k,v in model.policy.state_dict().items())
    model.save(tmp_path/'latest');loaded=RecurrentPPO.load(tmp_path/'latest',device='cpu')
    assert loaded.gamma==loaded.gae_lambda==1
    env.close()
    assert w.retired==1 and len([r for r in w.records if r['kind']=='task_terminal'])==4


def test_close_before_reset_closes_transport_without_creating_a_life():
    w=World();b=backend(w);b.close();b.close()
    assert w.closed and w.epoch==0
