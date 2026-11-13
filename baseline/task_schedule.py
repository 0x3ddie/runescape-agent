"""Registered episode starts for one upstream vector environment.

The schedule selects task conditions only between lives, never policy actions.
Native clocks, scoring and lifecycle validation remain in ResourceEpisode.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import numpy as np

from baseline.resource_task import ResourceBackend, need


class ScheduledResourceBackend:
    n_envs = 1

    def __init__(self, fixtures, make_episode, record, close_transport, shuffle=None):
        need(bool(fixtures), 'empty task schedule')
        self.fixtures=deepcopy(fixtures)
        # A numpy Generator: each pass over the fixtures goes in its own random order, so goals
        # stay balanced (each pass has each once) and their order is not predictable.
        self.shuffle=shuffle;self.order=list(range(len(fixtures)))
        self.make_episode=make_episode
        self.record=record
        self.close_transport=close_transport
        self.episode=None
        self.index=0
        self.closed=False
        self.failed=False

    def _next(self):
        need(not self.closed and not self.failed, 'closed/failed task schedule')
        previous=None
        if self.episode is not None:
            need(not self.episode.active and not self.episode.bound, 'cannot replace active task')
            previous=self.episode.previous_identity
        if self.shuffle is not None and self.index % len(self.fixtures)==0:
            self.order=[int(k) for k in self.shuffle.permutation(len(self.fixtures))]
        fixture=deepcopy(self.fixtures[self.order[self.index % len(self.fixtures)]])
        self.record('scheduled_episode',index=self.index,fixture=fixture)
        self.episode=self.make_episode(fixture)
        self.episode.previous_identity=deepcopy(previous)
        self.index+=1
        return self.episode.reset(fixture['reset_seed'])

    def reset(self, seed=None):
        # SB3's algorithm seed does not override the configured environment
        # fixtures or pretend to seed the native game's RNG.
        need(self.index==0, 'upstream reset would discard a scheduled episode')
        self.record('algorithm_reset_seed',seed=seed)
        try:
            return ResourceBackend.batch(self._next())
        except Exception:
            self.failed=True
            raise

    def step(self, actions):
        need(self.episode is not None and not self.closed and not self.failed, 'invalid schedule step')
        actions=np.asarray(actions)
        need(actions.shape==(1,) and np.issubdtype(actions.dtype,np.integer), 'invalid resource action batch')
        try:
            obs,reward,done,info=self.episode.step(int(actions[0]))
            if done:
                final,final_info=deepcopy(obs),deepcopy(info)
                obs=self._next()
                info=dict(final_observation=final,final_info=final_info,episode=final_info['episode'])
            return ResourceBackend.batch(obs),np.asarray([reward],np.float32),np.asarray([done],bool),np.asarray([False],bool),[info]
        except Exception:
            self.failed=True
            raise

    def close(self):
        if self.closed:return
        try:
            if self.episode is not None:
                if self.episode.bound:
                    # A fixed transition budget may stop inside a life. Preserve
                    # its capture, but do not manufacture a terminal training
                    # reward or include it in completed evaluation denominators.
                    try:
                        reply=self.episode._control('read',identity=self.episode.bound_identity)
                        self.record('collection_tail',schedule_index=self.index-1,capture=reply['audit'],
                            scored=False,reason='collection stopped before wrapper terminal')
                    finally:self.episode.close()
                else:self.episode.close()
            else:self.close_transport()
        finally:self.closed=True


class FreshWorldSchedule:
    """One training slot whose every episode runs in a world of its own (scripts/stack.py world service), so no
    episode starts in a world an earlier one changed. choose(k) gives the slot's k-th fixture;
    open_episode(world, fixture, record) joins the world and returns its ResourceEpisode. While an episode runs,
    the next one is prepared in the background: its world acquired and joined, its native reset and scripted
    setup done (ResourceEpisode.prepare_start), its rows held back; a lockstep world does not tick while it
    waits. At the boundary the finished episode's connection closes, its world is released (stopped), and the
    prepared one's rows are written and its task clock armed (start). The rewards paid in an episode must sum to
    its certified return."""
    n_envs = 1

    def __init__(self, choose, worlds, engine, open_episode, record):
        self.choose=choose;self.worlds=worlds;self.engine=engine;self.open_episode=open_episode;self.record=record
        self.episode=None;self.world=None;self.index=0;self.paid=0.;self.closed=False;self.failed=False
        self.pool=ThreadPoolExecutor(max_workers=1);self.prepared=None

    def _prepare(self, index):
        fixture=self.choose(index);rows=[]
        world=self.worlds.acquire(self.engine)
        try:
            episode=self.open_episode(world,fixture,lambda kind,**data:rows.append((kind,data)))
            try:episode.prepare_start(fixture['reset_seed'])
            except BaseException:
                episode.close()
                raise
        except BaseException:
            self.worlds.release(world['id'])
            raise
        return fixture,world,episode,rows

    def _retire(self):
        episode,world=self.episode,self.world
        self.episode=self.world=None
        try:
            if episode is not None and episode.bound:
                # A fixed transition budget may stop inside a life: keep its capture, unscored.
                try:
                    reply=episode._control('read',identity=episode.bound_identity)
                    self.record('collection_tail',schedule_index=self.index-1,capture=reply['audit'],
                                scored=False,reason='collection stopped before wrapper terminal')
                finally:episode.close()
            elif episode is not None:episode.close()
        finally:
            if world is not None:self.worlds.release(world['id'])

    def _next(self):
        need(not self.closed and not self.failed,'closed/failed task schedule')
        self._retire()
        pending=self.prepared or self.pool.submit(self._prepare,self.index)
        self.prepared=None
        fixture,self.world,self.episode,rows=pending.result()
        self.record('scheduled_episode',index=self.index,fixture=fixture,
                    world={k:self.world[k] for k in ('id','engine','tree','block','boot_id')})
        for kind,data in rows:self.record(kind,**data)
        self.episode.record=self.record
        self.index+=1
        self.paid=0.
        obs=self.episode.start()
        self.prepared=self.pool.submit(self._prepare,self.index)
        return obs

    def reset(self, seed=None):
        need(self.index==0,'upstream reset would discard a scheduled episode')
        self.record('algorithm_reset_seed',seed=seed)
        try:
            return ResourceBackend.batch(self._next())
        except Exception:
            self.failed=True
            raise

    def step(self, actions):
        need(self.episode is not None and not self.closed and not self.failed,'invalid schedule step')
        actions=np.asarray(actions)
        need(actions.shape==(1,) and np.issubdtype(actions.dtype,np.integer),'invalid resource action batch')
        try:
            obs,reward,done,info=self.episode.step(int(actions[0]))
            self.paid+=reward
            if done:
                need(abs(self.paid-info['reward_bounds'][0])<=1e-6,'paid rewards disagree with the certified total')
                final,final_info=deepcopy(obs),deepcopy(info)
                obs=self._next()
                info=dict(final_observation=final,final_info=final_info,episode=final_info['episode'])
            return ResourceBackend.batch(obs),np.asarray([reward],np.float32),np.asarray([done],bool),np.asarray([False],bool),[info]
        except Exception:
            self.failed=True
            raise

    def close(self):
        if self.closed:return
        self.closed=True
        try:self._retire()
        finally:
            pending,self.prepared=self.prepared,None
            try:
                if pending is not None:
                    try:_,world,episode,_=pending.result()
                    except Exception:world=None  # its own failure already released what it held
                    if world is not None:
                        try:episode.close()
                        finally:self.worlds.release(world['id'])
            finally:self.pool.shutdown(wait=False)


class ParallelSchedules:
    """Several single-agent schedules stepped concurrently as one vector env.

    Each agent's step is mostly waiting on its own game connection, so one
    thread per agent overlaps them. Any agent's failure closes every agent.
    """
    def __init__(self, schedules):
        need(bool(schedules), 'no agent schedules')
        self.schedules=list(schedules)
        self.n_envs=len(self.schedules)
        self.pool=ThreadPoolExecutor(max_workers=self.n_envs)
        self.closed=False

    def _all(self, calls):
        futures=[self.pool.submit(call) for call in calls]
        try:
            return [future.result() for future in futures]
        except Exception:
            self.close()
            raise

    def reset(self, seed=None):
        results=self._all([lambda s=s:s.reset(seed=seed) for s in self.schedules])
        return {k:np.concatenate([r[k] for r in results]) for k in results[0]}

    def step(self, actions):
        actions=np.asarray(actions)
        need(actions.shape==(self.n_envs,), 'invalid parallel action batch')
        results=self._all([lambda s=s,i=i:s.step(actions[i:i+1]) for i,s in enumerate(self.schedules)])
        obs={k:np.concatenate([r[0][k] for r in results]) for k in results[0][0]}
        rewards,terminated,truncated=(np.concatenate([r[j] for r in results]) for j in (1,2,3))
        return obs,rewards,terminated,truncated,[info for r in results for info in r[4]]

    def close(self):
        if self.closed:return
        self.closed=True
        errors=[]
        for schedule in self.schedules:
            try:schedule.close()
            except Exception as error:errors.append(error)
        self.pool.shutdown(wait=False)
        if errors:raise errors[0]
