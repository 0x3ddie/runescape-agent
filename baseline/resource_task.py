"""Finite native-time resource episodes for the upstream learner.

Transport-independent task semantics. No action planner or private PPO code.
The native evaluator owns the clock and terminal resource evidence. Ordinary
reset setup, if supplied, finishes before arm and is excluded from the rollout.
"""
from copy import deepcopy
import math
import numpy as np

from baseline.native_combat import SCHEMA as COMBAT_SCHEMA, audit_native_combat
from baseline.native_slot_flow import owned_containers
from baseline.native_value import GATHERING, audit_native_value
from baseline.evidence import EvidenceError
from obs import project
from act_contract import acts_from_config
from feedback_contract import episode_start

# Every certified gathered item, weighted by the game's price (baseline/native_value.py).
VALUE_SCORING = 'native-gathering-value-v1'
# goal-v2: the episode asks for one of four activities.
# Gathering goals return min(certified newly produced items of the goal's families retained, 2) / 2;
# melee returns min(certified eligible kills attributable to this agent, 1). Zero after a death.
# Prices, other activities, XP and banking add nothing. The policy and critic see goal_vec.
GOAL_SCORING = 'goal-activity-v2'
GOALS = ('wood', 'fish', 'mine', 'melee')  # the activity tasks' goals; world slot b asks for GOALS[b % 4]
# goal-v3: goal_vec has six entries in this order, GOALS first, so a goal's entry is its
# index in GOALS. Cook and fire have no task yet (fire-v1 and cook-v1 come with the port). goal-v2 had GOALS' four.
GOAL_VEC = GOALS + ('cook', 'fire')
GOAL_ITEMS = 2
GOAL_TARGETS = dict(wood=GOAL_ITEMS, fish=GOAL_ITEMS, mine=GOAL_ITEMS, melee=1)
# Declared families (base ids) by the game's own tables: every log, every raw fish, and every
# mining_table rock_output except rune essence and quest items (clay, copper, tin, iron, silver,
# gold, mithril, adamantite, runite, coal, blurite). Bark, charcoal, slimey eel and gems (a separate
# roll) are none. Draynor production so far: logs, oak and willow logs; raw shrimps and anchovies.
GOAL_FAMILIES = dict(wood=frozenset((1511, 1513, 1515, 1517, 1519, 1521, 2862)),
                     fish=frozenset((317, 321, 327, 331, 335, 345, 349, 359, 371, 377, 383, 389, 395, 2148)),
                     mine=frozenset((434, 436, 438, 440, 442, 444, 447, 449, 451, 453, 668)))
# Eligible melee enemies (npc type ids): chicken and chicken_brown, the enemy of the native start sets
# (Entrana and its arena). A kill counts once per enemy life: this agent's first hit finds it at
# full health, nobody else hits it, and this agent's hit defeats it before the task closes.
MELEE_ENEMIES = frozenset((41, 1017))


def goal_items(report, goal, bound='retained_lower'):
    """Certified outcomes of the goal: retained items of its families (report['families'],
    baseline/native_value.py), or for melee eligible kills (report['kills'], lower or upper)."""
    if goal == 'melee':
        kills = report.get('kills')
        need(isinstance(kills, dict) and all(type(kills.get(k)) is int and kills[k] >= 0 for k in ('lower', 'upper'))
             and kills['lower'] <= kills['upper'], 'melee needs a combat audit with certified kills')
        return kills[bound.split('_')[-1]]
    return sum(family[bound] for base, family in report['families'].items() if int(base) in GOAL_FAMILIES[goal])


def goal_return(report, goal, bound='retained_lower'):
    target = GOAL_TARGETS[goal]
    return min(goal_items(report, goal, bound), target) / target if report['survived'] else 0.


def need(condition, message):
    if not condition:
        raise EvidenceError(message)


class ResourceClock:
    def __init__(self, armed, horizon):
        need(type(horizon) is int and 1 <= horizon <= 10000, 'invalid task horizon')
        need(armed.get('schema')=='native-resource-capture-v2' and armed.get('failure') is None and
             armed.get('closed') is False and armed.get('lifecycleObserved') is True,
             'invalid armed resource evidence')
        need(type(armed.get('flowEvidenceVersion')) is int and armed['flowEvidenceVersion']==1,
             'native flow scoring requires write/ground coverage at arm')
        self.owned_containers = deepcopy(owned_containers(armed))
        need(armed.get('targetScope')==GATHERING and armed.get('targets'),'value scoring requires gathering-v1 at arm')
        self.identity=deepcopy(armed['identity'])
        need(set(self.identity)=={'worldBootId','accountId','accountEpoch','incarnationId'} and
             all(isinstance(self.identity[k],str) and self.identity[k] for k in ('worldBootId','accountId','incarnationId')) and
             type(self.identity['accountEpoch']) is int and self.identity['accountEpoch']>0,'invalid armed identity')
        self.start=armed['startTick'];self.end=armed['endTick']
        need(type(self.start) is int and self.start>=0 and self.end==self.start+horizon,
             'armed horizon mismatch')
        self.horizon=horizon;self.last_tick=self.start;self.last_status='active'

    def accept(self, reply):
        need(reply.get('type')=='resource_budget' and reply.get('identity')==self.identity,
             'task clock identity changed')
        b=reply.get('budget',{})
        need(set(b)=={'schema','startTick','endTick','observedTick','horizon','remainingTicks','status'} and
             b['schema']=='resource-budget-v1','invalid public budget schema')
        need(all(type(b[k]) is int for k in ('startTick','endTick','observedTick','horizon','remainingTicks')),
             'invalid public budget integers')
        tick,status=b['observedTick'],b['status']
        need(b['startTick']==self.start and b['endTick']==self.end and b['horizon']==self.horizon,
             'task budget changed')
        need(self.last_tick<=tick<=self.end and b['remainingTicks']==self.end-tick,
             'task clock regressed or remaining time disagrees')
        need(status in ('active','horizon','death') and
             ((status=='horizon' and tick==self.end) or (status!='horizon' and tick<self.end)),
             'invalid task closure')
        if self.last_status!='active':
            need(status==self.last_status and tick==self.last_tick,'closed task clock changed')
        self.last_tick=tick;self.last_status=status
        return np.asarray([b['remainingTicks']/self.horizon,self.horizon/10000],dtype=np.float32)

    def score(self, trace, scale):
        need(self.last_status!='active','cannot score an active task')
        need(owned_containers(trace)==self.owned_containers, 'owned containers changed during episode')
        report=audit_native_value(trace)
        need(report['identity']==self.identity and trace['startTick']==self.start and trace['endTick']==self.end
             and trace['observedTick']==self.last_tick and trace['terminal']['reason']==self.last_status,
             'terminal trace and task clock disagree')
        need(type(scale) in (int,float) and math.isfinite(scale) and scale>0,'invalid reward scale')
        # Complete native evidence may still leave a mixed stack's origin
        # uncertain. The reward is only its certified lower bound; it does not
        # discard that legal trajectory or invent an origin.
        return report['reward_eligible_lower']/scale,report


class ResourceEpisode:
    """One actor, one owned world, first-life finite-horizon episodes.

Reward is paid as it is earned: each step pays the change in the value held (target
items carried, plus those deposited this episode, at the game's price), and the
terminal step pays the certified total minus everything paid before. The sum is
exactly the certified total, so the best policy is the one for a terminal-only reward
(Ng et al. 1999); a death settles to zero. Held value starts at zero: arm requires no
target items. With a goal (goal-v1) the same holds for min(goal items, 2) / 2: losing an
item takes its progress back, and the terminal audit settles the estimate.

control(operation, identity=..., horizon=...) returns the full native protocol
reply. record(kind, **data) receives every raw row; evaluation keeps them all, training keeps
one effect row per decision and the full rows of selected episodes (baseline/telemetry.py).
prepare, when given, uses ordinary setup actions and returns the last public
observation. Neither hook is ever called to choose actions inside the rollout.
"""
    def __init__(self, transport, control, cfg, horizon, reward_scale, record, prepare=None, goal=None):
        # Raises unless cfg asks for item-v2 and act-v3.
        acts_from_config(cfg)
        need(cfg.get('lifecycle','episodic-v1')=='episodic-v1' and not cfg.get('resetPrelude'),
             'resource task requires fresh first-life setup')
        need(type(horizon) is int and 1<=horizon<=10000,'invalid task horizon')
        need(type(reward_scale) in (int,float) and math.isfinite(reward_scale) and reward_scale>0,'invalid task reward scale')
        need(goal is None or goal in GOALS,'invalid goal')
        # With a goal the held measure counts the goal's items, not coins (see _held).
        self.goal=goal;self.scoring=VALUE_SCORING if goal is None else GOAL_SCORING
        self.transport=transport;self.control=control;self.cfg=deepcopy(cfg)
        self.horizon=horizon;self.scale=reward_scale;self.record=record;self.prepare=prepare
        self.clock=None;self.raw=None;self.projected=None;self.failed=False;self.closed=False
        self.banked=0;self.carried=0;self.bank_was_open=False;self.prices={};self.paid=0
        self.active=False;self.bound=False;self.prepared=False;self.decisions=0;self.previous_identity=None
        self.bound_identity=None

    def _control(self, operation, **kwargs):
        reply=self.control(operation,**kwargs)
        self.record('resource_control',operation=operation,response=reply)
        return reply

    def _carried(self):
        return sum(self.prices.get(i.get('id'),0)*i['count'] for i in self.raw.get('inventory',[]))

    def _held(self):
        """Paid so far, in reward units times the scale: the value held, or with a goal the
        progress min(goal items held, 2) / 2. An estimate the terminal audit settles."""
        held=self.banked+self.carried
        return held if self.goal is None else min(held,GOAL_ITEMS)/GOAL_ITEMS*self.scale

    def _track_bank(self):
        # Target items leave the inventory with the bank open only by a deposit, and
        # return by a withdrawal; drops are masked and death ends the episode.
        carried=self._carried()
        if self.raw.get('bankOpen') or self.bank_was_open:self.banked+=self.carried-carried
        self.carried=carried;self.bank_was_open=bool(self.raw.get('bankOpen'))

    def _budget(self):
        reply=self._control('budget',identity=self.clock.identity)
        values=self.clock.accept(reply)
        result=project(self.raw)
        result['budget_vec']=values
        # Critic only (policy.py): the value banked this episode, which a death takes
        # back; the actor sees the bank only while it is open.
        result['critic_vec']=np.asarray([self.banked/self.scale],dtype=np.float32)
        if self.goal is not None:
            result['goal_vec']=np.asarray([g==self.goal for g in GOAL_VEC],dtype=np.float32)
        self.projected=result
        return result

    def reset(self, seed):
        self.prepare_start(seed)
        return self.start()

    def prepare_start(self, seed):
        """The setup before the task clock (reset): the native reset and scripted setup, ending in a live idle
        observation. start() arms the clock; a lockstep world does not tick in between."""
        need(not self.failed and not self.closed and not self.active and not self.bound,'invalid resource reset lifecycle')
        try:
            self.raw=self.transport.reset(seed=seed,cfg=deepcopy(self.cfg))
            self.record('reset',seed=seed,cfg=self.cfg,scoring=self.scoring,goal=self.goal,observation=self.raw)
            if self.prepare is not None:self.raw=self.prepare(self.transport,self.raw,self.record)
            need(not self.raw.get('done') and not self.raw.get('truncated') and not self.raw.get('optionBusy')
                 and not self.raw.get('modalOpen'),'setup did not yield a live idle public observation')
            self.raw=episode_start(self.raw)
            self.record('episode_start',observation=self.raw)
            project(self.raw)
            self.prepared=True
        except Exception:
            self.failed=True
            raise

    def start(self):
        need(not self.failed and not self.closed and not self.active and not self.bound and self.prepared,
             'invalid resource start lifecycle')
        self.prepared=False
        try:
            identity=self._control('identity')['identity']
            if self.previous_identity is not None:
                previous=self.previous_identity
                need(identity['worldBootId']==previous['worldBootId'] and identity['accountId']==previous['accountId'] and
                     identity['accountEpoch']>previous['accountEpoch'] and identity['incarnationId']!=previous['incarnationId'],
                     'reset did not produce a fresh incarnation in the same owned world/account')
            self.bound_identity=deepcopy(identity)
            # A lost arm response is uncertain, not proof that no capture started.
            self.bound=True
            armed=self._control('arm',identity=identity,horizon=self.horizon,targets=GATHERING)
            need(armed.get('identity')==identity and armed['audit'].get('identity')==identity,'arm identity mismatch')
            self.clock=ResourceClock(armed['audit'],self.horizon)
            targets=armed['audit']['targets']
            if self.goal is None:
                self.prices={t[k]:t['value'] for t in targets for k in ('base','note') if t[k] is not None}
            elif self.goal=='melee':
                # Kills settle at the end from the combat audit; nothing held pays before it.
                need(armed['audit'].get('combat',{}).get('schema')==COMBAT_SCHEMA,'melee needs the native combat capture at arm')
                self.prices={}
            else:
                need(GOAL_FAMILIES[self.goal]<={t['base'] for t in targets},'goal families not all certified at arm')
                self.prices={t[k]:1 for t in targets for k in ('base','note')
                             if t[k] is not None and t['base'] in GOAL_FAMILIES[self.goal]}
            self.banked=0;self.carried=self._carried();self.bank_was_open=bool(self.raw.get('bankOpen'));self.paid=0
            need(self.carried==0,'target items held at arm')
            self.previous_identity=deepcopy(identity);self.decisions=0
            result=self._budget()
            need(self.clock.last_status=='active','task expired during reset')
            self.active=True
            return result
        except Exception:
            self.failed=True
            raise

    def step(self, action):
        need(not self.failed and not self.closed and self.active,'invalid resource step lifecycle')
        try:
            need(type(action) is int and 0<=action<96 and self.projected['mask'][action], 'invalid/masked policy action')
            self.decisions+=1
            # Policy inference/transport time consumes native ticks too. If the
            # horizon closed before this request, do not dispatch into the next life.
            self._budget()
            executed=None
            if self.clock.last_status=='active':
                self.record('requested',action=action)
                self.raw=self.transport.step(action);executed=action
                self._track_bank()
                self.record('observed',action=action,observation=self.raw)
                self._budget()
            transition=dict(requested_action=action,executed_action=executed,
                            native_tick=self.clock.last_tick,status=self.clock.last_status,scoring=self.scoring)
            if self.clock.last_status=='active':
                need(not self.raw.get('done') and not self.raw.get('truncated'),'transport ended before native task closure')
                held=self._held()
                reward=(held-self.paid)/self.scale;self.paid=held
                self.record('task_transition',**transition,reward=reward,terminated=False)
                return self.projected,reward,False,{}
            trace=self._control('read',identity=self.clock.identity)['audit']
            total,report=self.clock.score(trace,self.scale)
            total_upper=report['retained_upper']/self.scale if report['survived'] else 0.0
            if self.goal is not None:
                # Kills are certified for every goal (a diagnostic elsewhere); melee requires them.
                if self.goal=='melee' or 'combat' in trace:report['kills']=audit_native_combat(trace,MELEE_ENEMIES)
                total,total_upper=goal_return(report,self.goal),goal_return(report,self.goal,'retained_upper')
            reward=total-self.paid/self.scale  # settles every earlier payment to the certified total
            self.record('task_transition',**transition,reward=reward,terminated=True)
            self.record('task_terminal',report=report,reward=total,reward_upper=total_upper,
                        reward_scale=self.scale,scoring=self.scoring,goal=self.goal)
            released=self._control('release',identity=self.clock.identity)['audit']
            need(released==trace,'released trace changed')
            self.bound=False;self.active=False
            return self.projected,reward,True,{'resource_report':report,'scoring':self.scoring,'goal':self.goal,
                'reward_bounds':[total,total_upper],'episode':{
                'r':total,'l':self.decisions,'native_ticks':report['delivered_ticks']}}
        except Exception:
            self.failed=True
            raise

    def close(self):
        if self.closed:return
        try:
            if self.bound and self.bound_identity is not None:
                try:self._control('retire',identity=self.bound_identity)
                except Exception as error:self.record('retirement_failed',error=str(error))
        finally:
            self.transport.close();self.closed=True;self.active=False


class ResourceBackend:
    """One-world backend implementing the existing GameVecEnv bridge protocol."""
    n_envs=1

    def __init__(self, episode):
        self.episode=episode;self.seed=0

    @staticmethod
    def batch(obs):
        return {k:np.expand_dims(v,0) for k,v in obs.items()}

    def reset(self, seed=None):
        if seed is not None:self.seed=seed
        return self.batch(self.episode.reset(self.seed))

    def step(self, actions):
        need(np.asarray(actions).shape==(1,) and np.issubdtype(np.asarray(actions).dtype,np.integer),'invalid resource action batch')
        obs,reward,done,info=self.episode.step(int(actions[0]))
        if done:
            final=deepcopy(obs);final_info=deepcopy(info);self.seed+=1
            obs=self.episode.reset(self.seed)
            info={'final_observation':final,'final_info':final_info,'episode':final_info['episode']}
        # Task horizon is a terminal objective, not a TimeLimit bootstrap target.
        return self.batch(obs),np.asarray([reward],np.float32),np.asarray([done],bool),np.asarray([False],bool),[info]

    def close(self):
        self.episode.close()
