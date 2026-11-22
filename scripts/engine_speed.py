"""Engine speed study: how fast lockstep worlds of the real engine play free-play-v1's native protocol, and where the
wall time goes. Measurement only: no training, no reward, no monitor, and no native-watch-v1 run (no run.json).

Runs inside scripts/stack.py, whose engines profile themselves with RL_TICK_PROFILE=1 (env-ts/engine/tick-profile.ts):

    RL_TICK_PROFILE=1 NODE_RANDOM_EVENTS=false .venv/bin/python scripts/stack.py --bun BUN --output OUT/stack \\
        --timeout-seconds 600 --lockstep --xp-rate 1 --worlds N -- \\
        .venv/bin/python scripts/engine_speed.py --output OUT/speed.json --stack OUT/stack --form watch \\
            --world village-world-v2 --actor random --ticks 3000 --sites SITE ...

One thread a world, as sim/watch.py. Forms:
- `watch`: sim/watch.py play_world, as native-watch-v1 plays it (the captures armed, a budget control before and after
  each step, every row written to the world's evidence file), with a timer around each part.
- `lean`: the transport alone (the native reset, the setup steps, then steps until the ticks are spent), as a training
  collector with no per-step control and no evidence would play it.
- `batch`: shared-lockstep-v1 (env-ts/engine/batch.ts, engines started with RL_SHARED_AGENTS=N): one connection a
  world carries `--agents` agents; one request answers every agent whose decision is due and returns the next ones; the
  engine deals the start places (`places`: the mainland grid, or `--centres` with `--radius`); each reset runs the
  watch's setup; at each life's end the XP paid from the replies is compared with the engine's ledger; no capture. Ticks are the engines' own cycles (the profile), decisions the steps sent, setup steps included.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time

PARTS = ('reset', 'setup', 'step', 'control', 'actor', 'project', 'evidence', 'decode')


class Clock:
    """One agent's wall seconds by part, and the engine's own step timing (obs.timing, whole milliseconds)."""

    def __init__(self):
        self.parts = dict.fromkeys(PARTS, 0.0)
        self.engine = dict(dispatch=0, tick=0, obs=0, total=0)
        self.steps = 0
        self.controls = {}

    def add(self, part, seconds):
        self.parts[part] += seconds

    def timed(self, part, fn):
        def call(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.add(part, time.perf_counter() - t0)
        return call


LOCAL = threading.local()


def mac_state():
    """Load averages, the power source and the lid (macOS; None where a read fails)."""
    def read(argv):
        try:
            return subprocess.run(argv, capture_output=True, text=True, timeout=10).stdout
        except OSError:
            return None
    battery, lid = read(['pmset', '-g', 'batt']), read(['ioreg', '-r', '-k', 'AppleClamshellState', '-d', '1'])
    return dict(load=[round(x, 2) for x in os.getloadavg()], time=time.time(),
                power=battery.splitlines()[0].strip() if battery else None,
                lid_closed=None if not lid or 'AppleClamshellState' not in lid else '"AppleClamshellState" = Yes' in lid)


def engine_pids(stack):
    """The engines scripts/stack.py launched ({name: pid}), from its supervisor log."""
    out = {}
    for line in (Path(stack) / 'supervisor.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('event') == 'launched' and row['name'].startswith('engine'):
            out[row['name']] = row['owner_pid']
    return out


def processes(pids):
    """{name: (resident MiB, cpu seconds)} of the named processes, by ps."""
    out = {}
    for name, pid in pids.items():
        text = subprocess.run(['ps', '-o', 'rss=,cputime=', '-p', str(pid)], capture_output=True, text=True).stdout.split()
        if len(text) == 2:
            parts = [float(x) for x in text[1].replace('-', ':').split(':')]
            seconds = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
            out[name] = (round(int(text[0]) / 1024, 1), round(seconds, 2))
    return out


def profile(transport, reset=False):
    return transport._rpc(dict(cmd='profile', reset=reset), timeout_s=10.0)


def timed_transport(transport, clock):
    """The transport with its step and reset timed, and the engine's own step timing summed."""
    step, reset = transport.step, transport.reset

    def timed_step(action):
        t0 = time.perf_counter()
        raw = step(action)
        clock.add('step', time.perf_counter() - t0)
        clock.steps += 1
        for key in clock.engine:
            clock.engine[key] += raw.get('timing', {}).get(key, 0)
        return raw
    transport.step, transport.reset = timed_step, clock.timed('reset', reset)
    return transport


def watch_world(k, link, site, actor_name, ticks, deadline, out, world, key, seed, excluded):
    """World k played by sim/watch.py play_world (module docstring, `watch`)."""
    import sim.watch as watch
    from sim import free
    transport, service = link
    clock = LOCAL.clock
    timed_transport(transport, clock)
    actor = dict(wait=free.Wait, random=free.Random)[actor_name]()
    actor.act = clock.timed('actor', actor.act)
    rules = watch.world_rules(world)
    return watch.play_world(k, transport, service, site, watch.SITES[site], watch.Records(key, k, rules['start_group']),
                            actor, seed, ticks, deadline, out / f'world-{k}.jsonl.gz', excluded)


def lean_world(k, transport, site, actor_name, ticks, deadline, world, key, seed):
    """Agent k played on the transport alone (module docstring, `lean`): lives until `ticks` ticks are spent."""
    import sim.watch as watch
    from obs import project_v6
    from sim import free
    clock = LOCAL.clock
    timed_transport(transport, clock)
    actor = dict(wait=free.Wait, random=free.Random)[actor_name]()
    act, project = clock.timed('actor', actor.act), clock.timed('project', project_v6)
    records, spent, lives = watch.Records(key, k, watch.world_rules(world)['start_group']), 0, []
    while spent < ticks and time.monotonic() < deadline:
        record = records.next()
        raw = transport.reset(record['reset_seed'], watch.life_cfg(watch.SITES[site], record))
        t0 = time.perf_counter()
        raw = watch.setup(record)(transport, raw, lambda kind, **data: None)
        clock.add('setup', time.perf_counter() - t0)
        actor.reset(watch.life_seed(seed, k, len(lives)))
        start = raw['tick']
        while spent + raw['tick'] - start < ticks and time.monotonic() < deadline:
            raw = transport.step(int(act(project(raw), None)))
            if raw.get('done') or raw.get('truncated'):
                break
        spent += raw['tick'] - start
        lives.append(dict(ticks=raw['tick'] - start, done=bool(raw.get('done'))))
    transport.close()
    return dict(world=k, site=site, lives=lives, armed_ticks=spent, status='complete' if spent >= ticks else 'cut')


def batch_world(w, transport, agents, actor_name, ticks, deadline, world, key, seed, centres=None, radius=None):
    """World w's `agents` agents on one shared-lockstep-v1 connection (env-ts/engine/batch.ts) until the world has run
    `ticks` ticks: each life from the watch's next start record with its setup in the reset, a new life at each death,
    and at each life's end the XP paid from the step replies compared with the engine's ledger."""
    import sim.watch as watch
    from obs import project_v6
    from sim import free
    clock, group = LOCAL.clock, watch.world_rules(world)['start_group']

    def call(request):  # the request path: send, the engine's work and wait, then the JSON decode apart
        t0 = time.perf_counter()
        transport.sock.sendall((json.dumps(request) + '\n').encode())
        line = transport._readline(900.0)
        t1 = time.perf_counter()
        reply = json.loads(line)
        clock.add('decode', time.perf_counter() - t1)
        clock.add('step', t1 - t0)
        if reply.get('type') in ('error', 'fatal'):
            raise RuntimeError(f"{reply.get('code')}: {reply.get('message')}")
        return reply
    call(dict(cmd='hello', protocol='shared-lockstep-v1'))
    place = dict(cmd='places', count=agents, seed=seed + w)
    if centres:
        place.update(centres=centres, radius=radius)
    tiles = call(place)['tiles']
    records = [watch.Records(key, w * agents + k, group) for k in range(agents)]
    actors = [dict(wait=free.Wait, random=free.Random)[actor_name]() for _ in range(agents)]
    act, project = clock.timed('actor', lambda k, raw: actors[k].act(raw, None)), clock.timed('project', project_v6)
    lives, paid, last, errors = [0] * agents, [None] * agents, [None] * agents, []
    count = dict(resets=0, deaths=0, ledgers=0, ledger_mismatch=0, ledger_grants=0, eats=0, killers=0, replies_checked=0,
                 reply_mismatch=0)

    def reset(k):
        record = records[k].next()
        actors[k].reset(watch.life_seed(seed, w * agents + k, lives[k]))
        lives[k] += 1
        return dict(op='reset', slot=k, seed=record['reset_seed'], cfg=watch.life_cfg((tiles[k]['x'], tiles[k]['z']), record),
                    setup=dict(wield=record['weapon'] if record['worn'] else None))

    def settle(k, ledger):  # the life's paid XP against the engine's own ledger
        if ledger is None or paid[k] is None:
            return
        count['ledgers'] += 1
        count['ledger_grants'] += len(ledger['grants'])
        if [int(x) for x in paid[k]] != ledger['total']:
            count['ledger_mismatch'] += 1
    items, start = [reset(k) for k in range(agents)], None
    while True:
        reply = call(dict(cmd='batch', items=items))
        start = reply['tick'] if start is None else start
        items = []
        for result in reply['results']:
            k, kind = result['slot'], result['kind']
            if kind in ('reset', 'error', 'left'):
                settle(k, result.get('ledger'))
            if kind == 'error':  # a reset or step that failed: a new life from the next record
                errors.append(f"{result.get('code')}: {result.get('message')}")
                if len(errors) > agents:
                    raise RuntimeError(f'more failures than agents: {errors[:3]}')
                paid[k] = None
                items.append(reset(k))
                continue
            if kind == 'reset':
                count['resets'] += 1
                paid[k], last[k] = [0] * len(result['pay']['xp']), result['pay']['xp']
            else:
                clock.steps += 1
                for part in ('dispatch', 'obs'):
                    clock.engine[part] += result['obs'].get('timing', {}).get(part, 0)
                xp = result['pay']['xp']
                paid[k] = [p + now - before for p, now, before in zip(paid[k], xp, last[k])]
                last[k] = xp
                count['replies_checked'] += 1  # the pay so far against the engine's running ledger total
                count['reply_mismatch'] += [int(x) for x in paid[k]] != result['pay']['ledger']
            if result['done']:
                count['deaths'] += bool(result['dead'])
                count['eats'] += result['facts']['eats']
                count['killers'] += bool((result['facts'].get('death') or {}).get('killers'))
                items.append(reset(k))
            else:
                items.append(dict(op='step', slot=k, action=int(act(k, project(result['obs'])))))
        if reply['tick'] - start >= ticks or time.monotonic() >= deadline:
            break
    return dict(world=w, agents=agents, armed_ticks=(reply['tick'] - start) * agents, world_ticks=reply['tick'] - start,
                failures=len(errors), failure_sample=errors[:3], **count,
                status='complete' if reply['tick'] - start >= ticks else 'cut')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stack', type=Path, required=True, help="the stack's --output (its supervisor log names the engines)")
    parser.add_argument('--form', choices=('watch', 'lean', 'batch'), required=True)
    parser.add_argument('--world', default='village-world-v2')
    parser.add_argument('--actor', choices=('wait', 'random'), default='random')
    parser.add_argument('--ticks', type=int, required=True, help='ticks an agent plays (armed ticks in the watch form)')
    parser.add_argument('--sites', nargs='+', default=[], help="one site an agent, from sim/watch.py SITES (not batch)")
    parser.add_argument('--agents', type=int, default=1, help='agents a world (batch form only: RL_SHARED_AGENTS)')
    parser.add_argument('--centres', default=None, help="batch form: start places near these x,z centres ('x,z;x,z')")
    parser.add_argument('--radius', type=int, default=16, help='batch form: the places radius around each centre')
    parser.add_argument('--guard', type=float, default=480.0)
    parser.add_argument('--key', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    from baseline.world import JsonlEvidence, ResourceService, join, worlds as stack_worlds
    from vec_adapter import EngineEnvServer
    import sim.watch as watch
    available = stack_worlds()
    batch = args.form == 'batch'
    agents = len(available) if batch else len(available) * args.agents  # threads: one a world in the batch form
    if args.agents > 1 and not batch:
        raise SystemExit('several agents a world: batch form only (shared-lockstep-v1, RL_SHARED_AGENTS)')
    centres = [[int(v) for v in pair.split(',')] for pair in args.centres.split(';')] if args.centres else None
    if not batch and (len(args.sites) != agents or not set(args.sites) <= set(watch.SITES)):
        raise SystemExit(f'{agents} sites of sim/watch.py SITES, one an agent')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f'{args.output} exists')
    evidence_dir = args.output.with_suffix('')
    excluded = None
    if args.form == 'watch':  # the world's excluded objects, read once from its archive (sim/watch.py check_settings)
        excluded = watch.check_settings(output=evidence_dir, world=args.world, actor=args.actor,
                                        sites=list(watch.SITES)[:4], rotation=0, key=args.key, seed=args.seed,
                                        ticks=args.ticks, guard=args.guard)['excluded']
    evidence_dir.mkdir()
    clocks = [Clock() for _ in range(agents)]
    call = JsonlEvidence.__call__

    def evidence(self, kind, **data):  # every thread's evidence rows, timed on that thread's clock
        t0 = time.perf_counter()
        try:
            return call(self, kind, **data)
        finally:
            LOCAL.clock.add('evidence', time.perf_counter() - t0)
    JsonlEvidence.__call__ = evidence
    control = ResourceService.__call__

    def timed_control(self, operation, **kwargs):  # the capture controls (watch form), timed on the calling thread's clock
        t0 = time.perf_counter()
        try:
            return control(self, operation, **kwargs)
        finally:
            seconds, clock = time.perf_counter() - t0, LOCAL.clock
            clock.add('control', seconds)
            c = clock.controls.setdefault(operation, [0, 0.0])
            c[0], c[1] = c[0] + 1, c[1] + seconds
    ResourceService.__call__ = timed_control
    project = watch.project_v6
    watch.project_v6 = lambda raw: LOCAL.clock.timed('project', project)(raw)
    links = []
    per_world = 1 if batch else args.agents  # connections a world
    for k in range(agents):
        world = available[k // per_world]
        links.append(join(world, lambda kind, **data: None, k) if args.form == 'watch'
                     else (EngineEnvServer(world['env_port'], f'speed{k}'), None))
    pids = engine_pids(args.stack)
    for transport, _ in links[::per_world]:
        profile(transport, reset=True)
    before, cpu0, state0 = processes(pids), resource.getrusage(resource.RUSAGE_SELF), mac_state()
    profiles, results = [None] * len(available), [None] * agents
    remaining, counted = [per_world] * len(available), threading.Lock()
    started = time.monotonic()
    deadline = started + args.guard

    def play(k):
        LOCAL.clock = clocks[k]
        transport, w = links[k][0], k // per_world
        close = transport.close

        def read_then_close():  # the world's last agent reads its profile before it leaves
            with counted:
                remaining[w] -= 1
                last = remaining[w] == 0
            if last:
                profiles[w] = profile(transport)
            close()
        transport.close = read_then_close
        t0 = time.monotonic()
        try:
            if batch:
                results[k] = batch_world(k, transport, args.agents, args.actor, args.ticks, deadline, args.world,
                                         args.key, args.seed, centres, args.radius)
                transport.close()
            elif args.form == 'watch':
                results[k] = watch_world(k, links[k], args.sites[k], args.actor, args.ticks, deadline, evidence_dir,
                                         args.world, args.key, args.seed, excluded)
            else:
                results[k] = lean_world(k, transport, args.sites[k], args.actor, args.ticks, deadline, args.world,
                                        args.key, args.seed)
        except Exception as error:
            results[k] = dict(world=k, status='stopped', cause=f'{type(error).__name__}: {error}')
            transport.close()
        results[k]['wall'] = round(time.monotonic() - t0, 3)
    threads = [threading.Thread(target=play, args=(k,), daemon=True) for k in range(agents)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - started
    after, cpu1, state1 = processes(pids), resource.getrusage(resource.RUSAGE_SELF), mac_state()
    ticks = sum(p['ticks'] for p in profiles if p)
    decisions = sum(c.steps for c in clocks)
    played = sum((r or {}).get('armed_ticks') or 0 for r in results)  # each agent's own ticks (armed ticks: watch form)
    report = dict(
        schema='engine-speed-v1', form=args.form, actor=args.actor, world=args.world, worlds=len(available),
        agents_per_world=args.agents, ticks_asked=args.ticks, sites=args.sites, wall=round(wall, 3),
        world_ticks=ticks, decisions=decisions,
        ticks_per_s_per_world=round(ticks / wall / len(available), 2), ticks_per_s=round(ticks / wall, 2),
        decisions_per_s=round(decisions / wall, 2), ticks_per_decision=round(ticks * args.agents / decisions, 2) if decisions else None,
        agent_ticks=played, agent_ticks_per_s=round(played / wall, 2),
        python_cpu_s=round(cpu1.ru_utime + cpu1.ru_stime - cpu0.ru_utime - cpu0.ru_stime, 2),
        engines={name: dict(rss_mib=after[name][0], cpu_s=round(after[name][1] - before[name][1], 2))
                 for name in after if name in before},
        mac=dict(before=state0, after=state1),
        agents=[dict(result={key: v for key, v in (r or {}).items() if key != 'lives'},
                     lives=len((r or {}).get('lives') or []),
                     parts={key: round(v, 3) for key, v in c.parts.items()}, steps=c.steps, engine_ms=c.engine,
                     controls={key: [n, round(s, 3)] for key, (n, s) in c.controls.items()})
                for r, c in zip(results, clocks)],
        profiles=profiles)
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    print(f"[speed] {args.form} {len(available)} worlds x {args.agents}: {ticks} ticks, {decisions} decisions in "
          f"{wall:.1f} s: {report['ticks_per_s_per_world']} ticks/s a world, {report['decisions_per_s']} decisions/s",
          flush=True)


if __name__ == '__main__':
    main()
