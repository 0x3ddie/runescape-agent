"""Native provenance flow: the graph the slot-flow audit (native_slot_flow.py) scores.

Flow capacities are observed quantities, not invented physical log identities.
Two integral min-cost flows bound production retained after ambiguous stack splits.
"""
from collections import deque

from baseline.native_trace import LOGS, family_ids, require
from baseline.evidence import integer


class Flow:
    def __init__(self):
        self.nodes = 2
        self.edges = []
        self.produced = 0

    def node(self):
        result = self.nodes
        self.nodes += 1
        return result

    def edge(self, start, end, count, cost=0):
        if count:
            self.edges.append((start, end, count, cost))

    def bounds(self, retained):
        """Send *all* produced mass to retained, lost, or remaining-ground sinks."""
        edges = self.edges + [(node, 1, count, 1) for node, count in retained]
        values = []
        for sign in (1, -1):
            graph = [[] for _ in range(self.nodes)]
            for start, end, cap, cost in edges:
                forward = [end, cap, sign * cost, len(graph[end])]
                reverse = [start, 0, -sign * cost, len(graph[start])]
                graph[start].append(forward)
                graph[end].append(reverse)
            sent = total_cost = 0
            while sent < self.produced:
                distance, previous = [float('inf')] * self.nodes, [None] * self.nodes
                distance[0] = 0
                queue, queued = deque([0]), {0}
                while queue:
                    start = queue.popleft()
                    queued.remove(start)
                    for index, edge in enumerate(graph[start]):
                        end, cap, cost, _ = edge
                        if cap and distance[end] > distance[start] + cost:
                            distance[end] = distance[start] + cost
                            previous[end] = (start, index)
                            if end not in queued:
                                queued.add(end)
                                queue.append(end)
                require(previous[1] is not None, 'infeasible resource flow')
                amount, end = self.produced - sent, 1
                while end:
                    start, index = previous[end]
                    amount = min(amount, graph[start][index][1])
                    end = start
                end = 1
                while end:
                    start, index = previous[end]
                    edge = graph[start][index]
                    edge[1] -= amount
                    graph[end][edge[3]][1] += amount
                    end = start
                sent += amount
                total_cost += amount * distance[1]
            values.append(sign * total_cost)
        return tuple(values)


def slots(snapshot):
    return {(location, row['slot'], row['id']): row['count']
            for location in ('inventory', 'bank') for row in snapshot[location]}


class Provenance:
    def __init__(self, ids=family_ids(LOGS)):
        self.ids = ids
        self.flow = Flow()
        self.account = {}
        self.ground = {}
        self.tiles = {}
        self.objects = {}
        self.last_order = 0
        self.transfers = self.uncertain_ops = 0

    def change(self, current, after):
        """Keep unchanged units; expose only removed/added quantities to allocation."""
        removed, added = [], []
        for key in sorted(current.keys() | after.keys()):
            old_node, old_count = current.get(key, (None, 0))
            new_count = after.get(key, 0)
            if old_count == new_count:
                continue
            new_node = self.flow.node() if new_count else None
            kept = min(old_count, new_count)
            if kept:
                self.flow.edge(old_node, new_node, kept)
            if old_count > kept:
                removed.append((old_node, old_count - kept))
            if new_count > kept:
                added.append((new_node, new_count - kept))
            if new_count:
                current[key] = (new_node, new_count)
            else:
                current.pop(key, None)
        return removed, added

    def allocate(self, removed, added, uncertain=False):
        incoming, outgoing = sum(n for _, n in removed), sum(n for _, n in added)
        hub = self.flow.node()
        for node, count in removed:
            self.flow.edge(node, hub, count)
        for node, count in added:
            self.flow.edge(hub, node, count)
        # Unknown opcodes might destroy old units and independently introduce
        # external ones even at zero net change. Never certify such replacement.
        self.flow.edge(hub, 1, incoming if uncertain else max(0, incoming - outgoing))

    def object(self, row):
        fields = {'captureObjectId', 'id', 'count', 'level', 'x', 'z',
                  'lifecycle', 'isActive', 'receiver64'}
        require(isinstance(row, dict) and set(row) == fields, 'invalid ground object fields')
        key = integer(row['captureObjectId'], 'ground identity', 1)
        count = integer(row['count'], 'ground count', 1)
        require(row['id'] in self.ids and type(row['isActive']) is bool,
                'invalid ground resource or active state')
        for field in ('id', 'level', 'x', 'z', 'lifecycle'):
            integer(row[field], 'ground ' + field)
        require(row['level'] <= 3 and isinstance(row['receiver64'], str), 'invalid ground metadata')
        identity = tuple(row[field] for field in ('id', 'level', 'x', 'z', 'lifecycle'))
        require(self.objects.get(key, identity) == identity, 'ground identity changed location/type')
        self.objects[key] = identity
        return key, count if row['isActive'] else 0

    def tile(self, rows, coordinate):
        require(isinstance(rows, list), 'missing tile snapshot')
        result, seen = {}, set()
        for row in rows:
            key, count = self.object(row)
            require(key not in seen and tuple(row[k] for k in ('level', 'x', 'z')) == coordinate,
                    'duplicate or misplaced ground object')
            seen.add(key)
            if count:
                result[key] = count
        return result

    def ground_changes(self, events):
        """Replay entry/exit snapshots; nested methods are not separate transfers."""
        calls, timeline = {}, []
        for event in events:
            require(event.get('method') in ('addObj', 'removeObj', 'changeObj'), 'unknown ground method')
            call = integer(event.get('callId'), 'ground call', 1)
            end = integer(event.get('endId'), 'ground return', 1)
            root = integer(event.get('rootCallId'), 'ground root', 1)
            parent = event.get('parentCallId')
            if parent is not None:
                integer(parent, 'ground parent', 1)
            require(call < end and call not in calls, 'invalid ground call interval')
            calls[call] = event
            self.object(event.get('before'))
            self.object(event.get('after'))
            require(event['before']['captureObjectId'] == event['after']['captureObjectId'],
                    'ground argument identity changed')
            coordinate = tuple(event['before'][k] for k in ('level', 'x', 'z'))
            timeline.extend([(call, True, event, coordinate), (end, False, event, coordinate)])
        # Native receipts are emitted at return. Reject reordered or missing trees.
        require([e['endId'] for e in events] == sorted(e['endId'] for e in events), 'ground return order')
        require(len({row[0] for row in timeline}) == len(timeline), 'duplicate ground ordering')
        stack, first, latest = [], {}, {}
        for order, entering, event, coordinate in sorted(timeline):
            require(order > self.last_order, 'ground ordering gap/duplicate')
            self.last_order = order
            call, root, parent = event['callId'], event['rootCallId'], event['parentCallId']
            if entering:
                require(parent == (stack[-1] if stack else None) and root == (stack[0] if stack else call),
                        'missing or crossing ground call')
                stack.append(call)
            else:
                require(stack and stack.pop() == call, 'ground return without matching entry')
            snapshot = self.tile(event['tileBefore' if entering else 'tileAfter'], coordinate)
            argument = event['before' if entering else 'after']
            if argument['isActive']:
                require(snapshot.get(argument['captureObjectId']) == argument['count'],
                        'active ground argument absent from tile')
            if coordinate not in first:
                if coordinate in self.tiles:
                    require(self.tiles[coordinate] == snapshot, 'unobserved ground mutation')
                else:
                    # First observed preexisting objects are external. Stable
                    # instance IDs never imply ownership by this account.
                    for key, count in snapshot.items():
                        require(key not in self.ground, 'ground identity appears on two tiles')
                        self.ground[key] = (self.flow.node(), count)
                first[coordinate] = snapshot
            elif entering:
                require(latest[coordinate] == snapshot, 'ground state gap at call entry')
            latest[coordinate] = snapshot
        require(not stack, 'unterminated ground call')
        after = {key: count for key, (_, count) in self.ground.items()}
        for coordinate, snapshot in latest.items():
            for key in first[coordinate]:
                after.pop(key, None)
            after.update(snapshot)
            self.tiles[coordinate] = snapshot
        return self.change(self.ground, after)

    def op(self, event):
        before, after = slots(event['before']), slots(event['after'])
        require({k: n for k, (_, n) in self.account.items()} == before, 'account flow state gap')
        ground_removed, ground_added = self.ground_changes(event['ground'])
        total = lambda values: sum(n for _, n in values)
        opcode = event['opcode']
        require(type(event.get('resourceMutation')) is bool, 'missing resource write coverage')
        require(event['resourceMutation'] or before == after, 'unmarked resource change')
        known = event['productionSource'] or opcode in (4302, 4305, 4307, 4308, 4319, 4320, 4321, 4309, 4311, 4312, 3510)
        known = known or not event['resourceMutation']
        if known:
            removed, added = self.change(self.account, after)
        else:
            # A setter can replace contents even if a slot's count stays equal
            # or increases. Preserve no unproven origin across unknown opcodes.
            removed = list(self.account.values())
            self.account = {k: (self.flow.node(), n) for k, n in after.items()}
            added = list(self.account.values())
        if event['productionSource']:
            require(not removed and total(added) == 1 and not ground_removed and not ground_added,
                    'unsupported production flow')
            self.flow.produced += 1
            for node, count in added:
                self.flow.edge(0, node, count)
        elif opcode in (4319, 4320, 4321) and not event['ground']:
            require(total(removed) == total(added), 'nonconserving native bank transfer')
            self.allocate(removed, added)
            self.transfers += 1
        elif opcode in (4309, 4311, 4312):
            require(not added and total(removed) == total(ground_added), 'nonconserving native drop')
            self.allocate(removed, ground_added)
            self.allocate(ground_removed, [])  # ground capacity eviction is a loss
            self.transfers += 1
        elif opcode == 3510:
            require(not removed and not ground_added and total(added) == total(ground_removed),
                    'nonconserving native pickup')
            self.allocate(ground_removed, added)
            self.transfers += 1
        elif opcode == 4302 and not event['ground']:
            require(not removed, 'external add removes existing units')
            self.allocate([], added)
        elif opcode in (4305, 4307, 4308) and not event['ground']:
            require(not added, 'native removal adds units')
            self.allocate(removed, [])
        else:
            # Unknown scripts do not receive a source exemption. Re-version all
            # observed slots, including equal-count replacement, conservatively.
            # Their old/new native quantities alone cannot prove a transfer.
            self.allocate(removed + ground_removed, added + ground_added, uncertain=True)
            self.uncertain_ops += 1

    def finish(self):
        for node, count in self.ground.values():
            self.flow.edge(node, 1, count)
        return self.flow.bounds(list(self.account.values()))

