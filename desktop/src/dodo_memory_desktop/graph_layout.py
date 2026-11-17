"""Stable placement for real catalogue clusters and their members."""

import math
from collections import defaultdict, deque
from functools import partial
from hashlib import blake2b

HUB_GAP = 178
CLOUD_RADIUS = 82
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))
FOLDERS = {"root", "drive", "folder", "directory", "project", "repository"}


def _distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _polar(radius, angle):
    return radius * math.cos(angle), radius * math.sin(angle)


def _hub_weights(nodes, edges, hubs):
    weights = {hub: defaultdict(float) for hub in hubs}
    for edge in edges:
        if edge["kind"] in {"contains", "inferred", "topic"}:
            continue
        source, target = (nodes.get(edge[key]) for key in ("source", "target"))
        if source is None or target is None:
            continue
        a, b = (node.get("cluster_id") or node["id"] for node in (source, target))
        if a in weights and b in weights and a != b:
            weight = math.log2(1 + edge.get("count", 1))
            weights[a][b] += weight
            weights[b][a] += weight
    return weights


def _place_hubs(hubs, weights, previous):
    placed = {hub: previous[hub] for hub in hubs if previous.get(hub, (0, 0, None))[2] == (hub,)}
    if len(placed) == len(hubs):
        return placed
    slots = [
        _polar(126 * math.sqrt(slot), GOLDEN_ANGLE * slot) for slot in range(max(32, 5 * len(hubs)))
    ]
    for point in placed.values():
        slots = [slot for slot in slots if _distance(slot, point) >= HUB_GAP**2]
    remaining = set(hubs).difference(placed)
    while remaining:
        hub = min(
            remaining,
            key=lambda hub: (
                -sum(value for other, value in weights[hub].items() if other in placed),
                -sum(weights[hub].values()),
                hub,
            ),
        )
        linked = [
            (placed[other], weight) for other, weight in weights[hub].items() if other in placed
        ]

        # The weighted centre minimises squared distance to existing neighbours and the origin.
        weight_sum = 0.12 + sum(weight for _, weight in linked)
        centre = tuple(
            sum(point[axis] * weight for point, weight in linked) / weight_sum for axis in (0, 1)
        )
        point = min(slots, key=partial(_distance, centre))
        placed[hub] = (*point, (hub,))
        slots = [slot for slot in slots if _distance(slot, point) >= HUB_GAP**2]
        remaining.remove(hub)
    return placed


def _cloud_slots():
    # The coarse slots keep overview members apart. Intermediate slots only fill on deep zoom.
    coarse, fine = [], []
    for row in range(-24, 25):
        for column in range(-24, 25):
            x, y = 4 * (column + row / 2), 2 * math.sqrt(3) * row
            if 0 < x * x + y * y <= CLOUD_RADIUS**2:
                (fine if row % 2 or column % 2 else coarse).append((x, y))
    return coarse, fine


CLOUD_SLOTS = _cloud_slots()


def _member_target(node, hub, parent):
    identity = node["id"]
    seed = int.from_bytes(blake2b(identity.encode(), digest_size=4).digest())
    angle = (seed & 0xFFFF) * math.tau / 65536
    folder = node.get("kind") in FOLDERS
    distance = 22 if folder else 13
    if parent == hub:
        distance = (44 if folder else 26) + (seed >> 16) % 31
    x, y = _polar(distance, angle)
    return parent[0] + x, parent[1] + y


def _place_members(nodes, hub, members, placed, previous):
    centre = placed[hub]
    children = defaultdict(list)
    occupied = set()
    for identity in sorted(members):
        parent = nodes[identity].get("parent_id")
        parent = parent if parent in members else hub
        children[parent].append(identity)
        if (old := previous.get(identity)) and old[2] == (hub, parent):
            placed[identity] = old
            occupied.add(old[:2])

    if all(identity in placed for identity in members):
        return

    slots = [
        {(centre[0] + x, centre[1] + y) for x, y in layer}.difference(occupied)
        for layer in CLOUD_SLOTS
    ]
    pending = deque([hub])
    while pending:
        parent_id = pending.popleft()
        parent = placed[parent_id]
        for identity in children[parent_id]:
            pending.append(identity)
            if identity in placed:
                continue
            node = nodes[identity]
            target = _member_target(node, centre, parent)
            free = slots[0] or slots[1]
            point = min(free, key=lambda slot: (_distance(slot, target), slot))
            free.remove(point)
            placed[identity] = (*point, (hub, parent_id))


def cluster_positions(nodes, edges, hubs, previous):
    """Place fresh hubs near related hubs and keep prior placements stable."""
    hub_ids = tuple(hub for hub in hubs if hub in nodes)
    placed = _place_hubs(hub_ids, _hub_weights(nodes, edges, hub_ids), previous)
    by_hub = defaultdict(set)
    for identity, node in nodes.items():
        if identity not in placed and node.get("cluster_id") in placed:
            by_hub[node["cluster_id"]].add(identity)
    for hub, members in by_hub.items():
        _place_members(nodes, hub, members, placed, previous)
    return placed
