"""Preflow push algorithm for computing max-flow"""

from enum import Enum, auto
from typing import Set, List
from unittest import TestCase
from collections import OrderedDict

import networkx as nx


class SearchDirection(Enum):
    """Specify the direction of search"""

    # Traverse to inward and outward facing arcs
    ALL = auto()
    # Traverse along inward facing arcs
    BACKWARDS = auto()
    # Travers along outward facing arcs
    FORWARD = auto()


def bfs_search(
    network: nx.DiGraph,
    source: int,
    direction: SearchDirection = SearchDirection.FORWARD,
):
    """Create a new BFSearch iterator on the given network from source
    in the direction specified.
    """
    to_visit: List[int] = [source]
    labeled: Set[int] = set()

    while to_visit:
        cur = to_visit.pop()

        # If this node has already been visited, don't visit it again
        if cur in labeled:
            continue

        # Add cur to labeled
        labeled.add(cur)

        if direction == SearchDirection.FORWARD:
            edges = network.out_edges(cur)
        elif direction == SearchDirection.BACKWARDS:
            edges = network.in_edges(cur)
        else:
            edges = list(network.in_edges(cur)) + list(network.out_edges(cur))

        for edge in edges:
            if direction == SearchDirection.FORWARD and edge[1] not in labeled:
                to_visit.append(edge[1])
            elif direction == SearchDirection.BACKWARDS and edge[0] not in labeled:
                to_visit.append(edge[0])
            elif edge[0] != cur and edge[0] not in labeled:
                to_visit.append(edge[0])
            elif edge[1] != cur and edge[1] not in labeled:
                to_visit.append(edge[1])

        yield cur


def preflow_push(network: nx.DiGraph, source: int, sink: int) -> nx.DiGraph:
    """Compute the max-flow in the given network from the source to the sink"""
    residual = nx.DiGraph()
    residual.add_nodes_from(network.nodes)

    residual.add_edges_from(
        (u, v, {"capacity": c}) for (u, v, c) in network.edges(data="capacity")
    )

    # Assign distances using BFS traversal
    residual.nodes[sink]["distance_label"] = 0
    for node in bfs_search(residual, sink, SearchDirection.BACKWARDS):
        if node == sink:
            continue
        min_label = min(
            residual.nodes[e[1]]["distance_label"]
            for e in residual.out_edges(node)
            if "distance_label" in residual.nodes[e[1]]
        )

        residual.nodes[node]["distance_label"] = min_label + 1

    if "distance_label" not in residual.nodes[source]:
        raise ValueError("Source and Sink are not connected")

    residual.nodes[source]["distance_label"] = len(residual.nodes)

    # Queue for active nodes
    next_nodes = OrderedDict()

    # Push flow along all arcs from source
    for (u, v, c) in list(residual.out_edges(source, data="capacity")):
        residual.nodes[v]["excess"] = c
        next_nodes[v] = None
        residual.remove_edge(u, v)
        residual.add_edge(v, u, capacity=c)

    # Iterate through active nodes until the solution is found
    while next_nodes:
        cur = next_nodes.popitem(False)[0]

        while residual.nodes[cur]["excess"] > 0:
            cur_label = residual.nodes[cur]["distance_label"]
            # Find an admissible arc to push flow along
            admissible_arc = next(
                (
                    (u, v, c)
                    for (u, v, c) in residual.out_edges(cur, data="capacity")
                    if residual.nodes[v]["distance_label"] + 1 == cur_label and c > 0
                ),
                None,
            )
            if admissible_arc is not None:
                # Push flow along arc
                (u, v, c) = admissible_arc
                delta = min(residual.nodes[cur]["excess"], c)

                # Increase v's excess
                residual.nodes[v]["excess"] = residual.nodes[v].get("excess", 0) + delta
                # Decrease u's excess
                residual.nodes[u]["excess"] -= delta

                # Add v to active nodes list
                if v not in (source, sink):
                    next_nodes[v] = None

                # correct residual arcs
                residual.get_edge_data(u, v)["capacity"] -= delta
                if (v, u) in residual.edges:
                    residual.get_edge_data(v, u)["capacity"] += delta
                else:
                    residual.add_edge(v, u, capacity=delta)

            else:
                # If no such arc exists, relabel.
                residual.nodes[cur]["distance_label"] = min(
                    residual.nodes[v]["distance_label"] + 1
                    for (_u, v, c) in residual.out_edges(cur, data="capacity")
                    if c > 0
                )

                if residual.nodes[cur]["excess"] > 0:
                    next_nodes[cur] = None
                break

    flow_value = residual.nodes[sink]["excess"]
    flows = {}
    for (u, v, c) in residual.edges(data="capacity"):
        if (v, u) in network.edges:
            flows[(v, u)] = c

    return flow_value, flows


class TestPreflowPush(TestCase):
    """Tests for preflow_push"""

    def test_simple(self):
        """Test a simple graph for max-flow"""
        graph = nx.DiGraph()

        graph.add_edges_from(
            [
                (0, 1, {"capacity": 15}),
                (0, 3, {"capacity": 4}),
                (1, 2, {"capacity": 12}),
                (2, 3, {"capacity": 3}),
                (2, 5, {"capacity": 7}),
                (3, 4, {"capacity": 10}),
                (4, 1, {"capacity": 5}),
                (4, 5, {"capacity": 10}),
            ]
        )

        flow_value, flows = preflow_push(graph, 0, 5)

        self.assertEqual(flow_value, 14)
        self.assertEqual(
            flows,
            {
                (0, 1): 10,
                (0, 3): 4,
                (2, 3): 3,
                (1, 2): 10,
                (2, 5): 7,
                (4, 5): 7,
                (3, 4): 7,
            },
        )
