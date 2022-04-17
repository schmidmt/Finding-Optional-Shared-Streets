"""Basic utilities for working with this library"""

from collections import defaultdict
from itertools import combinations, islice
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool
from functools import partial
from pprint import pprint
from copy import deepcopy

from scipy.interpolate import UnivariateSpline
import networkx as nx
import geopandas as gpd
import numpy as np
import shapely
from tqdm import tqdm


# Moore et al (2013) safe breaking distances
MOORE_AFTER_BREAK_SPLINE = UnivariateSpline(
    [20, 30, 40, 50, 60, 70, 80, 90, 100],
    [3.9, 6, 11, 18, 27, 39, 54, 58, 84],
)
MOORE_BEFORE_BREAK_SPLINE = UnivariateSpline(
    [20, 30, 40, 50, 60, 70, 80, 90, 100],
    [6, 8, 11, 14, 17, 19, 22, 25, 28],
)

MOORE_SAFE_BREAKING_DISTANCE = lambda x: MOORE_AFTER_BREAK_SPLINE(
    x
) + MOORE_BEFORE_BREAK_SPLINE(x)


def capacity_moore(lanes: float, max_speed: float):
    """Maximum capacity for a road network based on
    "Maximum flow in road networks with speed-dependent capacities-application to Bangkok traffic", Moore et al, 2013
    """
    return 1000 * max_speed / MOORE_SAFE_BREAKING_DISTANCE(max_speed) * lanes


def drive_network_to_capacitated_network(G: nx.DiGraph, method=capacity_moore):
    """Add capacities to the network G using a particular method"""
    G = G.copy()
    for u, v, i in G.edges:
        edge_data = G.get_edge_data(u, v, i)
        raw_lanes = edge_data.get("lanes")
        if raw_lanes is None:
            lanes = 1
        elif isinstance(raw_lanes, str):
            lanes = int(raw_lanes) / 2  # TODO: Consider oneways
        elif isinstance(raw_lanes, list):
            lanes = sum(int(x) for x in raw_lanes) / 2
        else:
            raise ValueError(f"No condition for lanes as {type(raw_lanes)}")

        edge_data["capacity"] = method(lanes, edge_data["speed_kph"])

    return G


def grid_nodes(nodes: gpd.GeoDataFrame, n: int):
    nodes = nodes.copy()
    (x_left, y_bottom, x_right, y_top) = nodes.total_bounds
    rect_width = (x_right - x_left) / (n + 1)
    rect_height = (y_top - y_bottom) / (n + 1)

    nodes["x_idx"] = np.floor((nodes.x - x_left) / rect_width).astype(np.int32)
    nodes["y_idx"] = np.floor((nodes.y - y_bottom) / rect_height).astype(np.int32)
    nodes = nodes.reset_index()

    def frame_to_central_point(frame):
        x_idx = frame.iloc[0]["x_idx"]
        y_idx = frame.iloc[0]["y_idx"]
        bbox = shapely.geometry.Polygon(
            [
                (x_left + x_idx * rect_width, y_bottom + y_idx * rect_height),
                (x_left + (x_idx + 1) * rect_width, y_bottom + y_idx * rect_height),
                (
                    x_left + (x_idx + 1) * rect_width,
                    y_bottom + (y_idx + 1) * rect_height,
                ),
                (x_left + x_idx * rect_width, y_bottom + (y_idx + 1) * rect_height),
            ]
        )
        return frame.iloc[frame.distance(bbox.centroid).argmin()]

    return nodes.groupby(["x_idx", "y_idx"]).apply(frame_to_central_point)


def chunker(it, size):
    iterator = iter(it)
    while chunk := list(islice(iterator, size)):
        yield chunk


def all_pairs_max_flow(network: nx.DiGraph, nodes: List[int], pool: Optional[Pool]=None):
    """Compute the max flow between all pairs of nodes on the given network"""
    
    if pool is None:
        pool = Pool()

    max_flows = {}
    max_flow_values = {}

    n = len(nodes)
    its = chunker(combinations(nodes, 2), pool._processes)
    par = partial(_chunked_all_pairs_max_flow, network=network)
    for batch_values, batch_flows in tqdm(pool.imap_unordered(par, its), total=n * (n - 1) // 2 // pool._processes, desc="All pairs Max Flow"):
        max_flows.update(batch_flows)
        max_flow_values.update(batch_values)

    return (max_flow_values, max_flows)

def _chunked_all_pairs_max_flow(pairs: List[Tuple[int, int]], network: nx.DiGraph):
    max_flows = {}
    max_flow_values = {}
    
    for i, j in pairs:
        # make sure the pairs for max-flow are ordered, since max-flow is symmetric (TODO: Maybe?)
        i, j = min([i, j]), max([i, j])
        max_flow_value, max_flow = nx.maximum_flow(network, i, j)
        max_flows[(i, j)] = max_flow
        max_flow_values[(i, j)] = max_flow_value
        
    return (max_flow_values, max_flows)


def all_arc_removal_impact(network: nx.DiGraph, nodes: List[int], pool: Optional[Pool] = None):
    """
    Compute the impact of an arc removal on the max-flow between the given nodes
    """

    if pool is None:
        pool = Pool()

    unperturbed_flow_values, unperturned_flows = all_pairs_max_flow(
        network, nodes, pool
    )
    
    for i, j in combinations(nodes, 2):
        i, j = min([i, j]), max([i, j])
        assert (i, j) in unperturbed_flow_values
        assert (i, j) in unperturned_flows
        
    
    impacts = defaultdict(lambda: defaultdict(list))
    
    p_impact = partial(
            arc_removal_impact,
            network=network,
            nodes=nodes,
            unperturbed_flow_values=unperturbed_flow_values,
            unperturned_flows=unperturned_flows,
        )
    for node, impact in tqdm(pool.imap_unordered(p_impact, network.edges()), total=len(network.edges()), desc="Removing arcs and determining impact"):
        impacts[node] = impact

    return impacts


def arc_removal_impact(
    arc: (int, int),
    network: nx.DiGraph,
    nodes: List[int],
    unperturbed_flow_values,
    unperturned_flows,
):
    """ """
    u, v = arc
    impacts = []
    network = network.copy()
    edge_data = network.get_edge_data(u, v)
    edge_data["capacity"] = 0
    
    nodes_to_rerun = []
    
    for i, j in combinations(nodes, 2):
        # make sure the pairs for max-flow are ordered, since max-flow is symmetric (TODO: Maybe?)
        i, j = min([i, j]), max([i, j])
        if unperturned_flows[(i, j)][u][v] == 0:
            impacts.append(0)
        else:
            nodes_to_rerun.append((i, j))
            
    #perturbed_flow_values, _perturned_flows = all_pairs_max_flow(network, nodes_to_rerun)
    
    for i, j in nodes_to_rerun:
        max_flow_value, _max_flow = nx.maximum_flow(network, i, j)
        impacts.append(unperturbed_flow_values[(i, j)] - max_flow_value)
        
    return (u, v), impacts
