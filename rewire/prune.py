# prune.py
"""
This file implements edge pruning based on the maximum curvature. 
"""
from math import inf

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from utils.adjacency import Adjacency
from utils.curvature import Curvature


def dict_argmax(d: dict[dict[int, float]]):
    best_key = None
    best = -inf
    for k1, sub_dict in d.items():
        for k2, v in sub_dict.items():
            if v > best:
                best = v
                best_key = (k1, k2)
    return best_key


def prune_argmax(
    data: Data, rewire_iterations: int, curvature_mtd: str = "bfc"
) -> list:
    G = to_networkx(data)
    remove_list = []
    curvature = Curvature(curvature_mtd, {})

    for i in range(rewire_iterations):
        cd = curvature.compute_curvature_dict(G)
        # Get the dict argmax
        edge = dict_argmax(cd)
        # Remove this edge
        G.remove_edge(*edge)
        remove_list.append(edge)
    return remove_list


def prune_rewire(data: Data, rewire_iterations: int, curvature_mtd: str) -> Data:
    """Removes edges from the graph based on the maximum curvature."""
    # Prevent removing more edges than possible
    # or disconnecting the graph
    nnodes = data.num_nodes
    nedges = data.edge_index.shape[1]
    rewire_iterations = min(rewire_iterations, nedges // 2, nedges - nnodes)
    remove_list = prune_argmax(data, rewire_iterations, curvature_mtd)

    # Now we need to loop through and remove these from the data
    remove_inds = []
    remove_type = []
    for edge in remove_list:
        ind = (data.edge_index[0] == edge[0]) & (data.edge_index[1] == edge[1])
        ind = ind.nonzero(as_tuple=False).view(-1)
        remove_inds.append(ind)
        remove_type.append(data.edge_type[ind])
        ind = (data.edge_index[0] == edge[1]) & (data.edge_index[1] == edge[0])
        ind = ind.nonzero(as_tuple=False).view(-1)
        remove_inds.append(ind)
        remove_type.append(data.edge_type[ind])

    if len(remove_inds) == 0:
        return data

    # Convert to tensors
    remove_inds = torch.concatenate(remove_inds)
    remove_type = torch.concatenate(remove_type)

    # Figure out impact on lower_intersection and upper_union
    counts = torch.bincount(data.edge_type, minlength=5)
    out = torch.zeros_like(data.edge_type)
    tmp = data.edge_type == Adjacency.LOWER.value
    out[tmp] = torch.arange(counts[Adjacency.LOWER.value])
    tmp = data.edge_type == Adjacency.UPPER.value
    out[tmp] = torch.arange(counts[Adjacency.UPPER.value])

    lower_removes = remove_inds[remove_type == Adjacency.LOWER.value]
    lower_int_removes = out[lower_removes]
    lower_int_removes = set(lower_int_removes.tolist())
    upper_removes = remove_inds[remove_type == Adjacency.UPPER.value]
    upper_int_removes = out[upper_removes]
    upper_int_removes = set(upper_int_removes.tolist())

    # Remove the edges
    keep_inds = torch.ones(data.edge_index.size(1), dtype=torch.bool)
    keep_inds[remove_inds] = False

    data.edge_index = data.edge_index[:, keep_inds]
    data.edge_type = data.edge_type[keep_inds]

    # Update lower_intersection and upper_union
    if len(lower_int_removes) > 0:
        data.lower_intersection = [
            item
            for idx, item in enumerate(data.lower_intersection)
            if idx not in lower_int_removes
        ]
    if len(upper_int_removes) > 0:
        data.upper_union = [
            item
            for idx, item in enumerate(data.upper_union)
            if idx not in upper_int_removes
        ]

    return data
