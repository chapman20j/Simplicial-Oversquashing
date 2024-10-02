# utils.py
"""
Contains function for building MLPs, picking pooling functions, and 
batching lower_intersection and upper_union. 

Neural network from make_base_net inspired by
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py

"""
import torch
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import global_add_pool, global_mean_pool


def make_base_net(input_dim, output_dim):
    """Makes a 2 layer MLP"""
    return Sequential(
        Linear(input_dim, output_dim),
        ReLU(),
        BatchNorm1d(output_dim),
        Linear(output_dim, output_dim),
        ReLU(),
        BatchNorm1d(output_dim),
    )


def get_pooling_fn(pooling):
    """Gets the pooling function"""
    if pooling == "mean":
        return global_mean_pool
    elif pooling == "none":
        return lambda x, batch: x
    elif pooling == "sum":
        return global_add_pool
    else:
        raise ValueError(f"Pooling {pooling} not currently supported")


def batch_common_inds(
    lower_intersection,
    upper_union,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batches lower_intersection and upper_union

    Args:
        lower_intersection: intersection of lower adjacent cells
        upper_union: union of upper adjacent cells
        batch: batch indices

    Returns:
        batched lower_intersection
        batched upper_union
    """
    # count the number of nodes in each graph
    num_nodes = torch.bincount(batch)

    tot = 0
    lower_out = []
    upper_out = []
    ngraphs = len(num_nodes)
    for i in range(ngraphs):
        lower_out.append(tot + torch.tensor(lower_intersection[i], dtype=torch.long))
        upper_out.append(tot + torch.tensor(upper_union[i], dtype=torch.long))
        tot += num_nodes[i]

    return torch.concat(lower_out, dim=0), torch.concat(upper_out, dim=0)
