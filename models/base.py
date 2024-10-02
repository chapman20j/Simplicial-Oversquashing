# base.py
""" 
Modules to simplify simplicial message passing
"""

from typing import Callable, Union

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, torch_sparse


class MultiDimensionalModule(torch.nn.Module):
    """This class allows for processing nodes of different dimensions separately.

    Args:
        module_fn: function to create a module for a given dimension
        max_dim: maximum dimension of cells in the complex.
    """

    def __init__(
        self,
        net_fn: Union[Callable, list[Callable]],
        max_dim: int,
    ):

        super(MultiDimensionalModule, self).__init__()

        self.max_dim = max_dim
        if isinstance(net_fn, list):
            self.net = torch.nn.ModuleList([net_fn[i]() for i in range(max_dim + 1)])
        else:
            self.net = torch.nn.ModuleList([net_fn() for _ in range(max_dim + 1)])

    def forward(
        self,
        x: Tensor,
        cell_dimensions: Tensor,
    ):
        out = None
        for i in range(self.max_dim + 1):
            mask = cell_dimensions == i
            tmp = self.net[i](x[mask])
            if out is None:
                out = torch.zeros(x.shape[0], tmp.shape[1], device=x.device)
            out[mask] = tmp
        return out


def masked_edge_index(edge_index: Adj, edge_mask: Tensor) -> Adj:
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout="coo")


class Messenger(MessagePassing):
    """Does message passing along different relations.

    Args:
        num_relations: Number of relations to consider.
        message_fn: Function to produce message modules.
        aggr: Aggregation method.
    """

    def __init__(
        self,
        num_relations: int,
        message_fn: Union[Callable, list[Callable]],
        aggr="sum",
    ):
        super(Messenger, self).__init__(aggr=aggr)
        self.num_relations = num_relations

        if isinstance(message_fn, list):
            self.convs = ModuleList([message_fn[i]() for i in range(num_relations)])
        else:
            self.convs = ModuleList([message_fn() for _ in range(num_relations)])

    def forward(self, x, edge_index, edge_type):
        msgs = []
        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)
            msgs.append(self.propagate(tmp, x=self.convs[i](x)))
        return msgs

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

    def reset_parameters(self):
        for conv in self.convs:
            reset(conv)
        return
