# cin.py
"""
Based on https://github.com/twitter-research/cwn/
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity, Linear, Module
from torch_geometric.nn import GINConv, MessagePassing
from torch_geometric.nn.inits import reset

from models.base import MultiDimensionalModule
from models.utils import get_pooling_fn, make_base_net
from utils.adjacency import Adjacency

from ..utils import batch_common_inds


class UpDownConv(MessagePassing):
    """Module that performs message passing when relations are UPPER or LOWER.
    This enables intersections and unions to be computed.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        eps: float = 0.0,
        train_eps: bool = False,
        multi_dimensional: bool = False,
    ):
        super(UpDownConv, self).__init__(aggr="sum")
        msg_fn = lambda: Linear(2 * input_dim, input_dim)
        update_fn = lambda: make_base_net(input_dim, output_dim)
        if multi_dimensional:
            self.msg_nn = MultiDimensionalModule(msg_fn, 2)
            self.update_nn = MultiDimensionalModule(update_fn, 2)
        else:
            self.msg_nn = msg_fn()
            self.update_nn = update_fn()
        self.multi_dimensional = multi_dimensional

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, x, edge_index, common_ind, cell_dimension):

        edge_attr = x[common_ind]

        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )
        if self.multi_dimensional:
            out = (1 + self.eps) * x + self.msg_nn(out, cell_dimension)
            return self.update_nn(out, cell_dimension)
        else:
            out = (1 + self.eps) * x + self.msg_nn(out)
            return self.update_nn(out)

    def reset_parameters(self):
        reset(self.msg_nn)
        reset(self.update_nn)
        self.eps.data.fill_(self.initial_eps)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_attr.shape[0] == 0:
            return torch.cat([x_j, torch.zeros_like(x_j)], dim=-1)
        return torch.cat([x_j, edge_attr], dim=-1)


class CINConv(Module):

    def __init__(
        self,
        eps,
        train_eps,
        input_dim,
        hidden_dim,
        output_dim,
        multi_dim: str = True,
        max_dim: int = 2,
    ):
        super(CINConv, self).__init__()
        self.boundary_net = GINConv(Identity(), eps=eps, train_eps=train_eps)
        self.rewire_net = GINConv(Identity(), eps=eps, train_eps=train_eps)
        update_fn = lambda: make_base_net(input_dim, hidden_dim)
        out_fn = lambda: Linear(3 * hidden_dim, output_dim)
        if multi_dim:
            self.boundary_feature_net = MultiDimensionalModule(update_fn, max_dim)
            self.rewire_feature_net = MultiDimensionalModule(update_fn, max_dim)
            self.out_mlp = MultiDimensionalModule(out_fn, max_dim)
        else:
            self.boundary_feature_net = update_fn()
            self.rewire_feature_net = update_fn()
            self.out_mlp = out_fn()

        self.upper_net = UpDownConv(input_dim, hidden_dim, eps, train_eps, multi_dim)
        self.multi_dim = multi_dim

    def forward(self, x, edge_index, edge_type, upper_ind, cell_dimension):
        # Boundary features
        boundary_edge_ind = edge_index[:, edge_type == Adjacency.BOUNDARY.value]
        boundary = self.boundary_net(x, boundary_edge_ind)
        if self.multi_dim:
            boundary = self.boundary_feature_net(boundary, cell_dimension)
        else:
            boundary = self.boundary_feature_net(boundary)

        # Rewire features
        rewire_edge_ind = edge_index[:, edge_type == Adjacency.REWIRE.value]
        rewire = self.rewire_net(x, rewire_edge_ind)
        if self.multi_dim:
            rewire = self.rewire_feature_net(rewire, cell_dimension)
        else:
            rewire = self.rewire_feature_net(rewire)

        # Upper features
        edge_type_mask = edge_type == Adjacency.UPPER.value
        upper_edge_ind = edge_index[:, edge_type_mask]
        upper = self.upper_net(x, upper_edge_ind, upper_ind, cell_dimension)

        comb = torch.cat([boundary, rewire, upper], dim=-1)

        if self.multi_dim:
            out = self.out_mlp(comb, cell_dimension)
        else:
            out = self.out_mlp(comb)

        return F.relu(out)


class CIN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_layers,
        hidden_dim,
        dropout,
        train_eps: bool = True,
        multi_dim: bool = False,
        max_dim: int = 2,
        pooling="mean",
    ):
        super().__init__()
        self.conv1 = CINConv(
            0, train_eps, num_features, hidden_dim, hidden_dim, multi_dim, max_dim
        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                CINConv(
                    0, train_eps, hidden_dim, hidden_dim, hidden_dim, multi_dim, max_dim
                )
            )
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)
        self.dropout = dropout
        self.pooling = get_pooling_fn(pooling)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_type = data.edge_type
        cell_dimension = data.cell_dimension

        # * use batch to modify upper_ind
        lower_ind, upper_ind = batch_common_inds(
            data.lower_intersection, data.upper_union, batch
        )

        x = F.relu(self.conv1(x, edge_index, edge_type, upper_ind, cell_dimension))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_type, upper_ind, cell_dimension))
        x = self.pooling(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
