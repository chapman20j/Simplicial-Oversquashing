# rgin.py
"""
RGINConv adapted from https://github.com/kedar2/FoSR
RGIN neural network based on pytorch benchmark
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
"""


import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GINConv
from torch_geometric.nn.inits import reset

from models.utils import get_pooling_fn, make_base_net


class RGINConv(torch.nn.Module):
    def __init__(
        self,
        eps,
        train_eps,
        input_dim,
        output_dim,
        num_relations,
    ):
        super(RGINConv, self).__init__()
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(input_dim, output_dim)
        convs = ModuleList()
        for _ in range(self.num_relations):
            convs.append(
                GINConv(
                    make_base_net(input_dim, output_dim), eps=eps, train_eps=train_eps
                )
            )
        self.convs = ModuleList(convs)

    def reset_parameters(self):
        reset(self.self_loop_conv)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type == i]
            x_new += conv(x, rel_edge_index)
        return x_new


class RGIN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_layers,
        hidden_dim,
        num_relations,
        dropout=0.5,
        train_eps: bool = True,
        pooling="mean",
    ):
        super().__init__()
        self.conv1 = RGINConv(
            eps=0,
            train_eps=train_eps,
            input_dim=num_features,
            output_dim=hidden_dim,
            num_relations=num_relations,
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                RGINConv(
                    eps=0,
                    train_eps=train_eps,
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_relations=num_relations,
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
        x = self.conv1(x, edge_index, edge_type)
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)

        x = self.pooling(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
