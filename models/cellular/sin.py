# sin.py
"""
Based on https://github.com/twitter-research/cwn/
"""

import torch
import torch.nn.functional as F
from torch.nn import Identity, Linear, Module, ModuleList, Parameter

from models.base import Messenger, MultiDimensionalModule
from models.utils import get_pooling_fn, make_base_net
from utils.adjacency import str_to_adj_list


class SINConv(Module):

    def __init__(
        self,
        eps,
        train_eps,
        input_dim,
        hidden_dim,
        output_dim,
        num_relations,
        multi_dim: str = False,
        max_dim: int = 2,
        relations: str = "bur",
    ):
        super().__init__()
        self.num_relations = num_relations
        self.allowed_relations = str_to_adj_list(relations)

        self.message_nets = Messenger(
            num_relations=num_relations, message_fn=Identity, aggr="sum"
        )
        self.train_eps = train_eps
        self.eps_init = eps
        if train_eps:
            self.eps = Parameter(torch.empty(num_relations))
        else:
            self.eps = [eps] * num_relations

        self.rel_update_net = ModuleList()
        for i in range(num_relations):
            if multi_dim:
                self.rel_update_net.append(
                    MultiDimensionalModule(
                        net_fn=lambda: make_base_net(input_dim, hidden_dim),
                        max_dim=max_dim,
                    )
                )
            else:
                self.rel_update_net.append(make_base_net(input_dim, hidden_dim))

        if multi_dim:
            self.final_net = MultiDimensionalModule(
                net_fn=lambda: make_base_net(
                    hidden_dim * len(self.allowed_relations), output_dim
                ),
                max_dim=max_dim,
            )
        else:
            self.final_net = make_base_net(
                hidden_dim * len(self.allowed_relations), output_dim
            )
        self.multi_dim = multi_dim

        self.reset_parameters()

    def reset_parameters(self):
        self.message_nets.reset_parameters()
        if self.train_eps:
            # set to eps_init
            self.eps.data.fill_(self.eps_init)

    def forward(self, x, edge_index, edge_type, cell_dimensions):
        # Compute all messages for different relations
        msgs = self.message_nets(x, edge_index, edge_type)

        # Add (1+eps) * x to all messages
        for i in range(self.num_relations):
            msgs[i] += (1 + self.eps[i]) * x

        # Pass these into an MLP. Only include the allowed relations
        tmp = []
        for i in range(self.num_relations):
            if i in self.allowed_relations:
                if self.multi_dim:
                    tmp.append(self.rel_update_net[i](msgs[i], cell_dimensions))
                else:
                    tmp.append(self.rel_update_net[i](msgs[i]))

        # Concatenate all the messages
        num_append = len(self.allowed_relations) - len(tmp)
        for _ in range(num_append):
            # this happens with the base graph
            tmp.append(torch.zeros_like(tmp[0], device=tmp[0].device))
        out = torch.cat(tmp, dim=-1)

        # Return the final output
        if self.multi_dim:
            return self.final_net(out, cell_dimensions)

        return self.final_net(out)


class SIN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_layers,
        hidden_dim,
        num_relations,
        dropout,
        train_eps: bool = True,
        multi_dim: bool = False,
        max_dim: int = 2,
        pooling="mean",
    ):
        super().__init__()
        self.conv1 = SINConv(
            0,
            train_eps,
            num_features,
            hidden_dim,
            hidden_dim,
            num_relations,
            multi_dim,
            max_dim,
        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                SINConv(
                    0,
                    train_eps,
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    num_relations,
                    multi_dim,
                    max_dim,
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
        x = F.relu(self.conv1(x, edge_index, edge_type, cell_dimension))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_type, cell_dimension))

        x = self.pooling(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
