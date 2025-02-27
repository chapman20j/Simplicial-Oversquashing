# transfer_cinpp.py
"""
Based on https://github.com/twitter-research/cwn/
Modified network head for graph transfer tasks. 
"""

import torch
import torch.nn.functional as F

from models.cellular.cinpp import CINppConv
from models.utils import batch_common_inds, get_pooling_fn


class CINpp(torch.nn.Module):
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
        self.conv1 = CINppConv(
            0, train_eps, num_features, hidden_dim, hidden_dim, multi_dim, max_dim
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                CINppConv(
                    0, train_eps, hidden_dim, hidden_dim, hidden_dim, multi_dim, max_dim
                )
            )
        self.convs.append(
            CINppConv(
                0, train_eps, hidden_dim, hidden_dim, num_classes, multi_dim, max_dim
            )
        )
        self.dropout = dropout
        self.pooling = get_pooling_fn(pooling)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_type = data.edge_type
        cell_dimension = data.cell_dimension
        root_mask = data.root_mask

        # * use batch to modify upper_ind
        if batch is None:
            lower_ind = data.lower_intersection
            upper_ind = data.upper_union
        else:
            lower_ind, upper_ind = batch_common_inds(
                data.lower_intersection, data.upper_union, batch
            )

        x = F.relu(
            self.conv1(x, edge_index, edge_type, upper_ind, lower_ind, cell_dimension)
        )
        n = len(self.convs)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, upper_ind, lower_ind, cell_dimension)

            if i < n - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x[root_mask]

    def __repr__(self):
        return self.__class__.__name__
