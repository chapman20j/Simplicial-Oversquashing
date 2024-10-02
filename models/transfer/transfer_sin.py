# transfer_sin.py
"""
Based on https://github.com/twitter-research/cwn/
Modified network head for graph transfer tasks. 
"""

import torch
import torch.nn.functional as F

from models.cellular.sin import SINConv


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
        for _ in range(num_layers - 1):
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
        self.convs.append(
            SINConv(
                0,
                train_eps,
                hidden_dim,
                hidden_dim,
                num_classes,
                num_relations,
                multi_dim,
                max_dim,
            )
        )
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_type = data.edge_type
        cell_dimension = data.cell_dimension
        root_mask = data.root_mask
        x = F.relu(self.conv1(x, edge_index, edge_type, cell_dimension))
        n = len(self.convs)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, cell_dimension)
            if i < n - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x[root_mask]

    def __repr__(self):
        return self.__class__.__name__
