# gcn.py
"""
Based on pytorch benchmark
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

from models.utils import get_pooling_fn


class GCN(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, num_layers, hidden_dim, dropout, pooling="mean"
    ):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)
        self.pooling = get_pooling_fn(pooling)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        x = self.pooling(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
