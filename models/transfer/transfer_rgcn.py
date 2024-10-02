# transfer_rgcn.py
"""
RGCN based on pytorch benchmark
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
Modified network head for graph transfer tasks. 
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_layers,
        hidden_dim,
        num_relations,
        dropout,
    ):
        super().__init__()
        self.conv1 = RGCNConv(num_features, hidden_dim, num_relations)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        self.convs.append(RGCNConv(hidden_dim, num_classes, num_relations))
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_type = data.edge_type
        root_mask = data.root_mask
        x = F.relu(self.conv1(x, edge_index, edge_type))
        n = len(self.convs)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < n - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x[root_mask]

    def __repr__(self):
        return self.__class__.__name__
