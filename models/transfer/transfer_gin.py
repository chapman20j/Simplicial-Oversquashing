# transfer_gin.py
"""
Based on pytorch benchmark
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
Modified network head for graph transfer tasks. 
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from models.utils import make_base_net


class GIN(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, num_layers, hidden_dim, dropout, pooling="mean"
    ):
        super().__init__()
        self.conv1 = GINConv(make_base_net(num_features, hidden_dim), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(
                GINConv(make_base_net(hidden_dim, hidden_dim), train_eps=True)
            )
        self.convs.append(
            GINConv(make_base_net(hidden_dim, num_classes), train_eps=True)
        )
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        root_mask = data.root_mask
        x = F.relu(self.conv1(x, edge_index))
        n = len(self.convs)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < n - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x[root_mask]

    def __repr__(self):
        return self.__class__.__name__
