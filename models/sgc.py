# sgc.py
"""
The original SGC is training free but only for node-level tasks. 
We apply pooling to turn this into graph-level tasks.

Finally we end with 2 linear layers since we need some minimal training. 
Based on https://github.com/Tiiiger/SGC/blob/master/utils.py
"""

import torch
from torch.nn import Linear, Module

from models.utils import get_pooling_fn


class SGC(Module):

    def __init__(self, num_features, num_classes, num_layers, pooling="mean"):
        super(SGC, self).__init__()
        self.W = Linear(num_features, num_classes)
        self.num_layers = num_layers
        self.pooling = get_pooling_fn(pooling)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        nodes = x.shape[0]
        edge_index = torch.sparse_coo_tensor(
            edge_index, torch.ones(edge_index.shape[1], device=x.device), (nodes, nodes)
        )
        for _ in range(self.num_layers):
            x = torch.spmm(edge_index, x)
        x = self.pooling(x, batch)
        x = self.W(x)
        return x
