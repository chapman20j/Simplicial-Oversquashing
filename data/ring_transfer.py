# ring_transfer.py
"""
Functions to create the RingTransfer dataset.
Adapted from https://github.com/twitter-research/cwn/
"""

from copy import deepcopy

import numpy as np
import torch
from torch_geometric.data import Data

from data.relational_structure import (
    data_to_relational_graph,
    none_complex,
    propagate_features,
)


def generate_ring_transfer_graph(nodes: int, target_label: np.array) -> Data:
    """Generates a ring graph with a target label."""
    opposite_node = nodes // 2

    # Initialise the feature matrix with a constant feature vector
    x = np.ones((nodes, len(target_label)))

    x[0, :] = 0.0
    x[opposite_node, :] = target_label
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []
    for i in range(nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

    # Add the edges that close the ring
    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])

    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a mask for the target node of the graph
    root_mask = torch.zeros(nodes, dtype=torch.bool)
    root_mask[0] = 1

    # Add the label of the graph as a graph label
    y = torch.tensor([np.argmax(target_label)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, root_mask=root_mask, y=y)


def generate_ring_transfer_graph_dataset(
    nodes: int,
    classes: int = 5,
    samples: int = 1000,
    pretransform: str = "none",
    init_method: str = "sum",
) -> list[Data]:
    """Creates the RingTransfer dataset.

    Args:
        nodes: size of the ring graph.
        classes: number of classes. Defaults to 5.
        samples: dataset size. Defaults to 1000.
        pretransform: graph lifting method to use. Defaults to "none".
        init_method: feature propagation method for graph lifting. Defaults to "sum".

    Raises:
        ValueError: Invalid graph lift

    Returns:
        RingTransfer dataset
    """
    # Generate the dataset
    data_list = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_ring_transfer_graph(nodes, target_class)
        data_list.append(graph)

    complex_list = []
    if pretransform == "none":
        return [none_complex(data) for data in data_list]
    elif pretransform == "clique":
        # Do the complex computation once
        first_complex, simplex_to_id, _ = data_to_relational_graph(
            data_list[0],
            pretransform,
            max_dimension=2,
        )
        # Get relevant information
        complex_num_nodes = first_complex.num_nodes
        edge_index = first_complex.edge_index.detach().clone()
        edge_type = first_complex.edge_type.detach().clone()
        lower = deepcopy(first_complex.lower_intersection)
        upper = deepcopy(first_complex.upper_union)
        for data in data_list:
            feat, cell_dimension = propagate_features(
                data.x, simplex_to_id, complex_num_nodes, init_method
            )
            rm = 0
            for i in range(len(data.root_mask)):
                if data.root_mask[i]:
                    rm = i
                    break
            root_mask = torch.zeros(complex_num_nodes, dtype=int)
            root_mask[rm] = 1
            complex_list.append(
                Data(
                    x=feat,
                    edge_index=edge_index.detach().clone(),
                    edge_type=edge_type.detach().clone(),
                    y=data.y,
                    cell_dimension=cell_dimension,
                    root_mask=root_mask.bool(),
                    lower_intersection=deepcopy(lower),
                    upper_union=deepcopy(upper),
                )
            )
    else:
        raise ValueError(f"Pretransform {pretransform} not found.")

    return complex_list
