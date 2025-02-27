# nmatch.py
"""
Functions to create the NeighborsMatch dataset.

Originally from: https://github.com/kedar2/FoSR
Modified to handle efficient complex construction
"""

from copy import deepcopy

import networkx as nx
import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected

from data.relational_structure import (
    data_to_relational_graph,
    none_complex,
    propagate_features,
)


def path_of_cliques(num_cliques: int, size_of_clique: int) -> nx.Graph:
    """Creates path of cliques graph."""
    nx_graph = nx.Graph([])
    for i in range(num_cliques):
        for j in range(size_of_clique):
            for k in range(j):
                nx_graph.add_edge(i * size_of_clique + j, i * size_of_clique + k)
        if i != num_cliques - 1:
            nx_graph.add_edge((i + 1) * size_of_clique - 1, (i + 1) * size_of_clique)
    return nx_graph


def create_neighborsmatch_labels(
    nx_graph: nx.Graph, root_vertex: int, vertices_to_label: list[int]
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Generates a dataset for the neighborsmatch problem

    Args:
        nx_graph: base graph
        root_vertex: Vertex where classification is to be done
        vertices_to_label: Vertices considered for matching.

    Returns:
        vertex_features: node features
        y: label
        root_mask: Mask for the root node
    """
    num_classes = len(vertices_to_label)
    num_nodes = len(list(nx_graph.nodes))
    class_labels = torch.randperm(num_classes)
    vertex_index_list = []
    matching_entry = torch.randint(0, num_classes, ())
    for i in range(len(nx_graph.nodes)):
        if i in vertices_to_label:
            entry = class_labels[vertices_to_label.index(i)]
            if entry == matching_entry:
                y = i
        elif i == root_vertex:
            entry = matching_entry
        else:
            entry = torch.tensor(num_classes)
        vertex_index_list.append(entry)
    vertex_one_hot_tensor = one_hot(torch.stack(vertex_index_list))

    # encoding of node numbers, so the root can distinguish between them
    node_indictors = one_hot(torch.arange(len(nx_graph.nodes)))
    vertex_features = torch.concat([vertex_one_hot_tensor, node_indictors], dim=1).to(
        dtype=torch.float32
    )
    root_mask = torch.zeros(num_nodes, dtype=int)
    root_mask[root_vertex] = 1
    root_mask = root_mask.bool()
    return vertex_features, y, root_mask


@torch.no_grad()
def create_neighborsmatch_dataset(
    nx_graph: nx.Graph,
    root_vertex: int,
    vertices_to_label: list[int],
    sample_size: int,
    pretransform: str,
    init_method: str = "mean",
) -> list[Data]:
    """Creates a dataset for the neighborsmatch problem

    Args:
        nx_graph: base graph
        root_vertex: classification node
        vertices_to_label: nodes to consider for matching
        sample_size: dataset size
        pretransform: graph lift to use.
        init_method: feature propagation method for graph lift. Defaults to "mean".

    Raises:
        ValueError: Invalid graph lift

    Returns:
        Dataset for the neighborsmatch problem
    """
    data_list = []
    edge_index = from_networkx(nx_graph).edge_index
    edge_index = to_undirected(edge_index)
    for i in range(sample_size):
        x, y, root_mask = create_neighborsmatch_labels(
            nx_graph, root_vertex, vertices_to_label
        )
        data_list.append(Data(x=x, y=y, edge_index=edge_index, root_mask=root_mask))

    complex_list = []
    if pretransform == "none":
        return [none_complex(data) for data in data_list]
    elif pretransform in ["clique", "ring"]:
        # Do the complex computation once
        first_complex, simplex_to_id, simplices = data_to_relational_graph(
            data_list[0],
            pretransform,
            max_dimension=2,
        )
        # Get relevant information
        complex_num_nodes = first_complex.num_nodes
        simplex_to_id = {s: i for i, s in enumerate(simplices)}
        edge_index = first_complex.edge_index.clone()
        edge_type = first_complex.edge_type.clone()
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
                    edge_index=edge_index.clone(),
                    edge_type=edge_type.clone(),
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
