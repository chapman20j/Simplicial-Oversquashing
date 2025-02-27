from copy import deepcopy

import networkx as nx
import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from data.relational_structure import (
    data_to_relational_graph,
    none_complex,
    propagate_features,
)


def generate_binary_tree_with_lines(max_depth: int) -> tuple[list, list]:
    r"""Creates a graph which is a binary tree with small line graphs off the leaves
    For example, a tree with max_depth=3 would look like:
             root
            /    \
           1      2
          / \    / \
         L1 L2  L3  L4
    where each Li is a line graph *-*-*
    eg 1-L1 looks like 1-*-*-T
    where T is a node considered for labeling

    Args:
        max_depth: depth of the binary tree

    Returns:
        Graph edges
        Nodes considered for labeling
    """
    edges = []
    node_counter = 1
    leaves = []
    line_end_nodes = []

    def helper(parent, depth):
        nonlocal node_counter
        if depth < max_depth:
            node_counter += 1
            left_child = node_counter
            edges.append((parent, left_child))
            helper(left_child, depth + 1)

            node_counter += 1
            right_child = node_counter
            edges.append((parent, right_child))
            helper(right_child, depth + 1)
        else:
            if parent != 1:
                leaves.append(parent)

    helper(1, 1)

    for leaf in leaves:
        node_counter += 1
        line_node1 = node_counter
        node_counter += 1
        line_node2 = node_counter

        edges.append((leaf, line_node1))
        edges.append((line_node1, line_node2))

        line_end_nodes.append(line_node2)

    return edges, line_end_nodes


def generate_binary_tree_with_cycles(max_depth: int) -> tuple[list, list]:
    r"""Creates a graph which is a binary tree with small cycle graphs off the leaves
    For example, a tree with max_depth=3 would look like:
             root
            /    \
           1      2
          / \    / \
         C1 C2  C3  C4
    where each Ci is a cycle graph with 4 nodes
                           *
                          / \
    eg 1-C1 looks like 1-*   T
                          \ /
                           *
    where T is a node considered for labeling
    
    Args:
        max_depth: depth of the binary tree

    Returns:
        Graph edges
        Nodes considered for labeling
    """
    edges = []
    node_counter = 1
    leaves = []
    cycle_nodes = []

    def helper(parent, depth):
        nonlocal node_counter
        if depth < max_depth:
            node_counter += 1
            left_child = node_counter
            edges.append((parent, left_child))
            helper(left_child, depth + 1)

            node_counter += 1
            right_child = node_counter
            edges.append((parent, right_child))
            helper(right_child, depth + 1)
        else:
            if parent != 1:
                leaves.append(parent)

    helper(1, 1)

    for leaf in leaves:
        node_counter += 1
        cycle_node1 = node_counter
        node_counter += 1
        cycle_node2 = node_counter
        node_counter += 1
        cycle_node3 = node_counter

        edges.append((leaf, cycle_node1))
        edges.append((cycle_node1, cycle_node2))
        edges.append((cycle_node2, cycle_node3))
        edges.append((cycle_node3, leaf))

        cycle_nodes.append(cycle_node2)

    return edges, cycle_nodes


def synthetic_trees(max_depth: int, cycles: bool) -> Data:
    """Creates a tree graph with lines or cycles at the leaves and returns a Data object

    Args:
        max_depth: Depth of the tree part
        cycles: Whether to use cycles or lines

    Returns:
        Graph stored in a Data object
    """
    if cycles:
        edges, vertices_to_label = generate_binary_tree_with_cycles(max_depth)
    else:
        edges, vertices_to_label = generate_binary_tree_with_lines(max_depth)
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edges)

    data = from_networkx(nx_graph)
    vertices_to_label = [x - 1 for x in vertices_to_label]

    root_vertex = 0
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

    data.x = vertex_features
    data.root_mask = root_mask
    data.y = y

    return data


@torch.no_grad()
def create_tree_dataset(
    max_depth: int,
    cycles: bool,
    sample_size: int,
    pretransform: str,
    init_method: str = "mean",
) -> list[Data]:
    """Creates the Tree dataset.

    Args:
        max_depth: Depth of the tree part.
        cycles: whether to use cycles.
        sample_size: dataset size.
        pretransform: graph lifting method to use. Defaults to "none".
        init_method: feature propagation method for graph lifting. Defaults to "mean".

    Raises:
        ValueError: Invalid graph lift

    Returns:
        Tree dataset
    """

    data_list = []
    for i in range(sample_size):
        data_list.append(synthetic_trees(max_depth, cycles))

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
