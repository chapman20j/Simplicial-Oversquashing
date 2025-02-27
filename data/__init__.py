# data/__init__.py
"""
Function for loading datasets.
Available datasets include:
TUDataset:
    MUTAG
    ENZYMES
    PROTEINS
    IMDB-BINARY
    NCI1

WebKB:
    TEXAS
    WISCONSIN
    CORNELL

Planetoid:
    CORA
    CITESEER

Molecular:
    Zinc

Synthetic:
    ringtransfer
    nmatch
    tree
"""

import torch

import pretransform_datasets as pdata
from data.nmatch import create_neighborsmatch_dataset, path_of_cliques
from data.ring_transfer import generate_ring_transfer_graph_dataset
from data.tree import create_tree_dataset

num_classes_dict = {
    "MUTAG": 2,
    "ENZYMES": 6,
    "PROTEINS": 2,
    "IMDB-BINARY": 2,
    "NCI1": 2,
}


def get_dataset(name, complex_mtd, **kwargs):
    """Function to preprocess and retrieve datasets.

    Args:
        name: Name of dataset
        complex_mtd: Method for complex construction. One of "none", "clique", or "ring"

    Raises:
        ValueError: Invalid dataset name

    Returns:
        Processed dataset.
    """

    if name in num_classes_dict:
        dataset = list(pdata.get_tu_data(name, pdata.methods[complex_mtd], complex_mtd))

        if name in ["IMDB-BINARY", "NCI1"]:
            for graph in dataset:
                n = graph.num_nodes
                graph.x = torch.ones((n, 1))
    elif name in pdata.webkb_datasets:
        dataset = list(
            pdata.get_webkb_data(name, pdata.methods[complex_mtd], complex_mtd)
        )
    elif name in pdata.planetoid_datasets:
        dataset = list(
            pdata.get_planetoid(name, pdata.methods[complex_mtd], complex_mtd)
        )
    elif "ZINC" in name:
        tmp = pdata.get_zinc(name, pdata.methods[complex_mtd], complex_mtd)
        dataset = [list(tmp[i]) for i in range(3)]

        for i in range(3):
            for graph in dataset[i]:
                graph.x = graph.x.to(torch.float32)
        dataset = tuple(dataset)
    elif name == "ringtransfer":
        num_classes = kwargs.get("num_classes", 5)
        num_datapoints = kwargs.get("num_datapoints", 1000)
        dataset = generate_ring_transfer_graph_dataset(
            kwargs["nodes"], num_classes, num_datapoints, complex_mtd, "sum"
        )
    elif name == "nmatch":
        num_datapoints = kwargs.get("num_datapoints", 1000)
        num_cliques = kwargs["num_cliques"]
        clique_size = kwargs["clique_size"]
        root_vertex = num_cliques * clique_size - 1
        nx_graph = path_of_cliques(num_cliques, clique_size)
        vertices_to_label = list(range(clique_size - 1))
        dataset = create_neighborsmatch_dataset(
            nx_graph, root_vertex, vertices_to_label, num_datapoints, complex_mtd
        )
    elif name == "tree":
        cycles = kwargs.get("cycles", False)
        max_depth = kwargs.get("max_depth", 3)
        dataset = create_tree_dataset(max_depth, cycles, 1000, complex_mtd, "mean")
    else:
        raise ValueError(f"Invalid dataset: {name}")

    return dataset
