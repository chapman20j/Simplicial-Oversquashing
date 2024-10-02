# data/__init__.py
"""
Function for loading datasets.
Available datasets include:
    MUTAG
    ENZYMES
    PROTEINS
    IMDB-BINARY
    NCI1
    ringtransfer
    nmatch
"""

import torch

import pretransform_datasets as pdata
from data.nmatch import create_neighborsmatch_dataset, path_of_cliques
from data.ring_transfer import generate_ring_transfer_graph_dataset

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
        complex_mtd: Method for complex construction. One of "none" or "clique".

    Raises:
        ValueError: Invalid dataset name

    Returns:
        Processed dataset.
    """

    if name in num_classes_dict.keys():
        dataset = list(pdata.get_tu_data(name, pdata.methods[complex_mtd], complex_mtd))

        if name in ["IMDB-BINARY", "NCI1"]:
            for graph in dataset:
                n = graph.num_nodes
                graph.x = torch.ones((n, 1))

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
        G = path_of_cliques(num_cliques, clique_size)
        vertices_to_label = list(range(clique_size - 1))
        dataset = create_neighborsmatch_dataset(
            G, root_vertex, vertices_to_label, num_datapoints, complex_mtd
        )
    else:
        raise ValueError(f"Invalid dataset: {name}")

    return dataset
