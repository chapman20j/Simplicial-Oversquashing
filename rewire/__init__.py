# rewire/__init__.py
"""
Implements function to apply rewiring to a dataset.
"""
import torch
from torch_geometric.data import Data

from rewire import borf, fosr, prune, sdrf


def rewire(dataset: list[Data], rewiring_method: str, rewire_iterations: int):
    """Applies rewiring to a dataset

    Args:
        dataset: Torch geometric dataset
        rewiring_method: Rewiring method. One of "none", "fosr", "sdrf", "afr4"
        rewire_iterations: Number of rewiring iterations.

    Raises:
        ValueError: Invalid rewiring method.
    """

    if rewiring_method == "fosr":
        for i in range(len(dataset)):
            edge_index, edge_type, _ = fosr.edge_rewire(
                dataset[i].edge_index.numpy(),
                edge_type=dataset[i].edge_type,
                rewire_iterations=rewire_iterations,
            )
            dataset[i].edge_index = torch.tensor(edge_index)
            dataset[i].edge_type = torch.tensor(edge_type)
    elif rewiring_method == "sdrf":
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(
                dataset[i],
                loops=rewire_iterations,
                remove_edges=False,
                is_undirected=True,
            )
    elif rewiring_method == "none":
        return
    elif rewiring_method == "afr4":
        for i in range(len(dataset)):
            dataset[i].edge_index, dataset[i].edge_type = borf.borf5(
                dataset[i],
                loops=rewire_iterations,
                remove_edges=False,
            )
    elif rewiring_method == "prune":
        for i in range(len(dataset)):
            # NOTE: This modifies in place
            prune.prune_rewire(
                dataset[i],
                rewire_iterations=rewire_iterations,
                curvature_mtd="bfc",
            )
    elif rewiring_method == "prune1d":
        for i in range(len(dataset)):
            # NOTE: This modifies in place
            prune.prune_rewire(
                dataset[i],
                rewire_iterations=rewire_iterations,
                curvature_mtd="1d",
            )
    else:
        raise ValueError("Invalid rewiring method")
