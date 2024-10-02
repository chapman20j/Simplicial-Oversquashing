# pre_transform.py
"""
This code performs clique construction. 
"""

import os

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from data.relational_structure import data_to_relational_graph


# Can modify this for custom dataset path
def get_dataset_path() -> str:
    return "/tmp/"


def none(data: Data) -> Data:
    """Converts a torch_geometric Data object to a complex."""
    return data_to_relational_graph(
        data,
        "none",
        max_dimension=2,
    )[0]


def clique(data: Data) -> Data:
    """Converts a torch_geometric Data object to a clique complex."""
    return data_to_relational_graph(
        data,
        "clique",
        max_dimension=2,
    )[0]


def get_tu_data(name, mtd, mtd_name) -> TUDataset:
    """Gets TUDataset with pre-transform method and saves it in memory.

    Args:
        name: Dataset of interest
        mtd: Complex construction method
        mtd_name: Complex construction method name

    Returns:
        Pre-transformed TUDataset
    """

    p = get_dataset_path()
    p = os.path.join(p, "TUDATASET", name, mtd_name)
    return TUDataset(
        root=p,
        name=name.upper(),
        pre_transform=mtd,
    )


datasets = [
    "MUTAG",
    "ENZYMES",
    "PROTEINS",
    "NCI1",
    "IMDB-BINARY",
]


methods = {
    "none": none,
    "clique": clique,
}


if __name__ == "__main__":

    for dt in datasets:
        print(dt)
        for method_name, method in methods.items():
            get_tu_data(dt, method, method_name)
