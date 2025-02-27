# pre_transform.py
"""
This code performs clique construction. 
"""

import os
import time

from torch_geometric.data import Data
from torch_geometric.datasets import ZINC, Planetoid, TUDataset, WebKB

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


def ring(data: Data) -> Data:
    """Converts a torch_geometric Data object to a ring complex."""
    return data_to_relational_graph(
        data,
        "ring",
        max_dimension=1,
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


def get_zinc(name: str, mtd, mtd_name):
    subset = "FULL" not in name

    p = get_dataset_path()
    p = os.path.join(p, "ZINC", name, mtd_name)

    return (
        ZINC(root=p, subset=subset, split="train", pre_transform=mtd),
        ZINC(root=p, subset=subset, split="val", pre_transform=mtd),
        ZINC(root=p, subset=subset, split="test", pre_transform=mtd),
    )


def get_webkb_data(name, mtd, mtd_name):
    p = get_dataset_path()
    p = os.path.join(p, "WebKB", name, mtd_name)
    return WebKB(root=p, name=name, pre_transform=mtd)


def get_planetoid(name, mtd, mtd_name):
    p = get_dataset_path()
    p = os.path.join(p, "Planetoid", name, mtd_name)
    return Planetoid(root=p, name=name, pre_transform=mtd)


datasets = [
    "MUTAG",
    "ENZYMES",
    "PROTEINS",
    "NCI1",
    "IMDB-BINARY",
]
webkb_datasets = ["TEXAS", "WISCONSIN", "CORNELL"]
planetoid_datasets = ["CORA", "CITESEER"]

methods = {
    "none": none,
    "clique": clique,
    "ring": ring,
}


if __name__ == "__main__":

    time_dict = dict()

    for dt in datasets:
        print(dt)
        for method_name, method in methods.items():
            start = time.perf_counter()
            get_tu_data(dt, method, method_name)
            end = time.perf_counter()
            time_dict[(dt, method_name)] = end - start

    for dt in ["ZINC"]:
        print(dt)
        for mtd_name, mtd in methods.items():
            start = time.perf_counter()
            get_zinc(dt, mtd, mtd_name)
            end = time.perf_counter()
            time_dict[(dt, mtd_name)] = end - start

    print("webkb")
    for dt in webkb_datasets:
        print(dt)
        for mtd_name, mtd in methods.items():
            start = time.perf_counter()
            get_webkb_data(dt, mtd, mtd_name)
            end = time.perf_counter()
            time_dict[(dt, mtd_name)] = end - start

    print("planetoid")
    for dt in planetoid_datasets:
        print(dt)
        for mtd_name, mtd in methods.items():
            if dt == "CITESEER" and mtd_name == "ring":
                continue
            start = time.perf_counter()
            get_planetoid(dt, mtd, mtd_name)
            end = time.perf_counter()
            time_dict[(dt, mtd_name)] = end - start

    print(time_dict)
