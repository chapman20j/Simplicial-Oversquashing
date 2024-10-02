# weighted_curvature.py
"""
This file:
1. defines the weighted curvature (WC) and Negative WC (NWC)
2. creates WC and NWC scatter plots for a dataset
3. Computes the curvature distribution for a dataset
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ray
import seaborn as sns
from ray.experimental import tqdm_ray
from sklearn.metrics import r2_score

import pretransform_datasets as pdata
from utils.curvature import Curvature


def weighted_curvature(
    graph: nx.Graph, curvature_method: str = "or"
) -> tuple[float, dict]:
    """Computes curvature statistics for a graph
        weighted curvature
        betweenness centrality
        weighted curvature
        negative weighted curvature

    Args:
        graph: graph
        curvature_method: curvature method to use. Defaults to "or".

    Returns:
        weighted curvature and other statistics
    """
    curv_fn = Curvature(curvature_method, {})
    curv_dict = curv_fn.compute_curvature_dict(graph)

    # Compute betweenness centrality
    bet_cen = nx.edge_betweenness_centrality(graph, weight="weight")

    # Compute weighted curvature
    mult = [curv_dict[u][v] * bet_cen[(u, v)] for u, v in graph.edges]
    neg = sum([x for x in mult if x > 0])
    info = {
        "curvature": [curv_dict[u][v] for u, v in graph.edges],
        "betweenness": [bet_cen[(u, v)] for u, v in graph.edges],
        "weighted_curvature": mult,
        "nwc": neg,
    }
    return sum(mult) * 2, info


@ray.remote
def ray_weighted_curvature(
    graph: nx.Graph, ray_pbar: tqdm_ray.tqdm, curvature_method: str
) -> tuple[float, dict]:
    """Computes curvature statistics for a graph using ray
        weighted curvature
        betweenness centrality
        weighted curvature
        negative weighted curvature

    Args:
        graph: graph
        ray_pbar: tqdm_ray progress bar
        curvature_method: curvature method to use. Defaults to "or".

    Returns:
        weighted curvature and other statistics
    """
    curv_fn = Curvature(curvature_method, {})
    curv_dict = curv_fn.compute_curvature_dict(graph)

    # Compute betweenness centrality
    bet_cen = nx.edge_betweenness_centrality(graph, weight="weight")

    ray_pbar.update.remote(1)

    # Compute weighted curvature
    mult = [curv_dict[u][v] * bet_cen[(u, v)] for u, v in graph.edges]
    neg = sum([x for x in mult if x < 0])
    info = {
        "curvature": [curv_dict[u][v] for u, v in graph.edges],
        "betweenness": [bet_cen[(u, v)] for u, v in graph.edges],
        "weighted_curvature": mult,
        "nwc": neg,
    }
    return sum(mult) * 2, info


def wc_scatter(
    graph_data: np.ndarray,
    complex_data: np.ndarray,
    save_path: str,
    name: str,
):
    """Function to make a scatter plot of graph data vs complex data
    Includes a linear fit and R^2 value
    Also includes a y=x line

    Args:
        graph_data: data for graphs
        complex_data: data for corresponding complexes
        save_path: path to save results
        name: name of the data collected
    """
    # Fit a line to this and get R^2
    coeff = np.polyfit(graph_data, complex_data, 1)
    r2 = r2_score(complex_data, np.polyval(coeff, graph_data))
    symb = "+" if coeff[1] > 0 else ""
    eqn_string = f"$y = {coeff[0]:.2f}x {symb} {coeff[1]:.2f}$     $R^2 = {r2:.2f}$"

    # Scatter plot with lines
    plt.rcParams.update({"font.size": 18})
    plt.scatter(graph_data, complex_data, color="blue", label="Data")
    plt.plot(
        graph_data,
        np.polyval(coeff, graph_data),
        color="aqua",
        label=eqn_string,
    )
    plt.plot(graph_data, graph_data, color="red", label="$y=x$")
    plt.xlabel("Graph")
    plt.ylabel("Complex")
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{name}_scatter.png"))
    plt.clf()

    # Save raw data
    np.save(os.path.join(save_path, f"{name}_graph.npy"), graph_data)
    np.save(os.path.join(save_path, f"{name}_complex.npy"), complex_data)
    print(f"Data for {name} scatter saved at {save_path}")


if __name__ == "__main__":
    COMPLEX_METHOD = "clique"

    # Make argparser
    parser = argparse.ArgumentParser(description="Weighted Curvature")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MUTAG",
        help="Base dataset to use",
    )
    parser.add_argument(
        "--curvature",
        type=str,
        default="or",
        help="Curvature method to use",
    )
    args = parser.parse_args()

    path = f"./results/weighted_curvature_{args.curvature}/{args.dataset}/"
    if not os.path.exists(path):
        os.makedirs(path)

    # * Load Data
    graph_dataset = pdata.get_tu_data(args.dataset, pdata.methods["none"], "none")
    complex_dataset = pdata.get_tu_data(
        args.dataset, pdata.methods[COMPLEX_METHOD], COMPLEX_METHOD
    )

    num_graphs = len(graph_dataset)

    # * Compute Curvature
    ray.init(num_cpus=5)
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    pbar = remote_tqdm.remote(total=len(graph_dataset))

    wc_graph_comp = [
        ray_weighted_curvature.remote(
            nx.from_edgelist(graph.edge_index.T.numpy()), pbar, args.curvature
        )
        for graph in graph_dataset
    ]
    full_wc_graph = ray.get(wc_graph_comp)
    pbar.close.remote()

    pbar = remote_tqdm.remote(total=len(graph_dataset))
    wc_complex_comp = [
        ray_weighted_curvature.remote(
            nx.from_edgelist(sc.edge_index.T.numpy()), pbar, args.curvature
        )
        for sc in complex_dataset
    ]
    full_wc_complex = ray.get(wc_complex_comp)
    pbar.close.remote()

    # Let the pbar finish
    time.sleep(0.2)

    ray.shutdown()

    # * Weighted Curvature Scatter Plot
    wc_graph = np.array([wc[0] for wc in full_wc_graph])
    wc_complex = np.array([wc[0] for wc in full_wc_complex])

    wc_scatter(wc_graph, wc_complex, path, "wc")

    # Again for negative
    neg_wc_graph = np.array([wc[1]["nwc"] for wc in full_wc_graph])
    neg_wc_complex = np.array([wc[1]["nwc"] for wc in full_wc_complex])

    wc_scatter(neg_wc_graph, neg_wc_complex, path, "neg_wc")

    # * Dataset Curvature Distribution
    # combine all the edge curvatures
    graph_summary = []
    complex_summary = []
    for i in range(num_graphs):
        graph_summary.extend(full_wc_graph[i][1]["curvature"])
        complex_summary.extend(full_wc_complex[i][1]["curvature"])

    # Plot the distribution
    plt.rcParams.update({"font.size": 18})
    sns.kdeplot(graph_summary, label="None", alpha=0.5, fill=True)
    sns.kdeplot(
        complex_summary,
        label=f"{COMPLEX_METHOD.capitalize()}",
        alpha=0.5,
        fill=True,
    )
    plt.legend(title="Lift")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "curvature_kde_summary.png"))
    plt.clf()
