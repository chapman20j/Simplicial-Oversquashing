# borf.py
"""
Adapted from  https://github.com/Weber-GeoML/AFRC_Rewiring/
"""

import networkx as nx
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from utils.adjacency import Adjacency

from .FormanRicci4 import FormanRicci4


def _preprocess_data(
    data: Data, is_undirected: bool = False
) -> tuple[nx.Graph, int, np.ndarray]:
    # Get necessary data information
    N = data.x.shape[0]
    m = data.edge_index.shape[1]

    # Compute the adjacency matrix
    if not "edge_type" in data.keys():
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type

    # Convert graph to Networkx
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()

    return G, N, edge_type


def _find_threshold(curv_vals: np.ndarray) -> float:
    """
    Model the curvature distribution with a mixture of two Gaussians.
    Find the midpoint between the means of the two Gaussians.
    """
    gmm = GaussianMixture(n_components=2, random_state=0).fit(curv_vals)

    mean1 = gmm.means_[0][0]
    std1 = np.sqrt(gmm.covariances_[0][0][0])

    mean2 = gmm.means_[1][0]
    std2 = np.sqrt(gmm.covariances_[1][0][0])

    threshold = (mean1 * std2 + mean2 * std1) / (std1 + std2)

    return (threshold, mean1, std1, mean2, std2)


# afrc-4 based rewiring
# Modified this to put all info into a separate edge_index and edge_type
def borf5(
    data: Data,
    loops: int = 10,
    remove_edges: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Preprocess data
    edge_index = data.edge_index
    G, N, edge_type = _preprocess_data(data)

    # Rewiring begins
    current_iteration = 0

    for _ in range(loops):
        afrc = FormanRicci4(G)
        afrc.compute_afrc_4()
        _C = sorted(afrc.G.edges, key=lambda x: afrc.G[x[0]][x[1]]["AFRC_4"])

        curvature_values = [afrc.G[edge[0]][edge[1]]["AFRC_4"] for edge in _C]

        # find the bounds
        if current_iteration == 0:
            lower_bound, mean1, std1, mean2, std2 = _find_threshold(
                np.array(curvature_values).reshape(-1, 1)
            )
            if mean1 > mean2:
                upper_bound = mean1 + std1
            else:
                upper_bound = mean2 + std2

        # Get top negative and positive curved edges
        most_pos_edges = [
            edge for edge in _C if afrc.G[edge[0]][edge[1]]["AFRC_4"] > upper_bound
        ]
        most_neg_edges = [
            edge for edge in _C if afrc.G[edge[0]][edge[1]]["AFRC_4"] < lower_bound
        ]
        # If no negative edges break
        if len(most_neg_edges) == 0:
            break

        # Remove edges
        if remove_edges:
            for u, v in most_pos_edges:
                if G.has_edge(u, v):
                    G.remove_edge(u, v)

        # Add edges
        for u, v in most_neg_edges:
            if len(set(G.neighbors(u)) - set(G.neighbors(v))) > 0:
                w = np.random.choice(list(set(G.neighbors(u)) - set(G.neighbors(v))))
                G.add_edge(v, w)
                edge_index = np.append(edge_index, [v, w])
                edge_type = np.append(edge_type, Adjacency.REWIRE.value)
                # add attributes "AFRC", "triangles", and "weight" to each added edge
                G[v][w]["AFRC"] = 0.0
                G[v][w]["triangles"] = 0
                G[v][w]["weight"] = 1.0
                G[v][w]["AFRC_4"] = 0.0
                G[v][w]["quadrangles"] = 0

            elif len(set(G.neighbors(v)) - set(G.neighbors(u))) > 0:
                w = np.random.choice(list(set(G.neighbors(v)) - set(G.neighbors(u))))
                G.add_edge(u, w)
                edge_index = np.append(edge_index, [u, w])
                edge_type = np.append(edge_type, Adjacency.REWIRE.value)
                # add attributes "AFRC", "triangles", and "weight" to each added edge
                G[u][w]["AFRC"] = 0.0
                G[u][w]["triangles"] = 0
                G[u][w]["weight"] = 1.0
                G[u][w]["AFRC_4"] = 0.0
                G[u][w]["quadrangles"] = 0

            else:
                pass

    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    if not torch.is_tensor(edge_type):
        edge_type = torch.tensor(edge_type, dtype=torch.long)

    return edge_index, edge_type
