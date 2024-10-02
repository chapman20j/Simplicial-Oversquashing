# curvature.py
"""
This file implements some curvature methods used to analyze the graphs. 

Adapted the following from https://github.com/jakubbober/discrete-curvature-rewiring/
    compute_curvature_1d
    compute_curvature_augmented
    compute_curvature_haantjes
    bfc_edge

https://github.com/hieubkvn123/revisiting-gnn-curvature/
    ollivier_ricci_edge
"""

import networkx as nx
import numpy as np
import ot
from networkx import Graph


def compute_curvature_1d(G: Graph, edge: tuple) -> int:
    return 4 - G.degree[edge[0]] - G.degree[edge[1]]


def compute_curvature_augmented(G: Graph, edge: tuple) -> int:
    v1_nbhd = set(G.neighbors(edge[0]))
    v2_nbhd = set(G.neighbors(edge[1]))
    triangles = v1_nbhd.intersection(v2_nbhd)
    return 4 - G.degree[edge[0]] - G.degree[edge[1]] + 3 * len(triangles)


def compute_curvature_haantjes(G: Graph, edge: tuple) -> int:
    v1_nbhd = set(G.neighbors(edge[0]))
    v2_nbhd = set(G.neighbors(edge[1]))
    triangles = v1_nbhd.intersection(v2_nbhd)
    return len(triangles)


def bfc_edge(G: Graph, edge: tuple) -> float:
    """Balanced Forman curvature computation for a given edge in a graph.

    Args:
        G: (undirected) graph under consideration.
        edge: edge under consideration.

    Returns:
        Balanced Forman curvature for the edge under consideration.
    """

    v1, v2 = edge

    deg1 = G.degree[v1]
    deg2 = G.degree[v2]
    deg_min = min(deg1, deg2)
    if deg_min == 1:
        return 0
    deg_max = max(deg1, deg2)

    S1_1 = set(G[v1])
    S1_2 = set(G[v2])

    triangles = S1_1.intersection(S1_2)
    squares_1 = set(
        k
        for k in S1_1.difference(S1_2)
        if k != v2 and set(G[k]).intersection(S1_2).difference(S1_1.union({v1}))
    )
    squares_2 = set(
        k
        for k in S1_2.difference(S1_1)
        if k != v1 and set(G[k]).intersection(S1_1).difference(S1_2.union({v2}))
    )
    ntri = len(triangles)
    ns1 = len(squares_1)
    ns2 = len(squares_2)

    out = 2 / deg1 + 2 / deg2 - 2 + 2 * ntri / deg_max + ntri / deg_min
    if ns1 == 0 or ns2 == 0:
        return out

    A = nx.adjacency_matrix(G).todense()

    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    idx1 = node_to_index[v1]
    idx2 = node_to_index[v2]
    s1_indices = [node_to_index[k] for k in squares_1]
    s2_indices = [node_to_index[k] for k in squares_2]

    A_v1 = A[idx1]
    A_v2 = A[idx2]

    # Compute gamma
    tmp1 = (A_v2 - np.multiply(A_v1, A_v2)).T
    gamma1 = np.max((A[s1_indices] @ tmp1), axis=0)
    tmp2 = (A_v1 - np.multiply(A_v2, A_v1)).T
    gamma2 = np.max((A[s2_indices] @ tmp2), axis=0)
    gamma = max(gamma1, gamma2) - 1

    return out + (ns1 + ns2) / (gamma * deg_max)


def ollivier_ricci_edge(G: Graph, edge: tuple, alpha=0.0, method="OTD") -> float:
    """Computes the Ollivier-Ricci curvature of an edge in a graph.

    Args:
        G: Graph
        edge: Edge (u, v) for which the curvature is to be computed.
        alpha: Mass remaining at the node. Defaults to 0.0.
        method: Optimal transport method. Defaults to "OTD".

    Raises:
        ValueError: Invalid optimal transport method.

    Returns:
        Ollivier-Ricci curvature of the edge.
    """

    u, v = edge
    # Assume the graph has edge weights under 'weight' attribute;
    # if not, weights are considered as 1.
    if not nx.get_edge_attributes(G, "weight"):
        weight = 1
    else:
        weight = G[u][v]["weight"]

    # Neighborhoods of u and v
    neighbors_u = list(G.neighbors(u))
    neighbors_v = list(G.neighbors(v))

    # Mass distributions at u and v
    mass_u = np.array(
        [alpha if x == u else (1 - alpha) / len(neighbors_u) for x in [u] + neighbors_u]
    )
    mass_v = np.array(
        [alpha if x == v else (1 - alpha) / len(neighbors_v) for x in [v] + neighbors_v]
    )

    # Cost matrix based on shortest path distances
    distances = np.array(
        [
            [
                nx.shortest_path_length(G, source=x, target=y, weight="weight")
                for y in [v] + neighbors_v
            ]
            for x in [u] + neighbors_u
        ]
    )

    # Compute optimal transport
    if method == "OTD":
        # Compute OTD using Earth Mover's Distance
        curvature, _ = ot.emd2(mass_u, mass_v, distances, log=True, return_matrix=True)
    else:
        raise ValueError("Unsupported method. Choose 'OTD'.")

    # Ricci curvature: 1 - transportation cost / edge weight
    ricci_curvature = 1 - curvature / weight

    return ricci_curvature


class Curvature:
    """Class to simplify curvature computation."""

    def __init__(
        self,
        curvature_method: str,
        curvature_kwargs: dict,
    ):
        self.curvature_method = curvature_method
        self.curvature_kwargs = curvature_kwargs  # typically empty

        if self.curvature_method == "1d":
            self.curvature_fn = compute_curvature_1d
        elif self.curvature_method == "augmented":
            self.curvature_fn = compute_curvature_augmented
        elif self.curvature_method == "haantjes":
            self.curvature_fn = compute_curvature_haantjes
        elif self.curvature_method == "bfc":
            self.curvature_fn = bfc_edge
        elif self.curvature_method == "or":
            self.curvature_fn = ollivier_ricci_edge
        else:
            raise ValueError(f"Invalid curvature method: {self.curvature_method}")

    def compute_curvature(self, G: Graph, edge: tuple):
        v1, v2 = edge
        return self.curvature_fn(G, (v1, v2), **self.curvature_kwargs)

    def compute_curvature_dict(self, G: Graph):
        curv_dict = {}
        for v1, v2 in G.edges():
            if v1 not in curv_dict:
                curv_dict[v1] = {}
            curv_dict[v1][v2] = self.compute_curvature(G, (v1, v2))
            if not G.is_directed():
                if v2 not in curv_dict:
                    curv_dict[v2] = {}
                curv_dict[v2][v1] = self.compute_curvature(G, (v2, v1))
        return curv_dict

    def compute_mean_curvature(self, G: Graph):
        d = self.compute_curvature_dict(G)
        out = 0
        for x in d.values():
            out += sum(x.values())
        return out / nx.number_of_edges(G)

    def compute_mean_curvature_randomized(self, G: Graph, iterations: int = 100):
        curvatures = []
        for _ in range(iterations):
            edge = np.random.choice(G.edges())
            curvatures.append(self.compute_curvature(G, edge))
        return np.mean(curvatures)
