# graph_lift.py
"""
Functions to test the impact of graph lifting on curvature.
This script produces
1. Edge level curvature plots for the original and lifted graph
2. Information about the graph and its lifted complex
"""

import argparse
import sys
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph
from torch_geometric.utils import from_networkx, to_networkx

from data.relational_structure import data_to_relational_graph
from utils.curvature import Curvature
from weighted_curvature import weighted_curvature


# * Datasets
def make_dumbbell(num_nodes_bell: int) -> Graph:
    """Makes a dumbbell graph with two complete graphs connected by an edge."""
    # Create two complete graphs
    G1 = nx.complete_graph(num_nodes_bell)
    G2 = nx.complete_graph(num_nodes_bell)

    # Combine the two complete graphs into a single graph
    dumbbell_graph = nx.disjoint_union(G1, G2)

    # Add a path (handle) between the two complete graphs
    handle_nodes = [
        num_nodes_bell - 1,
        num_nodes_bell,
    ]  # Connect the last node of G1 to the first node of G2
    dumbbell_graph.add_edge(*handle_nodes)
    return dumbbell_graph


def make_long_dumbbell(num_nodes_bell: int, bridge_len: int) -> Graph:
    """Creates a long dumbbell graph with two complete graphs connected
    by a path of length bridge_len.
    eg bridge_len = 2 -> K_n - * - * - K_n

    Args:
        num_nodes_bell: Number of nodes in each complete graph
        bridge_len: Number of nodes in the path connecting the two complete graphs

    Returns:
        Long dumbbell graph
    """
    # Create two complete graphs
    G1 = nx.complete_graph(num_nodes_bell)
    G2 = nx.complete_graph(num_nodes_bell)

    # Combine the two complete graphs into a single graph
    ld_graph = nx.disjoint_union(G1, G2)

    # Add a path (handle) between the two complete graphs
    path = (
        [num_nodes_bell - 1]
        + list(range(2 * num_nodes_bell, 2 * num_nodes_bell + bridge_len))
        + [num_nodes_bell]
    )

    # Create a new node
    tmp = len(path)
    for i in range(tmp - 1):
        ld_graph.add_edge(path[i], path[i + 1])

    return ld_graph


def ring_of_cliques_graph(cliques: int, nodes_per_clique: int):
    """Creates a ring of cliques graph"""
    # check cliques and nodes_per_clique are positive integers
    assert cliques > 0 and nodes_per_clique > 0

    # create ring of cliques graph
    G = nx.Graph()
    for i in range(cliques):
        # Create a fully connected subgraph (clique) for each community
        clique = nx.complete_graph(nodes_per_clique)
        # Relabel nodes to ensure unique node labels across communities
        mapping = {node: node + i * nodes_per_clique for node in clique.nodes()}
        clique = nx.relabel_nodes(clique, mapping)
        G = nx.compose(G, clique)

    # Add edges between cliques
    total_nodes = cliques * nodes_per_clique
    for i in range(cliques):
        target = ((i + 1) * nodes_per_clique) % total_nodes
        source = (target - 1) % total_nodes
        G.add_edge(source, target)
    return G


# * Plot positions
long_dumbbell_graph_pos = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1),
    4: (5, 5),
    5: (5, 6),
    6: (6, 5),
    7: (6, 6),
    8: (2, 2),
    9: (3, 3),
    10: (4, 4),
}

DELTA = 0.1
DELTA2 = 0.3
long_dumbbell_complex_pos = {
    0: (0, 0),
    1: (0.75, DELTA),
    2: (1.25, -DELTA),
    3: (2, 0),
    4: (6, 0),
    5: (6.75, DELTA),
    6: (7.25, -DELTA),
    7: (8, 0),
    8: (3, 0),
    9: (4, 0),
    10: (5, 0),
    11: ([0.58, 1 - 0.9 * DELTA2]),
    12: ([0, 1 + 0.1 * DELTA2]),
    13: ([0.85, 1 + 1.0 * DELTA2]),
    14: ([1, 1]),
    15: ([1.55, 1 - 0.6 * DELTA2]),
    16: ([2, 1 + 0.6 * DELTA2]),
    17: (2.5, 1),
    18: (6.58, 1 - 0.9 * DELTA2),  #
    19: (6, 1 + 0.1 * DELTA2),
    20: (6.85, 1 + 1.0 * DELTA2),
    21: (5.5, 1),
    22: (7.0, 1),  #
    23: (7.55, 1 - 0.6 * DELTA2),
    24: (8, 1 + 0.6 * DELTA2),
    25: (3.5, 1),
    26: (4.5, 1),
    27: (1.0, 3),
    28: (0, 2),
    29: (0.75, 2 + DELTA),
    30: (1.25, 2 - DELTA),
    31: (2, 2),
    32: (7.0, 3),
    33: (6, 2),
    34: (6.75, 2 + DELTA),
    35: (7.25, 2 - DELTA),
    36: (8, 2),
}


# * Plotting functions


def num_to_str(num):
    if num >= 0 and num < 26:
        return chr(num + 65)
    else:
        raise ValueError("Number out of range")


def plot_graph_curvature_before_after(
    orig_graph: Graph,
    new_graph: Graph,
    orig_curv_dict: dict,
    new_curv_dict: dict,
    simplicies: list,
    show_plots: bool,
    save_path: Optional[str],
    before_pos: Optional[dict] = None,
    after_pos: Optional[dict] = None,
):
    """Plots graph and complex with color-coded edge curvature

    Args:
        orig_graph: base graph
        new_graph: lifted graph
        orig_curv_dict: orig_graph curvature dictionary
        new_curv_dict: new_graph curvature dictionary
        simplicies: list of simplicies
        show_plots: shows plots if true
        save_path: saves figures if not None
        before_pos: positions for orig_graph. Defaults to None.
        after_pos: positions for new_graph. Defaults to None.
    """

    # Get edge weights for original graph
    orig_edge_weights = [orig_curv_dict[u][v] for u, v in orig_graph.edges()]

    # Get edge weights for rewired graph
    rewired_edge_weights = [new_curv_dict[u][v] for u, v in new_graph.edges()]

    # Determine the minimum and maximum edge weights for both graphs
    min_weight = min(min(orig_edge_weights, rewired_edge_weights))
    max_weight = max(max(orig_edge_weights, rewired_edge_weights))

    # Normalize the edge weights for both graphs
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)

    # Use the updated method to get the colormap
    cmap = plt.get_cmap("coolwarm")
    edge_color_orig = [cmap(norm(weight)) for weight in orig_edge_weights]
    edge_color_rewired = [cmap(norm(weight)) for weight in rewired_edge_weights]

    # Create the plot and the axis for the colorbar
    fig, (ax1, ax2, cax) = plt.subplots(
        nrows=1, ncols=3, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1, 0.05]}
    )

    # Draw the original graph with edge curvature
    node_labels = {
        node: "".join(map(num_to_str, simplicies[node])) for node in orig_graph.nodes()
    }
    nx.draw(
        orig_graph,
        pos=before_pos,
        with_labels=True,
        labels=node_labels,
        edge_color=edge_color_orig,
        edge_cmap=cmap,
        edge_vmin=min_weight,
        edge_vmax=max_weight,
        ax=ax1,
    )
    ax1.set_title("Graph")

    # Draw the rewired graph with edge curvature
    node_labels = {
        node: "".join(map(num_to_str, simplicies[node])) for node in new_graph.nodes()
    }
    nx.draw(
        new_graph,
        pos=after_pos,
        with_labels=True,
        labels=node_labels,
        edge_color=edge_color_rewired,
        edge_cmap=cmap,
        edge_vmin=min_weight,
        edge_vmax=max_weight,
        ax=ax2,
    )
    ax2.set_title("Complex")
    # Create the colorbar
    cbar = plt.colorbar(
        mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        label="Curvature",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "_curvature.png")
    if show_plots:
        plt.show()
    plt.clf()
    plt.close()


def edge_density(graph):
    return graph.number_of_edges() / (
        graph.number_of_nodes() * (graph.number_of_nodes() - 1)
    )


def curvature_experiment(
    graph_structure: str,
    curvature_type: str,
    complex_method: str = "clique",
    num_nodes: int = 4,
    num_cliques: int = 4,
    bridge_len: int = 3,
    max_dimension: int = 2,
    print_info: bool = True,
    show_plots: bool = True,
    save_path: str = None,
    before_pos: Optional[dict] = None,
    after_pos: Optional[dict] = None,
):
    """Plots graph before and after complex construction with curvature
    Prints basic graph information
    Produces KDE plot of curvature distribution

    Args:
        graph_structure: name of graph to use
        curvature_type: name of curvature method to use
        complex_method: complex method to use for lifting. Defaults to "clique".
        num_nodes: nodes in complete subgraphs. Defaults to 4.
        num_cliques: cliques in ring_of_cliques. Defaults to 4.
        bridge_len: path length in long_dumbbell. Defaults to 3.
        max_dimension: max cell dimension for graph lift. Defaults to 2.
        print_info: prints to file if False. Print to console otherwise. Defaults to True.
        show_plots: whether to show plots. Defaults to True.
        save_path: path to save figures. Defaults to None.
        before_pos: position for plotting original graph. Defaults to None.
        after_pos: position for plotting lifted graph. Defaults to None.

    Raises:
        ValueError: Invalid graph structure
    """
    if graph_structure == "complete":
        graph = nx.complete_graph(num_nodes)
    elif graph_structure == "dumbbell":
        graph = make_dumbbell(num_nodes)
    elif graph_structure == "long_dumbbell":
        graph = make_long_dumbbell(num_nodes, bridge_len)
    elif graph_structure == "ring_of_cliques":
        graph = ring_of_cliques_graph(num_cliques, num_nodes)
    else:
        raise ValueError("Invalid graph structure")

    data = from_networkx(graph)
    data.num_nodes = graph.number_of_nodes()
    sc, _, simplicies = data_to_relational_graph(
        data,
        method=complex_method,
        max_dimension=max_dimension,
    )
    complex_nx = to_networkx(sc).to_undirected()

    # Write these print statements to a file
    if print_info:
        outputfile = sys.stdout
    else:
        outputfile = open(save_path + "_info.txt", "w")

    curv_fn = Curvature(curvature_type, {})

    orig = curv_fn.compute_curvature_dict(graph)
    rewired = curv_fn.compute_curvature_dict(complex_nx)

    # * Print additional information
    print("Simplicies: ", file=outputfile)
    print(simplicies, file=outputfile)
    print(len(simplicies), file=outputfile)

    print("Number of Nodes:", file=outputfile)
    print(graph.number_of_nodes(), file=outputfile)
    print(complex_nx.number_of_nodes(), file=outputfile)

    print("Edge Density: ", file=outputfile)
    print(edge_density(graph), file=outputfile)
    print(edge_density(complex_nx), file=outputfile)

    print("Node Degrees: ", file=outputfile)
    degrees = complex_nx.degree()
    print(degrees, file=outputfile)

    print("Algebraic Connectivity: ", file=outputfile)
    print(nx.algebraic_connectivity(graph), file=outputfile)
    print(nx.algebraic_connectivity(complex_nx), file=outputfile)

    print("Weighted Curvature: ", file=outputfile)
    wcg, _ = weighted_curvature(graph)
    wcc, _ = weighted_curvature(complex_nx)
    print(wcg, file=outputfile)
    print(wcc, file=outputfile)

    # * Plot original and rewired graph with curvature
    plot_graph_curvature_before_after(
        graph,
        complex_nx,
        orig,
        rewired,
        simplicies,
        show_plots,
        save_path,
        before_pos,
        after_pos,
    )


if __name__ == "__main__":
    possible_curvature_methods = ["or", "rf", "bfc", "augmented", "1d", "haantjes"]
    possible_graphs = ["complete", "dumbbell", "long_dumbbell", "ring_of_cliques"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curvature",
        type=str,
        default="or",
        help=f"Curvature method. Options: {possible_curvature_methods}",
        choices=possible_curvature_methods,
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="complete",
        help=f"Graph structure. Options: {possible_graphs}",
        choices=possible_graphs,
    )
    args = parser.parse_args()

    if args.graph == "long_dumbbell":
        before_pos = long_dumbbell_graph_pos
        after_pos = long_dumbbell_complex_pos
    else:
        before_pos = None
        after_pos = None

    curvature_experiment(
        args.graph,
        args.curvature,
        complex_method="clique",
        print_info=True,
        show_plots=True,
        save_path=None,
        max_dimension=10,
        before_pos=before_pos,
        after_pos=after_pos,
    )
