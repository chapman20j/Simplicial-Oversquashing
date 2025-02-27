# diagram_tree.py
"""
Code to produce diagrams of the tree task. 
"""
import matplotlib.pyplot as plt
import networkx as nx

from data.tree import generate_binary_tree_with_cycles, generate_binary_tree_with_lines

use_cycles = False


def make_graph(max_depth: int, cycles: bool):
    if cycles:
        edges, vertices_to_label = generate_binary_tree_with_cycles(max_depth)
    else:
        edges, vertices_to_label = generate_binary_tree_with_lines(max_depth)
    G = nx.Graph()
    G.add_edges_from(edges)
    return G, vertices_to_label


graph, vtl = make_graph(3, use_cycles)

# Make labels the corresponding letter of the alphabet
labels = {node: chr(64 + node) for node in graph.nodes()}
# make the nodes with vtl a different color
node_color = [
    "cornflowerblue" if node not in vtl else "green" for node in graph.nodes()
]
node_color[0] = "red"
# The graph looks like
#   cycle         cycle
#       \         /
#        E - A - B
#       /         \
#   cycle         cycle

if use_cycles:
    pos = {
        1: (0, 0),
        2: (1, 0),
        3: (2, 1),
        4: (2, -1),
        5: (-1, 0),
        6: (-2, 1),  # F
        7: (-2, -1),  # G
        8: (2.5, 1),
        9: (2.5, 1.5),
        10: (2, 1.5),  # J
        11: (2.5, -1),
        12: (2.5, -1.5),
        13: (2, -1.5),
        14: (-2.5, 1),  # N
        15: (-2.5, 1.5),
        16: (-2, 1.5),
        17: (-2.5, -1),
        18: (-2.5, -1.5),
        19: (-2, -1.5),
    }
else:
    pos = {
        1: (0, 0),
        2: (1, 0),
        3: (2, 1),
        4: (2, -1),
        5: (-1, 0),
        6: (-2, 1),  # F
        7: (-2, -1),  # G
        8: (2.5, 1.5),
        9: (3, 2),
        10: (2.5, -1.5),  # J
        11: (3, -2),
        12: (-2.5, 1.5),
        13: (-3, 2),
        14: (-2.5, -1.5),  # N
        15: (-3, -2),
        16: (-2, 1.5),
        17: (-2.5, -1),
        18: (-2.5, -1.5),
        19: (-2, -1.5),
    }

nx.draw(
    graph,
    pos,
    with_labels=True,
    labels=labels,
    node_size=500,
    node_color=node_color,
    font_size=10,
    font_color="black",
    font_weight="bold",
    edge_color="gray",
)

plt.tight_layout()
plt.show()
