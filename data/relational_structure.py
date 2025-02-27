# relational_structure.py
"""
Functions for lifting graphs to simplicial complexes and then to relational graphs.
This file takes in graphs stored as torch_geometric Data objects.
A relational structure has the following attributes:
    x: Node features
    edge_index: Edge indices
    edge_type: Edge types
    cell_dimension: Dimension of the cell
    lower_intersection: Intersection of two lower adjacent cells (if in complex)
    upper_union: Union of two upper adjacent cells (if in complex)

Adapted from https://github.com/twitter-research/cwn/
"""


import networkx as nx
import torch
from gudhi import SimplexTree  # pylint: disable=no-name-in-module
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_undirected

from utils.adjacency import Adjacency


def none_complex(data: Data):
    """Returns the none-complex of a torch_geometric Data object."""
    if not hasattr(data, "x") or data.x is None:
        data.x = torch.ones((data.num_nodes, 1))
    data.cell_dimension = torch.zeros(data.num_nodes, dtype=torch.long)
    data.edge_type = (
        torch.zeros(data.num_edges, dtype=torch.long) + Adjacency.UPPER.value
    )
    # NOTE: Important that these are lists for graph batching.
    data.lower_intersection = []
    data.upper_union = []
    return data


def data_to_simplex_tree(data: Data) -> SimplexTree:
    """Constructs a simplex tree from a torch_geometric Data object."""

    st = SimplexTree()
    # Add vertices to the simplex.
    for v in range(data.num_nodes):
        st.insert([v], filtration=0.0)

    # Add edges to the simplex.
    for edge in data.edge_index.T:
        st.insert(edge.tolist(), filtration=1.0)

    return st


def get_simplex_to_id(data: Data, st: SimplexTree, max_dimension: int = 2):
    """Returns a mapping from simplices to their ids and a list of simplices.

    Args:
        data: The base graph.
        st: Simplex tree containing nodes and edges.
        max_dimension: Max cell dimension we want in the simplex tree. Defaults to 2.

    Returns:
        simplex_to_id: mapping from simplices to their ids
        simplices: list of simplices
    """
    simplex_to_id = {}
    simplices = []
    val = 0
    # Add nodes
    for i in range(data.num_nodes):
        fs = frozenset([i])
        simplex_to_id[fs] = val
        val += 1
        simplices.append(fs)

    # Add edges
    for i, edge in enumerate(data.edge_index.T):
        tmp = frozenset(edge.tolist())
        if tmp not in simplex_to_id:
            simplex_to_id[tmp] = val
            simplices.append(tmp)
            val += 1

    # Add higher-dimensional cells
    for i, simplex in enumerate(st.get_skeleton(max_dimension)):
        if len(simplex[0]) >= 3:
            fs = frozenset(simplex[0])
            if fs not in simplex_to_id:
                simplex_to_id[fs] = val
                simplices.append(fs)
                val += 1

    return simplex_to_id, simplices


def nontrivial_intersection(s1: frozenset, s2: frozenset) -> bool:
    """Returns true if the intersection of two sets is nontrivial."""
    return any(x in s1 for x in s2)


def compute_relations(
    st: SimplexTree, simplex_to_id: dict, max_dimension: int
) -> tuple[list, list, list, list]:
    """Computes simplicial complex adjacencies.

    Args:
        st: Simplex tree of the simplicial complex
        simplex_to_id: mapping from simplices to their ids
        max_dimension: Max dimension of cells in the complex

    Returns:
        edge_index: edges
        edge_type: edge types encoding relations
        lower_intersection: intersection of lower adjacent cells
        upper_union: union of upper adjacent cells
    """
    edge_index = []
    edge_type = []

    upper_union = []
    lower_intersection = []

    dim_maps = {i: [] for i in range(max_dimension + 1)}

    upper_adj = set()

    for cell in st.get_skeleton(max_dimension):
        fcell = frozenset(cell[0])
        ind1 = simplex_to_id[fcell]
        dim_maps[len(cell[0]) - 1].append(fcell)
        boundary = [frozenset(b[0]) for b in st.get_boundaries(cell[0])]
        for b in boundary:
            ind2 = simplex_to_id[b]

            # Add boundary
            edge_index.append([ind1, ind2])
            edge_type.append(Adjacency.BOUNDARY.value)

            # Add coboundary
            edge_index.append([ind2, ind1])
            edge_type.append(Adjacency.COBOUNDARY.value)

        # Add upper
        n = len(boundary)
        for i in range(n):
            for j in range(i + 1, n):
                s1 = simplex_to_id[boundary[i]]
                s2 = simplex_to_id[boundary[j]]
                fs = frozenset([s1, s2])
                if fs not in upper_adj:
                    upper_adj.add(fs)
                    edge_index.append([s1, s2])
                    edge_type.append(Adjacency.UPPER.value)

                    edge_index.append([s2, s1])
                    edge_type.append(Adjacency.UPPER.value)

                    upper_union.append(ind1)
                    upper_union.append(ind1)

    # Add lower adjacencies
    for dim in range(max_dimension):
        n = len(dim_maps[dim])

        for i in range(n - 1):
            for j in range(i + 1, n):
                s1 = dim_maps[dim][i]
                s2 = dim_maps[dim][j]
                if nontrivial_intersection(s1, s2):
                    sid1 = simplex_to_id[s1]
                    sid2 = simplex_to_id[s2]

                    edge_index.append([sid1, sid2])
                    edge_type.append(Adjacency.LOWER.value)

                    edge_index.append([sid2, sid1])
                    edge_type.append(Adjacency.LOWER.value)

                    intersection = s1.intersection(s2)
                    li = simplex_to_id[intersection]
                    lower_intersection.append(li)
                    lower_intersection.append(li)

    return (
        edge_index,
        edge_type,
        lower_intersection,
        upper_union,
    )


def propagate_features(
    graph_x: Tensor,
    simplex_to_id: dict,
    complex_num_nodes: int,
    init_method: str,
    max_dim: int = 2,
) -> tuple[Tensor, Tensor]:
    """Propagate features from lower-dimensional cells to higher-dimensional cells.

    Args:
        graph_x: graph node features
        simplex_to_id: mapping from simplices to their ids
        complex_num_nodes: nodes in the simplicial complex
        init_method: method to initialize higher-dimensional cell features
        max_dim: Max cell dimension to consider. Defaults to 2.

    Returns:
        complex_x: node features of the simplicial complex
        cell_dim: dimension of each cell
    """
    if init_method == "mean":
        init_fn = lambda x: torch.mean(x, dim=0, dtype=torch.float)
    elif init_method == "sum":
        init_fn = lambda x: torch.sum(x, dim=0, dtype=torch.float)
    else:
        raise ValueError(f"Method {init_method} not recognized")

    # Create node features
    glen = graph_x.size(1)
    complex_x = torch.zeros(complex_num_nodes, glen)
    cell_dim = torch.zeros(complex_num_nodes, dtype=torch.long)
    for simplex, sid in simplex_to_id.items():
        valid_indices = [idx for idx in simplex]
        if valid_indices:
            complex_x[sid] = init_fn(graph_x[valid_indices])
            cell_dim[sid] = min(len(simplex) - 1, max_dim)
    return complex_x, cell_dim


def data_to_relational_graph_clique(
    data: Data, max_dimension: int = 2
) -> tuple[Data, dict[frozenset, int], list[frozenset]]:
    """Computes the relational structure from clique complex lifting of a graph.

    Args:
        data: base graph
        max_dimension: Max cell dimension. Defaults to 2.

    Returns:
        relational structure: Data object containing the relational structure
        simplex_to_id: mapping from simplices to their ids
        simplices: list of simplices
    """
    st = data_to_simplex_tree(data)
    # Expand the simplex tree up to max_dimension
    st.expansion(max_dimension)

    simplex_to_id, simplices = get_simplex_to_id(data, st, max_dimension)
    edge_index, edge_type, lower_intersection, upper_union = compute_relations(
        st, simplex_to_id, max_dimension
    )
    num_simplices = len(simplices)
    if not hasattr(data, "x") or data.x is None:
        graph_x = torch.ones((data.num_nodes, 1))
    else:
        graph_x = data.x
    complex_x, cell_dim = propagate_features(
        graph_x, simplex_to_id, num_simplices, "mean", max_dim=max_dimension
    )
    return (
        Data(
            x=complex_x,
            edge_index=torch.tensor(edge_index).T,
            edge_type=torch.tensor(edge_type),
            y=data.y,
            cell_dimension=cell_dim,
            lower_intersection=lower_intersection,
            upper_union=upper_union,
        ),
        simplex_to_id,
        simplices,
    )


def get_rings(graph: Data, max_k=7, chordless=True):
    """Gets rings from a graph.

    Args:
        graph: graph to compute rings for
        max_k: Maximum ring size to consider. Defaults to 7.
        chordless: Whether to get chordless rings. Defaults to True.

    Returns:
        rings in the graph
    """
    # NOTE: Modified to use networkx now
    # NOTE: chordless only get rings that are chordless cycles

    if max_k == -1:
        max_k = graph.num_nodes

    graph_nx = to_networkx(graph, to_undirected=True, remove_self_loops=True)
    rings = []
    sorted_rings = set()

    if chordless:
        cycles = nx.algorithms.cycles.chordless_cycles(graph_nx, length_bound=max_k)
    else:
        cycles = nx.algorithms.cycles.simple_cycles(graph_nx, length_bound=max_k)
    # Sort the cycles by length
    cycles = sorted(cycles, key=len)

    for iso in cycles:
        tmp = frozenset(iso)
        if tmp not in sorted_rings:
            rings.append(iso)

    return rings


def compute_relations_ring(
    graph: Data,
    st: SimplexTree,
    simplex_to_id: dict,
    simplices: list,
    max_dimension: int,
    max_k: int,
    chordless: bool,
) -> tuple[list, list, list, list]:
    """Computes simplicial complex adjacencies for ring lift

    Args:
        graph: base graph
        st: Simplex tree of the simplicial complex
        simplex_to_id: mapping from simplices to their ids
        simplices: list of the simplices
        max_dimension: Max dimension of cells in the complex
        max_k: maximum ring size
        chordless: whether rings should be chordless

    Returns:
        edge_index: edges
        edge_type: edge types encoding relations
        lower_intersection: intersection of lower adjacent cells
        upper_union: union of upper adjacent cells
    """
    edge_index, edge_type, lower_intersection, upper_union = compute_relations(
        st, simplex_to_id, 2
    )

    rings = get_rings(graph, max_k, chordless)
    cob = dict()
    ring_sid = len(simplex_to_id)

    for ring in rings:
        n = len(ring)
        simplex_to_id[frozenset(ring)] = ring_sid
        simplices.append(frozenset(ring))

        boundary = []

        for i in range(len(ring)):
            e = frozenset([ring[i], ring[(i + 1) % n]])
            e_sid = simplex_to_id[e]

            # Boundary
            edge_index.append([ring_sid, e_sid])
            edge_type.append(Adjacency.BOUNDARY.value)

            # Co-boundary
            edge_index.append([e_sid, ring_sid])
            edge_type.append(Adjacency.COBOUNDARY.value)

            # Store Co-boundaries for lower computation
            if e_sid in cob:
                cob[e_sid].add(ring_sid)
            else:
                cob[e_sid] = {ring_sid}

            # Store boundary for upper computation
            boundary.append(e_sid)

        # Upper Adjacencies
        nbndry = len(boundary)
        for i in range(nbndry - 1):
            for j in range(i + 1, nbndry):
                edge_index.append([boundary[i], boundary[j]])
                edge_type.append(Adjacency.UPPER.value)
                edge_index.append([boundary[j], boundary[i]])
                edge_type.append(Adjacency.UPPER.value)
                upper_union.append(ring_sid)
                upper_union.append(ring_sid)

        ring_sid += 1

    # Lower Adjacencies
    new_edges = set()
    for e, rset in cob.items():
        # Make all connections
        rlist = list(rset)
        n = len(rlist)
        for i in range(n - 1):
            for j in range(i + 1, n):
                fs1 = frozenset([rlist[i], rlist[j]])
                fs2 = frozenset([rlist[j], rlist[i]])
                if fs1 not in new_edges:
                    new_edges.add(fs1)
                    lower_intersection.append(e)
                if fs2 not in new_edges:
                    new_edges.add(fs2)
                    lower_intersection.append(e)

    for edge in new_edges:
        e = list(edge)
        edge_index.append(e)
        edge_type.append(Adjacency.LOWER.value)

    return edge_index, edge_type, lower_intersection, upper_union


def data_to_relational_graph_ring(
    data: Data, max_dimension: int = 1, max_k=7, chordless: bool = True
) -> tuple[Data, dict[frozenset, int], list[frozenset]]:
    """Computes the relational structure from ring complex lifting of a graph.

    Args:
        data: base_graph
        max_dimension: Max cell dimension. Defaults to 1.
        max_k: maximum ring size. Defaults to 7.
        chordless: whether rings should be chordless. Defaults to True.

    Returns:
        relational structure: Data object containing the relational structure
        simplex_to_id: mapping from simplices to their ids
        simplices: list of simplices
    """
    st = data_to_simplex_tree(data)
    # Expand the simplex tree up to max_dimension
    st.expansion(1)

    simplex_to_id, simplices = get_simplex_to_id(data, st, 1)
    edge_index, edge_type, lower_intersection, upper_union = compute_relations_ring(
        data, st, simplex_to_id, simplices, 1, max_k, chordless
    )
    num_simplices = max(simplex_to_id.values()) + 1
    if not hasattr(data, "x") or data.x is None:
        graph_x = torch.ones((data.num_nodes, 1))
    else:
        graph_x = data.x
    complex_x, cell_dim = propagate_features(
        graph_x, simplex_to_id, num_simplices, "mean", max_dim=2
    )
    return (
        Data(
            x=complex_x,
            edge_index=torch.tensor(edge_index).T,
            edge_type=torch.tensor(edge_type),
            y=data.y,
            cell_dimension=cell_dim,
            lower_intersection=lower_intersection,
            upper_union=upper_union,
        ),
        simplex_to_id,
        simplices,
    )


def data_to_relational_graph(
    data: Data,
    method: str,
    max_dimension: int = 2,
) -> tuple[Data, dict[frozenset, int], list[frozenset]]:
    """Computes the relational structure from a torch_geometric Data object.

    Args:
        data: base graph
        method: Graph lifting method. One of "none", "clique", or "ring".
        max_dimension: Max cell dimension. Defaults to 2.

    Raises:
        ValueError: Invalid method

    Returns:
        relational structure: Data object containing the relational structure
        simplex_to_id: mapping from simplices to their ids
        simplices: list of simplices
    """
    data.edge_index = to_undirected(data.edge_index)
    if method == "none":
        d = none_complex(data)
        simplices = [frozenset([i]) for i in range(d.num_nodes)]
        simplex_to_id = {s: i for i, s in enumerate(simplices)}
        return d, simplex_to_id, simplices
    elif method == "clique":
        return data_to_relational_graph_clique(data, max_dimension)
    elif method == "ring":
        return data_to_relational_graph_ring(data, 1)
    else:
        raise ValueError(f"Method {method} not recognized")
