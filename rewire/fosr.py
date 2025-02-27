# fosr.py
"""
Adapted from https://github.com/kedar2/FoSR
"""
from math import inf
from typing import Optional, Union

import numpy as np
import torch
from numba import jit

from utils.adjacency import Adjacency


@jit(nopython=True)
def choose_edge_to_add(
    x: np.ndarray, edge_index: np.ndarray, degrees: np.ndarray
) -> tuple[int, int]:
    # chooses edge (u, v) to add which minimizes y[u]*y[v]
    n = x.size
    m = edge_index.shape[1]
    y = x / ((degrees + 1) ** 0.5)
    products = np.outer(y, y)
    for i in range(m):
        u = edge_index[0, i]
        v = edge_index[1, i]
        products[u, v] = inf
    for i in range(n):
        products[i, i] = inf
    smallest_product = np.argmin(products)
    return (smallest_product % n, smallest_product // n)


@jit(nopython=True)
def compute_degrees(edge_index: np.ndarray, num_nodes: Optional[int] = None):
    # returns array of degrees of all nodes
    if num_nodes is None:
        num_nodes = np.max(edge_index) + 1
    degrees = np.zeros(num_nodes)
    m = edge_index.shape[1]
    for i in range(m):
        degrees[edge_index[0, i]] += 1
    return degrees


@jit(nopython=True)
def add_edge(edge_index: np.ndarray, u: int, v: int) -> np.ndarray:
    new_edge = np.array([[u, v], [v, u]])
    return np.concatenate((edge_index, new_edge), axis=1)


@jit(nopython=True)
def adj_matrix_multiply(edge_index: np.ndarray, x: np.ndarray) -> np.ndarray:
    # given an edge_index, computes Ax, where A is the corresponding adjacency matrix
    n = x.size
    y = np.zeros(n)
    m = edge_index.shape[1]
    for i in range(m):
        u = edge_index[0, i]
        v = edge_index[1, i]
        y[u] += x[v]
    return y


@jit(nopython=True)
def compute_spectral_gap(edge_index: np.ndarray, x: np.ndarray) -> float:
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    degrees = compute_degrees(edge_index, num_nodes=n)
    y = adj_matrix_multiply(edge_index, x / (degrees**0.5)) / (degrees**0.5)
    for i in range(n):
        if x[i] > 1e-9:
            return 1 - y[i] / x[i]
    return 0.0


@jit(nopython=True)
def _edge_rewire(
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    x: Optional[np.ndarray] = None,
    rewire_iterations: int = 50,
    initial_power_iters: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    if x is None:
        x = 2 * np.random.random(n) - 1
    degrees = compute_degrees(edge_index, num_nodes=n)
    for i in range(initial_power_iters):
        x = x - x.dot(degrees**0.5) * (degrees**0.5) / sum(degrees)
        y = x + adj_matrix_multiply(edge_index, x / (degrees**0.5)) / (degrees**0.5)
        x = y / np.linalg.norm(y)
    for I in range(rewire_iterations):
        i, j = choose_edge_to_add(x, edge_index, degrees=degrees)
        edge_index = add_edge(edge_index, i, j)
        degrees[i] += 1
        degrees[j] += 1
        edge_type = np.append(edge_type, Adjacency.REWIRE.value)
        edge_type = np.append(edge_type, Adjacency.REWIRE.value)
        x = x - x.dot(degrees**0.5) * (degrees**0.5) / sum(degrees)
        y = x + adj_matrix_multiply(edge_index, x / (degrees**0.5)) / (degrees**0.5)
        x = y / np.linalg.norm(y)
    return edge_index, edge_type, x


def edge_rewire(
    edge_index: np.ndarray,
    x: Optional[np.ndarray] = None,
    edge_type: Optional[Union[np.ndarray, torch.Tensor]] = None,
    rewire_iterations: int = 50,
    initial_power_iters: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = edge_index.shape[1]
    n = np.max(edge_index) + 1
    if x is None:
        x = 2 * np.random.random(n) - 1
    if edge_type is None:
        edge_type = np.zeros(m, dtype=np.int64)
    elif torch.is_tensor(edge_type):
        edge_type = edge_type.detach().cpu().numpy()
    return _edge_rewire(
        edge_index,
        edge_type=edge_type,
        x=x,
        rewire_iterations=rewire_iterations,
        initial_power_iters=initial_power_iters,
    )
