# adjacency.py
"""
Defines Adjacency enum for consistent relation encoding. 
"""
from enum import Enum

NUM_ADJACENCIES: int = 5


class Adjacency(Enum):
    UPPER = 0
    LOWER = 1
    BOUNDARY = 2
    COBOUNDARY = 3
    REWIRE = 4


def str_to_adj_list(a: str):
    out = []
    if "b" in a:
        out.append(Adjacency.BOUNDARY)
    if "c" in a:
        out.append(Adjacency.COBOUNDARY)
    if "u" in a:
        out.append(Adjacency.UPPER)
    if "l" in a:
        out.append(Adjacency.LOWER)
    if "r" in a:
        out.append(Adjacency.REWIRE)
    return [x.value for x in out]
