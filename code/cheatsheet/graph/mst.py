from heapq import *
from ..tree.union_find import UnionFind
class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbors = []  # [(key, weight)]

class Graph:
    def __init__(self):
        self.vertices = {}  # {key: vertex}

def prim(graph, start):
    visited = set() # {key}
    heap = [(0, None, start.key)]
    mst = [] # [(from, to, weight)]
    total_weight = 0

    while heap:
        weight, u_key, v_key = heappop(heap)
        if v_key in visited:
            continue
        visited.add(v_key)
        mst.append((u_key, v_key, weight))
        total_weight += weight

        v = graph.vertices[v_key]
        for neighbor_key, weight in v.neighbors:
            if neighbor_key not in visited:
                heappush(heap, (weight, v_key, neighbor_key))

    return mst, total_weight

def kruskal(graph):
    n = len(graph.vertices)
    edges = []

    for v in graph.vertices.values():
        for neighbor_key, weight in v.neighbors:
            edges.append((weight, v.key, neighbor_key))

    edges.sort()

    union_find = UnionFind(n)
    mst = [] # [(from, to, weight)]
    total_weight = 0

    for weight, u_key, v_key in edges:
        if union_find.find(u_key) != union_find.find(v_key):
            union_find.union(u_key, v_key)
            mst.append((u_key, v_key, weight))
            total_weight += weight
    
    return mst, total_weight

if __name__ == "__main__":
    pass