from collections import defaultdict, deque

class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbors = []  # [key]

class Graph:
    def __init__(self):
        self.vertices = {}  # {key: vertex}

def topological_sort(graph: Graph):
    in_degree = defaultdict(int)
    for u in graph.vertices.values():
        for v_key in u.neighbors:
            in_degree[v_key] += 1

    queue = deque()
    topo_order = []

    for u in graph.vertices.values():
        if in_degree[u.key] == 0:
            queue.append(u.key)

    while queue:
        u_key = queue.popleft()
        topo_order.append(u_key)
        for v_key in graph.vertices[u_key].neighbors:
            in_degree[v_key] -= 1
            if in_degree[v_key] == 0:
                queue.append(v_key)
    if len(topo_order) != len(graph.vertices):
        return
    return topo_order

if __name__ == "__main__":
    pass