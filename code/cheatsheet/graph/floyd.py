class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbors = []  # [(key, weight)]

class Graph:
    def __init__(self):
        self.vertices = {}  # {key: vertex}

def floyd(graph: Graph):
    vertices = graph.vertices.values()
    dist = {v.key: {u.key: float('inf') for u in vertices} for v in vertices}

    for v in graph.vertices.values():
        dist[v.key][v.key] = 0
        for u in v.neighbors:
            dist[v.key][u[0]] = u[1]

    for k in vertices:
        for i in vertices:
            for j in vertices:
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist