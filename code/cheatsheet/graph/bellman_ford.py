class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbors = []  # [(key, weight)]

class Graph:
    def __init__(self):
        self.vertices = {}  # {key: vertex}

def bellman_ford(graph: Graph, start: Vertex):
    distances = {key: float('inf') for key in graph.vertices}
    distances[start.key] = 0

    for _ in range(len(graph.vertices) - 1):
        for vertex in graph.vertices.values():
            for neighbor_key, weight in vertex.neighbors:
                if distances[vertex.key] + weight < distances[neighbor_key]:
                    distances[neighbor_key] = distances[vertex.key] + weight

    for vertex in graph.vertices.values():
        for neighbor_key, weight in vertex.neighbors:
            if distances[vertex.key] + weight < distances[neighbor_key]:
                return 
    return distances

if __name__ == '__main__':
    g = Graph()
    g.vertices = {
        'A': Vertex('A'),
        'B': Vertex('B'),
        'C': Vertex('C'),
        'D': Vertex('D'),
        'E': Vertex('E'),
        'F': Vertex('F'),
    }
    g.vertices['A'].neighbors = [('B', 6), ('C', 7)]
    g.vertices['B'].neighbors = [('D', 5), ('E', 8)]
    g.vertices['C'].neighbors = [('E', 5)]