from bellman_ford import Graph, Vertex, bellman_ford
from dijkstra import dijkstra

def johnson(graph: Graph):
    virtual_vertex = Vertex(-1)
    for vertex in graph.vertices.values():
        virtual_vertex.neighbors.append([vertex.key, 0])
    graph.vertices[-1] = virtual_vertex
    h = bellman_ford(graph, virtual_vertex)
    if h is None:
        return 
    for vertex in graph.vertices.values():
        for neighbor in vertex.neighbors:
            neighbor[1] += h[vertex.key] - h[neighbor[0]]
    full_distances = {}
    for v in graph.vertices.values():
        if v.key == -1:
            continue
        distances = dijkstra(graph, v)
        adjusted = {}
        for u, d in distances.items():
            d['distance'] = d['distance'] + h[v.key] - h[u]
            adjusted[u] = d
        full_distances[v.key] = adjusted
    return full_distances

