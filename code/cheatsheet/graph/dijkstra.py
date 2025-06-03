from heapq import *

class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbors = []  # [(key, weight)]

class Graph:
    def __init__(self):
        self.vertices = {}  # {key: vertex}

def dijkstra(graph: Graph, start: Vertex):
    path = {key: {'distance': float("inf"), 'path': []} for key in graph.vertices}
    path[start.key]['distance'] = 0
    heap = [(0, start.key)]
    while heap:
        current_distance, current_vertex_key = heappop(heap)

        for neighbor_key, weight in graph.vertices[current_vertex_key].neighbors:
            new_distance = current_distance + weight
            if new_distance < path[neighbor_key]['distance']:
                path[neighbor_key]['distance'] = new_distance
                path[neighbor_key]['path'] = path[current_vertex_key]['path'] + [(current_vertex_key, weight)]
                heappush(heap, (new_distance, neighbor_key))
    return path

if __name__ == "__main__":
    pass