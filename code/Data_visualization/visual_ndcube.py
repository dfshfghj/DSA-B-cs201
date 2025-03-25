import matplotlib.pyplot as plt

class Node:
    def __init__(self, x):
        self.x = x
        self.connections = []
    def connect(self, node):
        if node not in self.connections:
            self.connections.append(node)
            node.connections.append(self)
    def __repr__(self):
        return str(bin(self.x))[2:]
def bin2vec(n, dim):
    vec = [0] * dim
    b = str(bin(n))[2:]
    for i in range(-1, -len(b)-1, -1):
        vec[i] = int(b[i])
    return vec
def vec2bin(vec):
    return sum([2**(len(vec) - i - 1) * vec[i] for i in range(len(vec))])

def build_hypercube(dimension):
    total_nodes = 2 ** dimension
    nodes = [0] * total_nodes
    connections = [0] * (dimension * total_nodes // 2)
    for i in range(total_nodes):
        nodes[i] = Node(bin2vec(i, dimension))
    for i in range(total_nodes):
        for j in range(dimension):
            neighbor_id = i ^ (1 << j)
            nodes[i].connect(nodes[neighbor_id])
    
    return nodes

dimension = 3
hypercube = build_hypercube(dimension)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
for node in hypercube:
    print(*node.x)
    ax.scatter(*node.x)
    #print(f"{node}: {[n for n in node.connections]}")
plt.show()