# Assignment #D: 图 & 散列表

Updated 2042 GMT+8 May 20, 2025

2025 spring, Complied by <mark>张景天 物理学院</mark>



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### M17975: 用二次探查法建立散列表

http://cs101.openjudge.cn/practice/17975/

<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



思路：谁知道输入数据还能重复啊



代码：

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]

hash_list = [None] * m
pos_list = [None] * n
for i in range(n):
    num = num_list[i]
    pos = num % m

    if hash_list[pos] is None or hash_list[pos] == num:
        hash_list[pos] = num
        pos_list[i] = pos
    else:
        sgn = 1
        k = 1
        while True:
            current = (pos + sgn * k**2) % m
            if hash_list[current] is not None or hash_list[current] == num:
                sgn *= -1
                if sgn == 1:
                    k += 1
            else:
                hash_list[current] = num
                pos_list[i] = current
                break
print(" ".join([str(i) for i in pos_list]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250526135359862](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250526135359862.png)



### M01258: Agri-Net

MST, http://cs101.openjudge.cn/practice/01258/

思路：



代码：

```python
import sys
from heapq import *
input = sys.stdin.read

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

data = input().split()
index = 0
while index < len(data):
    N = int(data[index])
    index += 1
    graph = Graph()
    for i in range(N):
        graph.vertices[i] = Vertex(i)
    for i in range(N):
        for j in range(N):
            if int(data[index]) != 0:
                graph.vertices[i].neighbors.append((j, int(data[index])))
            index += 1

    mst, total_weight = prim(graph, graph.vertices[0])
    print(total_weight)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250526135510544](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250526135510544.png)



### M3552.网络传送门旅游

bfs, https://leetcode.cn/problems/grid-teleportation-traversal/

思路：0-1bfs

发现`visited`还得多剪枝，不如直接记录`d[x][y]`

代码：

```python
class Solution:
    def minMoves(self, matrix) -> int:
        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]
        m = len(matrix)
        n = len(matrix[0])
        traveraler = defaultdict(list)
        for i in range(m):
            for j in range(n):
                if matrix[i][j] not in ".#":
                    traveraler[matrix[i][j]].append((i, j))
        queue = deque([(0, 0, 0)])
        visited = [[False] * n for _ in range(m)]

        while queue:
            num, x, y = queue.popleft()

            if visited[x][y]:
                continue
            visited[x][y] = True

            if x == m-1 and y == n-1:
                return num

            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny] and matrix[nx][ny] != "#":
                    queue.append((num+1, nx, ny))

            for nx, ny in traveraler[matrix[x][y]]:
                if not visited[nx][ny]:
                    queue.appendleft((num, nx, ny))
            traveraler[matrix[x][y]] = []
        return -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250526135542684](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250526135542684.png)



### M787.K站中转内最便宜的航班

Bellman Ford, https://leetcode.cn/problems/cheapest-flights-within-k-stops/

思路：



代码：

```python
class Solution:
    def findCheapestPrice(self, n: int, flights, src: int, dst: int, k: int) -> int:
        prev_distance = [float("inf")] * n
        curr_distance = [float("inf")] * n
        prev_distance[src] = 0
        for i in range(1, k+2):
            for u, v, w in flights:
                curr_distance[v] = min(curr_distance[v], prev_distance[u] + w)
            prev_distance, curr_distance = curr_distance, curr_distance.copy()
        return curr_distance[dst] if curr_distance[dst] != float("inf") else -1
        
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250526135612845](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250526135612845.png)



### M03424: Candies

Dijkstra, http://cs101.openjudge.cn/practice/03424/

思路：一开始edges用的`set`被重边阻击了，现在换成`list`，希望以后不用再改了



代码：

```python
from heapq import *

class Vertex:
    def __init__(self, key):
        self.key = key
        self.edges = [] # [(key, weight)]

class Graph:
    def __init__(self):
        self.vertices = {} # {key: vertex}

def dijkstra(graph, start):
    heap = [(0, start.key)]
    distances = {v: float('inf') for v in graph.vertices}
    distances[start.key] = 0
    while heap:
        current_distance, current_key = heappop(heap)
        if current_distance > distances[current_key]:
            continue
        current_vertex = graph.vertices[current_key]
        for neighbor_key, weight in current_vertex.edges:
            distance = current_distance + weight
            if distance < distances[neighbor_key]:
                distances[neighbor_key] = distance
                heappush(heap, (distance, neighbor_key))
    return distances


N, M = map(int, input().split())
graph = Graph()
for i in range(N):
    graph.vertices[i+1] = Vertex(i+1)

for _ in range(M):
    a, b, c = map(int, input().split())
    graph.vertices[a].edges.append((b, c))

distances = dijkstra(graph, graph.vertices[1])
print(distances[N])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250526135634213](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250526135634213.png)



### M22508:最小奖金方案

topological order, http://cs101.openjudge.cn/practice/22508/

思路：拓扑排序模板



代码：

```python
from collections import defaultdict, deque

class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbors = {}  # {key: vertex}

class Graph:
    def __init__(self):
        self.vertices = {}  # {key: vertex}

def topological_sort(graph: Graph):
    in_degree = defaultdict(int)
    for u in graph.vertices.values():
        for v in u.neighbors.values():
            in_degree[v.key] += 1

    queue = deque()
    topo_order = []

    for u in graph.vertices.values():
        if in_degree[u.key] == 0:
            queue.append(u)

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in u.neighbors.values():
            in_degree[v.key] -= 1
            if in_degree[v.key] == 0:
                queue.append(v)

    return topo_order

n, m = map(int, input().split())
graph = Graph()
for i in range(n):
    graph.vertices[i] = Vertex(i)
for _ in range(m):
    a, b = map(int, input().split())
    graph.vertices[a].neighbors[b] = graph.vertices[b]

topo_order = topological_sort(graph)
nums = [100] * n
for i in range(len(topo_order) - 1, -1, -1):
    v = topo_order[i]
    for u in v.neighbors.values():
        if nums[v.key] <= nums[u.key]:
            nums[v.key] = nums[u.key] + 1
print(sum(nums))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250526135712739](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250526135712739.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

作业题基本都是模板，oop模板泛用性可能更强一点？(虽然代码量加了)









