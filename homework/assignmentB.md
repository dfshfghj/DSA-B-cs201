# Assignment #B: 图为主

Updated 2223 GMT+8 Apr 29, 2025

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

### E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

思路：bfs模板



代码：

```python
from collections import deque

T = int(input())
dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]
for _ in range(T):
    R, C = map(int, input().split())

    mat = [list(input()) for _ in range(R)]
    for i in range(R):
        for j in range(C):
            if mat[i][j] == 'S':
                x0, y0 = i, j
            elif mat[i][j] == 'E':
                x1, y1 = i, j
    def bfs(mat, x0, y0, x1, y1):
        queue = deque([(x0, y0, 0)])
        visited = [[False] * C for _ in range(R)]
        while queue:
            x, y, time = queue.popleft()
            if x == x1 and y == y1:
                return time
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < R and 0 <= ny < C and mat[nx][ny] != '#' and not visited[nx][ny]:
                    queue.append((nx, ny, time + 1))
                    visited[nx][ny] = True
        return "oop!"
    
    print(bfs(mat, x0, y0, x1, y1))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527013140762](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250527013140762.png)



### M3532.针对图的路径存在性查询I

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

思路：节点分为若干块，各块之间互不连通，块内互相连通，只需要记录分界点，用二分查找确定所属的块



代码：

```python
class Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:
        part = []
        for i in range(n - 1):
            if nums[i + 1] - nums[i] > maxDiff:
                part.append(i)
        answer = []
        for query in queries:
            u, v = query
            answer.append(bisect_left(part, u) == bisect_left(part, v))

        return answer
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527013206140](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250527013206140.png)



### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：找到60%处的学生将其调至85分，其实就是直接解方程



代码：

```python
def transform(data, a):
    return [a*x + 1.1**(a*x) for x in data]

def check(data, a, target):
    new_data = transform(data, a)
    cnt = 0
    for x in new_data:
        if x >= target:
            cnt += 1
    return cnt < 0.6 * len(data)

def bisect_left(x, lo, hi, check):
    while lo < hi:
        mid = (lo + hi) // 2
        if check(mid, x):
            lo = mid + 1
        else:
            hi = mid
    return lo

data = list(map(float, input().split()))
data.sort()
print(bisect_left(85, 0, 1000000000, lambda mid, x: check(data, mid/1000000000, x)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527013225387](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250527013225387.png)



### Msy382: 有向图判环 

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：拓扑排序判环



代码：

```python
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

n, m = map(int, input().split())
graph = Graph()
for i in range(n):
    graph.vertices[i] = Vertex(i)
for _ in range(m):
    u_key, v_key = map(int, input().split())
    graph.vertices[u_key].neighbors.append(v_key)

if topological_sort(graph):
    print("No")
else:
    print("Yes")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527004217406](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250527004217406.png)



### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：每次dijkstra得到了全图上的单源最短路径，同一起点可以复用



代码：

```python
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

import sys
input = sys.stdin.read

data = input().split()
graph = Graph()
index = 0
p = int(data[index])
index += 1
for _ in range(p):
    graph.vertices[data[index]] = Vertex(data[index])
    index += 1
Q = int(data[index])
index += 1
for _ in range(Q):
    u_key, v_key, w = data[index], data[index+1], int(data[index+2])
    index += 3
    graph.vertices[u_key].neighbors.append((v_key, w))
    graph.vertices[v_key].neighbors.append((u_key, w))
R = int(data[index])
index += 1
pathes = {}
result = []
for _ in range(R):
    start_key, end_key = data[index], data[index+1]
    index += 2
    if start_key not in pathes:
        path = dijkstra(graph, graph.vertices[start_key])
        pathes[start_key] = path
    else:
        path = pathes[start_key]
    end_path = path[end_key]["path"]
    path_str = ""
    for v_key, weight in end_path:
        path_str += f"{v_key}->({weight})->"
    path_str += end_key
    result.append(path_str)
print("\n".join(result))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527013302565](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250527013302565.png)



### T28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：剪枝策略：

+ 对于奇数的$n$可以剪掉一半的格点（因为必须从黑格点出发）

+ 如果有孤立的格点立即剪枝

从第二条可以衍生出所谓的Warnsdorff启发式算法：先走最孤立的格点

将剪枝条件抽象为一个启发式函数，并用于指导搜索方向，即从0-1式的指导方式变为平滑函数式的指导，提供了更多的信息，A*亦是如此



代码：

```python
n = int(input())
sr, sc = map(int, input().split())

dx = [1, 2, -1, -2, 1, 2, -1, -2]
dy = [2, 1, -2, -1, -2, -1, 2, 1]

visited = [[False] * n for _ in range(n)]

def get_priority(x, y):
    priority = 8
    for i in range(8):
        nx = x + dx[i]
        ny = y + dy[i]
        if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny]:
            priority -= 1
    return priority

def dfs(x, y, depth):
    if depth == n**2:
        return True

    visited[x][y] = True

    for i in sorted(range(8), key=lambda i: get_priority(x + dx[i], y + dy[i]), reverse=True):
        nx = x + dx[i]
        ny = y + dy[i]

        if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny]:
            if dfs(nx, ny, depth + 1):
                return True
    visited[x][y] = False
    return False


if n % 2 == 1 and (sr + sc) % 2 == 1:
    print("fail")

else:
    if dfs(sr, sc, 1):
        print("success")
    else:
        print("fail")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527013320639](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250527013320639.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

五一假期不得空闲，cupt迫在眉睫，数算内容只得先放一放









