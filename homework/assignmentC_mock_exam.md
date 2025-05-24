# Assignment #C: 202505114 Mock Exam

Updated 1518 GMT+8 May 14, 2025

2025 spring, Complied by <mark>张景天 物理学院</mark>



> **说明：**
>
> 1. **⽉考**：AC?<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E06364: 牛的选举

http://cs101.openjudge.cn/practice/06364/

思路：签到题



代码：

```python
N, K = map(int, input().split())
nums = []
for i in range(N):
    a, b = map(int, input().split())
    nums.append((a, b, i+1))
nums.sort(key=lambda x: x[0], reverse=True)

print(max(nums[:K], key=lambda x: x[1])[2])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250514193817195](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250514193817195.png)



### M04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

思路：做过的题目



代码：

```python
from math import comb

n = int(input())

print(comb(2 * n, n) // (n + 1))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250514193838959](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250514193838959.png)



### M05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

思路：模拟



代码：

```python
from collections import deque
n = int(input())
cards = input().split()

queue_list = [deque([]) for _ in range(9)]
for i in range(n):
    queue_list[int(cards[i][-1]) - 1].append(cards[i])

for i in range(9):
    print(f"Queue{i+1}:{' '.join(list(queue_list[i]))}")

new_cards = []
for i in range(9):
    while queue_list[i]:
        new_cards.append(queue_list[i].popleft())

new_queue_list = [deque([]) for _ in range(4)]
dic = {'A':0, 'B':1, 'C':2, 'D':3}
chars = ['A', 'B', 'C', 'D']
for i in range(len(new_cards)):
    new_queue_list[dic[new_cards[i][0]]].append(new_cards[i])

for i in range(4):
    print(f"Queue{chars[i]}:{' '.join(list(new_queue_list[i]))}")

new_cards = []
for i in range(4):
    while new_queue_list[i]:
        new_cards.append(new_queue_list[i].popleft())
print(' '.join(new_cards))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250514193903518](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250514193903518.png)



### M04084: 拓扑排序

http://cs101.openjudge.cn/practice/04084/

思路：模板



代码：

```python
from collections import defaultdict
from heapq import heappush, heappop

def topological_sort(graph):
    indegree = defaultdict(int)
    result = []

    heap = []

    for u in graph:
        for v in graph[u]:
            indegree[v] += 1

    for u in graph:
        if indegree[u] == 0:
            heappush(heap, u)

    while heap:
        u = heappop(heap)
        result.append(u)

        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                heappush(heap, v)
    return result

n, a = map(int, input().split())
graph = defaultdict(list)
for i in range(1, n+1):
    graph[i] = []
for i in range(a):
    start, end = map(int, input().split())
    graph[start].append(end)

topo_order = topological_sort(graph)

print(" ".join([f"v{topo_order[i]}" for i in range(len(topo_order))]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250514193922235](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250514193922235.png)



### M07735:道路

Dijkstra, http://cs101.openjudge.cn/practice/07735/

思路：模板



代码：

```python
from heapq import heappush, heappop

class Vertex:
    def __init__(self, id):
        self.id = id
        self.connections = []

class Graph:
    def __init__(self):
        self.vertexes = {}

K = int(input())
N = int(input())
R = int(input())

graph = Graph()
for i in range(1, N + 1):
    graph.vertexes[i] = Vertex(i)

for _ in range(R):
    S, D, L, T = map(int, input().split())
    graph.vertexes[S].connections.append((graph.vertexes[D], L, T))

start = graph.vertexes[1]
end = graph.vertexes[N]

def bfs(start_vertex, end_vertex):
    INF = 2147483647
    dp = [dict() for _ in range(N + 1)]
    heap = [(0, 0, start_vertex.id)]

    while heap:
        curr_len, curr_cost, curr_id = heappop(heap)

        if curr_cost in dp[curr_id] and dp[curr_id][curr_cost] <= curr_len:
            continue
        dp[curr_id][curr_cost] = curr_len

        if curr_id == end.id:
            return curr_len

        for neighbor_vertex, length, toll in graph.vertexes[curr_id].connections:
            new_cost = curr_cost + toll
            new_len = curr_len + length

            if new_cost > K:
                continue

            if new_cost not in dp[neighbor_vertex.id] or dp[neighbor_vertex.id][new_cost] > new_len:
                heappush(heap, (new_len, new_cost, neighbor_vertex.id))

    return -1

result = bfs(start, end)
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250514193945195](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250514193945195.png)



### T24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

思路：层次遍历建树模板+及其简单的dp



代码：

```python
from collections import deque
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def generate_tree(nums):
    if not nums:
        return None
    root = Node(nums[0])
    queue = deque([root])
    i = 1
    while queue and i < len(nums):
        node = queue.popleft()
        node.left = Node(nums[i])
        queue.append(node.left)
        i += 1
        if i < len(nums):
            node.right = Node(nums[i])
            queue.append(node.right)
        i += 1
    return root

def max_value(root, mode):
    if not root:
        return 0
    
    if mode == 0:
        return max(max_value(root.left, 1), max_value(root.left, 0)) + max(max_value(root.right, 1), max_value(root.right, 0))
    else:
        return max_value(root.left, 0) + max_value(root.right, 0) + root.val
    
N = int(input())
nums = list(map(int, input().split()))
root = generate_tree(nums)
print(max(max_value(root, 0), max_value(root, 1)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250514194010484](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250514194010484.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

和别的课冲突了，一边上课一边倒着把后两题做掉了，其他看着还简单。基本都是模板题，不需要动脑子，尤其是最后一道题，我认为它与T的tag严重不符，用了约5分钟就做出来了，最难的反倒是拓扑排序，因为刚学不太熟练，如果期末考这样的话就轻松ak了。









