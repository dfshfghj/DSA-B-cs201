# Assignment #B: 图为主

Updated 2223 GMT+8 Apr 29, 2025

2025 spring, Complied by <mark>同学的姓名、院系</mark>



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





### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### Msy382: 有向图判环 

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





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





## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>











