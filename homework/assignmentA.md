# Assignment #A: Graph starts

Updated 1830 GMT+8 Apr 22, 2025

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

### M19943:图的拉普拉斯矩阵

OOP, implementation, http://cs101.openjudge.cn/practice/19943/

要求创建Graph, Vertex两个类，建图实现。

思路：



代码：

```python
class Vertex:
    def __init__(self):
        self.connection = set([])


class Graph:
    def __init__(self):
        self.vertexes = []

    def degrees(self):
        n = len(self.vertexes)
        deg = [0] * n
        for i in range(n):
            deg[i] = len(self.vertexes[i].connection)
        return deg

    def adjacency(self):
        n = len(self.vertexes)
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if self.vertexes[j] in self.vertexes[i].connection:
                    adj[i][j] = adj[j][i] = 1
        return adj

    def negative_adjacency(self):
        n = len(self.vertexes)
        neg_adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if self.vertexes[j] in self.vertexes[i].connection:
                    neg_adj[i][j] = neg_adj[j][i] = -1
        return neg_adj

    def laplacian(self):
        n = len(self.vertexes)
        neg_adj = self.negative_adjacency()
        deg = self.degrees()
        for i in range(n):
            neg_adj[i][i] += deg[i]
        return neg_adj


n, m = map(int, input().split())
graph = Graph()
for i in range(n):
    graph.vertexes.append(Vertex())

for i in range(m):
    a, b = map(int, input().split())
    graph.vertexes[a].connection.add(graph.vertexes[b])
    graph.vertexes[b].connection.add(graph.vertexes[a])

laplacian = graph.laplacian()
for i in range(n):
    print(*laplacian[i])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429230435003](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250429230435003.png)



### LC78.子集

backtracking, https://leetcode.cn/problems/subsets/

思路：



代码：

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums):
            if len(nums) == 1:
                return [[], [nums[0]]]
            sub_sets = dfs(nums[1:])
            return sub_sets + [[nums[0]] + sub_set for sub_set in sub_sets]
        return dfs(nums)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429230505602](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250429230505602.png)



### LC17.电话号码的字母组合

hash table, backtracking, https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

思路：



代码：

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        dic = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        def dfs(digits):
            if len(digits) == 0:
                return []
            if len(digits) == 1:
                return [char for char in dic[digits[0]]]
            prev_combines = dfs(digits[1:])
            combines = []
            for char in dic[digits[0]]:
                combines += [char + combine for combine in prev_combines]
            return combines
        return dfs(digits)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429230529067](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250429230529067.png)



### M04089:电话号码

trie, http://cs101.openjudge.cn/practice/04089/

思路：之前用字典树写过类似的代码所以感觉没有难度。一开始没有发现可能有重复的字符串，用node的children不为空判断为前缀出现了问题，还是要靠打标记。



代码：

```python
class Node:
    def __init__(self, val=None):
        self.val = val
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, text):
        node = self.root
        has_prefix = False
        for word in text:
            if word not in node.children:
                node.children[word] = Node(word)
            node = node.children[word]
            if node.is_end:
                has_prefix = True
        node.is_end = True
        return has_prefix


t = int(input())
for i in range(t):
    n = int(input())
    trie = Trie()
    has_prefix = False
    data = []
    for i in range(n):
        text = input()
        data.append(text)
    data.sort(key=lambda x: len(x))
    for text in data:
        if trie.insert(text):
            has_prefix = True
            break
    if has_prefix:
        print('NO')
    else:
        print('YES')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429230547400](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250429230547400.png)



### T28046:词梯

bfs, http://cs101.openjudge.cn/practice/28046/

思路：构建词桶是一个很有意思的想法，这使得建图很快，然后就是经典的bfs.

把queue打成了deque导致试了好几次都报错，写的太快导致的



代码：

```python
from collections import deque


def generate_graph(words):
    graph = {}
    for word in words:
        for i in range(4):
            pot = word[:i] + "_" + word[i+1:]
            if pot in graph:
                graph[pot].append(word)
            else:
                graph[pot] = [word]
    return graph


def bfs(start, end, graph):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        for i in range(4):
            pot = current[:i] + "_" + current[i+1:]
            for new_word in graph.get(pot, []):
                if new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, path + [new_word]))
    return None


n = int(input())
words = []
for i in range(n):
    words.append(input())
start, end = input().split()
graph = generate_graph(words)
path = bfs(start, end, graph)
if path:
    print(" ".join(path))
else:
    print("NO")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429230640214](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250429230640214.png)



### T51.N皇后

backtracking, https://leetcode.cn/problems/n-queens/

思路：学着用位棋盘写了一遍，当然考试的时候还是用集合更直观，避免细节出错。



代码：

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def placeQueen(row, columns, diagonalsL, diagonalsR, solution, solutions):
            if row == n:
                display = [["."] * n for _ in range(n)]
                for i in range(n):
                    display[i][solution[i]] = "Q"
                    display[i] = "".join(display[i])
                solutions.append(display)
                return
            
            for col in range(n):
                columnBit = 1 << col
                diagonalLBit = 1 << (row + col)
                diagonalRBit = 1 << (row - col + n - 1)
                
                if columns & columnBit or diagonalsL & diagonalLBit or diagonalsR & diagonalRBit:
                    continue
                newColumns = columns | columnBit
                newDiagonalsL = diagonalsL | diagonalLBit
                newDiagonalsR = diagonalsR | diagonalRBit
                
                solution.append(col)
                placeQueen(row + 1, newColumns, newDiagonalsL, newDiagonalsR, solution, solutions)
                solution.pop()
        
        solutions = []
        placeQueen(0, 0, 0, 0, [], solutions)
        return solutions
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429230659072](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250429230659072.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>



小北探索没有办法自己加知识库，于是按照讲义里的教程在本地粗略的实现了一个知识库，效果还不错。

目前在补每日选做，说实话很多题就是套了个壳，模板性很强，权当作保持熟练度。





