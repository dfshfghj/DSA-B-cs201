# Assignment #6: 回溯、树、双向链表和哈希表

Updated 1526 GMT+8 Mar 22, 2025

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

### LC46.全排列

backtracking, https://leetcode.cn/problems/permutations/

思路：这样写要注意放入排列时要深拷贝



代码：

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        visited = [0] * len(nums)
        permute_list = []

        def backtracking(result, visited, permute_list):
            if len(result) == len(nums):
                permute_list.append(result[:])

            for i in range(len(nums)):
                if not visited[i]:
                    visited[i] = 1
                    result.append(nums[i])
                    backtracking(result, visited, permute_list)
                    result.pop()
                    visited[i] = 0
        
        backtracking([], visited, permute_list)
        return permute_list
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw6-leetcode-46](D:\dfshfghj\DSA-B-cs201\img\hw6-leetcode-46.png)



### LC79: 单词搜索

backtracking, https://leetcode.cn/problems/word-search/

思路：本来用`visited`数组的，后来ai告诉我只用将`board`改掉就可以了，有道理。



代码：

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]

        def backtracking(x, y, index):
            if index == len(word):
                return True
            if x < 0 or x >= m or y < 0 or y >= n or board[x][y] != word[index]:
                return False

            temp = board[x][y]
            board[x][y] = "#"

            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if backtracking(nx, ny, index + 1):
                    return True
            board[x][y] = temp
            return False
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if backtracking(i, j, 0):
                    return True

        return False
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw6-leetcode-79](D:\dfshfghj\DSA-B-cs201\img\hw6-leetcode-79.png)



### LC94.二叉树的中序遍历

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

思路：



代码：

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        
        def inorder(node):
            inorder_traversal = []
            if node:
                inorder_traversal += inorder(node.left)
                inorder_traversal += [node.val]
                inorder_traversal += inorder(node.right)
                return inorder_traversal
            else:
                return []
        
        return inorder(root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw6-leetcode-94](D:\dfshfghj\DSA-B-cs201\img\hw6-leetcode-94.png)



### LC102.二叉树的层序遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

思路：



代码：

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        result = []
        queue = deque([root])
        while queue:
            level = len(queue)
            current_level = []
            for i in range(level):
                node = queue.popleft()
                current_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(current_level)
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw6-leetcode-102](D:\dfshfghj\DSA-B-cs201\img\hw6-leetcode-102.png)



### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

思路：好久没写dp了有点手生



代码：

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        is_palindrome = [[True] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                is_palindrome[i][j] = (s[i] == s[j]) and is_palindrome[i + 1][j - 1]
        
        partition_result = []

        def backtracking(result, pos, partition_result):
            if pos == n:
                partition_result.append(result[:])
                return 
            for i in range(pos, n):
                if is_palindrome[pos][i]:
                    result.append(s[pos : i + 1])
                    backtracking(result, i + 1, partition_result)
                    result.pop()
        
        backtracking([], 0, partition_result)
        return partition_result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw6-leetcode-131](D:\dfshfghj\DSA-B-cs201\img\hw6-leetcode-131.png)



### LC146.LRU缓存

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

思路：之前用哈希表和链表手搓过一个lrucache，突然发现有`OrderedDict`，学了以下。



代码：

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw6-leetcode-146](D:\dfshfghj\DSA-B-cs201\img\hw6-leetcode-146.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

忙死了，要打CUPT还要准备期中考，数算先放一下，之后会补的。









