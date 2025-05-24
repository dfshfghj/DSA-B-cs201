# Assignment #9: Huffman, BST & Heap

Updated 1834 GMT+8 Apr 15, 2025

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

### LC222.完全二叉树的节点个数

dfs, https://leetcode.cn/problems/count-complete-tree-nodes/

思路：使用dfs，只是最后一层不用遍历了（因为是完全二叉树）



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        def compute_depth(node):
            depth = 0
            while node:
                node = node.left
                depth += 1
            return depth
        left_depth = compute_depth(root.left)
        right_depth = compute_depth(root.right)

        if left_depth == right_depth:
            return (2 ** left_depth) + Solution.countNodes(self, root.right)
        else:
            return (2 ** right_depth) + Solution.countNodes(self, root.left)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422235320720](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250422235320720.png)



### LC103.二叉树的锯齿形层序遍历

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

思路：就是在之前层序遍历的代码基础上小改一下，加一个反向



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        result = []
        queue = deque([root])
        left_to_right = True
        while queue:
            level_size = len(queue)
            level_nodes = deque()
            
            for _ in range(level_size):
                node = queue.popleft()
                if left_to_right:
                    level_nodes.append(node.val)
                else:
                    level_nodes.appendleft(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(list(level_nodes))
            left_to_right = not left_to_right
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422235337552](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250422235337552.png)



### M04080:Huffman编码树

greedy, http://cs101.openjudge.cn/practice/04080/

思路：每次从取出两个权值最小的节点，合并成一个新的节点



代码：

```python
import heapq

def huffman(n, weights):
    if n == 1:
        return weights[0]
    heapq.heapify(weights)
    
    total_cost = 0
    while len(weights) > 1:
        w1 = heapq.heappop(weights)
        w2 = heapq.heappop(weights)
        combined_weight = w1 + w2
        total_cost += combined_weight
        heapq.heappush(weights, combined_weight)
    return total_cost

n = int(input())
weights = list(map(int, input().split()))
print(huffman(n, weights))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422235353891](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250422235353891.png)



### M05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/

思路：一个建树，一个层次遍历，都是之前写过的，搬过来

问ai助教得到了`list(dict.fromkeys(nums))`的好操作



代码：

```python
from collections import deque

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def insert(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    elif val > root.val:
        root.right = insert(root.right, val)
    return root

def generate_tree(nums):
    root = None
    for num in nums:
        root = insert(root, num)
    return root

def level_order_traversal(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        result.append(str(node.val))
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

nums = list(map(int, input().strip().split()))
unique_nums = list(dict.fromkeys(nums))
root = generate_tree(unique_nums)
traversal = level_order_traversal(root)
print(" ".join(traversal))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422235409981](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250422235409981.png)



### M04078: 实现堆结构

手搓实现，http://cs101.openjudge.cn/practice/04078/

类似的题目是 晴问9.7: 向下调整构建大顶堆，https://sunnywhy.com/sfbj/9/7

思路：拿列表实现的思路可以参考python内置库的代码，但是用指针实现的思路目前还没有成功，主要是没有办法快速的找到最大的那个插入点，代码附在下面，还没有调通。



代码：

```python
class BinaryHeap:
    def __init__(self):
        self.heap = []

    def _siftdown(self, i):
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i

            if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest != i:
                self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
                i = smallest
            else:
                break

    def _siftup(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i] < self.heap[parent]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def heappush(self, item):
        self.heap.append(item)
        self._siftup(len(self.heap) - 1)

    def heappop(self):
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._siftdown(0)
        return root

heap = BinaryHeap()
n = int(input())
for i in range(n):
    cmd = list(map(int, input().split()))
    type = cmd[0]
    if type == 1:
        u = cmd[1]
        heap.heappush(u)
    elif type == 2:
        print(heap.heappop())
        
#----------------------------------------------------        
class HeapNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

    def __lt__(self, other):
        return self.val < other.val


class BinaryHeap:
    def __init__(self):
        self.root = None
        self.lastnode = None

    def _siftdown(self, node):
        while True:
            left = node.left
            right = node.right
            smallest = node

            if left and left.val < smallest.val:
                smallest = left
            if right and right.val < smallest.val:
                smallest = right
            if smallest != node:
                node.val, smallest.val = smallest.val, node.val
                node = smallest
            else:
                break

    def _siftup(self, node):
        while node.parent and node.val < node.parent.val:
            node.val, node.parent.val = node.parent.val, node.val
            node = node.parent

    def heappush(self, item):
        node = HeapNode(item)
        if self.root:
            self.lastnode.left = node
            node.parent = self.lastnode
            self._siftup(node)
            if node.val > self.lastnode.val:
                self.lastnode = node
        else:
            self.root = node
            self.lastnode = node

    def heappop(self):
        node = self.lastnode
        if self.root:
            if self.lastnode.parent:
                returnitem = self.root.val
                self.root.val = node.val
                self._siftdown(self.root)
                self.lastnode = self.lastnode.parent
                self.lastnode.left = self.lastnode.right = None
                return returnitem
            else:
                returnitem = self.root.val
                self.root = self.lastnode = None
                return returnitem
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422235428979](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250422235428979.png)



### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/

思路：怎么重复了



代码：

```python
import heapq

def huffman(n, weights):
    if n == 1:
        return weights[0]
    heapq.heapify(weights)
    
    total_cost = 0
    while len(weights) > 1:
        w1 = heapq.heappop(weights)
        w2 = heapq.heappop(weights)
        combined_weight = w1 + w2
        total_cost += combined_weight
        heapq.heappush(weights, combined_weight)
    return total_cost

n = int(input())
weights = list(map(int, input().split()))
print(huffman(n, weights))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422235353891](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250422235353891.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

发现树其实定式很多，本身不难，难点在于结合计概时期的greedy或dp.不过代码变长了细节需要格外在意，都集成成函数之后复用就可以避免重写一遍的随机错误。









