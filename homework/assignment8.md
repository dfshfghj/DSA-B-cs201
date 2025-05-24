# Assignment #8: 树为主

Updated 1704 GMT+8 Apr 8, 2025

2025 spring, Complied by <mark>张景天 物理学院</mark>



> **说明：**eb
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

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：递归地将数组拆成左右两半



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums: return None
        n = len(nums)
        root = TreeNode(nums[n//2])
        root.left = Solution.sortedArrayToBST(self, nums[:n//2])
        root.right = Solution.sortedArrayToBST(self, nums[n//2+1:])
        return root
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416235316809](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250416235316809.png)



### M27928:遍历树

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：看不懂题目，看了题解之后才明白要干什么



代码：

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.children = []
    
    def __lt__(self, other):
        return self.val < other.val

n = int(input())
is_root = {}
nodes = {}
for i in range(n):
    nums = list(map(int, input().split()))
    if nums[0] in nodes: node = nodes[nums[0]]
    else:
        node = Node(nums[0])
        nodes[nums[0]] = node
        is_root[node] = True

    for j in nums[1:]:
        if j in nodes: child = nodes[j]
        else:
            child = Node(j)
            nodes[j] = child
        is_root[child] = False
        nodes[nums[0]].children.append(child)

for node in is_root:
    if is_root[node]:
        root = node
        break

def dfs(root):
    if not root.children: print(root.val)
    else:
        for node in sorted([root] + root.children):
            if node == root: print(root.val)
            else: dfs(node)

dfs(root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416235253040](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250416235253040.png)



### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：好像就是普通的dfs?



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(pre, root):
            num = 0
            if root.left:
                num += dfs(pre*10 + root.val, root.left)
            if root.right:
                num += dfs(pre*10 + root.val, root.right)
            if not root.left and not root.right:
                return pre * 10 + root.val
            return num
        return dfs(0, root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416235051748](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250416235051748.png)



### M22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/22158/

思路：前序用来确定根节点，中序用来确定左右子树



代码：

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def generate_tree(pre, mid):
    if not pre:
        return None
    root = pre[0]
    mid_left, mid_right = mid.split(root)
    pre_left, pre_right = pre[1:len(mid_left)+1], pre[len(mid_left)+1:]
    root = Node(root)
    root.left = generate_tree(pre_left, mid_left)
    root.right = generate_tree(pre_right, mid_right)
    return root


def postfix(root):
    if root:
        return postfix(root.left) + postfix(root.right) + root.val
    else:
        return ""


while True:
    try:
        pre = input()
        mid = input()
        root = generate_tree(pre, mid)
        print(postfix(root))
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416235025738](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250416235025738.png)



### T24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：遇到(把当前节点压入stack，遇到)弹出节点，stack末尾的节点就是当前父节点



代码：

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.children = []


def extract(expr):
    stack = []
    node = None
    root = Node(expr[0])
    for i in expr:
        if i == "(":
            stack.append(node)
        elif i == ")":
            root = stack.pop()
        elif i not in " ,":
            node = Node(i)
            if stack:
                stack[-1].children.append(node)
    return root


def prefix(node):
    if node:
        pre = node.val
        for child in node.children:
            pre += prefix(child)
        return pre
    else:
        return ""


def postfix(node):
    if node:
        pre = ""
        for child in node.children:
            pre += postfix(child)
        return pre + node.val
    else:
        return ""


expr = input()
root = extract(expr)
print(prefix(root))
print(postfix(root))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416234959606](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250416234959606.png)



### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：用双链表找到要更新的前后两个值，在heap里_siftdown调整顺序就好了，然而heapq是用list实现的，空有指针没有索引也无济于事，后来发现可以用bisect查找出来，然而并没有调出来，于是参考了题解，leetcode还能用一些库，放到oj上都不敢想



代码：

```python
class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        pairs = SortedList()
        indexes = SortedList(range(len(nums)))
        count = 0

        for i, (x, y) in enumerate(pairwise(nums)):
            if x > y:
                count += 1
            pairs.add((x + y, i))

        ans = 0
        while count > 0:
            ans += 1

            s, i = pairs.pop(0)
            k = indexes.bisect_left(i)
            nxt = indexes[k + 1]
            if nums[i] > nums[nxt]:
                count -= 1
            if k > 0:
                pre = indexes[k - 1]
                if nums[pre] > nums[i]:
                    count -= 1
                if nums[pre] > s:
                    count += 1
                pairs.remove((nums[pre] + nums[i], pre))
                pairs.add((nums[pre] + s, pre))
            if k + 2 < len(indexes):
                nxt2 = indexes[k + 2]
                if nums[nxt] > nums[nxt2]:
                    count -= 1
                if s > nums[nxt2]:
                    count += 1
                pairs.remove((nums[nxt] + nums[nxt2], nxt))
                pairs.add((s + nums[nxt2], i))

            nums[i] = s
            indexes.remove(nxt)

        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250416234916372](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250416234916372.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

期中考试终于结束了，可以专心数算，打算试试leetcode周赛









