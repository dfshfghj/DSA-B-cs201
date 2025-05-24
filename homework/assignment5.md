# Assignment #5: 链表、栈、队列和归并排序

Updated 1348 GMT+8 Mar 17, 2025

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

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：从`mergeSort`里把`merge`照抄过来.



代码：

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        node_1, node_2 = list1, list2
        merge_list = ListNode()
        current_node = merge_list
        while node_1 and node_2:
            if node_1.val < node_2.val:
                current_node.next = node_1
                current_node = current_node.next
                node_1 = node_1.next
            else:
                current_node.next = node_2
                current_node = current_node.next
                node_2 = node_2.next
        if node_1 == None:
            current_node.next = node_2
        if node_2 == None:
            current_node.next = node_1
        return merge_list.next
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw5-leetcode-21](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw5-leetcode-21.png)



### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>

题解的破坏性的反转链表我感觉不太优雅，我用的是将后半段单链表扩展为双链表，这样至少没有破坏原有的结构。

代码：

```python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        fast, slow = head, head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        half = slow

        if fast.next:
            slow = slow.next
        slow.prev = None
        while slow.next:
            slow.next.prev = slow
            slow = slow.next
        
        start, end = head, slow
        is_palindrome = start.val == end.val
        while is_palindrome:
            if start.val != end.val:
                is_palindrome = False
            if start == half or not end.prev:
                break
            start = start.next
            end = end.prev
        return is_palindrome
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw5-leetcode-234](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw5-leetcode-234.png)



### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双链表实现。</mark>

模拟，没什么好说的.

代码：

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

class BrowserHistory:
    def __init__(self, homepage: str):
        self.page = Node(homepage)

    def visit(self, url: str) -> None:
        self.page.next = Node(url)
        self.page.next.prev = self.page
        self.page = self.page.next

    def back(self, steps: int) -> str:
        i = 0
        while self.page.prev and i < steps:
            self.page = self.page.prev
            i += 1
        return self.page.val
    def forward(self, steps: int) -> str:
        i = 0
        while self.page.next and i < steps:
            self.page = self.page.next
            i += 1
        return self.page.val


"""
class BrowserHistory:

    def __init__(self, homepage: str):
        self.back_stack = []
        self.forward_stack = []
        self.page = homepage

    def visit(self, url: str) -> None:
        self.back_stack.append(self.page)
        self.forward_stack = []
        self.page = url

    def back(self, steps: int) -> str:
        i = 0
        while self.back_stack and i < steps:
            self.forward_stack.append(self.page)
            self.page = self.back_stack.pop()
            i += 1
        return self.page

    def forward(self, steps: int) -> str:
        i = 0
        while self.forward_stack and i < steps:
            self.back_stack.append(self.page)
            self.page = self.forward_stack.pop()
            i += 1
        return self.page
"""
   
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw5-leetcode-1472](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw5-leetcode-1472.png)



### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：

从表达式建立语法树，然后对其进行后序遍历。使用`ast`会方便一些，主要是省去了很多繁杂的`if`判断。

如果要用`stack`的话，由递归与`stack`的等价性立即得到第二段代码

后来发现，这不就是Dijkstra的调度场算法吗.

> 直接根据递归与使用栈的等价性，是否就可以推导出调度场算法？
>
> 
>
> 假设我们有一个递归函数 `parse_expression`，它的逻辑如下：
>
> 1. 找到优先级最低的运算符。
> 2. 递归解析左子表达式。
> 3. 递归解析右子表达式。
> 4. 将当前运算符输出。
>
> 要将其转换为非递归形式，可以这样推导：
>
> 1. 使用一个栈来保存尚未处理的运算符（即操作符栈）。
> 2. 遍历表达式时，遇到操作数直接输出；遇到操作符则根据优先级决定是否压入栈中。
> 3. 遇到右括号时，弹出栈中的操作符，直到匹配到左括号。
> 4. 最后清空栈，确保所有操作符都被正确输出。
>
> 这正是调度场算法的核心逻辑。

感觉还是表达式树好想象，不过调度场算法可以作为一个板子背下来.

代码：

```python
#recursion
import ast

operator_to_str = {ast.Add: '+', 
                   ast.Sub: '-',
                   ast.Mult: '*',
                   ast.Div: '/'}
def postfix(node):
    if isinstance(node, ast.Constant):
        return str(node.value)
    elif isinstance(node, ast.BinOp):
        return f'{postfix(node.left)} {postfix(node.right)} {operator_to_str[type(node.op)]}'
    

n = int(input())
for i in range(n):
    expr = input()
    tree = ast.parse(expr, mode='eval')
    print(postfix(tree.body))
    
    
#stack
def infix_to_postfix(expression):
    precedence = {
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2
    }

    def is_operator(token):
        return token in precedence

    output = []
    operators = []
    i = 0
    while i < len(expression):
        char = expression[i]

        if char.isdigit() or char == '.':
            num = []
            while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
                num.append(expression[i])
                i += 1
            output.append(''.join(num))
            continue

        elif char == '(':
            operators.append(char)

        elif char == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()

        elif is_operator(char):
            while (operators and operators[-1] != '(' and
                   precedence.get(operators[-1], 0) >= precedence.get(char, 0)):
                output.append(operators.pop())
            operators.append(char)

        i += 1

    while operators:
        output.append(operators.pop())

    return ' '.join(output)

n = int(input())
for i in range(n):
    expr = input()
    postfix_expr = infix_to_postfix(expr)
    print(postfix_expr)
    
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw5-oj-24591](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw5-oj-24591.png)



### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>

`deque`模拟.

代码：

```python
from collections import deque
while True:
    n, p, m = map(int, input().split())
    if n == 0:
        break
    q = deque([_ for _ in range(1, n+1)])
    pop = []
    for i in range(1, p):
        q.append(q.popleft())
    while q:
        for i in range(1, m):
            q.append(q.popleft())
        pop.append(str(q.popleft()))
    print(','.join(pop))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw5-oj-03253](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw5-oj-03253.png)



### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：就是求逆序对数，本质上是`sort`.背`merge sort`的板子.赶超事件发生在`merge`中.

ps.被静态检查报了好几次`Compile Error`

代码：

```python
# pylint: skip-file

def mergeSort(array):
    if len(array) == 1:
        return array
    mid = len(array) // 2
    return merge(mergeSort(array[:mid]), mergeSort(array[mid:]))

def merge(array_1, array_2):
    global count
    i, j = 0, 0
    merge_array = []
    while True:
        if array_1[i] <= array_2[j]:
            merge_array.append(array_1[i])
            i += 1
        else:
            merge_array.append(array_2[j])
            count += len(array_1) - i
            j += 1
        if i == len(array_1):
            merge_array += array_2[j:]
            break
        if j == len(array_2):
            merge_array += array_1[i:]
            break
    return merge_array

N = int(input())
array = [0] * N
count = 0
for i in range(N):
    array[N-i-1] = int(input())
mergeSort(array)
print(count)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw5-oj-20018](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw5-oj-20018.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

最近很忙，勉强跟进每日选做









