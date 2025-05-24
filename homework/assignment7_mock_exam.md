# Assignment #7: 20250402 Mock Exam

Updated 1624 GMT+8 Apr 2, 2025

2025 spring, Complied by <mark>张景天 物理学院</mark>



> **说明：**
>
> 1. **⽉考**：AC?<mark>None</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E05344:最后的最后

http://cs101.openjudge.cn/practice/05344/



思路：



代码：

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.prev = None

n, k = map(int, input().split())
start_node = prev_node = Node(1)
for i in range(2, n+1):
    node = Node(i)
    node.prev = prev_node
    prev_node.next = node
    prev_node = node
node.next = start_node
start_node.prev = node
killed = []
while node.next != node:
    for i in range(k):
        node = node.next
    killed.append(str(node.val))
    node.prev.next = node.next
    node.next.prev = node.prev
print(' '.join(killed[:-1]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408235034918](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250408235034918.png)



### M02774: 木材加工

binary search, http://cs101.openjudge.cn/practice/02774/



思路：之前同样的一道题换壳。最大长度$l$对应切的段数$k(l)=\sum\limits_{i=1}^{N}\lfloor\frac{L_i}{l}\rfloor$



代码：

```python
from math import floor
n, k = map(int, input().split())
nums = [int(input()) for _ in range(n)]
def check(l):
    return -sum(map(lambda x: floor(x / l), nums)) <= -k
def bisect_right():
    lo = 1
    hi = max(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if check(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo - 1
print(bisect_right())
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408235056473](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250408235056473.png)



### M07161:森林的带度数层次序列存储

tree, http://cs101.openjudge.cn/practice/07161/



思路：



代码：

```python
from collections import deque
class Node:
    def __init__(self, val, degree):
        self.val = val
        self.degree = degree
        self.children = []
def post_order(node, out):
    for child in node.children:
        post_order(child, out)
    out.append(node.val)
n = int(input())
out = []
for i in range(n):
    seq = input().split()

    nodes = deque()
    root = Node(seq[0], degree=int(seq[1]))
    nodes.append(root)
    i = 2
    while nodes and i < len(seq) - 1:
        current_node = nodes.popleft()
        degree = current_node.degree
        for _ in range(degree):
            child = Node(seq[i], int(seq[i+1]))
            current_node.children.append(child)
            nodes.append(child)
            i += 2
    post_order(root, out)

print(" ".join(out))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408235116888](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250408235116888.png)



### M18156:寻找离目标数最近的两数之和

two pointers, http://cs101.openjudge.cn/practice/18156/



思路：



代码：

```python
T = int(input())
S = sorted(list(map(int, input().split())))
left, right = 0, len(S) - 1
closest = float("inf")
while left < right:
    s = S[left] + S[right]
    if abs(s - T) < abs(closest - T) or (abs(s - T) == abs(closest - T) and s < closest):
        closest = s
        #result = (S[left], S(right))
    if s == T:
        closest = T
        break
    elif s < T:
        left += 1
    else:
        right -= 1
print(closest)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408235136946](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250408235136946.png)



### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/



思路：



代码：

```python
primes = []
is_prime = [True] * 10002
is_prime[0] = is_prime[1] = False
for i in range(2, 10002):
    if is_prime[i]:
        primes.append(i)
    for p in primes:
        if i * p > 10001:
            break
        is_prime[i * p] = False
        if i % p == 0:
            break
T = int(input())
for case in range(1, T+1):
    n = int(input())
    nums = []
    for i in range(1, n, 10):
        if is_prime[i]:
            nums.append(str(i))
    print(f"Case{case}:")
    if not nums:
        print("NULL")
    else:
        print(" ".join(nums))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408235152603](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250408235152603.png)



### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/



思路：



代码：

```python
class Team:
    def __init__(self, name):
        self.name = name
        self.accept = [0] * 26
        self.tries = 0
    def to_tuple(self):
        return (- len([_ for _ in self.accept if _ == 1]), self.tries, self.name)

M = int(input())
teams_dic = {}
teams = []
for i in range(M):
    name, num, accept = input().split(",")
    if name not in teams_dic:
        teams_dic[name] = Team(name)
        teams.append(teams_dic[name])
    teams_dic[name].tries += 1
    if accept == "yes":
        teams_dic[name].accept[ord(num) - ord("A")] = 1

data = sorted([team.to_tuple() for team in teams])
for i in range(min(12, len(data))):
    print(f"{i + 1} {data[i][2]} {-data[i][0]} {data[i][1]}")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20250408235220235](C:\Users\13706\AppData\Roaming\Typora\typora-user-images\image-20250408235220235.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

冲突未参加月考，月考简单，基本都是背板，细节凭感觉就对了；忙，日后会跟进选做









