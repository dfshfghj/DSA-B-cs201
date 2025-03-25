# Assignment #4: 位操作、栈、链表、堆和NN

Updated 1203 GMT+8 Mar 10, 2025

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

### 136.只出现一次的数字

bit manipulation, https://leetcode.cn/problems/single-number/



<mark>请用位操作来实现，并且只使用常量额外空间。</mark>



代码：

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        single_number = 0
        for num in nums:
            single_number ^= num
        return single_number
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw4-leetcode-0136.png (1036×670)](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw4-leetcode-0136.png)



### 20140:今日化学论文

stack, http://cs101.openjudge.cn/practice/20140/



思路：使用递归，一层层把[]里的字符解压出来。

`stack`和递归本质上是一样的所以就不管`stack`标签了。



代码：

```python
s = input()

def extract(text):
    extract_text = ''
    mul_str = ''
    i = 0
    while i < len(text):
        if text[i] == '[':
            j, part = extract(text[i+1:])
            extract_text += part
            i += j + 1
        elif text[i] == ']':
            return i, int(mul_str) * extract_text
        elif text[i].isdigit():
            mul_str += text[i]
        else:
            extract_text += text[i]
        i += 1
    return i, extract_text
print(extract(s)[1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw4-oj-20140.png (934×774)](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw4-oj-20140.png)



### 160.相交链表

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/



思路：第一个思路是用集合存下A链表的节点，遍历B链表时找到第一个在集合中出现的节点。

后来看到要求设计一个时间复杂度 `O(m + n)` 、仅用 `O(1)` 内存的解决方案，想到双指针让两个指针停在所求节点，可以从对称性的要求出发，让两个指针把AB都跑一遍就对称了，会同时跑到相交的节点。



代码：

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        nodeA = headA
        nodeB = headB
        while nodeA != nodeB:
            if nodeA:
                nodeA = nodeA.next
            else:
                nodeA = headB
            if nodeB:
                nodeB = nodeB.next
            else:
                nodeB = headA
        return nodeA
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw4-leetcode-0160.png (1011×668)](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw4-leetcode-0160.png)



### 206.反转链表

linked list, https://leetcode.cn/problems/reverse-linked-list/



思路：在遍历一遍的同时把指向后一个节点的指针指向前一个即可。



代码：

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        current = head
        while current:
            current.next, prev, current = prev, current, current.next
        return prev
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw4-leetcode-0206.png (1019×672)](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw4-leetcode-0206.png)



### 3478.选出和最大的K个元素

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/



思路：观察数据范围得知复杂度为$O(n\log n)$，故考虑使用堆维护最大的$k$个元素，另外还需要维护和值，否则有无法接受的$O(n k)$额外开销。初始对`nums1`排序之后就只需要遍历一遍就可以解决。

发现`heappushpop`似乎会比先`heappush`再`heappop`快。

代码：

```python
class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        import heapq
        indexs = sorted(enumerate(nums1), key=lambda x: x[1])
        heap = [0] * k
        max_sum = [0] * len(nums1)
        j = 0
        s = 0
        for i in range(len(indexs)):
            while indexs[j][1] < indexs[i][1]:
                s += nums2[indexs[j][0]]
                s -= heapq.heappushpop(heap, nums2[indexs[j][0]])
                j += 1
            max_sum[indexs[i][0]] = s
        return max_sum
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![hw4-leetcode-3478.png (1015×672)](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw4-leetcode-3478.png)



### Q6.交互可视化neural network

https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises

**Your task:** configure a neural network that can separate the orange dots from the blue dots in the diagram, achieving a loss of less than 0.2 on both the training and test data.

**Instructions:**

In the interactive widget:

1. Modify the neural network hyperparameters by experimenting with some of the following config settings:
   - Add or remove hidden layers by clicking the **+** and **-** buttons to the left of the **HIDDEN LAYERS** heading in the network diagram.
   - Add or remove neurons from a hidden layer by clicking the **+** and **-** buttons above a hidden-layer column.
   - Change the learning rate by choosing a new value from the **Learning rate** drop-down above the diagram.
   - Change the activation function by choosing a new value from the **Activation** drop-down above the diagram.
2. Click the Play button above the diagram to train the neural network model using the specified parameters.
3. Observe the visualization of the model fitting the data as training progresses, as well as the **Test loss** and **Training loss** values in the **Output** section.
4. If the model does not achieve loss below 0.2 on the test and training data, click reset, and repeat steps 1–3 with a different set of configuration settings. Repeat this process until you achieve the preferred results.

给出满足约束条件的<mark>截图</mark>，并说明学习到的概念和原理。

![hw4-nn.png (1463×1115)](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw4-nn.png)

+ 正则化：提高泛化能力

  - L1正则化：$L_(w) =\lambda \|w\|_1$

    L1正则化倾向于产生稀疏的权重矩阵，即许多权重值会变为零。能够进行特征选择，自动挑选出那些对目标变量影响较大的特征；
  - L2正则化：$L_(w) =\lambda \|w\|_2^2$
    
    可以微分，计算梯度下降时比较方便。

+ 激活函数：
  - 双曲正切$\tanh$函数：存在梯度消失问题；

  - ReLU$max(0,x)$函数：简单且计算效率高，有效缓解了梯度消失问题，并加速了深层网络的训练。目前是许多类型网络的默认选择。可能导致某些神经元“死亡”，即对于负输入，ReLU输出为零，这些神经元及其权重将不再更新。(换为Leaky ReLU 或ELU解决)

+ 隐藏层大小：层数越多能拟合越复杂的函数，但是容易过拟合。这很容易理解，参数多的极端就是拉格朗日插值。事实上，上图的神经网络已经有过拟合的倾向了，本来应该是一个圆，变得奇形怪状就是被噪声影响了。

+ 批次大小：调大批次大小可以并行计算，但是需要更久收敛，实验中也观察到了这一点。

做了一个高维数据投影至三维的数据可视化(iris)，看起来效果不错，后续大概也可以做一个类似的可视化neural network：

![raw.githubusercontent.com/dfshfghj/DSA-B-cs201/13cecec6fb317f67d5d8180dd9d8970f8602a668/img/visual_iris.svg](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/13cecec6fb317f67d5d8180dd9d8970f8602a668/img/visual_iris.svg)

## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

Leetcode热题100已经过半，积累了很多模板，感觉实力有很大提升。

