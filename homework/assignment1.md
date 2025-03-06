# Assignment #1: 虚拟机，Shell & 大语言模型

Updated 2309 GMT+8 Feb 20, 2025

2025 spring, Complied by <mark>张景天 物理学院</mark>



**作业的各项评分细则及对应的得分**

| 标准                                 | 等级                                                         | 得分 |
| ------------------------------------ | ------------------------------------------------------------ | ---- |
| 按时提交                             | 完全按时提交：1分<br/>提交有请假说明：0.5分<br/>未提交：0分  | 1 分 |
| 源码、耗时（可选）、解题思路（可选） | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于2个：0分 | 1 分 |
| AC代码截图                           | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于：0分 | 1 分 |
| 清晰头像、PDF文件、MD/DOC附件        | 包含清晰的Canvas头像、PDF文件以及MD或DOC格式的附件：1分<br/>缺少上述三项中的任意一项：0.5分<br/>缺失两项或以上：0分 | 1 分 |
| 学习总结和个人收获                   | 提交了学习总结和个人收获：1分<br/>未提交学习总结或内容不详：0分 | 1 分 |
| 总得分： 5                           | 总分满分：5分                                                |      |
>
> 
>
> **说明：**
>
> 1. **解题与记录：**
>       - 对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>    
>2. **课程平台与提交安排：**
> 
>   - 我们的课程网站位于Canvas平台（https://pku.instructure.com ）。该平台将在第2周选课结束后正式启用。在平台启用前，请先完成作业并将作业妥善保存。待Canvas平台激活后，再上传你的作业。
> 
>       - 提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
> 
>3. **延迟提交：**
> 
>   - 如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
> 
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：依题意执行即可，使用`math.gcd`约分。



代码：

```python
from math import gcd
class Fraction:
    def __init__(self, numerator, denominator):
        divisor = gcd(numerator, denominator)
        self.numerator = numerator // divisor
        self.denominator = denominator // divisor
    def __str__(self):
        return f'{self.numerator}/{self.denominator}'
    def __add__(self, fraction):
        return Fraction(self.numerator * fraction.denominator + self.denominator * fraction.numerator, fraction.denominator * self.denominator)
    
if __name__ == '__main__':
    a, b, c, d = map(int, input().split())
    f1 = Fraction(a, b)
    f2 = Fraction(c ,d)
    print(f1 + f2)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw1-oj-27653.png)



### 1760.袋子里最少数目的球

 https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/



思路：由贪心求出$Operations = \sum\limits_{i=0}^{n-1}\lfloor\frac{(nums)_i-\delta}{Cost}\rfloor$，这直接给出了$Cost=Cost(operations)$的单调递减隐函数关系，要求$Cost(maxOperations)$的值，只要用二分法解方程即可。

$Cost = \frac{\sum{(nums)_i}}{maxOperations+n}$不失为一个较好的近似，从这里开始二分法可以优化：

$$maxOperations>=Operations>=\frac{\sum(nums)_i}{Cost}-n$$

$$Cost \in \left[\frac{\sum(nums)_i}{maxOperations+n},\max(nums)_i\right]$$

这样范围进一步缩小。另外也许可以使用割线法求解（优化不会很大，就不用了）。

代码：

```python
class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        def check(cost, maxOperations):
            Operations = sum((x-1) // cost for x in nums)
            return Operations <= maxOperations

        left, right = max(sum(nums) // (maxOperations + len(nums)), 1), max(nums)
        while right>left:
            mid = (left + right)//2
            flag = check(mid, maxOperations)
            if flag:
                right = mid
            else:
                left = mid + 1
        return left
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw1-leetcode-1760.png)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135



思路：由贪心得知，给定$Cost$时$M$最小的方案如下：向fajo月中加入一天直到无法加入后开启一个新的fajo月，又知$Cost$与$M$负相关，于是同上题，二分求解。



代码：

```python
N, M = map(int, input().split())
nums = []
for i in range(N):
    nums.append(int(input()))
def check(cost):
    part_cost = 0
    m = 1
    for num in nums:
        part_cost += num
        if part_cost > cost:
            part_cost = num
            m += 1
    return m
left, right = max(nums), sum(nums)
while left<right:
    mid = (left+right) // 2
    if check(mid) > M:
        left = mid + 1
    else:
        right = mid
print(left)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw1-oj-04135.png)



### 27300: 模型整理

http://cs101.openjudge.cn/practice/27300/



思路：简单的两次排序。



代码：

```python
n = int(input())
model_dic = {}
model_list = []
def key(param):
    return float(param[:-1]) * {'M': 1e6, 'B': 1e9}[param[-1]]
for i in range(n):
    model, param = input().split('-')
    if model in model_dic:
        model_dic[model].append(param)
    else:
        model_dic[model] = [param]
for model in model_dic:
    model_dic[model].sort(key=key)
    model_list.append([model, model_dic[model]])
model_list.sort()
for model in model_list:
    print(model[0] + ': ' + ', '.join(model[1]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw1-oj-27300.png)



### Q5. 大语言模型（LLM）部署与测试

本任务旨在本地环境或通过云虚拟机（如 https://clab.pku.edu.cn/ 提供的资源）部署大语言模型（LLM）并进行测试。用户界面方面，可以选择使用图形界面工具如 https://lmstudio.ai 或命令行界面如 https://www.ollama.com 来完成部署工作。

测试内容包括选择若干编程题目，确保这些题目能够在所部署的LLM上得到正确解答，并通过所有相关的测试用例（即状态为Accepted）。选题应来源于在线判题平台，例如 OpenJudge、Codeforces、LeetCode 或洛谷等，同时需注意避免与已找到的AI接受题目重复。已有的AI接受题目列表可参考以下链接：
https://github.com/GMyhf/2025spring-cs201/blob/main/AI_accepted_locally.md

请提供你的最新进展情况，包括任何关键步骤的截图以及遇到的问题和解决方案。这将有助于全面了解项目的推进状态，并为进一步的工作提供参考。

本人的电脑配置为 16GB 内存，无显卡，经测试最多运行 16B 的大模型。

在 LM Studio 中使用 Deepseek Coder V2 Lite 16B 尝试解题，实验结果示于下表：

|             <mark>Accept</mark>             | <mark style="background-color: red;">Not Accept</mark> |
| :-----------------------------------------: | :----------------------------------------------------: |
|              27653: Fraction类              |                    27300: 模型整理                     |
|               04135: 月度开销               |                                                        |
| 18161: 矩阵运算(先乘再加)**(经提醒后改正)** |                                                        |
|               04140: 方程求解               |                                                        |
|                                             |                                                        |

<center>table 1：Deepseek Coder V2 Lite 16B accepted locally</center>

由此可见，AI容易解决算法模板题，然而在细节问题与对题文的理解上可能出问题：例如，在 27300: 模型整理 中，AI始终无法理解题目并不需要对模型参量数值做格式上的处理，不是改变 B 与 M 的单位就是保留多位小数。

不排除有些题目的题文本身比较模糊。

### Q6. 阅读《Build a Large Language Model (From Scratch)》第一章

作者：Sebastian Raschka

请整理你的学习笔记。这应该包括但不限于对第一章核心概念的理解、重要术语的解释、你认为特别有趣或具有挑战性的内容，以及任何你可能有的疑问或反思。通过这种方式，不仅能巩固你自己的学习成果，也能帮助他人更好地理解这一部分内容。

英语文献读起来还是费劲，第一章尚有部分未读完。之前在计概时已尝试使用BERT模型解决问题，并且研究了一下 Attention is all you need这篇论文。似乎目前的大模型还都是使用transformer，只是参数量变大了很多。

问题：

- 目前有没有比transformer更先进的架构，优化方向在哪里？

- 需要大量的参数以及训练语料才能训练出一个不那么像人的AI，这是否意味着当前的训练算法对信息的利用效率是极低的，至少比生物低几个数量级，如果从信息论的观点来看，训练信息量与模型在某个特定问题上的准确率之间的极限在哪里？（最好有定量关系）



## 2. 学习总结和个人收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

云端虚拟机很好用，使用VScode远程连接后更加方便了。我也曾在Linux上运行过一些数值模拟软件，可惜云端的算力不足，不然就可以挂着comsol算个一整天。虚拟机的网速比本地快多了，有些文件可以下载在虚拟机上直接操作。

之前计概不在闫老师班里，刷题有些少，目前正在补刷 Leetcode 热题 100 ，发现很多算法的细节已经记不清楚了，例如二分区间的开闭，要试几次才对，往往知道算法但是在细节上不断报错，被一些模拟题折磨。

最小化最大值往往导向二分，这确实是一个有用的经验，但是我尚未厘清其本质原因。我的想法是把二分当作一个找函数零点的工具使用，至于函数关系即从最大(最小)化中贪心得到。可能是由于我思考题目时往往先考虑有没有数学上的突破口。

这学期专业课压力比较小，所以来选闫老师的课了[doge]。

