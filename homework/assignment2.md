

# Assignment #2: 深度学习与大语言模型

Updated 2204 GMT+8 Feb 25, 2025

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

### 18161: 矩阵运算

matrices, http://cs101.openjudge.cn/practice/18161



思路：$P_{ik}=C_{ik} + A_{ij}B_{jk}$.



代码：

```python
n, m = map(int, input().split())
A = []
for i in range(n):
    A.append(list(map(int, input().split())))
p, q = map(int, input().split())
B = []
for i in range(p):
    B.append(list(map(int, input().split())))
r, s = map(int, input().split())
C = []
for i in range(r):
    C.append(list(map(int, input().split())))
if m == p and n == r and q == s:
    for i in range(n):
        for j in range(m):
            for k in range(q):
                C[i][k] += A[i][j] * B[j][k]
        print(*C[i])
else:
    print('Error!')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw2-oj-18161.png)



### 19942: 二维矩阵上的卷积运算

matrices, http://cs101.openjudge.cn/practice/19942/




思路：直接计算。话说卷积可以使用fft解决，复杂度可以到$O(n^2m^2 logn logm)$，不过在小矩阵情况下常数很大。



代码：

```python
m, n, p, q = map(int, input().split())
A = []
B = []
for i in range(m):
    A.append(list(map(int, input().split())))
for i in range(p):
    B.append(list(map(int, input().split())))
for i in range(m+1-p):
    lst = []
    for j in range(n+1-q):
        s = 0
        for k in range(p):
            for l in range(q):
                s += A[i+k][j+l]*B[k][l]
        lst.append(s)
    print(*lst)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw2-oj-19942.png)



### 04140: 方程求解

牛顿迭代法，http://cs101.openjudge.cn/practice/04140/

请用<mark>牛顿迭代法</mark>实现。

$f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{1}{2}f''(x_0)(x-x_0)^2 + \cdots$，估算误差$\delta \approx \frac{1}{2f'(x_n)}f''(x_n)(x_{n+1}-x_n)^2 = \frac{f''(x_n)f(x_n)^2}{f'(x_n)^3}$.

因为大语言模型的训练过程中涉及到了梯度下降（或其变种，如SGD、Adam等），用于优化模型参数以最小化损失函数。两种方法都是通过迭代的方式逐步接近最优解。每一次迭代都基于当前点的局部信息调整参数，试图找到一个比当前点更优的新点。理解牛顿迭代法有助于深入理解基于梯度的优化算法的工作原理，特别是它们如何利用导数信息进行决策。

> **牛顿迭代法**
>
> - **目的**：主要用于寻找一个函数 $f(x)$ 的根，即找到满足 $f(x)=0$ 的 $x$ 值。不过，通过适当变换目标函数，它也可以用于寻找函数的极值。
> - **方法基础**：利用泰勒级数的一阶和二阶项来近似目标函数，在每次迭代中使用目标函数及其导数的信息来计算下一步的方向和步长。
> - **迭代公式**：$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $ 对于求极值问题，这可以转化为$ x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)} $，这里 $f'(x)$ 和 $f''(x)$ 分别是目标函数的一阶导数和二阶导数。
> - **特点**：牛顿法通常具有更快的收敛速度（尤其是对于二次可微函数），但是需要计算目标函数的二阶导数（Hessian矩阵在多维情况下），并且对初始点的选择较为敏感。
>
> **梯度下降法**
>
> - **目的**：直接用于寻找函数的最小值（也可以通过取负寻找最大值），尤其在机器学习领域应用广泛。
> - **方法基础**：仅依赖于目标函数的一阶导数信息（即梯度），沿着梯度的反方向移动以达到减少函数值的目的。
> - **迭代公式**：$ x_{n+1} = x_n - \alpha \cdot \nabla f(x_n) $ 这里 $\alpha$ 是学习率，$\nabla f(x_n)$ 表示目标函数在 $x_n$ 点的梯度。
> - **特点**：梯度下降不需要计算复杂的二阶导数，因此在高维空间中相对容易实现。然而，它的收敛速度通常较慢，特别是当目标函数的等高线呈现出椭圆而非圆形时（即存在条件数大的情况）。
>
> **相同与不同**
>
> - **相同点**：两者都可用于优化问题，试图找到函数的极小值点；都需要目标函数至少一阶可导。
> - **不同点**：
>   - 牛顿法使用了更多的局部信息（即二阶导数），因此理论上收敛速度更快，但在实际应用中可能会遇到计算成本高、难以处理大规模数据集等问题。
>   - 梯度下降则更为简单，易于实现，特别是在高维空间中，但由于只使用了一阶导数信息，其收敛速度可能较慢，尤其是在接近极值点时。
>



代码：

```python
x = 6
err = 1e-10
while True:
    delta = (x**3 -5*x**2 +10*x -80)/(3*x**2 -10*x + 10)
    x = x - delta
    if delta <= err**(1/2):
        break
print(format(x, '.9f'))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw2-oj-04140.png)



### 06640: 倒排索引

data structures, http://cs101.openjudge.cn/practice/06640/



思路：注意使用`set`去重，一开始看样例以为不会有重复的...



代码：

```python
N = int(input())
dic = {}
for i in range(1, N+1):
    doc = input().split()
    for j in range(1, int(doc[0]) + 1):
        if doc[j] in dic:
            dic[doc[j]].add(i)
        else:
            dic[doc[j]] = set([i])
M = int(input())
for i in range(M):
    word = input()
    if word in dic:
        print(*sorted(dic[word]))
    else:
        print('NOT FOUND')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw2-oj-06640.png)



### 04093: 倒排索引查询

data structures, http://cs101.openjudge.cn/practice/04093/



思路：使用集合操作是很直观的，要求出现时用`&`求交集，要求不出现时用`-`删去这些元素。



代码：

```python
N = int(input())
l = []
t_set = set()
for i in range(N):
    l.append(set(map(int, input().split()[1:])))
for i in l:
    t_set = t_set | i
M = int(input())
for i in range(M):
    q = input().split()
    q_set = t_set
    for j in range(N):
        if q[j] == '1':
            q_set = q_set & l[j]
        elif q[j] == '-1':
            q_set = q_set - l[j]
    if q_set:
        print(*sorted(q_set))
    else:
        print('NOT FOUND')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw2-oj-04093.png)



### Q6. Neural Network实现鸢尾花卉数据分类

在http://clab.pku.edu.cn 云端虚拟机，用Neural Network实现鸢尾花卉数据分类。

参考链接，https://github.com/GMyhf/2025spring-cs201/blob/main/LLM/iris_neural_network.md

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, num_classes=3):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = IrisNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

sample = X_test[0].unsqueeze(0)
prediction = torch.argmax(model(sample), dim=1)
print(f"\nSample prediction: True class {y_test[0].item()}, "
      f"Predicted class {prediction.item()}")
```



使用sklearn集成的`MLPClassifier`可以更方便的实现神经网络：

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42, learning_rate_init=0.01, verbose=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
```

训练结果如下：

```
...
Iteration 100, loss = 0.08176150

Accuracy: 1.0

Classification Report:
               precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

将上述`pytorch`代码封装为相同的`MLPClassifier`类：
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

class MLPClassifier():
    def __init__(self, hidden_layer_sizes=(10,), max_iter=100, random_state=42, batch_size=16, learning_rate_init=0.01, verbose=True):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.verbose = verbose
    def fit(self, X, y):
        self.model = IrisNet(input_size=len(X[0]), hidden_size=self.hidden_layer_sizes[0], num_classes=len(set(y)))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        num_epochs = self.max_iter
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_X.size(0)
            if self.verbose:
                epoch_loss = running_loss / len(train_loader.dataset)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.tolist()

class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, num_classes=3):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

```

训练结果如下：

```
Epoch [10/100], Loss: 0.2387
Epoch [20/100], Loss: 0.0989
Epoch [30/100], Loss: 0.0655
Epoch [40/100], Loss: 0.0532
Epoch [50/100], Loss: 0.0505
Epoch [60/100], Loss: 0.0415
Epoch [70/100], Loss: 0.0428
Epoch [80/100], Loss: 0.0425
Epoch [90/100], Loss: 0.0374
Epoch [100/100], Loss: 0.0386
Accuracy: 0.9666666666666667

Classification Report:
               precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```



当然对于这种简单的数据分类问题，也可以使用传统的SVM算法：
做替换

```python
from sklearn.svm import SVC

model = SVC(kernel='linear', C=1.0)
```

训练结果如下：
```
Accuracy: 0.9666666666666667

Classification Report:
               precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.89      0.94         9
   virginica       0.92      1.00      0.96        11

    accuracy                           0.97        30
   macro avg       0.97      0.96      0.97        30
weighted avg       0.97      0.97      0.97        30
```

由于iris数据集太小了，超参数优化不明显。

可以试试sentiment140的文本情感分类，这是我计概时的一个大作业项目，还是挺有意思的。

## 2. 学习总结和个人收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

在做每日选做，大部分题目没有问题。回溯的题目之前没怎么做过（leetcode上数独做了半天仍然WA遂破防，之后就没做过回溯了）需要做一些模板题，并查集、马拉车之类的做过一遍之后就差不多会了。还是要多做题多背板。

使用强化学习方法训练$2048$游戏的AI：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import math
import copy
from itertools import count
import random
import os
import sys
from collections import namedtuple
BOARD_SIZE = 4
TARGET = 2048

def init_board():
    board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    add_random_tile(board)
    add_random_tile(board)
    return board

def add_random_tile(board):
    empty = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == 0]
    if not empty:
        return
    i, j = random.choice(empty)
    board[i][j] = 4 if random.random() < 0.1 else 2

def print_board(board):
    os.system('clear')  # 清屏
    print("-" * (BOARD_SIZE * 7 + 1))
    for row in board:
        print("|", end="")
        for num in row:
            if num == 0:
                print("      |", end="")
            else:
                print(f"{num:^6}|", end="")
        print()
        print("-" * (BOARD_SIZE * 7 + 1))

def slide_and_merge(board):
    reward = 0
    new_board = []
    for line in board:
        new_line = [num for num in line if num != 0]
        merged_line = []
        skip = False
        for i in range(len(new_line)):
            if skip:
                skip = False
                continue
            if i + 1 < len(new_line) and new_line[i] == new_line[i+1]:
                merged_line.append(new_line[i] * 2)
                reward += new_line[i]
                skip = True
            else:
                merged_line.append(new_line[i])
        merged_line += [0] * (BOARD_SIZE - len(merged_line))
        new_board.append(merged_line)
    return new_board, reward

def move_left(board):
    return slide_and_merge(board)

def reverse(board):
    return [list(reversed(row)) for row in board]

def transpose(board):
    return [list(row) for row in zip(*board)]

def move_right(board):
    result = move_left(reverse(board))
    return reverse(result[0]), result[1]

def move_up(board):
    result = move_left(transpose(board))
    return transpose(result[0]), result[1]

def move_down(board):
    result = move_right(transpose(board))
    return transpose(result[0]), result[1]

def boards_equal(b1, b2):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if b1[i][j] != b2[i][j]:
                return False
    return True

def can_move(board):
    for row in board:
        if 0 in row:
            return True
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if j + 1 < BOARD_SIZE and board[i][j] == board[i][j + 1]:
                return True
            if i + 1 < BOARD_SIZE and board[i][j] == board[i + 1][j]:
                return True
    return False

def reached_target(board):
    for row in board:
        if any(num >= TARGET for num in row):
            return True
    return False
def count_empty(board):
    return sum(row.count(0) for row in board)

def calculate_smoothness(board):
    """计算平滑度：相邻单元格差值的总和（差值越小越好）"""
    smooth = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE - 1):
            smooth -= abs(board[i][j] - board[i][j+1])
    for j in range(BOARD_SIZE):
        for i in range(BOARD_SIZE - 1):
            smooth -= abs(board[i][j] - board[i+1][j])
    return smooth

def calculate_monotonicity(board):
    """计算单调性：如果行或列单调性好则奖励"""
    mono_rows = 0
    mono_cols = 0
    for row in board:
        if all(row[i] >= row[i+1] for i in range(BOARD_SIZE-1)) or all(row[i] <= row[i+1] for i in range(BOARD_SIZE-1)):
            mono_rows += 1
    for col in zip(*board):
        if all(col[i] >= col[i+1] for i in range(BOARD_SIZE-1)) or all(col[i] <= col[i+1] for i in range(BOARD_SIZE-1)):
            mono_cols += 1
    return mono_rows + mono_cols

def evaluate_board(board):
    empty = count_empty(board)
    smoothness = calculate_smoothness(board)
    monotonicity = calculate_monotonicity(board)
    # 权重矩阵（蛇形布局），鼓励大数集中在角落
    weights = [
        [65536, 32768, 16384, 8192],
        [512,   1024,  2048,  4096],
        [256,    512,  1024,  2048],
        [128,    256,   512,  1024]
    ]
    weight_score = sum(board[i][j] * weights[i][j] for i in range(BOARD_SIZE) for j in range(BOARD_SIZE))
    return weight_score + empty * 500 + smoothness * 5 + monotonicity * 100

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

def get_state(board):
    """将棋盘转换为网络输入"""
    return torch.tensor([[board[i][j] for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)], dtype=torch.float).view(-1).unsqueeze(0)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def select_action(state, steps_done, policy_net):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            state = state.to(device)
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], dtype=torch.long)

BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10

policy_net = DQN(BOARD_SIZE*BOARD_SIZE, 4).to(device)
target_net = DQN(BOARD_SIZE*BOARD_SIZE, 4).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.5)
memory = ReplayMemory(10000)

steps_done = 0

num_episodes = 1000
for i_episode in range(num_episodes):
    board = init_board()
    state = get_state(board)
    cumulative_reward = 0 
    for t in count():
        action = select_action(state, steps_done, policy_net)
        steps_done += 1
        move_func = [move_up, move_down, move_left, move_right]
        move_result = move_func[action.item()](copy.deepcopy(board))
        new_board = move_result[0]
        if boards_equal(board, new_board):
            continue
        board = new_board
        add_random_tile(board)
        
        reward =math.log2(move_result[1] + 1) if can_move(board) else -1
        cumulative_reward += reward
        reward = torch.tensor([reward], dtype=torch.float)

        next_state = get_state(board) if can_move(board) else None

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model(policy_net, target_net, memory, optimizer)

        if not can_move(board):
            print(max(map(max, board)))
            print(f"Episode {i_episode + 1}, Score: {int(cumulative_reward)}")
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
torch.save(policy_net.state_dict(), 'policy_net_model.pth')
print("Model saved successfully.")
```



可惜训练效果不好，也许这种简单的游戏不如用朴素的估价函数然后搜索？五子棋似乎也是这样，简单估价加上4到5层搜索就可以碾压我了。

另外一个思路是优化权重函数的参数，使用遗传算法之类的，只是评估方法没有想好。
