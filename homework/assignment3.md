# Assignment #3: 惊蛰 Mock Exam

Updated 1641 GMT+8 Mar 5, 2025

2025 spring, Complied by <mark>张景天 物理学院</mark>



> **说明：**
>
> 1. **惊蛰⽉考**：<mark>未参加</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E04015: 邮箱验证

strings, http://cs101.openjudge.cn/practice/04015



思路：依题意模拟即可。之前做过了。



代码：

```python
while True:
    try:
        emails = input()
    except EOFError:
        break
    yes_or_no = True
    if '@' not in emails:
        yes_or_no = False
    
    else:
        index = emails.find('@')
        if '@' in emails[index + 1:]:
            yes_or_no = False

    if emails[0] == '@' or emails[0] == '.':
        yes_or_no = False

    if emails[-1] == '@' or emails[-1] == '.':
        yes_or_no = False

    index0 = emails.find('@')
    if '.' not in emails[index0 + 1:]:
        yes_or_no = False
    if '@.' in emails or '.@'  in emails:
        yes_or_no = False

    if yes_or_no:
        print('YES')
    else:
        print('NO')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw3-oj-04015.png)



### M02039: 反反复复

implementation, http://cs101.openjudge.cn/practice/02039/



思路：直接模拟就好了



代码：

```python
n = int(input())
text = input()
output = ''
data = [[0 for _ in range(n)] for __ in range(len(text)//n)]
line = []
for i in range(len(text)):
    if (i//n)%2 == 1:
        data[i//n][-i%n-1] = text[i]
    else:
        data[i//n][i%n] = text[i]

m = len(data)
for i in range(len(text)):
    output += data[i%m][i//m]
print(output)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw3-oj-02039.png)



### M02092: Grandpa is Famous

implementation, http://cs101.openjudge.cn/practice/02092/



思路：读懂英文题干后发现只要统计一下出现顺序然后排序一下就行了，感到很疑惑(⊙o⊙)？



代码：

```python
while True:
    N, M = map(int, input().split())
    if N == 0:
        break
    data = []
    playlist = [0 for _ in range(10001)]
    for __ in range(N):
        rank = list(map(int, input().split()))
        for player in rank:
            playlist[player] += 1
    playlist = sorted(enumerate(playlist), key=lambda x: x[1])
    second_score = playlist[-2][1]
    second_players = []
    i = -2
    while True:
        if playlist[i][1] == second_score:
            second_players.append(playlist[i][0])
        else:
            break
        i -= 1
    output = [0] * len(second_players)
    for i in range(len(second_players)):
        output[i] = second_players[len(second_players) - i - 1]
    print(*output)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw3-oj-02092.png)



### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/



思路：如果按照朴素的做法复杂度为$O(L^2 d^2)$，显然超时，注意到垃圾分布是稀疏的，于是反过来：

​	在每个垃圾周围$d$的范围内给二维数组$+ i$，然后找二维数组极大值。复杂度$O(d^2 n)$。

如果要优化的话可以用一个`set`存下有可能的点，这样最后就不用遍历所有点了。

ps.发现有个求极值非常快的方法`max(map(max, data))`

代码：

```python
d = int(input())
n = int(input())
data = [[0 for _ in range(1025)] for __ in range(1025)]
for ___ in range(n):
    x, y, i = map(int, input().split())
    for xi in range(max(x-d, 0), min(x+d+1,1025)):
        for eta in range(max(y-d, 0), min(y+d+1, 1025)):
            data[xi][eta] += i
max_score = max(map(max, data))
num = 0
for xi in range(1025):
    for eta in range(1025):
        if data[xi][eta] == max_score:
            num += 1
print(num, max_score)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw3-oj-04133.png)



### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/



思路：当dfs模板题来做了，保持字典序只要让`dx` `dy`按字典序即可。

第一次WA查了好久没有发现原来是`Scenario #{i+1}:`全部当`Scenario #1:`输出了

代码：

```python
n = int(input())
dy = [-1, 1, -2, 2, -2, 2, -1, 1]
dx = [-2, -2, -1, -1, 1, 1, 2, 2]
def is_valid(x, y, visited):
    if 0 <= x < q and 0 <= y < p and not visited[x][y]:
        return True
    else:
        return False

def dfs(x, y, depth, visited, way):
    visited[x][y] = True
    way.append(f"{chr(ord('A')+x)}{y+1}")
    if depth == p * q:
        return True
    for i in range(8):
        nx = x + dx[i]
        ny = y + dy[i]
        if is_valid(nx, ny, visited):
            if dfs(nx, ny, depth + 1, visited, way):
                return True
    visited[x][y] = False
    way.pop()
for i in range(n):
    p, q = map(int, input().split())
    visited = [[False for _ in range(p)] for _ in range(q)]
    way = []
    dfs(0, 0, 1, visited, way)
    print(f'Scenario #{i+1}:')
    if way:
        print(''.join(way))
    else:
        print('impossible')
    print()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw3-oj-02488.png)



### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/



思路：首先最小的序列是组合每一行的最小值，所以需要`sort`之后取最小，这之后尝试把每一行的索引往后挪来找到其余的最小值。

接下来的思路是问AI得到的：考虑所有m行很复杂，所以先考虑两行，定义``merge_min(s, b, n)`从中取出`n`个最小的和，再以此类推处理`m`行。`merge_min`是通过`heap`实现的高效寻找。

有点巧妙。



代码：

```python
import heapq
def merge_min(s, b, n):
    visited = set()
    heap = []
    heapq.heappush(heap, (s[0] + b[0], 0, 0))
    visited.add((0, 0))
    res = []
    while len(res) < n and heap:
        sum_val, i, j = heapq.heappop(heap)
        res.append(sum_val)
        if i + 1 < len(s) and (i+1, j) not in visited:
            new_sum = s[i+1] + b[j]
            heapq.heappush(heap, (new_sum, i+1, j))
            visited.add((i+1, j))
        if j + 1 < len(b) and (i, j+1) not in visited:
            new_sum = s[i] + b[j+1]
            heapq.heappush(heap, (new_sum, i, j+1))
            visited.add((i, j+1))
    return res

T = int(input())
for _ in range(T):
    m, n = map(int, input().split())
    arrays = []
    for _ in range(m):
        row = list(map(int, input().split()))
        row.sort()
        arrays.append(row)
    current = arrays[0].copy()
    for i in range(1, m):
        b = arrays[i]
        new_current = merge_min(current, b, n)
        current = new_current
    print(' '.join(map(str, current)))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/hw3-oj-06648.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

本次考试由于与课冲突未能参加，之后模拟考了一下，可以AC前5题，最后一题可以想到大概是heap但是没什么实现思路，看了答案之后释然，确实是水平不足。

自学了一下c++，发现真的快：

下面这个auto_2048代码，同样使用Alpha-Beta剪枝，可以比python快50~80倍（甚至没有加lru_cache）

c++写起来还是很舒服的，只不过很多python中可以直接拿来用的东西没有，比如lru_cache

一开始用的是Cython，把原来的py复制过来直接编译为pyd之后就可以快一倍，不过之后想要快就很麻烦了

再下面的Cython代码至今还是跑不起来，反观c++很快就调通了

吐槽Cython的语法简直是一坨，既要又要导致的

c++拿了MVP	cython就是躺赢狗（确信）

```cpp
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
using namespace std;

const int SIZE = 4;
const float INF = 2147483647;
bool canMove(int board[SIZE][SIZE]);
void initBoard(int board[SIZE][SIZE]);
void printBoard(const int board[SIZE][SIZE]);
void reverseRow(int row[SIZE]);
void addRandomTile(int board[SIZE][SIZE]);
bool slideRowLeft(int row[SIZE]);
bool move(int board[SIZE][SIZE], char dir);
char getRandomMove(int board[SIZE][SIZE]);
int countEmpty(int board[SIZE][SIZE]);
int calculateSmoothness(int board[SIZE][SIZE]);
int calculateMonotonicity(int board[SIZE][SIZE]);
int evaluateBoard(int board[SIZE][SIZE]);
int getDynamicDepth(int board[SIZE][SIZE]);
float expectimaxAB(int board[SIZE][SIZE], int depth, bool is_player, float alpha, float beta);
char chooseBestMove(int board[4][4]);

int main() {
    int board[SIZE][SIZE];
    initBoard(board);
    char input;
    bool running = true;

    while (running) {
        cout << "\033[2J\033[1;1H";
        printBoard(board);
        if (!canMove(board)) {
            cout << "No more moves available. Game Over!" << endl;
            break;
        }
        char input = chooseBestMove(board);
        cout << "choose dir:" << input << endl;
        if (input == 'q') break;
        cout << "automatically move " << input << endl;
        addRandomTile(board);
    }

    return 0;
}
bool canMove(int board[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j){
            if (board[i][j] == 0) return true;
            if (i < SIZE - 1 && board[i][j] == board[i + 1][j]) return true;
            if (j < SIZE - 1 && board[i][j] == board[i][j + 1]) return true;
        }
    return false;
}

void initBoard(int board[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            board[i][j] = 0;
    srand(42);
    addRandomTile(board);
    addRandomTile(board);
}

void printBoard(const int board[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j)
            cout << board[i][j] << "\t";
        cout << endl;
    }
}

void reverseRow(int row[SIZE]) {
    for (int i = 0; i < SIZE / 2; ++i) {
        int temp = row[i];
        row[i] = row[SIZE - i - 1];
        row[SIZE - i - 1] = temp;
    }
}

void addRandomTile(int board[SIZE][SIZE]) {
    int empty_count = 0;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            if (board[i][j] == 0) empty_count += 1;
        }
    }
    if (empty_count == 0) return ;
    int pos = rand() % empty_count;
    int count = 0;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            if (board[i][j] == 0) {
                if (pos == count) board[i][j] = (rand() % 10 < 9) ? 2 : 4;
                count += 1;
            }
        }
    }
}
bool slideRowLeft(int row[SIZE]) {
    bool changed = false;
    int newRow[SIZE] = {0};
    int idx = 0;
    for (int i = 0; i < SIZE; ++i) {
        if (row[i] != 0) {
            if (idx > 0 && newRow[idx-1] == row[i]) {
                newRow[idx-1] *= 2;
                changed = true;
            } else {
                newRow[idx++] = row[i];
            }
        }
    }
    bool isEqual = true;
    for (int i = 0; i < SIZE; ++i) {
        if (newRow[i] != row[i]) {
            isEqual = false;
            break;
        }
    }
    if (!isEqual) changed = true;
    for (int i = 0; i < SIZE; ++i) row[i] = newRow[i];
    return changed;
}

bool move(int board[SIZE][SIZE], char dir) {
    int originalBoard[SIZE][SIZE];
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            originalBoard[i][j] = board[i][j];

    if (dir == 'a') {
        for (int i = 0; i < SIZE; ++i) {
            slideRowLeft(board[i]);
        }
    } else if (dir == 'd') {
        for (int i = 0; i < SIZE; ++i) {
            reverseRow(board[i]);
            slideRowLeft(board[i]);
            reverseRow(board[i]);
        }
    } else if (dir == 'w') {
        int transposed[SIZE][SIZE] = {0};
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                transposed[i][j] = board[j][i];
        for (int i = 0; i < SIZE; ++i)
            slideRowLeft(transposed[i]);
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                board[j][i] = transposed[i][j];
    } else if (dir == 's') {
        int transposed[SIZE][SIZE] = {0};
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                transposed[i][j] = board[j][i];
        for (int i = 0; i < SIZE; ++i) {
            reverseRow(transposed[i]);
            slideRowLeft(transposed[i]);
            reverseRow(transposed[i]);
        }
        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                board[j][i] = transposed[i][j];
    }

    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            if (originalBoard[i][j] != board[i][j])
                return true;
    return false;
}
void boardCopy(int board[SIZE][SIZE], int board_copy[SIZE][SIZE]) {
    memcpy(board_copy, board, SIZE*SIZE*sizeof(int));
}
char getRandomMove(int board[SIZE][SIZE]) {
    char moves[] = {'w', 'a', 's', 'd'};
    for (int i = 0; i < 4; ++i) {
        int board_copy[4][4];
        boardCopy(board, board_copy);
        if (move(board_copy, moves[i])) return moves[i];
    }
    return 'q';
}
int countEmpty(int board[SIZE][SIZE]) {
    int empty_count = 0;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            if (board[i][j] == 0) empty_count += 1;
        }
    }
    return empty_count;
}
int calculateSmoothness(int board[SIZE][SIZE]) {
    int smooth = 0;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE - 1; ++j) {
            smooth -= abs(board[i][j] - board[i][j+1]);
        }
    }
    for (int i = 0; i < SIZE - 1; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            smooth -= abs(board[i][j] - board[i+1][j]);
        }
    }
    return smooth;
}
int calculateMonotonicity(int board[SIZE][SIZE]) {
    int mono_rows = 0;
    int mono_cols = 0;
    bool is_increasing = true;
    bool is_decreasing = true;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE - 1; ++j) {
            if (board[i][j] < board[i][j+1]) is_decreasing = false;
            if (board[i][j] > board[i][j+1]) is_increasing = false;
        }
        if (is_decreasing || is_increasing) mono_rows += 1;
    }
    is_increasing = true;
    is_decreasing = true;
    for (int j = 0; j < SIZE; ++j) {
        for (int i = 0; i < SIZE - 1; ++i) {
            if (board[i][j] < board[i+1][j]) is_decreasing = false;
            if (board[i][j] > board[i+1][j]) is_increasing = false;
        }
        if (is_decreasing || is_increasing) mono_cols += 1;
    }
    return mono_rows + mono_cols;
}
int evaluateBoard(int board[SIZE][SIZE]) {
    int empty = countEmpty(board);
    int smoothness = calculateSmoothness(board);
    int monotonicity = calculateMonotonicity(board);
    int weights[SIZE][SIZE] = {
        {65536, 32768, 16384, 8192},
        {512,   1024,  2048,  4096},
        {256,    512,  1024,  2048},
        {128,    256,   512,  1024}
    };
    int weight_score = 0;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            weight_score += weights[i][j] * board[i][j];
        }
    }
    return weight_score + empty * 500 + smoothness * 5 + monotonicity * 100;
}
int getDynamicDepth(int board[SIZE][SIZE]) {
    int empty_cells = countEmpty(board);
    if (empty_cells >= 6) return 4;
    else if (empty_cells >= 3) return 5;
    else return 6;
}
float expectimaxAB(int board[SIZE][SIZE], int depth, bool is_player, float alpha, float beta) {
    if (depth == 0 || !canMove(board)) return evaluateBoard(board);
    if (is_player) {
        float best = -INF;
        float a = alpha;
        char moves[] = {'w', 'a', 's', 'd'};
        for (int i = 0; i < SIZE; ++i) {
            int new_board[SIZE][SIZE];
            boardCopy(board, new_board);
            if (!move(new_board, moves[i])) continue;
            float val = expectimaxAB(new_board, depth - 1, false, a, beta);
            best = max(best, val);
            a = max(a, best);
            if (a >= beta) break;
        }
        return best;
    }
    else {
        int empty_count = 0;
        float total = 0;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                if (board[i][j] == 0) {
                    empty_count += 1;
                int board_copy[SIZE][SIZE];
                boardCopy(board, board_copy);
                board_copy[i][j] = 2;
                float val = expectimaxAB(board_copy, depth - 1, true, alpha, beta);
                total += 0.9 * val;
                boardCopy(board, board_copy);
                board_copy[i][j] = 4;
                val = expectimaxAB(board_copy, depth - 1, true, alpha, beta);
                total += 0.1 * val;
                }
            }
        }
    if (empty_count == 0) return evaluateBoard(board);
    return total / empty_count;
    }
}
char chooseBestMove(int board[4][4]) {
    char moves[] = {'w', 'a', 's', 'd'};
    char best_move = 'q';
    float best_score = -INF;
    int best_new_board[SIZE][SIZE] ={0};
    int depth = getDynamicDepth(board);
    cout << depth << endl;
    for (int i = 0; i < SIZE; ++i) {
        int new_board[SIZE][SIZE];
        boardCopy(board, new_board);
        if (!move(new_board, moves[i])) continue;
        float score = expectimaxAB(new_board, depth, false, -INF, INF);
        if (score > best_score) {
            best_score = score;
            best_move = moves[i];
            boardCopy(new_board, best_new_board);
        }
    }
    boardCopy(best_new_board, board);
    return best_move;
}
```



```cython
import random
import os
import time
#random.seed(40)
cdef extern from "stdlib.h":
    int rand()  
    int abs(int x)  
    int printf(const char *format, ...)  

cdef extern from "math.h":
    float INFINITY
    
cdef int BOARD_SIZE = 4
cdef int TARGET = 2048

cdef void print_board(int board[4][4])  :
    cdef int i, j
    # 使用 ANSI 转义码清屏
    printf("\033[H\033[J")
    printf("-" * ( 4 * 7 + 1))
    printf("\n")
    for i in range( 4):
        printf("|")
        for j in range( 4):
            if board[i][j] == 0:
                printf("      |")
            else:
                printf("%6d|", board[i][j])
        printf("\n")
        printf("-" * ( 4 * 7 + 1))
        printf("\n")

cdef void add_random_tile(int board[4][4])  :
    cdef int empty_count = 0
    cdef int i, j
    
    for i in range( 4):
        for j in range( 4):
            if board[i][j] == 0:
                empty_count += 1
    if empty_count == 0:
        return
    cdef int pos = rand() % empty_count
    cdef int count = 0
    for i in range( 4):
        for j in range( 4):
            if board[i][j] == 0:
                if count == pos:
                    #board[i][j] = (rand() % 10 < 9) ? 2 : 4
                    board[i][j] = 2 if (rand() % 10 < 9) else 4
                    return 
                count += 1

cdef void init_board(int board[4][4])  :
    cdef int i, j
    for i in range( 4):
        for j in range( 4):
            board[i][j] = 0
    add_random_tile(board)
    add_random_tile(board)

cdef int slide_and_merge(int board[4][4], int new_board[4][4])  :
    cdef int reward = 0
    cdef int new_line[ 4]
    #cdef int merged_line[ 4]
    cdef int count = 0
    cdef int merged_count = 0
    cdef bint skip = False
    for i in range( 4):
        
        for j in range( 4):
            if board[i][j] != 0:
                new_line[count] = board[i][j]
                count += 1
        
        for j in range(count):
            if skip:
                skip = False
                continue
            if j + 1 < count and new_line[j] == new_line[j+1]:
                new_board[i][merged_count] = new_line[j] * 2
                reward += new_line[j]
                skip = True
            else:
                new_board[i][merged_count] = new_line[j]
            merged_count += 1
    return reward

cdef int move_left(int board[4][4], int new_board[4][4])  :
    return slide_and_merge(board, new_board)

cdef void reverse(int board[4][4])  :
    cdef int i, j
    cdef int temp
    for i in range( 4):
        for j in range( 4):
            temp = board[i][j]
            board[i][j] = board[i][ 4 - 1 - j]
            board[i][ 4 - 1 - j] = temp

cdef void transpose(int board[4][4])  :
    cdef int temp
    cdef int i, j
    for i in range( 4):
        for j in range(i + 1,  4):
            temp = board[i][j]
            board[i][j] = board[j][i]
            board[j][i] = temp

cdef int move_right(int board[4][4], int new_board[4][4])  :
    cdef int reversed_board[4][4]
    cdef int i,j
    for i in range( 4):
        for j in range( 4):
            reversed_board[i][j] = board[i][j]
    reverse(reversed_board)
    reward = move_left(reversed_board, new_board)
    reverse(new_board)
    return reward

cdef int move_up(int board[4][4], int new_board[4][4])  :
    cdef int transposed_board[4][4]
    cdef int i,j
    for i in range( 4):
        for j in range( 4):
            transposed_board[i][j] = board[i][j]
    transpose(transposed_board)
    reward = move_left(transposed_board, new_board)
    transpose(new_board)
    return reward

cdef int move_down(int board[4][4], int new_board[4][4])  :
    cdef int transposed_board[4][4]
    cdef int i,j
    for i in range( 4):
        for j in range( 4):
            transposed_board[i][j] = board[i][j]
    transpose(transposed_board)
    reward = move_right(transposed_board, new_board)
    transpose(new_board)
    return reward

cdef bint boards_equal(int b1[4][4], int b2[4][4])  :
    cdef i, j
    for i in range( 4):
        for j in range( 4):
            if b1[i][j] != b2[i][j]:
                return False
    return True

cdef bint can_move(int board[4][4])  :
    for i in range( 4):
        for j in range( 4):
            if j + 1 <  4 and board[i][j] == board[i][j+1]:
                return True
            if i + 1 <  4 and board[i][j] == board[i+1][j]:
                return True
    return False

cdef bint reached_target(int board[4][4])  :
    cdef int i, j
    for i in range( 4):
        for j in range( 4):
            if board[i][j] >= TARGET:
                return True
    return False

cdef int count_empty(int board[4][4])  :
    cdef int empty_count = 0
    cdef int i, j
    for i in range( 4):
        for j in range( 4):
            if board[i][j] == 0:
                empty_count += 1
    return empty_count

cdef int calculate_smoothness(int board[4][4])  :
    """计算平滑度：相邻单元格差值的总和（差值越小越好）"""
    cdef int smooth = 0
    cdef int i, j
    for i in range( 4):
        for j in range( 4 - 1):
            smooth -= abs(board[i][j] - board[i][j+1])
    for j in range( 4):
        for i in range( 4 - 1):
            smooth -= abs(board[i][j] - board[i+1][j])
    return smooth

cdef int calculate_monotonicity(int board[4][4])  :
    """计算单调性：如果行或列单调性好则奖励"""
    cdef int mono_rows = 0
    cdef int mono_cols = 0
    cdef int i, j
    cdef bint is_increasing = True
    cdef bint is_decreasing = True
    for i in range( 4):
        

        for j in range( 4 - 1):
            if board[i][j] < board[i][j + 1]:
                is_decreasing = False
            if board[i][j] > board[i][j + 1]:
                is_increasing = False

        if is_increasing or is_decreasing:
            mono_rows += 1
    for j in range( 4):
        is_increasing = True
        is_decreasing = True

        for i in range( 4 - 1):
            if board[i][j] < board[i + 1][j]:
                is_decreasing = False
            if board[i][j] > board[i + 1][j]:
                is_increasing = False

        if is_increasing or is_decreasing:
            mono_cols += 1
    return mono_rows + mono_cols

cdef int evaluate_board(int board[4][4])  :
    cdef int empty = count_empty(board)
    cdef int smoothness = calculate_smoothness(board)
    cdef int monotonicity = calculate_monotonicity(board)
    # 权重矩阵（蛇形布局），鼓励大数集中在角落
    cdef int weights[4][4]
    weights[0][0] = 65536; weights[0][1] = 32768; weights[0][2] = 16384; weights[0][3] =  8192; 
    weights[1][0] =   512; weights[1][1] =  1024; weights[1][2] =  2048; weights[1][3] =  4096; 
    weights[2][0] =   256; weights[2][1] =   512; weights[2][2] =  1024; weights[2][3] =  2048; 
    weights[3][0] =   128; weights[3][1] =   256; weights[3][2] =   512; weights[3][3] =  1024; 
    cdef int weight_score = 0
    cdef int i, j
    for i in range( 4):
        for j in range( 4):
            weight_score += board[i][j] * weights[i][j]
    return weight_score + empty * 500 + smoothness * 5 + monotonicity * 100

cdef int get_dynamic_depth(int board[4][4]):
    """根据当前空格数量动态调整搜索深度：空格越少，局势越紧张，搜索深度加深"""
    cdef int empty_cells = count_empty(board)
    if empty_cells >= 6:
        return 2
    elif empty_cells >= 3:
        return 3
    else:
        return 4


cdef float expectimax_ab(int board[4][4], int depth, bint is_player, float alpha, float beta)  :
    cdef float best = - INFINITY
    cdef float a = alpha
    cdef int new_board[4][4]
    cdef float reward
    cdef float val
    cdef int i, j, _
    cdef int (*move_func[4])(int[4][4], int[4][4])
    move_func[0] = move_up
    move_func[1] = move_down
    move_func[2] = move_left
    move_func[3] = move_right
    cdef int tile, row, col
    cdef float prob
    cdef float total = 0.0
    cdef int empty_count = 0
    cdef int board_copy[4][4]
    if depth == 0 or not can_move(board):
        return evaluate_board(board)
    
    if is_player:
        for _ in range(4):
            for i in range( 4):
                for j in range( 4):
                    new_board[i][j] = 0
            reward = move_func[_](board, new_board)
            if boards_equal(board, new_board):
                continue
            val = expectimax_ab(new_board, depth - 1, False, a, beta)
            best = max(best, val)
            a = max(a, best)
            if a >= beta:
                break
        return best
    else:
        for i in range( 4):
            for j in range( 4):
                if board[i][j] == 0:
                    empty_count += 1
                    for tile, prob in [(2, 0.9), (4, 0.1)]:
                        for row in range( 4):
                            for col in range( 4):
                                board_copy[row][col] = board[row][col]
                        board_copy[i][j] = tile
                        val = expectimax_ab(board_copy, depth - 1, True, alpha, beta)
                        total += prob * val
        if empty_count == 0:
            return evaluate_board(board)
        else:
            return total / empty_count

cdef tuple choose_best_move(int board[4][4]):
    cdef int (*move_func[4])(int[4][4], int[4][4])
    move_func[0] = move_up
    move_func[1] = move_down
    move_func[2] = move_left
    move_func[3] = move_right
    cdef char moves[4]
    moves[0] = 'w'
    moves[1] = 's'
    moves[2] = 'a'
    moves[3] = 'd'
    cdef char best_move = '?'
    cdef float best_score = -INFINITY
    cdef int best_new_board[4][4]
    cdef int best_reward = 0
    cdef int row, col
    cdef int new_board[4][4]
    cdef int reward
    cdef int i, depth
    cdef float score
    depth = get_dynamic_depth(board)
    for i in range(4):
        reward = move_func[i](board, new_board)
        if boards_equal(board, new_board):
            continue
        score = expectimax_ab(new_board, depth, False, -INFINITY, INFINITY)
        if score > best_score:
            best_score = score
            best_move = moves[i]
            for row in range( 4):
                for col in range( 4):
                    best_new_board[row][col] = new_board[row][col]
            best_reward = reward
    for row in range( 4):
        for col in range( 4):
            board[row][col] = best_new_board[row][col]
    return chr(best_move), best_reward

def main():
    t1 = time.time()
    cdef int board[4][4]
    init_board(board)
    score = 0
    has_printed_win = False
    print_board(board)
    while True:
        if not can_move(board):
            print("游戏结束，无法移动！")
            break
        move, reward = choose_best_move(board)
        print(move)
        score += reward
        if move == '?':
            print("没有有效移动，游戏结束！")
            break
        print(f"自动移动：{move}", score)
        add_random_tile(board)
        print_board(board)
        if reached_target(board) and not has_printed_win:
            print("已达到 2048！继续自动移动以获得更高分数。")
            has_printed_win = True
    t2 = time.time()
    print(t2 - t1)

```









