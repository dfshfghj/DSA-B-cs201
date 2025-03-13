# 高维数据分类可视化



## 1.高维数据向低维投影

一般地，将一个$n$维线性空间中的点$x$投影至其$3$维子空间中，取$n$维空间中的线性无关向量$\alpha_1,\alpha_2,\alpha_3$为一组基，设其在基$\alpha_i$下的坐标为$c = (c_1, c_2, c_3)^{T}$，即
$$
x' = c_1 \alpha_1 + c_2 \alpha_2 + c_3 \alpha_3
$$
用矩阵形式表示
$$
x' = Ac
$$
其中$A = (\alpha_1, \alpha_2, \alpha_3)$.

由于投影要求距离$||x-Ac||$最小，使用最小二乘法
$$
A^{T}(x-Ac) = 0
$$
解出
$$
c = (A^{T}A)^{-1}A^{T}x
$$

## 2.最大化分离效果

由于投影损失了一部分信息，需要调整子空间的基使得数据点在其中分离现象最明显。

在$n$维空间中找到最佳分类超平面法向量为$n$，使选取的子空间与其正交。显然$n$应当作为一个基，这之后通过基$e_i$的Schmidt正交化得到剩余的基。



将$n$维空间中的数据点投影到$3$维空间中以供可视化，以$4$维的iris数据集为例：

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
'''
use SVM model to find the best hyperplane


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear', C=1.0)
#model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42, learning_rate_init=0.01, verbose=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print(model.coef_)
print(model.intercept_)
'''
def projection(x, coef):
    A, B = coef[0], coef[1]
    x0 = x[0] - A*(A*x[0] + B*x[1]) / (A**2 + B**2)
    x1 = x[1] - B*(A*x[0] + B*x[1]) / (A**2 + B**2)
    x2 = x[2]
    x3 = x[3]
    return (x1, x2, x3)
# result of model.coef_: normal vector of hyperplanes
coef_1 = [-0.42806316, 0.34038049, -0.88031519, -0.91871856]
coef_2 = [-0.06063568, 0.14174227, -0.54598541, -0.55220094]
coef_3 = [ 0.2076526, 0.78543726, -2.40938941, -2.11358406]
coef_x = [1, 1, 1, 1] # a simple example
coef = coef_1
x_a, y_a, z_a = [], [], []
for i in range(50):
    pos = projection(X[i], coef=coef)
    x_a.append(pos[0])
    y_a.append(pos[1])
    z_a.append(pos[2])
x_b, y_b, z_b = [], [], []
for i in range(50, 100):
    pos = projection(X[i], coef=coef)
    x_b.append(pos[0])
    y_b.append(pos[1])
    z_b.append(pos[2])
x_c, y_c, z_c = [], [], []
for i in range(100, 150):
    pos = projection(X[i], coef=coef)
    x_c.append(pos[0])
    y_c.append(pos[1])
    z_c.append(pos[2])
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(x_a, y_a, z_a, label='setosa')
ax.scatter(x_b, y_b, z_b, label='versicolor')
ax.scatter(x_c, y_c, z_c, label='virginica')
plt.legend()
'''
plot 4 axis of the origin coordinate if you want


axis_1 = projection([10, 0, 0, 0], coef=coef)
axis_2 = projection([0, 10, 0, 0], coef=coef)
axis_3 = projection([0, 0, 10, 0], coef=coef)
axis_4 = projection([0, 0, 0, 10], coef=coef)
ax.plot([0, axis_1[0]], [0, axis_1[1]], [0, axis_1[2]])
ax.plot([0, axis_2[0]], [0, axis_2[1]], [0, axis_2[2]])
ax.plot([0, axis_3[0]], [0, axis_3[1]], [0, axis_3[2]])
ax.plot([0, axis_4[0]], [0, axis_4[1]], [0, axis_4[2]])'
'''
plt.show()
```

![visual_iris.eps](https://raw.githubusercontent.com/dfshfghj/DSA-B-cs201/refs/heads/main/img/visual_iris.svg) 
