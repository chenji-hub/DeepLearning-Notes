# Kaggle 深度学习实战笔记

## 01_Kaggle_单神经元实战笔记 (A Single Neuron)

💡 **实战目标**：使用 TensorFlow/Keras 框架，为一个红酒质量数据集构建最基础的线性模型（单个神经元），体会理论在代码中的实现。

### 1. 核心代码实现
在 Keras 中搭建一个接受 11 个输入特征，输出 1 个预测结果的单层神经网络：

```python
from tensorflow import keras
from tensorflow.keras import layers

# 创建一个带有 1 个线性单元的网络
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[11])
])
```

---

## 02_Kaggle_构建深度神经网络 (Deep Neural Networks)

本篇文档是基于 Kaggle [Intro to Deep Learning] 课程第一部分的学习总结。主要探讨了深度神经网络的核心组件：全连接层与激活函数，并基于 `Concrete`（混凝土）数据集进行了实际的代码构建。

### 1. 核心理论：深度神经网络的基石

深度学习的强大之处在于**模块化 (Modularity)**，即将简单的功能单元堆叠成复杂的网络，从而学习数据中极其复杂的非线性关系。

#### 1.1 全连接层 (Dense Layers)
神经网络将神经元组织成“层”。当我们将一组接收相同输入的线性单元集合在一起时，就得到了一个**全连接层（稠密层，Dense Layer）**。
* **物理意义**：你可以将每一层看作是执行某种数据转换。通过构建极深的网络层叠结构，模型可以一步步将原始数据提炼成更高级的特征。
* **局限性**：如果仅仅堆叠全连接层，而没有任何其他操作，其效果等同于单个线性层，永远无法脱离“直线和平面”的线性世界。

#### 1.2 激活函数 (Activation Functions)
为了拟合现实世界中复杂的曲线和非线性规律，我们必须引入**激活函数**。最经典的是整流函数（Rectified Linear Unit，简称 **ReLU**）。
* **ReLU 函数**：$f(x) = \max(0, x)$。它将负数部分截断为 0，正数部分保持原样。这种在数据中制造“弯折”的特性，赋予了神经网络非线性的表达能力。

---

### 2. 实战演练：预测混凝土抗压强度

在理解了理论之后，我们使用 TensorFlow 和 Keras 构建了一个真实的神经网络架构，目标是根据 8 种不同的配方特征，预测混凝土的最终抗压强度（典型的**回归任务**）。

#### 2.1 确定输入形状 (Input Shape)
在构建模型时，必须显式告诉模型输入特征的数量。`Concrete` 数据集剥离目标列后，还有 8 列特征，因此输入形状定为 `[8]`。

```python
# 明确输入维度
input_shape = [8]
```

#### 2.2 搭建多层顺序模型 (Sequential Model)
使用 `keras.Sequential` 容器，按数据流动的顺序依次堆叠网络层：
* 包含三个拥有 512 个神经元的庞大隐藏层，均使用 ReLU 激活函数。
* 最后一层为输出层，由于是回归任务（预测具体数值），因此只保留 1 个神经元且**不使用**任何激活函数（保持纯线性输出）。

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # 第一层隐藏层：指定神经元数量、激活函数，并必须定义 input_shape
    layers.Dense(units=512, activation='relu', input_shape=[8]),
    
    # 隐藏层 2 和 3：Keras 会自动推断输入维度，无需再写 input_shape
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    
    # 线性输出层（回归任务专用）
    layers.Dense(units=1)
])
```

#### 2.3 进阶代码技巧：激活层解耦
通常，激活函数作为 `Dense` 层的参数传入。但在更高级的网络架构中（如未来需要插入 Batch Normalization 层），我们需要将线性计算和非线性激活拆分为两个独立的层。二者在逻辑上完全等效，但拆分后更为灵活：

```python
model = keras.Sequential([
    # 第一层：拆分为独立的 Dense 计算层 + 独立的 Activation 激活层
    layers.Dense(units=32, input_shape=[8]),
    layers.Activation('relu'),
    
    # 第二层：同样拆分
    layers.Dense(units=32),
    layers.Activation('relu'),
    
    # 输出层保持不变
    layers.Dense(units=1)
])
```

#### 2.4 探索 ReLU 的高级替代方案
虽然 ReLU 表现优异，但在处理负数时直接归零可能会导致“神经元死亡”。Keras 提供了诸如 `elu`、`selu` 和 `swish` 等变体，它们在负半轴允许平滑的非零输出，有助于加速特定模型的收敛。

我们可以将 Keras 中的层实例化后当做函数调用，以可视化这些高级激活函数的实际数学曲线：

```python
import tensorflow as tf
import matplotlib.pyplot as plt

# 实例化一个 ELU (指数线性单元) 激活层
activation_layer = layers.Activation('elu')

# 生成 -3.0 到 +3.0 的 100 个测试点
x = tf.linspace(-3.0, 3.0, 100)
# 将数据通过 ELU 层进行转换 (层可以像函数一样被调用)
y = activation_layer(x) 

# 使用 Matplotlib 绘制函数曲线
plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("ELU Activation Function")
plt.show()
```

