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
* **ReLU 函数**：f(x) = max(0, x)。它将负数部分截断为 0，正数部分保持原样。这种在数据中制造“弯折”的特性，赋予了神经网络非线性的表达能力。

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

## 03_Kaggle_随机梯度下降与模型训练 (Stochastic Gradient Descent)

💡 **实战目标**：理解神经网络是如何“学习”的，掌握损失函数、优化器的概念，并探究批大小 (Batch Size) 和学习率 (Learning Rate) 对训练过程的深刻影响。

### 1. 核心理论：让模型学会学习
当网络刚刚搭建好时，内部的权重是随机初始化的。我们需要通过**训练 (Training)** 来调整这些权重，使其具备预测能力。

* **损失函数 (Loss Function)**：网络的“评分标准”。它衡量模型预测值与真实目标之间的误差。回归任务中最常用的是 **MAE (平均绝对误差)**。损失函数的存在就是为了告诉模型：“你的终极目标是让这个误差降到最低”。
* **优化器 (Optimizer)**：网络的“纠错引擎”。它根据损失函数的反馈，决定如何微调权重。深度学习中最广泛使用的是**随机梯度下降 (SGD)** 及其最强大的变体 **Adam**（自带自适应学习率，无需繁琐调参）。

### 2. 训练机制：Epoch 与 Batch
模型并不是一次性看遍所有数据，而是分批次、分周期地进行学习：
* **小批量 (Minibatch / Batch Size)**：模型每次微调权重前，抓取的数据样本数量。
* **训练周期 (Epoch)**：模型将整个训练数据集完整看一遍的过程。

### 3. 实战演练：编译、训练与评估
基于汽车燃油经济性 (Fuel Economy) 数据集，在数据预处理完成后，我们进行完整的模型训练流程：

#### 3.1 编译模型 (Compile)
使用 `model.compile()` 为模型装配优化器和损失函数：
```python
model.compile(
    optimizer='adam',
    loss='mae'
)
```

#### 3.2 启动训练 (Fit)
将特征数据 `X` 和目标数据 `y` 喂给模型，设定批大小和训练周期，并将训练过程的各种指标记录在 `history` 对象中：
```python
history = model.fit(
    X, y,
    validation_data=(X,y),
    batch_size=128,
    epochs=200
)
```
> 在真实的机器学习项目中，必须使用验证集来防止模型“死记硬背”（过拟合）。可以使用 `validation_data=(X_valid, y_valid)` 或直接加入参数 `validation_split=0.2`（让 Keras 自动划出 20% 的验证集）。

#### 3.3 绘制并评估损失曲线 (Loss Curve)
将 `history` 转换为 Pandas DataFrame 并绘制折线图，以直观观察模型的学习进度。

**进阶技巧**：使用 `.loc[5:]` 跳过前 5 轮误差极大的初始阶段，可以防止图表的 Y 轴比例尺被撑爆，从而更清晰地观察后期平缓区域的收敛细节。

```python
import pandas as pd

history_df = pd.DataFrame(history.history)
# 跳过前 5 轮绘制曲线，放大并观察后期收敛细节
history_df.loc[5:, ['loss']].plot();
```
**评估标准**：当损失曲线逐渐从下降转为“水平铺平 (levelled off)”时，说明模型已收敛到了当前的极限，此时再继续增加 Epoch 也不会产生有益的优化了。

### 4. 核心洞察：超参数的权衡 (Trade-offs)
通过调节 SGD 模拟器，我们得出了深度学习调参的两大黄金法则：

1. **批大小 (Batch Size) 决定“平滑度”**：
   * **Batch 太小**：每次采样的随机噪点多，更新方向震荡剧烈（曲线像心电图），但有时能帮模型跳出局部最优解。
   * **Batch 很大**：更新方向极度平滑稳定，目标明确，但每轮计算更耗时。
2. **学习率 (Learning Rate) 决定“步伐大小”**：
   * **学习率太小**：极其安全、一定能下坡，但需要极长的训练时间才能走到谷底。
   * **学习率太大**：走得飞快，但容易在最佳答案两侧反复横跳（Overshoot），甚至导致训练直接飞跃谷底，模型彻底崩溃。

## 04_Kaggle_过拟合与欠拟合 (Overfitting and Underfitting)

💡 **实战目标**：理解深度学习中的核心权衡（信号与噪声），掌握模型容量 (Capacity) 对拟合状态的影响，并学会使用早停法 (Early Stopping) 在最佳平衡点终止训练，防止模型过度拟合。

### 1. 核心理论：信号与噪声的博弈
在训练神经网络时，训练数据中包含两类截然不同的信息：
* **信号 (Signal)**：代表数据的潜在真实规律。它是能够**泛化 (Generalize)** 的部分，可以帮助模型对从未见过的全新数据做出准确预测。
* **噪声 (Noise)**：仅在训练数据中存在的随机波动和偶然模式。它对预测新数据毫无价值，模型如果过度学习噪声，就会导致泛化能力下降。

我们通过观察**训练损失 (Training Loss)** 和**验证损失 (Validation Loss)** 的学习曲线来识别两者的博弈：
* **学习信号时**：训练损失和验证损失都会持续下降。
* **学习噪声时**：训练损失会继续走低，但验证损失会止跌回升。两条曲线之间拉开的**差距 (Gap)**，直接反映了模型学到了多少噪声。

### 2. 拟合机制：欠拟合与过拟合
这种权衡关系在模型训练中会引发两个经典的拟合问题：
* **欠拟合 (Underfitting)**：由于模型没有学习到足够的信号，导致训练集和验证集的损失都无法降到理想水平。这通常是因为模型结构过于简单，无法捕捉复杂的规律。
* **过拟合 (Overfitting)**：由于模型学习了太多的噪声，导致其在训练集上表现极佳，但在验证集上误差反弹。这通常是因为模型记忆力过剩，或者训练时间过长。

### 3. 实战演练：数据防漏、编译与早停控制
基于 Spotify 数据集预测歌曲受欢迎程度的任务，我们在构建高容量网络时，进行完整的防御性训练流程：

#### 3.1 引入分组拆分防止信号泄露 (Grouped Split)
在切分训练集和验证集时，如果使用传统的随机打乱，同一歌手的不同歌曲会同时散落两端，导致模型靠“认歌手”作弊。我们使用 `GroupShuffleSplit` 按歌手进行分组拆分，确保同一歌手的所有歌曲要么全在训练集，要么全在验证集，迫使模型学习普适的音频特征：

```python
from sklearn.model_selection import GroupShuffleSplit

# 按歌手特征进行分组切分
splitter = GroupShuffleSplit(train_size=0.75)
train, test = next(splitter.split(X, y, groups=artists))
X_train, X_valid = X.iloc[train], X.iloc[test]
y_train, y_valid = y.iloc[train], y.iloc[test]
```

#### 3.2 配置早停法回调函数 (Early Stopping)
使用 `callbacks.EarlyStopping` 定义一个监控探头，为模型的决策过程设定容错余地：

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001,           # 最小改善阈值：验证损失至少下降 0.001 才算有效进步
    patience=5,                # 耐心值：容忍连续 5 个 epoch 验证损失没有实质改善
    restore_best_weights=True  # 恢复权重：强行停止后，自动回滚到验证损失最低那一轮的参数
)
```

#### 3.3 启动带早停监控的训练 (Fit)
将早停回调函数以列表形式传入 `model.fit()`。此时我们可以放心地将上限设为 50 轮甚至更高，系统会自动判定何时掐断训练：

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping]  # 传入早停回调函数
)
```

### 4. 核心洞察：模型容量与决策权衡 (Trade-offs)
通过在低容量（线性模型）与高容量（多层深网络）之间进行调参对比，我们得出了深度学习拟合控制的两大黄金法则：

1. **模型容量 (Capacity) 决定拟合上限**：
   * **容量太低**：如无隐藏层的线性模型（`layers.Dense(1)`），只能拟合一条直线，无法处理复杂的非线性关系，必然导致**欠拟合**。
   * **容量过高**：通过增加网络深度（层数）和宽度（神经元数量）并配合 `relu` 激活函数，模型能捕捉极其复杂的规律，但记忆力过强也会使其极易走向**过拟合**。
2. **早停法参数决定“容错余地”**：
   * **Patience 设定过小**：模型会缺乏耐心，容易把批次数据带来的随机波动（Random Batch Variation）误判为过拟合，从而导致训练过早误停。
   * **黄金调优策略**：深度学习的最佳实践是“给模型足够大的容量使其能够充分拟合，再配合合理的早停参数，在验证损失的最低点（即信号与噪声交换的最完美平衡点）安全刹车并自动回档”。
