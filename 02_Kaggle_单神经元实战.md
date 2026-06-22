# 02_Kaggle_单神经元实战笔记 (A Single Neuron)

> 💡 **实战目标**：使用 TensorFlow/Keras 框架，为一个红酒质量数据集构建最基础的线性模型（单个神经元），体会理论在代码中的实现。

## 1. 核心代码实现

在 Keras 中搭建一个接受 11 个输入特征，输出 1 个预测结果的单层神经网络：

```python
from tensorflow import keras
from tensorflow.keras import layers

# 创建一个带有 1 个线性单元的网络
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[11])
])