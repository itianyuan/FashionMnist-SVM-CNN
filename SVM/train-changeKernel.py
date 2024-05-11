import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import LinearSVM, SVMWithSigmoidKernel
import matplotlib.pyplot as plt

# 加载fashion-mnist数据集
train = pd.read_csv('../data/fashion/fashionmnist_train.csv')
df_train = train.copy()

# 分离data和label
X = df_train.iloc[:, 1:].values
Y = df_train.iloc[:, 0].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 切分训练集验证集
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=0)

# 转换为PyTorch的张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.int64)

# 创建LinearSVM模型
input_dim = x_train.shape[1]
num_classes = len(np.unique(y_train))
linear_svm_model = LinearSVM(input_dim, num_classes)

# 创建SVMWithSigmoidKernel模型
sigmoid_kernel_model = SVMWithSigmoidKernel(input_dim, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
linear_optimizer = torch.optim.SGD(linear_svm_model.parameters(), lr=0.01)
sigmoid_optimizer = torch.optim.SGD(sigmoid_kernel_model.parameters(), lr=5)

# 训练LinearSVM模型
linear_losses = []
for epoch in range(1000):
    print("range"+str(epoch))
    # 前向传播
    outputs = linear_svm_model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播与优化
    linear_optimizer.zero_grad()
    loss.backward()
    linear_optimizer.step()

    linear_losses.append(loss.item())  # 保存每个epoch的损失值

# 训练SVMWithSigmoidKernel模型
sigmoid_losses = []
for epoch in range(1000):
    print("range"+str(epoch))
    # 前向传播
    outputs = sigmoid_kernel_model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播与优化
    sigmoid_optimizer.zero_grad()
    loss.backward()
    sigmoid_optimizer.step()

    sigmoid_losses.append(loss.item())  # 保存每个epoch的损失值

# 绘制损失曲线
plt.plot(linear_losses, label='LinearSVM')
plt.plot(sigmoid_losses, label='SVMWithSigmoidKernel')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()