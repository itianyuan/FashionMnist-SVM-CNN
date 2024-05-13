import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import LinearSVM
from model import SVMWithSigmoidKernel
from model import SVMWithGaussianKernel
from model import PolynomialSVM

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import cv2

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

# 创建线性支持向量机模型
input_dim = x_train.shape[1]
num_classes = len(np.unique(y_train))

# model = LinearSVM(input_dim, num_classes)
# model = SVMWithSigmoidKernel(input_dim, num_classes)
# model = SVMWithGaussianKernel(input_dim, num_classes)
model = PolynomialSVM(input_dim, num_classes)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)

# 训练模型
num_epochs = 1000
losses = []
for epoch in tqdm(range(num_epochs)):
    # 前向传播
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())  # 保存每个epoch的损失值

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 绘制损失曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 保存模型
torch.save(model.state_dict(), 'svm_model.pth')



# 加载fashion-mnist测试集数据集
test = pd.read_csv('../data/fashion/fashionmnist_test.csv')
df_test = test.copy()

# 分离data和label
X_test = df_test.iloc[:, 1:].values
Y_test = df_test.iloc[:, 0].values

# 数据标准化
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# 转换为PyTorch的张量
x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(Y_test, dtype=torch.int64)

# 创建线性支持向量机模型
input_dim = X_test.shape[1]
num_classes = len(np.unique(Y_test))

# model = LinearSVM(input_dim, num_classes)
# model = SVMWithSigmoidKernel(input_dim, num_classes)
# model = SVMWithGaussianKernel(input_dim, num_classes)
model = PolynomialSVM(input_dim, num_classes)

# 加载模型权重
model.load_state_dict(torch.load('svm_model.pth'))

# 测试模型
with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print('Test Accuracy:', accuracy)