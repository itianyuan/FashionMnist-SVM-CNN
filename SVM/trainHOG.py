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
from skimage.feature import hog

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

# 提取HOG特征
def extract_hog_features(image):
    # 将图像重新调整为28x28
    image = image.reshape(28, 28)
    # 将图像转换为灰度图
    gray_image = image.astype(np.uint8)
    # 提取HOG特征
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
    return features

# 提取训练集和验证集的HOG特征
x_train_hog = np.array([extract_hog_features(image) for image in x_train])
x_val_hog = np.array([extract_hog_features(image) for image in x_val])

# 转换为PyTorch的张量
x_train_tensor = torch.tensor(x_train_hog, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
x_val_tensor = torch.tensor(x_val_hog, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.int64)

# 创建线性支持向量机模型
input_dim = x_train_hog.shape[1]  # 更新输入维度为HOG特征的长度
num_classes = len(np.unique(y_train))

# model = LinearSVM(input_dim, num_classes)
# model = SVMWithSigmoidKernel(input_dim, num_classes)
# model = SVMWithGaussianKernel(input_dim, num_classes)
model = PolynomialSVM(input_dim, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5)

# 训练模型
num_epochs = 1000
losses = []
for epoch in tqdm(range(num_epochs)):
    # 前向传播
    outputs = model(x_train_tensor)  # 不需要再修改输入维度
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

# 测试模型
with torch.no_grad():
    outputs = model(x_val_tensor)
    _, predicted = torch.max(outputs, 1)
    total = y_val_tensor.size(0)
    correct = (predicted == y_val_tensor).sum().item()
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')
