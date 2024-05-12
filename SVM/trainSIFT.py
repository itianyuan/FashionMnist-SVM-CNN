import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from model import LinearSVM
from model import SVMWithSigmoidKernel
import matplotlib.pyplot as plt

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

# 提取SIFT特征
def extract_sift_features(image):
    # 将图像重新调整为28x28
    image = image.reshape(28, 28)
    # 将图像转换为灰度图
    gray_image = image.astype(np.uint8)
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    # 检测关键点和计算描述符
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    # 如果检测到的关键点数量大于10，则只保留前10个关键点的描述符
    print(len(keypoints))
    if len(keypoints) > 10:
        keypoints = keypoints[:10]
        descriptors = descriptors[:10]
    # 如果没有检测到关键点，则返回全零数组
    if descriptors is None:
        descriptors = np.zeros((10, 128), dtype=np.float32)
    return descriptors

# 提取训练集和验证集的SIFT特征，并填充成相同的大小
max_descriptors = 10  # 提取的关键点数量
x_train_sift = np.array([np.vstack((extract_sift_features(image), np.zeros((max_descriptors - extract_sift_features(image).shape[0], 128)))) for image in x_train])
x_val_sift = np.array([np.vstack((extract_sift_features(image), np.zeros((max_descriptors - extract_sift_features(image).shape[0], 128)))) for image in x_val])


# 转换为PyTorch的张量
x_train_tensor = torch.tensor(x_train_sift, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
x_val_tensor = torch.tensor(x_val_sift, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.int64)

# 创建线性支持向量机模型
input_dim = max_descriptors * 128  # 更新输入维度
num_classes = len(np.unique(y_train))
model = SVMWithSigmoidKernel(input_dim, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 训练模型
num_epochs = 200
losses = []
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_train_tensor.view(-1, input_dim))  # 修改输入维度
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
# torch.save(model.state_dict(), 'svm_model.pth')

# 测试模型
with torch.no_grad():
    outputs = model(x_val_tensor.view(-1, input_dim))
    _, predicted = torch.max(outputs, 1)
    total = y_val_tensor.size(0)
    correct = (predicted == y_val_tensor).sum().item()
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')
