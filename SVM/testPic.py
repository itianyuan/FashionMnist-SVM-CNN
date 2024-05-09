import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import LinearSVM
from dic import class_names  # 导入类别名字的字典
import matplotlib.pyplot as plt

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
model = LinearSVM(input_dim, num_classes)

# 加载模型权重
model.load_state_dict(torch.load('svm_model.pth'))

# 测试模型
with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print('Test Accuracy:', accuracy)

# 可视化预测结果
correct_indices = (predicted == y_test_tensor).nonzero().squeeze().numpy()
incorrect_indices = (predicted != y_test_tensor).nonzero().squeeze().numpy()

# 随机选择一些正确分类的样本
num_samples = 10
correct_samples = X_test[correct_indices][:num_samples]
correct_labels = Y_test[correct_indices][:num_samples]

# 随机选择一些错误分类的样本
incorrect_samples = X_test[incorrect_indices][:num_samples]
incorrect_labels = Y_test[incorrect_indices][:num_samples]

# 可视化正确分类的样本
plt.figure(figsize=(10, 5))
for i in range(num_samples):
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(correct_samples[i].reshape(28, 28), cmap='gray')
    plt.title(f'True: \n{class_names[correct_labels[i]]}')
    plt.axis('off')

# 可视化错误分类的样本
for i in range(num_samples):
    plt.subplot(2, num_samples, num_samples + i + 1)
    plt.imshow(incorrect_samples[i].reshape(28, 28), cmap='gray')
    plt.title(f'Pred: \n{class_names[predicted[incorrect_indices[i]].item()]}\nTrue: \n{class_names[incorrect_labels[i]]}')
    plt.axis('off')

plt.tight_layout()
plt.show()
