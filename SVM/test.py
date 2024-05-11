import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import LinearSVM
from model import SVMWithSigmoidKernel

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
# model = SVMWithSigmoidKernel(input_dim, num_classes)


# 加载模型权重
model.load_state_dict(torch.load('svm_model.pth'))

# 测试模型
with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print('Test Accuracy:', accuracy)
