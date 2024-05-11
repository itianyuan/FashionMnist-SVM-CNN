import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import SVMWithSigmoidKernel
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

# 定义学习率列表
learning_rates = [0.01, 0.1, 1, 10]

# 训练模型并记录损失
losses_dict = {}

for lr in learning_rates:
    print("lr"+str(lr))
    model = SVMWithSigmoidKernel(input_dim=x_train.shape[1], num_classes=len(np.unique(y_train)))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    for epoch in range(200):
        print("range"+str(epoch))
        # 前向传播
        outputs = model(x_train_tensor)
        loss = nn.CrossEntropyLoss()(outputs, y_train_tensor)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # 保存每个epoch的损失值

    losses_dict[lr] = losses

# 绘制损失曲线
plt.figure(figsize=(10, 6))
for lr, losses in losses_dict.items():
    plt.plot(losses, label=f'LR={lr}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss with Different Learning Rates')
plt.legend()
plt.grid(True)
plt.show()