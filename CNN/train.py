import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

from tqdm import tqdm, trange

import time


# 检查设备是否支持CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义FashionMNISTDataset类用于数据处理
class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28).astype(np.float32)
        self.Y = np.array(data.iloc[:, 0])
        del data
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]
        return (item, label)

# 定义超参数
LR = 0.001
EPOCHES = 25
BATCH_SIZE = 256

# 加载训练集和测试集数据
train_dataset = FashionMNISTDataset(csv_file='../data/fashion/fashionmnist_test.csv')
test_dataset = FashionMNISTDataset(csv_file='../data/fashion/fashionmnist_train.csv')

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        out = self.pool1(self.layer1(x))
        out = self.pool2(self.layer3(self.layer2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(16 * 12 * 12, 10)  # 这里的12 * 12是根据输入大小和池化层结果推断的

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 创建模型实例、损失函数和优化器
# model = SimpleCNN().to(device)
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 定义绘制损失函数的函数
def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, color='blue', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

# train函数
def train(model, train_loader, criterion, optimizer, num_epochs=EPOCHES):
    losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            losses.append(loss.item())
            if (i + 1) % 100 == 0:
                print('Epoch : %d/%d, Iter : %d/%d, Loss : %.4f' % (
                    epoch + 1, num_epochs,
                    i + 1, len(train_loader.dataset) // BATCH_SIZE,
                    loss.item()
                ))

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 绘制损失曲线
    plot_loss(losses)

# 定义测试函数
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float().to(device)
            outputs = model(images).cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('CNN acc：%.2f %%' % (100 * correct / total))

# 记录开始时间
start_time = time.time()
# 训练模型
train(model, train_loader, criterion, optimizer)
# 记录结束时间
end_time = time.time()

# 计算程序运行时间
execution_time = end_time - start_time
print("程序运行时间：", execution_time, "秒")


# 在测试集上评估模型性能
test(model, test_loader)