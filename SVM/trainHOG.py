import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.svm import LinearSVC
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

# 提取HOG特征
def extract_hog_features(images):
    hog_features = []
    for image in images:
        # Reshape the image to 2D
        gray_image = image.reshape(28, 28)
        # Extract HOG features
        features = hog(gray_image, orientations=9, pixels_per_cell=(4, 4),
                       cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
        hog_features.append(features)
    return np.array(hog_features)

x_train_hog = extract_hog_features(x_train)
x_val_hog = extract_hog_features(x_val)

# 创建线性支持向量机模型
model = LinearSVC()

# 训练模型
model.fit(x_train_hog, y_train)

# 评估模型
train_accuracy = model.score(x_train_hog, y_train)
val_accuracy = model.score(x_val_hog, y_val)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# 保存模型
import joblib
joblib.dump(model, 'svm_model.pkl')
