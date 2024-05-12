import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dic import class_names

# 加载fashion-mnist数据集
train = pd.read_csv('../data/fashion/fashionmnist_train.csv', header=None)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(train.iloc[:, 1:].values)

# 提取SIFT特征
def extract_sift_keypoints(image):
    # 将图像重新调整为28x28
    image = image.reshape(28, 28)
    # 将图像转换为8位无符号整数类型
    image = (image * 255).astype(np.uint8)
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    # 检测关键点
    keypoints, _ = sift.detectAndCompute(image, None)
    # 将关键点绘制在图像上
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    return image_with_keypoints

# 随机选择每个类别的一张图像，并可视化其原图和SIFT关键点
class_labels = np.unique(train.iloc[:, 0].values)
num_classes = len(class_labels)
plt.figure(figsize=(15, 20))

for i, label in enumerate(class_labels):
    # 随机选择该类别的一张图像
    image_indices = np.where(train.iloc[:, 0].values == label)[0]
    random_index = np.random.choice(image_indices)
    image = X[random_index]

    # 提取SIFT关键点
    image_with_keypoints = extract_sift_keypoints(image)

    # 显示原始图像
    plt.subplot(num_classes, 2, i*2 + 1)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Class {class_names[label]} - Original Image')
    plt.axis('off')

    # 显示SIFT关键点
    plt.subplot(num_classes, 2, i*2 + 2)
    plt.imshow(image_with_keypoints, cmap='gray')
    plt.title(f'Class {class_names[label]} - Image with SIFT Keypoints')
    plt.axis('off')

plt.tight_layout()
plt.show()
