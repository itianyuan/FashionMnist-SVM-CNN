import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dic import class_names
from skimage.feature import hog
from skimage import feature, exposure

# 加载fashion-mnist数据集
train = pd.read_csv('../data/fashion/fashionmnist_train.csv', header=None)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(train.iloc[:, 1:].values)

# 提取HOG特征
def extract_hog_features(image):
    # 将图像重新调整为28x28
    image = image.reshape(28, 28)
    # 提取HOG特征
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys",
                              visualize=True)
    return hog_image

# 随机选择每个类别的一张图像，并可视化其原图和HOG特征
class_labels = np.unique(train.iloc[:, 0].values)
num_classes = len(class_labels)
plt.figure(figsize=(15, 20))

for i, label in enumerate(class_labels):
    # 随机选择该类别的一张图像
    image_indices = np.where(train.iloc[:, 0].values == label)[0]
    random_index = np.random.choice(image_indices)
    image = X[random_index]

    # 提取HOG特征
    hog_image = extract_hog_features(image)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # 显示原始图像
    plt.subplot(num_classes, 2, i*2 + 1)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Class {class_names[label]} - Original Image')
    plt.axis('off')

    # 显示HOG特征图
    plt.subplot(num_classes, 2, i*2 + 2)
    plt.imshow(hog_image_rescaled, cmap='gray')
    plt.title(f'Class {class_names[label]} - HOG Feature Image')
    plt.axis('off')

plt.tight_layout()
plt.show()
