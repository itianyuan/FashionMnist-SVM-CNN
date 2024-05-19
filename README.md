# FashionMNIST分类任务

该仓库用于存放机器学习课程小组作业的代码

本代码仓库包含了两个部分，分别使用SVM和CNN对FashionMNIST数据集进行分类。

## 参与人员及分工

- 邹培源
- 王心怡

## 主要文件结构

```
root
│  convertCSV.py	# 将数据集转换为CSV格式
│  requirements.txt	# 依赖库版本
├─DataAndGraph    # 测试结果（数据和图表）
├─CNN	# CNN
│      train.py	# CNN训练及测试
│      trainPic.py	# 改变学习率，可视化loss的CNN
├─data	# 数据集
│  └─fashion	# 数据集源文件（不含CSV，需手动生成）
│          t10k-images-idx3-ubyte
│          t10k-labels-idx1-ubyte
│          train-images-idx3-ubyte
│          train-labels-idx1-ubyte
└─SVM	# SVM
    │  dic.py	# 数据集字典
    │  model.py	# 类定义
    │  svm_model.pth	# 网络参数
    │  test.py	# 测试程序
    │  testPic.py	# 测试程序（含可视化）
    │  train.py	# 训练程序（使用像素值特征）
    │  trainHOG.py	# 训练程序（使用HOG特征）
    │  trainSIFT.py	# 训练程序（使用SIFT特征）
    │  train-changeLR.py	# 改变学习率训练
    │  train-changeKernel.py	# 改变核函数训练
```

## 数据集

FashionMNIST数据集是一个包含60,000个训练样本和10,000个测试样本的数据集，用于图像分类任务。每个样本都是一个28x28像素的灰度图像，属于10个不同的类别之一，包括T恤、裤子、套衫、连衣裙等。

数据集包含在本代码仓库中，但代码中所使用的CSV格式文件由于文件太大，未包含在仓库中。请执行convertCSV.py生成该文件。

数据集下载链接：[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)

## 使用说明

1. 克隆本代码仓库：

```
git clone https://github.com/itianyuan/FashionMnist-SVM-CNN.git
```
2. 下载FashionMNIST数据集，并将其解压缩后放置于相应的数据文件夹中。
3. 分别运行两个模型的代码，包括train和test。

## 环境要求

- Python 3.8
- matplotlib\==3.4.3
  numpy\==1.22.3
  pandas\==1.3.4
  scikit_learn\==0.24.2
  torch\==2.2.1
