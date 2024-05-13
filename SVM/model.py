import torch
import torch.nn as nn


class LinearSVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

class SVMWithSigmoidKernel(nn.Module):
    def __init__(self, input_dim, num_classes, alpha=1.0, c=0.0):
        super(SVMWithSigmoidKernel, self).__init__()
        self.alpha = alpha
        self.c = c
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.randn(input_dim, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        # 计算核矩阵
        kernel_matrix = torch.tanh(self.alpha * torch.mm(x, self.weights) + self.c)
        # 计算类别得分
        scores = kernel_matrix + self.bias
        return scores

class SVMWithGaussianKernel(nn.Module):
    def __init__(self, input_dim, num_classes, gamma=1.0, c=0.0):
        super(SVMWithGaussianKernel, self).__init__()
        self.gamma = gamma
        self.c = c
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.randn(input_dim, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        # 计算核矩阵
        dist = torch.cdist(x, self.weights.T, p=2)  # 计算样本和支持向量之间的欧氏距离
        kernel_matrix = torch.exp(-self.gamma * dist ** 2)  # 高斯核函数的计算
        # 计算类别得分
        scores = kernel_matrix + self.bias
        return scores

class PolynomialSVM(nn.Module):
    def __init__(self, input_dim, num_classes, alpha=1.0, degree=1, bias=0.0):
        super(PolynomialSVM, self).__init__()
        self.alpha = alpha
        self.degree = degree
        self.bias = bias
        self.num_classes = num_classes
        # 权重参数
        self.weights = nn.Parameter(torch.randn(input_dim, num_classes))
        # 偏置参数
        self.bias_param = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        # 计算核矩阵
        kernel_matrix = torch.pow((self.alpha * torch.mm(x, self.weights) + self.bias), self.degree)
        # 计算类别得分
        scores = kernel_matrix + self.bias_param
        return scores