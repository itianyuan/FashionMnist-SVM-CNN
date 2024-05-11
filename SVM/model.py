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