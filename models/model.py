import torch
from torch.nn import Module, Flatten, Linear


class NeuralNet(Module):
    """Build a simple toy model"""

    def __init__(self, n):
        super(NeuralNet, self).__init__()
        self.n = n
        self.flatten = Flatten()
        self.linear = Linear(n, n)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)
