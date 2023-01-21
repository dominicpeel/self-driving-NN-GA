import numpy as np


class Network:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagate(self, a0):
        self.layers = [a0]
        for index, weight in enumerate(self.weights):
            self.layers.append(self.sigmoid(
                np.dot(weight, self.layers[index]) + self.bias[index]))
        return self.layers[-1]
