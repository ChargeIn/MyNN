# Florian Plesker 2026588

import numpy as np
from PIL import Image


# Describing the shape of an tensor
class Shape:

    def __init__(self, arr):
        self.shapeArray = arr


# Representation of the data
class Tensor:

    def __init__(self, shape, elements):
        self.elements = elements
        self.shape = shape


# Converts pictures to tensors
class InputLayer:

    @staticmethod
    def forward(img):
        img = img.reshape((-1, 1))
        return Tensor(Shape(img.shape), img)


# Representation of Layers

class FullyConnected:

    def __init__(self, shape_out, bias):
        self.weights = np.random.rand(1, shape_out)
        self.bias = bias

    def get_weights(self):
        return self.weights

    def forward(self, inputs):
        print(inputs.elements.shape)
        print(self.weights.shape)
        out = np.matmul(inputs.elements, self.weights) + self.bias


class ReLuLayer:

    @staticmethod
    def forward(inputs):
        return np.maximum(inputs, 0)


class Softmax:

    @staticmethod
    def forward(inputs):
        exp = np.exp(inputs)
        return exp/np.sum(exp)
