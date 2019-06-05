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
    def forward(img_name):
        img = Image.open("src/" + img_name)
        arr = np.array(img)
        return Tensor(Shape(arr.size), arr)


# Representation of Layers

class FullyConnected:

    def __init__(self, shape_out, bias):
        self.weights = np.array(shape_out, float)
        self.bias = bias

    def get_weights(self):
        return self.weights

    def forward(self, inputs):
        out = np.matmul(inputs, self.weights) + self.bias


class ReLuLayer:

    @staticmethod
    def forward(inputs):
        return np.maximum(inputs, 0)


class Softmax:

    @staticmethod
    def forward(inputs):
        exp = np.exp(inputs)
        return exp/np.sum(exp)
