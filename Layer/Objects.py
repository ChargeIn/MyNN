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
        return img


# Representation of Layers

class FullyConnected:

    def __init__(self, shape_in, shape_out, bias):
        self.inputs = np.zeros(shape_in)
        self.weights = np.random.rand(shape_out, shape_in[0])
        self.bias = bias

    def get_weights(self):
        return self.weights

    def forward(self, inputs):
        self.inputs = inputs
        return np.matmul(self.weights, inputs) + self.bias

    def backward(self, lastlayer):
        self.deltas = lastlayer
        return np.matmul(lastlayer, np.transpose(self.weights))


class ReLuLayer:

    def __init__(self, shape):
        self.inputs = np.zeros(shape)

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(inputs, 0)

    def backward(self, lastlayer):
        return np.matmul(np.diag(np.where(self.inputs > 0, 1, 0)), lastlayer)


class Softmax:

    def __init__(self, shape):
        self.inputs = np.zeros(shape)

    def forward(self, inputs):
        self.inputs = inputs
        exp = np.exp(inputs)
        return exp/np.sum(exp)

    def backward(self, lastlayer):
        return np.matmul(lastlayer, np.diagflat(self.inputs) - np.matmul(self.inputs, np.transpose(self.inputs)))

