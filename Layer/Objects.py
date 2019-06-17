# Florian Plesker 2026588

import numpy as np
import scipy.signal as sc


# Describing the shape of an tensor
class Shape:

    def __init__(self, arr):
        self.shapeArray = arr


# Representation of the data
class Tensor:

    def __init__(self, elements):
        self.elements = np.array(elements)
        self.deltas = {}

    def get_deltas(self):
        if "deltas" not in self.deltas:
            return np.zeros(self.elements.shape)
        return self.deltas["deltas"]

    def add_deltas(self, update):
        self.deltas["deltas"] = self.get_deltas() + update

    def update(self, rate, batchsize):
        self.elements = self.elements - rate * self.get_deltas()/batchsize
        self.deltas = {}


# Converts pictures to tensors
class InputLayer:

    @staticmethod
    def forward(img):
        img = img.reshape((1, -1))
        return img


# Representation of Layers
class Conv2DLayer:

    def __init__(self, filters, count):
        self.filters = filters
        self.count = count

    def forward(self, inputs):
        out = np.zeros([int(inputs.shape[0] - np.floor(self.filters.shape[1]/2) + 1), int(inputs.shape[1]
                        - np.floor(self.filters.shape[2]/2) + 1, self.count)])
        for i in range(0, self.count):
            for j in range(0, self.filters.shape[3]):
                out[i, :, :] += sc.convolve2d(inputs, self.filters[i, j, :, :])

        return out

    #def backprop(self, lastlayer):



class FullyConnected:

    def __init__(self, shape_in, shape_out, bias):
        self.inputs = Tensor(np.zeros(shape_in))
        self.weights = Tensor(np.random.rand(shape_in[1], shape_out))
        self.bias = Tensor(bias)

    def get_weights(self):
        return self.weights

    def forward(self, inputs):
        self.inputs.elements = inputs
        return np.matmul(inputs, self.weights.elements) + self.bias.elements

    def backprop(self, lastlayer):
        self.bias.add_deltas(lastlayer)
        self.weights.add_deltas(np.matmul(np.transpose(self.inputs.elements), lastlayer))
        return np.matmul(lastlayer, np.transpose(self.weights.elements))

    def update(self, rate, batchsize):
        self.bias.update(rate, batchsize)
        self.weights.update(rate, batchsize)


class ReLuLayer:

    def __init__(self, shape):
        self.inputs = Tensor(np.zeros(shape))

    def forward(self, inputs):
        self.inputs.elements = inputs
        return np.maximum(inputs, 0)

    def backprop(self, lastlayer):
        return np.matmul(lastlayer, np.diag(np.where(self.inputs.elements > 0, 1, 0).flatten()))

    def update(self, rate, batchsize):
        return


class Softmax:

    def __init__(self, shape):
        self.inputs = Tensor(np.zeros(shape))

    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1))
        self.inputs.elements = exp/np.sum(exp, axis=1)
        return self.inputs.elements

    def backprop(self, lastlayer):
        s = self.inputs.elements.reshape(-1, 1)
        return np.matmul(lastlayer, np.diagflat(s) - np.dot(s, s.T))

    def update(self, rate, btachsize):
        return
