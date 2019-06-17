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

    def __init__(self, shape):
        self.shape = shape

    def forward(self, img):
        self.shape = img.shape
        img = img.reshape((1, -1))
        return img

    def backprop(self, lastlayer):
        return lastlayer.reshape(self.shape)

    def update(self, rate, batchsize):
        return


# Representation of Layers
class Conv2DLayer:

    def __init__(self, filters, bias):
        self.filters = Tensor(filters)
        self.bias = Tensor(bias)
        self.inputs = {}

    def forward(self, inputs):
        self.inputs["in"] = inputs
        d1 = inputs.shape[0] - self.filters.shape[2] + 1
        d2 = inputs.shape[1] - self.filters.shape[3] + 1
        out = np.zeros([self.filters.shape[0], d1, d2])
        for i in range(0, self.filters.shape[0]):
            for j in range(0, self.filters.shape[1]):
                out[i, :, :] += sc.convolve2d(inputs, self.filters[i, j, :, :], mode="valid")

        return out + self.bias

    def backprop(self, lastlayer):
        print(lastlayer.shape)
        newfilter = np.transpose(self.filters, [1, 0, 2, 3])
        out = np.zeros(self.inputs["in"].shape)
        for i in range(0, newfilter.shape[0]):
            for j in range(0, newfilter.shape[1]):
                newfilter[i, j, :, :] = np.rot90(newfilter[i, j, :, :], 2)
                out += sc.convolve2d(lastlayer[i], newfilter[i, j, :, :])

        return out

    def update(self, rate, batchsize):



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
