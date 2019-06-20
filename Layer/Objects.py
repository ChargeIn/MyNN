# Florian Plesker 2026588

import numpy as np
import scipy.signal as sc
import scipy.special as sp
import scipy.ndimage as con


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
        self.elements = self.elements - rate * self.get_deltas() / batchsize
        self.deltas = {}


# Converts pictures to tensors
class InputLayer:

    @staticmethod
    def forward(img):
        return np.array([img])

    @staticmethod
    def backprop(lastlayer):
        return lastlayer

    @staticmethod
    def update(rate, batchsize):
        return


# converts the elements between convLayer and fullyconnected
class ReshapeLyer:

    def __init__(self):
        self.shape = {}

    def forward(self, inputs):
        self.shape["s"] = inputs.shape
        return inputs.reshape((1, -1))

    def backprop(self, lastlayer):
        return lastlayer.reshape(self.shape["s"])

    @staticmethod
    def update(rate, batchsize):
        return


# Representation of Layers

class FullyConnected:

    def __init__(self, shape_in, shape_out):
        self.inputs = Tensor(np.zeros(shape_in))
        self.weights = Tensor(np.random.rand(shape_in[1], shape_out))
        self.bias = Tensor(np.ones(shape_out))

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


# Representation of Layers
class Conv2DLayer:

    def __init__(self, filters, bias):
        for i in range(0, filters.shape[0]):
            for j in range(0, filters.shape[1]):
                filters[i, j, :, :] = np.rot90(filters[i, j, :, :], 2)
        self.filters = Tensor(filters)
        self.bias = Tensor(bias)
        self.inputs = {}

    def forward(self, inputs):
        self.inputs["in"] = inputs
        d1 = inputs.shape[1] - self.filters.elements.shape[2] + 1
        d2 = inputs.shape[2] - self.filters.elements.shape[3] + 1
        out = np.zeros([self.filters.elements.shape[0], d1, d2])
        for i in range(0, self.filters.elements.shape[0]):
            for j in range(0, self.filters.elements.shape[1]):
                out[i, :, :] += sc.convolve2d(inputs[j, :, :], self.filters.elements[i, j, :, :], mode="valid") \
                                + self.bias.elements[i, j, :]
        return np.array([out])

    def backprop(self, lastlayer):
        newfilter = np.transpose(self.filters.elements, [1, 0, 2, 3])
        out = np.zeros(self.inputs["in"].shape)
        for i in range(0, newfilter.shape[0]):
            for j in range(0, newfilter.shape[1]):
                out[j, :, :] += sc.convolve2d(lastlayer[i, :, :], np.rot90(newfilter[i, j, :, :], 2), mode="full")

        # generating updates for deltas
        filter_update = np.zeros(self.filters.elements.shape)
        bias_update = np.zeros(self.bias.elements.shape)
        for i in range(0, self.filters.elements.shape[0]):
            for j in range(0, self.filters.elements.shape[1]):
                filter_update[i, j, :, :] = sc.convolve2d(self.inputs["in"][j, :, :], np.rot90(lastlayer[i, :, :],2),
                                                          mode="valid")
                bias_update[i, j, :] = np.array(np.sum(lastlayer[j, :, :]))
        self.filters.add_deltas(filter_update)
        self.bias.add_deltas(bias_update)
        return out

    def update(self, rate, batchsize):
        self.filters.update(rate, batchsize)
        self.bias.update(rate, batchsize)


class ReLuLayer:

    def __init__(self):
        self.inputs = {}

    def forward(self, inputs):
        self.inputs["in"] = Tensor(inputs)
        return np.maximum(inputs, 0)

    def backprop(self, lastlayer):
        return np.matmul(lastlayer, np.diag(np.where(self.inputs["in"].elements > 0, 1, 0).flatten()))

    def update(self, rate, batchsize):
        return


class Sigmoid:

    def __init__(self):
        self.inputs = {}

    def forward(self, inputs):
        self.inputs["in"] = sp.expit(inputs)
        return self.inputs["in"]

    def backprop(self, lastlayer):
        return self.inputs["in"] * (1 - self.inputs["in"]) * lastlayer

    def update(self, rate, batchsize):
        return


class Softmax:

    def __init__(self):
        self.inputs = {}

    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1))
        self.inputs["in"] = Tensor(exp / np.sum(exp, axis=1))
        return self.inputs["in"].elements

    def backprop(self, lastlayer):
        s = self.inputs["in"].elements.reshape(-1, 1)
        return np.matmul(lastlayer, np.diagflat(s) - np.dot(s, s.T))

    def update(self, rate, batchsize):
        return
