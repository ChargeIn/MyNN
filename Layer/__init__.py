# Florian Plesker 2026588
from Layer.Objects import *
import tensorflow as tf
import numpy as np


def mean_square(output, original):
    return np.linalg.norm(original - output)


def mean_square_derivation(output, original):
    return output - original


def cross_entropy(output, original):
    log = - np.log(output)
    return np.multiply(log, original)


def int_to_vector(i, shape):
    vector = np.zeros([1, shape])
    vector[0, i] = 1
    return vector


def minst(layers):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # starting parameter
    batchsize = 50
    learning_rate = 0.1
    epoche = 1
    output = 0
    loss = 0
    while True:
        for i in range(0, y_train.size, batchsize):
            loss = 0
            for j in range(0, batchsize):
                inputs = x_train[i + j]
                x2 = int_to_vector(y_train[i + j], 10)

                for l in layers:
                    inputs = l.forward(inputs)

                output = mean_square_derivation(inputs, x2)
                for l in layers[::-1]:
                    output = l.backprop(output)

            for l in layers:
                l.update(learning_rate, batchsize)

            print("\rEpoche: " + str(epoche) + " Training: " + str(i) + " / " + str(y_train.size), end="")

        print(" \rEpoche: " + str(epoche) + " Training: " + str(y_train.size) + " / " + str(y_train.size))
        for i in range(0, y_test.size, batchsize):
            inputs = x_test[i]
            x2 = int_to_vector(y_test[i], 10)
            for l in layers:
                inputs = l.forward(inputs)
            loss += mean_square(inputs, x2)
        print("Epoche: " + str(epoche) + " Loss:" + str(loss / batchsize))
        epoche += 1


if __name__ == '__main__':

    inputs = InputLayer()
    reshape = reshapeLyer()
    filter1 = np.array([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]])
    filter2 = np.array([[[1, 0, -1], [0, 0, 0], [-1, 0, 1]]])
    filters = np.array([filter1, filter2])
    bias = np.array([[[1]], [[1]]])
    convLayer = Conv2DLayer(filters, bias)
    fully = FullyConnected([1, 1352], 10)#784
    fully2 = FullyConnected([1, 100], 10)
    reLu1 = ReLuLayer()
    reLu = ReLuLayer()
    soft = Softmax()

    layers = [inputs, convLayer, reshape, fully, reLu, soft]
    # layers = [inputs, reshape, fully, reLu1, fully2, reLu, soft]
    #minst(layers)

    #fully connected test
    inputTest = InputLayer()
    reshapeTest = reshapeLyer()
    fullyTest = FullyConnected([1, 3], 2)
    fullyTest.bias.elements = np.array([0.0000, 0.0000, 0.0000])
    fullyTest.weights.elements = np.array([[-0.5057, 0.3987, -0.8943], [0.3356, 0.1673, 0.8321], [-0.3485, -0.4597, -0.1121]])
    fullyTest2 = FullyConnected([1, 3], 2)
    fullyTest2.weights.elements = np.array([[0.4047, 0.9563], [-0.8192, -0.1274], [0.3662, -0.7252]])
    fullyTest2.bias.elements = np.array([[0.0000, 0.0000]])
    sigmoidTest = Sigmoid()
    softmaxTest = Softmax()

    # forward
    input = [0.4183, 0.5209, 0.0291]
    label = [0.7095, 0.0942]
    input = inputTest.forward(input)
    input = reshapeTest.forward(input)
    input = fullyTest.forward(input)
    print("forward fully1: ", input)
    input = sigmoidTest.forward(input)
    print("forward sigmoid: ", input)
    input = fullyTest2.forward(input)
    print("forward fully 2: ", input)
    input = softmaxTest.forward(input)
    print("forward softmax: ", input)

    # backward
    back = np.array([[-1.4901, -0.1798]])
    back = softmaxTest.backprop(back)
    print("backward softmax: ", back)
    back = fullyTest2.backprop(back)
    print("backward fully2: ", back)
    back = sigmoidTest.backprop(back)
    print("backward sigmoid: ", back)
    back = fullyTest.backprop(back)
    print("backward fully1: ", back)

