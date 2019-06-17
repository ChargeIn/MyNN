# Florian Plesker 2026588
from Layer.Objects import *
import tensorflow as tf
import numpy as np


def meanSquare(output, original):
    return np.linalg.norm(output - original)


def meanSquareDerivation(output, original):
    return output - original


def crossEntropy(output, original):
    log = - np.log(output)
    return np.multiply(log, original)


def intToVector(i, shape):
    vector = np.zeros([1, shape])
    vector[0, i] = 1
    return vector


if __name__ == '__main__':

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    inputLayer = InputLayer()
    fully = FullyConnected([1, 784], 10, 1)
    filter1 = np.array([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]])
    filter2 = np.array([[[1, 0, -1], [0, 0, 0], [-1, 0, 1]]])
    filters = np.array([filter1, filter2])
    convLayer = Conv2DLayer(filters, 2)
    reLu = ReLuLayer([10, 1])
    soft = Softmax([10, 1])

    # starting parameter
    batchsize = 50
    learning_rate = 0.1
    layers = [fully, reLu, soft]
    epoche = 1
    output = 0
    loss = 0
    old_loss = 999
    while True:
        for i in range(0, y_train.size, batchsize):
            loss = 0
            for j in range(0, batchsize):
                x1 = x_train[i + j]
                x2 = intToVector(y_train[i + j], 10)
                x1 = convLayer.forward(x1)
                inputs = inputLayer.forward(x1)
                for l in layers:
                    inputs = l.forward(inputs)

                output = meanSquareDerivation(inputs, x2)
                for l in layers[::-1]:
                    output = l.backprop(output)

            for l in layers:
                l.update(learning_rate, batchsize)

            print("\rEpoche: " + str(epoche) + " Training: " + str(i) + " / " + str(y_train.size), end="")

        print(" \rEpoche: " + str(epoche) + " Training: " + str(y_train.size) + " / " + str(y_train.size))

        for i in range(0, y_test.size, batchsize):
            x1 = x_test[i]
            x2 = intToVector(y_test[i], 10)
            inputs = inputLayer.forward(x1)
            for l in layers:
                inputs = l.forward(inputs)
            loss += meanSquare(inputs, x2)

        if np.abs(loss - old_loss) < 0.001:
            break
        old_loss = loss
        print("Epoche: " + str(epoche) + " Loss:" + str(loss / batchsize))
        epoche += 1
