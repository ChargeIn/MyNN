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


def vec_to_hit(i, o):
    if np.argmax(o, 1)[0] == i:
        return True
    return False


def minst(layers):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # starting parameter
    batchsize = 500
    learning_rate = 0.01
    epoche = 1
    output = 0
    loss = 0
    l = ""

    for ld in layers:
        l = l + str(ld) + " "
    print("Layers: " + l)

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
        acc = 0
        for i in range(0, y_test.size, batchsize):
            inputs = x_test[i]
            x2 = int_to_vector(y_test[i], 10)
            for l in layers:
                inputs = l.forward(inputs)
            loss += mean_square(inputs, x2)
            if vec_to_hit(y_test[i], inputs):
                acc += 1
        print("Epoche: " + str(epoche) + " Loss:" + str(loss / batchsize) + " Acc: " + str(acc/y_test.size))
        epoche += 1


if __name__ == '__main__':

    inputs = InputLayer()
    reshape = ReshapeLayer()
    filter1 = np.array([[[1, -1, -1], [1, 1, 0], [-1, -1, 1]]])
    filter2 = np.array([[[1, 1, 0], [1, 0, 1], [-1, 1, 0]]])
    filter3 = np.array([[[-1, -1, 1], [0, 0, 1], [1, -1, -1]]])
    filters = np.array([filter1, filter2, filter3])
    bias = np.array([[[0]], [[0]], [[0]]])
    convLayer = Conv2DLayer(filters, bias)
    fully = FullyConnected([1, 784], 10)#784#1352#2028
    fully2 = FullyConnected([1, 100], 10)
    reLu = ReLuLayer()
    sigmoid = Sigmoid()
    soft = Softmax()

    layers = [inputs, reshape, fully, reLu, soft]
    #layers = [inputs, convLayer, reshape, fully, sigmoid, fully2, soft]
    minst(layers)

    # # conv2d_test
    # inputTest = InputLayer()
    # reshapeTest = ReshapeLyer()
    # save = np.array([0.1, -0.2, 0.5, 0.6, 1.2, 1.4, 1.6, 2.2, 0.01, 0.2, -0.3, 4.0, 0.9, 0.3, 0.5, 0.65, 1.1, 0.7, 2.2, 4.4, 3.2, 1.7, 6.3, 8.2])
    # save2 = np.array([0.1, -0.2, 0.3, 0.4, 0.7, 0.6, 0.9, -1.1, 0.37, -0.9, 0.32, 0.17, 0.9, 0.3, 0.2, -0.7])
    # save3 = np.zeros([2, 4, 3])
    # save4 = np.zeros([2, 2, 2, 2])
    # pos = 0
    # for i in range(0, 2):
    #     for j in range(0, 3):
    #         for k in range(0, 4):
    #             save3[i, k, j] = save[pos]
    #             pos += 1
    # for i in range(0, 2):
    #     for j in range(0, 3):
    #         for k in range(0, 4):
    #             s1 = 4*j
    #             s2 = 4*3*i
    #             pos = k + s1 + s2
    #             save3[i, k, j] = save[pos]
    # pos = 0
    # for i in range(0, 2):
    #     for j in range(0, 2):
    #         for k in range(0, 2):
    #             for m in range(0, 2):
    #                 save4[i, j, m, k] = save2[pos]
    #                 pos += 1
    # kernel_forward = save4
    # convLayerTest = Conv2DLayer(kernel_forward, np.zeros([2, 2, 1]))
    # inputs = convLayerTest.forward(save3)
    # print(inputs)
    # modified = np.array([0.4, 0.3, -0.2, 0.1, 0.17, 0.32, -0.9, 0.37, -1.1, 0.9, 0.6, 0.7, -0.7, 0.2, 0.3, 0.9])
    # pos = 0
    # save4 = np.zeros([2, 2, 2, 2])
    # for i in range(0, 2):
    #     for j in range(0, 2):
    #         for k in range(0, 2):
    #             for m in range(0, 2):
    #                 save4[i, j, m, k] = modified[pos]
    #                 pos += 1
    # convLayerTest.filters.elements = save4
    # save3 = np.zeros([2, 3, 2])
    # save = np.array([0.1, 0.33, -0.6, -0.25, 1.3, 0.01, -0.5, 0.2, 0.1, -0.8, 0.81, 1.1])
    # pos = 0
    # for i in range(0, 2):
    #     for j in range(0, 2):
    #         for k in range(0, 3):
    #             save3[i, k, j] = save[pos]
    #             pos += 1
    # input = convLayerTest.backprop(save3)
    # print(input)
    # print(convLayerTest.filters.deltas["deltas"])

    # fully_test
    # inputTest = InputLayer()
    # reshapeTest = ReshapeLyer()
    # fullyTest = FullyConnected([1, 3], 2)
    # fullyTest.bias.elements = np.array([0.0000, 0.0000, 0.0000])
    # fullyTest.weights.elements = np.array([[-0.5057, 0.3987, -0.8943], [0.3356, 0.1673, 0.8321], [-0.3485, -0.4597, -0.1121]])
    # fullyTest2 = FullyConnected([1, 3], 2)
    # fullyTest2.weights.elements = np.array([[0.4047, 0.9563], [-0.8192, -0.1274], [0.3662, -0.7252]])
    # fullyTest2.bias.elements = np.array([[0.0000, 0.0000]])
    # sigmoidTest = Sigmoid()
    # softmaxTest = Softmax()
    #
    # # forward
    # input = [0.4183, 0.5209, 0.0291]
    # label = [0.7095, 0.0942]
    # input = inputTest.forward(input)
    # input = reshapeTest.forward(input)
    # input = fullyTest.forward(input)
    # print("forward fully1: ", input)
    # input = sigmoidTest.forward(input)
    # print("forward sigmoid: ", input)
    # input = fullyTest2.forward(input)
    # print("forward fully 2: ", input)
    # input = softmaxTest.forward(input)
    # print("forward softmax: ", input)
    #
    # # backward
    # back = np.array([[-1.4901, -0.1798]])
    # back = softmaxTest.backprop(back)
    # print("backward softmax: ", back)
    # back = fullyTest2.backprop(back)
    # print("backward fully2: ", back)
    # back = sigmoidTest.backprop(back)
    # print("backward sigmoid: ", back)
    # back = fullyTest.backprop(back)
    # print("backward fully1: ", back)

    # # Test for convolution
    # inte = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # fil = np.array([[1, 1], [1, 1]])
    # print("\n")
    # print(inte)
    # print(sc.convolve2d(inte, fil, mode="valid"))

