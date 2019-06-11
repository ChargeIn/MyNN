# Florian Plesker 2026588
from Layer.Objects import *
import tensorflow as tf


def meanSquare(output, original):
    return np.linalg.norm(output - original)/original.size


def crossEntropy(output, original):
    log = - np.log(output)
    return np.multiply(log, original)


if __name__ == '__main__':

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = x_train / 255.0, y_train / 255.0

    inputLayer = InputLayer()
    fully = FullyConnected([784, 1], 10, 1)
    reLu = ReLuLayer([10, 1])
    soft = Softmax([10, 1])

    layers = [fully, reLu, soft]
    while True:
        for i in range(y_train.size):
            x1 = x_train[i]
            x2 = y_train[i]
            print(x1.shape)
            inputs = inputLayer.forward(x1)
            print(inputs.shape)
            for l in layers:
                inputs = l.forward(inputs)
            output = meanSquare(inputs, x2)
            print(output)
            for l in layers[::-1]:
                output = l.backward(output)
                #l.update()
            print(output)
