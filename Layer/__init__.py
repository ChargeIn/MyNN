# Florian Plesker 2026588
from Layer.Objects import *
import tensorflow as tf


def meanSquare(output, original):
    return np.linalg.norm(output - original)/original.lenght()


def crossEntropy(output, original):
    log = - np.log(output)
    return np.multiply(log, original)


if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = x_train / 255.0, y_train / 255.0

    inputLayer = InputLayer
    fully = FullyConnected(100, 1)
    reLu = ReLuLayer
    soft = Softmax

    layers = [fully, reLu, soft]
    while True:
        for i in range(y_train.size):
            x1 = x_train[i]
            x2 = y_train[i]
            tensor = inputLayer.forward(x1)
            for l in layers:
                tensor = l.forward(tensor)
            print(tensor.elements.shape)
