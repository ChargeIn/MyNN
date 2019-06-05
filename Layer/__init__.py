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


