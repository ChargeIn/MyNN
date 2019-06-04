# Florian Plesker 2026588
from Layer.Objects import *


def loss(output, original):
    return np.linalg.norm(output - original)


if __name__ == '__main__':
    t1 = Tensor()
