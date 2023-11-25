import numpy as np

from Layers import AveragePoolingLayer as pool
from Layers import ConvolutionLayer as conv

# Max pooling base case.
def MaxBaseCaseTest():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    data = np.asarray(data)

    data = conv.buffer(data, 1, 1)

    result = pool.max_pool(data)
    print(result)
