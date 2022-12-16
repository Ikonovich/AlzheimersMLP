import numpy as np

from Layers import ConvolutionLayer as conv


# Base case. A filter composed of a 0D array containing the value 1
# should return a matrix identical to the input matrix.
def BaseCaseTest():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    data = np.asarray(data)

    kernel = np.asarray([[1]])

    result = conv.convolve(data, kernel)
    print(result)

