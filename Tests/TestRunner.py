import numpy as np

import ConvolutionTests as conv
import PoolingTest as pool
from Experimental import ConvolutionPrototype

if __name__ == "__main__":


    #pool.MaxBaseCaseTest()

    # data = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
    # kernel = np.asarray([[1, 0], [0, 1]])
    #
    # ConvolutionPrototype.TwoDconvolve(data=data, kernel=kernel)

    matr = np.zeros((2, 2, 2))
    matr[0][0][1] = 1
    matr[0][1][0] = 1
    print(matr)

    matr[1][0][0] = 1
    matr[1][1][1] = 1

    print(matr)

    nums = np.asarray([[1, 2], [3, 4]])

    print(np.matmul(nums, matr))
