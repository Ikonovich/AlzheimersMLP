import random
from datetime import time

from Debugger import Debugger
import numpy as np
from NeuralNetwork import NeuralNetwork



# A convolutional layer
class ConvolutionalLayer:
    def __init__(self,
                 network: NeuralNetwork,
                 activation: str,
                 input_size_x: int,
                 input_size_y: int,
                 num_filters: int,
                 filter_size_x: int,
                 filter_size_y: int,
                 padding_x: int = 0,
                 padding_y: int = 0,
                 step_size_x: int = 1,
                 step_size_y: int = 1,
                 dropout_modifier: float = 0.0,
                 # If buffer sides is true, 0s will be added to each side of the input data
                 # to allow edges and corners to be fully captured.
                 buffer_sides: bool = False):

        # Set the network and activation function
        # Also initializes a pretty printer as pprint
        super().__init__(network, activation, dropout_modifier)


        self.input_size = (input_size_x, input_size_y)
        self.filter_size = (filter_size_x, filter_size_y)
        self.step_size = (step_size_x, step_size_y)
        self.padding = (padding_x, padding_y)

        # Calculate the size of the output
        output_size_x = ((input_size_x + 2 * padding_x - filter_size_x) / step_size_x) + 1
        output_size_y = ((input_size_y + 2 * padding_y - filter_size_y) / step_size_y) + 1
        self.output_size = (output_size_x, output_size_y)


        # Stores the actual convolutional filters
        # as a list of filter_size_x by filter_size y arrays
        self.filters = []
        # Randomly generate initial filters.
        for i in range(num_filters):
            self.filters.append(np.random.randn(filter_size_x, filter_size_y))


    def forward_prop(self, data_in: np.array):

        # Iterate over the input data with each filter
        output = np.asarray([])
        for i in range(len(self.filters)):
            pass


    def convolve(self, data, kernel):
        # Flip the filter matrix
        kernel = np.flipud(np.fliplr(kernel))






    def back_prop(self):
        pass

    # Buffers the sides of 2 dimensional arrays with zeroes
def buffer(data_in: np.array, buffer_x: int, buffer_y: int):

    # Convert the input to a list to avoid numpy trying to keep the memory continuous
    # during the operation.
    data_in = data_in.tolist()

    # Insert the buffering values in the x dimension
    for row in data_in:
        for i in range(buffer_x):
            row.append(0)
            row.insert(0, 0)

    # Insert the buffering values in the y dimension
    size_y = len(data_in) + buffer_y * 2
    for i in range(buffer_y):
        data_in.append([0 for i in range(size_y)])
        data_in.insert(0, [0 for i in range(size_y)])

    # Convert back to a numpy array
    data_out = np.array(data_in, ndmin=2)
    return data_out

def speedtest():
    # Generate random arrays
    list_array = []
    mult_array = []
    for i in range(16):
        list_array.append([random.randint() for i in range(16)])
        mult_array.append([random.randint() for i in range(16)])

    # nparray = np.asarray(list_array)
    # cparray = cp.asarray(list_array)
    #
    # np_multarray = np.asarray(mult_array)
    # cp_multarray = cp.asarray(mult_array)
    #
    # print("Starting Numpy calculations.")
    # start = time.time()
    #
    # for i in range(10000):
    #     throwaway = np.matmul(nparray, np_multarray)
    #
    # print(f"Time taken by numpy: {time.time() - start}")
    #
    # print("Starting CuPy calculations.")
    # start = time.time()
    #
    # for i in range(10000):
    #     throwaway = cp.matmul(cparray, cp_multarray)
    #
    # print(f"Time taken by Cupy: {time.time() - start}")


if __name__ == "__main__":
    # buffer(np.random.randn(5, 5), 3, 3)

    speedtest()
