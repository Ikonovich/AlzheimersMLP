import random
from numba import jit

import numpy as np

from Layers.Layer import Layer

class ConvolutionalLayer(Layer):
    # A convolutional layer

    def __init__(
            self,
            input_size: tuple[int, int],
            num_filters: int,
            filter_size: tuple[int, int],
            padding_size: tuple[int, int] = (0, 0),
            step_size: tuple[int, int] = (1, 1),
            # If buffer sides is true, 0s will be added to each side of the input data
            # to allow edges and corners to be fully captured.
            buffer_sides: bool = False,
            dropout_modifier: float = 0.0,
            activation_string: str = "None",
            previous_layer=None):



        # Calculate the size of the output
        output_size_x = int((input_size[0] + 2 * padding_size[0] - filter_size[0]) / step_size[0]) + 1
        output_size_y = int((input_size[1] + 2 * padding_size[1] - filter_size[1]) / step_size[1]) + 1
        self.output_size = (num_filters, output_size_x, output_size_y)

        # Initialize the base class
        super().__init__(
            input_size=input_size,
            output_size=self.output_size,
            dropout_modifier=dropout_modifier,
            activation_string=activation_string,
            previous_layer=previous_layer)

        self.filter_size = filter_size
        self.step_size = step_size
        self.padding = padding_size

        # Stores the actual convolutional filters
        # as a list of filter_size_x by filter_size y arrays
        self.filters = []
        # Randomly generate initial filters.
        for i in range(num_filters):
            self.filters.append(np.random.default_rng().random(filter_size))

    def forward_prop(self, data_in: np.array):

        # Check to see if this is a 2D array, if so, add a third dimension.
        if data_in.ndim < 3:
            data_in = data_in[np.newaxis]

        # Store the input
        self.last_input = data_in
        # Iterate over the input data with each filter
        output = []
        for kernel in self.filters:
            output.append(convolve(data_in, kernel, self.activation))

        self.last_output = np.asarray(output)
        print(self.last_output)
        return self.last_output

    def hidden_back_prop(self, learn_rate):
        lower_layer = self.next_layer
        prime = self.derivative(self.last_output)
        self.last_delta = np.matmul(lower_layer.last_delta, lower_layer.weights.T) * prime
        self.back_prop(learn_rate=learn_rate)

    def back_prop(self, learn_rate):
        adjustedInput = self.last_input[np.newaxis]
        adjustedDelta = self.last_delta[np.newaxis]
        weight_updates = np.matmul(adjustedInput.T, adjustedDelta * learn_rate)
        bias_updates = self.last_delta * learn_rate * self.bias_lrn_modifier
        self.weights -= weight_updates
        self.bias -= bias_updates

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


# @jit(nopython=True)
def convolve(data, kernel, activation=None, stride_x=1, stride_y=1):
    # Store the size of the data in each dimension
    x_size_data = data.shape[1]
    y_size_data = data.shape[2]
    z_size_data = data.shape[0]
    # Store how big each slice of the input data should be in each dimension,
    # which is also the kernel size.
    # TODO: Fix this for 3 dimensions
    x_size_kernel = kernel.shape[0]
    y_size_kernel = kernel.shape[1]
    z_size_kernel = z_size_data

    # Calculate the size of the output
    output_size_x = int((x_size_data - x_size_kernel) / stride_x) + 1
    output_size_y = int((y_size_data - y_size_kernel) / stride_y) + 1

    # Initialize the output
    output = np.zeros(shape=(output_size_x, output_size_y))
    # Flip the filter matrix
    kernel = np.flipud(np.fliplr(kernel))
    # Get a flattened version of the filter
    flat_kernel = kernel.ravel()

    # We will divide by this to get normalized values
    kernel_size = x_size_kernel * y_size_kernel * z_size_kernel

    # Run the convolution loop
    # We iterate over the data in the x-axis, stopping when the filter runs
    # up against the edge of the data, then going on to the next step of the y-axis.

    # Store the index of the output to place the next calculation result.
    y_output_index = 0
    for y_index in range(0, (y_size_data - y_size_kernel + 1), stride_y):
        x_output_index = 0
        for x_index in range(0, (x_size_data - x_size_kernel + 1), stride_x):
            # Get a view of the numpy array from the relevant indices to the size of the
            # array. This does NOT copy any data.
            # At the same time, flatten it using ravel, which also returns a view.
            elements = data[0:z_size_kernel, x_index:x_index + x_size_kernel, y_index: y_index + y_size_kernel].ravel()

            # The dot product of two vectors is an elementwise multiplication
            # and summation, so let's do that.
            # We divide the result by the total number of values in the filter
            # to get a normalized value.
            result = elements.dot(flat_kernel) / kernel_size

            # If an activation function was supplied, apply it now.
            if activation is not None:
                result = activation(result)
            # Store the result of the calculation
            output[x_output_index][y_output_index] = result
            # Increment the x-index of the output by 1.
            x_output_index += 1

        # Increment the y-index of the output by 1.
        y_output_index += 1

    return output


if __name__ == "__main__":
    # buffer(np.random.randn(5, 5), 3, 3)

    speedtest()
