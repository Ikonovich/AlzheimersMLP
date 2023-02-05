import random
from numba import jit
import numpy as np

from Layers.Layer import Layer


class ConvolutionalLayer(Layer):
    # A convolutional layer

    def __init__(
            self,
            # Determines the shape of the input in the y and then x direction.
            input_shape: tuple[int, int],
            # Determines the total number of filters/kernels to be generated
            # with the provided parameters.
            num_filters: int,
            # Determines the shape of the filters in the y and then x direction.
            filter_shape: tuple[int, int],
            # Determines how much to pad the top/bottom and sides, respectively
            padding_size: tuple[int, int] = (0, 0),
            # Determines the size of each step in the y and then x direction.
            step_size: tuple[int, int] = (1, 1),
            activation_string: str = "None",
            previous_layer=None):

        # Calculate the size of the output
        output_size_x = int((input_shape[0] + 2 * padding_size[0] - filter_shape[0]) / step_size[0]) + 1
        output_size_y = int((input_shape[1] + 2 * padding_size[1] - filter_shape[1]) / step_size[1]) + 1
        self.output_shape = (num_filters, output_size_x, output_size_y)

        # Initialize the base class
        super().__init__(
            input_shape=input_shape,
            output_shape=self.output_shape,
            activation_string=activation_string,
            previous_layer=previous_layer)

        self.filter_size = filter_shape
        self.step_size = step_size
        self.padding = padding_size

        # Stores the actual convolutional filters
        # as a list of filter_size_x by filter_size y arrays
        self.filters = []
        # Randomly generate initial filters.
        for i in range(num_filters):
            self.filters.append(np.random.default_rng().random(filter_shape))

    def forward_prop(self, data_in: np.array):

        # Check to see if this is a 2D array, if so, add a third dimension.
        if data_in.ndim < 3:
            data_in = data_in[np.newaxis]

        # Store the input
        self.input = data_in
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
        delta = np.matmul(lower_layer.loss, lower_layer.weights.T)

        # If the delta isn't the correct shape (I.E., if the lower
        # layer flattened its input), reshape it.
        if delta.shape != self.output_shape:
            delta = delta.reshape(self.output_shape)

        self.back_prop(learn_rate=learn_rate)

    def back_prop(self, learn_rate):
        adjustedInput = self.input[np.newaxis]
        adjustedDelta = self.last_delta[np.newaxis]
        weight_updates = np.matmul(adjustedInput.T, adjustedDelta * learn_rate)
        self.weights -= weight_updates

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


# Convolves across the sample and generates filter updates from the loss.
def inverse_convolve(data, kernel, loss, derivative=None, stride_x=1, stride_y=1):
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
    loss_index = 0
    # Stores the updates to be returned
    update = 0
    for y_index in range(0, (y_size_data - y_size_kernel + 1), stride_y):
        for x_index in range(0, (x_size_data - x_size_kernel + 1), stride_x):
            # Get a view of the numpy array from the relevant indices to the size of the
            # array. This does NOT copy any data.
            # At the same time, flatten it using ravel, which also returns a view.
            elements = data[0:z_size_kernel, x_index:x_index + x_size_kernel, y_index: y_index + y_size_kernel].ravel()

            # Multiply each element times the kernel.
            # We divide the result by the total number of values in the filter
            # to get a normalized value.
            # Finally, we multiply by the appropriate loss data from the next
            # layer.
            update += elements * flat_kernel / kernel_size * loss[loss_index]

            # If an activation function was supplied, apply it now.
            if derivative is not None:
                result = derivative(result)


            # Increment the loss index by 1.
            loss_index += 1

    return update