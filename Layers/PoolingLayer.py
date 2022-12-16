from enum import Enum

import numpy as np
from numba import jit

from Layers.Layer import Layer

class PoolingType(Enum):
    Max = "Max",
    Average = "Average"

class PoolingLayer(Layer):

    def __init__(self,
                 previous_layer,
                 next_layer,
                 input_size: tuple[int, int],
                 filter_size: tuple[int, int],
                 step_size: tuple[int, int] = (1, 1),
                 pooling_type: PoolingType = PoolingType.Max):

        self.filter_size = filter_size

        self.step_size = step_size

        # Calculate the size of the output
        output_size_x = int((input_size[0] - filter_size[1]) / step_size[0]) + 1
        output_size_y = int((input_size[1] - filter_size[1]) / step_size[1]) + 1
        self.output_size = (output_size_x, output_size_y)

        # Set the base parameters
        super('Layer').__init__(previous_layer=previous_layer,
                                next_layer=next_layer,
                                input_size=input_size,
                                output_size=self.output_size,
                                dropout_modifier=0,
                                activation_string="None")

    # The functionality of this method is stored outside the class to allow the easy use
    # of numba to improve performance.
    def forward_prop(self, data):
        self.last_input = data

        self.last_output = max_pool(data, self.filter_size, self.step_size)
        return self.last_output

    def back_prop(self):
        lower_layer = self.next_layer
        # Get the elements that delta should be applied to - The maximums.
        self.last_delta = np.matmul(lower_layer.last_delta, lower_layer.weights.T) * prime

# Stored outside the class to make Numba easy to use
@jit(nopython=True)
def max_pool(
        data,
        filter_size: tuple[int, int] = (2, 2),
        step_size: tuple[int, int] = (2, 2)):

    # Store the size of the data in each dimension
    x_size_data = data.shape[0]
    y_size_data = data.shape[1]

    # Calculate the size of the output
    output_size_x = int((data.shape[0] - filter_size[0]) / step_size[0]) + 1
    output_size_y = int((data.shape[1] - filter_size[1]) / step_size[1]) + 1

    # Initialize the output
    output = np.zeros(shape=(output_size_x, output_size_y))

    # Initialize a binary map of maximum element.
    # Indexes that were never a maximum will be 0, otherwise they will be one.
    max_indices = np.zeros(shape=data.shape)


    # Ease of reading variables
    x_size_kernel = filter_size[0]
    y_size_kernel = filter_size[1]
    stride_x = step_size[0]
    stride_y = step_size[1]


    # Store the index of the output to place the next calculation result.
    y_output_index = 0

    # Run the convolution loop
    # We iterate over the data in the x-axis, stopping when the filter runs
    # up against the edge of the data, then going on to the next step of the y-axis.
    for y_index in range(0, (y_size_data - y_size_kernel + 1), stride_y):
        x_output_index = 0
        for x_index in range(0, (x_size_data - x_size_kernel + 1), stride_x):
            # Get a view of the numpy array from the relevant indices to the size of the
            # array. This does NOT copy any data.
            # At the same time, flatten it using ravel, which also returns a view.
            elements = data[x_index:x_index + x_size_kernel, y_index: y_index + y_size_kernel].ravel()

            # Get the max element from the area being convolved
            result_index = np.argmax(elements, axis=None)
            max_indices[result_index] += 1
            result = elements[result_index]
            # Store the result
            output[x_output_index][y_output_index] = result
            # Increment the x-index of the output by 1.
            x_output_index += 1

        # Increment the y-index of the output by 1.
        y_output_index += 1

    return output, max_indices
