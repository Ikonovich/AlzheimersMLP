import numba
from numba import jit, float32, float64
import numpy as np
import time

from Math import Functions

datum = np.random.randn(1000, 1000)
filt = np.random.randn(10, 10)

def normal(data, kernel, stride_x=1, stride_y=1):

    # Store the size of the data in each dimension
    x_size_data = data.shape[0]
    y_size_data = data.shape[1]
    # Store how big each slice of the input data should be in each dimension,
    # which is also the kernel size.
    x_size_kernel = kernel.shape[0]
    y_size_kernel = kernel.shape[1]


    # Calculate the size of the output
    output_size_x = int((x_size_data - x_size_kernel) / stride_x) + 1
    output_size_y = int((y_size_data - y_size_kernel) / stride_y) + 1

    # Initialize the output
    output = np.zeros(shape=(output_size_x, output_size_y))
    # Flip the filter matrix
    kernel = np.flipud(np.fliplr(kernel))
    # Get a flattened version of the filter
    flat_kernel = kernel.ravel()


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
            elements = data[x_index:x_index + x_size_kernel, y_index: y_index + y_size_kernel].ravel()

            # The dot product of two vectors is an elementwise multiplication
            # and summation, so let's do that.
            result = elements.dot(flat_kernel)
            # Store the result of the calculation
            output[x_output_index][y_output_index] = result
            # Increment the x-index of the output by 1.
            x_output_index += 1

        # Increment the y-index of the output by 1.
        y_output_index += 1

    return output

@jit(nopython=True)
def numba(data, kernel, stride_x=1, stride_y=1):

    # Store the size of the data in each dimension
    x_size_data = data.shape[0]
    y_size_data = data.shape[1]
    # Store how big each slice of the input data should be in each dimension,
    # which is also the kernel size.
    x_size_kernel = kernel.shape[0]
    y_size_kernel = kernel.shape[1]


    # Calculate the size of the output
    output_size_x = int((x_size_data - x_size_kernel) / stride_x) + 1
    output_size_y = int((y_size_data - y_size_kernel) / stride_y) + 1

    # Initialize the output
    output = np.zeros(shape=(output_size_x, output_size_y))
    # Flip the filter matrix
    kernel = np.flipud(np.fliplr(kernel))
    # Get a flattened version of the filter
    flat_kernel = kernel.ravel()


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
            elements = data[x_index:x_index + x_size_kernel, y_index: y_index + y_size_kernel].ravel()

            # The dot product of two vectors is an elementwise multiplication
            # and summation, so let's do that.
            result = elements.dot(flat_kernel)
            # Store the result of the calculation
            output[x_output_index][y_output_index] = result
            # Increment the x-index of the output by 1.
            x_output_index += 1

        # Increment the y-index of the output by 1.
        y_output_index += 1

    return output

# Testing normal execution, without Numba
start = time.time()
normal(datum, filt)
end = time.time()
print("Elapsed normal = %s" % (end - start))

# First iteration of Numba. Numba will have to compile the code during this iteration.
start = time.time()
numba(datum, filt)
end = time.time()
print("Elapsed numba = %s" % (end - start))

# 2nd iteration with Numba. Compilation should be complete, this should be a lot faster.
start = time.time()
numba(datum, filt)
end = time.time()
print("Elapsed numba (after Numba compilation) = %s" % (end - start))