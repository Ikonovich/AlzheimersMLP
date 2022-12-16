import math

import numpy as np
from Debugger import Debugger

def TwoDconvolve(data: np.ndarray, kernel: np.ndarray, step_size: tuple[int, int] = (1, 1)):
    # Convolution with 2D motion.
    # Depth of the kernel/filter must match depth of the data.
    #
    # If the shape of the inputs is 3D, we take depth to be
    # the 0-axis of the nd-arrays (referred to as z-axis or z),
    # setting x-axis = 1 and y-axis = 2.
    #
    # If the shape of the inputs is 2D, we set x-axis = 0 and
    # y-axis = 1
    #
    # Step_size dimensions are always treated as (x, y),
    # equivalent to x = 0 and y = 1.

    # Set when a warning message is given for a kernel not providing total coverage
    # of the input to a convolution function.
    incomplete_coverage_warning = False

    # Check that the number of dimensions of the kernel matches that of the data.
    if len(data.shape) != len(kernel.shape):
        raise ValueError("The number of dimensions of the inputs to a convolution did not match: "
                         f"Data: {len(data.shape)}, Kernel: {len(kernel.shape)}")

    # Then, see if we're convolving in 2D or 3D and designate our
    # x and y axes appropriately.
    if len(data.shape) == 3:
        # In this case, need to make sure the depth of the input matches the depth of the data.
        if data.shape[0] != kernel.shape[0]:
            raise ValueError("The depth of the inputs to a convolution did not match: "
                             f"Data: {data.shape[0]}, Kernel: {kernel.shape[0]}")

        x_axis = 1
        y_axis = 2
    elif len(data.shape) == 2:
        x_axis = 0
        y_axis = 1
    else:
        raise ValueError(f"Convolve input has invalid number of dimensions: {len(data.shape)}.")

    # Then, we calculate the number of steps to take in each axis
    x_step_count = ((data.shape[x_axis] - kernel.shape[x_axis]) / step_size[0]) + 1
    y_step_count = ((data.shape[y_axis] - kernel.shape[y_axis]) / step_size[1]) + 1

    if incomplete_coverage_warning == False and (x_step_count.is_integer() == False or y_step_count.is_integer()) == False:
        Debugger.warn("Supplied kernel will not convolve over entire input. "
                      f"Spaces remaining:\nX-axis: "
                      f"{(len(data[x_axis]) - len(kernel[x_axis])) % step_size[0]}"
                      f"\nY-axis: {(len(data(y_axis)) - len(kernel[y_axis])) % step_size[1]}")
        incomplete_coverage_warning = True

    # Iterate over each axis in the step sizes provided.
    # This will produce results in the shape
    # [ 1 2 3
    #   4 5 n..]

    # Initialize results matrix
    output = np.zeros((math.floor(x_step_count), math.floor(y_step_count)))
    # Get a flattened view of the kernel
    flat_kern = kernel.ravel()
    for y in range(0, data.shape[y_axis], step_size[1]):
        for x in range(0, data.shape[x_axis], step_size[0]):
            # Get the section of the data to convolve over and flatten it with a view.
            section = data[x: x + kernel.shape[x_axis], y: y + kernel.shape[y_axis]].ravel()

            result = section.dot(flat_kern)
            output[x][y] = result

    print(output)

