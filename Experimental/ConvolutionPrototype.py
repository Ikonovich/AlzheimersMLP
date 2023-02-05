import math

import numpy as np
from Debugger import Debugger

from Math import Functions

class ConvolutionPrototype:

    def __init__(self):
        self.filters = [np.random.uniform(-1, 1, (5, 5)) for i in range(1)]

        # Stores the most recent input
        self.last_input = None
        print(self.filters)
        self.decision_weights = np.random.uniform(-1, 1, (576, 10))  # Size of the output on mnist dataset

        # Stores the output from the convolution layer
        self.conv_output = None

        # Stores the output from the perceptron layer
        self.decision_output = None


    def iterate(self, data: np.ndarray):
        self.last_input = data
        results = list()

        for filter in self.filters:
            results.append(self.twoDconvolve(data, filter))

        self.conv_output = results
        output = self.perceptron(results)

    def perceptron(self, input):
        next = np.asarray(input).flatten()

        input_in = np.asarray(input).flatten()

        self.decision_output = np.matmul(input_in, self.decision_weights) / 576
        return Functions.sigmoid(self.decision_output)

    def backprop(self, expected):

        learn_rate = 0.5

        # Update the perceptron layer
        error = -(expected - self.decision_output)
        prime = Functions.sigmoid_prime(self.decision_output)

        delta = error * prime

        adjustedInput = np.asarray(self.conv_output).flatten()[np.newaxis]
        adjustedDelta = delta[np.newaxis]
        weight_updates = np.matmul(adjustedInput.T, adjustedDelta * learn_rate)
        self.decision_weights -= weight_updates

        # Update the convolution filters
        prime = list()
        for item in self.conv_output:
            prime.append(Functions.relu_prime(item))

        lower_weights = np.split(self.decision_weights, len(self.conv_output))
        lower_delta = np.split(delta, len(self.conv_output))

        last_delta = list()
        for i in range(len(self.conv_output)):
            conv_delta = np.matmul(lower_delta[i], lower_weights[i].T) * prime[i].ravel()
            last_delta.append(conv_delta)

        adjustedInput = self.last_input.tolist()

        for i in range(len(self.filters)):
            data = adjustedInput[i]
            delta = last_delta[i]
            data = np.asarray(data)[np.newaxis]
            delta = np.asarray(delta)[np.newaxis]
            filter_update = np.matmul(data.T, delta * learn_rate)
            self.filters[i] -= filter_update


    def twoDconvolve(self, data: np.ndarray, kernel: np.ndarray, step_size: tuple[int, int] = (1, 1)):
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
        for y in range(0, data.shape[y_axis] - kernel.shape[y_axis], step_size[1]):
            for x in range(0, data.shape[x_axis] - kernel.shape[x_axis], step_size[0]):
                # Get the section of the data to convolve over and flatten it with a view.
                section = data[x: x + kernel.shape[x_axis], y: y + kernel.shape[y_axis]].ravel()

                result = section.dot(flat_kern)

                output[x][y] = Functions.relu(result)

        # print(output)
        # print("Flattened: ")
        # print(output.flatten())
        return output

