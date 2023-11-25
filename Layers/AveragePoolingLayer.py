import numpy as np

from Layers.Layer import Layer


class AveragePoolingLayer(Layer):

    def __init__(self,
                 previous_layer=None,
                 next_layer=None,
                 input_shape: tuple[int, int] = None,
                 filter_shape: tuple[int, int] = (2, 2),
                 step_size: tuple[int, int] = (2, 2)):

        self.filter_shape = filter_shape

        self.step_size = step_size

        # Calculate the size of the output
        output_size_x = int((input_shape[0] - filter_shape[1]) / step_size[0]) + 1
        output_size_y = int((input_shape[1] - filter_shape[1]) / step_size[1]) + 1
        output_shape = (output_size_x, output_size_y)

        # Set the base parameters
        super().__init__(previous_layer=previous_layer,
                         input_shape=input_shape,
                         output_shape=output_shape)

    def forward_prop(self, data):
        self.input = data

        self.output = self.avg_pool(data, self.filter_shape, self.step_size)
        return self.output

    def output_back_prop(self, learn_rate):
        self.back_prop()

    def hidden_back_prop(self, learn_rate):
        self.back_prop()

    def back_prop(self):
        lower_layer = self.next_layer
        # Get the elements that delta should be applied to - The maximums.
        self.loss = lower_layer.delta
        np.reshape(self.loss, self.output_shape)
        self.delta = self.back_max_pool(self.input, self.loss, self.filter_shape, self.step_size)

    @staticmethod
    def avg_pool(
            data,
            filter_size: tuple[int, int] = (2, 2),
            step_size: tuple[int, int] = (2, 2)):

        # Store the size of the data in each dimension
        z_size_data = data.shape[0]
        x_size_data = data.shape[1]
        y_size_data = data.shape[2]

        # Ease of reading variables
        x_size_kernel = filter_size[0]
        y_size_kernel = filter_size[1]
        stride_x = step_size[0]
        stride_y = step_size[1]

        # Calculate the size of the output
        output_size_z = z_size_data
        output_size_x = int((x_size_data - x_size_kernel) / stride_x) + 1
        output_size_y = int((y_size_data - y_size_kernel) / stride_y) + 1

        # Initialize the output
        output = np.zeros(shape=(output_size_z, output_size_x, output_size_y))

        # Run the convolution loop for each layer in the z-axis
        # Store the index of the output to place the next calculation result.
        z_output_index = 0
        for z_index in range(z_size_data):
            y_output_index = 0
            # We iterate over the data in the x-axis, stopping when the filter runs
            # up against the edge of the data, then going on to the next step of the y-axis.
            for y_index in range(0, (y_size_data - y_size_kernel + 1), stride_y):
                x_output_index = 0
                for x_index in range(0, (x_size_data - x_size_kernel + 1), stride_x):
                    # Get a view of the numpy array from the relevant indices to the size of the
                    # array. This does NOT copy any data.
                    # At the same time, flatten it using ravel, which also returns a view.
                    elements = data[z_index, x_index:x_index + x_size_kernel, y_index: y_index + y_size_kernel].ravel()

                    # Get the average of the elements from the area being convolved
                    average = np.average(elements, axis=None)
                    # Store the result
                    output[z_output_index][x_output_index][y_output_index] = average
                    # Increment the x-index of the output by 1.
                    x_output_index += 1

                # Increment the y-index of the output by 1.
                y_output_index += 1

            return output

    @staticmethod
    def back_avg_pool(
            data_in,
            loss,
            filter_size: tuple[int, int] = (2, 2),
            step_size: tuple[int, int] = (2, 2)):

        # This function returns a matrix in the same shape as the input where the supplied networkLoss is
        # placed at the index of the corresponding max value.

        # make a deep copy of the data
        data = np.copy(data_in)
        # Store the size of the data in each dimension
        z_size_data = data.shape[0]
        x_size_data = data.shape[1]
        y_size_data = data.shape[2]

        # Ease of reading variables
        x_size_kernel = filter_size[0]
        y_size_kernel = filter_size[1]
        stride_x = step_size[0]
        stride_y = step_size[1]

        # Run the convolution loop for each layer in the z-axis

        # Store the index of the output to place the next calculation result.
        z_output_index = 0
        for z_index in range(z_size_data):
            y_output_index = 0
            # We iterate over the data in the x-axis, stopping when the filter runs
            # up against the edge of the data, then going on to the next step of the y-axis.
            for y_index in range(0, (y_size_data - y_size_kernel + 1), stride_y):
                x_output_index = 0
                for x_index in range(0, (x_size_data - x_size_kernel + 1), stride_x):
                    # Get a view of the numpy array from the relevant indices to the size of the
                    # array. This does NOT copy any data.
                    # At the same time, flatten it using ravel, which also returns a view.
                    elements = data[z_index, x_index:x_index + x_size_kernel, y_index: y_index + y_size_kernel].ravel()

                    # Normalize elements to the get the effect of each index on the output,
                    # then multiply by the networkLoss
                    summation = np.sum(elements)
                    elements = elements / summation
                    elements = elements * loss[z_output_index][x_output_index][y_output_index]

                    # Reshape elements and set the corresponding portion of the data matrix to it
                    elements = np.reshape(elements, filter_size)
                    data[z_index, x_index:x_index + x_size_kernel, y_index: y_index + y_size_kernel] = elements
                    # Increment the x-index of the output by 1.
                    x_output_index += 1

                # Increment the y-index of the output by 1.
                y_output_index += 1

            # Increment the z-index of the output by 1.
            z_output_index += 1
        return data

    def get_layer_string(self):
        return "Average Pooling Layer" \
            + f"\nLayer Input Size: {self.input_shape}" \
            + f"\nLayer Output Size: {self.output_shape}"
