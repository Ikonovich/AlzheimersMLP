import numpy as np

from Layers.Layer import Layer

# Randomly drops out the provided ratio of inputs
class DropoutLayer(Layer):
    # This layer drops out the provided percentage of neurons randomly, when training is true.


    # The provided ratio is as a fraction of 1. For example, if 0.3 is provided,
    # 30% of all inputs will be set to 0.
    def __init__(self, drop_ratio: float, input_shape: tuple, previous_layer: Layer):

        super().__init__(
            input_shape=input_shape,
            output_shape=input_shape,
            previous_layer=previous_layer)

        self.drop_ratio = drop_ratio

        self.input_shape = input_shape
        self.output_shape = input_shape

        # Stores a set of indexes that have been removed in the last iteration,
        # to avoid duplicate removals and assist in backprop
        self.removed = set()

        # Determine the linear length of the input
        if type(self.input_shape) is int:
            self._input_length = self.input_shape
        else:
            self._input_length = 1
            for value in self.input_shape:
                self._input_length *= value

        # Determines the number of neurons to be dropped from each iteration
        self.drop_count = min(int(self._input_length * self.drop_ratio), self._input_length)

    def forward_prop(self, data_in: np.array):
        if self.training == False:
            # If we're not training, don't perform any dropout.
            self.output = data_in

        else:
            # Performs the dropout operation

            # Stores the input value as a flattened view to make manipulating it easier
            flat_input = data_in.ravel()

            # Clear the removed indexes
            self.removed = set()
            # Stores the number of dropped neurons so far
            count = 0
            while count < self.drop_count:
                index = np.random.randint(0, self._input_length)
                if index not in self.removed:
                    flat_input[index] = 0
                    self.removed.add(index)
                    count += 1
            self.output = data_in
        return self.output

    def hidden_back_prop(self, learn_rate):
        self.back_prop(learn_rate)

    def output_back_prop(self, learn_rate):
        self.back_prop(learn_rate)

    def back_prop(self, learn_rate):
        delta = self.next_layer.delta

        view = delta.ravel()

        for index in self.removed:
            view[index] = 0

        self.delta = delta

    def get_layer_string(self):
        return f"\nLayer Type: Dropout" \
            + f"\nDropout Rate: {self.drop_ratio}" \
            + f"\nLayer Input Shape: {self.input_shape}" \
            + f"\nLayer Output Shape: {self.output_shape}"
