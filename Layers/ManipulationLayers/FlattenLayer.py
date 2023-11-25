import numpy as np

from Layers.Layer import Layer


class FlattenLayer(Layer):
    # Flattens the input during forward propagation, and reshapes it during back propagation.

    def __init__(self, input_shape, previous_layer=None):
        self.input_shape = input_shape

        size = 1
        for i in range(len(input_shape)):
            size *= input_shape[i]

        super().__init__(
            input_shape=input_shape,
            output_shape=[size],
            previous_layer=previous_layer)

    def forward_prop(self, x: np.ndarray):
        self.output = x.flatten()
        return self.output

    def output_back_prop(self, expected, learn_rate):
        return self.back_prop()

    def hidden_back_prop(self, learn_rate):
        return self.back_prop()

    def back_prop(self):
        lower_layer = self.next_layer
        # Reshape the backpropagated delta
        self.delta = np.reshape(lower_layer.delta, self.input_shape)
        return self.delta

    def get_layer_string(self):
        return "Flatten Layer" \
            + f"\nLayer Input Size: {self.input_shape}" \
            + f"\nLayer Output Size: {self.output_shape}"
