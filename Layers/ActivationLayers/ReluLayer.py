import numpy as np

from Layers.Layer import Layer


class ReluLayer(Layer):
    # Applies the ReLU activation to its input

    def __init__(self, input_shape, previous_layer=None):
        self.input_shape = input_shape
        self.output_shape = input_shape

        super().__init__(
            input_shape=self.input_shape,
            output_shape= self.output_shape,
            previous_layer=previous_layer)

    def forward_prop(self, x: np.ndarray):
        self.output = self.relu(x)
        return self.output

    def output_back_prop(self, expected, learn_rate=None):
        # The results are the last output of this layer
        error = -(expected - self.output)
        prime = self.relu_prime(self.output)
        self.loss = error * prime
        self.back_prop()

    def hidden_back_prop(self, learn_rate=None):
        lower_layer = self.next_layer
        prime = self.relu_prime(self.output)
        self.loss = lower_layer.delta * prime
        return self.back_prop()

    def back_prop(self, delta):
        prime = self.relu_prime(self.output)
        self.delta = self.delta
        self.loss = delta * prime
        return self.loss

    # Calculate relu
    @classmethod
    def relu(cls, x: np.ndarray):
        return np.maximum(x, 0)

    # Calculate derivative of relu
    @classmethod
    def relu_prime(cls, x: np.ndarray):
        result = np.zeros(shape=x.shape)
        for i, xi in np.ndenumerate(x):
            if xi > 0:
                result[i] = 1
        return result

    def get_layer_string(self):
        return f"\nLayer Type: ReLu" \
            + f"\nLayer Input Shape: {self.input_shape}" \
            + f"\nLayer Output Shape: {self.output_shape}"
