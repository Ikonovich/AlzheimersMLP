import numpy as np

from Layers.Layer import Layer


class Sigmoid(Layer):
    # Applies the Sigmoid activation to its input

    def __init__(self, input_shape, previous_layer=None):
        self.input_shape = input_shape
        self.output_shape = input_shape

        super().__init__(
            input_shape=self.input_shape,
            output_shape= self.output_shape,
            previous_layer=previous_layer)

    def forward_prop(self, x: np.ndarray):
        self.input = x
        self.output = self.sigmoid(x)
        return self.output

    def output_back_prop(self, expected, learn_rate=None):
        # The results are the last output of this layer
        error = -(expected - self.output)
        prime = self.sigmoid_prime(self.output)
        self.loss = error * prime
        self.back_prop()

    def hidden_back_prop(self, learn_rate=None):
        lower_layer = self.next_layer
        prime = self.sigmoid_prime(self.output)
        self.loss = lower_layer.delta * prime
        return self.back_prop()

    def back_prop(self):
        # Reshape the back propagated delta
        self.delta = self.loss

    # Calculate sigmoid of x
    @classmethod
    def sigmoid(cls, x: np.ndarray):
        result = []
        for num in x:
            if num >= 0:
                result.append(1. / (1. + np.exp(-num)))
            else:
                result.append(np.exp(num) / (1. + np.exp(num)))
        return result

    # Calculate derivative of sigmoid of x
    @classmethod
    def sigmoid_prime(cls, x: np.ndarray):
        sig = cls.sigmoid_two(x)
        return sig * (1.0 - sig)

    # Calculate sigmoid of x with a different method, better for the derivative
    @classmethod
    def sigmoid_two(cls, x: np.ndarray):
        return np.piecewise(x, [x > 0],
                            [lambda i: 1 / (1 + np.exp(-i)),
                             lambda i: np.exp(i) / (1 + np.exp(i))])

    def get_layer_string(self):
        return f"\nLayer Type: Sigmoid" \
            + f"\nLayer Input Shape: {self.input_shape}" \
            + f"\nLayer Output Shape: {self.output_shape}"