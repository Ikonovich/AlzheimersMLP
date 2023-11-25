import torch
from tensorflow import Tensor

from Layers.Layer import Layer


class MeanSquaredError:
    # Applies the Sigmoid activation to its input

    def __init__(self):
        self.actual = None
        self.expected = None
        self.loss = None
        self.prime = None  # Stores the online derivatives
        self.gradient = None

    def calculate_loss(self, actual: Tensor, expected: Tensor, grad: bool = True):
        self.actual = actual
        self.expected = expected
        self.loss, self.prime = self.mean_squared_error(
            actual=actual,
            expected=expected,
            grad=grad)
        return self.loss

    def back_prop(self):
        self.gradient = self.loss * self.prime
        return self.gradient

    # Calculate MSE and derivative
    # Value must be a 1D tensor
    @classmethod
    def mean_squared_error(cls, actual: Tensor, expected: Tensor, grad: bool = True):

        loss = torch.sum((torch.square(expected - actual) / len(expected)))

        prime = None
        if grad:
            prime = 2 * loss
        return loss, prime

    def get_layer_string(self):
        return f"\nLayer Type: Mean Squared Error" \
            + f"\nLayer Input Shape: {self.input_shape}" \
            + f"\nLayer Output Shape: {self.output_shape}"
