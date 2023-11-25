import torch
from tensorflow import Tensor

from Layers.Layer import Layer


class Softmax(Layer):
    # Applies the Sigmoid activation to its input

    def __init__(self, input_shape, previous_layer=None):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.prime = None  # Stores the online derivatives
        self.gradient = None

        super().__init__(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            previous_layer=previous_layer)

    def forward_prop(self, x: Tensor, grad: bool = True):
        self.input = x
        self.output, self.prime = self.softmax(x=x, grad=grad)
        return self.output

    def back_prop(self, loss, learn_rate=None):
        self.loss = loss
        self.gradient = loss * self.prime

    # Calculate softmax of x
    # Value must be a 1D tensor
    @classmethod
    def softmax(cls, x: Tensor, grad: True) -> (Tensor, Tensor | None):
        gradient = None

        max_x = torch.max(x)
        x = x - max_x  # Increases numerical stability
        exp_x = torch.exp(x)
        sum_x = torch.sum(exp_x)
        softmax_x = torch.div(exp_x, sum_x)

        # Calculate the derivative if necessary
        prime = None
        if grad:
            identity = torch.eye(softmax_x.shape[-1])
            grad_one = torch.einsum('ij,jk->ijk', softmax_x, identity)
            grad_two = torch.einsum('ij,jk->ijk', softmax_x, softmax_x)
            prime = grad_one - grad_two

        return softmax_x, prime

    def get_layer_string(self):
        return f"\nLayer Type: Softmax" \
            + f"\nLayer Input Shape: {self.input_shape}" \
            + f"\nLayer Output Shape: {self.output_shape}"
