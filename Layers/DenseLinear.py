import numpy as np
import torch
from torch import Tensor, float16, float32

from Math import Functions


# Represents a layer of a neural network
from Layers.Layer import Layer


class DenseLinear(Layer):

    def __init__(self, output_features: int | tuple[int], input_features: int = None, bias_modifier=0.0,
                 previous_layer=None):

        # Stores the input size for normalization
        self.input_size = None

        # Set the input and output sizes, dropout rate, and activation function
        # Also initializes a pretty printer as pprint
        super().__init__(
            input_shape=input_features,
            output_shape=output_features,
            previous_layer=previous_layer)

        # The bias learning modifier, applied on top of the
        # regular learning rate. I.E: Setting to 0.1 will cause the learning rate to
        # be 1/10th of the regular learning rate.
        # Set to 0 to disable bias.
        if bias_modifier is None:
            self.bias_lrn_modifier = 0.0
        else:
            self.bias_lrn_modifier = bias_modifier

        # Stores the per-neuron bias
        self.bias = torch.zeros(output_features, dtype=float16)

    # Handles forward propagation when given input data
    # The size of data_in must match the input size of this layer
    def forward_prop(self, data_in: Tensor) -> Tensor:
        # Unravels the input into a 1D array,
        self.input = data_in.ravel()
        self.output = (torch.matmul(self.input, self.weights) / self.input_size)

        if self.bias_lrn_modifier != 0:
            self.output += self.bias
        return self.output

    def get_output(self):
        return self.output

    # Performs backprop for the output layer. Takes in a list of expected results.
    # def output_back_prop(self, expected, learn_rate):
    #     # The results are the last output of this layer
    #     error = -(expected - self.output)
    #     prime = self.derivative(self.output)
    #
    #     self.networkLoss = error * prime
    #     self.back_prop(learn_rate=learn_rate)
    #
    # def hidden_back_prop(self, learn_rate):
    #     lower_layer = self.next_layer
    #     prime = self.derivative(self.output).ravel()
    #     self.networkLoss = lower_layer.delta * prime
    #     self.back_prop(learn_rate=learn_rate)
    #
    # def back_prop(self, learn_rate):
    #     # We store the delta to be back propagated before updating the weights
    #     self.delta = np.matmul(self.networkLoss, self.weights.T)
    #
    #     adjustedInput = self.input[np.newaxis]
    #     adjustedLoss = self.networkLoss[np.newaxis]
    #     weight_updates = np.matmul(adjustedInput.T, adjustedLoss * learn_rate)
    #     bias_updates = self.networkLoss * learn_rate * self.bias_lrn_modifier
    #     self.weights += weight_updates
    #     self.bias += bias_updates

    def back_prop(self, delta: Tensor):
        self.loss = delta
        adjustedInput = torch.unsqueeze(self.input, dim=-1)
        adjustedDelta = torch.unsqueeze(self.loss, dim=-1)
        weight_updates = np.matmul(adjustedInput, adjustedDelta)
        bias_updates = self.loss * self.bias_lrn_modifier

        self.delta = np.matmul(self.loss, self.weights.T)

        self.weights -= weight_updates
        self.bias -= bias_updates
        return self.delta

    def get_layer_string(self):
        return f"\nFully Connected Layer" \
                + f"\nLayer Input Size: {self.input_shape}" \
                + f"\nLayer Output Size: {self.output_shape}" \
                + f"\nBias Modifier: {self.bias_lrn_modifier}\n"

    # Set the input shape and initialize the weights
    # Set the input shape and initialize the weights
    def set_input_shape(self, input_shape: int | tuple[int]) -> None:
        self.input_shape = input_shape

        # Combine the shapes of the input and output to get the shape
        # of the weight matrix.
        dim_list = list()
        if type(self.input_shape) is int:
            dim_list.append(self.input_shape)
            self.input_size = self.input_shape
        else:
            dim = 1
            for dimension in self.input_shape:
                dim = dim * dimension
            dim_list.append(dim)
            self.input_size = dim
        if type(self.output_shape) is int:
            dim_list.append(self.output_shape)
        else:
            dim = 1
            for dimension in self.output_shape:
                dim = dim * dimension
            dim_list.append(dim)

        # Generate the weights
        self.weights = torch.rand(tuple(dim_list), dtype=float32)

    def save(self):
        self.saved_state = self.weights

    def load_saved(self, params=None):
        # Loads the stored parameters of the layer
        # If no params are provided, loads from self.saved_state
        # If params are provided, loads them
        if params is not None:
            if params.shape is not self.weights.shape:
                raise Exception("Parameter input and weights matrix must be the same dimensions.")
            self.weights = params
        else:
            self.weights = self.saved_state

