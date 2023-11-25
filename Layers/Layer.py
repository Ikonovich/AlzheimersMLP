import torch
from torch import float16

from Debugger import Debugger
from Math import Functions

import numpy as np


class Layer:
    # The basic layer class.

    def __init__(
            self,
            input_shape: int | tuple[int] = None,
            output_shape: int | tuple[int] = None,
            previous_layer=None):

        # Keeps saved parameters
        self.saved_state = None

        # Stores whether we are in a training phase.
        # Some layers, such as dropout, behave differently during training vs testing.
        self.training = False

        # Store the output shape of the layer
        self.output_shape = output_shape

        # If we're given an input shape, set it and create weights.
        # Otherwise, this must be done before running the network.
        self.input_shape = input_shape
        self.weights = None
        if input_shape is not None:
            self.set_input_shape(input_shape)

        # Stores references to previous and next layers.
        self.previous_layer = previous_layer
        self.next_layer = None

        # Stores the last provided input
        self.input = None
        # Stores the last calculated output
        self.output = None

        # Stores the last computed gradient of input wrt to weights, which is equal to input dot weights
        self.delta = None
        # Stores the last calculated networkLoss
        self.loss = None

    # Set the input shape and initialize the weights
    def set_input_shape(self, input_shape: tuple[int]):

        if self.input_shape is not None:

            # Combine the shapes of the input and output to get the shape
            # of the weight matrix.
            dim_list = list()
            if type(self.input_shape) is int:
                dim_list.append(self.input_shape)
            else:
                for dimension in self.input_shape:
                    dim_list.append(dimension)
            if type(self.output_shape) is int:
                dim_list.append(self.output_shape)
            elif self.output_shape is not None:
                for dimension in self.output_shape:
                    dim_list.append(dimension)

            # Generate the weights
            self.weights = torch.rand(tuple(dim_list), dtype=float16)

    def save(self):
        # Stores the parameters of the layer
        pass

    def load_saved(self, params=None):
        # Loads the stored parameters of the layer
        # If no params are provided, loads from self.saved_state
        # If params are provided, loads them
        pass