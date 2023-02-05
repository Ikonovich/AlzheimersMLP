import numpy as np
from numba import float32
from numba.experimental import jitclass

from Debugger import Debugger
from Math import Functions


class Layer:
    # The basic layer class.

    def __init__(
            self,
            input_shape: tuple[int, int] | int,
            output_shape: tuple[int, int] | int,
            dropout_modifier: float = 0.0,
            activation_string: str = "None",
            previous_layer=None):


        # Store the output shape of the layer
        self.output_shape = output_shape

        # If we're given an input shape, set it and create weights.
        # Otherwise, this must be done before running the network.
        self.input_shape = None
        self.weights = None
        if input_shape is not None:
            self.set_input_shape(input_shape)


        # Stores references to previous and next layers.
        self.previous_layer = previous_layer
        self.next_layer = None

        # Uses the provided activation string to get the activation function
        # as well as its derivative.
        self.activation_string = activation_string
        self.activation = Functions.ACTIVATION_FUNCTION_MAP[activation_string][0]
        self.derivative = Functions.ACTIVATION_FUNCTION_MAP[activation_string][1]


        # defines the drop-out rate
        self.dropout = dropout_modifier

        # Stores the last provided input
        self.input = None
        # Stores the last calculated output
        self.output = None
        # Stores the last calculated delta
        self.loss = None

    # Set the input shape and initialize the weights
    def set_input_shape(self, input_shape: tuple[int, int] | int):
        self.input_shape = input_shape

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
        else:
            for dimension in self.output_shape:
                dim_list.append(dimension)

        # Generate the weights
        self.weights = np.random.default_rng().random(tuple(dim_list))
