from numba import float32
from numba.experimental import jitclass

from Debugger import Debugger
from Math import Functions


class Layer:
    # The basic layer class.

    def __init__(
            self,
            input_size: tuple[int, int] | int,
            output_size: tuple[int, int] | int,
            dropout_modifier: float = 0.0,
            activation_string: str = "None",
            previous_layer=None):

        # Stores references to previous and next layers.
        self.previous_layer = previous_layer
        self.next_layer = None

        # Stores the input and output size of the layer
        self.input_size = None
        self.output_size = output_size

        # Uses the provided activation string to get the activation function
        # as well as its derivative.
        self.activation_string = activation_string
        self.activation = Functions.ACTIVATION_FUNCTION_MAP[activation_string][0]
        self.derivative = Functions.ACTIVATION_FUNCTION_MAP[activation_string][1]


        # defines the drop-out rate
        self.dropout = dropout_modifier

        # Stores the last provided input
        self.last_input = None
        # Stores the last calculated output
        self.last_output = None
        # Stores the last calculated delta
        self.last_delta = None

    # Used to set the actual input size
    def set_input_size(self, input_size):
        self.input_size = input_size
