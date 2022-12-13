from numba import float32
from numba.experimental import jitclass

from Debugger import Debugger
from Math import Functions


class Layer:
    # The basic layer class.

    def __init__(self, activation, dropout_modifier):

        # Uses the provided activation string to get the activation function
        # as well as its derivative.
        self.activation_string = activation
        self.activation = Functions.ACTIVATION_FUNCTION_MAP[activation][0]
        self.derivative = Functions.ACTIVATION_FUNCTION_MAP[activation][1]

        # defines the drop-out rate
        self.dropout = dropout_modifier

