import numpy as np

from Layers.Layer import Layer

# Randomly drops out the provided ratio of inputs
class DropoutLayer(Layer):


    # The provided ratio is as a fraction of 1. For example, if 0.3 is provided,
    # 30% of all inputs will be set to 0.
    def __init__(self, ratio):
        self.ratio = ratio


    def forward_prop(self, data_in: np.array):

        for item in np.nditer(data_in):
           if np.random.random() < self.ratio:
               item = 0


