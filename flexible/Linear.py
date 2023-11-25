import random

import Core
from flexible.Layer import Layer


class Linear(Layer):

    def __init__(self, network, input_size: int, output_size: int, useBias: bool):
        super().__init__(network, input_size, output_size)

        self.weights = Core.matrix(rows=output_size, cols=output_size)
        (random.random() - 0.5)

        self.outputs = [0 for i in range(output_size)]
        self.inputs = [0 for i in range(input_size)]

        if useBias:
            self.biasRate = 1
        else:
            self.biasRate = 0
        self.bias = [0 for i in range(output_size)]

        self.loss = [0 for i in range(output_size)]
        self.delta = [0 for i in range(input_size)]

    def forward(self, data: list[float]):

        for i in range(self.output_size):
            total = 0
            for j in range(self.input_size):
                weight = self.weights[i][j]
                total += weight * data[j]
            total += self.bias[i] * self.biasRate
            self.output[i] = total

    def backward(self, loss: list[float]):
        self.loss = loss

        for i in range(self.output_size):
            for j in range(self.input_size):
                value = self.input[j]
                self.weights[i][j] -= (loss[i] * value)

        self.delta = n


