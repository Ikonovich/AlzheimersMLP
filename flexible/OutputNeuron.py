from typing import Callable, cast

from flexible.Core import vectorMul, vectorAdd
from flexible.Neuron import Neuron


class OutputNeuron(Neuron):

    def __init__(self, network, layerId: int, neuronId: str, input_size: int,
                 activation: Callable[[float], tuple[float, float]] = None,
                 bias_mod: float = 0.1):
        super().__init__(network=network, layerId=layerId, neuronId=neuronId, input_size=input_size,
                         activation=activation, bias_mod=bias_mod)

    def backwardOutput(self, loss: float):
        weightedLoss = vectorMul(self.input, loss)
        delta = loss * self.output_prime
        self.loss = list()
        for i in range(len(self.weights)):
            self.loss.append(self.weights[i] * delta)
        weightShift = vectorMul(weightedLoss, self.output_prime)
        self.weights = vectorAdd(self.weights, weightShift)
        self.bias += loss * self.bias_mod

    @staticmethod
    def toDict(neuron: Neuron, inputIds: list[str], outputIds: list[str]) -> dict:
        neuronDict = Neuron.toDict(neuron=neuron,
                                   inputIds=inputIds,
                                   outputIds=outputIds)
        neuronDict["ty[e"] = "OutputNeuron"
        return neuronDict

    @staticmethod
    def fromDict(neuronDict: dict, network=None) -> "OutputNeuron":
        neuron = Neuron.fromDict(neuronDict=neuronDict,
                                 network=network)
        neuron = cast(OutputNeuron, neuron)
        return neuron
