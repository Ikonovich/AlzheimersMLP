from typing import cast

from flexible.Neuron import Neuron



class InputNeuron(Neuron):

    def __init__(self, network, layerId: int, neuronId: str):
        super().__init__(network, layerId=layerId, neuronId=neuronId, input_size=1)


    def forwardInput(self, data):
        self.output = data

    @staticmethod
    def toDict(neuron: Neuron, inputIds: list[str], outputIds: list[str]) -> dict:
        neuronDict = Neuron.toDict(neuron=neuron,
                                   inputIds=inputIds,
                                   outputIds=outputIds)
        neuronDict["type"] = "InputNeuron"
        return neuronDict

    @staticmethod
    def fromDict(neuronDict: dict, network=None) -> "InputNeuron":
        neuron = Neuron.fromDict(neuronDict=neuronDict,
                                 network=network)
        neuron = cast(InputNeuron, neuron)
        return neuron

