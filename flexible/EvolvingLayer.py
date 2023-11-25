import copy
import math
import random

import numpy

from Layer import Layer
from Neuron import Neuron, generateWeight
from Core import vectorMul, relu, vectorSub, argmax, vectorDiv, sigmoid
from flexible.DataTools import load_mnist
from flexible.InputNeuron import InputNeuron
from flexible.OutputNeuron import OutputNeuron


# Represents a subnetwork containing internal layers whose connections and neuron counts can change over time.
class EvolvingNetwork(Layer):
    # Stores a list of float ranges, beginning at 0, and their corresponding mutations
    MUTATIONS = [(0.1, "DUPLICATE_LAYER"), (0.4, "DUPLICATE_NEURON"), (0.7, "RANDOM_NEURON"), (1.0, "REMOVE_NEURON")]
    # Stores a list of possible activation functions. All are equally likely to be applied.
    ACTIVATIONS = [relu, sigmoid]

    # Num neurons is a list of integers defining the number of neurons each layer will have.
    # By default, each neuron in each layer is connected to every neuron in the layers above and below it.
    def __init__(self, network, input_size: int, output_size: int, numNeurons: list):
        super().__init__(network, input_size, output_size)

        # Maps IDs to neurons
        self.idToNeuron = dict()
        # Maps output and hidden neuron IDs to their inputs
        self.idToInput = dict()
        # Maps output and hidden neuron IDs to their inputs
        self.idToOutput = dict()
        self.numNeurons = numNeurons

        self.layers = list()

        self.inputNeurons = list()
        self.outputNeurons = list()

        self.receivedOutputs = 0

        self.initializeLayers()

    def initializeLayers(self):
        # Initialize input neurons
        for i in range(self.input_size):
            neuronId = f"I{i}"
            neuron = InputNeuron(self, layerId=0, neuronId=neuronId)
            self.inputNeurons.append(neuron)
            self.idToNeuron[neuronId] = neuron
        self.layers.append(self.inputNeurons)

        # Initialize hidden/evolving layers
        for i in range(len(self.numNeurons)):
            count = self.numNeurons[i]
            layer = list()
            for j in range(count):
                neuronId = f"{i + 1}{self.generateId()}"
                neuron = Neuron(network=self, layerId=i + 1, neuronId=neuronId, input_size=len(self.layers[i]),
                                activation="relu", bias_mod=0.1)
                self.idToNeuron[neuronId] = neuron
                # Connects every neuron in this layer to the layer before it
                self.idToInput[neuronId] = [neuron.neuronId for neuron in self.layers[i]]
                layer.append(neuron)
            self.layers.append(layer)

        # Initialize output layer
        for i in range(self.output_size):
            neuronId = f"O{i}"
            neuron = OutputNeuron(network=self, layerId=len(self.layers), neuronId=neuronId,
                                  input_size=len(self.layers[-1]), bias_mod=0.1)
            self.idToNeuron[neuronId] = neuron
            # Connects every neuron in this layer to the layer before it
            self.idToInput[neuronId] = [neuron.neuronId for neuron in self.layers[-1]]
            self.outputNeurons.append(neuron)
        self.layers.append(self.outputNeurons)

        # Built the output mappings.
        # TODO: Make this less brute force.
        for neuronId in self.idToNeuron:
            self.idToOutput[neuronId] = list()

        for neuronId in self.idToInput:
            inputs = self.idToInput[neuronId]
            for inputId in inputs:
                self.idToOutput[inputId].append(neuronId)

    def forward(self, data: list[float]):
        if len(data) != self.input_size:
            raise ValueError("Layer input length must match set input size.")

        for i in range(self.input_size):
            self.inputNeurons[i].forwardInput(data[i])

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            for neuron in layer:
                neuron.forward()

    def backward(self, loss: list[float]):

        for i in range(len(self.outputNeurons)):
            neuron = self.outputNeurons[i]
            neuron.backwardOutput(loss[i])

        for layer in reversed(self.layers[1:-1]):
            for i in range(len(layer)):
                neuron = layer[i]
                neuron.backward()

    # Returns a random neuron ID not in the idToNeuron map.
    def generateId(self) -> int:
        newId = random.randint(100000, 999999)
        while newId in self.idToNeuron:
            newId = random.randint(100000, 999999)
        return newId

    # Returns a new random neuron weight.
    def getOutput(self) -> list[float]:
        output = list()
        for neuron in self.outputNeurons:
            output.append(neuron.output)
        return output

    # Selects a random neuron from a random hidden layer and duplicates it.
    def duplicateNeuron(self):
        layer = random.choice(self.layers[1:-1])
        layerIndex = self.layers.index(layer)
        oldNeuron = random.choice(layer)
        newNeuron = copy.deepcopy(oldNeuron)
        newId = self.generateId()
        newNeuron.neuronId = newId
        inputs = copy.deepcopy(self.idToInput[oldNeuron.neuronId])
        outputs = copy.deepcopy(self.idToOutput[oldNeuron.neuronId])

        self.addNeuron(layerIndex, newNeuron, inputs, outputs)

    # Adds a neuron to the network
    def addNeuron(self, layer: int, neuron: Neuron, inputs: list[str], outputs: list[str]):
        newId = neuron.neuronId

        self.idToNeuron[newId] = neuron
        self.idToInput[newId] = inputs
        self.idToOutput[newId] = outputs

        # Update the output maps
        if len(outputs) != len(neuron.weights):
            raise IndexError("Number of provided inputs must match number of neuron weights.")

        for neuronId in self.idToInput[newId]:
            self.idToOutput[neuronId].append(newId)

        # Update the input maps and generate a new weight for each receiving neuron.
        for neuronId in self.idToOutput[newId]:
            self.idToInput[neuronId].append(newId)
            self.idToNeuron[neuronId].weights.append(generateWeight())


def generateEvolvingNetwork(dataset, numNeurons: list[int] = None) -> EvolvingNetwork:
    if numNeurons is None:
        numNeurons = [16]

    x, y = dataset[0]
    x = numpy.array(x).flatten().tolist()
    input_size = len(x)
    output_size = 10
    en = EvolvingNetwork(network=None, input_size=input_size, output_size=output_size, numNeurons=numNeurons)

    return en


def trainEvolvingNetwork(dataset, network) -> float:
    x, y = dataset[0]
    x = numpy.array(x).flatten().tolist()

    # Store our iteration count and how often we want to reset accumulated loss and accuracy.
    count = 0
    lossCount = 100
    # Store cumulative network low and accuracy.
    cumAcc = 0
    cumLoss = 0
    for x, y in dataset:
        # Flatten the data
        x = numpy.array(x).flatten().tolist()
        # Normalize
        x = vectorDiv(x, 256)
        network.forward(data=x)
        result = network.getOutput()

        if argmax(result) == y:
            cumAcc += 1

        # One-hot encode label
        expected = [0 for i in range(10)]
        expected[y] = 1
        error = vectorSub(expected, result)
        networkLoss = sum(vectorMul(error, error))
        lossPrime = vectorMul(error, 2)  # Not strictly necessary
        network.backward(vectorMul(lossPrime, 0.01))

        cumLoss += networkLoss
        count += 1
        if count % lossCount == 0:
            print(f"Last {lossCount} mean loss: {cumLoss / lossCount}. Total accurate: {cumAcc}")
            cumLoss = 0
            cumAcc = 0
    return cumLoss / lossCount


def intTest():
    en = EvolvingNetwork(network=None, input_size=1, output_size=1, numNeurons=[])

    # Generate a bunch of ints between 0 and 255 and
    # guess if they're 128 or above
    # ints = [bin(x)[2:] for x in range(255)]
    ints = [1, 2, 3, 4, 6, 7, 8, 9]

    # Store cumulative networkLoss and how often we restart it.
    lossCount = 50
    cumAcc = 0
    cumLoss = 0
    for x in range(1000):
        num = random.choice(ints)
        en.forward(data=[num])
        result = en.getOutput()[0]

        if num > 5:
            expected = 1
            if result > 0.5:
                cumAcc += 1
        else:
            expected = 0
            if result < 0.5:
                cumAcc += 1

        error = expected - result
        networkLoss = math.sqrt(error * error)
        lossPrime = 2 * error
        en.backward([lossPrime * 0.01])

        cumLoss += networkLoss
        if x % lossCount == 0:
            print(f"Last {lossCount} mean loss: {cumLoss / lossCount}. Total accurate: {cumAcc}")
            cumLoss = 0
            cumAcc = 0


if __name__ == "__main__":
    ds = load_mnist()
    net = generateEvolvingNetwork(ds, [16])

    trainEvolvingNetwork(ds, net)
