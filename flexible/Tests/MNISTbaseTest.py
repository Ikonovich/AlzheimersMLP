import unittest

from NeuralNetwork import NeuralNetwork
from flexible.DataTools import load_mnist
from flexible.EvolvingLayer import EvolvingNetwork, trainEvolvingNetwork


class TestStringMethods(unittest.TestCase):

    def testKnownGoodNewNeuronMutations(self):
        def

        ds = load_mnist()
        trainEvolvingNetwork(ds, [16])

if __name__ == "__main__":
    network = NeuralNetwork()

    layer = EvolvingNe(network, input_size=1, output_size=1)

