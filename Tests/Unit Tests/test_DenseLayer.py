import torch.nn as tnn
import torch
import NeuralNetwork as net


class DenseLayerTests:

    def __init__(self):
        # A test layer for testing with bias.
        self.layer_b = net.DenseLinear(output_features=4, input_features=4, bias_modifier=0.1)
        # Compare b is meant to be tested against layer b.
        self.compare_b = tnn.Linear(in_features=4, out_features=4, bias=True)

        # Set the layer weights to the same as compare_b
        self.layer_b.weights = self.compare_b.weight
        self.layer_b.bias = self.compare_b.bias

        # Model with bias
        self.layer_w = net.DenseLinear(output_features=4, input_features=4, bias_modifier=0.1)
        self.compare_w = tnn.Linear(in_features=4, out_features=4, bias=True)
        self.layer_w.weights = self.compare_w.weight

    def test_forward_prop(self):
        x_tensor = torch.rand((12, 12))
        x_numpy = x_tensor.numpy()

        result_w = self.layer_w.forward_prop(x_numpy)
        result_b = self.layer_w.forward_prop(x_numpy)

        baseline_w = self.compare_w(x_tensor)
        baseline_b = self.compare_b(x_tensor)

        print(f"Output: {result_w}")
        print(f"Output: {baseline_w}")


        print("Fin")
