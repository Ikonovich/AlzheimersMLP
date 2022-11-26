from cmath import exp
import random
from turtle import back, backward
import numpy as np

### Initializes the network
## Inputs
# n_inputs: number of inputs
# n_hidden_layers: number of hidden layers
# n_hidden_nodes: number of nodes per hidden layer
# n_outputs: number of outputs
## Outputs
# network: list of dicts containing each layer's weights and outputs
def initialize_network(n_inputs, n_hidden_layers, n_hidden_nodes, n_outputs):
    network = list()
    outputs = []
    n_previous_layer_nodes = n_inputs

    input_layer = {
        'outputs' : outputs
    }
    network.append(input_layer)

    for layer in range(n_hidden_layers):
        weights = np.random.uniform(size=(n_hidden_nodes,n_previous_layer_nodes)) # weights are dim (nodes in current layer, nodes in prev layer)
        biases = np.random.uniform(size=(n_hidden_nodes))
        hidden_layer = {
            'weights': weights,
            'biases' : biases,
            'outputs': outputs,
        }
        network.append(hidden_layer)
        n_previous_layer_nodes = n_hidden_nodes

    weights = np.random.uniform(size=(n_outputs,n_previous_layer_nodes))
    biases = np.random.uniform(size=(n_outputs))
    output_layer = {
        'weights' : weights,
        'biases' : biases,
        'outputs' : outputs,
    }
    network.append(output_layer)

    return network

### Calculate activation for a layer
## Inputs
# layer: an array consisting of the weights for all the nodes in the layer
# inputs: a 1D array consisting of the outputs from the previous layer
## Outputs
# activation: array of node outputs for this layer where the node output is equivalent
#             to the dot product of the weights and the inputs + the bias
def activate(layer, inputs):
    layer_w = layer['weights']
    layer_b = layer['biases']
    activation = np.matmul(inputs,layer_w.T) + layer_b
    return activation

### Calculate sigmoid of x
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

### Calculate derivative of sigmoid of x
def sigmoid_derivative(x):
    return x*(1.0-x)

### Forward propagate input through network and return output
### Save node outputs along the way
def forward_propagate(network,input):
    inputs = input
    for l, layer in enumerate(network):
        if l != 0:
            inputs = sigmoid(activate(layer,inputs))
        layer['outputs'] = inputs
    return inputs

### Backward propagate output through network and update weights
def backward_propagate(network,output,expected,learning_rate):
    errors = None # 1D array of previous layer's errors
    inputs = None

    # calculate errors for each layer
    for l, layer in enumerate(reversed(network)):
        l = len(network)-1 - l
        if l !=0:
            if l == len(network) - 1:
                errors = (output - expected) * sigmoid_derivative(output)
            else:
                outputs = layer['outputs']
                weights = network[l+1]['weights']
                try:
                    derivative = sigmoid_derivative(outputs)
                    weighted_errors = np.matmul(errors,weights)
                    errors = np.multiply(weighted_errors,derivative)
                except:
                    print(f'Errors: \nShape: {errors.shape}\n{errors}')
                    print(f'Weights: \nShape: {weights.shape}\n{weights}')
                    print(f'Outputs: \nShape: {outputs.shape}\n{outputs}')
                    return
            layer['errors'] = errors

    # update network weights
    for l, layer in enumerate(network):
        if l != 0:
            inputs = network[l-1]['outputs']
            errors = layer['errors']
            old_weights = layer['weights']
            old_biases = layer['biases']

            try:
                delta_w = learning_rate * np.matmul(errors.reshape((-1,1)),inputs.reshape((1,-1)))
            except:
                print(f'Errors: \nShape: {errors.reshape((-1,1)).shape}\n{errors.reshape((-1,1))}')
                print(f'Weights: \nShape: {old_weights.shape}\n{old_weights}')
                print(f'Inputs: \nShape: {inputs.reshape((1,-1)).shape}\n{inputs.reshape((1,-1))}')
                return
            delta_b = learning_rate * errors

            new_weights = old_weights + delta_w
            try:
                new_biases = old_biases + delta_b
            except:
                print(f'Level: {l}')
                print(f'Errors: \nShape: {errors.shape}\n{errors}')
                print(f'Biases: \nShape: {old_biases.shape}\n{old_biases}')
                print(f'Delta B: \nShape: {delta_b.shape}\n{delta_b}')
                return
            layer['weights'] = new_weights
            layer['biases'] = new_biases

### Trains the network according to the passed training data
def train(network,X_train,y_train,learning_rate):
    for s, sample in enumerate(X_train):
        expected = y_train[s]
        output = forward_propagate(network,sample)
        backward_propagate(network,output,expected,learning_rate)

### Tests the network against the passed testing data
### Returns the accuracy of the network over the testing data
def test(network,X_test,y_test):
    errors = []
    for s, sample in enumerate(X_test):
        expected = y_test[s]
        output = forward_propagate(network,sample)
        errors.append(abs(expected-output))
    return 1-np.mean(errors) # equivalent to accuracy


if __name__=='__main__':
    n_inputs = 2
    n_hidden_layers = 3
    n_hidden_nodes = 5
    n_outputs = 1 # binary classification test

    net = initialize_network(n_inputs,n_hidden_layers,n_hidden_nodes,n_outputs)

    input = np.arange(n_inputs)
    output = forward_propagate(net,input)
    # print(net)
    expected = [1]
    backward_propagate(net,output,expected,100)
    for l, layer in enumerate(net):
        print(f"Layer {l}")
        print(layer)
