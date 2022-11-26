import numpy as np
from keras.datasets import mnist
from tqdm import tqdm

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
    empty = []
    n_previous_layer_nodes = n_inputs

    input_layer = {
        'activations' : empty,
        'zs' : empty
    }
    network.append(input_layer)

    for layer in range(n_hidden_layers):
        weights = np.random.uniform(size=(n_hidden_nodes,n_previous_layer_nodes)) # weights are dim (nodes in current layer, nodes in prev layer)
        biases = np.random.uniform(size=(n_hidden_nodes))
        hidden_layer = {
            'weights': weights,
            'biases' : biases,
            'activations' : empty,
            'zs' : empty,
        }
        network.append(hidden_layer)
        n_previous_layer_nodes = n_hidden_nodes

    weights = np.random.uniform(size=(n_outputs,n_previous_layer_nodes))
    biases = np.random.uniform(size=(n_outputs))
    output_layer = {
        'weights' : weights,
        'biases' : biases,
        'activations' : empty,
        'zs' : empty,
    }
    network.append(output_layer)

    return network

### Calculate z for a layer
## Inputs
# layer: an array consisting of the weights for all the nodes in the layer
# inputs: a 1D array consisting of the outputs from the previous layer
## Outputs
# z: array of node outputs for this layer where the node output is equivalent
#             to the dot product of the weights and the inputs + the bias
def calculate(layer, inputs):
    layer_w = layer['weights']
    layer_b = layer['biases']
    z = np.matmul(inputs,layer_w.T) + layer_b
    return z

### Calculate sigmoid of x
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

### Calculate derivative of sigmoid of x
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig*(1.0-sig)

### Calculate tanh of x
def tanh(x):
    return np.divide((np.exp(2*x)-1.0),(np.exp(2*x)+1.0))

### Calculate derivative of tanh of x
def tanh_derivative(x):
    return 1.0-np.square(x)

### Turns int into categorical array
## Ex: 2 => [0,0,1,0,0,0,0,0,0,0]
def to_output_array(input: int,n_output: int):
    output_array = [0] * n_output
    output_array[input] = 1
    return output_array

### Turns categorical array to int
def to_int(cat_array: np.ndarray):
    output_val = 0
    output = 0
    for i, val in enumerate(cat_array):
        if val > output_val:
            output_val = val
            output = i

    return output

### Calculates error of output layer
def output_layer_error(output, expected, z):
    return (expected-output) * sigmoid_derivative(z)

### Calculates error of hidden layer
def hidden_layer_error(z, weights, errors):
    derivative = sigmoid_derivative(z)
    weighted_errors = np.matmul(errors,weights)
    return np.multiply(weighted_errors,derivative)

### Forward propagate input through network and return output
### Save node outputs along the way
def forward_propagate(network,input):
    inputs = input
    for l, layer in enumerate(network):
        if l != 0:
            z = calculate(layer,inputs)
            inputs = sigmoid(z)
            layer['zs'] = z
        layer['activations'] = inputs
    return inputs

### Backward propagate output through network and update weights
def backward_propagate(network,output,expected,learning_rate):
    errors = None # 1D array of previous layer's errors

    # calculate errors for each layer
    for l, layer in enumerate(reversed(network)):
        l = len(network)-1 - l
        if l !=0:
            z = layer['zs']
            if l == len(network) - 1:
                errors = output_layer_error(output,expected,z)
            else:
                weights = network[l+1]['weights'] # use the weights connecting this layer to the next
                errors = hidden_layer_error(z,weights,errors)

            layer['errors'] = errors

    # update network weights
    for l, layer in enumerate(network):
        if l != 0:
            inputs = network[l-1]['activations'] # use the inputs to this layer (outputs of last layer)
            errors = layer['errors']
            old_weights = layer['weights']
            old_biases = layer['biases']

            # multiplies errors and inputs such that the resulting array is the same size as the weights
            nubla_w = learning_rate * np.matmul(errors.reshape((-1,1)),inputs.reshape((1,-1))) # reshaped to perform matrix multiplication on 1D arrays
            nubla_b = learning_rate * errors

            new_weights = old_weights + nubla_w
            new_biases = old_biases + nubla_b

            layer['weights'] = new_weights
            layer['biases'] = new_biases

### Trains the network according to the passed training data
def train(network,X_train: np.ndarray,y_train: np.ndarray,learning_rate):
    n_outputs = len(network[-1]["biases"])
    for s in tqdm(range(len(X_train))):
        sample = X_train[s].flatten()
        expected = to_output_array(y_train[s],n_outputs)
        output = forward_propagate(network,sample)
        # print(f'Output: {output}\nExpected: {expected}')
        backward_propagate(network,output,expected,learning_rate)

### Tests the network against the passed testing data
### Returns the accuracy of the network over the testing data
def test(network,X_test: np.ndarray,y_test: np.ndarray):
    error = 0
    total = 0
    for s in tqdm(range(len(X_test))):
        sample = X_test[s].flatten()
        output = forward_propagate(network,sample)
        # expected = to_output_array(y_test[s],n_outputs)
        #error = output_layer_error(output,expected,network[-1]['zs'])
        
        output_num = to_int(output)
        expected_num = y_test[s]
        if output_num != expected_num:
            error += 1
            print(f'Actual: {output}')
            print(f'Expected Num: {expected_num}\tActual Num: {output_num}')
        total += 1
    return 1-(error/total) # equivalent to accuracy


if __name__=='__main__':
    learning_rate = 0.001

    # load data
    print("Loading data...")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # normalize values between 0 and 1
    train_X = np.array([np.divide(sample,255) for sample in train_X])
    test_X = np.array([np.divide(sample,255) for sample in test_X])

    num_samples, dimension_x, dimension_y = train_X.shape 

    n_inputs = dimension_x*dimension_y
    n_hidden_layers = 2
    n_hidden_nodes = 16
    n_outputs = 10

    # init network
    print("Creating network...")
    model = initialize_network(n_inputs,n_hidden_layers,n_hidden_nodes,n_outputs)

    train_portion = 15000
    # train model
    print("Training model...")
    train(model,train_X[:train_portion],train_y[:train_portion],learning_rate)


    # for layer in model:
    #     print(layer)
    test_portion = 5
    # test model
    print("Testing model...")
    accuracy = test(model,test_X[:test_portion],test_y[:test_portion])
    print(accuracy)