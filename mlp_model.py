import numpy as np
from keras.datasets import mnist
from tqdm import tqdm

hidden_activation_method = None
hidden_activation_derivative = None
output_activation_method = None
output_activation_derivative = None

class MLP_Model():
    ### Initializes the network
    ## Inputs
    # n_inputs: number of inputs
    # n_hidden_layers: number of hidden layers
    # n_hidden_nodes: number of nodes per hidden layer
    # n_outputs: number of outputs
    # configs: activation methods
    ## Outputs
    # network: list of dicts containing each layer's weights and outputs
    def __init__(self, n_inputs, n_hidden_layers, n_hidden_nodes, n_outputs, configs):
        self.hidden_activation_method = configs["hidden_activation_method"]
        self.hidden_activation_derivative = configs["hidden_activation_derivative"]
        self.output_activation_method = configs["output_activation_method"]
        self.output_activation_derivative = configs["output_activation_derivative"]

        network = list()
        empty = []
        n_previous_layer_nodes = n_inputs

        for layer in range(n_hidden_layers):
            weights = np.random.uniform(size=(n_hidden_nodes,n_previous_layer_nodes)) # weights are dim (nodes in current layer, nodes in prev layer)
            biases = np.random.uniform(size=(n_hidden_nodes,1))
            hidden_layer = {
                'weights': weights,
                'biases' : biases,
                'inputs' : empty,
                'outputs' : empty,
            }
            network.append(hidden_layer)
            n_previous_layer_nodes = n_hidden_nodes

        weights = np.random.uniform(size=(n_outputs,n_hidden_nodes))
        biases = np.random.uniform(size=(n_outputs,1))
        output_layer = {
            'weights' : weights,
            'biases' : biases,
            'inputs' : empty,
            'outputs' : empty,
        }
        network.append(output_layer)

        self.network = network

    def calculate(self, layer,inputs):
        weights = layer['weights']
        biases = layer['biases']
        # result = np.add(np.matmul(weights,inputs),biases)
        result = np.matmul(weights,inputs)
        return result

    ### Turns int into categorical array
    ## Ex: 2 => [0,0,1,0,0,0,0,0,0,0]
    def to_output_array(self, input: int,n_output: int):
        output_array = [0] * n_output
        output_array[input] = 1
        return np.array(output_array)

    ### Turns categorical array to int
    def to_int(self, cat_array: np.ndarray):
        output_val = 0
        output = 0
        for i, val in enumerate(cat_array):
            if val > output_val:
                output_val = val
                output = i

        return output

    def forward_propagate(self,input):
        layer_input = input
        for l, layer in enumerate(self.network):
            layer['inputs'] = layer_input
            z = self.calculate(layer,layer_input)
            if l == len(self.network) - 1:
                layer_input = self.output_activation_method(z) # output layer activation function
            else:
                layer_input = self.hidden_activation_method(z) # hidden layer activation function

            layer['outputs'] = layer_input

        return layer_input

    def back_propagate(self,expected,lr):
        errors = None
        for l, layer in enumerate(reversed(self.network)):
            l = len(self.network)-1 - l
            inputs = layer['inputs']
            outputs = layer['outputs']
            old_weights = layer['weights']
            if l == len(self.network) - 1:
                errors = expected - outputs
            else:
                weights = self.network[l+1]['weights'] # use the weights connecting this layer to the next
                errors = weights.T @ errors
            
            derivative = self.hidden_activation_derivative(outputs)
            layer['weights'] = old_weights + lr * ((errors*derivative) @ inputs.T)

    ### Trains the network according to the passed training data
    def train(self,X_train: np.ndarray,y_train: np.ndarray,learning_rate):
        n_outputs = len(self.network[-1]["biases"])
        for s in tqdm(range(len(X_train))):
            sample = X_train[s]
            sample = sample.reshape(sample.size,1)

            expected = y_train[s]
            expected = expected.reshape(expected.size,1)

            output = self.forward_propagate(sample)
            self.back_propagate(expected,learning_rate)

    ### Tests the network against the passed testing data
    ### Returns the accuracy of the network over the testing data
    def test(self,X_test: np.ndarray,y_test: np.ndarray):
        error = 0
        total = 0
        for s in tqdm(range(len(X_test))):
            sample = X_test[s]
            sample = sample.reshape(sample.size,1)
            
            output = self.forward_propagate(sample)
            
            output_num = self.to_int(output.flatten())
            expected_num = self.to_int(y_test[s])
            if output_num != expected_num:
                error += 1
            total += 1
        return 1-(error/total) # equivalent to accuracy

### Calculate sigmoid of x
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

### Calculate derivative of sigmoid of x
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig*(1.0-sig)

### Calculate relu
def relu(x):
    return np.maximum(x,0)

### Calculate derivative of relu
def relu_derivative(x):
    output = []
    output = np.zeros(shape=(x.size,))
    for i, xi in enumerate(x):
        if xi > 0:
            output[i] = 1
    return output.reshape(output.size,1)

### Calculate tanh of x
def tanh(x):
    x = np.round(x,5)
    return np.divide((np.exp(2*x)-1.0),(np.exp(2*x)+1.0))

### Calculate derivative of tanh of x
def tanh_derivative(x):
    x = np.round(x,5)
    return 1.0-np.square(x)

def run_mnist():
    learning_rate = 0.005

    # load data
    print("Loading data...")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # normalize values between 0 and 1
    train_X = np.array([np.divide(sample,255) for sample in train_X])
    test_X = np.array([np.divide(sample,255) for sample in test_X])

    num_samples, dimension_x, dimension_y = train_X.shape 

    n_inputs = dimension_x*dimension_y
    n_hidden_layers = 1
    n_hidden_nodes = 16
    n_outputs = 10
    method_configs = {
        "hidden_activation_method" : relu,
        "hidden_activation_derivative" : relu_derivative,
        "output_activation_method" : sigmoid,
        "output_activation_derivative" : sigmoid_derivative,
    }

    # init network
    print("Creating network...")
    model = MLP_Model(n_inputs,n_hidden_layers,n_hidden_nodes,n_outputs,method_configs)

    train_portion = 15000
    # train model
    print("Training model...")
    model.train(train_X[:train_portion],train_y[:train_portion],learning_rate)

    test_portion = 2000
    # test model
    print("Testing model...")
    accuracy = model.test(test_X[:test_portion],test_y[:test_portion])
    print(accuracy)

if __name__ == '__main__':
    run_mnist()