import numpy as np
from keras.datasets import mnist
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

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

        # Storing these to print after running
        self.learning_rate = None
        self.n_inputs = n_inputs
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.configs = configs

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
        result = np.matmul(weights,inputs) + biases
        #result = np.matmul(weights,inputs)
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
        input_size = len(input)
        for l, layer in enumerate(self.network):
            layer['inputs'] = layer_input
            z = self.calculate(layer,layer_input)
            max_z = max(z)
            if l == len(self.network) - 1:
                layer_input = self.output_activation_method(z)/max_z # output layer activation function
            else:
                layer_input = self.hidden_activation_method(z)/max_z # hidden layer activation function

            layer['outputs'] = layer_input
            input_size = len(layer_input)

        return layer_input

    def back_propagate(self,expected,lr):
        errors = None
        for l, layer in enumerate(reversed(self.network)):
            l = len(self.network)-1 - l
            inputs = layer['inputs']
            outputs = layer['outputs']
            old_weights = layer['weights']
            old_biases = layer['biases']
            derivative = None
            if l == len(self.network) - 1:
                errors = expected - outputs
                derivative = self.output_activation_derivative(outputs)
            else:
                weights = self.network[l+1]['weights'] # use the weights connecting this layer to the next
                errors = weights.T @ errors
                derivative = self.hidden_activation_derivative(outputs)
            
            layer['weights'] = old_weights + lr * ((errors*derivative) @ inputs.T)
            layer['biases'] = old_biases + lr * (errors*derivative)

    ### Trains the network according to the passed training data
    def train(self,X_train: np.ndarray,y_train: np.ndarray,learning_rate):
        n_outputs = len(self.network[-1]["biases"])
        self.learning_rate = learning_rate

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
        y_actual = []
        y_pred = [] # Stores the results
        for s in tqdm(range(len(X_test))):
            sample = X_test[s]
            sample = sample.reshape(sample.size,1)
            
            output = self.forward_propagate(sample)
            
            output_num = self.to_int(output.flatten())
            expected_num = self.to_int(y_test[s])
            y_pred.append(expected_num)
            y_actual.append(output_num)
            if output_num != expected_num:
                error += 1
            total += 1
        self.print_results(y_actual, y_pred, total, error)

    def print_results(self, y_actual, y_predicted, total, error):

        accuracy = 1 - (error/total)
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_actual,
            y_predicted,
            labels=[0,1,2,3])
        print(f"--- Parameters --- "
              f"\nLearning rate: {self.learning_rate}"
              f"\nActivation: {self.hidden_activation_method.__name__}"
              f"\nNumber of Hidden Layers: {self.n_hidden_layers}"
              f"\nNumber of Neurons per Hidden Layer: {self.n_hidden_nodes}"
              f"\nLearning rate: {self.learning_rate}"
              f"\n\n--- Results --- "
              f"\nAccuracy: {accuracy}"
              f"\nPrecision: {precision}"
              f"\nRecall: {recall}"
              f"\nSupport: {support}")


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

# calculate leaky relu
def leaky_relu(x):
    output = []
    for xi in x:
        if xi > 0:
            output.append(xi)
        else:
            output.append(0.01 * xi)

    return np.array(output)

### Calculate derivative of leaky relu
def leaky_relu_derivative(x):
    output = []
    output = np.zeros(shape=(x.size,))
    for i, xi in enumerate(x):
        if xi > 0:
            output[i] = 1
        else:
            output[i] = 0.01
    return output.reshape(output.size , 1)

### Calculate tanh of x
def tanh(x):
    x = np.round(x,5)
    return np.divide((np.exp(2*x)-1.0),(np.exp(2*x)+1.0))

### Calculate derivative of tanh of x
def tanh_derivative(x):
    x = np.round(x,5)
    return 1.0-np.square(x)

### Calculate softmax of x
def softmax(x):
    e = np.exp(x)
    return e / sum(e)

### Calculate derivative of softmax of x
def softmax_derivative(x):
    s = softmax(x)
    return -s * s.reshape(s.shape[0],1)

def run_mnist():
    learning_rate = 0.0000001

    # load data
    print("Loading data...")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # normalize values between 0 and 1
    train_X = np.array([np.divide(sample,255) for sample in train_X])
    test_X = np.array([np.divide(sample,255) for sample in test_X])

    # flatten samples
    X_train = []
    X_test = []
    for train, test in zip(train_X,test_X):
        X_train.append(train.flatten())
        X_test.append(test.flatten())

    train_X = np.array(X_train)
    test_X = np.array(X_test)

    # turn labels into categorical arrays
    y_train = []
    y_test = []
    for train, test in zip(train_y,test_y):
        y_train.append(MLP_Model.to_output_array(None,train,10))
        y_test.append(MLP_Model.to_output_array(None,test,10))

    train_y = np.array(y_train)
    test_y = np.array(y_test)

    num_samples, dimension = train_X.shape 

    n_inputs = dimension
    n_hidden_layers = 1
    n_hidden_nodes = 64
    n_outputs = 10
    method_configs = {
        "hidden_activation_method" : leaky_relu,
        "hidden_activation_derivative" : leaky_relu_derivative,
        "output_activation_method" : sigmoid,
        "output_activation_derivative" : sigmoid_derivative,
    }

    # init network
    print("Creating network...")
    model = MLP_Model(n_inputs,n_hidden_layers,n_hidden_nodes,n_outputs,method_configs)

    # train model
    n_epochs = 20
    print("Training model...")
    for epoch in range(n_epochs):
        model.train(train_X,train_y,learning_rate)

    # test model
    print("Testing model...")
    model.test(test_X,test_y)
    

if __name__ == '__main__':
    run_mnist()