import numpy as np
import torch
from keras.datasets import mnist, imdb
from torch import nn, float16, float32
from torchvision import datasets
from torchtext.datasets import IMDB

from DataHandling.DataTools import tokenize, one_hot_encode, split, load_glove, get_k_most_common, remove_elements, \
    find_k_nearest_embeddings, load_imdb, to_embeddings_index

import ModelParams

from Layers.MaxPoolingLayer import MaxPoolingLayer
from Tests.ComparisonModels.TorchConv import TorchConv
from Tests.ComparisonModels.TorchLinear import TorchLinear
from Layers.ActivationLayers.ReluLayer import ReluLayer
from Layers.ActivationLayers.Sigmoid import Sigmoid

from Layers.ConvolutionLayer import ConvolutionalLayer
from Layers.DenseLinear import DenseLinear
from Layers.ManipulationLayers.DropoutLayer import DropoutLayer
from Layers.ManipulationLayers.FlattenLayer import FlattenLayer
from NeuralNetwork import FromDictionary, NeuralNetwork

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from Tests.ComparisonModels.TorchRecurrent import TorchRNN


# This module stores a variety of tests for use with the NeuralNetwork class and associated files.


def prep_mnist(subset=False, subset_train_ratio=0.9):
    # Used to retrieve and prepare the mnist dataset for testing
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = torch.tensor(train_x, dtype=float32)
    test_x = torch.tensor(test_x, dtype=float32)

    if subset == True:
        subset_count = int(len(train_x) * 0.1)
        train_x = train_x[:subset_count]
        train_y = train_y[:subset_count]

    # Convert y from integers into one-hot encoding
    train_y_arr = []
    for val in train_y:
        newVec = [0] * 10
        newVec[int(val)] = 1
        train_y_arr.append(newVec)
    train_y = torch.asarray(train_y_arr)

    test_y_arr = []
    for val in test_y:
        newVec = [0] * 10
        newVec[val] = 1
        test_y_arr.append(newVec)
    test_y = torch.asarray(test_y_arr)

    # normalize values between 0 and 1
    train_x = [torch.divide(sample, 255) for sample in train_x]
    test_x = [torch.divide(sample, 255) for sample in test_x]

    labels = [i for i in range(0, 10)]

    return (train_x, train_y), (test_x, test_y), labels


def run_alzheimers(print_result=False, k_folds=False, k=10):

    train_x, train_y, test_x, test_y = mnist.load_data(1.0)
    n_inputs = train_x.shape[1]
    n_outputs = train_y.shape[1]
    labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

    train_x = np.array([np.divide(sample,255) for sample in train_x])
    test_x = np.array([np.divide(sample,255) for sample in test_x])

    for entry in ModelParams.alzheimers_param_list:
        perceptron = FromDictionary(entry,"alzheimers_test_results.txt", input_size=n_inputs)

        print("Beginning training iterations.")
        if k_folds == True:
            perceptron.k_folds(train_x, train_y, labels=labels, k=k)
        else:
            perceptron.train(train_x, train_y, labels=labels)
        print("Beginning validation iterations.")
        test_result = perceptron.test(test_x, test_y, labels=labels)

        if print_result == True:
            print(f"Testing Result: {test_result}")


def run_linear_mnist(print_result=False, k_folds=False, k=10):

    (train_x, train_y), (test_x, test_y), labels = prep_mnist(subset=False)


    perceptron = NeuralNetwork(
        learning_function="constant",
        lrn_rate_modifier=0.01,
        output_filename="linearmnist.txt",
        labels_in=labels)

    input_layer = DenseLinear(output_features=32, input_features=28 * 28, bias_modifier=0.0)
    relu_one = ReluLayer(input_shape=input_layer.output_shape, previous_layer=input_layer)

    hidden_layer = DenseLinear(output_features=16, input_features=32, bias_modifier=0.00)

    relu_two = ReluLayer(input_shape=hidden_layer.output_shape, previous_layer=hidden_layer)

    dropout_layer = DropoutLayer(drop_ratio=0.5, input_shape=relu_two.output_shape, previous_layer=relu_two)

    output_layer = DenseLinear(output_features=10, input_features=16, bias_modifier=0.0)


    perceptron.add_completed_layer(input_layer)
    perceptron.add_completed_layer(relu_one)
    perceptron.add_completed_layer(hidden_layer)
    perceptron.add_completed_layer(relu_two)
    perceptron.add_completed_layer(dropout_layer)
    perceptron.add_completed_layer(output_layer)

    print("Beginning training iterations.")
    if k_folds == True:
        perceptron.k_folds(train_x, train_y, labels=labels, k=k)
    else:
        perceptron.train(train_x, train_y, labels=labels)
    print("Beginning validation iterations.")
    test_result = perceptron.test(test_x, test_y, labels=labels)

    if print_result == True:
        print(f"Testing Result: {test_result[2]}")


def run_conv_mnist(print_result=True, k_folds=False, k=1):

    (train_x, train_y), (test_x, test_y), labels = prep_mnist(subset=False)

    perceptron = NeuralNetwork(
        learning_function="constant",
        lrn_rate_modifier=0.01,
        output_filename="convmnist.txt",
        labels_in=labels)

    conv_layer = ConvolutionalLayer(
        input_shape=train_x[0].shape,
        num_filters=6,
        filter_shape=(5, 5))

    max_pool_layer = MaxPoolingLayer(input_shape=conv_layer.output_shape, previous_layer=conv_layer)
    flatten_layer = FlattenLayer(input_shape=max_pool_layer.output_shape, previous_layer=max_pool_layer)

    output_layer = DenseLinear(output_features=10, input_features=flatten_layer.output_shape, bias_modifier=0.0)

    perceptron.add_completed_layer(conv_layer)
    perceptron.add_completed_layer(max_pool_layer)
    perceptron.add_completed_layer(flatten_layer)
    perceptron.add_completed_layer(output_layer)


    print("Beginning training iterations.")
    if k_folds == True:
        perceptron.k_folds(train_x, train_y, labels=labels, k=k)
    else:
        perceptron.train(train_x, train_y, labels=labels)
    print("Beginning validation iterations.")
    test_result = perceptron.test(test_x, test_y, labels=labels)

    print("Beginning validation iterations.")

    if print_result == True:
        print(f"Testing Result: {test_result[2]}")


def torch_mnist():

    # initialize the model
    model = TorchConv()
    # model = TorchLinear()

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 1
    # percentage of training set to use as validation
    valid_size = 0.2
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    # choose the training and testing datasets
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_index, valid_index = indices[split:], indices[:split]
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)


    # specify networkLoss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()
    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # number of epochs to train the model
    n_epochs = 50
    # initialize tracker for minimum validation networkLoss
    valid_loss_min = np.Inf  # set initial "min" to infinity

    accurate = 0
    total = 0
    for epoch in range(n_epochs):
        # monitor losses
        train_loss = 0
        valid_loss = 0

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for data, label in train_loader:
            labels = [0 for i in range(10)]
            labels[label] = 1
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the networkLoss
            loss = criterion(output, label)
            # backward pass: compute gradient of the networkLoss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training networkLoss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, label in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the networkLoss
            loss = criterion(output, label)
            # update running validation networkLoss
            valid_loss = loss.item() * data.size(0)

            if epoch == n_epochs - 1:
                for index in range(len(output)):
                    total += 1
                    result = np.argmax(output[index].detach().numpy())
                    if result == label[index]:
                        accurate += 1

        # print training/validation statistics
        # calculate average networkLoss over an epoch
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            valid_loss
        ))

        # save model if validation networkLoss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation networkLoss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss


    # Final validation
    accurate = 0
    total = 0
    model.load_state_dict(torch.load('model.pt'))
    model.eval()  # prep model for evaluation
    for data, label in valid_loader:
        output = model(data)

        for index in range(len(output)):
            total += 1
            result = np.argmax(output[index].detach().numpy())
            if result == label[index]:
                accurate += 1

    print(f"Accuracy: {accurate/total}")


def torch_recurrent_imdb():

    # get the full datasets
    x, y = load_imdb(shuffle=True)


    # Process the data
    # One-hot encode the labels
    # Get our complete vocabulary
    # Finally, remove the most common words
    data_x = list()
    data_y = list()
    vocab = set()
    for line, label in zip(x, y):
        tokens = tokenize(line=line, tolower=True, alphanumeric=True)
        data_x.append(tokens)
        data_y.append(one_hot_encode(label=label, num_labels=2))
        vocab.update(tokens)

    common = get_k_most_common(data=data_x, k=30)
    data_x = remove_elements(data=data_x, elements=common)

    # Get our word embeddings
    embed_size = 50
    words, embeddings, word2index = load_glove("C:\\Users\\evanh\\Documents\\GLOVE Embeddings", embed_size)

    # # Fun with embedding math
    # one = embeddings["sad".encode()]
    # two = embeddings["robot".encode()]
    # three = one + two
    # nearest = find_k_nearest_embeddings(embeddings, three, k=20)

    # Split into a training and test set
    train_x, train_y, test_x, test_y = split(data_x, data_y, 0.2)

    # Split some of our training data off into a validation set
    train_x, train_y, valid_x, valid_y = split(train_x, train_y, 0.1)

    # Create our embeddings matrix
    vocab_len = len(vocab)
    embed_matrix = np.zeros((len(words), embed_size))

    for i in range(len(words)):
        word = words[i]
        embed_matrix[i] = embeddings[word]
    for i, word in enumerate(vocab):
        word = word.encode()
        if word not in embeddings:
            # Generate embeddings for new words
            new_embed = np.random.normal(scale=0.6, size=(embed_size, ))
            embed_matrix[i] = new_embed
            embeddings[word] = new_embed
            words.append(word)
            word2index[word] = len(words) - 1


    # Convert our data to embeddings
    test_x = to_embeddings_index(test_x, word2index)
    train_x = to_embeddings_index(train_x, word2index)

    # initialize the model
    model = TorchRNN(torch.from_numpy(embed_matrix))

    # specify networkLoss function (categorical cross-entropy)
    criterion = nn.BCELoss()
    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # number of epochs to train the model
    n_epochs = 50
    # initialize tracker for minimum validation networkLoss
    valid_loss_min = np.Inf  # set initial "min" to infinity

    accurate = 0
    total = 0
    for epoch in range(n_epochs):
        # monitor losses
        train_loss = 0
        valid_loss = 0

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for data, label in zip(train_x, train_y):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # Convert the data to a tensor
            data = torch.tensor(data, dtype=torch.int32)
            label = torch.tensor(label, dtype=torch.float32)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the networkLoss
            label = label[None, :]
            loss = criterion(output, label)
            # backward pass: compute gradient of the networkLoss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training networkLoss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        accurate = 0
        total = 0
        for data, label in zip(test_x, test_y):
            # Convert the data to a tensor
            data = torch.tensor(data, dtype=torch.float32)
            labels = torch.tensor(label, dtype=torch.float32)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the networkLoss
            labels = label[None, :]
            loss = criterion(output, label)
            # update running validation networkLoss
            valid_loss = loss.item() * data.size(0)

            total += 1
            result = torch.argmax(output.detach())
            if result == torch.argmax(label):
                accurate += 1

        # print training/validation statistics
        # calculate average networkLoss over an epoch
        train_loss = train_loss / len(train_x)
        valid_loss = valid_loss / len(test_x)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            valid_loss
        ))

        # save model if validation networkLoss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation networkLoss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss

        print(f"Accuracy: {accurate / total}")

    # Final validation
    accurate = 0
    total = 0
    model.load_state_dict(torch.load('model.pt'))
    model.eval()  # prep model for evaluation
    for data, label in zip(test_x, test_y):
        # Convert the data to a tensor
        data = torch.tensor(data, dtype=torch.float32)
        output = model(data)

        total += 1
        result = torch.argmax(output.detach())
        if result == label:
            accurate += 1

    print(f"Accuracy: {accurate/total}. Sample count: {total}")

if __name__ == "__main__":

    # run_linear_mnist(k_folds=True, k=10)
    # run_conv_mnist(k_folds=True, k=10)

    # torch_mnist()
    torch_recurrent_imdb()
    #
    # row_one = [1, 2, 3, 4]
    # row_two = [5, 6, 7, 8]
    # row_three = [9, 10, 11, 12]
    # row_four = [13, 14, 15, 16]
    # matrix = [row_one, row_two, row_three, row_four]
    #
    # matrix = np.asarray(matrix)
    #
    # print(matrix)
    #
    # pool = MaxPoolingLayer.back_max_pool(matrix)
    #
    # print(pool)


    # data = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
    # kernel = np.asarray([[1, 0], [0, 1]])
    #
    # matr = np.zeros((2, 2, 2))
    # matr[0][0][1] = 1
    # matr[0][1][0] = 1
    # print(matr)
    #
    # matr[1][0][0] = 1
    # matr[1][1][1] = 1
    #
    # print(matr)
    #
    # nums = np.asarray([[1, 2], [3, 4]])
    #
    # result = np.matmul(nums, matr)
    # print(result)
    # flat = result.flatten()
    # print(flat)
    #
    # print(flat.reshape(result.shape))

