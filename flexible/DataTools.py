# Contains various tools for manipulating data
import math
import os
import pickle
import random
import re
from os.path import exists
from pathlib import Path
from typing import Collection, Sequence, Any, Callable

# import bcolz
import numpy as np
import torchvision
from scipy import spatial
from torch.utils.data import DataLoader, Dataset

# Pattern for removing non-alphanumeric characters

pattern = re.compile(r'[^a-zA-Z0-9\s]')


def to_embeddings_index(data: list[list[str]], word2index: dict[bytes, int]) -> list[list[int]]:
    result = list()
    for i in range(len(data)):
        item = data[i]
        index = [word2index[word.encode()] for word in item]
        result.append(index)
    return result


def get_k_most_common(data: list[list[Any]], k: int) -> set[Any]:
    # Returns the most common elements in the provided dataset
    # Store a dictionary of seen words
    counted = dict()
    # Count occurrences of each word
    for entry in data:
        for element in entry:
            if element in counted:
                counted[element] += 1
            else:
                counted[element] = 1

    # Sort the elements by number of occurrences
    sorted_elements = sorted(counted.items(), key=lambda x: x[1], reverse=True)
    # Remove everything after the top k
    sorted_elements = sorted_elements[:k]
    # Convert the remaining values into a set and return
    elements = set()
    for item in sorted_elements:
        elements.add(item[0])
    return elements


def remove_elements(data: list[list[Any]], elements: set[Any]):
    # Removes the provided elements from the provided dataset
    # Store a dictionary of seen words
    new_data = list()
    for line in data:
        new_line = [x for x in line if x not in elements]
        new_data.append(new_line)
    return new_data


# def load_glove(folder_path: str, size: int = 50):
#     # Loads glove
#     # Size options are 50, 100, 200, and 300.
#     # All other inputs will result in size 50 being returned.
#     if size == 100:
#         filepath = f'{folder_path}/glove.6B.100d'
#     elif size == 200:
#         filepath = f'{folder_path}/glove.6B.200d'
#     elif size == 300:
#         filepath = f'{folder_path}/glove.6B.300d'
#     else:
#         filepath = f'{folder_path}/glove.6B.50d'
#
#     vec_path = filepath + '_vectors.dat'
#     words_path = filepath + '_words.pkl'
#     word2index_path = filepath + '_indices.pkl'
#
#     if exists(vec_path) and exists(words_path) and exists(word2index_path):
#         vectors = bcolz.open(vec_path)[:]
#         words = pickle.load(open(words_path, 'rb'))
#         word2index = pickle.load(open(word2index_path, 'rb'))
#
#     else:
#         vectors = bcolz.carray(np.zeros(1), rootdir=filepath, mode='w')
#         words = []
#         index = 0
#         word2index = {}
#         with open(filepath + '.txt', 'rb') as f:
#             for string in f:
#                 line = string.split()
#                 word = line[0]
#                 words.append(word)
#                 word2index[word] = index
#                 index += 1
#                 vect = np.array(line[1:]).astype(np.float)
#                 vectors.append(vect)
#
#         vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=vec_path, mode='w')
#         vectors.flush()
#         pickle.dump(words, open(words_path, 'wb'))
#         pickle.dump(word2index, open(word2index_path, 'wb'))
#
#     embeddings = {word: vectors[word2index[word]] for word in words}
#     return words, embeddings, word2index


def find_k_nearest_embeddings(embeddings: dict[str, str], word_vector: str, k=5):
    # A function used to return the k-closest embeddings to a particular word
    results = sorted(embeddings.keys(), key=lambda word: spatial.distance.euclidean(embeddings[word], word_vector))
    return results[:k]


def tokenize(line: str, tolower: bool = False, alphanumeric: bool = False) -> list[str]:
    # Splits a string into individual tokens, optionally converting them to lowercase and removing
    # non-alphanumeric values

    # Remove linebreaks
    line.replace("<br>", " ")

    if tolower:
        line = line.lower()

    if alphanumeric:
        line = re.sub(pattern, ' ', line)
    return line.split()


def one_hot_encode(label: int, num_labels: int):
    label_vec = [0 for i in range(num_labels)]
    label_vec[label] = 1
    return label_vec


def split(data: Sequence, labels: Sequence, fraction: float, shuffle=False):
    # Splits a provided collection of samples and labels into two.
    # The second split will have the fraction of samples indicated by the fraction parameter.
    # Ex: 10 samples with a 0.2 fraction will result in a split of (8, 2)
    if len(data) != len(labels):
        raise IndexError

    if shuffle:
        data, labels = do_shuffle(data, labels)

    right_size = math.floor(len(data) * fraction)
    left_data, right_data = data[:right_size], data[right_size:]
    left_labels, right_labels = labels[:right_size], data[right_size:]

    return left_data, left_labels, right_data, right_labels


def do_shuffle(data, labels):
    # Shuffles a set of data and labels, while maintaining pairwise indices
    # Example: After shuffling, data[1] and labels[1] will (probably) move places, but will
    # always remain next to each other.
    if len(data) != len(labels):
        raise IndexError

    indices = [i for i in range(len(data))]
    random.shuffle(indices)
    new_data = list()
    new_labels = list()
    for index in indices:
        new_data.append(data[index])
        new_labels.append(labels[index])

    return new_data, new_labels


def load_imdb(shuffle=True) -> tuple[list[str], list[int]]:
    x = list()
    y = list()
    train_neg = load_text(r"C:\Users\evanh\Documents\Datasets\IMDB\train\neg")
    train_pos = load_text(r"C:\Users\evanh\Documents\Datasets\IMDB\train\pos")
    test_neg = load_text(r"C:\Users\evanh\Documents\Datasets\IMDB\test\neg")
    test_pos = load_text(r"C:\Users\evanh\Documents\Datasets\IMDB\test\pos")

    train_neg.extend(test_neg)
    train_pos.extend(test_pos)
    train_neg_y = [0 for i in range(len(train_neg))]
    train_pos_y = [1 for i in range(len(train_pos))]

    x.extend(train_neg)
    y.extend(train_neg_y)
    x.extend(train_pos)
    y.extend(train_pos_y)

    if shuffle:
        x, y = do_shuffle(x, y)

    return x, y


def load_mnist(train: bool = True, transform: Callable = None) -> Dataset:
    dataset = torchvision.datasets.MNIST(root="data/MNIST",
                                         train=train,
                                         download=True,
                                         transform=transform)
    return dataset


def load_text(folder_path: str) -> list[str]:
    # Given a folder path, recursively loads every .txt file in that folder into a separate string
    # and returns it as a list
    data = list()
    for entry in os.listdir(folder_path):
        ending = entry[-4:]
        path = os.path.join(folder_path, entry)
        if os.path.isdir(path):
            data.extend(load_text(entry))
        elif os.path.isfile(path) and ending == '.txt':
            txt = Path(path).read_text(encoding="utf8")
            txt = txt.replace('\n', '')
            data.append(txt)

    return data
