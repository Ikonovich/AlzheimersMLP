import numpy







import numpy as np

# shuffles the provided data
def shuffle(x_array, y_array):
    inputs = np.asarray(list(zip(x_array, y_array)))
    np.random.shuffle(inputs)
    x_array, y_array = np.asarray(list(zip(*inputs)))

    return x_array, y_array

# splits the provided data and returns it in a list
# Removes any remainder to ensure an even split
def split(x_array, y_array, numsplits):

    if x_array.shape != y_array.shape:
        raise Exception("Shape of x_array and y_array must be the same.")
    if len(x_array) != len(y_array):
        raise Exception("length of x_array and y_array must be equal.")

    removeCount = len(x_array) % numsplits
    if removeCount > 0:
        x_array = np.delete(x_array, [i for i in range(removeCount)], axis=1)
        y_array = np.delete(x_array, [i for i in range(removeCount)], axis=1)

    x_array = np.split(x_array, numsplits, axis=0)
    y_array = np.split(y_array, numsplits, axis=0)

    return x_array, y_array


# Merges two identically sized lists of np arrays for performing k-folds
def kfold_merge(x_array, y_array, drop_index=-1):
    start = 0
    if drop_index == start:
        x_out = x_array[0]
        y_out = y_array[0]
    else:
        start = 2
        x_out = x_array[1]
        y_out = y_array[1]

    for i in range(start, len(x_array)):

        if i != drop_index:
            x_out = np.concatenate((x_out, x_array[i]), axis=0)
            y_out = np.concatenate((y_out, y_array[i]), axis=0)

    return x_out, y_out
