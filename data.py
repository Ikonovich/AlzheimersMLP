import random
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def handle_folder(folder_name,k):
    list_dir = os.listdir(folder_name)
    k = min(k,len(list_dir))
    files = random.sample(list_dir,k)
    
    output = []
    for filename in files:
        img = os.path.join(folder_name,filename)
        with Image.open(img) as fp:
            (left,upper,right,lower) = (0,0,190,190)
            img_crop = fp.crop((left,upper,right,lower))
            img_crop = img_crop.convert("L")
            array = np.asarray(img_crop) 
            array = array.flatten()
            output.append(array)

    return output

def randomize(X,y):
    assert len(X) == len(y)
    rand_indices = np.arange(len(X))
    np.random.shuffle(rand_indices)
    X = X[rand_indices]
    y = y[rand_indices]

    return X,y

def load_images_to_arrays(dir,k):
    moderate_dir = os.path.join(dir,'ModerateDemented')
    mild_dir = os.path.join(dir,'MildDemented')
    very_mild_dir = os.path.join(dir,'VeryMildDemented')
    non_dir = os.path.join(dir,'NonDemented')

    X = []
    y = []
    
    # Categorical array structure: 
    ## ModerateDemented = [0,0,0,1]
    ## MildDemented = [0,0,1,0]
    ## VeryMildDemented = [0,1,0,0]
    ## NonDemented = [1,0,0,0]

    print("Loading moderate...")
    moderate = handle_folder(moderate_dir,k)
    X += moderate
    Y = [[0,0,0,1]] * len(moderate)
    y += Y

    print("Loading mild...")
    mild = handle_folder(mild_dir,k)
    X += mild
    Y = [[0,0,1,0]] * len(mild)
    y += Y

    print("Loading very mild...")
    very_mild = handle_folder(very_mild_dir,k)
    X += very_mild
    Y = [[0,1,0,0]] * len(very_mild)
    y += Y

    print("Loading non...")
    non = handle_folder(non_dir,k)
    X += non
    Y = [[1,0,0,0]] * len(non)
    y += Y

    X = np.array(X)
    y = np.array(y)
    
    X, y = randomize(X, y)
    return X, y

def load_training_data():
    parent_dir = os.getcwd()
    train_dir = os.path.join(parent_dir,'Images\data\\train')
    trainX, trainY = load_images_to_arrays(train_dir,6000)
    return trainX, trainY

def load_testing_data():
    parent_dir = os.getcwd()
    test_dir = os.path.join(parent_dir,'Images\data\\val')
    testX, testY = load_images_to_arrays(test_dir,896)
    return testX, testY

def load_data():
    print("Loading training data...")
    trainX, trainY = load_training_data()
    print("Loading testing data...")
    testX, testY = load_testing_data()

    # normalize values between 0 and 1
    trainX = np.array([np.divide(sample,255) for sample in trainX])
    testX = np.array([np.divide(sample,255) for sample in testX])

    return trainX, trainY, testX, testY

if __name__ == "__main__":
    trainX, trainY, testX, testY = load_data()

    model = MLPClassifier(solver="lbfgs", alpha = 1e-7,hidden_layer_sizes=(16,1),random_state=1)

    model.fit(trainX,trainY)
    print(model.score(testX,testY))
