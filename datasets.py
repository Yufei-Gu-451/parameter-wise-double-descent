import numpy as np
from sklearn import datasets as skdataset
import keras.datasets as kerasDatasets
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from PIL import Image


# All datasets return value are shuffled np array, col is feature size (dimension), row number is sample size.

def linear_regression_with_gaussian(sample_size=100, feautre_size=5, used_feature=3, noise_level=0.0, bias=0.0, y_dimension=1, random_state = None):
    X, y = skdataset.make_regression(n_samples=sample_size, n_features=feautre_size, n_informative=used_feature, bias=bias, n_targets=y_dimension, noise=noise_level, random_state=random_state)
    y = np.reshape(y, (-1, y_dimension))
    return X, y

# real world datasets, can be added with any type noise later
def load_CIFAR10(indices):
    (x_train, y_train), (x_test, y_test) = kerasDatasets.cifar10.load_data()
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    data = []
    label = []
    for i in range(60000):
        if y[i] in indices:
            data.append(np.array(Image.fromarray(X[i]).convert("L")).reshape(1024))
            label.append(y[i])
    
    return np.asarray(data), np.asarray(label)

def load_CIFAR100(indices):
    (x_train, y_train), (x_test, y_test) = kerasDatasets.cifar100.load_data()
    X = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    data = []
    label = []
    for i in range(60000):
        if y[i] in indices:
            data.append(np.array(Image.fromarray(X[i]).convert("L")).reshape(1024))
            label.append(y[i])
    
    return np.asarray(data), np.asarray(label)

def load_MNIST():
    (x_train, y_train), (x_test, y_test) = kerasDatasets.mnist.load_data()
    X = np.concatenate((x_train, x_test), axis=0)
    X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
    y = np.concatenate((y_train, y_test), axis=0)
    return X, y

def load_Fashion_MNIST():
    (x_train, y_train), (x_test, y_test) = kerasDatasets.fashion_mnist.load_data()
    X = np.concatenate((x_train, x_test), axis=0)
    X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
    y = np.concatenate((y_train, y_test), axis=0)
    return X, y

def load_boston():
    X = skdataset.load_boston()
    return X.data, X.target

def load_digits():
    X = skdataset.load_digits()
    return X.data, X.target
    

if __name__ == "__main__":
    # X, y = load_digits()
    # print(X.shape)
    # print(y.shape)
    
    # X, y = load_boston()
    # print(X.shape)
    # print(y.shape)
    
    # X, y = linear_regression_with_gaussian(sample_size=200, feautre_size=50, used_feature=50, noise_level=10, bias=10, y_dimension=5)
    # print(X.shape)
    # print(y.shape)
    
    indices = [0, 1]
    
    X, y = load_CIFAR100(indices)
    print(X.shape)
    print(y.shape)
    
    # X, y = load_CIFAR100()
    # print(X.shape)
    # print(y.shape)
    
    # X, y = load_MNIST()
    # print(X.shape)
    # print(y.shape)
