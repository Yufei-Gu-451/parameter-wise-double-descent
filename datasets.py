from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

# All datasets return value are shuffled np array, col is feature size (dimension), row number is sample size.

def linear_regression_with_gaussian(sample_size, feautre_size, used_feature, noise_level, bias=0.0, y_dimension=1, random_state = 42):
    X, y = make_regression(n_samples=sample_size, n_features=feautre_size, n_informative=used_feature, bias=bias, n_targets=y_dimension, noise=noise_level, random_state=random_state)
    y = np.reshape(y, (-1, 1))
    return X, y

if __name__ == "__main__":
    X, y = linear_regression_with_gaussian(10, 5, 2, 0)
    print(X.shape)
    print(y.shape)
