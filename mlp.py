from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


class MLP:
    def __init__(self, n_samples: int, n_features: int, noise):
        self.n_samples: int = n_samples
        self.n_features: int = n_features
        self.noise: int = noise

    def generate_dataset(self, n_samples, n_features, noise):
        return make_regression(n_samples=n_samples, n_features=n_features, noise=noise)

    def split_dataset(self):
        return train_test_split(self.data)

    def build_neural_network(self):
        raise NotImplementedError

    def train_neural_network(self):
        raise NotImplementedError
