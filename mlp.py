import pathlib
import random
import time
from enum import Enum

import matplotlib.pyplot
import numpy
from sklearn import model_selection, datasets
import typing
import torch
import torchvision
from torchvision.transforms import functional


class MLP:
    def __init__(self,
                 device: torch.device,
                 classify: bool,
                 n_train: int = int(5e3),
                 n_test: int = int(5e3),
                 neurons: int = int(1e5),
                 extra_layers: int = 0,
                 n_features: int = 1,
                 noise: int = 10,
                 random_state: int = 6839) -> None:
        self.__classify: bool = classify
        self.__classifications: typing.Optional[numpy.ndarray] = None
        self.__random_state: typing.Optional[int] = random_state
        self.__device: torch.device = device
        self.__n_features: int = n_features
        self.__n_outputs: int = 1
        # Generate dataset
        self.__x, self.__y = self.__generate_dataset(n_samples=n_train + n_test, n_features=n_features, noise=noise)
        # Split dataset
        self.__x_train, self.__x_test, \
            self.__y_train, self.__y_test = self.__split_dataset(test_size=n_test / (n_train + n_test))
        # Initialise Neural Network
        self.__neural_network: torch.nn.Sequential = self.__build_neural_network(neurons=neurons,
                                                                                 extra_layers=extra_layers)
        # Initialise optimiser
        self.__loss_function: torch.nn.modules.loss._Loss = self.__get_loss_function()
        self.__optimizer: torch.optim.Optimizer = self.__get_optimiser()
        # Initialise epoch counter
        self.__epochs: int = 0

    def __generate_dataset(self, n_samples: int, n_features: int, noise: int) \
            -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        def generate_classifier_dataset():
            data = torchvision.datasets.MNIST(root='./MNIST', download=True)
            xs = numpy.array(
                [torch.reshape(torchvision.transforms.functional.pil_to_tensor(data[item][0]), (-1,)).numpy()
                 for item in range(n_samples)])
            xs = xs / numpy.linalg.norm(xs)
            base_ys = numpy.array([data[item][1] for item in range(xs.shape[0])])
            self.__classifications = numpy.unique(base_ys)
            ys = numpy.array(
                [[1 if classification == y else 0 for classification in self.__classifications] for y in base_ys])
            self.__n_features = xs.shape[1]
            self.__n_outputs = self.__classifications.shape[0]
            return xs, ys

        def generate_regression_dataset():
            random.seed(self.__random_state)
            xs: numpy.ndarray
            ys: numpy.ndarray
            xs, ys = datasets.make_regression(n_samples=n_samples,
                                              n_features=n_features,
                                              noise=noise,
                                              random_state=self.__random_state)
            ys = numpy.reshape(ys, ys.shape + (1,))
            return xs, ys

        if self.__classify:
            return generate_classifier_dataset()
        else:
            return generate_regression_dataset()

    def __split_dataset(self, test_size: float) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_train: numpy.ndarray
        x_test: numpy.ndarray
        y_train: numpy.ndarray
        y_test: numpy.ndarray
        x_train, x_test, y_train, y_test = model_selection.train_test_split(self.__x, self.__y,
                                                                            test_size=test_size,
                                                                            random_state=self.__random_state)
        return torch.from_numpy(x_train).to(torch.float32).to(self.__device), \
            torch.from_numpy(x_test).to(torch.float32).to(self.__device), \
            torch.from_numpy(y_train).to(torch.float32).to(self.__device), \
            torch.from_numpy(y_test).to(torch.float32).to(self.__device)

    def __build_neural_network(self, neurons: int, extra_layers: int) -> torch.nn.Sequential:
        neural_network: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.__n_features, out_features=neurons),
            torch.nn.ReLU()
        )
        for layer in range(extra_layers):
            neural_network.append(torch.nn.Linear(in_features=neurons, out_features=neurons))
            neural_network.append(torch.nn.ReLU())
        neural_network.append(torch.nn.Linear(in_features=neurons, out_features=self.__n_outputs))
        if self.__classify:
            neural_network.append(torch.nn.Softmax(dim=1))
        return neural_network.to(self.__device)

    @staticmethod
    def __get_loss_function() -> torch.nn.modules.loss._Loss:
        return torch.nn.MSELoss()

    def __get_optimiser(self) -> torch.optim.Optimizer:
        return torch.optim.Adamax(self.__neural_network.parameters())

    def train_neural_network(self, epochs: int = 1000) -> None:
        self.__neural_network.train()
        for epoch in range(epochs):
            self.__optimizer.zero_grad()
            loss: torch.Tensor = self.__loss_function(self.__y_train, self.__neural_network(self.__x_train))
            loss.backward()
            self.__optimizer.step()
            self.__epochs += 1
        self.__neural_network.eval()

    @property
    def train_loss(self) -> float:
        with torch.no_grad():
            return self.__loss_function(self.__y_train,
                                        self.__neural_network(self.__x_train)).item()  # / self.__y_train.size(dim=0)

    @property
    def test_loss(self) -> float:
        with torch.no_grad():
            return self.__loss_function(self.__y_test,
                                        self.__neural_network(self.__x_test)).item()  # / self.__y_test.size(dim=0)

    def __evaluate(self, values: numpy.ndarray) -> torch.Tensor:
        self.__neural_network.eval()
        with torch.no_grad():
            return self.__neural_network(torch.from_numpy(values).to(self.__device).to(torch.float32))

    @property
    def epochs(self) -> int:
        return self.__epochs

    def plot_data(self, path: pathlib.Path, title: str, image_extension: str, image_name: str) -> None:
        test_results: numpy.ndarray = self.__evaluate(self.__x_test.cpu().numpy()).cpu().numpy()
        train_results: numpy.ndarray = self.__evaluate(self.__x_train.cpu().numpy()).cpu().numpy()
        y_train: numpy.ndarray = self.__y_train.cpu().numpy()
        y_test: numpy.ndarray = self.__y_test.cpu().numpy()
        assert (test_results.shape == y_test.shape)
        assert (train_results.shape == y_train.shape)
        matplotlib.pyplot.title(title)
        matplotlib.pyplot.xlabel("X")
        matplotlib.pyplot.ylabel("Y")
        # Should probably do PCA on x, y here rather than indexing 0
        matplotlib.pyplot.scatter(self.__x_test.cpu().numpy()[:, 0], y_test[:, 0], c='g', label='Test')
        matplotlib.pyplot.scatter(self.__x_test.cpu().numpy()[:, 0], test_results[:, 0], c='r',
                                  label='Neural Network')
        matplotlib.pyplot.scatter(self.__x_train.cpu().numpy()[:, 0], y_train[:, 0], c='b', label='Train')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.savefig(path / (image_name + image_extension))
        matplotlib.pyplot.close()

    def classifier_precisions(self):
        def classify(y: numpy.ndarray):
            return self.__classifications[y.argmax()]

        test_results: numpy.ndarray = self.__evaluate(self.__x_test.cpu().numpy()).cpu().numpy()
        train_results: numpy.ndarray = self.__evaluate(self.__x_train.cpu().numpy()).cpu().numpy()
        y_train: numpy.ndarray = self.__y_train.cpu().numpy()
        y_test: numpy.ndarray = self.__y_test.cpu().numpy()
        assert (test_results.shape == y_test.shape)
        assert (train_results.shape == y_train.shape)
        test_matches = [classify(test_results[row]) == classify(y_test[row]) for row in
                        range(test_results.shape[0])]
        train_matches = [classify(train_results[row]) == classify(y_train[row]) for row in
                         range(train_results.shape[0])]
        return train_matches.count(False) / len(train_matches), test_matches.count(False) / len(test_matches)


class DescentType(Enum):
    MODEL_WISE = 0
    SAMPLE_WISE = 1
    EPOCH_WISE = 2


class Descender:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path: pathlib.Path = pathlib.Path('figures')
    image_extension: str = '.svg'

    def __init__(self, classifier: bool):
        self.classify = classifier
        if self.file_path.exists() and self.file_path.is_dir():
            for file in self.file_path.glob(f'*{self.image_extension}'):
                file.unlink()
            self.file_path.rmdir()
        self.file_path.mkdir()

    def __plot_losses(self,
                      file_name: str,
                      title: str,
                      values: typing.List[int],
                      train_losses: typing.List[float],
                      test_losses: typing.List[float]):
        matplotlib.pyplot.title(title)
        matplotlib.pyplot.xlabel("Value")
        matplotlib.pyplot.ylabel("Loss")
        matplotlib.pyplot.plot(values[:len(train_losses)], train_losses, c='r', label='Train')
        matplotlib.pyplot.plot(values[:len(test_losses)], test_losses, c='g', label='Test')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.ylim(bottom=0)
        matplotlib.pyplot.savefig(self.file_path / (file_name + self.image_extension))
        matplotlib.pyplot.close()

    def __descend(self, values: typing.List[int], value_changed: str, descent_type: DescentType):
        train_losses: typing.List[float] = []
        test_losses: typing.List[float] = []
        train_precisions: typing.List[float] = []
        test_precisions: typing.List[float] = []
        mlp: typing.Optional[MLP] = None
        for value in values:
            print(value)
            t = time.time()
            # Create/recreate neural network
            if descent_type == DescentType.MODEL_WISE:
                mlp = MLP(device=self.device, classify=self.classify, neurons=value)
            elif descent_type == DescentType.SAMPLE_WISE:
                mlp = MLP(device=self.device, classify=self.classify, n_train=value)
            else:
                mlp = MLP(device=self.device, classify=self.classify) if mlp is None else mlp
            # Train
            if descent_type == DescentType.EPOCH_WISE:
                mlp.train_neural_network(epochs=1)
                print(f"Cumulative epochs: {mlp.epochs}")
            else:
                mlp.train_neural_network()
            train_loss: float = mlp.train_loss
            test_loss: float = mlp.test_loss
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Train: {train_loss}, Test: {test_loss}")
            if self.classify:
                train_precision, test_precision = mlp.classifier_precisions()
                train_precisions.append(train_precision)
                test_precisions.append(test_precision)
                self.__plot_losses(file_name="precisions",
                                   title=value_changed,
                                   values=values,
                                   train_losses=train_precisions,
                                   test_losses=test_precisions)
            else:
                mlp.plot_data(path=self.file_path,
                              title=f'{value_changed}={round(value, 2)}, Train Loss={round(train_loss, 2)}, Test Loss={round(test_loss, 2)}',
                              image_extension=self.image_extension,
                              image_name=str(value))
            self.__plot_losses(file_name="losses",
                               title=value_changed,
                               values=values,
                               train_losses=train_losses,
                               test_losses=test_losses)
            print(f"{time.time() - t} seconds")

    def model_wise_descend(self, values: typing.List[int]):
        self.__descend(values=values, value_changed='Neurons', descent_type=DescentType.MODEL_WISE)

    def sample_wise_descend(self, values: typing.List[int]):
        self.__descend(values=values, value_changed='Samples', descent_type=DescentType.SAMPLE_WISE)

    def epoch_wise_descend(self, count: int):
        self.__descend(values=[i for i in range(1, count)], value_changed='Epochs', descent_type=DescentType.EPOCH_WISE)


def main():
    classifier: Descender = Descender(classifier=True)
    # classifier.sample_wise_descend(values=[i for i in range(100, int(1e4), 100)])
    classifier.epoch_wise_descend(count=int(1e4))


if __name__ == '__main__':
    main()
