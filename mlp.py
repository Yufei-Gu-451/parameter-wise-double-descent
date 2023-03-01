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
                 n_samples: int = int(1e4),
                 n_features: int = 1,
                 noise: int = 10,
                 random_state: int = 6839,
                 test_size: float = 0.5,
                 neurons: int = int(1e5),
                 extra_layers: int = 0,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-3,
                 classify: bool = True) -> None:
        self.__classify: bool = classify
        self.__classifications: typing.Optional[numpy.ndarray] = None
        self.__random_state: typing.Optional[int] = random_state
        self.__device: torch.device = device
        self.__n_features: int = n_features
        self.__n_outputs: int = 1
        # Generate dataset
        self.__x: numpy.ndarray
        self.__y: numpy.ndarray
        self.__x, self.__y = self._generate_dataset(n_samples=n_samples, n_features=n_features, noise=noise)
        # Split dataset
        self.__x_train: torch.Tensor
        self.__x_test: torch.Tensor
        self.__y_train: torch.Tensor
        self.__y_test: torch.Tensor
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.__split_dataset(test_size=test_size)
        # Initialise Neural Network
        self.__neural_network: torch.nn.Sequential = self._build_neural_network(neurons=neurons,
                                                                                extra_layers=extra_layers)
        # Initialise optimiser
        self.__loss_function: torch.nn.modules.loss._Loss = self._get_loss_function()
        self.__optimizer: torch.optim.Optimizer = self._get_optimiser(lr=lr, weight_decay=weight_decay)
        # Initialise epoch counter
        self.__epochs: int = 0

    def _generate_dataset(self, n_samples: int, n_features: int, noise: int) \
            -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        def generate_classifier_dataset():
            data = torchvision.datasets.MNIST(root='./MNIST', download=True)
            xs = numpy.array(
                [torch.reshape(torchvision.transforms.functional.pil_to_tensor(data[item][0]), (-1,)).numpy() for item
                 in range(n_samples)])
            xs = xs / numpy.linalg.norm(xs)
            base_ys = numpy.array([data[item][1] for item in range(n_samples)])
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

    def _build_neural_network(self, neurons: int, extra_layers: int) -> torch.nn.Sequential:
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

    def _get_loss_function(self) -> torch.nn.modules.loss._Loss:
        return torch.nn.MSELoss()

    def _get_optimiser(self, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        return torch.optim.Adamax(self.__neural_network.parameters())

    def train_neural_network(self, epochs: int = 100000) -> None:
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

    def evaluate(self, values: numpy.ndarray) -> torch.Tensor:
        self.__neural_network.eval()
        with torch.no_grad():
            return self.__neural_network(torch.from_numpy(values).to(self.__device).to(torch.float32))

    @property
    def epochs(self) -> int:
        return self.__epochs

    def plot_data(self, path: pathlib.Path, title: str, image_extension: str, image_name: str) -> None:
        def classify(y: numpy.ndarray):
            return self.__classifications[y.argmax()]

        test_results: numpy.ndarray = self.evaluate(self.__x_test.cpu().numpy()).cpu().numpy()
        train_results: numpy.ndarray = self.evaluate(self.__x_train.cpu().numpy()).cpu().numpy()
        y_train: numpy.ndarray = self.__y_train.cpu().numpy()
        y_test: numpy.ndarray = self.__y_test.cpu().numpy()
        assert (test_results.shape == y_test.shape)
        assert (train_results.shape == y_train.shape)
        matplotlib.pyplot.title(title)
        matplotlib.pyplot.xlabel("X")
        matplotlib.pyplot.ylabel("Y")
        if self.__classify:
            test_matches = [classify(test_results[row]) == classify(y_test[row]) for row in
                            range(test_results.shape[0])]
            matplotlib.pyplot.bar(["Correct", "Incorrect"], [test_matches.count(True), test_matches.count(False)])
            matplotlib.pyplot.savefig(path / f'{image_name}_test{image_extension}')
            matplotlib.pyplot.close()
            train_matches = [classify(train_results[row]) == classify(y_train[row]) for row in
                             range(train_results.shape[0])]
            matplotlib.pyplot.bar(["Correct", "Incorrect"], [train_matches.count(True), train_matches.count(False)])
            matplotlib.pyplot.savefig(path / f'{image_name}_train{image_extension}')
            matplotlib.pyplot.close()
        else:
            # Should probably do PCA on x, y here rather than indexing 0
            matplotlib.pyplot.scatter(self.__x_test.cpu().numpy()[:, 0], y_test[:, 0], c='g', label='Test')
            matplotlib.pyplot.scatter(self.__x_test.cpu().numpy()[:, 0], test_results[:, 0], c='r',
                                      label='Neural Network')
            matplotlib.pyplot.scatter(self.__x_train.cpu().numpy()[:, 0], y_train[:, 0], c='b', label='Train')
            matplotlib.pyplot.legend()
            matplotlib.pyplot.savefig(path / f'{image_name}_test{image_extension}')
            matplotlib.pyplot.close()


class DescentType(Enum):
    MODEL_WISE = 0
    SAMPLE_WISE = 1
    EPOCH_WISE = 2


def descend(values: typing.List[int], value_changed: str, descent_type: DescentType):
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path: pathlib.Path = pathlib.Path('figures')
    image_extension: str = '.svg'
    if file_path.exists() and file_path.is_dir():
        for file in file_path.glob(f'*{image_extension}'):
            file.unlink()
        file_path.rmdir()
    file_path.mkdir()
    train_losses: typing.List[float] = []
    test_losses: typing.List[float] = []
    mlp: typing.Optional[MLP] = None
    for value in values:
        print(value)
        t = time.time()
        # Create/recreate neural network
        if descent_type == DescentType.MODEL_WISE:
            mlp = MLP(device=device, neurons=value)
        elif descent_type == DescentType.SAMPLE_WISE:
            mlp = MLP(device=device, n_samples=value)
        else:
            mlp = MLP(device=device) if mlp is None else mlp
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
        mlp.plot_data(file_path,
                      f'{value_changed}={round(value, 2)}, Train Loss={round(train_loss, 2)}, Test Loss={round(test_loss, 2)}',
                      image_extension=image_extension, image_name=str(value))
        matplotlib.pyplot.title(value_changed)
        matplotlib.pyplot.xlabel("Value")
        matplotlib.pyplot.ylabel("Loss")
        matplotlib.pyplot.plot(values[:len(train_losses)], train_losses, c='r', label='Train')
        matplotlib.pyplot.plot(values[:len(test_losses)], test_losses, c='g', label='Test')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.ylim(bottom=0)
        matplotlib.pyplot.savefig(file_path / f'loss{image_extension}')
        matplotlib.pyplot.close()
        print(f"{time.time() - t} seconds")


def model_wise_descend(values: typing.List[int]):
    descend(values=values, value_changed='Neurons', descent_type=DescentType.MODEL_WISE)


def sample_wise_descend(values: typing.List[int]):
    descend(values=values, value_changed='Samples', descent_type=DescentType.SAMPLE_WISE)


def epoch_wise_descend(count: int):
    descend(values=[i for i in range(1, count)], value_changed='Epochs', descent_type=DescentType.EPOCH_WISE)


if __name__ == '__main__':
    epoch_wise_descend(count=1000)
