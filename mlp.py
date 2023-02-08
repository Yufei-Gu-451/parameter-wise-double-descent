import pathlib

import matplotlib.pyplot
import numpy
from sklearn import model_selection, datasets
import typing
import torch


class MLP:
    def __init__(self,
                 device: torch.device,
                 n_samples: int = 1000,
                 n_features: int = 1,
                 noise: int = 10,
                 random_state: int = 42,
                 test_size: float = 0.9,
                 neurons: int = 25000,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5) -> None:
        self.__random_state: typing.Optional[int] = random_state
        self.__device: torch.device = device
        self.__n_features: int = n_features
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
        self.__y_train = torch.reshape(self.__y_train, (self.__y_train.shape + (1,)))
        self.__y_test = torch.reshape(self.__y_test, (self.__y_test.shape + (1,)))
        # Initialise Neural Network
        self.__neural_network: torch.nn.Sequential = self._build_neural_network(neurons=neurons)
        # Initialise optimiser
        self.__loss_function: torch.nn.modules.loss._Loss = self._get_loss_function()
        self.__optimizer: torch.optim.Optimizer = self._get_optimiser(lr=lr, weight_decay=weight_decay)
        # Initialise epoch counter
        self.__epochs: int = 0

    def _generate_dataset(self, n_samples: int, n_features: int, noise: int) \
            -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        return datasets.make_regression(n_samples=n_samples,
                                        n_features=n_features,
                                        noise=noise,
                                        random_state=self.__random_state)

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

    def _build_neural_network(self, neurons: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(torch.nn.Linear(in_features=self.__n_features, out_features=neurons),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(in_features=neurons, out_features=1),
                                   ).to(self.__device)

    def _get_loss_function(self) -> torch.nn.modules.loss._Loss:
        return torch.nn.MSELoss()

    def _get_optimiser(self, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        return torch.optim.Adamax(self.__neural_network.parameters(), lr=lr, weight_decay=weight_decay)

    def train_neural_network(self, epochs: int = 1000) -> None:
        self.__neural_network.train()
        for epoch in range(epochs):
            loss: torch.Tensor = self.__loss_function(self.__y_train, self.__neural_network(self.__x_train))
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
            self.__epochs += 1
        self.__neural_network.eval()

    @property
    def train_loss(self) -> float:
        with torch.no_grad():
            return self.__loss_function(self.__y_train,
                                        self.__neural_network(self.__x_train)).item() / self.__y_train.size(dim=0)

    @property
    def test_loss(self) -> float:
        with torch.no_grad():
            return self.__loss_function(self.__y_test,
                                        self.__neural_network(self.__x_test)).item() / self.__y_test.size(dim=0)

    def evaluate(self, values: numpy.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.__neural_network(torch.from_numpy(values).to(self.__device).to(torch.float32))

    @property
    def epochs(self) -> int:
        return self.__epochs

    def plot_data(self, path: pathlib.Path, title: str) -> None:
        matplotlib.pyplot.title(title)
        matplotlib.pyplot.xlabel("X")
        matplotlib.pyplot.ylabel("Y")
        matplotlib.pyplot.scatter(self.__x[:, 0], self.evaluate(self.__x).cpu().numpy(), c='r', label='Neural Network')
        matplotlib.pyplot.scatter(self.__x_test.cpu().numpy()[:, 0], self.__y_test.cpu().numpy(), c='g', label='Test')
        matplotlib.pyplot.scatter(self.__x_train.cpu().numpy()[:, 0], self.__y_train.cpu().numpy(),
                                  c='b', label='Train')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.savefig(path)
        matplotlib.pyplot.close()


if __name__ == '__main__':
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path: pathlib.Path = pathlib.Path('./figures')
    image_extension: str = '.png'
    if file_path.exists() and file_path.is_dir():
        for file in file_path.glob(f'*{image_extension}'):
            file.unlink()
        file_path.rmdir()
    file_path.mkdir()
    train_losses: typing.List[float] = []
    test_losses: typing.List[float] = []
    values: typing.List[int] = [i for i in range(10, 10000, 100)]
    for value in values:
        print(value)
        mlp = MLP(device=device, n_samples=value)
        mlp.train_neural_network()
        train_loss: float = mlp.train_loss
        test_loss: float = mlp.test_loss
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        mlp.plot_data(file_path / f'{value}{image_extension}',
                      f'Neurons={round(value, 2)}, Train Loss={round(train_loss, 2)}, Test Loss={round(test_loss, 2)}')
    matplotlib.pyplot.title('Losses')
    matplotlib.pyplot.xlabel("Value")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.plot(values, train_losses, c='r', label='Train')
    matplotlib.pyplot.plot(values, test_losses, c='g', label='Test')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(file_path / f'loss{image_extension}')
    matplotlib.pyplot.close()
