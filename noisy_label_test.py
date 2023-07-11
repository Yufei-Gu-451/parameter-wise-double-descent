import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from prefetch_generator import BackgroundGenerator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

import mnist_dd_exp

hidden_units = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100,
                120, 150, 200, 400, 600, 800, 1000]

sample_size = 4000
n_epochs = 2000
batch_size = 64
label_noise_ratio = 0.2

directory = "assets/MNIST/sub-set-3d/epoch=%d-noise-%d-model" % (n_epochs, label_noise_ratio * 100)

dictionary_path = os.path.join(directory, "dictionary.csv")
checkpoint_path = os.path.join(directory, "ckpt")
plots_path = os.path.join(directory, "plots")


def load_model(hidden_unit):
    model = mnist_dd_exp.Simple_FC(hidden_unit)

    checkpoint = torch.load(os.path.join(checkpoint_path, 'Simple_FC_%d.pth' % hidden_unit))
    model.load_state_dict(checkpoint['net'])

    return model

def add_gaussian_noise(image, mean, std_dev):
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = image + noise
    return noisy_image.astype(np.float32)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    org_train_dataset = torch.load('data/MNIST/subset.pth')
    noisy_train_dataset = torch.load('data/MNIST/subset-noise-20%.pth')

    assert(len(org_train_dataset) == len(noisy_train_dataset))

    test_dataset_list = []

    gaussian_mean = 0
    gaussian_std_dev = 0

    for i in range(len(org_train_dataset)):
        # assert (np.array(org_train_dataset[i][0]).all() == np.array(noisy_train_dataset[i][0]).all())

        if org_train_dataset[i][1] != noisy_train_dataset[i][1]:
            data = org_train_dataset[i][0].numpy().astype(np.float32)
            pertub_data = add_gaussian_noise(data, gaussian_mean, gaussian_std_dev)
            pertub_data = torch.tensor(pertub_data)

            test_dataset_list.append((pertub_data, org_train_dataset[i][1]))

    test_dataset = mnist_dd_exp.ListDataset(test_dataset_list)

    test_dataloader = mnist_dd_exp.DataLoaderX(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    accuracy_list = []

    for n in hidden_units:
        model = load_model(n)
        model.eval()

        correct = 0

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(test_dataloader):
                labels = torch.nn.functional.one_hot(labels, num_classes=10).float()

                outputs = model(inputs)

                _, predicted = outputs.max(1)
                correct += predicted.eq(labels.argmax(1)).sum().item()

        print(n, correct)

        accuracy_list.append(correct / 720)


    plt.figure(figsize=(10, 7))
    ax = plt.axes()
    scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)
    ax.set_xscale('function', functions=scale_function)
    plt.plot(hidden_units, accuracy_list, marker='o', label='Data Noise Test Error')
    plt.xticks([1, 5, 15, 40, 100, 250, 500, 1000])
    plt.legend()
    plt.xlabel('Number of hidden units (H)')
    plt.ylabel('Accuracy on Pertubated Noisy Data Points')
    plt.title('Gaussian standard deviation = %lf' % gaussian_std_dev)
    plt.savefig(os.path.join(plots_path, 'Pertubated Data Test - %lf.png' % gaussian_std_dev))
