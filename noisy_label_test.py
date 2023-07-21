import sklearn.decomposition
import torch
import torchvision.datasets as datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

import dd_exp
import datasets
import models

DATASET = 'CIFAR-10'

if DATASET == 'MNIST':
    hidden_units = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100,
                    120, 150, 200, 400, 600, 800, 1000]
elif DATASET == 'CIFAR-10':
    hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]

N_EPOCHS = 100
N_SAMPLES = 4000
BATCH_SIZE = 64

TEST_GROUP = 0
TEST_NUMBERS = [1]

label_noise_ratio = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_gaussian_noise(image, mean, std_dev):
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = image + noise
    return noisy_image.astype(np.float32)

def decomposition_by_SVD(mat, k):
    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(mat)

    # Choose top k important singular values
    Uk = U[:, :k]
    Sk = np.diag(S[0:k])
    Vk = V[:k, :]

    # recover the image
    imgMat_new = Uk @ Sk @ Vk
    return Uk

def get_test_losses(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        test_losses = []

        for row in reader:
            if int(row['Epoch']) == N_EPOCHS:
                test_losses.append(float(row['Test Loss']))

        return test_losses

def get_test_accuracy(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        test_accuracy = []

        for row in reader:
            if int(row['Epoch']) == N_EPOCHS:
                test_accuracy.append(float(row['Test Accuracy']))

        return test_accuracy

def get_train_accuracy(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        train_accuracy = []

        for row in reader:
            if int(row['Epoch']) == N_EPOCHS:
                train_accuracy.append(float(row['Train Accuracy']))

        return train_accuracy

def get_parameters(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        parameters = []

        for row in reader:
            if int(row['Epoch']) == N_EPOCHS:
                parameters.append(int(row['Parameters']) // 1000)

        return parameters

def load_dataset(dataset_path):
    org_train_dataset = torch.load(os.path.join(dataset_path, 'subset-clean.pth'))
    noisy_train_dataset = torch.load(os.path.join(dataset_path, 'subset-noise-20%.pth'))

    assert (len(org_train_dataset) == len(noisy_train_dataset))

    return org_train_dataset, noisy_train_dataset

def load_model(checkpoint_path, hidden_unit):
    if DATASET == 'MNIST':
        model = models.Simple_FC(hidden_unit)
    elif DATASET == 'CIFAR-10':
        model = models.FiveLayerCNN(hidden_unit)

    checkpoint = torch.load(os.path.join(checkpoint_path, 'Simple_FC_%d.pth' % hidden_unit))
    model.load_state_dict(checkpoint['net'])

    return model


def get_clean_noisy_dataloader(dataset_path):
    # Load the two dataset
    org_train_dataset, noisy_train_dataset = load_dataset(dataset_path)

    # Spilt the Training set to the ones with clean labels and the ones with random (noisy) labels
    clean_label_list, noisy_label_list = [], []

    for i in range(len(org_train_dataset)):
        data = org_train_dataset[i][0].numpy()

        if org_train_dataset[i][1] != noisy_train_dataset[i][1]:
            noisy_label_list.append((data, org_train_dataset[i][1]))
        else:
            clean_label_list.append((data, org_train_dataset[i][1]))

    clean_label_dataset = datasets.ListDataset(clean_label_list)
    noisy_label_dataset = datasets.ListDataset(noisy_label_list)

    clean_label_dataloader = dd_exp.DataLoaderX(clean_label_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                      num_workers=0, pin_memory=True)
    noisy_label_dataloader = dd_exp.DataLoaderX(noisy_label_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                      num_workers=0, pin_memory=True)

    return clean_label_dataloader, noisy_label_dataloader, len(noisy_label_list)

def get_hidden_features(model, dataloader):
    # Obtain the hidden features
    data, hidden_features, predicts, true_labels = [], [], [], []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            hidden_feature = model(inputs, path='half1')
            outputs = model(hidden_feature, path='half2')

            for input in inputs:
                input = input.cpu().detach().numpy()
                data.append(input)

            for hf in hidden_feature:
                hf = hf.cpu().detach().numpy()
                hidden_features.append(hf)

            for output in outputs:
                predict = output.cpu().detach().numpy().argmax()
                predicts.append(predict)

            for label in labels:
                true_labels.append(label)

    if DATASET == 'MNIST':
        image_size = 28 * 28
        feature_size = model.n_hidden_units
    elif DATASET == 'CIFAR-10':
        image_size = 32 * 32 * 3
        feature_size = model.n_hidden_units * 8
    else:
        raise NotImplementedError

    data = np.array(data).reshape(len(true_labels), image_size)
    hidden_features = np.array(hidden_features).reshape(len(true_labels), feature_size)
    predicts = np.array(predicts).reshape(len(true_labels), )
    true_labels = np.array(true_labels).reshape(len(true_labels), )

    return data, hidden_features, predicts, true_labels


def knn_prediction_test(k):
    print('\nKNN Prediction Test: k = %d\n' % k)
    accuracy_list, train_accuracy, test_accuracy, test_losses = [], [], [], []

    for i, test_number in enumerate(TEST_NUMBERS):
        directory = f"assets/{DATASET}/N=%d-3D/TEST-%d/epoch=%d-noise-%d-model-%d" % (
        N_SAMPLES, TEST_GROUP, N_EPOCHS, label_noise_ratio * 100, test_number)

        dataset_path = os.path.join(directory, 'dataset')
        clean_label_dataloader, noisy_label_dataloader, n_noisy_data = get_clean_noisy_dataloader(dataset_path)

        accuracy_list.append([])

        for n in hidden_units:
            # Initialize model with pretrained weights
            checkpoint_path = os.path.join(directory, "ckpt")
            model = load_model(checkpoint_path, n)
            model.eval()

            # Obtain the hidden features of the clean data set
            data, hidden_features, predicts, labels = get_hidden_features(model, clean_label_dataloader)
            data_2, hidden_features_2, predicts_2, labels_2 = get_hidden_features(model, noisy_label_dataloader)

            knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
            knn.fit(hidden_features, labels)

            correct = sum(knn.predict(hidden_features_2) == labels_2)
            accuracy_list[-1].append(correct / n_noisy_data)

            print('Test No = %d ; Hidden Units = %d ; Correct = %d' % (test_number, n, correct))

        # Get Parameters and dataset Losses
        dictionary_path = os.path.join(directory, "dictionary.csv")
        train_accuracy.append(get_train_accuracy(dictionary_path))
        test_accuracy.append(get_test_accuracy(dictionary_path))
        test_losses.append(get_test_losses(dictionary_path))

    accuracy_list = np.mean(np.array(accuracy_list), axis=0)
    train_accuracy = np.mean(np.array(train_accuracy), axis=0)
    test_accuracy = np.mean(np.array(test_accuracy), axis=0)
    test_losses = np.mean(np.array(test_losses), axis=0)

    # Plot the Diagram
    plt.figure(figsize=(10, 7))
    ax = plt.axes()

    if DATASET == 'MNIST':
        scale_function = (lambda x: x ** (1 / 3), lambda x: x ** 3)
        ax.set_xscale('function', functions=scale_function)

    ln1 = ax.plot(hidden_units, accuracy_list, marker='o', label='KNN Prediction Accuracy (Noisy Training Data)', color='red')
    ln2 = ax.plot(hidden_units, train_accuracy, marker='o', label='Train Accuracy (Full Train Set)', color='orange')
    ln3 = ax.plot(hidden_units, test_accuracy, marker='o', label='Test Accuracy (Full dataset Set)', color='green')
    ax.set_xlabel('Number of Hidden Neurons (N)')
    ax.set_ylabel('Accuracy Rate')

    ax2 = ax.twinx()
    ln4 = ax2.plot(hidden_units, test_losses, marker='o', label='Test Losses (Full dataset Set)', color='blue')
    ax2.set_ylabel('Cross Entropy Loss')
    ax2.set_ylim([0, 2.75])

    if DATASET == 'MNIST':
        plt.xticks([1, 5, 15, 40, 100, 250, 500, 1000])

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.grid()

    plt.title('%d-KNN Feature Extraction dataset' % k)
    plt.savefig(f'assets/{DATASET}/N=%d-3D/TEST-%d/%d-NN Feature Extraction Test (epoch = 4000).png'
                % (N_SAMPLES, TEST_GROUP, k))


'''
def class_center_distance_test(d):
    clean_label_dataloader, noisy_label_dataloader = get_clean_noisy_dataloader()

    mean_distance_to_classfier_centers, mean_distance_to_classfier_centers_2 = [], []

    for n in hidden_units:
        # Initialize model with pretrained weights
        model = load_model(n)
        model.eval()

        # Obtain the hidden features of the clean data set
        data, hidden_features, predicts, true_labels = get_hidden_features(model, clean_label_dataloader)

        # Perform SVD Decomposition to extract features
        # if n > d:
        #     hidden_features = decomposition_by_SVD(hidden_features, d)

        # Find the center for each class
        class_centers = []
        for cls in range(10):
            class_data = hidden_features[(true_labels == cls).flatten()]  # Subset of data points belonging to the class
            if len(class_data) == 0:
                class_center = np.zeros(d)  # Set to zeros if the array is empty
            else:
                class_center = np.mean(class_data, axis=0)  # Calculate the mean along each feature axis
            class_centers.append(class_center)

        # Compute average distance of hidden features of random labelled training images to the 10 centers
        distance = []

        for i in range(len(hidden_features)):
            p1 = hidden_features[i]
            p2 = class_centers[true_labels[i]]
            cosine_distance = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
            distance.append(cosine_distance)

        print(n, sum(distance) / len(distance))
        mean_distance_to_classfier_centers.append(sum(distance) / len(distance))


        # Obtain the hidden features of the noisy data set
        data_2, hidden_features_2, predicts_2, true_labels_2 = get_hidden_features(model, noisy_label_dataloader)

        # if n > d:
        #     hidden_features_2 = decomposition_by_SVD(hidden_features_2, d)

        # Compute average distance of hidden features of random labelled training images to the 10 centers
        distance_2 = []

        for i in range(len(hidden_features_2)):
            p1 = hidden_features_2[i]
            p2 = class_centers[true_labels_2[i]]
            cosine_distance = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
            distance_2.append(cosine_distance)

        print(n, sum(distance_2) / len(distance_2), '\n')
        mean_distance_to_classfier_centers_2.append(sum(distance_2) / len(distance_2))


    # Plot the Diagram
    plt.figure(figsize=(10, 7))
    ax = plt.axes()
    scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)
    ax.set_xscale('function', functions=scale_function)
    plt.plot(hidden_units, mean_distance_to_classfier_centers, marker='o',
             label='Average Distance from class centers - Clean')
    plt.plot(hidden_units, mean_distance_to_classfier_centers_2, marker='o',
                 label='Average Distance from class centers - Noisy')
    plt.xticks([0, 5, 15, 40, 100, 200, 400, 800])
    plt.legend()
    plt.xlabel('Number of Parameters (* 10^3)')
    plt.ylabel('Cosine Distance')
    plt.title('Class Center Distance dataset')
    plt.savefig(os.path.join(plots_path, 'Class Center Distance dataset.png'))


def label_noise_pertubation_test(gaussian_mean, gaussian_std_dev):
    org_train_dataset, noisy_train_dataset = load_dataset()

    test_dataset_list = []

    for i in range(len(org_train_dataset)):
        if org_train_dataset[i][1] != noisy_train_dataset[i][1]:
            data = org_train_dataset[i][0].numpy().astype(np.float32)
            pertub_data = add_gaussian_noise(data, gaussian_mean, gaussian_std_dev)
            pertub_data = torch.tensor(pertub_data)

            test_dataset_list.append((pertub_data, org_train_dataset[i][1]))

    test_dataset = mnist_dd_exp.ListDataset(test_dataset_list)

    test_dataloader = mnist_dd_exp.DataLoaderX(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                               pin_memory=True)

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
    plt.plot(hidden_units, accuracy_list, marker='o', label='Data Noise dataset Error')
    plt.xticks([1, 5, 15, 40, 100, 250, 500, 1000])
    plt.legend()
    plt.xlabel('Number of hidden units (H)')
    plt.ylabel('Accuracy on Pertubated Noisy Data Points')
    plt.title('Gaussian standard deviation = %lf' % gaussian_std_dev)
    plt.savefig(os.path.join(plots_path, 'Pertubated Data dataset - %lf.png' % gaussian_std_dev))
'''

if __name__ == '__main__':
    knn_prediction_test(1)
    knn_prediction_test(5)
    knn_prediction_test(10)
