import matplotlib.pyplot as plt
import re
import os

epochs = 100
noise = 0.0

directory = 'assets/CIFAR-10/records/epoch=%d-noise=%d' % (epochs, noise * 100)
plots_directory = os.path.join(directory, 'plots')
input_file = os.path.join(directory, 'epoch=%d-noise=%d.txt' % (epochs, noise * 100))

if not os.path.isdir(plots_directory):
    os.mkdir(plots_directory)

if __name__ == '__main__':
    array = []
    f = open(input_file, 'r')
    for line in f:
        pair = re.findall(r"(\d*\.*\d+)", line)
        print(pair)
        array.append(pair)

    hidden_units, parameters = [], []
    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    for pair in array:
        hidden_units.append(float(pair[0]))
        parameters.append(float(pair[1]) / 1000)
        train_losses.append(float(pair[2]))
        train_accs.append(float(pair[3]))
        test_losses.append(float(pair[4]))
        test_accs.append(float(pair[5]))

    scale_function = (lambda x: x**(1/3), lambda x: x**3)

    plt.figure(figsize=(5, 4))
    plt.plot(hidden_units, test_losses, marker='o', label='test')
    plt.plot(hidden_units, train_losses, marker='o', label='train')
    plt.xticks([1, 10, 20, 30, 40, 50, 60, 64])
    plt.xlabel('Number of hidden units (H)')
    plt.ylabel('Cross Entropy Loss')
    plt.title('CIFAR-10')
    plt.savefig(os.path.join(plots_directory, 'CrossEntropyLoss-ModelWidthK.png'))

    plt.figure(figsize=(6, 4.5))
    plt.plot(parameters, test_losses, marker='o', label='test')
    plt.plot(parameters, train_losses, marker='o', label='train')
    plt.xlabel('Number of parameters/weights (×10ˆ3)')
    plt.ylabel('Cross Entropy Loss')
    plt.title('CIFAR-10')
    plt.savefig(os.path.join(plots_directory, 'CrossEntropyLoss-Parameters.png'))

    plt.figure(figsize=(5, 4))
    plt.plot(hidden_units, test_accs, marker='o', label='test')
    plt.plot(hidden_units, train_accs, marker='o', label='train')
    plt.xticks([1, 10, 20, 30, 40, 50, 60, 64])
    plt.ylim([0, 1])
    plt.xlabel('Number of hidden units (H)')
    plt.ylabel('Accuracy')
    plt.title('CIFAR-10')
    plt.savefig(os.path.join(plots_directory, 'Accuracy-ModelWidthK.png'))

    plt.figure(figsize=(5, 4))
    plt.plot(parameters, test_accs, marker='o', label='test')
    plt.plot(parameters, train_accs, marker='o', label='train')
    plt.ylim([0, 1])
    plt.xlabel('Number of parameters/weights (×10ˆ3)')
    plt.ylabel('Accuracy')
    plt.title('CIFAR-10')
    plt.savefig(os.path.join(plots_directory, 'Accuracy-Parameters.png'))