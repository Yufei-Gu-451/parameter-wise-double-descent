import matplotlib.pyplot as plt
import os
import re
import math

epochs = 6000
weight_reuse = True

if weight_reuse:
    directory = 'assets/mnist/weight-reuse-case/epoch=%d-3/plots' % epochs
    input_file = 'assets/mnist/weight-reuse-case/epoch=%d-3/epoch=%d.txt' % (epochs, epochs)
else:
    directory = 'assets/mnist/standard-case/epoch=%d-w-wo-lr-decay/plots' % epochs
    input_file = 'assets/mnist/standard-case/epoch=%d-w-wo-lr-decay/epoch=%d-2.txt' % (epochs, epochs)

if not os.path.isdir(directory):
    os.mkdir(directory)

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

    #for n in hidden_units:
    #    parameters.append(int( (785 * n + (n + 1) * 10) / 1000) )

    scale_function = (lambda x: x**(1/3), lambda x: x**3)

    plt.figure(figsize=(5, 4))
    ax = plt.axes()
    ax.set_xscale('function', functions=scale_function)
    plt.plot(hidden_units, test_losses, marker='o', label='test')
    plt.plot(hidden_units, train_losses, marker='o', label='train')
    plt.xticks([1, 8, 20, 50, 100, 200, 400, 600, 1000])
    plt.xlabel('Number of hidden units (H)')
    plt.ylabel('Squared loss')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(directory, 'MSE-Neurons.png'))

    plt.figure(figsize=(6, 4.5))
    ax = plt.axes()
    #ax.set_xscale('function', functions=scale_function)
    ax.vlines([40], -0.003, 0.155, linestyles='dashed', color='black')
    plt.plot(parameters, test_losses, marker='o', label='test')
    plt.plot(parameters, train_losses, marker='o', label='train')
    #plt.xticks([5, 10, 40, 100])
    plt.yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
    #plt.xlim((10, 300))
    plt.ylim((-0.003, 0.155))
    plt.xlabel('Number of parameters/weights (×10ˆ3)')
    plt.ylabel('Squared loss')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(directory, 'MSE-Parameters.png'))

    plt.figure(figsize=(5, 4))
    ax = plt.axes()
    ax.set_xscale('function', functions=scale_function)
    plt.plot(hidden_units, test_accs, marker='o', label='test')
    plt.plot(hidden_units, train_accs, marker='o', label='train')
    plt.xticks([1, 5, 20, 50, 100, 200, 400, 600, 1000])
    plt.xlabel('Number of hidden units (H)')
    plt.ylabel('Accuracy')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(directory, 'Accuracy-Neurons.png'))

    plt.figure(figsize=(5, 4))
    ax = plt.axes()
    ax.set_xscale('function', functions=scale_function)
    plt.plot(parameters, test_accs, marker='o', label='test')
    plt.plot(parameters, train_accs, marker='o', label='train')
    plt.xticks([14, 40, 120, 300, 800])
    plt.xlabel('Number of parameters/weights (×10ˆ3)')
    plt.ylabel('Accuracy')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(directory, 'Accuracy-Parameters.png'))