import matplotlib.pyplot as plt
import os
import re
import math

epochs = 4000

directory = 'assets/MNIST/sub-set/epoch=%d-noise-20' % epochs
plots_directory = os.path.join(directory, 'plots')
input_file = os.path.join(directory, 'epoch=%d.txt' % epochs)

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

    #for n in hidden_units:
    #    parameters.append(int( (785 * n + (n + 1) * 10) / 1000) )


    scale_function = (lambda x: x**(1/4), lambda x: x**4)

    plt.figure(figsize=(5, 4))
    ax = plt.axes()
    ax.set_xscale('function', functions=scale_function)
    plt.plot(hidden_units, test_losses, marker='o', label='test')
    plt.plot(hidden_units, train_losses, marker='o', label='train')
    plt.xticks([1, 5, 15, 40, 100, 250, 500, 1000])
    #ax.vlines([4], -0.01, 2.55, linestyles='dashed', color='black')
    plt.ylim((-0.1, 2.6))
    plt.xlabel('Number of hidden units (H)')
    plt.ylabel('Squared loss')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(plots_directory, 'MSE-Neurons.png'))

    plt.figure(figsize=(6, 4.5))
    ax = plt.axes()
    ax.set_xscale('function', functions=scale_function)
    plt.plot(parameters, test_losses, marker='o', label='test')
    plt.plot(parameters, train_losses, marker='o', label='train')
    plt.xticks([1, 5, 15, 40, 100, 200, 400, 800])
    #plt.yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
    #plt.xlim((10, 300))
    #ax.vlines([40], -0.003, 0.09, linestyles='dashed', color='black')
    #plt.ylim((-0.003, 0.09))
    plt.xlabel('Number of parameters/weights (×10ˆ3)')
    plt.ylabel('Squared loss')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(plots_directory, 'MSE-Parameters.png'))

    plt.figure(figsize=(5, 4))
    ax = plt.axes()
    ax.set_xscale('function', functions=scale_function)
    plt.plot(hidden_units, test_accs, marker='o', label='test')
    plt.plot(hidden_units, train_accs, marker='o', label='train')
    plt.xticks([1, 5, 15, 40, 100, 250, 500, 1000])
    plt.xlabel('Number of hidden units (H)')
    plt.ylabel('Accuracy')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(plots_directory, 'Accuracy-Neurons.png'))

    plt.figure(figsize=(5, 4))
    ax = plt.axes()
    ax.set_xscale('function', functions=scale_function)
    plt.plot(parameters, test_accs, marker='o', label='test')
    plt.plot(parameters, train_accs, marker='o', label='train')
    plt.xticks([1, 5, 15, 40, 100, 200, 400, 800])
    plt.xlabel('Number of parameters/weights (×10ˆ3)')
    plt.ylabel('Accuracy')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(plots_directory, 'Accuracy-Parameters.png'))