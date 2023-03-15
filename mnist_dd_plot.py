import matplotlib.pyplot as plt
import os
import re
import math

epochs = 6000
weight_reuse = True

if weight_reuse:
    directory = 'assets/mnist/weight-reuse-case/epoch=%d/plots' % epochs
    input_file = 'assets/mnist/weight-reuse-case/epoch=%d/epoch=%d.txt' % (epochs, epochs)
else:
    directory = 'assets/mnist/standard-case/epoch=%d/plots' % epochs
    input_file = 'assets/mnist/standard-case/epoch=%d/epoch=%d.txt' % (epochs, epochs)

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
        #parameters.append(float(pair[1]) / 1000)
        train_losses.append(float(pair[1]))
        train_accs.append(float(pair[2]))
        test_losses.append(float(pair[3]))
        test_accs.append(float(pair[4]))

    for n in hidden_units:
        parameters.append(int( (785 * n + (n + 1) * 10) / 1000) )

    plt.figure(figsize=(5, 4))
    #ax = plt.axes()
    #ax.set_xscale('function', functions=(lambda x: x**(1/4), lambda x: x**4))
    plt.plot(hidden_units, test_losses, marker='o', label='test')
    plt.plot(hidden_units, train_losses, marker='o', label='train')
    #plt.xticks([1, 5, 20, 50, 100, 200, 400, 700, 1000])
    plt.xlabel('Number of hidden units (H)')
    plt.ylabel('Squared loss')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(directory, 'MSE-Neurons.png'))

    plt.figure(figsize=(6, 4.5))
    #ax = plt.axes()
    #ax.set_xscale('function', functions=(lambda x: x**(1/4), lambda x: x**4))
    plt.plot(parameters, test_losses, marker='o', label='test')
    plt.plot(parameters, train_losses, marker='o', label='train')
    #plt.xticks([14, 40, 100, 300, 800])
    #plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04])
    #plt.xlim((10, 300))
    #plt.ylim((-0.003, 0.1))
    plt.xlabel('Number of parameters/weights (×10ˆ3)')
    plt.ylabel('Squared loss')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(directory, 'MSE-Parameters.png'))

    plt.figure(figsize=(5, 4))
    #ax = plt.axes()
    #ax.set_xscale('function', functions=(lambda x: x**(1/4), lambda x: x**4))
    plt.plot(hidden_units, test_accs, marker='o', label='test')
    plt.plot(hidden_units, train_accs, marker='o', label='train')
    #plt.xticks([1, 5, 20, 50, 100, 200, 400, 700, 1000])
    plt.xlabel('Number of hidden units (H)')
    plt.ylabel('Accuracy')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(directory, 'Accuracy-Neurons.png'))

    plt.figure(figsize=(5, 4))
    #ax = plt.axes()
    #ax.set_xscale('function', functions=(lambda x: x**(1/4), lambda x: x**4))
    plt.plot(parameters, test_accs, marker='o', label='test')
    plt.plot(parameters, train_accs, marker='o', label='train')
    #plt.xticks([14, 40, 100, 300, 800])
    plt.xlabel('Number of parameters/weights (×10ˆ3)')
    plt.ylabel('Accuracy')
    plt.title('MNIST (n = 4×10ˆ3,d = 784,K = 10)')
    plt.savefig(os.path.join(directory, 'Accuracy-Parameters.png'))