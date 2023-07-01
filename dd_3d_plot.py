import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import csv
import os

n_epochs = 4000
label_noise_ratio = 0.2
gap = 50

directory = "assets/MNIST/sub-set-3d/epoch=%d-noise-%d" % (n_epochs, label_noise_ratio)

dictionary_path = os.path.join(directory, "dictionary.csv")
plots_path = os.path.join(directory, 'plots')

if not os.path.isdir(plots_path):
    os.mkdir(plots_path)

index, hidden_units, parameters, epochs = [], [], [], []
train_losses, train_accs, test_losses, test_accs = [], [], [], []

# Open a csv file for writing
with open(dictionary_path, "r", newline="") as infile:
    # Create a reader object
    reader = csv.DictReader(infile)

    i = -1
    for row in reader:
        if i == -1 or n == n_epochs // gap:
            hidden_units.append([])
            parameters.append([])
            epochs.append(([]))
            train_losses.append([])
            train_accs.append([])
            test_losses.append([])
            test_accs.append([])
            i, n = i + 1, 0

        hidden_units[i].append(int(row['Hidden Neurons']))
        parameters[i].append(int(row['Parameters']))
        epochs[i].append(int(row['Epoch']))
        train_losses[i].append(float(row['Train Loss']))
        train_accs[i].append(float(row['Train Accuracy']))
        test_losses[i].append(float(row['Test Loss']))
        test_accs[i].append(float(row['Test Accuracy']))

        n += 1

hidden_units = np.flipud(np.array(hidden_units))
parameters = np.flipud(np.array(parameters))
epochs = np.array(epochs)
train_losses = np.flipud(np.array(train_losses))
train_accs = np.flipud(np.array(train_accs))
test_losses = np.flipud(np.array(test_losses))
test_accs = np.flipud(np.array(test_accs))

print(hidden_units)

# Creating figure
fig = plt.figure(figsize=(30, 20))
ax = plt.axes(projection='3d')

ax.plot_wireframe(hidden_units, epochs, test_losses, color='orange')

plt.savefig(os.path.join(plots_path, 'Test_Loss-Hidden_Neurons.png'))
