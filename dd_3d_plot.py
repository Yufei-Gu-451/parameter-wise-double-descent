import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import csv
import os
import re

n_epochs = 4000
label_noise_ratio = 0.2

directory = "assets/MNIST/sub-set-3d/epoch=%d-noise-%d" % (n_epochs, label_noise_ratio)

dictionary_path = os.path.join(directory, "dictionary.csv")
plots_path = os.path.join(directory, 'plots')

if not os.path.isdir(plots_path):
    os.mkdir(plots_path)

hidden_units, parameters, epochs = [], [], []
train_losses, train_accs, test_losses, test_accs = [], [], [], []

# Open a csv file for writing
with open(dictionary_path, "r", newline="") as infile:
    # Create a reader object
    reader = csv.DictReader(infile)

    for row in reader:
        hidden_units.append(row['Hidden Neurons'])
        epochs.append(row['epochs'])
        parameters.append(row['Parameters'])
        train_losses.append(row['Train Loss'])
        train_accs.append(row['Train Accuracy'])
        test_losses.append(row['Test Loss'])
        test_accs.append(row['Test Accuracy'])


# Creating figure
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')

# Creating plot
ax.plot_surface(hidden_units, epochs, test_losses)

plt.savefig(os.path.join(plots_path, 'Test_Loss-Hidden_Neurons.png'))