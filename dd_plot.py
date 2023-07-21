import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import csv
import os

DATASET = 'CIFAR-10'

N_EPOCHS = 100
N_SAMPLES = 4000
BATCH_SIZE = 64

TEST_GROUP = 0
TEST_NUMBER = 1

label_noise_ratio = 0.2

directory = f"assets/{DATASET}/N=%d-3d/TEST-%d/epoch=%d-noise-%d-model-%d" \
                    % (N_SAMPLES, TEST_GROUP, N_EPOCHS, label_noise_ratio * 100, TEST_NUMBER)

dictionary_path = os.path.join(directory, "dictionary2.csv")
plots_path = os.path.join(directory, 'plots2')

if not os.path.isdir(plots_path):
    os.mkdir(plots_path)

index, hidden_units, parameters, epochs = [], [], [], []
train_losses, train_accs, test_losses, test_accs = [], [], [], []

with open(dictionary_path, "r", newline="") as infile:
    # Create a reader object
    reader = csv.DictReader(infile)

    for row in reader:
        if len(epochs) == 0 or int(row['Epoch']) < epochs[-1][-1]:
            hidden_units.append([])
            parameters.append([])
            epochs.append(([]))
            train_losses.append([])
            train_accs.append([])
            test_losses.append([])
            test_accs.append([])

        hidden_units[-1].append(int(row['Hidden Neurons']))
        parameters[-1].append(int(row['Parameters']))
        epochs[-1].append(int(row['Epoch']))
        train_losses[-1].append(float(row['Train Loss']))
        train_accs[-1].append(float(row['Train Accuracy']))
        test_losses[-1].append(float(row['Test Loss']))
        test_accs[-1].append(float(row['Test Accuracy']))

hidden_units = np.array(hidden_units)
parameters = np.array(parameters)
epochs = np.array(epochs)
train_losses = np.array(train_losses)
train_accs = np.array(train_accs)
test_losses = np.array(test_losses)
test_accs = np.array(test_accs)

print(hidden_units)


plt.figure(figsize=(10, 7))
plt.plot(hidden_units, test_losses[:, -1], marker='o', label='test', color='orange')
plt.plot(hidden_units, train_losses[:, -1], marker='o', label='train', color='blue')
plt.xlabel('Number of hidden units (H)')
plt.ylabel('Cross Entropy Loss')
plt.title(f'Double Descent on {DATASET} (N = %d)' % N_SAMPLES)
plt.savefig(os.path.join(plots_path, 'Losses-Hidden_Units-2D.png'))

my_col = cm.jet(test_accs/np.amin(test_accs))

fig2 = plt.figure(figsize=(15, 10))
ax = plt.axes(projection='3d')
ax.plot_surface(hidden_units, epochs, train_losses, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlabel('Train Losses')
plt.xlabel('Number of Hidden Units (H)')
plt.ylabel('Number of Epochs')
plt.title(f'Double Descent on {DATASET} (N = %d)' % N_SAMPLES)
plt.savefig(os.path.join(plots_path, 'Train_Losses-Hidden_Units-3D.png'))

fig3 = plt.figure(figsize=(15, 10))
ax = plt.axes(projection='3d')
ax.plot_surface(hidden_units, epochs, test_losses, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlabel('Test Losses')
plt.xlabel('Number of Hidden Units (H)')
plt.ylabel('Number of Epochs')
plt.title(f'Double Descent on {DATASET} (N = %d)' % N_SAMPLES)
plt.savefig(os.path.join(plots_path, 'Test_Losses-Hidden_Units-3D.png'))