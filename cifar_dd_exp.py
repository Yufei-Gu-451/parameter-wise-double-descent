import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

# ------------------------------------------------------------------------------------------


# Training Settings
lr_decay = True
CNN_widths = [64, 1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_epochs = 400
learning_rate = 0.1
label_noise_ratio = 0.2

directory = "assets/CIFAR-10/std/epoch=%d-noise=%d" % (n_epochs, label_noise_ratio * 100)

output_file = os.path.join(directory, "epoch=%d-noise=%d.txt" % (n_epochs, label_noise_ratio * 100))
checkpoint_path = os.path.join(directory, "ckpt")

if not os.path.isdir(directory):
    os.mkdir(directory)
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


# ------------------------------------------------------------------------------------------


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# Return the trainloader and testloader of MINST
def get_train_and_test_dataloader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    trainloader = DataLoaderX(trainset, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    testloader = DataLoaderX(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    print('Load CIFAR-10 dataset success;\n')

    return len(trainset), len(testset), trainloader, testloader

def add_label_noise(labels, noise_level):
    num_samples = labels.size(0)
    num_corrupt = int(num_samples * noise_level)
    noise_indices = np.random.choice(num_samples, num_corrupt, replace=False)
    noisy_labels = labels.clone()
    noisy_labels[noise_indices] = torch.randint(0, 10, (num_corrupt,), dtype=torch.long)
    return noisy_labels


# ------------------------------------------------------------------------------------------\


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class FiveLayerCNN(nn.Module):
    def __init__(self, k):
        super(FiveLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(k, 2 * k, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(2 * k)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(2 * k, 4 * k, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(4 * k)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(4 * k, 8 * k, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(8 * k)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.pool4 = nn.MaxPool2d(4)
        self.flatten = Flatten()
        self.fc = nn.Linear(8 * k, 10, bias=True)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ------------------------------------------------------------------------------------------


def train_and_evaluate_model(trainloader, testloader, model, optimizer, criterion, n_train_samples, n_test_samples):
    total_train_step = 0

    for i in range(n_epochs):
        model.train()
        cumulative_loss, correct, total = 0.0, 0, 0

        for data in trainloader:
            images, targets = data

            if label_noise_ratio > 0:
                targets = add_label_noise(targets, label_noise_ratio)

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()

            if total_train_step % 512 == 0:
                optimizer.param_groups[0]['lr'] = learning_rate / pow(1 + total_train_step // 512, 0.5)
                print("Learning Rate : ", optimizer.param_groups[0]['lr'])
            total_train_step = total_train_step + 1

        train_loss = cumulative_loss / len(trainloader)
        train_acc = correct / n_train_samples

        print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f" % (i, train_loss, train_acc))

    model.eval()
    cumulative_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for data in testloader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            cumulative_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_loss = cumulative_loss / len(testloader)
    test_acc = correct / n_test_samples

    return train_loss, train_acc, test_loss, test_acc


# ------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    torch.backends.cudnn.benchmark = True

    # Get the training and testing data of specific sample size
    n_train_samples, n_test_samples, trainloader, testloader = get_train_and_test_dataloader()

    # Main Training Unit
    for CNN_width in CNN_widths:
        # Generate the model with specific number of hidden_unit
        model = FiveLayerCNN(CNN_width)
        model = model.to(device)
        print("Model with %d hidden neurons successfully generated;" % CNN_width)

        parameters = sum(p.numel() for p in model.parameters())
        print('Number of parameters: %d' % sum(p.numel() for p in model.parameters()))

        # Set the optimizer and criterion
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # Train and evalute the model
        train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(trainloader, testloader,
                                model, optimizer, criterion, n_train_samples, n_test_samples)

        # Print training and evaluation outcome
        print("\nCNN_width : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; "
              "Test Acc : %.3f\n\n" % (CNN_width, parameters, train_loss, train_acc, test_loss, test_acc))

        # Write the training and evaluation output to file
        f = open(output_file, "a")
        f.write("CNN_width : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; "
                "Test Acc : %.3f\n" % (CNN_width, parameters, train_loss, train_acc, test_loss, test_acc))
        f.close()
