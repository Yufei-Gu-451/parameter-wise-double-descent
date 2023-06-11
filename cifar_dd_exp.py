import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np

#------------------------------------------------------------------------------------------


# Training Settings
weight_reuse = False
lr_decay = False
#hidden_units = [1, 50, 100]
CNN_widths = [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_epochs = 1000
momentum = 0.95
learning_rate = 0.01
lr_decay_rate = 0.9
sample_size = 4000

if weight_reuse:
    directory = "assets/cifar/weight-reuse-case/epoch=%d" % n_epochs
else:
    directory = "assets/cifar/standard-case/epoch=%d" % n_epochs

output_file = os.path.join(directory, "epoch=%d.txt" % n_epochs)
checkpoint_path = os.path.join(directory, "ckpt")

if not os.path.isdir(directory):
    os.mkdir(directory)
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


#------------------------------------------------------------------------------------------


from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# Return the trainloader and testloader of MINST
def get_train_and_test_dataloader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

    trainloader = DataLoaderX(trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=False)

    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    testloader = DataLoaderX(testset, batch_size=64, shuffle=False, num_workers=8, pin_memory=False)

    print('Load CIFAR-10 dataset success;')

    return trainloader, testloader


#------------------------------------------------------------------------------------------\


class FiveLayerCNN(nn.Module):
    def __init__(self, k):
        super(FiveLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(1, 2)

        self.conv2 = nn.Conv2d(k, 2 * k, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * k)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(2 * k, 4 * k, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * k)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(4 * k, 8 * k, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(8 * k)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(8)

        self.fc = nn.Linear(8 * k, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model(hidden_unit):
    model = FiveLayerCNN(hidden_unit)
    model = model.to(device)

    print("Model with %d hidden neurons successfully generated;" % hidden_unit)

    return model


#------------------------------------------------------------------------------------------


def train_and_evaluate_model(trainloader, testloader, model, optimizer, criterion, CNN_width):
    return train_loss, train_acc, test_loss, test_acc


#------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    torch.backends.cudnn.benchmark = True

    # Get the training and testing data of specific sample size
    trainloader, testloader = get_train_and_test_dataloader()

    # Main Training Unit
    for CNN_width in CNN_widths:
        # Generate the model with specific number of hidden_unit
        model = get_model(CNN_width)
        parameters = sum(p.numel() for p in model.parameters())
        print('Number of parameters: %d' % sum(p.numel() for p in model.parameters()))

        # Set the optimizer and criterion
        optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        # Train and evalute the model
        train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(trainloader, testloader, \
                                    model, optimizer, criterion, CNN_width)

        # Print training and evaluation outcome
        print("\nCNN_width : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; "
              "Test Acc : %.3f\n\n" % (CNN_width, parameters, train_loss, train_acc, test_loss, test_acc))

        # Write the training and evaluation output to file
        f = open(output_file, "a")
        f.write("CNN_width : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; "
                "Test Acc : %.3f\n" % (CNN_width, parameters, train_loss, train_acc, test_loss, test_acc))
        f.close()