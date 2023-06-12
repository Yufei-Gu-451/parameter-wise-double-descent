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
CNN_widths = [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_epochs = 100
learning_rate = 0.01
label_noise_ratio = 0.0

directory = "assets/CIFAR-10/std/epoch=%d-noise=%d" % (n_epochs, label_noise_ratio)

output_file = os.path.join(directory, "epoch=%d-noise=%d.txt" % (n_epochs, label_noise_ratio))
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

    if label_noise_ratio > 0:
        label_noise_transform = transforms.Lambda(lambda y: torch.tensor(np.random.randint(0, 10)))
        num_samples = len(trainset)
        num_noisy_samples = int(label_noise_ratio * num_samples)

        noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
        for idx in noisy_indices:
            trainset.targets[idx] = label_noise_transform(trainset.targets[idx])

    trainloader = DataLoaderX(trainset, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)


    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    testloader = DataLoaderX(testset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

    print('Load CIFAR-10 dataset success;')

    return trainloader, testloader


# ------------------------------------------------------------------------------------------\


class FiveLayerCNN(nn.Module):
    def __init__(self, k):
        super(FiveLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(k, 2 * k, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * k)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(1)

        self.conv3 = nn.Conv2d(2 * k, 4 * k, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * k)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(4 * k, 8 * k, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(8 * k)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

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


# ------------------------------------------------------------------------------------------


def train_and_evaluate_model(trainloader, testloader, model, optimizer, criterion):
    total_train_step = 0

    for i in range(n_epochs):
        model.train()
        cumulative_loss, correct, total = 0.0, 0, 0

        for data in trainloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            total_train_step = total_train_step + 1
            if total_train_step > 0 and total_train_step % 512 == 0:
                optimizer.param_groups[0]['lr'] = learning_rate / pow(1 + total_train_step // 512, 0.5)
                print("Learning Rate Decay: ", optimizer.param_groups[0]['lr'])

        train_loss = cumulative_loss / len(testloader)
        train_acc = correct / total
        print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f" % (i, train_loss, train_acc))

    model.eval()
    cumulative_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for data in testloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            cumulative_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_loss = cumulative_loss / len(testloader)
    test_acc = correct / total

    return train_loss, train_acc, test_loss, test_acc


# ------------------------------------------------------------------------------------------


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
        model = FiveLayerCNN(CNN_width)
        model = model.to(device)
        print("Model with %d hidden neurons successfully generated;" % CNN_width)

        parameters = sum(p.numel() for p in model.parameters())
        print('Number of parameters: %d' % sum(p.numel() for p in model.parameters()))

        # Set the optimizer and criterion
        optimizer = torch.optim.SGD(model.parameters(), momentum=0, lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # Train and evalute the model
        train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(trainloader, testloader,
                                                                              model, optimizer, criterion)

        # Print training and evaluation outcome
        print("\nCNN_width : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; "
              "Test Acc : %.3f\n\n" % (CNN_width, parameters, train_loss, train_acc, test_loss, test_acc))

        # Write the training and evaluation output to file
        f = open(output_file, "a")
        f.write("CNN_width : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; "
                "Test Acc : %.3f\n" % (CNN_width, parameters, train_loss, train_acc, test_loss, test_acc))
        f.close()
