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
lr_decay = True
hidden_units = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150, 200]
n_epochs = 4000
learning_rate = 0.05
sample_size = 4000
label_noise_ratio = 0.2

directory = "assets/MNIST/standard-case/epoch=%d-noise-20" % n_epochs

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
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

    if label_noise_ratio > 0:
        label_noise_transform = transforms.Lambda(lambda y: torch.tensor(np.random.randint(0, 10)))
        num_samples = len(trainset)
        num_noisy_samples = int(label_noise_ratio * num_samples)

        noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
        for idx in noisy_indices:
            trainset.targets[idx] = label_noise_transform(trainset.targets[idx])

    trainset = torch.utils.data.Subset(trainset, indices=np.arange(sample_size))
    trainloader = DataLoaderX(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)

    '''
    for images, targets in trainloader:
        for i in range(18):
            plt.subplot(3, 6, i + 1)
            plt.imshow(images[i][0], cmap='gray')
        plt.savefig('Label Noise Data.png')
        print(targets[:6])
        print(targets[6:12])
        print(targets[12:18])
        break
    '''

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    testloader = DataLoaderX(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    print('Load MINST dataset success;')
    return trainloader, testloader


#------------------------------------------------------------------------------------------


class simple_FC(nn.Module):
    def __init__(self, n_hidden):
        super(simple_FC, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n_hidden),
            nn.ReLU()
        )
        #self.dropout = nn.Dropout(0.6)
        self.classifier = nn.Linear(n_hidden, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# Set the neural network model to be used
def get_model(hidden_unit):
    model = simple_FC(hidden_unit)
    model = model.to(device)

    if hidden_unit == 1:
        torch.nn.init.xavier_uniform_(model.features[1].weight, gain=1.0)
        torch.nn.init.xavier_uniform_(model.classifier.weight, gain=1.0)
    else:
        torch.nn.init.normal_(model.features[1].weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.1)

        if weight_reuse:
            print('Use previous checkpoints to initialize the weights:')
            i = 1 # load the closest previous model for weight reuse
            while not os.path.exists(os.path.join(checkpoint_path, 'Simple_FC_%d.pth'%(hidden_unit-i))):
                print('     loading from simple_FC_%d.pth'%(hidden_unit-i))
                i += 1
            checkpoint = torch.load(os.path.join(checkpoint_path, 'Simple_FC_%d.pth'%(hidden_unit-i)))
            with torch.no_grad():
                model.features[1].weight[:hidden_unit-i, :].copy_(checkpoint['net']['features.1.weight'])
                model.features[1].bias[:hidden_unit-i].copy_(checkpoint['net']['features.1.bias'])
                model.classifier.weight[:, :hidden_unit-i].copy_(checkpoint['net']['classifier.weight'])
                model.classifier.bias.copy_(checkpoint['net']['classifier.bias'])

    print("Model with %d hidden neurons successfully generated;" % hidden_unit)

    print('Number of parameters: %d'%sum(p.numel() for p in model.parameters()))
    return model


#------------------------------------------------------------------------------------------


# Model Training
def train(trainloader, model, optimizer, criterion):
    model.train()
    cumulative_loss, correct, total = 0.0, 0, 0

    for idx, (inputs, labels) in enumerate(trainloader):
        # Calculate the training loss
        labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.argmax(1)).sum().item()

    train_loss = cumulative_loss / len(trainloader)
    train_acc = correct / total

    return model, train_loss, train_acc


# Model testing
def test(testloader, model):
    model.eval()
    cumulative_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            # Calculate the testing loss
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

    test_loss = cumulative_loss / len(testloader)
    test_acc = correct/total

    return test_loss, test_acc


# Train and Evalute the model
def train_and_evaluate_model(trainloader, testloader, model, optimizer, criterion, hidden_unit):
    train_loss, train_acc, epoch = 0.0, 0.0, 0

    # Stops the training within the pre-set epoch size or when the model fits the training set (99%)
    for epoch in range(n_epochs):
        if epoch % 50 == 0:
            optimizer.param_groups[0]['lr'] = learning_rate / pow(1 + epoch // 50, 0.5)
            print("Learning Rate : ", optimizer.param_groups[0]['lr'])

        # Train the model
        model, train_loss, train_acc = train(trainloader, model, optimizer, criterion)

        # Print the status of current training and testing outcome
        print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f" % (epoch, train_loss, train_acc))

    # Evaluate the model
    test_loss, test_acc = test(testloader, model)

    if weight_reuse:
        state = {
            'net': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(checkpoint_path, 'Simple_FC_%d.pth'%hidden_unit))
        print("Torch saved successfully！")

    return train_loss, train_acc, test_loss, test_acc


if __name__ == '__main__':
    # Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    torch.backends.cudnn.benchmark = True

    # Get the training and testing data of specific sample size
    trainloader, testloader = get_train_and_test_dataloader()

    # Main Training Unit
    for hidden_unit in hidden_units:
        # Generate the model with specific number of hidden_unit
        model = get_model(hidden_unit)
        parameters = sum(p.numel() for p in model.parameters())

        # Set the optimizer and criterion 
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # Train and evalute the model
        train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(trainloader, testloader, \
                                    model, optimizer, criterion, hidden_unit)

        # Print training and evaluation outcome
        print("\nHidden Neurons : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; "
              "Test Acc : %.3f\n\n" % (hidden_unit, parameters, train_loss, train_acc, test_loss, test_acc))

        # Write the training and evaluation output to file
        f = open(output_file, "a")
        f.write("Hidden Neurons : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; "
                "Test Acc : %.3f\n" % (hidden_unit, parameters, train_loss, train_acc, test_loss, test_acc))
        f.close()
