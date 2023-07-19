import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from prefetch_generator import BackgroundGenerator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


# ------------------------------------------------------------------------------------------


# Training Settings
hidden_units = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100,
                120, 150, 200, 400, 600, 800, 1000]

sample_size = 4000
batch_size = 64
label_noise_ratio = 0.2

n_epochs = 4000
learning_rate_decay = True
learning_rate = 0.05

weight_reuse = False
tSNE_Visualization = False
save_model = True

TEST_NUMBER = 5

directory = "assets/MNIST/sub-set-3d/epoch=%d-noise-%d-model-%d" % (n_epochs, label_noise_ratio * 100, TEST_NUMBER)
dataset_path = "data/MNIST/Test-%d" % TEST_NUMBER

dictionary_path = os.path.join(directory, "dictionary.csv")
tsne_path = os.path.join(directory, "t-SNE")
checkpoint_path = os.path.join(directory, "ckpt")

if not os.path.isdir(directory):
    os.mkdir(directory)
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
if tSNE_Visualization and not os.path.isdir(tsne_path):
    os.mkdir(tsne_path)
if save_model and not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print('Using device : ', torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))


# ------------------------------------------------------------------------------------------


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.data = []
        self.targets = []

        for i in range(len(data_list)):
            self.data.append(data_list[i][0])
            self.targets.append(data_list[i][1])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def get_list(self):
        list = []
        for i in range(self.__len__()):
            list.append([self.data[i], int(self.targets[i])])

        return list


def create_training_set():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    train_dataset = torch.utils.data.Subset(train_dataset, indices=np.arange(sample_size))
    torch.save(list(train_dataset), os.path.join(dataset_path, 'subset-clean.pth'))

    train_dataset_2_list = torch.load(os.path.join(dataset_path, 'subset-clean.pth'))
    train_dataset_2 = ListDataset(train_dataset_2_list)

    label_noise_transform = transforms.Lambda(lambda y: torch.tensor(np.random.randint(0, 10)))
    num_noisy_samples = int(label_noise_ratio * len(train_dataset_2))

    noisy_indices = np.random.choice(len(train_dataset_2), num_noisy_samples, replace=False)
    for idx in noisy_indices:
        train_dataset_2.targets[idx] = label_noise_transform(train_dataset_2.targets[idx])

    print("%d Label Noise added to Train Data;\n" % (label_noise_ratio * 100))

    torch.save(train_dataset_2.get_list(), os.path.join(dataset_path, 'subset-noise-20%.pth'))


# Return the train_dataloader and test_dataloader of MINST
def get_train_and_test_dataloader():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torch.load(os.path.join(dataset_path, 'subset-noise-20%.pth'))

    train_dataloader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    test_dataloader = DataLoaderX(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print('Load MINST dataset success;')

    return train_dataloader, test_dataloader


# ------------------------------------------------------------------------------------------


class Simple_FC(nn.Module):
    def __init__(self, n_hidden):
        self.n_hidden_neuron = n_hidden

        super(Simple_FC, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n_hidden),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n_hidden, 10)

    def get_n_hidden_neuron(self):
        return self.n_hidden_neuron

    def forward_half1(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def forward_half2(self, x):
        out = self.classifier(x)
        return out

    def forward(self, x, path='all'):
        if path == 'all':
            x = self.forward_half1(x)
            x = self.forward_half2(x)
        elif path == 'half1':
            x = self.forward_half1(x)
        elif path == 'half2':
            x = self.forward_half2(x)
        else:
            raise NotImplementedError

        return x


def load_model_from_checkpoint(model, hidden_unit):
    print(' Weight Reuse: Use previous checkpoints to initialize the weights:')
    i = 1

    # load the closest previous model for weight reuse
    while not os.path.exists(os.path.join(checkpoint_path, 'Simple_FC_%d.pth' % (hidden_unit - i))):
        i += 1

    print('     loading from simple_FC_%d.pth' % (hidden_unit - i))

    checkpoint = torch.load(os.path.join(checkpoint_path, 'Simple_FC_%d.pth' % (hidden_unit - i)))

    with torch.no_grad():
        model.features[1].weight[:hidden_unit - i, :].copy_(checkpoint['net']['features.1.weight'])
        model.features[1].bias[:hidden_unit - i].copy_(checkpoint['net']['features.1.bias'])
        model.classifier.weight[:, :hidden_unit - i].copy_(checkpoint['net']['classifier.weight'])
        model.classifier.bias.copy_(checkpoint['net']['classifier.bias'])

    return model


# Set the neural network model to be used
def get_model(hidden_unit, device):
    model = Simple_FC(hidden_unit)
    model = model.to(device)

    if hidden_unit == 1:
        torch.nn.init.xavier_uniform_(model.features[1].weight, gain=1.0)
        torch.nn.init.xavier_uniform_(model.classifier.weight, gain=1.0)
    else:
        torch.nn.init.normal_(model.features[1].weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.1)

        if weight_reuse:
            model = load_model_from_checkpoint(model, hidden_unit)

    print("Model with %d hidden neurons successfully generated;" % hidden_unit)

    print('Number of parameters: %d' % sum(p.numel() for p in model.parameters()))

    return model


# ------------------------------------------------------------------------------------------


# Model Training
def train(train_dataloader, model, optimizer, criterion):
    model.train()
    cumulative_loss, correct, total = 0.0, 0, 0

    for idx, (inputs, labels) in enumerate(train_dataloader):
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

    train_loss = cumulative_loss / len(train_dataloader)
    train_acc = correct / total

    return model, train_loss, train_acc


# Model testing
def test(model, test_dataloader):
    model.eval()
    cumulative_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

    test_loss = cumulative_loss / len(test_dataloader)
    test_acc = correct/total

    return test_loss, test_acc


def model_t_sne(model, trainloader, hidden_unit):
    model.eval()

    hidden_features, predicts = [], []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)

            hidden_feature = model(inputs, path='half1')
            outputs = model(hidden_feature, path='half2')

            for hf in hidden_feature:
                hidden_features.append(hf.cpu().detach().numpy())

            for output in outputs:
                predict = output.cpu().detach().numpy().argmax()
                predicts.append(predict)

    hidden_features = np.array(hidden_features)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(hidden_features)

    plt.figure(figsize=(30, 20))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=predicts, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.title('t-SNE Hidden Features Visualization (N = %d)' % hidden_unit)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(os.path.join(tsne_path, 't-SNE_Hidden_Features_%d.jpg' % hidden_unit))


def model_save(model, epoch, test_accuracy):
    state = {
        'net': model.state_dict(),
        'acc': test_accuracy,
        'epoch': epoch,
    }
    torch.save(state, os.path.join(checkpoint_path, 'Simple_FC_%d.pth' % hidden_unit))
    print("Torch saved successfully!")


def status_save(hidden_unit, epoch, parameters, train_loss, train_acc, test_loss, test_acc):
    dictionary = {'Hidden Neurons': hidden_unit, 'Epoch': epoch, 'Parameters': parameters, 'Train Loss': train_loss,
                  'Train Accuracy': train_acc, 'Test Loss': test_loss, 'Test Accuracy': test_acc}

    with open(dictionary_path, "a", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())

        # Write the data rows
        writer.writerow(dictionary)
        print('Done writing dict to a csv file')


# Train and Evalute the model
def train_and_evaluate_model(trainloader, testloader, model, optimizer, criterion):
    train_loss, train_acc, test_loss, test_acc, epoch = 0.0, 0.0, 0.0, 0.0, 0
    parameters = sum(p.numel() for p in model.parameters())
    hidden_unit = model.get_n_hidden_neuron()

    for epoch in range(1, n_epochs + 1):
        model, train_loss, train_acc = train(trainloader, model, optimizer, criterion)
        print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f" % (epoch, train_loss, train_acc))

        if epoch % 100 == 0:
            test_loss, test_acc = test(model, testloader)
            status_save(hidden_unit, epoch, parameters, train_loss, train_acc, test_loss, test_acc)
            print("Hidden Neurons : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; "
                  "Test Acc : %.3f\n" % (hidden_unit, parameters, train_loss, train_acc, test_loss, test_acc))

            if learning_rate_decay:
                optimizer.param_groups[0]['lr'] = learning_rate / pow(1 + epoch // 50, 0.5)
                print("Learning Rate : ", optimizer.param_groups[0]['lr'])

    if tSNE_Visualization:
        model_t_sne(model, trainloader, hidden_unit)

    if save_model:
        model_save(model, test_acc, epoch)

    return


if __name__ == '__main__':
    # Initialize Status Dictionary
    dictionary = {'Hidden Neurons': 0, 'Epoch': 0, 'Parameters': 0, 'Train Loss': 0,
                  'Train Accuracy': 0, 'Test Loss': 0, 'Test Accuracy': 0}

    with open(dictionary_path, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
        writer.writeheader()

    # Create the training dataset
    create_training_set()

    # Get the training and testing data of specific sample size
    train_dataloader, test_dataloader = get_train_and_test_dataloader()

    # Main Training Unit
    for hidden_unit in hidden_units:
        # Generate the model with specific number of hidden_unit
        model = get_model(hidden_unit, device)

        # Set the optimizer and criterion
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # Train and evaluate the model
        train_and_evaluate_model(train_dataloader, test_dataloader, model, optimizer, criterion)
