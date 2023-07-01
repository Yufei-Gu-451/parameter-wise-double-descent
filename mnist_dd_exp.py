import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import os


# ------------------------------------------------------------------------------------------


# Training Settings
# hidden_units = [20]
hidden_units = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150, 200, 400]

sample_size = 4000
batch_size = 128
label_noise_ratio = 0.2

n_epochs = 4000
learning_rate_decay = True
learning_rate = 0.1

weight_reuse = False
tSNE_Visualization = False
save_model = False
hebbian_learning = False
hebbian_learning_rate = 1

directory = "assets/MNIST/sub-set-3d/epoch=%d-noise-%d" % (n_epochs, label_noise_ratio)

dictionary_path = os.path.join(directory, "dictionary.csv")
tsne_path = os.path.join(directory, "t-SNE")
checkpoint_path = os.path.join(directory, "ckpt")

if not os.path.isdir(directory):
    os.mkdir(directory)
if not os.path.isdir(tsne_path):
    os.mkdir(tsne_path)
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


# ------------------------------------------------------------------------------------------


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# Return the train_dataloader and test_dataloader of MINST
def get_train_and_test_dataloader():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

    if label_noise_ratio > 0:
        label_noise_transform = transforms.Lambda(lambda y: torch.tensor(np.random.randint(0, 10)))
        num_samples = len(train_dataset)
        num_noisy_samples = int(label_noise_ratio * num_samples)

        noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
        for idx in noisy_indices:
            train_dataset.targets[idx] = label_noise_transform(train_dataset.targets[idx])

    train_dataset = torch.utils.data.Subset(train_dataset, indices=np.arange(sample_size))

    train_dataloader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

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

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    test_dataloader = DataLoaderX(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

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

        if hebbian_learning:
            self.features.requires_grad_(False)

        # self.dropout = nn.Dropout(0.6)
        self.classifier = nn.Linear(n_hidden, 10)
        self.softmax = nn.Softmax(dim=1)

    def get_n_hidden_neuron(self):
        return self.n_hidden_neuron

    def forward_half1(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def forward_half2(self, x):
        out = self.classifier(x)
        out = self.softmax(out)
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

    '''
    if hidden_unit == 1:
        torch.nn.init.xavier_uniform_(model.features[1].weight, gain=1.0)
        torch.nn.init.xavier_uniform_(model.classifier.weight, gain=1.0)
    else:
        torch.nn.init.normal_(model.features[1].weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.1)

        if weight_reuse:
            model = load_model_from_checkpoint(model, hidden_unit)
    '''

    nn.init.kaiming_uniform_(model.features[1].weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(model.classifier.weight, mode='fan_in', nonlinearity='relu')

    print("Model with %d hidden neurons successfully generated;" % hidden_unit)

    print('Number of parameters: %d' % sum(p.numel() for p in model.parameters()))

    return model


# ------------------------------------------------------------------------------------------

class Hebbian_Learning():
    def __init__(self):
        self.iterating_counts = 0
        self.past_products = 0
        self.learning_rate = hebbian_learning_rate

    def average_products(self):
        if self.iterating_counts == 0:
            return 0
        else:
            return self.past_products / self.iterating_counts

    def add_count(self):
        self.iterating_counts += 1

    def add_product(self, product):
        self.past_products += product

    def get_learning_rate(self):
        return self.learning_rate * pow(0.95, self.iterating_counts // sample_size)


def hebbian_train(model, inputs, HL):
    for input in inputs:
        hidden_feature = model(input, path='half1')
        hidden_feature = hidden_feature.cpu().detach().numpy()[0].reshape(model.n_hidden_neuron, 1)
        input = input.cpu().detach().numpy().reshape(1, 784)

        # Activation Threshold
        delta = HL.get_learning_rate() * np.subtract(hidden_feature * input, HL.average_products())

        # Gradient Threshold
        threshold_1 = np.percentile(delta, 90)
        threshold_2 = np.percentile(delta, 10)

        delta = np.add(np.where(delta >= threshold_1, delta, 0), np.where(delta <= threshold_2, delta, 0))

        # Replace parameters in-place
        state_dict = model.state_dict()
        parameters = state_dict['features.1.weight'].cpu().detach().numpy()
        state_dict['features.1.weight'] = torch.from_numpy(np.subtract(parameters, delta)).to(device)
        model.load_state_dict(state_dict)

        if HL.iterating_counts % 500 == 0:
            np.set_printoptions(threshold=sys.maxsize)
            print(HL.iterating_counts, HL.get_learning_rate())
            print(np.count_nonzero(delta < 0), np.count_nonzero(parameters < 0))
            print(delta.mean(), parameters.mean())
            print()

        #assert((parameters - delt == model.state_dict()['features.1.weight'].cpu().detach().numpy().all())

        # Add count
        HL.add_count()
        HL.add_product(hidden_feature * input)

    return model, HL


# Model Training
def train(train_dataloader, model, optimizer, criterion, HL=None):
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

        if hebbian_learning:
            model, HL = hebbian_train(model, inputs, HL)

        cumulative_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.argmax(1)).sum().item()

    train_loss = cumulative_loss / len(train_dataloader)
    train_acc = correct / total

    return model, train_loss, train_acc, HL


# Model testing
def test(test_dataloader):
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
    print("Torch saved successfully！")

def status_save(hidden_unit, epoch, parameters, train_loss, train_acc, test_loss, test_acc):
    dictionary = {'Hidden Neurons': hidden_unit, 'Epoch': epoch, 'Parameters': parameters, 'Train Loss': train_loss,
                  'Train Accuracy': train_acc, 'Test Loss': test_loss, 'Test Accuracy': test_acc}

    with open("person.csv", "w", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())

        # Write the data rows
        writer.writerow(dictionary)
        print('Done writing dict to a csv file')


# Train and Evalute the model
def train_and_evaluate_model(trainloader, testloader, model, optimizer, criterion, hidden_unit):
    train_loss, train_acc, test_loss, test_acc, epoch = 0.0, 0.0, 0.0, 0.0, 0
    parameters = sum(p.numel() for p in model.parameters())

    # Train the model
    if hebbian_learning:
        HL = Hebbian_Learning()
    else:
        HL = None

    # Stops the training within the pre-set epoch size or when the model fits the training set (99%)
    for epoch in range(1, n_epochs + 1):
        model, train_loss, train_acc, HL = train(trainloader, model, optimizer, criterion, HL)
        print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f" % (epoch, train_loss, train_acc))

        if epoch % 50 == 0:
            test_loss, test_acc = test(testloader)
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

    return train_loss, train_acc, test_loss, test_acc


if __name__ == '__main__':
    # Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    torch.backends.cudnn.benchmark = True

    dictionary = {'Hidden Neurons': 0, 'Epoch': 0, 'Parameters': 0, 'Train Loss': 0,
                  'Train Accuracy': 0, 'Test Loss': 0, 'Test Accuracy': 0}

    with open("person.csv", "w", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())

        # Write the header row
        writer.writeheader()

    # Get the training and testing data of specific sample size
    train_dataloader, test_dataloader = get_train_and_test_dataloader()

    # Main Training Unit
    for hidden_unit in hidden_units:
        # Generate the model with specific number of hidden_unit
        model = get_model(hidden_unit, device)
        parameters = sum(p.numel() for p in model.parameters())

        # Set the optimizer and criterion 
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        # Train and evaluate the model
        train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(train_dataloader, test_dataloader,
                                                                              model, optimizer, criterion, hidden_unit)
