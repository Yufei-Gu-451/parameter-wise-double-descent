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
hidden_units = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150, 200, 400, 700, 1000]
n_epochs = 6000
momentum = 0.95
learning_rate = 0.01
lr_decay_rate = 0.9
sample_size = 4000

if weight_reuse:
    directory = "assets/mnist/weight-reuse-case/epoch=%d-2" % n_epochs
else:
    directory = "assets/mnist/standard-case/epoch=%d-2" % n_epochs

output_file = os.path.join(directory, "epoch=%d.txt" % n_epochs)
checkpoint_path = os.path.join(directory, "ckpt")

if not os.path.isdir(directory):
    os.mkdir(directory)
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


#------------------------------------------------------------------------------------------

class simple_FC(nn.Module):
    def __init__(self, n_hidden):
        super(simple_FC, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n_hidden),
            nn.ReLU()
        )
        self.classifier = nn.Linear(n_hidden, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# Return the trainloader and testloader of MINST
def get_train_and_test_dataloader():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.MNIST(root='./data', train=True,
                                            download=True, transform = transform_train)
    trainset_sub = torch.utils.data.Subset(trainset, indices = np.arange(4000))
    trainloader = torch.utils.data.DataLoader(trainset_sub, batch_size=128,
                                            shuffle=True, num_workers=4)

    testset = datasets.MNIST(root='./data', train=False,
                                        download=True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=4)

    print('Load MINST dataset success;')

    return trainloader, testloader


# Set the neural network model to be used
def get_model(hidden_unit):
    model = simple_FC(hidden_unit)
    model = model.to(device)

    if hidden_unit == 1:
        torch.nn.init.xavier_uniform_(model.features[1].weight, gain=1.0)
        torch.nn.init.xavier_uniform_(model.classifier.weight, gain=1.0)
    #elif hidden_unit > 30:
    #    torch.nn.init.normal_(model.features[1].weight, mean=0.0, std=0.1)
    #    torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.1)
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


# Model Training
def train(trainloader, model, optimizer, criterion):
    model.train()
    cumulative_loss, correct, total = 0.0, 0, 0

    for inputs, labels in trainloader:
        # Calculate the training loss
        labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Calculate the Testing Loss
        cumulative_loss += loss.item()
        # Calculate the Training Accuracy
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
        for inputs, labels in testloader:
            # Calculate the testing loss
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Calculate the Testing Loss
            cumulative_loss += loss.item()
            # Calculate the Testing Accuracy
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
    while epoch < n_epochs:
        # Perform weight decay before the interpolation threshold
        # LR decay by lr_decay_rate percent after every `500` epochs
        if epoch > 1 and epoch % 500 == 1 and lr_decay:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * lr_decay_rate

        # Train the model
        model, train_loss, train_acc = train(trainloader, model, optimizer, criterion)

        # Print the status of current training and testing outcome
        epoch += 1
        print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f ; LR : %.3f" 
                  % (epoch, train_loss, train_acc, optimizer.param_groups[0]['lr']))

        if hidden_unit < 30 and train_acc == 1:
            break

    # Evaluate the model
    test_loss, test_acc = test(testloader, model)

    state = {
        'net': model.state_dict(),
        'acc': test_acc,
        'epoch': epoch,
    }
    torch.save(state, os.path.join(checkpoint_path, 'Simple_FC_%d.pth'%hidden_unit))
    print("Torch saved successfully！");

    return train_loss, train_acc, test_loss, test_acc


if __name__ == '__main__':
    # Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device : ', device)

    # Get the training and testing data of specific sample size
    trainloader, testloader = get_train_and_test_dataloader()

    # Main Training Unit
    for hidden_unit in hidden_units:
        # Generate the model with specific number of hidden_unit
        model = get_model(hidden_unit)
        parameters = sum(p.numel() for p in model.parameters())

        # Set the optimizer and criterion 
        optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=learning_rate)
        criterion = torch.nn.MSELoss()

        # Train and evalute the model
        train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(trainloader, testloader, \
                                    model, optimizer, criterion, hidden_unit)

        # Print training and evaluation outcome
        print("\nHidden Neurons : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; Test Acc : %.3f\n\n" \
                % (hidden_unit, parameters, train_loss, train_acc, test_loss, test_acc))

        # Write the training and evaluation output to file
        f = open(output_file, "a")
        f.write("Hidden Neurons : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; Test Acc : %.3f\n" \
                % (hidden_unit, parameters, train_loss, train_acc, test_loss, test_acc))
        f.close()