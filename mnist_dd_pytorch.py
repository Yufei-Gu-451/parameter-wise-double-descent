import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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


def get_train_and_test_dataloader(sample_size):
    my_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    subset = list(range(0, sample_size, 1))
    trainset = datasets.MNIST(root='./data', train=True,
                                            download=True, transform = my_transform)
    trainset_1 = torch.utils.data.Subset(trainset, subset)
    trainloader = torch.utils.data.DataLoader(trainset_1, batch_size=128,
                                            shuffle=True, num_workers=4)

    testset = datasets.MNIST(root='./data', train=False,
                                        download=True, transform = my_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=4)

    print('Load MINST dataset success')

    return trainloader, testloader


def train_and_evaluate_model(trainloader, testloader, model, hidden_unit, optimizer, criterion, n_epochs, lr_decay_rate):
    train_losses, test_losses = [], []
    train_acc, test_acc, epoch = 0, 0, 0
    
    while epoch <= n_epochs and train_acc < 0.99:
        if epoch >= 1:
            print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; Test Acc : %.3f ; LR : %.3f" 
                  % (epoch, train_losses[-1], train_acc, test_losses[-1], test_acc, optimizer.param_groups[0]['lr']))
        epoch += 1

        if hidden_unit <= 50 and epoch % 100 == 1:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * lr_decay_rate
        elif hidden_unit > 50:
            optimizer.param_groups[0]['lr'] = 0.01

        model.train()
        
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

        train_losses.append(train_loss/len(trainloader))
        train_acc = correct/total
        
        
        model.eval()
        
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()
            
        test_losses.append(test_loss/len(testloader))
        test_acc = correct/total

    return train_losses[-1], train_acc, test_losses[-1], test_acc


if __name__ == '__main__':
    # Training Settings
    hidden_units = [55, 60]
    n_epochs = 1000
    momentum = 0.95
    learning_rate = 0.01
    lr_decay_rate = 0.9
    sample_size = 4000

    # Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device :', device)

    trainloader, testloader = get_train_and_test_dataloader(sample_size)

    # Main Training Unit
    for hidden_unit in hidden_units:
        model = simple_FC(hidden_unit)

        if hidden_unit == 1:
            torch.nn.init.xavier_uniform_(model.features[1].weight, gain=1.0)
            torch.nn.init.xavier_uniform_(model.classifier.weight, gain=1.0)
        else:
            torch.nn.init.normal_(model.features[1].weight, mean=0.0, std=0.1)
            torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.1)

        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=learning_rate)
        criterion = torch.nn.MSELoss()

        train_loss, train_acc, test_loss, test_acc = train_and_evaluate_model(trainloader, testloader, \
                                    model, hidden_unit, optimizer, criterion, n_epochs, lr_decay_rate)
        print("\nHidden Neurons : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; Test Acc : %.3f\n\n" \
                % (hidden_unit, train_loss, train_acc, test_loss, test_acc))

        f = open("plots/epoch=1000/epoch=1000.txt", "a")
        f.write("Hidden Neurons : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; Test Acc : %.3f\n" \
                % (hidden_unit, train_loss, train_acc, test_loss, test_acc))
        f.close()