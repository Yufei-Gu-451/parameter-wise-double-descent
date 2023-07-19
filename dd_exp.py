import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
# from sklearn.manifold import TSNE
import csv
import os

import models
import datasets


# ------------------------------------------------------------------------------------------


# Training Settings
DATASET = 'CIFAR-10'

N_EPOCHS = 4000
N_SAMPLES = 4000
BATCH_SIZE = 64

TEST_GROUP = 0
TEST_NUMBERS = [0]

label_noise_ratio = 0.2

learning_rate_decay = True
learning_rate = 0.05

save_model = True

#tSNE_Visualization = False

# ------------------------------------------------------------------------------------------


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# Return the train_dataloader and test_dataloader of MINST
def get_train_and_test_dataloader(dataset_path):
    train_dataset = datasets.save_and_create_training_set(DATASET, sample_size=N_SAMPLES,
                        label_noise_ratio=label_noise_ratio, dataset_path=dataset_path)

    train_dataloader = DataLoaderX(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    test_dataset = datasets.get_test_dataset(DATASET=DATASET)

    test_dataloader = DataLoaderX(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f'Load {DATASET} dataset success;')

    return train_dataloader, test_dataloader


# ------------------------------------------------------------------------------------------



# Set the neural network model to be used
def get_model(hidden_unit, device):
    if DATASET == 'MNIST':
        model = models.Simple_FC(hidden_unit)
        model.initialize()
    elif DATASET == 'CIFAR-10':
        model = models.FiveLayerCNN(hidden_unit)

    model = model.to(device)

    print("Model with %d hidden neurons successfully generated;" % hidden_unit)

    print('Number of parameters: %d' % sum(p.numel() for p in model.parameters()))

    return model


# ------------------------------------------------------------------------------------------


# Model Training
def train(model, device, train_dataloader, optimizer, criterion):
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
def test(model, device, test_dataloader):
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


# ------------------------------------------------------------------------------------------



'''
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
'''

def model_save(model, epoch, test_accuracy, checkpoint_path):
    state = {
        'net': model.state_dict(),
        'acc': test_accuracy,
        'epoch': epoch,
    }
    torch.save(state, os.path.join(checkpoint_path, 'Simple_FC_%d.pth' % hidden_unit))
    print("Torch saved successfully!")


def status_save(n_hidden_units, epoch, parameters, train_loss, train_acc, test_loss, test_acc, dictionary_path):
    print("Hidden Neurons : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; dataset Loss : %f ; "
          "dataset Acc : %.3f\n" % (n_hidden_units, parameters, train_loss, train_acc, test_loss, test_acc))

    print('Writing to a csv file...')
    dictionary = {'Hidden Neurons': hidden_unit, 'Epoch': epoch, 'Parameters': parameters, 'Train Loss': train_loss,
                  'Train Accuracy': train_acc, 'dataset Loss': test_loss, 'dataset Accuracy': test_acc}

    with open(dictionary_path, "a", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())

        # Write the data rows
        writer.writerow(dictionary)
        print('Done writing to a csv file\n')


# Train and Evalute the model
def train_and_evaluate_model(model, device, trainloader, testloader, optimizer, criterion, dictionary_path, checkpoint_path):
    train_loss, train_acc, test_loss, test_acc, epoch = 0.0, 0.0, 0.0, 0.0, 0
    parameters = sum(p.numel() for p in model.parameters())
    n_hidden_units = model.n_hidden_units

    for epoch in range(1, N_EPOCHS + 1):
        model, train_loss, train_acc = train(model, device, trainloader, optimizer, criterion)
        print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f" % (epoch, train_loss, train_acc))

        if epoch % 100 == 0:
            test_loss, test_acc = test(model, device, testloader)

            status_save(n_hidden_units, epoch, parameters, train_loss, train_acc, test_loss, test_acc,
                        dictionary_path=dictionary_path)

            if learning_rate_decay:
                optimizer.param_groups[0]['lr'] = learning_rate / pow(1 + epoch // 50, 0.5)
                print("Learning Rate : ", optimizer.param_groups[0]['lr'])

    '''
    if tSNE_Visualization:
        model_t_sne(model, trainloader, hidden_unit)
    '''

    if save_model:
        model_save(model, test_acc, epoch, checkpoint_path=checkpoint_path)

    return


if __name__ == '__main__':
    # Initialization of used device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    # Initialization of hidden units
    if DATASET == 'MNIST':
        hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80,
                        90, 100, 120, 150, 200, 400, 600, 800, 1000]
    elif DATASET == 'CIFAR-10':
        hidden_units = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]

    # Main Program
    for test_number in TEST_NUMBERS:
        # Define the roots and paths
        directory = f"assets/{DATASET}/N=%d-3d/TEST-%d/epoch=%d-noise-%d-model-%d" \
                    % (N_SAMPLES, TEST_GROUP, N_EPOCHS, label_noise_ratio * 100, test_number)

        dataset_path = os.path.join(directory, 'dataset')
        dictionary_path = os.path.join(directory, "dictionary.csv")
        checkpoint_path = os.path.join(directory, "ckpt")
        tsne_path = os.path.join(directory, "t-SNE")

        if not os.path.isdir(directory):
            os.mkdir(directory)
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)
        if save_model and not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        '''
        if tSNE_Visualization and not os.path.isdir(tsne_path):
            os.mkdir(tsne_path)
        '''

        # Initialize Status Dictionary
        dictionary = {'Hidden Neurons': 0, 'Epoch': 0, 'Parameters': 0, 'Train Loss': 0,
                      'Train Accuracy': 0, 'Test Loss': 0, 'Test Accuracy': 0}

        with open(dictionary_path, "a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
            writer.writeheader()

        # Get the training and testing data of specific sample size
        train_dataloader, test_dataloader = get_train_and_test_dataloader(dataset_path=dataset_path)

        # Main Training Unit
        for hidden_unit in hidden_units:
            # Generate the model with specific number of hidden_unit
            model = get_model(hidden_unit, device)

            # Set the optimizer and criterion
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            criterion = torch.nn.CrossEntropyLoss()
            criterion = criterion.to(device)

            # Train and evaluate the model
            train_and_evaluate_model(model, device, train_dataloader, test_dataloader, optimizer, criterion,
                                        dictionary_path=dictionary_path, checkpoint_path=checkpoint_path)
