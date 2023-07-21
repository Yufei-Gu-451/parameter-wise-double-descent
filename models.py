import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))

class FiveLayerCNN(nn.Module):
    def __init__(self, k):
        self.n_hidden_units = k

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

    def forward(self, x, path='all'):
        if path == 'all':
            x = self.forward_1(x)
            x = self.forward_2(x)
            x = self.forward_3(x)
            x = self.forward_4(x)
            x = self.forward_5(x)
        elif path == 'half1':
            x = self.forward_1(x)
            x = self.forward_2(x)
            x = self.forward_3(x)
            x = self.forward_4(x)
        elif path == 'half2':
            x = self.forward_5(x)
        else:
            raise NotImplementedError

        return x

    def forward_1(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        return x

    def forward_2(self, x):
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        return x

    def forward_3(self, x):
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        return x

    def forward_4(self, x):
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        return x

    def forward_5(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Simple_FC(nn.Module):
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units

        super(Simple_FC, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n_hidden_units),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n_hidden_units, 10)

        if self.n_hidden_units == 1:
            torch.nn.init.xavier_uniform_(self.features[1].weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)
        else:
            torch.nn.init.normal_(self.features[1].weight, mean=0.0, std=0.1)
            torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.1)

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