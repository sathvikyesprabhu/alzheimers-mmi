import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DNN, self).__init__()
        # my network is composed of only affine layers
        # self.bn = nn.BatchNorm1d(in_channels)
        self.fc1 = nn.Linear(in_channels, 10)
        self.fc1_drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        # x = self.bn(x)
        # x = F.sigmoid(self.fc1(x))
        # x = F.sigmoid(self.fc2(x))
        x = ((torch.sigmoid(self.fc1(x))))
        x = self.fc1_drop(x)
        x = ((torch.sigmoid(self.fc2(x))))
        return x