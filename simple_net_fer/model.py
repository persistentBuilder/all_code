import torch
import torch.nn as nn
from torch.nn import functional as F


class simpleNet(nn.Module):
    def __init__(self, num_classes):
        super(simpleNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 50, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(50)
        self.conv1_drop = nn.Dropout2d(p=0.1)

        self.conv2 = nn.Conv2d(50, 100, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(100)
        self.conv2_drop = nn.Dropout2d(p=0.1)

        self.conv3 = nn.Conv2d(100, 150, kernel_size=5)
        self.conv3_bn = nn.BatchNorm2d(150)
        self.conv3_drop = nn.Dropout2d(p=0.2)

        # fully connected layers
        self.fc1 = nn.Linear(9600, 300)
        self.fc1_bn = nn.BatchNorm1d(300)
        self.fc1_drop = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(300, 300)
        self.fc2_bn = nn.BatchNorm1d(300)
        self.fc2_drop = nn.Dropout(p=0.4)

        self.fc3 = nn.Linear(300, self.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop((self.conv1(x))), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop((self.conv2(x))), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop((self.conv3(x))), 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1_drop((self.fc1(x))))
        # x = F.relu(self.fc1_drop(self.fc1_bn(self.fc1(x))))
        x = F.relu(self.fc2_drop((self.fc2(x))))
        # x = F.relu(self.fc2_drop(self.fc2_bn(self.fc2(x))))
        x = self.fc3(x)

        return x
