import torch
import torch.nn as nn
from torch.nn import functional as F


class simpleNet(nn.Module):
    def __init__(self, num_classes):
        super(simpleNet, self).__init__()
        self.num_classes = num_classes
        self.branches = int(self.num_classes/2)
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.fc1 = []
        self.fc2 = []
        self.prelu_fc1 = []
        for branch in range(0, self.branches):
            self.fc1.append(nn.Linear(128*3*3, 32))
            self.prelu_fc1.append(nn.PReLU())
            self.fc2.append(nn.Linear(32, self.num_classes))

    def forward(self, x):

        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))  
        x = F.max_pool2d(x, 2)
        x_size = x.size()
        x = x.view(-1, x_size[1]*x_size[2]*x_size[3])

        return out
