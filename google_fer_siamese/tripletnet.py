import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class FECNet(nn.Module):
    def __init__(self, embeddingnet):
        super(FECNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        dist_c = F.pairwise_distance(embedded_y, embedded_z, 2)
        return dist_a, dist_b, dist_c, embedded_x, embedded_y, embedded_z


class EmbeddNet(nn.Module):
    def __init__(self):
        super(EmbeddNet, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2')
        self.resnet.requires_grad = False
        self.fc1 = nn.Linear(512, 16)
        self.fc1.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(8820, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 8820)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)
