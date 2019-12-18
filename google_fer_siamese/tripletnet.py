import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from densenet import *


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
        comp_resnet_pretrained = InceptionResnetV1(pretrained='vggface2')
        modules = list(comp_resnet_pretrained.children())[:-8]
        self.resnet = nn.Sequential(*modules)
        self.resnet.requires_grad = False
        self.densenet = DenseNet(growthRate=64, depth=9, input_channels=896, reduction=0.5, bottleneck=True, nClasses=1)
        self.densenet.requires_grad = True
        self.dense_drop = nn.Dropout(p=0.5)
        self.dense_drop.requires_grad = True
        self.fc1 = nn.Linear(288, 16)
        self.fc1.requires_grad = True
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc1_drop.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        x = self.dense_drop(self.densenet(x))
        x = self.fc1_drop(self.fc1(x))
        x = F.normalize(x, p=2, dim=1)
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
