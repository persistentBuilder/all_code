import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from densenet import *


class extendNet(nn.Module):
    def __init__(self, num_classes):
        super(extendNet, self).__init__()
        self.num_classes = num_classes
        comp_resnet_pretrained = InceptionResnetV1(pretrained='vggface2')
        modules = list(comp_resnet_pretrained.children())[:-8]
        self.resnet = nn.Sequential(*modules)
        self.resnet.requires_grad = False
        self.densenet = DenseNet(growthRate=64, depth=9, input_channels=896, reduction=0.5, bottleneck=True, nClasses=1)
        self.densenet.requires_grad = True
        self.dense_drop = nn.Dropout(p=0.5)
        self.dense_drop.requires_grad = True
        self.fc1 = nn.Linear(288, self.num_classes)
        self.fc1.requires_grad = True
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc1_drop.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        x = self.dense_drop(self.densenet(x))
        x = self.fc1_drop(self.fc1(x))
        #x = F.normalize(x, p=2, dim=1)
        return x

