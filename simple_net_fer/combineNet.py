from __future__ import print_function, division
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from heatmapNet import  heatmapNet
from extendNet import extendNet

class combineNet(nn.Module):
    def __init__(self, num_classes):
        super(combineNet, self).__init__()
        # conv layers
        self.num_classes = num_classes
        self.heatmap_net = heatmapNet(self.num_classes, compute_embedding=True)
        self.extend_net = extendNet(self.num_classes, compute_embedding=True)
        self.dense_drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(588, self.num_classes)
        self.fc1_drop = nn.Dropout(p=0.4)

    def forward(self, x, y):
        img_embedding = self.extend_net(x)
        heatmap_embedding = self.heatmap_net(y)
        out = torch.cat([img_embedding, heatmap_embedding], dim=1)
        out = self.fc1_drop(self.fc1(out))
        return out

