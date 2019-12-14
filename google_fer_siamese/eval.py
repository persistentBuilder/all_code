import operator
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F


def pred_accurate(dist_xy, dist_yz, dist_xz):
    return np.logical_and(dist_xy < dist_yz, dist_xy < dist_yz).sum()

def triplet_loss(dista, distb, margin):
    return max(0, dista-distb+margin)


def compute_loss(embedded_x, embedded_y, embedded_z, margin):
    dist_xy = F.pairwise_distance(embedded_x, embedded_y, 2)
    dist_yz = F.pairwise_distance(embedded_y, embedded_z, 2)
    dist_xz = F.pairwise_distance(embedded_x, embedded_z, 2)

    acc = pred_accurate(dist_xy.detach().numpy(), dist_yz.detach().numpy(), dist_xz.detach().numpy())

    loss = triplet_loss(dist_xy, dist_xz, margin) + triplet_loss(dist_xy, dist_yz, margin)

    return loss, acc