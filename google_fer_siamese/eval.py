import operator
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate



def pred_accurate(dist12, dist23, dist13):
    if dist12 < dist23 and dist12 < dist13:
        return True
    else:
        return False

def triplet_loss(dista, distb, margin):
    return max(0, dista-distb+margin)


def compute_loss(embed_1, embed_2, embed_3, margin):
    dist_12 = distance(embed_1, embed_2)
    dist_23 = distance(embed_2, embed_3)
    dist_13 = distance(embed_1, embed_3)

    acc = pred_accurate(dist_12, dist_23, dist_13)

    loss = triplet_loss(dist_12, dist_13, margin) + triplet_loss(dist_12, dist_23, margin)

    return loss, acc