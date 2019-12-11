import torch
import torch.nn as nn
from torch.nn import functional as F
import datasets
from data_loader import create
import argparse
from model import simpleNet
import random

parser = argparse.ArgumentParser("Branch Loss Example")
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--use-gpu', type=bool, default=False)
parser.add_argument('--include-neutral', type=bool, default=False)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
args = parser.parse_args()

lr_model = args.lr-model
epochs = args.max-epoch
use_gpu = args.use-gpu
batch_size = args.batch-size
include_neutral = args.include-neutral

num_classes = 8 if include_neutral else 7


