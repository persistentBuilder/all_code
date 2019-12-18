from dataset import SiameseGoogleFer
import time
from torchvision import datasets, transforms
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import argparse

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

def fetch_dataset(current_division, divisions, path):
    SiameseGoogleFer(path, train_flag=False, transform=transform,  divisions=divisions,
                     current_division=current_division, load_in_memory=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Parallel download dataset")
    parser.add_argument('--train-fetch', type=int, default=1)
    args = parser.parse_args()
    divisions = 50
    processes = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    if args.train-fetch == 1:
        path = "data/faceexp-comparison-data-train-public.csv"
    else:
        path = "data/faceexp-comparison-data-test-public.csv"

    for current_division in range(0, divisions):
        p = Process(target=fetch_dataset, args=(current_division, divisions, path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


