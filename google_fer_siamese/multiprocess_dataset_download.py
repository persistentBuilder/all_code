from dataset import SiameseGoogleFer
import time
from torchvision import datasets, transforms
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from multiprocessing import Process
import argparse

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

def fetch_dataset(current_division, divisions, path, train_flag):
    x = SiameseGoogleFer(path=path, train_flag=train_flag, transform=transform,  divisions=divisions,
                         current_division=current_division, load_in_memory=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Parallel download dataset")
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--divisions', type=int, default=50)
    args = parser.parse_args()
    divisions = args.divisions
    if args.train == 1:
        path = "data/faceexp-comparison-data-train-public.csv"
        train_flag = True
    else:
        path = "data/faceexp-comparison-data-test-public.csv"
        train_flag = False
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    if divisions == 1:
        fetch_dataset(current_division=0, divisions=divisions, path=path, train_flag=train_flag)
    else:
        processes = []
        for current_division in range(0, divisions):
            p = Process(target=fetch_dataset, args=(current_division, divisions, path, train_flag))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
