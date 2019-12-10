import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms

from dataset import cohnKanadeDataLoad


class CKPLUS(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        #pin_memory = True if use_gpu else False
        pin_memory =  False

        path_file = "relevant_images_paths.txt"
        trainset = cohnKanadeDataLoad(path_file=path_file, train_flag=True, include_neutral=False)
        testset = cohnKanadeDataLoad(path_file=path_file, train_flag=False, include_neutral=False)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )


        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

__factory = {
    'ck+': CKPLUS,
}


def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers)