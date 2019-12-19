import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms

from dataset import CohnKanadeDataLoad


class CKPLUS(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        pin_memory = True if use_gpu else False

        path_file = "relevant_images_paths.txt"
        trainset = CohnKanadeDataLoad(path_file=path_file, train_flag=True, include_neutral=False, transform=transform)
        testset = CohnKanadeDataLoad(path_file=path_file, train_flag=False, include_neutral=False, transform=transform)

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
        self.num_classes = trainset.num_classes

    # def __iter__(self):
    #     for i, data in enumerate(self.dataloader):
    #         if i * self.opt.batch_size >= self.opt.max_dataset_size:
    #             break
    #         yield data

__factory = {
    'ck+': CKPLUS,
}


def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers)