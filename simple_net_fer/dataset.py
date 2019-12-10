from __future__ import print_function, division
import os, io
import torch
import numpy as np
import PIL
from PIL import Image
import cv2
import yaml
import math
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from skimage.transform import resize


class cohnKanadeDataLoad(Dataset):

    def __init__(self, path_file, train_flag=True, include_neutral=False, transform=None):

        f = open(path_file, "r")
        self.all_lines = f.readlines()
        self.transform = transform
        self.train_flag = train_flag
        self.include_neutral = include_neutral
        self.all_gt = []
        self.imgs = []
        self.gt = []
        self.num_classes = 8 if self.include_neutral else 7
        self.test_flag = not train_flag

        distinct_persons = set()
        for line in self.all_lines:
            line = line[:-1]
            person_set = line.rsplit("/",1)[-1].split("_")[0]
            distinct_persons.add(person_set)

        train_set, test_set = get_train_and_test_set(distinct_persons)

        cnt = 0
        for line in self.all_lines:
            line = line[:-1]
            person_set = line.rsplit("/", 1)[-1].split("_")[0]
            if (self.train_flag and person_set in train_set) or (self.test_flag and person_set in test_set):
                seq_num = int(line.rsplit("/",1)[-1].rsplit("_",1)[-1].split(".")[0])
                if not include_neutral and seq_num == 1:
                    cnt = cnt + 1
                    continue

                img = get_image_from_path(line)
                self.imgs.append(img)
                ground_truth = get_gt_from_path(line, seq_num)
                self.gt.append(ground_truth)
            cnt = cnt + 1
        f.close()


    def get_train_and_test_set(self, distinct_persons, test_set_ratio=0.1):
         total_persons = len(distinct_persons)
         train_length = int(total_persons * (1-test_set_ratio))
         train_set = []
         test_set = []
         cnt = 0
         for person in distinct_persons:
             if cnt < train_length:
                 train_set.append(person)
             else:
                 test_set.append(person)
             cnt = cnt + 1
         return train_set, test_set

    def get_image_from_path(self, path):
        img = io.imread(path)
        return img

    def get_gt_from_path(self, path, seq_num):
        if seq_num == 1:
            return 8

        g = open(gt_file, "r")
        gt_lines = g.readlines()

        g.close()
        return

    def __getitem__(self, index):
        img = self.imgs[index]
        label = to_categorical(self.gt[index], self.num_classes)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def to_categorical(self, y, num_classes):
        return np.eye(num_classes, dtype='uint8')[y]

# train_dataset = ucfDataLoad('/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucfTrainTestlist/testlist01.txt',0)
# train_dataset = retDataset('/home/aryaman.g/pyTorchLearn/biwiTrain.txt',1)
# train_dataset = biwiDataset('/ssd_scratch/cvit/aryaman.g/biwiHighResHM/allHM',1)
# print(train_dataset.__getitem__(2))