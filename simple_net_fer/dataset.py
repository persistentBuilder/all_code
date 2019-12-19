from __future__ import print_function, division
import os, io
import torch
import numpy as np
import PIL
from PIL import Image
import yaml
import math
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import cv2

class CohnKanadeDataLoad(Dataset):

    def __init__(self, path_file, train_flag=True, include_neutral=False, transform=None):

        f = open(path_file, "r")
        self.all_lines = f.readlines()
        self.transform = transform
        self.train_flag = train_flag
        self.include_neutral = include_neutral
        self.all_gt = []
        self.imgs = []
        self.image_resize_height = 224
        self.image_resize_width = 224
        self.gt = []
        self.num_classes = 8 if self.include_neutral else 7
        self.test_flag = not train_flag

        distinct_persons = set()
        for line in self.all_lines:
            line = line[:-1]
            person_set = line.rsplit("/", 1)[-1].split("_")[0]
            distinct_persons.add(person_set)

        train_set, test_set = self.get_train_and_test_set(distinct_persons)

        cnt = 0
        for line in self.all_lines:
            line = line[:-1]
            person_set = line.rsplit("/", 1)[-1].split("_")[0]
            if (self.train_flag and person_set in train_set) or (self.test_flag and person_set in test_set):
                seq_num = int(line.rsplit("/",1)[-1].rsplit("_",1)[-1].split(".")[0])
                if not include_neutral and seq_num == 1:
                    cnt = cnt + 1
                    continue

                img = self.get_image_from_path(line)
                self.imgs.append(img)
                ground_truth = self.get_gt_from_path(line, seq_num)
                self.gt.append(ground_truth)
            cnt = cnt + 1
        f.close()

    def resize_face_image(self, img):
        return cv2.resize(img, (self.image_resize_width, self.image_resize_height), interpolation=cv2.INTER_CUBIC)

    def get_train_and_test_set(self, distinct_persons, test_set_ratio=0.1):
        total_persons = len(distinct_persons)
        train_length = int(total_persons * (1 - test_set_ratio))
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

        #img = Image.open(path)
        img = cv2.imread(path)
        return img

    def get_gt_from_path(self, path, seq_num):
        if seq_num == 1:
            return 8

        components = path.rsplit("/", 4)
        gt_path = components[0] + '/' + 'Emotion/' + components[2] + '/' + components[3] + '/'
        gt_file_name = os.listdir(gt_path)[0]
        g = open(gt_path + gt_file_name, "r")
        gt_lines = g.readlines()
        ground_truth = int(float(gt_lines[0][:-1]))
        g.close()
        return ground_truth

    def __getitem__(self, index):

        img = self.resize_face_image(self.imgs[index])
        if self.transform:
            img = self.transform(img)
        #label = self.to_categorical(self.gt[index]-1)
        return img, self.gt[index]-1

    def __len__(self):
        return 10
        #return len(self.imgs)

    def to_categorical(self, y):
        return np.eye(self.num_classes, dtype='int')[y]
