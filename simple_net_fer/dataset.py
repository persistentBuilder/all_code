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
#from mtcnn import MTCNN
from numpy import load
import pandas as pd
import time
from AUmaps import *


class CohnKanadeDataLoad(Dataset):

    def __init__(self, path_file, train_flag=True, include_neutral=False, input_type="fuse", transform=None,
                 divide_distinct_persons=False):

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
        self.face_detector = MTCNN()
        self.input_type = input_type
        self.ddp = divide_distinct_persons
        self.heatmap_detector = AUdetector('shape_predictor_68_face_landmarks.dat', enable_cuda=torch.cuda.is_available())

        distinct_persons = set()
        for line in self.all_lines:
            line = line[:-1]
            if not self.ddp:
                person_set = line.rsplit("/", 1)[-1].rsplit("_", 2)[0]
            else:
                person_set = line.rsplit("/", 1)[-1].split("_")[0]
            distinct_persons.add(person_set)

        train_set, test_set = self.get_train_and_test_set(distinct_persons)

        cnt = 0
        for line in self.all_lines:
            line = line[:-1]
            base = '/home/aryaman.g/projects/FER/dataset/'
            line = base + line.split("/", 5)[-1]
            person_set = line.rsplit("/", 1)[-1].split("_")[0]
            if (self.train_flag and person_set in train_set) or (self.test_flag and person_set in test_set):
                seq_num = int(line.rsplit("/",1)[-1].rsplit("_",1)[-1].split(".")[0])
                if not include_neutral and seq_num == 1:
                    cnt = cnt + 1
                    continue

                ground_truth = self.get_gt_from_path(line, seq_num)
                heatmap_path = "heatmaps/" + line.rsplit("/", 1)[-1].split(".")[0] + '.npy'

                if self.input_type == "fuse":
                    try:
                        img = self.resize_face_image(self.get_image_from_path(line))
                        heatmap = self.resize_heatmap(self.get_au_heatmap(img))
                        fused_img = self.fuse_heatmap_into_img(img, heatmap)
                        if self.transform:
                            fused_img = self.transform(fused_img)
                        self.imgs.append(fused_img)
                        self.gt.append(ground_truth)
                    except:
                        continue
                elif self.input_type == "stack":
                    try:
                        img = self.resize_face_image(self.get_image_from_path(line))
                        heatmap = self.read_saved_heatmap(heatmap_path)
                        if self.transform:
                            img = self.transform(img)
                        self.imgs.append([img, heatmap])
                        self.gt.append(ground_truth)
                    except:
                        continue
                elif self.input_type == "only_heatmap":
                    heatmap = self.read_saved_heatmap(heatmap_path)
                    img = self.resize_face_image(self.get_image_from_path(line, without_face_crop=True))/255
                    img = np.stack([img[:, :, 0], img[:, :, 1], img[:, :, 2]], axis=0)
                    heatmap_and_img = np.concatenate((heatmap, img), axis=0)
                    self.imgs.append(heatmap_and_img)
                    self.gt.append(ground_truth)
                else:
                    try:
                        img = self.resize_face_image(self.get_image_from_path(line))
                        self.imgs.append(img)
                        self.gt.append(ground_truth)
                    except:
                        continue
            cnt = cnt + 1
        f.close()

    def fuse_heatmap_into_img(self, img, heatmap):
        red_channel = np.zeros([img.shape[0], img.shape[1]])
        green_channel = np.zeros([img.shape[0], img.shape[1]])
        blue_channel = np.zeros([img.shape[0], img.shape[1]])
        for i in range(0, heatmap.shape[0]):
            red_channel = np.add(red_channel, np.multiply(heatmap[i, :, :], img[:, :, 0]))
            green_channel = np.add(green_channel, np.multiply(heatmap[i, :, :], img[:, :, 1]))
            blue_channel = np.add(blue_channel, np.multiply(heatmap[i, :, :], img[:, :, 2]))
        fused_image = np.stack([red_channel, green_channel, blue_channel], axis=0)
        return fused_image

    def get_au_heatmap(self, img):
        dum, heatmap, dum_img = AUdetector.detectAU(img)
        return heatmap

    def read_saved_heatmap(self, heatmap_path):
        return load(heatmap_path)

    def resize_heatmap(self, heatmap):
        pass

    def resize_face_image(self, img):
        return cv2.resize(img, (self.image_resize_width, self.image_resize_height), interpolation=cv2.INTER_CUBIC)

    def get_train_and_test_set(self, distinct_persons, test_set_ratio=0.1):
        total_persons = len(distinct_persons)
        print("total distinct sets: ", total_persons)
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

    def get_face_from_img(self, img):
        results = self.face_detector.detect_faces(img)
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face_img = img[y1:y2, x1:x2]
        return face_img

    def get_image_from_path(self, path, without_face_crop=False):
        #img = Image.open(path)
        if without_face_crop:
            return cv2.imread(path)
        face_img = self.get_face_from_img(cv2.imread(path))
        return face_img

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

        img = self.imgs[index]
        if self.input_type == "only_heatmap":
            img = torch.Tensor(img)
        if self.transform and self.input_type == "only_img":
            img = self.transform(img)
        #label = self.to_categorical(self.gt[index]-1)
        return img, self.gt[index]-1

    def __len__(self):
        return len(self.imgs)

    def to_categorical(self, y):
        return np.eye(self.num_classes, dtype='int')[y]


class AffectNetDataset(Dataset):

    def __init__(self, path, train_flag=True, base_path='/ssd_scratch/cvit/aryaman.g/affectnet',
                 transform=None, include_neutral=True, use_heatmap=-1):

        self.path_imgs = []
        self.ground_truth = []
        self.train_flag = train_flag
        self.include_neutral = include_neutral
        self.face_location = []
        self.base_path = base_path
        self.image_resize_height = 224
        self.image_resize_width = 224
        self.transform = transform
        self.num_classes = 8 if self.include_neutral else 7
        self.heatmap_detector = AUdetector('shape_predictor_68_face_landmarks.dat', enable_cuda=torch.cuda.is_available())
        self.use_heatmap = use_heatmap


        data = pd.read_csv(self.base_path + '/' + path)
        data['subDirectory_filePath'] = data['subDirectory_filePath'].apply(lambda x: x.split("/")[-1])
        data = data.sort_values(by=['subDirectory_filePath'])
        all_file_pd_series = data['subDirectory_filePath']
        all_file_paths = np.array(all_file_pd_series)
        set_of_path = set(all_file_paths)
        all_face_locations = list([list(data['face_x']), list(data['face_y']), list(data['face_width']),
                                   list(data['face_height'])])
        all_face_locations = np.transpose(all_face_locations)
        all_ground_truth = list(data['expression'])
        lf = os.listdir(self.base_path)
        useful_dir = []

        for l in lf:
            if len(l) >= 18 and l[:18] == "Manually_Annotated" and l[-6:-2] == "part":
                useful_dir.append(l)

        begin_time = time.time()
        series_search_time = 0
        count=0
        for dir_name in useful_dir:
            dir_path = self.base_path + '/' + dir_name
            images_list = os.listdir(dir_path)
            for image in images_list:
                if image in set_of_path:
                    ss_begin = time.time()
                    idx = np.searchsorted(all_file_paths, image)
                    series_search_time += time.time()-ss_begin
                    if all_ground_truth[idx] >= 8:
                        continue
                    self.path_imgs.append(dir_path + '/' + all_file_paths[idx])
                    self.face_location.append(all_face_locations[idx])
                    self.ground_truth.append(all_ground_truth[idx])
                    count += 1
                    if count % 1000 == 0:
                        print(count, time.time()-begin_time, series_search_time)

    def resize_face_image(self, img):
        return cv2.resize(img, (self.image_resize_width, self.image_resize_height), interpolation=cv2.INTER_CUBIC)

    def get_image_from_path(self, path, face_location):
        x1, y1, width, height = face_location
        x2, y2 = x1 + width, y1 + height
        img = cv2.imread(path)
        face_img = img[y1:y2, x1:x2]
        return face_img

    def get_au_heatmap(self, img):
        dum, heatmap, dum_img = AUdetector.detectAU(img)
        return heatmap

    def get_heatmap_modified_image(self, heatmap, img):
        modified_img = img
        return modified_img

    def __getitem__(self, index):
        img = self.resize_face_image(self.get_image_from_path(self.path_imgs[index], self.face_location[index]))
        if self.transform:
            img = self.transform(img)
        if self.use_heatmap == -1:
            return img, self.ground_truth[index]
        elif self.use_heatmap == 1:
            heatmap = self.get_au_heatmap(img)
            return heatmap, self.ground_truth[index]
        elif self.use_heatmap == 2:
            heatmap = self.get_au_heatmap(img)
            return [img, heatmap], self.ground_truth[index]
        elif self.use_heatmap == 3:
            heatmap = self.get_au_heatmap(img)
            mod_img = self.get_heatmap_modified_image(heatmap, img)
            return mod_img, self.ground_truth[index]

    def __len__(self):
        return len(self.path_imgs)