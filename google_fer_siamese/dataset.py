import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import requests
import torch
import cv2
from torchvision import transforms


class SiameseGoogleFer(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, path, train_flag=True, transform=None, write_only_face=True, divisions=1, current_division=0):

        self.f = open(path, "r")
        self.train_flag = train_flag
        self.write_only_face = write_only_face
        self.all_lines = self.f.readlines()
        self.divisions = divisions

        self.current_lines = self.get_lines(current_division)
        
        if self.train_flag:
            failed_path = "data/failed_read_train.txt"
        else:
            failed_path = "data/failed_read_test.txt"

        print(len(self.current_lines))
        self.all_triplets = []
        self.triplets = []
        self.image_resize_height = 224
        self.image_resize_width = 224
        self.transform = transform

        self.g = open(failed_path, "w")
        line_num = 0
        for line in self.current_lines:
            line_num += 1
            #print(line_num)
            line_components = line.split(",")
            url_1 = line_components[0][1:-1]
            url_2 = line_components[5][1:-1]
            url_3 = line_components[10][1:-1]
            try:
                img_1 = self.download_and_load_image(url_1, line_num)
                img_2 = self.download_and_load_image(url_2, line_num)
                img_3 = self.download_and_load_image(url_3, line_num)
            except:
                continue

            if img_1.shape[0] == self.image_resize_height and img_1.shape[1] == self.image_resize_width and \
                img_2.shape[0] == self.image_resize_height and img_2.shape[1] == self.image_resize_width and \
                img_3.shape[0] == self.image_resize_height and img_3.shape[1] == self.image_resize_width:
                face_image_1 = img_1
                face_image_2 = img_2
                face_image_3 = img_3
            else:
                face_image_1 = self.select_face_region(img_1, line_components[1:5], img_1.shape[0], img_1.shape[1])
                face_image_2 = self.select_face_region(img_2, line_components[6:10], img_2.shape[0], img_2.shape[1])
                face_image_3 = self.select_face_region(img_3, line_components[11:15], img_3.shape[0], img_3.shape[1])
                cv2.imwrite(self.get_path(url_1, line_num), self.resize_face_image(face_image_1))
                cv2.imwrite(self.get_path(url_2, line_num), self.resize_face_image(face_image_2))
                cv2.imwrite(self.get_path(url_3, line_num), self.resize_face_image(face_image_3))
            #to do detect face
            #if not (self.detect_face(face_image_1) and self.detect_face(face_image_2)
            #        and self.detect_face(face_image_3)):
            #    continue


            strong_flag, annotation = self.check_strong_annotation(line_components)
            if not strong_flag:
                continue
            if annotation == 1:
                self.triplets.append([face_image_2, face_image_3, face_image_1])
            elif annotation == 2:
                self.triplets.append([face_image_2, face_image_1, face_image_2])
            else:
                self.triplets.append([face_image_1, face_image_2, face_image_3])
        self.f.close()
        self.g.close()

    def get_lines(self, current_division):
        l = len(self.all_lines) 
        start = int((l/self.divisions)*current_division)
        end = int((l/self.divisions)*(current_division+1)) if current_division < self.divisions - 1 else l 
        return self.all_lines[start:end]

    def get_majority_element(self, arr):
        d = {}
        max_el, max_count = 0, 0
        for el in arr:
            if d.get(el) != None :
                d[el] = d[el] + 1
            else:
                d[el] = 1
            if d[el] > max_count:
                max_el = el
                max_count = d[el]
        return max_el, max_count

    def check_strong_annotation(self, line_components):

        votes = []
        for vote in range(17, len(line_components), 2):
            votes.append(int(line_components[vote]))
        annotation, count = self.get_majority_element(votes)
        if count >= int(len(votes) * (2 / 3)):
            return True, annotation
        else:
            return False, annotation

    def select_face_region(self, img, bounding_box, height, width):

        top_left_column = max(0, int(float(bounding_box[0])*width))
        bottom_right_column = min(width, int(float(bounding_box[1])*width))
        top_left_row = max(0, int(float(bounding_box[2])*height))
        bottom_right_row = min(height, int(float(bounding_box[3])*height))
        return img[top_left_row:bottom_right_row, top_left_column:bottom_right_column, :]

    def download_and_save_image_from_url(self, url, line_num):

        image_path = self.get_path(url, line_num)
        with open(image_path, 'wb') as handle:
            response = requests.get(url, stream=True)

            if not response.ok:
                #print(response)
                self.g.write(url+"\n")
                raise("url couldn't be downloaded")

            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)

        return self.saved_image_load(url, line_num)

    def get_path(self, url, line_num):
        image_name = url.split("/")[-1]
        line_str = str(line_num).zfill(8) + "_"
        base_path = "data/images/"
        if self.train_flag:
            base_directory = base_path + "train/"
        else:
            base_directory = base_path + "test/"
        image_path = base_directory + line_str + image_name
        return image_path

    def saved_image_load(self, url, line_num):
        image_path = self.get_path(url, line_num)
        try:
            return cv2.imread(image_path)
        except:
            return None

    def download_and_load_image(self, url, line_num):
        #check if not already downloaded
        saved_image = self.saved_image_load(url,line_num)
        if saved_image is None:
            saved_image = self.download_and_save_image_from_url(url, line_num)
        return saved_image

    def resize_face_image(self, img):
        return cv2.resize(img, (self.image_resize_width, self.image_resize_height), interpolation=cv2.INTER_CUBIC)

    def detect_face(self, img):
        pass

    def __getitem__(self, index):
        anchor_img, positive_img, negative_img = self.triplets[index]
        anchor_img = self.resize_face_image(anchor_img)
        positive_img = self.resize_face_image(positive_img)
        negative_img = self.resize_face_image(negative_img)
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.triplets)
