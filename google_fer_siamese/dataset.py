import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import requests
import torch
import cv2
from torchvision import transforms
from mtcnn import MTCNN


class SiameseGoogleFer(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, path, train_flag=True, transform=None, write_only_face=True, divisions=1, current_division=0,
                 load_in_memory=True):

        self.f = open(path, "r")
        self.train_flag = train_flag
        self.write_only_face = write_only_face
        self.all_lines = self.f.readlines()
        self.divisions = divisions
        self.load_in_memory = load_in_memory

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
        self.face_detector = MTCNN()

        self.g = open(failed_path, "w")
        line_num = 0
        for line in self.current_lines:
            line_num += 1
            #print
            line_components = line.split(",")
            face_image_1, face_image_2, face_image_3 = self.get_face_images(line_components, line_num)

            if not self.check_valid_comb(line_components, face_image_1, face_image_2, face_image_3):
                continue

            if self.load_in_memory:
                strong_flag, annotation = self.check_strong_annotation(line_components)

                face_image_1, face_image_2, face_image_3 = self.shuffle_based_on_annotations(annotation, face_image_1,
                                                                                             face_image_2, face_image_3)
                self.triplets.append([face_image_1, face_image_2, face_image_3])

        self.f.close()
        self.g.close()

    def get_lines(self, current_division):
        l = len(self.all_lines)
        start = int((l/self.divisions)*current_division)
        end = int((l/self.divisions)*(current_division+1)) if current_division < self.divisions - 1 else l 
        return self.all_lines[start:end]

    def get_face_images(self, line_components, line_num):

        url_for_image = []
        url_for_image.append(line_components[0][1:-1])
        url_for_image.append(line_components[5][1:-1])
        url_for_image.append(line_components[10][1:-1])
        try:
            imgs = []
            for url in url_for_image:
                imgs.append(self.download_and_load_image(url, line_num))
        except:
            return None, None, None

        for img in imgs:
            if img is None:
                return None, None, None

        imgs_are_face_imgs = True
        for img in imgs:
            imgs_are_face_imgs = imgs_are_face_imgs and img.shape[0] == self.image_resize_height and \
                                 img.shape[1] == self.image_resize_width

        face_images = []
        for i in range(0, len(imgs)):
            img = imgs[i]
            if imgs_are_face_imgs:
                face_images.append(img)
            else:
                face_image = self.resize_face_image(self.select_face_region(img, line_components[i*5+1:(i+1)*5],
                                                                            img.shape[0], img.shape[1]))
                face_images.append(face_image)
                try:
                    cv2.imwrite(self.get_path(url_for_image[i], line_num), face_image)
                except:
                    raise("error couldn't write")
        return face_images[0], face_images[1], face_images[2]

    def check_valid_comb(self, line_components, face_image_1, face_image_2, face_image_3):
        strong_flag, annotation = self.check_strong_annotation(line_components)
        any_face_image_is_none = (face_image_1 is None) or (face_image_2 is None) or (face_image_3 is None)
        if any_face_image_is_none:
            return False
        face_detect = self.check_if_single_face_present(face_image_1) and \
                      self.check_if_single_face_present(face_image_2) and \
                      self.check_if_single_face_present(face_image_3)

        if (not strong_flag) or any_face_image_is_none or (not face_detect):
            return False
        return True

    def check_if_single_face_present(self, img):
        results = self.face_detector.detect_faces(img)
        return len(results) == 1

    def shuffle_based_on_annotations(self, annotation, face_image_1, face_image_2, face_image_3):
        if annotation == 1:
            return face_image_2, face_image_3, face_image_1
        elif annotation == 2:
            return face_image_2, face_image_1, face_image_2
        else:
            return face_image_1, face_image_2, face_image_3

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
        image_path = base_directory + image_name
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

    def __getitem__(self, index):
        if self.load_in_memory:
            anchor_img, positive_img, negative_img = self.triplets[index]
        else:
            line_components = self.current_lines[index].split(",")
            strong_flag, annotation = self.check_strong_annotation(line_components)
            face_image_1, face_image_2, face_image_3 = self.get_face_images(line_components, index)
            if (not strong_flag) or (face_image_1 is None):
                raise("either not strong annotation or image could not be loaded")

            anchor_img, positive_img, negative_img = self.shuffle_based_on_annotations(annotation, face_image_1,
                                                                                       face_image_2, face_image_3)

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
