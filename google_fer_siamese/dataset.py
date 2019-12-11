import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SiameseGoogleFer(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, path, train_flag=True):

        f = open(path, "r")
        self.all_lines = f.readlines()
        self.all_triplets = []
        self.triplets = []
        self.train_flag = train_flag
        if train_flag:
            self.lines = self.all_lines[0:int(len(self.all_lines)*0.9)]
        else:
            self.lines = self.all_lines[int(len(self.all_lines)*0.9):len(self.all_lines)]

        for line in self.lines:
            line_components = line.split(",")
            url_1 = line_components[0][1:-1]
            url_2 = line_components[5][1:-1]
            url_3 = line_components[10][1:-1]
            #image_name_1 = url_1.split("/")[-1]
            img_1 = self.download_and_load_image(url_1)
            img_2 = self.download_and_load_image(url_2)
            img_3 = self.download_and_load_image(url_3)
            bounding_box_1 = []
            for j in range(1,5):
                bounding_box_1.append(line_components[j])

            bounding_box_2 = []
            for j in range(6, 10):
                bounding_box_2.append(line_components[j])

            bounding_box_3 = []
            for j in range(11, 15):
                bounding_box_3.append(line_components[j])

            face_image_1 = self.select_face_region(img_1, bounding_box_1)
            face_image_2 = self.select_face_region(img_2, bounding_box_2)
            face_image_3 = self.select_face_region(img_3, bounding_box_3)
            if line_components[15] == 'TWO_CLASS_TRIPLET':
                if line_components[17] == 1:
                    self.triplets.append([face_image_2, face_image_3, face_image_1])
                elif line_components[17] == 2:
                    self.triplets.append([face_image_2, face_image_1, face_image_2])
                else:
                    self.triplets.append([face_image_1, face_image_2, face_image_3])

    def select_face_region(self, img, bounding_box):
        pass

    def download_and_save_image_from_url(self, url):
        pass

    def path_of_saved_image(self, url):
        pass

    def download_and_load_image(self, url):
        #check if not already downloaded
        saved_image_path = self.path_of_saved_image(url)
        if saved_image_path is None:
            saved_image_path = self.download_and_save_image_from_url(url)
        return np.asarray(Image.open(saved_image_path))

    def __getitem__(self, index):
        return self.triplets[index]

    def __len__(self):
        return len(self.triplets)


