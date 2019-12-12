import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import requests

class SiameseGoogleFer(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, path, train_flag=True):

        f = open(path, "r")
        self.all_lines = f.readlines()[0:100]
        self.all_triplets = []
        self.triplets = []
        self.train_flag = train_flag

        # if train_flag:
        #     self.lines = self.all_lines[0:int(len(self.all_lines)*0.9)]
        # else:
        #     self.lines = self.all_lines[int(len(self.all_lines)*0.9):len(self.all_lines)]
        line_num = 0
        for line in self.all_lines:
            line_num = line_num + 1
            line_components = line.split(",")
            url_1 = line_components[0][1:-1]
            url_2 = line_components[5][1:-1]
            url_3 = line_components[10][1:-1]
            #image_name_1 = url_1.split("/")[-1]
            img_1 = self.download_and_load_image(url_1, line_num)
            img_2 = self.download_and_load_image(url_2, line_num)
            img_3 = self.download_and_load_image(url_3, line_num)
            bounding_box_1 = []
            for j in range(1,5):
                bounding_box_1.append(line_components[j])

            bounding_box_2 = []
            for j in range(6, 10):
                bounding_box_2.append(line_components[j])

            bounding_box_3 = []
            for j in range(11, 15):
                bounding_box_3.append(line_components[j])

            face_image_1 = self.select_face_region(img_1, bounding_box_1, img_1.shape[0], img_1.shape[1])
            face_image_2 = self.select_face_region(img_2, bounding_box_2, img_2.shape[0], img_2.shape[1])
            face_image_3 = self.select_face_region(img_3, bounding_box_3, img_3.shape[0], img_3.shape[1])
            if line_components[15] == 'TWO_CLASS_TRIPLET':
                if line_components[17] == 1:
                    self.triplets.append([face_image_2, face_image_3, face_image_1])
                elif line_components[17] == 2:
                    self.triplets.append([face_image_2, face_image_1, face_image_2])
                else:
                    self.triplets.append([face_image_1, face_image_2, face_image_3])

    def select_face_region(self, img, bounding_box, height, width):

        top_left_column = max(0, int(bounding_box[0]*width))
        bottom_right_column = min(width, bounding_box[1]*width)
        top_left_row = max(0, int(bounding_box[2]*height))
        bottom_right_row = min(height, int(bounding_box[3]*height))
        return img[top_left_row:bottom_right_row, top_left_column:bottom_right_column, :]

    def download_and_save_image_from_url(self, url, line_num):

        image_name = url.split("/")[-1]
        line_str = str(line_num).zfill(8) + "_"
        image_path = "data/images/" + line_str + image_name

        with open(image_path, 'wb') as handle:
            response = requests.get(url, stream=True)

            if not response.ok:
                print(response)

            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)

        return self.saved_image_load(url, line_num)


    def saved_image_load(self, url, line_num):
        image_name = url.split("/")[-1]
        line_str = str(line_num).zfill(8) + "_"
        image_path = "data/images/" + line_str + image_name
        try:
            np.asarray(Image.open(image_path))
        except:
            return None


    def download_and_load_image(self, url, line_num):
        #check if not already downloaded
        saved_image = self.saved_image_load(url,line_num)
        if saved_image is None:
            saved_image = self.download_and_save_image_from_url(url, line_num)
        return saved_image

    def __getitem__(self, index):
        return self.triplets[index]

    def __len__(self):
        return len(self.triplets)
