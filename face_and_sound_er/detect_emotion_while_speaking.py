import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from extendNet import extendNet
import torchvision.transforms as transforms
from imutils import face_utils
import datetime
import argparse
import imutils
import time


class Frame(object):

    def __init__(self, img, face_detector, shape_predictor, frame_num):
        self.img = img
        self.detector = face_detector
        self.shape_predictor = shape_predictor
        self.m_start, self.m_end = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
        self.detect_faces_rects()
        self.frame_num = frame_num
        self.face_rects = None #face_rects

    def detect_faces_rects(self, from_gray=False):
        if from_gray:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.face_rects = self.detector(gray, 0)
        else:
            self.face_rects = self.detector(self.img, 1)

    def update_face_images(self, img):
        self.img = img
        self.detect_faces_rects()

    def get_best_overlap_with_face(self, current_rect):

        def area_of_rect(face_rect):
            return abs(face_rect.bottom()-face_rect.top())*abs(face_rect.right()-face_rect.left())

        for face_rect in self.face_rects:
            olp = self.overlap(current_rect, face_rect)
            if olp > area_of_rect(face_rect)*0.5:
                return face_rect
        return None

    def get_mouth_for_face(self, rect):
        margin = 10
        shape = self.shape_predictor(self.img, rect)
        shape = face_utils.shape_to_np(shape)

        mouth_shape = shape[self.m_start:self.m_end + 1]

        leftmost_x = min(x for x, y in mouth_shape) - margin
        bottom_y = min(y for x, y in mouth_shape) - margin
        rightmost_x = max(x for x, y in mouth_shape) + margin
        top_y = max(y for x, y in mouth_shape) + margin

        mouth_img = self.img[bottom_y:top_y, leftmost_x:rightmost_x]
        return mouth_img

    @staticmethod
    def overlap(rect1, rect2):
        dx = min(rect1.right(), rect2.right()) - max(rect1.left(), rect2.left())
        dy = min(rect1.bottom(), rect2.top()) - max(rect1.bottom(), rect2.top())
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0

    def get_face_from_rect(self, rect):
        return self.img[rect.top():rect.bottom(), rect.left(): rect.right()]


def is_speaking(prev_img, curr_img, debug=False, threshold=500, width=400, height=400):
    """
    Args:
        prev_img:
        curr_img:
    Returns:
        Bool value if a person is speaking or not
    """
    prev_img = cv2.resize(prev_img, (width, height))
    curr_img = cv2.resize(curr_img, (width, height))

    diff = cv2.absdiff(prev_img, curr_img)
    norm = np.sum(diff) / (width*height) * 100
    if debug:
        print(norm)
    return norm > threshold


def resize_face_image(img):
    return cv2.resize(img, (image_resize_width, image_resize_height), interpolation=cv2.INTER_CUBIC)

#
# def get_faces_from_frame(img):
#     detected_faces = face_detector(img, 1)
#     face_images = []
#     for i, d in enumerate(detected_faces):
#         face_images.append(img[d.top():d.bottom(), d.left(): d.right()])
#     return face_images


def load_model(model_path, model=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def strong_pred(value, prediction):
    if value > 3 and prediction != 0:
        return True
    return False


def check_emotion_in_face(face_image, model=None, model_path=None):
    if model is None:
        model = load_model(model_path)
        model.eval()
    output = model(face_image.unsqueeze(0))
    values, predictions = torch.max(output.data, 1)
    if strong_pred(values.item(), predictions.item()):
        return predictions.item()
    else:
        return -1


def check_for_lip_movement(curr_frame, prev_frame, curr_face_rect):

    print(curr_frame.frame_num, prev_frame.frame_num)
    prev_face_rect = prev_frame.get_best_overlap_with_face(curr_face_rect)
    if prev_face_rect is None:
        return False
    prev_mouth_img = prev_frame.get_mouth_for_face(prev_face_rect)
    curr_mouth_img = curr_frame.get_mouth_for_face(curr_face_rect)
    return is_speaking(prev_mouth_img, curr_mouth_img, threshold=args.threshold)


def write_frame(face_img, emotion_label, video_name, frame_num):
    img_name_to_write = video_name+"""_{:06d}_""".format(frame_num) + str(emotion_label) + ".jpg"
    print(img_name_to_write)
    cv2.imwrite(img_name_to_write, face_img)


def main():
    video_file = args.video_path
    video_name = video_file.split("/")[-1]

    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    shape_predictor = dlib.shape_predictor(args.shape_predictor)
    face_detector = dlib.get_frontal_face_detector()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    model = extendNet(num_classes=8)
    model = nn.DataParallel(model)
    model = load_model(
        model_path="/home/aryaman.g/projects/all_code/simple_net_fer/runs/affectnet_model/model_best.pth.tar",
        model=model)
    model.eval()
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count = frame_count + 1

        curr_frame = Frame(frame, face_detector, shape_predictor, frame_count)
        if prev_frame is not None and curr_frame.face_rects is not None:
            for curr_face_rect in curr_frame.face_rects:
                if check_for_lip_movement(curr_frame, prev_frame, curr_face_rect):
                    face_img = curr_frame.get_face_from_rect(curr_face_rect)
                    emotion_label = check_emotion_in_face(transform(resize_face_image(face_img)), model=model)
                    if emotion_label > 0:
                        print(frame_count)
                        write_frame(face_img, emotion_label, video_name, frame_count)
        prev_frame = curr_frame
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default="")
    parser.add_argument("--shape-predictor", required=True, help="path to facial landmark predictor")
    parser.add_argument("--threshold", type=int, default=500, help="threshold of speaking or not")
    parser.add_argument("--width", type=int, default=800, help="width of window")
    args = parser.parse_args()
    image_resize_width = 224
    image_resize_height = 224
    main()
