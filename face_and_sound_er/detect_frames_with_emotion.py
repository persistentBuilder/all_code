import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from extendNet import extendNet
import torchvision.transforms as transforms


def resize_face_image(img):
    return cv2.resize(img, (image_resize_width, image_resize_height), interpolation=cv2.INTER_CUBIC)


def get_faces_from_frame(img):
    detected_faces = face_detector(img, 1)
    face_images = []
    for i, d in enumerate(detected_faces):
        face_images.append(img[d.top():d.bottom(), d.left(): d.right()])
    return face_images


def load_model(model_path, model=None):
    checkpoint = torch.load(model_path)
    #print(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def strong_pred(value, prediction):
    if value>3 and prediction!=0:
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


def check_for_lip_movement():
    pass


def write_frame(face_img, emotion_label, video_name, frame_num):
    img_name_to_write = video_name+"""_{:06d}_""".format(frame_num) + str(emotion_label) + ".jpg"
    print(img_name_to_write)
    cv2.imwrite(img_name_to_write, face_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default="")
    args = parser.parse_args()
    frame_window_for_movement = 24
    video_file = args.video_path
    video_name = video_file.split("/")[-1]
    video_base_path = video_file.rsplit("/", 1)[0]
    image_resize_width = 224
    image_resize_height = 224
    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    face_detector = dlib.get_frontal_face_detector()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    model = extendNet(num_classes=8)
    model = nn.DataParallel(model)
    model = load_model(model_path="/home/aryaman.g/projects/all_code/simple_net_fer/runs/affectnet_model/model_best.pth.tar", model=model)
    model.eval()

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_count = frame_count + 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = get_faces_from_frame(frame)
        for face_img in faces:
            print(type(face_img))
            emotion_label = check_emotion_in_face(transform(resize_face_image(face_img)), model=model)
            if emotion_label > 0:
                print(frame_count)
                write_frame(face_img, emotion_label, video_name, frame_count)

    cap.release()
    cv2.destroyAllWindows()
