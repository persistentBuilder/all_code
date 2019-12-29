import cv2
import dlib
import numpy as np
import torch
import argparse

def get_faces_from_frame(img):
    detected_faces = face_detector(img, 1)
    face_images = []
    for i, d in enumerate(detected_faces):
        face_images.append(img[d.top():d.bottom(), d.left(): d.right()])
    return face_images


def load_model(model_path, model=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def strong_pred(output):
    pass

def check_emotion_in_face(face_image, model=None, model_path=None):
    if model is None:
        model = load_model(model_path)
    model.eval()
    output = model(face_image)
    if strong_pred(output):
        return torch.max(output.data, 1)[0]
    else:
        return -1

def check_for_lip_movement():
    pass


def write_frame(face_img, emotion_label, video_name, frame_num):
    cv2.imwrite(face_img, video_name+"""_{06d}_""".format(frame_num) + str(emotion_label) + ".jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default="")
    args = parser.parse_args()
    frame_window_for_movement = 24
    video_file = args.video_path
    video_name = video_file.split("/", 1)[-1]
    video_base_path = video_file.rsplit("/", 1)[0]
    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    face_detector = dlib.get_frontal_face_detector()

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_count = frame_count + 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = get_faces_from_frame()
        for face_img in faces:
            emotion_label = check_emotion_in_face(face_img)
            if emotion_label >= 0:
                write_frame(face_img, emotion_label, video_name, frame_count)

    cap.release()
    cv2.destroyAllWindows()
    detector = dlib.get_frontal_face_detector()