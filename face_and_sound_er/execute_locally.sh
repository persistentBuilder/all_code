#!/usr/bin/env bash

source /Users/aryaman/research/deep_learning_venv/bin/activate

python detect_emotion_while_speaking.py --video-path="/Users/aryaman/research/FER_datasets/video/ErinBrockavich_shot_2.mp4"\
       --shape-predictor="/Users/aryaman/research/all_code/simple_net_fer/shape_predictor_68_face_landmarks.dat"\
       --model-path="/Users/aryaman/research/all_code/simple_net_fer/runs/affectnet_model/model_best.pth.tar"