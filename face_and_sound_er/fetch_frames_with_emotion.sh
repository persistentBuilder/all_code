#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=11000
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=END

source /home/aryaman.g/packages/keras_tf_venv3/bin/activate
#mkdir -p /ssd_scratch/cvit/aryaman.g/
#mkdir -p /ssd_scratch/cvit/aryaman.g/affectnet
python detect_emotion_while_speaking.py --video-path="/home/aryaman.g/projects/all_code/face_and_sound_er/videos/ErinBrockavich_shot_2.mp4"\
       --shape-predictor="/home/aryaman.g/projects/all_code/simple_net_fer/shape_predictor_68_face_landmarks.dat"\
       --model-path="/home/aryaman.g/projects/all_code/simple_net_fer/runs/affectnet_model/model_best.pth.tar"


