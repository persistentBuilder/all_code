#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=81000
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate
#mkdir -p /ssd_scratch/cvit/aryaman.g/
#mkdir -p /ssd_scratch/cvit/aryaman.g/affectnet
python detect_frames_with_emotion.py --video-path="/home/aryaman.g/projects/all_code/face_and_sound_er/videos/ErinBrockavich_shot_2.mp4"
