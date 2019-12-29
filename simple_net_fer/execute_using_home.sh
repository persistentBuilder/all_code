#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnode18
#SBATCH --mem=51000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate
module load cuda/10.0
#module load cudnn/7-cuda-10.0

python main.py --epochs=100 --lr=0.001 --batch_size=64 --net="extendNet" --dataset="affectnet" --save-checkpoint=1
