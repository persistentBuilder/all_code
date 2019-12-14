#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --gres=gpu:2
#SBATCH --mem=51000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate
module load cuda/9.0
module load cudnn/7-cuda-9.0

MY_NODE_HOME=/ssd_scratch/cvit/aryaman.g
mkdir -p $MY_NODE_HOME

CODE_DIR=/ssd_scratch/cvit/aryaman.g/google_fer_siamese

if [ ! -d "$CODE_DIR" ]; then
    cp -r ../google_fer_siamese $MY_NODE_HOME
fi

cd $CODE_DIR
mkdir -p $CODE_DIR/data
mkdir -p $CODE_DIR/data/images
mkdir -p "$CODE_DIR/data/images/train"
mkdir -p "$CODE_DIR/data/images/test"
cp /home/aryaman.g/projects/FER/FEC_dataset/* $CODE_DIR/data/

python train.py