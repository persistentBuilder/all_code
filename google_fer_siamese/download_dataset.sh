#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=81000
#SBATCH --cpus-per-task=10
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate

python multiprocess_dataset_download.py --train=0 --divisions=50
