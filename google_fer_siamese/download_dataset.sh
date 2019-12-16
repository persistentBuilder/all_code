#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=81000
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate

python download_dataset_using_dataloader.py
