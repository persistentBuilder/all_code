#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=81000
#SBATCH --cpus-per-task=10
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate
mkdir -p /ssd_scratch/cvit/aryaman.g/
mkdir -p /ssd_scratch/cvit/aryaman.g/affectnet
python multiprocess_affectnet_download --save_path="/ssd_scratch/cvit/aryaman.g/affectnet/"
