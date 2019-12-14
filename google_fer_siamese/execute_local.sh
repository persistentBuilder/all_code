#!/usr/bin/env

source /Users/aryaman/research/deep_learning_venv/bin/activate


MY_NODE_HOME=/Users/aryaman/research/temp
mkdir -p $MY_NODE_HOME

CODE_DIR=/Users/aryaman/research/temp/google_fer_siamese

if [ ! -d "$CODE_DIR" ]; then
    cp -r ../google_fer_siamese $MY_NODE_HOME
fi

cd $CODE_DIR
mkdir -p "data"
mkdir -p "data/images"
mkdir -p "data/images/train"
mkdir -p "data/images/test"


python train.py