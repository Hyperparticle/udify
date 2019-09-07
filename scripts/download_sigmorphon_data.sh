#!/usr/bin/env bash

# Download SIGMORPHON 2019 Shared Task data
DATA="https://github.com/sigmorphon/2019.git"

GIT_DIR="data/sigmorphon-2019"
DATASET_DIR="${GIT_DIR}/task2"


echo "Downloading SIGMORPHON data..."

git clone ${DATA} ${GIT_DIR}


echo "Generating multilingual dataset..."

mkdir -p "data/sigmorphon-2019/multilingual"

python concat_treebanks.py data/sigmorphon-2019/multilingual --dataset_dir ${DATASET_DIR}
