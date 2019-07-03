#!/usr/bin/env bash

DATASET_DIR="data/ud-treebanks-v2.3"

echo "Generating multilingual dataset..."

mkdir -p "data/ud"
mkdir -p "data/ud/multilingual"

python concat_treebanks.py data/ud/multilingual --dataset_dir ${DATASET_DIR}