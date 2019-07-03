#!/usr/bin/env bash

# Can download UD 2.3 or 2.4
UD_2_3="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2895/ud-treebanks-v2.3.tgz?sequence=1&isAllowed=y"
UD_2_4="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/ud-treebanks-v2.4.tgz?sequence=4&isAllowed=y"

DATASET_DIR="data/ud-treebanks-v2.3"

ARCHIVE="ud_data.tgz"


echo "Downloading UD data..."

curl ${UD_2_3} -o ${ARCHIVE}

tar -xvzf ${ARCHIVE} -C ./data
mv ${ARCHIVE} ./data


echo "Generating multilingual dataset..."

mkdir -p "data/ud"
mkdir -p "data/ud/multilingual"

python concat_treebanks.py data/ud/multilingual --dataset_dir ${DATASET_DIR}
