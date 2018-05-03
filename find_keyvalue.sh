#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
PKL_DIR="/hd/skip-gram_minho/VTW_demo/mecab_dic2_indeed_ncs_len1del/w8s16e32b128"
#SAVE_DIR=$(echo ${PKL_DIR/word_embedding_data/find_key})
SAVE_DIR=$PKL_DIR
echo $SAVE_DIR
STEPS="100000 200000 300000 400000 500000"
export PYTHON_PATH="./"
do_python(){
    python find_keyvalue.py --pkl_dir=$1 \
        --save_dir=$2\
        --step=$3
}

for STEP in $STEPS
do
    do_python ${PKL_DIR} ${SAVE_DIR} ${STEP}
    echo $STEP
done


