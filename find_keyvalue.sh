export CUDA_VISIBLE_DEVICES=0
PKL_DIR="/hd/skip-gram_minho/VTW_demo/SAS_ncspartial2_job_0508/w8s16e128b128"
#SAVE_DIR=$(echo ${PKL_DIR/word_embedding_data/find_key})
SAVE_DIR=$PKL_DIR
echo $SAVE_DIR
#STEPS="100000 2000000"
STEPS="100000 2000000"
export PYTHON_PATH="./"
do_python(){
    python find_keyvalue.py --pkl_dir=$1 \
        --save_dir=$2\
        --step=$3
}
echo 1
for STEP in $STEPS
do
    do_python ${PKL_DIR} ${SAVE_DIR} ${STEP}
    echo $STEP
done


