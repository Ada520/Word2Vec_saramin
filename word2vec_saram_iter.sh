export CUDA_VISIBLE_DEVICES=0
CURRENT_PATH=$PWD
EMBEDDING_SIZE=$1
BATCH_SIZE=128
SKIP_WINDOW=8
NUM_SKIPS=16
NUM_SAMPLED=8
NUM_STEPS=2000001
LOG_DIR="/hd/skip-gram_minho/VTW_demo/indeed_ncs02and20_job_line_step2000000/w${SKIP_WINDOW}s${NUM_SKIPS}e${EMBEDDING_SIZE}b${BATCH_SIZE}"
python word2vec_saram_StepVersion.py --log_dir=${LOG_DIR}\
    --embedding_size=${EMBEDDING_SIZE}\
    --batch_size=${BATCH_SIZE}\
    --skip_window=${SKIP_WINDOW}\
    --num_skips=${NUM_SKIPS}\
    --num_sampled=${NUM_SAMPLED}\
    --num_steps=${NUM_STEPS}
cp word2vec_saram_StepVersion.py ${LOG_DIR}/word2vec_saram_StepVersion.py
STEPS="100000 500000 1000000 1500000 2000000"
export PYTHON_PATH="./"
do_python(){
    python find_keyvalue.py --pkl_dir=$1 \
        --save_dir=$2\
        --step=$3
}

for STEP in $STEPS
do
    do_python ${LOG_DIR} ${LOG_DIR} ${STEP}
    echo $STEP
done

cd ${LOG_DIR}
sh ${CURRENT_PATH}/convert_json/convert_json.sh
cd ${CURRENT_PATH}

