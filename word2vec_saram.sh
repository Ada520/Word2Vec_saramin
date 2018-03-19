export CUDA_VISIBLE_DEVICES=0
EMBEDDING_SIZE=128
BATCH_SIZE=128
SKIP_WINDOW=8
NUM_SKIPS=16
NUM_SAMPLED=8
NUM_STEPS=500001
LOG_DIR="/hd/skip-gram_minho/logs_saramin_nltk2_w${SKIP_WINDOW}s${NUM_SKIPS}e${EMBEDDING_SIZE}b${BATCH_SIZE}"
python word2vec_saram.py --log_dir=${LOG_DIR}\
    --embedding_size=${EMBEDDING_SIZE}\
    --batch_size=${BATCH_SIZE}\
    --skip_window=${SKIP_WINDOW}\
    --num_skips=${NUM_SKIPS}\
    --num_sampled=${NUM_SAMPLED}\
    --num_steps=${NUM_STEPS}
cp word2vec_saram.py ${LOG_DIR}/word2vec_saram.py
cp word2vec_saram.sh ${LOG_DIR}/word2vec_saram.sh

