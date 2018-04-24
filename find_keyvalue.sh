PKL_DIR="/hd/skip-gram_minho/logs_saramin_nltk2_w8s16e128b128"
SAVE_DIR=${PKL_DIR}
export PYTHON_PATH="./"
python find_keyvalue.py --pkl_dir=${PKL_DIR} \
    --save_dir=${SAVE_DIR}
