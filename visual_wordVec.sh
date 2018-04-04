EMBEDDING_TALBE="/hd/skip-gram_minho/logs_saramin_nltk2_w8s16e128b128/embedding.pkl"
REVERSE_DICTIONARY="/hd/skip-gram_minho/logs_saramin_nltk2_w8s16e128b128/reverse_dictionary.pkl"
SAVE_FIG_NAME="/home/minhopark2115/Picture/wordVec.png"
PLOT_ONLY=1000

python ./visual_wordVec.py --plot_only=${PLOT_ONLY} \
    --embedding_table=${EMBEDDING_TALBE} \
    --reverse_dictionary=${REVERSE_DICTIONARY} \
    --save_fig_name=${SAVE_FIG_NAME}

