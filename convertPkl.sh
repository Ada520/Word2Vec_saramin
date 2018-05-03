cd "/hd/word_embedding_data/indeed_mecab(word_threshold=3)/(window=8)(step=500001)"
(source ~/tensorflow/bin/activate)
mkdir ./text_data

python -m pickle dictionary.pkl > ./text_data/dictionary.txt
python -m pickle reverse_dictionary.pkl > ./text_data/reverse_dictionary.txt
python -m pickle count.pkl> ./text_data/count.txt
