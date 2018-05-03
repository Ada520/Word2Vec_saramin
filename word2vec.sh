EMBEDDING_SIZE="32 64 128"
python ./word2vec_saram_builddataset.py

for ES in $EMBEDDING_SIZE
do
    sh word2vec_saram_iter.sh $ES
done


