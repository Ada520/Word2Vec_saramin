EMBEDDING_SIZE="32 64 128"
DATAFILE_PATH="/hd/ncs_worknet/indeed_SAS_20180510.txt"
DATAPKL_DIR="/hd/ncs_worknet/20180510indeedpkl"
python ./word2vec_saram_builddataset.py --datapkl_dir=$DATAPKL_DIR \
    --datafile_path=$DATAFILE_PATH

for ES in $EMBEDDING_SIZE
do
    sh word2vec_saram_iter.sh $DATAPKL_DIR $ES
done


