import pickle
import argparse
import os
import sys
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pkl_dir',
    type=str,
    default=current_path,
    help='pickle file directory')
parser.add_argument(
    '--data_name',
    type=str,
    default='reverse_dictionary',
    help='choose one count, data, dictionary, reverse_dictionary, embedding')

FLAGS, unparsed = parser.parse_known_args()
data_name=['count', 'data', 'dictionary', 'reverse_dictionary', 'embedding']
assert FLAGS.data_name in data_name
with open(os.path.join(FLAGS.pkl_dir,FLAGS.data_name+".pkl"),"rb") as f:
    rd = pickle.load(f)

for i in range(1000):
    print(i, " : ", rd[i])
