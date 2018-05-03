# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
import re
import numpy as np
from six.moves import urllib
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import nltk
import time
nltk.download('punkt')

from nltk.tokenize import word_tokenize

from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "NanumGothic"

from matplotlib import font_manager, rc

#font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
#rc('font', family=font_name)
# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default='/hd/ncs_indeed_data/ncs02and20_indeed_job_line',
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()


# filename = "C:\\Users\\Tmax\\Desktop\\WordVec\\vtw\\parsed_data\\parsed_data_indeed_wordcount=3.txt"

filename = "/hd/ncs_indeed_data/parsed_data_04.txt"

#datapath = "vanilaVersion"
#filename = "/home/byounggeon_kim/WordVec/vtw/parsed_data/indeed_mecab(word_threshold=3).txt"


if not os.path.exists(os.path.join(FLAGS.log_dir)):
    os.makedirs(os.path.join(FLAGS.log_dir))

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with open(filename, encoding='UTF-8') as f:
        # data = tf.compat.as_str(f.read()).split(",")
        #      data = re.split('[/W+\s\,\.]',tf.compat.as_str(f.read()))
        data = word_tokenize(tf.compat.as_str(f.read()))
#        data[:] = (value for value in data if value != "(")
#        data[:] = (value for value in data if value != ")")
#        data[:] = (value for value in data if value != ".")
#        data[:] = (value for value in data if value != ":")
#        data[:] = (value for value in data if value != "[")
#        data[:] = (value for value in data if value != "]")
        data[:] = (value for value in data if value != ",")
#        data[:] = (value for value in data if value != "-")

    return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 35000


def build_dataset(words, n_words):
    """1 length word delete"""
#    stamp=time.time()
#    print(len(words))
#    for i in range(len(words),0,-1):
#        if len(words[i-1])==1:
#            del(words[i-1])
#        if i%100000 == 0 :
#            now=time.time()
#            print("%f %% completed... collapse time: %f" %((len(words)-i)/(len(words)),now-stamp))
#            stamp=now
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)



with open(os.path.join(FLAGS.log_dir, "data.pkl"), "wb") as f:
    pickle.dump(data, f)

with open(os.path.join(FLAGS.log_dir, "count.pkl"), "wb") as f:
    pickle.dump(count, f)

with open(os.path.join(FLAGS.log_dir, "dictionary.pkl"), "wb") as f:
    pickle.dump(dictionary, f)

with open(os.path.join(FLAGS.log_dir, "reverse_dictionary.pkl"), "wb") as f:
    pickle.dump(reverse_dictionary, f)
