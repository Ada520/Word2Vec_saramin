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
    default='/hd/word_embedding_data',
    help='The log directory for TensorBoard summaries.')
parser.add_argument(
    '--datapkl_dir',
    type=str,
    default=None,
    help='data, dic, reverse_dic, count pkl directory path')
parser.add_argument(
    '--embedding_size',
    type=int,
    default=100,
    help='word embedding size')
parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='data batch size')
parser.add_argument(
    '--skip_window',
    type=int,
    default=10,
    help='How many words to consider left and right')
parser.add_argument(
    '--num_skips',
    type=int,
    default=4,
    help='How many times to reuse an input to generate a label.')
parser.add_argument(
    '--num_sampled',
    type=int,
    default=64,
    help='Number of negative examples to sample')
parser.add_argument(
    '--num_steps',
    type=int,
    default=500001,
    help='Number of max steps of training')
parser.add_argument(
    '--vocabulary_size',
    tpye=int,
    default=35000,
    help='Number of vocabulary size')
FLAGS, unparsed = parser.parse_known_args()
num_steps = FLAGS.num_steps
skip_window = FLAGS.skip_window  # How many words to consider left and right.
batch_size = FLAGS.batch_size
embedding_size = FLAGS.embedding_size  # Dimension of the embedding vector.
num_skips = FLAGS.num_skips  # How many times to reuse an input to generate a label.
num_sampled = FLAGS.num_sampled  # Number of negative examples to sample.

data_index = 0

# filename = "C:\\Users\\Tmax\\Desktop\\WordVec\\vtw\\parsed_data\\parsed_data_indeed_wordcount=3.txt"

#datapath = "0418_sm_R1_Version"
#filename = "/hd/ncs_indeed_data/parsed_data.txt"

#datapath = "vanilaVersion"
#filename = "/home/byounggeon_kim/WordVec/vtw/parsed_data/indeed_mecab(word_threshold=3).txt"


#if not os.path.exists(os.path.join(FLAGS.log_dir, datapath,"dim=%d_window=%d_step=%d"%(embedding_size,skip_window,num_steps))):
#    os.makedirs(os.path.join(FLAGS.log_dir, datapath,"dim=%d_window=%d_step=%d"%(embedding_size,skip_window,num_steps)))

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with open(filename, encoding='UTF-8') as f:
        # data = tf.compat.as_str(f.read()).split(",")
        #      data = re.split('[/W+\s\,\.]',tf.compat.as_str(f.read()))
        data = word_tokenize(tf.compat.as_str(f.read()))
        data[:] = (value for value in data if value != "(")
        data[:] = (value for value in data if value != ")")
        data[:] = (value for value in data if value != ".")
        data[:] = (value for value in data if value != ":")
        data[:] = (value for value in data if value != "[")
        data[:] = (value for value in data if value != "]")
        data[:] = (value for value in data if value != ",")
        data[:] = (value for value in data if value != "-")

    return data
# Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(words, n_words):
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
#vocabulary = read_data(filename)
#print('Data size', len(vocabulary))
def load_pickle(path, picklename):
    with open(os.path.join(path, picklename+".pkl"), "rb") as f:
        pickledata = pickle.load(f)
    return pickledata


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    plt.savefig(filename)

vocabulary_size = FLAGS.vocabulary_size



class Word2vecModel(object):

    def __init__(self):
        super(Word2vecModel, self).__init__()
        self.num_steps = FLAGS.num_steps
        self.skip_window = FLAGS.skip_window
        self.batch_size = FLAGS.batch_size
        self.embedding_size = FLAGS.embedding_size
        self.num_skips = FLAGS.num_skips
        self.num_sampled = FLAGS.num_sampled
        self.vocabulary_size = vocabulary_size
    def build_model(self):

        # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[shlf.batch_size, 1])
            #valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                embbedings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embbedings, train_inputs)

        with tf.device('/cpu:0'):
            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [vocabulary_size, embedding_size],
                        stddev=1.0/ math.sqrt(embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)

        self.embbedings = embbedings
        self.nce_weights = nce_weights
        self.loss = loss

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=tf.GraphKeys.GLOBAL_STEP)

        self.global_step = global_step




# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
#data, count, dictionary, reverse_dictionary = build_dataset(
#    vocabulary, vocabulary_size)
#del vocabulary  # Hint to reduce memory.

# To reduce data loading time, this code used pickled data

def main() :
#    datapkl_path = "/hd/ncs_indeed_data/SAS_ncs2_job"
    datapkl_path = FLAGS.datapkl_dir
    print(datapkl_path)
    data = load_pickle(datapkl_path, "data")
    count = load_pickle(datapkl_path, "count")
    dictionary = load_pickle(datapkl_path, "dictionary")
    reverse_dictionary = load_pickle(datapkl_path, "reverse_dictionary")

    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])



    # Step 3: Function to generate a training batch for the skip-gram model.





    batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
              reverse_dictionary[labels[i, 0]])

    # Step 4: Build and train a skip-gram model.



    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        with tf.device('/cpu:0'):
            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [vocabulary_size, embedding_size],
                        stddev=1.0 / math.sqrt(embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)
        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Merge all summaries.
        merged = tf.summary.merge_all()

        # Add variable initializer.
        init = tf.global_variables_initializer()

        # Create a saver.
        saver = tf.train.Saver()

    # Step 5: Begin training.
    num_steps = FLAGS.num_steps
    disp_num = 2000

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact', random_state=0)
    with tf.Session(graph=graph) as session:
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips,
                                                        skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
            # Feed metadata variable to session for visualizing the graph in TensorBoard.
            _, summary, loss_val = session.run(
                [optimizer, merged, loss],
                feed_dict=feed_dict,
                run_metadata=run_metadata)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            # Add returned summaries to writer in each step.
            # writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
        # if step == (num_steps - 1):
        #      writer.add_run_metadata(run_metadata, 'step%d' % step)

            if (step + 1) % disp_num == 0:
                if step > 0:
                    average_loss /= disp_num
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step + 1, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if (step + 1) % (disp_num * 5) == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        # print(nearest[k])
                        close_word = reverse_dictionary[nearest[k]]
                        log_str += ' %s,' % (close_word)
                    print(log_str)
                print('')
            if (step + 1) % (disp_num * 10) == 0:
                print('Visualizing...')
                final_embeddings = normalized_embeddings.eval()
                plot_only = 500
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact', random_state=0)

                low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
                labels = [reverse_dictionary[i] for i in range(plot_only)]

                #plot_with_labels(low_dim_embs, labels, os.path.join(FLAGS.log_dir, 'tsne_%d.png' % (step+1)))
                plot_with_labels(low_dim_embs, labels, os.path.join(FLAGS.log_dir, 'tsne_%d.png' % (step+1)))
                print('')
                with open(os.path.join(FLAGS.log_dir, "embedding%d.pkl"%(step+1)), "wb") as f:
                    pickle.dump(final_embeddings, f)
    # Write corresponding labels for the embeddings.
        with open(FLAGS.log_dir + '/metadata.tsv', 'w', encoding="UTF8") as f:
            for i in range(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        # Save the model for checkpoints.
        saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

    writer.close()


    # Step 6: Visualize the embeddings.


    # pylint: disable=missing-docstring
    # Function to draw visualization of distance between embeddings.



    #with open(os.path.join(FLAGS.log_dir, file ,"_%d"%(num_steps), "embedding.pkl"), "wb") as f:
    #    pickle.dump(final_embeddings, f)

    with open(os.path.join(FLAGS.log_dir, "data.pkl"), "wb") as f:
        pickle.dump(data, f)

    with open(os.path.join(FLAGS.log_dir, "count.pkl"), "wb") as f:
        pickle.dump(count, f)

    with open(os.path.join(FLAGS.log_dir, "dictionary.pkl"), "wb") as f:
        pickle.dump(dictionary, f)

    with open(os.path.join(FLAGS.log_dir, "reverse_dictionary.pkl"), "wb") as f:
        pickle.dump(reverse_dictionary, f)

if __name__ == "__main__":
    main()

