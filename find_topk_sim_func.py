import tensorflow as tf
import numpy as np
import argparse
import pickle
import os
from collections import namedtuple
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pkl_dir',
        type=str,
        default=None,
        help='pickle file directory')
    FLAGS, unparsed = parser.parse_known_args()


nearest_topk = namedtuple("nearest_topk","source,target,value")

#valid_examples = np.random.choice(valid_window, valid_size, replace=False)

def find_topk(reverse_dictionary, embedding, valid_examples):
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    valid_embeddings = tf.nn.embedding_lookup(embedding,valid_dataset)
    nearest_topk_list=[]
    similarity = tf.matmul(
        valid_embeddings, embedding, transpose_b=True)
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        init.run()
        sim = similarity.eval()

        for i in range(len(valid_examples)):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8 #arg
            nearest = (-sim[i, :]).argsort()[1:top_k +1]
            nearest_val = (-sim[i, :])
            nearest_val.sort()
            nearest_val=-nearest_val
            log_str = "Nearest to %s:"% valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                close_sim_val = nearest_val[k+1]
                log_str = '%s %s %f,' % (log_str, close_word, close_sim_val)
                nearest_topk_value=nearest_topk(valid_word, close_word, close_sim_val)
                nearest_topk_list.append(nearest_topk_value)
            print(log_str)
    return nearest_topk_list

