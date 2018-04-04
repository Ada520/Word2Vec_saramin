import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
plt.rcParams["font.family"] = "NanumGothic"
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    '--plot_only',
    type=int,
    default=500,
    help="How many dots in the plot")
parser.add_argument(
    '--embedding_table',
    type=str,
    default=None,
    help='embedding table pickle file')
parser.add_argument(
    '--reverse_dictionary',
    type=str,
    default=None,
    help='reverse dictionary pickle file')
parser.add_argument(
    '--save_fig_name',
    type=str,
    default=None,
    help='save figure name')
FLAGS, unparsed = parser.parse_known_args()

with open(FLAGS.embedding_table,"rb") as f:
    embedding_table = pickle.load(f)
with open(FLAGS.reverse_dictionary,"rb") as f:
    reverse_dictionary = pickle.load(f)
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))
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


tsne = TSNE(
    perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
plot_only = FLAGS.plot_only
low_dim_embs = tsne.fit_transform(embedding_table[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels, FLAGS.save_fig_name)
