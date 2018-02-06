import os
import sys
import pickle

from sklearn.manifold import TSNE

from tsne_hack import extract_sequence
from visualize import savegif

def main(dataset):
    data_path = './data/%s.pkl' % dataset 
    with open(data_path, 'rb') as f:
        X, labels = pickle.load(f)

    Y_seq = extract_sequence(TSNE(verbose=True), X)
    with open('mnist.pkl', 'wb') as f:
        pickle.dump(Y_seq, f)

    if not os.path.exists('figures'):
        os.mkdir('figures')

    lo = Y_seq.min(axis=0).min(axis=0).max()
    hi = Y_seq.max(axis=0).max(axis=0).min()
    limits = ([lo, hi], [lo, hi])
    fig_name = '%s-tsne' % dataset
    fig_path = './figures/%s.gif' % fig_name
    savegif(Y_seq, labels, fig_name, fig_path, limits=limits)

if __name__ == '__main__':
    dataset = 'mnist70k'
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    main(dataset)
