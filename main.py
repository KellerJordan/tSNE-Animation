import os
import sys
import pickle
import argparse

from sklearn.manifold import TSNE

from tsne_hack import extract_sequence
from visualize import savegif

def main(args):
    data_path = './data/%s.pkl' % args.dataset 
    with open(data_path, 'rb') as f:
        X, labels = pickle.load(f)

    tsne = TSNE(n_iter=args.early_iters, verbose=True)
    tsne._EXPLORATION_N_ITER = args.early_iters
    Y_seq = extract_sequence(tsne, X)
    with open('mnist.pkl', 'wb') as f:
        pickle.dump(Y_seq, f)

    if not os.path.exists('figures'):
        os.mkdir('figures')

    lo = Y_seq.min(axis=0).min(axis=0).max()
    hi = Y_seq.max(axis=0).max(axis=0).min()
    limits = ([lo, hi], [lo, hi])
    fig_name = '%s-%d-%d-tsne' % (args.dataset, args.num_iters, args.early_iters)
    fig_path = './figures/%s.gif' % fig_name
    savegif(Y_seq, labels, fig_name, fig_path, limits=limits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist70k')
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--early_iters', type=int, default=250)
    args = parser.parse_args()
    main(args)
