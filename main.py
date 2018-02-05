import os
import pickle

from sklearn.manifold import TSNE

from tsne_hack import extract_sequence
from visualize import savegif

def main():
    data_path = './data/mnist70k.pkl' 
    with open(data_path, 'rb') as f:
        X, labels = pickle.load(f)

    Y_seq = extract_sequence(TSNE(verbose=True), X)
    with open('mnist.pkl', 'wb') as f:
        pickle.dump(Y_seq, f)

    if not os.path.exists('figures'):
        os.mkdir('figures')

    savegif(Y_seq, labels, 'mnist70k-tsne', './figures/mnist70k-tsne.gif')

if __name__ == '__main__':
    main()
