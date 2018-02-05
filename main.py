import pickle

from sklearn.manifold import TSNE

from tsne_hack import extract_sequence
from visualize import savegif

data_path = '../TriMap-PyTorch/data/mnist70k.pkl' # change this to your situation
with open(data_path, 'rb') as f:
    X, labels = pickle.load(f)

Y_seq = extract_sequence(TSNE(verbose=True), X)
with open('mnist.pkl', 'wb') as f:
    pickle.dump(Y_seq, f)

visualize.savegif(Y_seq, labels, 'mnist10k-tsne', './mnist10k-tsne.gif')
