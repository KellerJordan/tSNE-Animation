"""
Code modified from
https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py

Run this from within /data/!
"""

import numpy as np
from urllib import request
import gzip
import pickle

filename = [
    ["training_images","train-images-idx3-ubyte.gz"],
    ["test_images","t10k-images-idx3-ubyte.gz"],
    ["training_labels","train-labels-idx1-ubyte.gz"],
    ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    
    # modification: merge training and test datasets
    mnist_merge = {}
    mnist_merge['images'] = np.concatenate([mnist['training_images'], mnist['test_images']])
    mnist_merge['labels'] = np.concatenate([mnist['training_labels'], mnist['test_labels']])
    datasets = {}
    datasets['mnist70k'] = (mnist_merge['images'], mnist_merge['labels'])
#     datasets['mnist10k'] = (mnist_merge['images'][:10000], mnist_merge['labels'][:10000])
#     datasets['mnist2500'] = (mnist_merge['images'][:2500], mnist_merge['labels'][:2500])
#     datasets['mnist250'] = (mnist_merge['images'][:250], mnist_merge['labels'][:250])

    for name, data in datasets.items():
        with open('%s.pkl' % name, 'wb') as f:
            pickle.dump(data, f)
    print("Save complete.")

if __name__ == '__main__':
    download_mnist()
    save_mnist()
