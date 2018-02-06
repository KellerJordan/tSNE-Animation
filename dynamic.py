import pickle
from visualize import savegif

def main(dataset):
    
    data_path = './data/%s.pkl' % dataset 
    with open(data_path, 'rb') as f:
        X, labels = pickle.load(f)

    with open('mnist.pkl', 'rb') as f:
        Y_seq = pickle.load(f)
    
    fig_name = '%s-tsne' % dataset
    fig_path = './figures/%s.gif' % fig_name
    savegif(Y_seq, labels, fig_name, fig_path)

if __name__ == '__main__':
    main('mnist70k')
