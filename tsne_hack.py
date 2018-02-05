import time

import numpy as np
from sklearn import manifold

def extract_sequence(tsne, X):
    sklearn_grad = manifold.t_sne._gradient_descent
    Y_seq = []

    # modified from sklearn source https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/manifold/t_sne.py#L442
    # to save the sequence of embeddings at each training iteration
    def _gradient_descent(objective, p0, it, n_iter,
                          n_iter_check=1, n_iter_without_progress=300,
                          momentum=0.8, learning_rate=200.0, min_gain=0.01,
                          min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(np.float).max
        best_error = np.finfo(np.float).max
        best_iter = i = it

        tic = time()
        for i in range(it, n_iter):

            # save the current state
            Y_seq.append(p.copy().reshape(-1, 2))

            error, grad = objective(p, *args, **kwargs)
            grad_norm = linalg.norm(grad)

            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

            if (i + 1) % n_iter_check == 0:
                toc = time()
                duration = toc - tic
                tic = toc

                if verbose >= 2:
                    print("[t-SNE] Iteration %d: error = %.7f,"
                          " gradient norm = %.7f"
                          " (%s iterations in %0.3fs)"
                          % (i + 1, error, grad_norm, n_iter_check, duration))

                if error < best_error:
                    best_error = error
                    best_iter = i
                elif i - best_iter > n_iter_without_progress:
                    if verbose >= 2:
                        print("[t-SNE] Iteration %d: did not make any progress "
                              "during the last %d episodes. Finished."
                              % (i + 1, n_iter_without_progress))
                    break
                if grad_norm <= min_grad_norm:
                    if verbose >= 2:
                        print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                              % (i + 1, grad_norm))
                    break

        return p, error, i

    # replace with modified gradient descent
    manifold.t_sne._gradient_descent = _gradient_descent
    # train given tsne object with new gradient function
    X_proj = tsne.fit_transform(X)
    # return to default version
    manifold.t_sne._gradient_descent = sklearn_grad
    
    return Y_seq
  
