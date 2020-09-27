import numpy as np

def augment(X):
    N, d = X.shape
    Xb = -1*np.ones((N, 1))
    X = np.hstack((X, Xb))
    return X


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


def net_signal(w, x):
    w = np.reshape(w, (-1,len(w)))
    x = np.reshape(x, (len(x), -1))
    return w.dot(x)