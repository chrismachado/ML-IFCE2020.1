import torch
import numpy as np
import matplotlib.pyplot as plt


def predict(x, clf):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    ans = clf.predict(x)
    return ans.numpy()


def plot_decision_boundary(pred_func, X, y):
    if X.shape[1] != 2:
        return

    n_classes = len(np.unique(y))
    plot_colors = "bry"

    x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
    y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, levels=4)
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)

    plt.show()
