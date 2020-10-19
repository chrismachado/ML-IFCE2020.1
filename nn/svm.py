import torch
import numpy as np
import matplotlib.pyplot as plt

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


class SVM:
    def run(self, N, target):
        x_train = 2 * torch.rand(N, 2) - 1
        w = torch.zeros((2, 1))

        w[0, 0] = target[1, 1] - target[0, 1]
        w[1, 0] = -(target[1, 0] - target[0, 0])
        b = target[1, 0] * target[0, 1] - target[0, 0] * target[1, 1]
        x_test = 2 * torch.rand(1000, 2) - 1

        y_test = x_test @ w + b
        y_test[y_test > 0] = 1
        y_test[y_test <= 0] = -1

        y_train = x_train @ w + b  # xwT + b
        y_train[y_train > 0] = 1
        y_train[y_train <= 0] = -1

        if (y_train == 1).all() or (y_train == -1).all():
            print('Trying again, only one class was detected...')
            return run(N, target)

        xx, yy = torch.meshgrid([torch.arange(-1.5, 1.5, 0.01), torch.arange(-1.5, 1.5, 0.01)])
        full_square = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)

        y_test = full_square @ w + b
        y_test[y_test > 0] = 1
        y_test[y_test <= 0] = -1

        plt.contourf(xx, yy, y_test.reshape(xx.shape), cmap=plt.cm.terrain, alpha=0.8)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.flag)

        return x_train, x_test, y_train, y_test

    def prepare_to_solver(self, x, y):
        m, n = x.shape
        y = y.reshape(-1, 1) * 1.
        X_dash = y * x
        H = np.dot(X_dash, X_dash.T) * 1.

        H = H.astype(np.double)
        y = y
        return m, n, H


    def solve(self, m, n, H, x, y):
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(-np.eye(m))
        h = cvxopt_matrix(np.zeros(m))
        A = cvxopt_matrix(y.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))

        cvxopt_solvers.options['show_progress'] = True
        cvxopt_solvers.options['abstol'] = 1e-10
        cvxopt_solvers.options['reltol'] = 1e-10
        cvxopt_solvers.options['feastol'] = 1e-10

        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        w = ((y * alphas).T @ x).reshape(-1, 1)
        S = (alphas > 1e-4).flatten()
        b = y[S] - np.dot(x[S], w)

        return w, alphas, b
