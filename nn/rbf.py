import torch
import numpy as np


from sklearn.model_selection import train_test_split

class RBF(object):
    def __init__(self, x, y, gamma, c):
        x = torch.tensor(x)
        y = torch.tensor(y)

        self.x_train, \
        self.x_test, \
        self.y_train, \
        self.y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)
        
        self.set_params(c=c, gamma=gamma, x=x)

    def set_params(self, c, gamma, x=None):
        if c > self.x_train.shape[0]:
            raise ValueError("Defina um valor correto para as centroides")

        if x is None:
            x = self.x_train
            is_train = True
        else:
         is_train = False

        self.c, _x = define_centroid(c=c, x=x)
        self.gamma = gamma

        if is_train:
            self.x_train = _x

    def fit(self):
        z = torch.cdist(self.x_train, self.c)

        bias = torch.ones((z.shape[0], 1))
        z = torch.cat((bias, z), dim=1)

        return torch.inverse(z.T @ z) @ z.T @ self.y_train
        

    def predict(self, thetas):
        z = torch.cdist(self.x_test, self.c)
        z = torch.exp(-self.gamma * z) 

        bias = torch.ones((z.shape[0], 1))
        z = torch.cat((bias, z), dim=1)

        return z @ thetas

    def rmse(self, y_pred, y=None):
        if y is not None:
            return torch.sqrt(torch.mean((y - y_pred) ** 2))
            
        y = self.y_test
        return torch.sqrt(torch.mean((y - y_pred) ** 2))

def define_centroid(c, x):
    centroids = []
    indexes = []

    while len(centroids) < c:
        index = torch.randint( x.shape[0], (1,)).item()
        indexes.append(index)

        if index in indexes:
            centroids.append(x.T[:, index].tolist())

    x = np.delete(x.numpy(), indexes, axis=0)

    return torch.tensor(centroids).double(), torch.tensor(x)