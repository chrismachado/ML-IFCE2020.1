import torch

class KNN(object):
    def __init__(self, K, samples_train, samples_targets, regression=False):
        self.K = K
        self.samples = samples_train
        self.targets = samples_targets
        self.regression = regression

    def predict(self, x):
        cdists = torch.cdist(x, self.samples)
        idxs = cdists.argsort()[:, :self.K]
        knns = self.targets[idxs]

        if self.regression:
            return knns.mean(dim=1)
        
        return knns.mode(dim=1).values
