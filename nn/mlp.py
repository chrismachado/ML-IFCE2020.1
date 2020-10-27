import torch
import numpy as np

from torch.nn import Linear
from torch.nn import CrossEntropyLoss
from torch.nn import Module
from torch.optim import SGD
from torch.nn.functional import relu, softmax


class MLP(Module):
    def __init__(self, input_layer,
                 hidden_layer,
                 output_layer,
                 epochs,
                 lr,
                 train_samples,
                 test_samples):
        super(Module, self).__init__()
        self.input_layer = Linear(input_layer, hidden_layer)
        self.hidden_layer = Linear(hidden_layer, output_layer)
        self.epochs = epochs
        self.lr = lr
        self.train_samples = train_samples
        self.test_samples = test_samples

        self.loss = 0
        self.loss_per_epoch = []
        self.loss_per_iter = []

        self.last_acc = 0

    def forward(self, X):
        out = self.input_layer(X)
        out = relu(out)
        out = self.hidden_layer(out)
        return out

    def fit(self, **options):
        last_epoch = options.pop('last_epoch', None)
        show_progress = options.pop('show_progress', None)
        per_iter = options.pop('per_iter', None)
        per_iter_frac = options.pop('per_iter_frac', 100)

        optimizer = SGD(self.parameters(), lr=self.lr)
        loss_func = CrossEntropyLoss()

        _iter = 0
        for epoch in range(self.epochs):
            self.train()  # modo treinamento
            for i, (targets, labels) in enumerate(self.train_samples):
                targets = targets.view(-1, targets.shape[1])
                labels = labels

                optimizer.zero_grad()

                outputs = self.forward(targets)

                self.loss = loss_func(outputs, labels)  # soft max: cross entropy

                self.loss.backward()
                self.loss_per_iter.append(self.loss.item())
                optimizer.step()

                _iter += 1
                if per_iter and i % per_iter_frac:
                    self.__hit_rate(_iter)
            if (epoch % 2 == 0 and show_progress) or \
                    (last_epoch and epoch == self.epochs - 1):
                self.__hit_rate(epoch + 1)
            self.loss_per_epoch.append(self.loss.item())

    def __hit_rate(self, _iter):
        self.eval()  # modo avaliacao

        hit = 0
        total = 0
        with torch.no_grad():
            for targets, labels in self.test_samples:
                targets = targets.view(-1, targets.shape[1])
                outputs = self.forward(targets)

                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)

                hit += (pred == labels).sum().float()
            acc = 100 * hit / total

            print(
                f'Epoch: {_iter: 3d} | Loss: {self.loss.item(): .3f} | Acc: {acc: .2f}% ')
            self.last_acc = acc

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        pred = softmax(self.forward(x), dim=1)
        _, ans = torch.max(pred, 1)
        return ans
