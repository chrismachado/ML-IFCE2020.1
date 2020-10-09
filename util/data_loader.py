import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def data_loader(targets, labels):
    batch_size = 10
    train_samples, test_samples, train_labels, test_labels = train_test_split(targets, labels, test_size=0.2)
    train_samples = torch.FloatTensor(train_samples)
    train_labels = torch.LongTensor(train_labels)

    test_samples = torch.FloatTensor(test_samples)
    test_labels = torch.LongTensor(test_labels)
    trainD = TensorDataset(train_samples, train_labels)
    testD = TensorDataset(test_samples, test_labels)
    train_loader = DataLoader(trainD, batch_size, shuffle=True)
    test_loader = DataLoader(testD, batch_size, shuffle=False)

    return train_loader, test_loader