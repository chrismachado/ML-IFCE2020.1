import torch


def reader(path, splitter=','):
    samples, target = [], []
    with open(path, 'r') as csv_file:
        for line in csv_file.readlines():
            line_ = list(map(float, line.split(splitter)))
            samples.append(line_[:-1])
            target.append(line_[-1])
    return torch.tensor(samples), torch.tensor(target, dtype=int)


def regression_reader(path):
    samples = []
    try:
        with open(path, 'r') as csv_file:
            for line in csv_file.readlines():
                line_ = list(map(float, line.split(' ')))
                samples.append(line_)
    except:
        print('Algum erro ocorreu')
    finally:
        return torch.tensor(samples)


def mlp_reader(path, splitter=','):
    samples, target = [], []
    with open(path, 'r') as csv_file:
        for line in csv_file.readlines():
            line_ = list(map(float, line.split(splitter)))
            samples.append(line_)
    return torch.tensor(samples)
