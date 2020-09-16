import torch

class LogisticRegression(object):
    def __init__(self, x_train, x_test, y_train, y_test, digits, learning_rate=1e-3):
        self.x_train = torch.tensor(x_train, dtype=float)
        self.x_test = torch.tensor(x_test, dtype=float)

        self.y_train = torch.reshape(torch.tensor(y_train, dtype=float), (y_train.shape[0], 1))
        self.y_test = torch.reshape(torch.tensor(y_test, dtype=float), (y_test.shape[0], 1))
        
        self.lr = learning_rate
        self.digits = digits     

    def __filter(self, train_or_test):
        if train_or_test == 'train':
            x, y = self.x_train, self.y_train
        else:
            x, y = self.x_test, self.y_test

        digit1, digit2 = self.digits
        
        idx_digit1, idx_digit2, idx_nondigit = [], [], []
        
        for idx in range(y.shape[0]):
            if digit1 == y[idx]:
                idx_digit1.append(idx)
            elif digit2 == y[idx]:
                idx_digit2.append(idx)
            else:
                idx_nondigit.append(idx)

        return idx_digit1, idx_digit2, idx_nondigit

    def __label_to_binclass(self, train_or_test='train', labels=(1, -1)):
        idx1, idx2, _ = self.__filter(train_or_test=train_or_test)
        
        if train_or_test == 'train':
            x = self.x_train.clone().detach()
            y = self.y_train.clone().detach()
        else:
            x = self.x_test.clone().detach()
            y = self.y_test.clone().detach()

        for idx in idx1:
            y[idx] = labels[0]

        for idx in idx2:
            y[idx] = labels[1]

        idxs = idx1 + idx2
        idxs.sort()
        x = x[idxs]
        y = y[idxs]

        return x, y

    def fit(self, stop):
        x, y = self.__label_to_binclass(train_or_test='train')

        theta = torch.zeros((x.shape[1], 1), dtype=float)

        for _ in range(stop):
            g = torch.mean((y * x) / (1 + torch.exp( y * (x @ theta))), 
                            axis=0, 
                            keepdim=True).T
            theta += self.lr*g
    
        return theta

    def predict(self, theta):
        x, _ = self.__label_to_binclass(train_or_test='test')

        y = 1 / (1 + torch.exp(-x @ theta))

        ones = torch.ones(y.shape[0], 1)

        return torch.where(y >= 0.5, ones, -ones)
        
    def acc(self, ypred):
        _, y = self.__label_to_binclass(train_or_test='test')
        hitrate = 0
        missclass = []
        for i in range(y.shape[0]):
            if ypred[i] == y[i]:
                hitrate += 1.0 * 100 / y.shape[0]
            else:
                missclass.append(i)
             
        
        return hitrate, missclass
        
    def show_miss(self, plt, miss):
        sample = miss[torch.randint(0, len(miss) - 1, (1,))]
        x, y = self.__label_to_binclass(labels=self.digits)
        img = torch.reshape(x[sample], (28, 28))

        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {y[sample].item()}')

        return plt    

    def set_digits(self, digits):
        self.digits = digits

class Regression(object):
    def __init__(self, x, y, size=0.8, ones=True):
        x = torch.tensor(x)
        y = torch.tensor(y)
        
        if ones:
            bias = torch.ones((x.shape[0], 1))
            x = torch.cat((bias, x), dim=1)        
        
        size = int(size * x.shape[0])
        
        self.x_train = x[:size]
        self.y_train = y[:size]
        
        self.x_test = x[size:]
        self.y_test = y[size:]
        
    def fit(self):
        x = self.x_train
        y = self.y_train
        
        # return theta
        # @ → multiply matrix
        return torch.inverse(x.T @ x) @ x.T @ y
     
    def predict(self, theta, x=None):
        if x is not None:
            return x @ theta
        
        x = self.x_test
        # return y predicted
        return x @ theta
          
    def rmse(self, y_pred, y=None):
        if y is not None:
            return torch.sqrt(torch.mean((y - y_pred) ** 2))
            
        y = self.y_test
        return torch.sqrt(torch.mean((y - y_pred) ** 2))
        
        