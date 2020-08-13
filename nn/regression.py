import torch
        
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
        
        