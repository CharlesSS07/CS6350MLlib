import numpy as np

class Primal():
    
    def __init__(self, X, y, kernel, C):
        
        self.kernel = kernel
        
        self.X = np.concatenate([X, list(map(self.kernel, X))])
        self.y = y
        
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        
        assert self.n==len(y), 'Data must be paired with labels.'
        
        self.alphas = np.zeros(shape=self.m, dtype=np.float64)
        self.w = np.zeros(shape=self.m, dtype=np.float64)
        self.b = 0
        
        self.C = C
        
        self.epoch = 1
        
    def objective(self):
        # page 3 of
        # https://users.cs.utah.edu/~zhe/teach/pdf/svm-sgd.pdf
        regularization_term = (1/2) * np.matmul(self.w.T, self.w)
        preds = np.matmul(self.w.T, self.data_processed) + self.b
        empiracle_loss_term = self.C * np.sum( np.max( 0, 1 - self.labels * preds ) )
        return regularization_term + empiracle_loss_term
    
    def gradient(self):
        pass
    
    def sgd(self, epochs=10000):

        eta = 1

        while self.epoch<epochs:
            if self.epoch%100:
                print('Epoch:', self.epoch)
            np.random.shuffle(self.X.T)
            for i, x in enumerate(self.X.T):
                if (self.y[i] * np.dot(x, self.w)) < 1:
                    self.w = self.w + eta * ( (x * self.y[i]) + (-2 * (1/self.epoch) * self.w) )
                else:
                    self.w = self.w + eta * ( -2  *(1/self.epoch)* self.w )
            self.epoch += 1
        
    def predict(self, example):
        
        x = np.concatenate([example, list(map(self.kernel, example))])
        
        print(x, self.w.shape, self.b)
        
        return np.sign(np.matmul(self.w.T, x) + self.b)