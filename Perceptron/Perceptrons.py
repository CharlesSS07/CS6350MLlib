import numpy as np

class StandardPerceptron:
    
    def __init__(self, data: np.array, labels: list, r: float, T: int):
        
        assert len(data.shape)==2, 'individual example in data must be flattened'
        
        assert len(data)==len(labels), 'must be a label for each index in data'
        
        # line data with ones in last row. this is for the bias parameter
        # "fold bias parameter into weights"
        tmp_data = np.ones((data.shape[0], data.shape[1]+1))
        tmp_data[:,:-1] = data
        data = tmp_data.astype(np.float64)
        del tmp_data
        
        l = data.shape[1]
        
        self.w = np.zeros(shape=l, dtype=np.float64)
        
        data_order = list(range(0, len(data)))
        for epoch in range(T):
            np.random.shuffle(data_order)
            for i in data_order:
                x_i = data[i]
                y_i = labels[i]
                s = y_i*np.matmul(self.w, x_i.T)
                if s <= 0: #y_i!=y_prime
                    self.w+=r*(y_i*x_i)
                # elif s > 0:
                #     pass
                # else:
                #     pass
                    
    
    def predict(self, x):
        
        return np.sign(np.matmul(self.w, x.T))

class VotedPerceptron:
    
    def __init__(self, data: np.array, labels: list, r: float, T: int):
        
        assert len(data.shape)==2, 'individual example in data must be flattened'
        
        assert len(data)==len(labels), 'must be a label for each index in data'
        
        # line data with ones in last row. this is for the bias parameter
        # "fold bias parameter into weights"
        tmp_data = np.ones((data.shape[0], data.shape[1]+1))
        tmp_data[:,:-1] = data
        data = tmp_data.astype(np.float64)
        del tmp_data
        
        l = data.shape[1]
        
        self.w = [np.zeros(shape=l, dtype=np.float64)]
        self.C = [0]
        
        data_order = list(range(0, len(data)))
        for epoch in range(T):
            np.random.shuffle(data_order)
            for i in data_order:
                x_i = data[i]
                y_i = labels[i]
                if y_i * np.matmul( self.w[-1], x_i.T ) <= 0:
                    self.w.append( self.w[-1] + r*(y_i*x_i) )
                    self.C.append(1)
                else:
                    self.C[-1]+=1
        
        self.w = np.array(self.w)
        self.C = np.array(self.C)
        print(self.C.shape, self.w.shape, np.sum(self.C)/T)
    
    def predict(self, x):
        
        return np.sign( np.sum( np.sign( np.multiply( self.C,  np.matmul(self.w, x.T) ) ) ) )


class AveragePerceptron:
    
    def __init__(self, data: np.array, labels: list, r: float, T: int):
        
        assert len(data.shape)==2, 'individual example in data must be flattened'
        
        assert len(data)==len(labels), 'must be a label for each index in data'
        
        # line data with ones in last row. this is for the bias parameter
        # "fold bias parameter into weights"
        tmp_data = np.ones((data.shape[0], data.shape[1]+1))
        tmp_data[:,:-1] = data
        data = tmp_data.astype(np.float64)
        del tmp_data
        
        l = data.shape[1]
        
        w = np.zeros(shape=l, dtype=np.float64)
        self.a = np.zeros(shape=l, dtype=np.float64)
        
        data_order = list(range(0, len(data)))
        for epoch in range(T):
            np.random.shuffle(data_order)
            for i in data_order:
                x_i = data[i]
                y_i = labels[i]
                # y_prime = self.predict(x_i)
                if y_i*np.matmul(w, x_i.T) <= 0: #y_i!=y_prime
                    w+=r*(y_i*x_i)
                self.a+=w
    
    def predict(self, x):
        
        return np.sign(np.matmul(self.a, x.T))


