import numpy as np

class activations:
    
    @staticmethod
    def identity(x):
        return x
    @staticmethod
    def d_identity_dx(x):
        return np.ones(shape=x.shape)

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    @staticmethod
    def d_sigmoid_dx(x):
        return activations.sigmoid(x)*(1-activations.sigmoid(x))

class DenseLayer:
    
    def __init__(self, activation_function, dactivation_function, use_bias=False):
        self.activation_function=activation_function
        self.dactivation_function=dactivation_function
        self.use_bias=use_bias
    
    def foward_pass(self, x):
        x = np.array(x)
        # print(x)
        s = np.matmul(self.w, x)
        if self.use_bias:
            return np.array([1, *self.activation_function(s)])
        else:
            return np.array(self.activation_function(s))
    
    def backward_pass(self, x):
        x = np.array(x)
        
        # print('w.shape', self.w.shape)
        
        # print('x.shape', x.shape)
        
        s = np.matmul(self.w, x)
        
        # print('s.shape', s.shape)
        
        dactivation = np.array(self.dactivation_function(s))
        
        # print('dactivation.shape', dactivation.shape)
        
        # d/dx
        dsdx = self.w.T
        
        dx = np.matmul(dsdx, dactivation)
        
        # d/dw
        dsdw = x
        
        dw = np.array([ x*dsdw for x in dactivation ])
        
        # print('dw', dw)
        
        return dx, dw

def evaluate_nn(layers:list, x):
    
    state = x
    
    for l in layers:
        state = l.foward_pass(state)
    
    return state