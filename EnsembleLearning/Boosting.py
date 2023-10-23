import pandas as pd
import numpy as np

class AbstractWeakBoostableClassifier:
    
    def __init__(self):
        pass
    
    def __predict__(self, x):
        raise Exception('__predict__ must be overridden')
    
    def predict(self, x):
        return np.sign(self.__predict__(x))
    
    def __call__(self, x):
        return self.predict(x)

class Booster:
    
    def __init__(self, find_classifier_callback: callable, iterations: int, labels: list, data: pd.DataFrame):
        '''
        :param find_classifier_callback: must be a callable function which returns a function for invoking a new trained/found classifier. The output of the classifier function must be in {-1, 1}. Its input is an example.
        '''
        assert len(labels)==len(data), 'data does not match labels'
        
        self.classifiers = []
        self.alphas = []

        m = data.shape[0]

        D = np.ones(m)/m # initializing weights to 1/m

        for iteration in range(iterations):
            classifier = find_classifier_callback(data)
            self.classifiers.append(classifier)
            
            preds = list(map(classifier, data.iloc))
            
            epsilon = (1/2) - ( (1/2) * np.sum([ D_i * y_i * p_i for D_i, y_i, p_i in zip(D, labels, preds) ]))
            # from page 25 of https://users.cs.utah.edu/~zhe/teach/pdf/ensemble-learning.pdf

            alpha = (1/2) * np.log( (1-epsilon)/epsilon )
            # from page 54 of https://users.cs.utah.edu/~zhe/teach/pdf/ensemble-learning.pdf
            self.alphas.append(alpha)
            
            np.multiply(
                D,
                np.array(
                    [
                        np.exp( (-alpha) * y_i * p_i )
                        for y_i, p_i in
                        zip(labels, preds)
                    ]
                ),
                out=D
            )
            
            # normalize D to have sum of 1
            np.multiply(D, 1/np.sum(D), out=D)
        
    def predict(self, x):
        
        return np.sign(np.sum(
            [
                a_i*p_i
                for a_i, p_i in
                zip(
                    self.alphas,
                    map(lambda c:c(x), self.classifiers)
                )
            ]
        ))
            
        