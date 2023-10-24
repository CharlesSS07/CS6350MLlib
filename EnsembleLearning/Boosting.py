import pandas as pd
import numpy as np
from tqdm.auto import tqdm

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
        
        labels = np.array(labels, dtype=np.float64)
        
        self.classifiers = []
        self.alphas = []

        m = data.shape[0]
        D = np.ones(m, dtype=np.float64)/m # initializing weights to 1/m

        for iteration in tqdm(list(range(iterations)), desc='Training Boosted Classifier'):
            classifier = find_classifier_callback(data, D)
            self.classifiers.append(classifier)
            
            preds = np.array([classifier(x) for _,x in data.iterrows()], dtype=np.float64)
            
            correct = np.multiply(labels, preds)
            # correct should be - for incorrect, and + for correct predictions
            
            # epsilon = np.sum(D[correct<0])
            epsilon = (1/2) - ( (1/2) * np.sum(np.multiply(D, correct)) )
            # from page 25 of https://users.cs.utah.edu/~zhe/teach/pdf/ensemble-learning.pdf
            alpha = (1/2) * np.log( (1-epsilon)/epsilon )
            # from page 54 of https://users.cs.utah.edu/~zhe/teach/pdf/ensemble-learning.pdf
            self.alphas.append(alpha)
            
            np.multiply(
                D,
                np.exp( (-alpha) * correct ),
                out=D
            )
            
            # normalize D to have sum of 1
            np.multiply(D, 1/np.sum(D), out=D)
        
    def predict(self, x):
        
        return np.sign(np.sum(
            [
                a_i*h_x
                for a_i, h_x in
                zip(
                    self.alphas,
                    map(lambda h:h(x), self.classifiers)
                )
            ]
        ))
            
        