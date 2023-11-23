import pandas as pd
import numpy as np

import sys
sys.path.append('../../')

print('HW4 Practice Problem (2)')

from SVM import Primal

train = np.array(pd.read_csv(
    '../../data/bank-note/train.csv',
    names=['variance', 'skewness', 'curtosis', 'entropy', 'y']
))

# need to map labels from 0,1 to -1,1
def map_labels(y01):
    return 2 * (y01-0.5)

X = train[:,:-1].T
y = map_labels(train[:,-1])

primal = Primal(X, y, lambda x: x**2, C=1)

print('Doing SGD on Primal SVM')
primal.sgd()

test = np.array(pd.read_csv(
    '../../data/bank-note/test.csv',
    names=['variance', 'skewness', 'curtosis', 'entropy', 'y']
))

print('Predicting on test set:')

primal.predict(test[0,:-1])