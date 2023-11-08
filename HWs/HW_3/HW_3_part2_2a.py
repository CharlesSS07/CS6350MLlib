'''
We will implement Perceptron for a binary classification task — bank-note authentication. Please download the data “bank-note.zip” from Canvas. The features and labels are listed in the file “bank-note/data-desc.txt”. The training data are stored in the file 'bank-note/train.csv', consisting of 872 examples. The test data are stored in 'bank-note/test.csv', and comprise of 500 examples. In both the training and testing datasets, feature values and labels are separated by commas.
(a) [16 points] Implement the standard Perceptron. Set the maximum number of epochs T to 10. Report your learned weight vector, and the average prediction error on the test dataset.'''

import pandas as pd
import numpy as np

import sys
sys.path.append('../../')

print('HW3 Practice Problem (2a)')

from Perceptron.Perceptrons import StandardPerceptron

train = np.array(pd.read_csv(
    '../../data/bank-note/train.csv',
    names=['variance', 'skewness', 'curtosis', 'entropy', 'y']
))

# need to map labels from 0,1 to -1,1
def map_labels(y01):
    return 2 * (y01-0.5)

p = StandardPerceptron(
    data=train[:,:-1],
    labels=map_labels(train[:,-1]),
    r=0.01,
    T=10
)

test = np.array(pd.read_csv(
    '../../data/bank-note/test.csv',
    names=['variance', 'skewness', 'curtosis', 'entropy', 'y']
))
test_labels = map_labels(test[:,-1])
test[:,-1] = 1

errors = 0
for i in range(len(test)):
    if p.predict(test[i])!=test_labels[i]:
        errors+=1

print('Testing Errors:', errors)

print('W:', p.w)