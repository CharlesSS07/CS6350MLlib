
import numpy as np
import pandas as pd

import sys
sys.path.append('../../')

from DecisionTree.ID3 import ID3
from DecisionTree.purity import entropy, majority_error, gini_index

def check_example(example, node):
    global errors
    if type(node)==str:
        if not example['label']==node:
            return False
        return True
    
    keys = list(node.keys())
    
    attribute = keys[0].split('=')[0]
    
    value = example[attribute]
    
    return check_example(example, node[attribute+'='+value])

def compute_error(dataframe, model):
    error = 0
    for index, row in dataframe.iterrows():
        if not check_example(row, model):
            error+=1
    return error/len(dataframe)

train = pd.read_csv(
    '../../data/car-4/train.csv',
    names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
)

test = pd.read_csv(
    '../../data/car-4/test.csv',
    names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
)

entropy_results = pd.DataFrame(columns=['training_error', 'testing_error'])
majority_error_results = pd.DataFrame(columns=['training_error', 'testing_error'])
gini_index_results = pd.DataFrame(columns=['training_error', 'testing_error'])

for i in [1, 2, 3, 4, 5, 6]:

    model = ID3(
        train,
        'label',
        ['unacc', 'acc', 'good', 'vgood'],
        purity_metric=entropy,
        max_depth=i
    )
    
    entropy_results.loc[i] = [compute_error(train, model), compute_error(test, model)]
    
    del model
    
    model = ID3(
        train,
        'label',
        ['unacc', 'acc', 'good', 'vgood'],
        purity_metric=majority_error,
        max_depth=i
    )
    
    majority_error_results.loc[i] = [compute_error(train, model), compute_error(test, model)]
    
    del model
    
    model = ID3(
        train,
        'label',
        ['unacc', 'acc', 'good', 'vgood'],
        purity_metric=gini_index,
        max_depth=i
    )
    
    gini_index_results.loc[i] = [compute_error(train, model), compute_error(test, model)]
    
    del model
    
entropy_results.to_csv('./results/HW_1_part2_2b_entropy.csv')
majority_error_results.to_csv('./results/HW_1_part2_2b_majority_error.csv')
gini_index_results.to_csv('./results/HW_1_part2_2b_gini_index.csv')

print('Information Gain (Entropy) Error Results')
print(entropy_results)

print('Majority Error Results')
print(majority_error_results)

print('Gini Index Results')
print(gini_index_results)
