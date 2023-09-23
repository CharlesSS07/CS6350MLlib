
import numpy as np
import pandas as pd

import sys
sys.path.append('../')

from ID3 import ID3
from DecisionTree.purity_metrics import majority_error, gini_index

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

def compute_error(dataframe):
    error = 0
    for index, row in dataframe.iterrows():
        if not check_example(row, model):
            error+=1
    return error/len(dataframe)

train = pd.read_csv(
    '../data/car-4/train.csv',
    names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
)

test = pd.read_csv(
    '../data/car-4/test.csv',
    names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
)

majority_error_results = pd.DataFrame(columns=['training_error', 'testing_error'])
gini_index_results = pd.DataFrame(columns=['training_error', 'testing_error'])

for i in [1, 2, 3, 4, 5, 6]:
    model = ID3(
        train,
        'label',
        binary_purity_metric=majority_error,
        max_depth=i
    )
    
    majority_error_results.loc[i] = [compute_error(train), compute_error(test)]
    
    model = ID3(
        train,
        'label',
        binary_purity_metric=gini_index,
        max_depth=i
    )
    
    training_error = compute_error(train)
    
    testing_error = compute_error(test)
    
    gini_index_results.loc[i] = [compute_error(train), compute_error(test)]
    
majority_error_results.to_csv('./HW_1_part2_2b_majority_error.csv')
gini_index_results.to_csv('./HW_1_part2_2b_gini_index.csv')

print('Majority Error Results')
print(majority_error_results)

print('Gini Index Results')
print(gini_index_results)
