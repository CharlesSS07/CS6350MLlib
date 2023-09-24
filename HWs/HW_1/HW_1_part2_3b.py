
# 3b) Let us consider ``unknown'' as a particular attribute value, and hence we do not have any missing attributes for both training and test. Vary the maximum  tree depth from $1$ to $16$ --- for each setting, run your algorithm to learn a decision tree, and use the tree to  predict both the training  and test examples. Again, if your tree cannot grow up to $16$ levels, stop at the maximum level. Report in a table the average prediction errors on each dataset when you use information gain, majority error and gini index heuristics, respectively.

import numpy as np
import pandas as pd

import sys
sys.path.append('../../')

from DecisionTree.ID3 import ID3, error, most_common_label
from DecisionTree.purity import entropy, majority_error, gini_index

train = pd.read_csv(
    '../../data/bank-4/train.csv',
    names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
)

bank_data_values = {
    'age': ['< 50 precentile', '> 50 precentile'],
    'job': ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],
    'marital': ["married","divorced","single"],
    'education': ["unknown","secondary","primary","tertiary"],
    'default': ["yes","no"],
    'balance': ['< 50 precentile', '> 50 precentile'],
    'housing': ["yes","no"],
    'loan': ["yes","no"],
    'contact': ["unknown","telephone","cellular"],
    'day': ['< 50 precentile', '> 50 precentile'],
    'month': ['apr', 'aug', 'dec', 'feb', 'jan', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep'],
    'duration': ['< 50 precentile', '> 50 precentile'],
    'campaign': ['< 50 precentile', '> 50 precentile'],
    'pdays': ['not previously contacted', '< 50 precentile', '> 50 precentile'],
    'previous': ['< 50 precentile', '> 50 precentile'],
    'poutcome': ["unknown","other","failure","success"],
    'y': ["yes","no"]
}
# train_values

median_age = np.median(train.age)
median_balance = np.median(train.balance)
median_day = np.median(train.day)
median_duration = np.median(train.duration)
median_campaign = np.median(train.campaign)
median_pdays = np.median(train.pdays[train.pdays!=-1])
median_previous = np.median(train.previous)

def discretize_bank_data(ds):
    return pd.DataFrame(
        {
            'age': [ '< 50 precentile' if age>median_age else '> 50 precentile' for age in ds.age],
            'job': ds.job,
            'marital': ds.marital,
            'education': ds.education,
            'default': ds.default,
            'balance': [ '< 50 precentile' if balance>median_balance else '> 50 precentile' for balance in ds.balance],
            'housing': ds.housing,
            'loan': ds.loan,
            'contact': ds.contact,
            'day': [ '< 50 precentile' if day>median_day else '> 50 precentile' for day in ds.day],
            'month': ds.month,
            'duration': [ '< 50 precentile' if duration>median_duration else '> 50 precentile' for duration in ds.duration],
            'campaign': [ '< 50 precentile' if campaign>median_campaign else '> 50 precentile' for campaign in ds.campaign],
            'pdays': [ 'not previously contacted' if pdays==-1 else ('< 50 precentile' if pdays>median_pdays else '> 50 precentile') for pdays in ds.pdays],
            'previous': [ '< 50 precentile' if previous>median_previous else '> 50 precentile' for previous in ds.previous],
            'poutcome': ds.poutcome,
            'y': ds.y
        }
    )

def replace_unknowns_with_majority_label(ds):
    
    for k in ds:
        majority_label = most_common_label(ds[k])
        ds[k] = [ i if i!='unknown' else majority_label for i in ds[k]]
    
    return ds

train_discretized = replace_unknowns_with_majority_label(discretize_bank_data(train))

test_discretized = replace_unknowns_with_majority_label(discretize_bank_data(pd.read_csv(
    '../../data/bank-4/test.csv',
    names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
)))

del train # make sure I don't accidentally use wrong dataset


results = {}
results['information gain'] = pd.DataFrame(columns=['training_error', 'testing_error'])
results['majority error'] = pd.DataFrame(columns=['training_error', 'testing_error'])
results['gini index'] = pd.DataFrame(columns=['training_error', 'testing_error'])

for i in range(1,17):

    print(f'{i}/16')
    
    model = ID3(
        train_discretized,
        'y',
        bank_data_values['y'],
        attribute_values=bank_data_values,
        purity_metric=entropy,
        max_depth=i
    )
    
    results['information gain'].loc[i] = [error(train_discretized, model, 'y'), error(test_discretized, model, 'y')]
    
    del model
    
    model = ID3(
        train_discretized,
        'y',
        bank_data_values['y'],
        attribute_values=bank_data_values,
        purity_metric=majority_error,
        max_depth=i
    )
    
    results['majority error'].loc[i] = [error(train_discretized, model, 'y'), error(test_discretized, model, 'y')]
    
    del model
    
    model = ID3(
        train_discretized,
        'y',
        bank_data_values['y'],
        attribute_values=bank_data_values,
        purity_metric=gini_index,
        max_depth=i
    )
    
    results['gini index'].loc[i] = [error(train_discretized, model, 'y'), error(test_discretized, model, 'y')]
    
    del model

print('Information Gain Results')

print(results['information gain'])

print('Majroity Error Results')

print(results['majority error'])

print('Gini Index Results')

print(results['gini index'])
