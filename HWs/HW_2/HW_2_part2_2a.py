'''
2. We will implement the boosting and bagging algorithms based on decision trees. Let us test them on the bank marketing dataset in HW1 (bank.zip in Canvas). We use the same approach to convert the numerical features into binary ones. That is, we choose the media (NOT the average) of the attribute values (in the training set) as the threshold, and examine if the feature is bigger (or less) than the threshold. For simplicity, we treat “unknown” as a particular attribute value, and hence we do not have any missing attributes for both training and test.

a) Modify your decision tree learning algorithm to learn decision stumps — trees with only two levels. Specifically, compute the information gain to select the best feature to split the data. Then for each subset, create a leaf node. Note that your decision stumps must support weighted training examples. Based on your decision stump learning algorithm, implement AdaBoost algorithm. Vary the number of iterations T from 1 to 500, and examine the training and test errors. You should report the results in two figures. The first figure shows how the training and test errors vary along with T. The second figure shows the training and test errors of all the decision stumps learned in each iteration. What can you observe and conclude? You have had the results for a fully expanded decision tree in HW1. Comparing them with Adaboost, what can you observe and conclude?

Steps:

-[x] Get a decision stump working on banking data w/ information gain
-[x] Make decision stumps support weighted training examples
-[x] Implement AdaBoost algorithim
-[ ] Graph training & testing error plots

'''

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import sys
sys.path.append('../../')

from DecisionTree.ID3 import ID3, error, most_common_label
from DecisionTree.purity import entropy, majority_error, gini_index
from EnsembleLearning.Boosting import Booster, AbstractWeakBoostableClassifier

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

del train 

def binarize_labels(label):
    if label=='yes':
        return 1
    if label=='no':
        return -1
    raise Exception('This should not happen')

class BoostableDecisionTreeClassifier(AbstractWeakBoostableClassifier):
    
    def __init__(self, decision_tree: dict):
        self.decision_tree = decision_tree
    
    def __predict__(self, x: pd.DataFrame):
        node = self.decision_tree
        
        while True:
            if not type(node)==dict:
                return binarize_labels(node)
            keys = list(node.keys())
            attribute = keys[0].split('=')[0]
            # breakpoint()
            value = x[attribute]
            
            node = node[attribute+'='+str(value)]

def find_classifier_callback(examples, example_weights):
    model = ID3(
        examples,
        'y',
        bank_data_values['y'],
        attribute_values=bank_data_values,
        purity_metric=entropy,
        max_depth=2, # just a stump
        example_weights=example_weights
    )
    return BoostableDecisionTreeClassifier(model)

def measure_training_and_testing_errors(classifier):
    testing_errors = 0
    for i, row in test_discretized.iterrows():
        if binarize_labels(row['y'])!=classifier.predict(row):
            testing_errors+=1
            
    training_errors = 0
    for i, row in train_discretized.iterrows():
        if binarize_labels(row['y'])!=classifier.predict(row):
            training_errors+=1
    
    return testing_errors, training_errors

boosted_results = pd.DataFrame(columns=['iterations', 'training_error', 'testing_error'])

max_iteration = 500

# slow way
# for i, iterations in enumerate([*range(1, 20), 40, 60, 80, 100, 150, 200, 300, 400, 500]):
    
#     print('Iterations:', iterations)
    
#     b = Booster(
#         find_classifier_callback=find_classifier_callback,
#         iterations=iterations,
#         labels=list(map(binarize_labels, train_discretized['y'])),
#         data=train_discretized
#     )
    
#     testing_errors, training_errors = measure_training_and_testing_errors(b)
    
#     boosted_results.loc[i] = [iterations, testing_errors, training_errors]
    
#     del b

# only need to train one classifier with 500 iterations, and can then truncate alphas and classifiers lists

# fast way
b = Booster(
    find_classifier_callback=find_classifier_callback,
    iterations=max_iteration,
    labels=list(map(binarize_labels, train_discretized['y'])),
    data=train_discretized
)

# This is an optimization so that evaluating error for every t from 0 to 500 is not painstaking
# It basically pre-processes the raw predictions for every single subclassifier into a format
# which makes evaluating the boosted prediction given only the classifiers from 0 to iterations
# really easy and fast. Reduces the number of redundant calculations.
subclassifier_training_preds = np.array(
    [
        np.multiply(
            b.alphas,
            [
                h(x)
                for h in b.classifiers
            ]
        )
        for _,x in train_discretized.iterrows()
    ]
).T
labels_training = np.array(
    [binarize_labels(row['y']) for _,row in train_discretized.iterrows()])

subclassifier_test_preds = np.array(
    [
        np.multiply(
            b.alphas,
            [
                h(x)
                for h in b.classifiers
            ]
        )
        for _,x in test_discretized.iterrows()
    ]
).T
labels_testing = np.array(
    [binarize_labels(row['y']) for _,row in test_discretized.iterrows()])

for i, iterations in tqdm(list(enumerate(range(1, max_iteration))), desc='Evaluating Boosted Classifiers'):
    
    testing_errors = np.sum(
        [ p_i!=y_i for p_i, y_i in zip(
            np.sign(np.sum(
                subclassifier_test_preds[:iterations], axis=0
            )), labels_testing) ])
    training_errors = np.sum(
        [ p_i!=y_i for p_i, y_i in zip(
            np.sign(np.sum(
                subclassifier_training_preds[:iterations], axis=0
            )), labels_testing) ])
    
    boosted_results.loc[i] = [iterations, testing_errors, training_errors]
    
boosted_results.to_csv('./results/HW_2_part2_2a_errors_to_iterations.csv')
# See Plot HW_2_part2_2a_results.ipynb notebook for plotting of above data.

classifier_errors = pd.DataFrame(columns=['classifier', 'training_error', 'testing_error'])

# Because the boosted model with 500 classifier shares 499 of its classifiers with the boosted model with 499 classifiers, and etc. down to 1, we can just use the final boosted models classifiers and evaluate the errors to get the second plot
for i, c in tqdm(list(enumerate(b.classifiers)), desc='Evaluating Booster SubClassifiers'):
    
    testing_errors, training_errors = measure_training_and_testing_errors(c)
    
    classifier_errors.loc[i] = [i+1, testing_errors, training_errors]
    
classifier_errors.to_csv('./results/HW_2_part2_2a_classifier_errors.csv')

# Plotting
