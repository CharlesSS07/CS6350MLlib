
import pandas as pd
import numpy as np

from DecisionTree.purity import purity
from DecisionTree.purity import majority_error

def most_common_label(labels, label_weights=None):
    uniq = np.unique(labels)
    if label_weights is None:
        shares = [sum(labels==l) for l in uniq]
    else:
        assert len(label_weights)==len(labels)
        shares = [sum(label_weights[labels==l]) for l in uniq]
    return uniq[np.argmax(shares)]

def ID3(
    dataframe,
    label_attribute,
    label_values,
    attribute_values=None, # dictionary of attributes in dataframe to values each attribute can take on
    purity_metric=majority_error,
    max_depth=6,
    example_weights=None
):

    assert(type(dataframe)==pd.DataFrame)
    
    # if attribute_values is not provided, assume dataframe has all present values
    if attribute_values is None:
        attributes = list(dataframe.columns)
        attributes.remove(label_attribute) # don't compte gain of labels column on labels...
        attribute_values = {k:np.unique(dataframe[k]) for k in attributes }
    
    return __ID3__(
        dataframe = dataframe.drop(columns=label_attribute),
        attribute_values = attribute_values,
        labels = np.array( # make sure labels is a np array
             dataframe[label_attribute]
        ),
        label_values=label_values,
        purity_metric=purity_metric,
        max_depth=max_depth,
        example_weights=example_weights
    )

def __ID3__(dataframe, attribute_values, labels, label_values, purity_metric, max_depth, example_weights=None):

    if max_depth<=0:
        return most_common_label(labels, example_weights)
    
    uniq = np.unique(labels)
    
    if len(uniq)<=1:
        return uniq[0] # return the only label
    
    match len(dataframe.columns):
        case 0: # attributes empty
            # find most common label
            return most_common_label(labels, example_weights)
        case 1: # only one attribute, can skip computing gain
            A = dataframe.columns[0]
        case _: # many attributes, compute gain
            ip = purity(
                S=dataframe,
                y=labels,
                A_values=attribute_values,
                y_values=label_values,
                purity_metric=purity_metric,
                example_weights=example_weights
            )

            keys = []
            values = []
            for k in ip:
                keys.append(k)
                values.append(ip[k])
            
            A = keys[np.argmax(values)] # max gain attribute
    
    root_node = {}
    
    for k in attribute_values[A]:
        
        subset = dataframe[A]==k
        subdataframe = dataframe[subset]
        
        if len(subdataframe)==0: # S_v is empty
            # add leaf node with most common value of Label in S
            root_node[A+'='+str(k)] = most_common_label(labels, example_weights)
        else:
            root_node[A+'='+str(k)] = __ID3__(
                subdataframe.drop(columns=A, inplace=False),
                attribute_values, # don't need to prune attributes, gain computation should ignore attributes not in S
                labels[subset],
                label_values=label_values,
                purity_metric=purity_metric,
                max_depth=max_depth-1,
                example_weights=example_weights[subset] if not example_weights is None else None
            )
    
    return root_node

def check_example(example, id3_node, label_attribute):
    global errors
    if type(id3_node)!=dict:
        if not example[label_attribute]==id3_node:
            return False
        return True
    
    keys = list(id3_node.keys())
    
    attribute = keys[0].split('=')[0]
    
    value = example[attribute]
    
    return check_example(example, id3_node[attribute+'='+value], label_attribute)

def error(dataframe, model, label_attribute):
    error = 0
    for index, row in dataframe.iterrows():
        if not check_example(row, model, label_attribute):
            error+=1
    return error/len(dataframe)
