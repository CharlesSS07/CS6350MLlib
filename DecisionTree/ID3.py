
import pandas as pd
import numpy as np

from DecisionTree.purity import purity
from DecisionTree.purity import majority_error

def most_common_label(labels):
    uniq = np.unique(labels)
    shares = [sum(labels==l) for l in uniq]
    return uniq[np.argmax(shares)]

def ID3(
    dataframe,
    label_attribute,
    label_values,
    attribute_values=None, # dictionary of attributes in dataframe to values each attribute can take on
    purity_metric=majority_error,
    max_depth=6
):

    assert(type(dataframe)==pd.DataFrame)
    
    # if attribute_values is not provided, assume dataframe has all present values
    if attribute_values is None:
        attributes = list(dataframe.columns)
        attributes.remove(label_attribute) # don't compte gain of labels column on labels...
        attribute_values = {k:np.unique(dataframe[k]) for k in attributes }
    
#    print(attribute_values)
    return __ID3__(
        dataframe = dataframe.drop(columns=label_attribute),
        attribute_values = attribute_values,
        labels = np.array( # make sure labels is a np array
             dataframe[label_attribute]
        ),
        label_values=label_values,
        purity_metric=purity_metric,
        max_depth=max_depth
    )

def __ID3__(dataframe, attribute_values, labels, label_values, purity_metric, max_depth):

    if max_depth<=0:
        return most_common_label(labels)
    
    uniq = np.unique(labels)
    
    if len(uniq)<=1:
#        print(uniq, len(labels))
        return uniq[0] # return the only label
    
#    print(len(dataframe.columns), max_depth)
    match len(dataframe.columns):
        case 0: # attributes empty
            # find most common label
            return most_common_label(labels)
        case 1: # only one attribute, can skip computing gain
            A = dataframe.columns[0]
        case _: # many attributes, compute gain
            ip = purity(
                S=dataframe,
                y=labels,
                A_values=attribute_values,
                y_values=label_values,
                purity_metric=purity_metric
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
#            print(subdataframe, labels)
            return most_common_label(labels)
        
        root_node[A+'='+str(k)] = __ID3__(
            subdataframe.drop(columns=A, inplace=False),
            attribute_values, # don't need to prune attributes, gain computation should ignore attributes not in S
            labels[subset],
            label_values=label_values,
            purity_metric=purity_metric,
            max_depth=max_depth-1
        )
    
    return root_node
