
import pandas as pd
import numpy as np

from DecisionTree.gain import binary_gain
from DecisionTree.purity_metrics import majority_error

def most_common_label(labels):
    uniq = np.unique(labels)
    shares = [sum(labels==l) for l in uniq]
    return uniq[np.argmax(shares)]

def ID3(
    dataframe,
    label_attribute,
    attribute_values=None, # dictionary of attributes in dataframe to values each attribute can take on
    binary_purity_metric=majority_error
):

    assert(type(dataframe)==pd.DataFrame)
    
    # if attribute_values is not provided, assume dataframe has all present values
    if attribute_values is None:
        attributes = list(dataframe.columns)
        attributes.remove(label_attribute) # don't compte gain of labels column on labels...
        attribute_values = {k:np.unique(dataframe[k]) for k in attributes }
    
    return __ID3__(
        dataframe = dataframe.drop(label_attribute, axis=1),
        attribute_values = attribute_values,
        labels = np.array( # make sure labels is a np array
             dataframe[label_attribute]
        ),
        binary_purity_metric=binary_purity_metric
    )

def __ID3__(dataframe, attribute_values, labels, binary_purity_metric):
    
    uniq = np.unique(labels)
    
    if len(uniq)<=1:
        return uniq[0] # return the only label
    
    if len(attribute_values)==0: # attributes empty
        # find most common label
        return most_common_label(labels)
    
    root_node = {}
    
    gain = binary_gain(dataframe, labels, attribute_values, binary_purity_metric=binary_purity_metric)
    
    keys = []
    values = []
    for k in gain:
        keys.append(k)
        values.append(gain[k])
    
    A = keys[np.argmax(values)] # max gain attribute
    
    for k in attribute_values[A]:
#        print(A+'='+k)
        
        subset = dataframe[A]==k
        subdataframe = dataframe[subset].drop(A, axis=1)
        
        if len(subdataframe)==0: # S_v is empty
            # add leaf node with most common value of Lable in S
            return most_common_label(labels)
        
        root_node[A+'='+k] = __ID3__(
            subdataframe,
            attribute_values, # don't need to prune attributes, gain computation should ignore attributes not in S
            labels[subset],
            binary_purity_metric=binary_purity_metric
        )
    
    return root_node
