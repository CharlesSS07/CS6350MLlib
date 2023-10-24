
import numpy as np

def entropy(x, unique_x_values, x_weights):

    x = np.array(x)

    # make sure x is 1-d
    assert(len(x.shape)==1)
    
    l = len(x)
    
    if x_weights is None:
        x_weights = np.ones(shape=l, dtype=np.float64)/l
    else:
        assert len(x_weights)==l
    
    ps = np.array([ np.sum(x_weights[x==i]) for i in unique_x_values ])
    
    ps = ps[ps>0]/l # ignore 0's because log2 dosent like them and they cancel out anyway
    
    return - np.multiply(np.log2(ps), ps).sum()

def majority_error(x, unique_x_values, x_weights):
    '''
    If you only guess the majority label, what is your error?
    '''

    x = np.array(x)

    # make sure x is 1-d
    assert(len(x.shape)==1)

    l = len(x)
    
    if x_weights is None:
        x_weights = np.ones(shape=l, dtype=np.float64)/l
    else:
        assert len(x_weights)==l
    
    shares = [ np.sum(x_weights[x==u_i]) for u_i in unique_x_values]

    return min(shares)

def gini_index(x, unique_x_values, x_weights):

    x = np.array(x)

    # make sure x are 1-d, and boolean
    assert(len(x.shape)==1)
    
    l = len(x)
    
    if x_weights is None:
        x_weights = np.ones(shape=l, dtype=np.float64)/l
    else:
        assert len(x_weights)==l
    
    return 1 - np.sum([ np.sum(x_weights[x==i])**2 for i in unique_x_values ])


def purity(S, y, A_values, y_values, example_weights, purity_metric=entropy):
    '''
    :param S: pandas dataframe. number of rows should match lenght of y
    :param y: list of labels, same order as lists in S
    :param A_values: dict of names to attributes in S (keys), and the possible values for the corresponding attribute
    :binary_information_measure_algorithim: should be a function which takes a boolean list, and returns the a measure of information content as a number
    '''

    y = np.array(y)
    ly = len(y)
    
    assert len(S)==ly
    
    if example_weights is None:
        example_weights = np.ones(shape=ly, dtype=np.float64)/ly
    else:
        assert len(example_weights)==ly

    H = purity_metric(y, y_values, example_weights)

    res = {}

    for A in S.columns:

        res[A] = H

        for a in np.unique(A_values[A]):
            
            subset = np.where(np.array(S[A])==a)
            y_subset = y[subset]
            w_subset = example_weights[subset]
            
            ls = len(y_subset)

            if ls > 0: #( ls/ly ) * 
                res[A] -= ( ls/ly ) * purity_metric(y_subset, y_values, w_subset)
    
    assert len(res)!=0
    
    return res
