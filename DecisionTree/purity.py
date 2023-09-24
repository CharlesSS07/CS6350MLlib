
import numpy as np

def entropy(x, unique_x_values):

    x = np.array(x)

    # make sure x is 1-d
    assert(len(x.shape)==1)
    
    l = len(x)
    
    counts = [ np.sum(x==i) for i in unique_x_values ]
    
    ps = [ c/l for c in counts if c!=0 ]

    return - sum([p_i*np.log2(p_i) for p_i in ps])

def majority_error(x, unique_x_values):
    '''
    If you only guess the majority label, what is your error?
    '''

    x = np.array(x)

    # make sure x is 1-d
    assert(len(x.shape)==1)

    l = len(x)
    
    shares = [np.sum(x==u_i)/l for u_i in unique_x_values]

    return min(shares)

def gini_index(x, unique_x_values):

    x = np.array(x)

    # make sure x are 1-d, and boolean
    assert(len(x.shape)==1)
    
    return 1 - sum([ (np.sum(x==i)/len(x))**2 for i in unique_x_values ])

def purity(S, y, A_values, y_values, purity_metric=entropy):
    '''
    S -- pandas dataframe. number of rows should match lenght of y
    y -- list of labels, same order as lists in S
    A_values -- dict of names to attributes in S (keys), and the possible values for the corresponding attribute
    binary_information_measure_algorithim -- should be a function which takes a boolean list, and returns the a measure of information content as a number
    '''

    y = np.array(y)
    ly = len(y)

    H = purity_metric(y, y_values)

    res = {}

    for A in S.columns:

        res[A] = H

        for a in np.unique(A_values[A]):

            subset = y[np.where(np.array(S[A])==a)]
            ls = len(subset)

            if ls > 0:
                res[A] -= ( ls/ly ) * purity_metric(subset, y_values)

    return res
