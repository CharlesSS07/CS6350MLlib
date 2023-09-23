
import numpy as np

def binary_entropy(x):

    x = np.array(x)

    # make sure x are 1-d, and boolean
    assert(len(x.shape)==1)
    assert(x.dtype==bool)

    p_pos = np.sum(x)/len(x)
    p_neg = 1 - p_pos

    # prevent nan values from showing up due to limit approaching infinity...
    if p_pos==0 or p_neg==0:
        return 0

    return -(p_pos * np.log2(p_pos)) - (p_neg * np.log2(p_neg))

#def binary_majority_error(x):
#
#    x = np.array(x)
#
#    # make sure x are 1-d, and boolean
#    assert(len(x.shape)==1)
#    assert(x.dtype==bool)
#
#    p_pos = np.sum(x)/len(x)
#    p_neg = 1 - p_pos
#
#    # print(p_pos, p_neg, min(p_pos, p_neg))
#
#    return min(p_pos, p_neg)

def majority_error(x):
    '''
    If you only guess the majority label, what is your error?
    '''

    x = np.array(x)

    # make sure x are 1-d, and boolean
    assert(len(x.shape)==1)
    
    uniq = np.unique(x)

    l = len(x)
    
    shares = [np.sum(x==u_i)/l for u_i in uniq]
    majority_label_idx = np.argmax(shares)

    return shares[majority_label_idx]

def gini_index(x):

    x = np.array(x)

    # make sure x are 1-d, and boolean
    assert(len(x.shape)==1)
    
    return 1 - sum([ np.sum(x==i)/len(x) for i in np.unique(x) ])

