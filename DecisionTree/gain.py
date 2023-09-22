
import numpy as np

from DecisionTree.purity_metrics import majority_error

def binary_gain(S, y, A_values, binary_purity_metric=majority_error):
    '''
    S -- pandas dataframe. number of rows should match lenght of y
    y -- list of labels, same order as lists in S
    A_values -- dict of names to attributes in S (keys), and the possible values for the corresponding attribute
    binary_information_measure_algorithim -- should be a function which takes a boolean list, and returns the a measure of information content as a number
    '''

    y = np.array(y)

    H = binary_purity_metric(y)

    res = {}

    for A in S.columns:

        res[A] = H*1

        for a in np.unique(A_values[A]):

            subset = y[np.where(np.array(S[A])==a)]

            if len(subset) > 0:
                res[A] -= ( len(subset)/len(y) ) * binary_purity_metric(subset)

    return res

