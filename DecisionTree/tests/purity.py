
import sys
sys.path.append('../../')

from DecisionTree.purity import entropy, majority_error, gini_index, purity

def test_entropy_binary_example_001():
    
    e = [*([True,]*15), *([False,]*15)] # p=0.5, entropy should be 1
    
    assert(entropy(e, [True, False])==1)
    
    e = [False,]*15 # p=1, entropy should be 0
    
    assert(entropy(e, [True, False])==0)

test_entropy_binary_example_001()

def test_majority_error_binary_example_001():
    
    e = [*([True,]*15), *([False,]*5)]
    
    assert(majority_error(e, [True, False])==1/4)
    
    e = [False,]*10
    
    assert(majority_error(e, [True, False])==0)

test_majority_error_binary_example_001()

def test_gini_index_binary_example_001():
    
    e = [*([True,]*15), *([False,]*5)]
    
    assert(gini_index(e, [True, False])==3/8)
    
    e = [*([True,]*5), *([False,]*15)]
    
    assert(gini_index(e, [True, False])==3/8)

test_gini_index_binary_example_001()

def test_purity_0():
    
    # 100 % pure example
    
    # since x = y, more splits should not increase (0) the purity
    
    import pandas as pd
    
    import numpy as np
    
    # create a 4-value attribute so no longer binary
    x = [ str(a)+' '+str(b) for a,b in zip(np.random.uniform(size=100)>0.5, np.random.uniform(size=100)>0.5) ]
    
    y = x
    
    y_values = ['True True', 'True False', 'False True', 'False False']
    x_values = y_values
    
    df = pd.DataFrame({'x':x})
    
    H = entropy(y, y_values)
    
    assert(
        purity(
            S=df,
            y=y,
            A_values={'x': x_values},
            y_values=y_values,
            purity_metric=entropy
        )['x']==H
    )
    
    H = majority_error(y, y_values)
    
    assert(
        purity(
            S=df,
            y=y,
            A_values={'x': x_values},
            y_values=y_values,
            purity_metric=majority_error
        )['x']==H
    )
    
    H = gini_index(y, y_values)
    
    assert(
        purity(
            S=df,
            y=y,
            A_values={'x': x_values},
            y_values=y_values,
            purity_metric=gini_index
        )['x']==H
    )

test_purity_0()

def test_purity_low():
    
    # low purity example
    
    # generate decorrelated x and y. more splits should barley increase the purity
    
    import pandas as pd
    
    import numpy as np
    
    # create a 4-value attribute so no longer binary
    x = [ str(a)+' '+str(b) for a,b in zip(np.random.uniform(size=100)>0.5, np.random.uniform(size=100)>0.5) ]
    
    y = np.random.uniform(size=100)>0.5
    
    y_values = [True, False]
    x_values = ['True True', 'True False', 'False True', 'False False']
    
    df = pd.DataFrame({'x':x})
    
    H = entropy(y, y_values)
    
    print(
        purity(
            S=df,
            y=y,
            A_values={'x': x_values},
            y_values=y_values,
            purity_metric=entropy
        )['x'], H
    )
    
    H = majority_error(y, y_values)
    
    print(
        purity(
            S=df,
            y=y,
            A_values={'x': x_values},
            y_values=y_values,
            purity_metric=majority_error
        )['x'], H
    )
    
    H = gini_index(y, y_values)
    
    print(
        purity(
            S=df,
            y=y,
            A_values={'x': x_values},
            y_values=y_values,
            purity_metric=gini_index
        )['x'], H
    )

test_purity_low()
