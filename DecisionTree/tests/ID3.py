
import sys
sys.path.append('../../')

from DecisionTree.ID3 import ID3, error, most_common_label
from DecisionTree.purity import entropy, majority_error, gini_index, purity

import pandas as pd

tennis = pd.DataFrame({
    'Outlook':  'S S O R R R O S S R S O O R'.split(' '),
    'Temp':     'H H H M C C C M C M M M H M'.split(' '),
    'Humidity': 'H H H H N N N H N N N H N H'.split(' '),
    'Wind':     'W S W W W S S W W W S S W S'.split(' '),
    'Play?': [False, False, True, True, True, False, True, False, True, True, True, True, True, False]
})


def test_ID3_example_001():
    
    depth = 10
    i = 0

    results = {}
    results['information gain'] = pd.DataFrame(columns=['training_error'])
    results['majority error'] = pd.DataFrame(columns=['training_error'])
    results['gini index'] = pd.DataFrame(columns=['training_error'])
    
    model = ID3(
        tennis,
        'Play?',
        [True, False],
        attribute_values={
            'Outlook':  'S O R'.split(' '),
            'Temp':     'H M C'.split(' '),
            'Humidity': 'H N L'.split(' '),
            'Wind':     'S W'.split(' ')
        },
        purity_metric=entropy,
        max_depth=depth
    )
    
    results['information gain'].loc[i] = [error(tennis, model, 'Play?')]
    
    del model
    
    model = ID3(
        tennis,
        'Play?',
        [True, False],
        attribute_values={
            'Outlook':  'S O R'.split(' '),
            'Temp':     'H M C'.split(' '),
            'Humidity': 'H N L'.split(' '),
            'Wind':     'S W'.split(' ')
        },
        purity_metric=majority_error,
        max_depth=depth
    )
    
    results['majority error'].loc[i] = [error(tennis, model, 'Play?')]
    
    del model
    
    model = ID3(
        tennis,
        'Play?',
        [True, False],
        attribute_values={
            'Outlook':  'S O R'.split(' '),
            'Temp':     'H M C'.split(' '),
            'Humidity': 'H N L'.split(' '),
            'Wind':     'S W'.split(' ')
        },
        purity_metric=majority_error,
        max_depth=depth
    )
    
    results['gini index'].loc[i] = [error(tennis, model, 'Play?')]
    
    del model
    
    print('Information Gain Results')

    print(results['information gain'])

    print('Majroity Error Results')

    print(results['majority error'])

    print('Gini Index Results')

    print(results['gini index'])

test_ID3_example_001()
