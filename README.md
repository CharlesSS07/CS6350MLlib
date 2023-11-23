# CS6350MLlib
This is a machine learning library developed by CHARLES STRAUSS for CS6350 at the University of Utah

# Documentation

## Decision Trees

### Example

```python
from DecisionTree.ID3 import ID3
from DecisionTree.purity import entropy, majority_error, gini_index

# Load in data
import pandas as pd
train = pd.DataFrame(...)

# Train a ID3 Decision Tree
model = ID3(
    dataframe=train,
    label_attribute='label', # the attribute in the dataframe which you are using as a label
    label_values=['unacc', 'acc', 'good', 'vgood'], # the possible values of label_attribute
    # attribute_values={'attribute':['A', 'B', 'C', ...]} (optional), used to outline attributes/values missing from dataframe, which are not the label attribute/values
    purity_metric=entropy, # the algorithim for measuring purity, used for information gain, etc. Import purity measures from DecisionTree.purity (entropy, majority_error, gini_index, ...) or use any function here
    max_depth=i # depth to stop growing this tree at
)

# Lookup a classification:

classification = model['attribute1=value']['attribute3=value']['attribute4=value']
```

In the above example, we import my Decision Tree code, and give an example of setting up a decision tree and training it on a pandas dataframe. `ID3` takes the training dataset, the attribute which should be used for labelling, it's values, the purity algorithim, and the maximum depth to grow the tree to.

## AdaBoost

### Example

```python
from DecisionTree.ID3 import ID3, error, most_common_label
from DecisionTree.purity import entropy, majority_error, gini_index
from EnsembleLearning.Boosting import Booster, AbstractWeakBoostableClassifier

# Load in data
import pandas as pd
train = pd.DataFrame(...)

# Implement AbstractWeakBoostableClassifier to use my decision tree
class BoostableDecisionTreeClassifier(AbstractWeakBoostableClassifier):
    
    def __init__(self, decision_tree: dict):
        self.decision_tree = decision_tree
    
    def __predict__(self, x: pd.DataFrame):
        # this code extracts the prediction from the decision tree
        node = self.decision_tree
        
        while True:
            if not type(node)==dict:
                return binarize_labels(node)
            keys = list(node.keys())
            attribute = keys[0].split('=')[0]
            value = x[attribute]
            node = node[attribute+'='+str(value)]

# The Booster needs a function that trains a new decision tree on every call.
# This is that function.
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

# Train the boosted decision tree
b = Booster(
    find_classifier_callback=find_classifier_callback,
    iterations=max_iteration,
    labels=list(map(binarize_labels, train_discretized['y'])),
    data=train_discretized
)

print(b.predict()) # Display the predictions of the boosted decision tree.
```
The boosted decision tree library (AdaBoost) works by allowing the user to specify the weak classifier to use for boosting. This is done by implementing a `BoostableDecisionTreeClassifier`, with logic for storing and predicting on a model, and then supplying a callback function (`find_classifier_callback`) in this case, which trains up a new classifier with the current weights (supplied to that function as shown). The `Booster` class trains up `iterations` many of the weak classifiers supplied by the callback function.

## Perceptrons

### Example

```python
from Perceptron.Perceptrons import StandardPerceptron
from Perceptron.Perceptrons import VotedPerceptron
from Perceptron.Perceptrons import AveragePerceptron

import pandas as pd
train = pd.DataFrame(...)

# need to map labels from 0,1 to -1,1
def map_labels(y01):
    return 2 * (y01-0.5) # or whatever works for your data...

sp = StandardPerceptron(
    data=train[:,:-1],
    labels=map_labels(train[:,-1]),
    r=0.01,
    T=10
)

vp = VotedPerceptron(
    data=train[:,:-1],
    labels=map_labels(train[:,-1]),
    r=0.0001,
    T=10
)

ap = AveragePerceptron(
    data=train[:,:-1],
    labels=map_labels(train[:,-1]),
    r=0.0001,
    T=10
)
```

In this example, we start out by importing the classes for the standard, voted, and average perceptrons, and then training up one of each of these classes of perceptron on some data, with some parameters which could make sense.
