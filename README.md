# CS6350MLlib
This is a machine learning library developed by CHARLES STRAUSS for CS6350 at the University of Utah

# Grading

I set up a [google colab](https://colab.research.google.com/drive/1o6SQprPrUDDwlHZ-xnCCq2a5SEda0XC0?usp=sharing) which runs my code.

Note that I have seperated the work for each homework problem into its own file, which can be found in the HWs folder under the corresponding weeks homework.

Also, note that this project will require a working installation of pandas and numpy to run.

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


