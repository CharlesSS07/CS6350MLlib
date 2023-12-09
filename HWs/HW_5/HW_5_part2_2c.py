


print('Running "HWs/HW_5/HW_5_part2_2b.py"')
print()

import numpy as np

import sys
sys.path.append('../../NeuralNetworks/')

import layers
import BackProp



import pandas as pd
train = pd.read_csv(
    '../../data/bank-note/train.csv',
    names=['variance', 'skewness', 'curtosis', 'entropy', 'y']
)



X = np.array(train)[:,:-1]
X = np.concatenate([np.ones(shape=X.shape[0])[np.newaxis].T, X], axis=1) # for bias trick
Y = (np.array(train)[:,-1] * 2) - 1



def build_nn(width):
    a = layers.DenseLayer(layers.activations.sigmoid, layers.activations.d_sigmoid_dx)
    b = layers.DenseLayer(layers.activations.sigmoid, layers.activations.d_sigmoid_dx)
    c =  layers.DenseLayer(layers.activations.sigmoid, layers.activations.d_sigmoid_dx)

    a.w = np.zeros(shape=[width, X.shape[1]])
    b.w = np.zeros(shape=[width, width])
    c.w = np.zeros(shape=[1, width])
    
    return [a, b, c]




def loss(y, y_true):
    return (1/2) * ((y-y_true) **2)



def train_nn(width=10, T=30, lr_0=0.001, lr_d=1):
    
    def lr_sched(t):
        return lr_0/(1+((lr_0/lr_d)*t))
    
    nn_layers = build_nn(width)
    mean_loss = []
    
    for t in range(0, T):
        lr = lr_sched(t)
        X_samples = list(range(0, len(X)))
        np.random.shuffle(X_samples)
        for i in X_samples:
            dLdws = reversed(BackProp.compute_gradient_via_backprop(
                layers=nn_layers,
                x=X[i],
                y_true=Y[i],
                dLdy_fn=lambda _Y, _y_true: _Y-_y_true # derivative of arbitrary loss function
            ))
            # do the update
            for dLdw, layer in zip(dLdws, nn_layers):
                layer.w -= lr * dLdw
        mean_loss.append(
            np.mean(loss(np.array([ layers.evaluate_nn(nn_layers, x) for x in X ]), Y)))
    return mean_loss


import matplotlib.pyplot as plt


experiments = [
    {
        'width':5
    },
    {
        'width':10
    },
    {
        'width':25
    },
    {
        'width':50
    },
    {
        'width':100
    }
]

for i, experiment in enumerate(experiments):
    
    print(f'Training NNs from zero initialization with width={experiment["width"]}\n\n')
    plt.title(f'Zero Initialization of NNs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    print(f'{i}/{len(experiments)}')
    plt.plot(train_nn(
        width=experiment['width'], 
        T=40,
        lr_0=0.001,
        lr_d=10
    ), label=f'width={experiment["width"]}')
plt.legend()
plt.show()