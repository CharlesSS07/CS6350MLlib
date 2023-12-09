
print('Running "HWs/HW_5/HW_5_part2_2a.py"')
print()

import numpy as np

import sys
sys.path.append('../../NeuralNetworks/')

import layers
import BackProp

# set up the neural network like shown in the problem
z1 = layers.DenseLayer(layers.activations.sigmoid, layers.activations.d_sigmoid_dx, use_bias=True)
z2 = layers.DenseLayer(layers.activations.sigmoid, layers.activations.d_sigmoid_dx, use_bias=True)
y =  layers.DenseLayer(layers.activations.identity, layers.activations.d_identity_dx)

# set the weights to the weights given in the HW
z1.w = np.array(
    [
        [-1, -2, -3],
        [ 1,  2,  3]
    ]
)
z2.w = np.array(
    [
        [-1, -2, -3],
        [ 1,  2,  3]
    ]
)
y.w = np.array(
    [
        [ -1,  2, -1.5]
    ]
)

# compute the gradient using the example data from part 1 problem 3
gradients = BackProp.compute_gradient_via_backprop(
    layers=[z1, z2, y],
    x=np.array([1, 1, 1]),
    y_true=1,
    dLdy_fn=lambda Y, y_true: Y-y_true # derivative of arbitrary loss function
)

print('Gradient of layer 3:', gradients[0])
print()

print('Gradient of layer 2:', gradients[1])
print()

print('Gradient of layer 1:', gradients[2])
print()