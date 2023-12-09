import numpy as np

def compute_gradient_via_backprop(layers:list, x, y_true, dLdy_fn):
    
    states = [x]
    
    for l in layers:
        states.append(l.foward_pass(states[-1]))
    
    y = states[-1]
    
    states = states[:-1]
    
    dLdy = dLdy_fn(y, y_true)
    
    # dldws = []
    dldzs = []
    
    dLdws = []
    
    # print(len(layers), len(states))
    
    for l, s in reversed(list(zip(layers, states))):
        # print(l, s)
        dldz, dldw = l.backward_pass(s)
        dLdws.append( dLdy * np.prod(dldzs) * dldw )
        dldzs.append(dldz)
    
    return dLdws
    
# def compute_gradient_via_backprop(x, y_true):
#     Z1 = np.array([1, *z1.foward_pass(x)])
#     Z2 = np.array([1, *z2.foward_pass(Z1)])
#     Y =  y.foward_pass(Z2)
#     print(Y)
#     L = (1/2) * (Y-y_true)**2
    
#     dLdy = Y-y_true
    
#     dydz2, dydw3 = y.backward_pass(Z2)
    
#     # print('dydz2\n', dydz2)
    
#     dLdw3 = dLdy*dydw3
    
#     print('dLdw3', dLdw3)
    
#     dz2dz1, dz2dw2 = z2.backward_pass(Z1)
    
#     # print('dz2dz1\n', dz2dz1)
    
#     dLdw2 = dLdy * dydz2 * dz2dw2
    
#     print('dLdw2', dLdw2)
    
#     dz1dx, dz1dw1 = z1.backward_pass(x)
    
#     # print('dz1dx\n', dz1dx)
    
#     dLdw1 = (dLdy * dydz2 * dz2dz1).T * dz1dw1 # why did this need that transpose?
    
#     print('dLdw1', dLdw1)
    
# compute_gradient_via_backprop(np.array([1,1,1]), np.array(1))