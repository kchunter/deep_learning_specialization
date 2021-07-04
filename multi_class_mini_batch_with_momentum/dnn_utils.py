import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def softmax(Z):
    """
    Implements stable version of the softmax activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of softmax(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    exps = np.exp(Z - np.max(Z))
    A = exps / np.sum(exps, axis=0, keepdims=True)
    cache = Z
    
    return A, cache


def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def softmax_backward(dA, cache):
    """
    Implement the backward propagation for an n^[L] SOFTMAX units.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    m = Z.shape[1]
    
    exps = np.exp(Z - np.max(Z))
    A = exps / np.sum(exps, axis=0, keepdims=True)
    jacs = np.zeros((Z.shape[0], Z.shape[0]))
    
    for i in range(m):
        jac = np.diag(A[:, i])
        for j in range(jac.shape[0]):
            for k in range(jac.shape[0]):
                if j == k:
                    jac[j][k] = A[:, i][j] * (1 - A[:, i][j])
                else:
                    jac[j][k] = -A[:, i][j]*A[:, i][k]

        jacs = jacs + jac
    
    jacs = jacs / m
    dZ = dA * jacs
    
    assert (dZ.shape == Z.shape)
    
    return dZ



def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
