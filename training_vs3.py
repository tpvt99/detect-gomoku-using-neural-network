import numpy as np
import time
import matplotlib.pyplot as plt
import scipy
from PIL import Image
#import mnist_loader3


def initialize_parameters(layer_dimensions):
    """
    Arguments:
    layer_dimensions -- python array (list containing dimensions of each layer. For ex: [4,5,10,2] is 3 layer because the first is for X(input)

    Return:
    parameters -- python dictionary contains weights and bias, with the format
            Wl -- the weights at layer l
            bl -- the biases at layer l

    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dimensions)
    for i in range(1,L):
        parameters["W" + str(i)] = np.random.randn(layer_dimensions[i], layer_dimensions[i-1]) * np.sqrt(2/(layer_dimensions[i-1]))
        parameters["b" + str(i)] = np.zeros((layer_dimensions[i], 1))

        assert(parameters["W" + str(i)].shape == (layer_dimensions[i], layer_dimensions[i-1]))
        assert(parameters["b" + str(i)].shape == (layer_dimensions[i], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Arguments:
    A is the activation of the previous layer (size_previous_layer, num_of_ex)
    W is the weights matrix: numpy array matrix (size_current_layer, size_previous_layer)
    b is the biases matrix: np matrix (size_current_layer, 1)

    Returns:
    Z
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    assert (Z.shape == (W.shape[0], A.shape[1]))

    return Z, cache

def sigmoid(z):
    return 1./ (1 + np.exp(-z))

def activation_forward(Z, activation):
    """
    Arguments:
    Z -- np matrix (current_size_layer, no_of_examples)

    Returns:
    A -- np matrix (current_size_layer, no_of_examples)
    """
    if activation == "relu":
        A = np.multiply(Z, (Z>=0))
    elif activation == "sigmoid":
        A = sigmoid(Z)

    cache = Z

    return A, cache


# combine linear_forward and activation_forward
def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments:
    A is the activation of the previous layer (size_previous_layer, num_of_ex)
    W is the weights matrix: numpy array matrix (size_current_layer, size_previous_layer)
    b is the biases matrix: np matrix (size_current_layer, 1)
    activation is the string contains "relu" and "sigmoid"

    Returns:
    A -- the activation of the next layer (size_next_layer, num_of_ex)
    cache -- the cache contains linear_cache and activation_cache
    """

    # linear cache -- containing A_prev(l-1), W(l), b(l) to feed in linear backward
    # activation cache -- containing Z(l) to feed in activation backward

    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation_forward(Z, activation)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Arguments:
    X -- the input np matrix
    parameters -- the dictionary containing W and b

    Returns
    AL -- the np activation matrix of the end layer
    caches -- list of caches, each item contains W and b of current layer l and A of previous layer l-1
    """
    caches = []
    L = len(parameters) // 2
    A = X

    for i in range(1,L):
        W = parameters["W" + str(i)]
        b = parameters["b" + str(i)]
        A, cache = linear_activation_forward(A, W, b, "relu")
        caches.append(cache)

    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A, WL, bL, "sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y, parameters, lambd):
    """
    Arguments:
    AL - probability vector corresponding to label predictions, shape (output_layer, number of examples)
    Y - true label vector of examples, shape (output_layer, number of examples)

    Returns:
    cost -- cross entropy cost
    """

    m = Y.shape[1]
    L = len(parameters)//2
    L2_regularization = 0
    for i in range(L):
        W = parameters["W" + str(i+1)]
        L2_regularization += np.sum(np.power(W,2))

    J = -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL))) + lambd / (2*m) * L2_regularization

    cost = np.squeeze(J)
    assert(cost.shape == ())

    return cost

def linear_backward(dZ, cache):
    """
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output ( of current layer l)
    cache -- the linear cache

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1)
    dW -- Gradient of the cost with respect to the weights ( of the current layer l)
    db -- Gradient of the cost with respect to the biases ( of the current layer l)
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    assert ( dA_prev.shape == A_prev.shape)
    assert ( dW.shape == W.shape)
    assert ( db.shape == b.shape)

    return dA_prev, dW, db
    
def activation_backward(dA, cache, activation):
    """
    Arguments:
    dA -- Gradient of the cost with respect to the activation ( of the current layer l)
    cache -- contains the cache
    activation -- string contains the activation name

    Returns:
    dZ -- Gradient of the cost with respect to the linear output ( of current layer l)
    """
    Z = cache

    if activation == "sigmoid":
        A = sigmoid(Z)
        g_derivative = np.multiply(A, 1-A)
    elif activation == "relu":
        g_derivative = (Z >= 0)
    
    dZ = np.multiply(dA, g_derivative)
    return dZ

def linear_activation_backward(dA, cache, activation):
    """
    Arguments:
    dA -- Gradient of the cost with respect to the activation ( of current layer l)
    cache -- cache containing (linear_cache, activation_cache)
    activation -- the string containing "relu" or "sigmoid"

    Returns:
    dA_prev
    dW
    db
    """
    
    linear_cache, activation_cache = cache

    dZ = activation_backward(dA, activation_cache, activation)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return (dA_prev, dW, db)

def L_model_backward(AL, Y, caches, parameters, lambd):
    """
    Arguments:
    AL
    caches
    Y

    Returns:
    grads -- A dictionary with gradients
    """
    grads = {}
    L = len(caches)
    m = AL.shape[0]
    Y = Y.reshape(AL.shape)

    # Initialize backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    cache = caches[-1]
    dA_prev, dW, db = linear_activation_backward(dAL, cache, "sigmoid")
    grads["dW" + str(L)] = dW + lambd/m * parameters["W" + str(L)]
    grads["db" + str(L)] = db
    
    for i in range(L-1, 0, -1):
        cache = caches[i-1]
        dA_prev, dW, db =  linear_activation_backward(dA_prev, cache, "relu")
        grads["dW" + str(i)] = dW + lambd/m * parameters["W" + str(i)]
        grads["db" + str(i)] = db

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Arguments:
    parameters
    grads
    learning_rate

    Returns:
    parameters
    """
    L = len(parameters) // 2
    for i in range(1,L+1):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * grads["dW" + str(i)]
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * grads["db" + str(i)]

    return parameters

def main_training(X, Y,  layer_dims, learning_rate = 3, epochs = 2500, lambd = 0.7, X_test = None, Y_test = None, print_cost = True):
    # intialize parameters
    parameters = initialize_parameters(layer_dims)
    costs = []

    for i in range(epochs):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y, parameters, lambd)

        grads = L_model_backward(AL, Y, caches, parameters, lambd)

        parameters = update_parameters(parameters, grads, learning_rate)

        print("Cost after %i: %f" %(i,cost))
        costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning_rate " + str(learning_rate))
    plt.show()

    return parameters


train_in, train_re, test_in, test_re =  np.load('training_inputs.npy'), np.load('training_results.npy'), np.load('test_inputs.npy'), np.load('test_results.npy')

X,Y = train_in, train_re

X = X.reshape(X.shape[0], 784).T
print(X.shape)

Y = Y.reshape(Y.shape[0], 3).T
print(Y.shape)

X_test,Y_test = test_in, test_re
X_test = X_test.reshape(X_test.shape[0], 784).T
print(X_test.shape)
Y_test = Y_test.reshape(Y_test.shape[0], 3).T
print(Y_test.shape)

parameters = main_training(X, Y, layer_dims = [784,10,3], learning_rate = 0.1, epochs = 100, lambd = 1.5, X_test = X_test, Y_test = Y_test)

np.save('parameters.npy', parameters)
