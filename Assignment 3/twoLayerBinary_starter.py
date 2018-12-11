'''
This file implements a two layer neural network for a binary classifier

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''

import numpy as np
from mod_load_mnist import mnist
import matplotlib.pyplot as plt
from cycler import cycler
import pdb
import os

def tanh(Z):
    '''
    computes tanh activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def tanh_der(dA, cache):
    '''
    computes derivative of tanh activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    tan_h = tanh(cache["Z"])[0]
    dZ = dA * (1-tan_h)**2
    return dZ

def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = 1/(1+np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):
    '''
    computes derivative of sigmoid activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    sig = sigmoid(cache["Z"])[0]
    dZ = dA*sig*(1-sig)
    return dZ

def initialize_2layer_weights(n_in, n_h, n_fin):
    '''
    Initializes the weights of the 2 layer network

    Inputs: 
        n_in input dimensions (first layer)
        n_h hidden layer dimensions
        n_fin final layer dimensions

    Returns:
        dictionary of parameters
    '''
    # initialize network parameters
    ### CODE HERE
    W1 = np.random.randn(n_h, n_in) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_fin, n_h) * 0.01
    b2 = np.zeros((n_fin, 1))

    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A, W and b
        to be used for derivative
    '''
    ### CODE HERE
    Z = W.dot(A) + b 

    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache

def cost_estimate(A2, Y):
    '''
    Estimates the cost with prediction A2

    Inputs:
        A2 - numpy.ndarray (1,m) of activations from the last layer
        Y - numpy.ndarray (1,m) of labels
    
    Returns:
        cost of the objective function
    '''
    ### CODE HERE


    cost = -np.sum(Y * np.log(A2) + (1-Y) * np.log(1-A2), axis=1)/Y.shape[1]

    return cost

def linear_backward(dZ, cache, W, b):
    '''
    Backward propagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    # CODE HERE
    dA_prev = W.T.dot(dZ)
    dW = np.dot(dZ, cache['A'].T)
    db = np.sum(dZ, axis=1, keepdims=True)
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache) 
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE

    A1, cache1 = layer_forward(X, parameters['W1'], parameters['b1'], "sigmoid")
    A2, cache2 = layer_forward(A1, parameters['W2'], parameters['b2'], "sigmoid")    

    YPred = np.ceil(A2-0.5)

    return YPred

def test_error(X, Y, parameters):
    A1, cache1 = layer_forward(X, parameters['W1'], parameters['b1'], "sigmoid")
    A2, cache2 = layer_forward(A1, parameters['W2'], parameters['b2'], "sigmoid")
    
    return cost_estimate(A2, Y)

def two_layer_network(X, Y, vX, vY, net_dims, num_iterations=2000, learning_rate=0.1):
    '''
    Creates the 2 layer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    
    A0 = X
    costs = []
    val_costs = []
    for ii in range(num_iterations):
        # Forward propagation
        ### CODE HERE
        A1, cache1 = layer_forward(A0, parameters['W1'], parameters['b1'], "sigmoid")
        A2, cache2 = layer_forward(A1, parameters['W2'], parameters['b2'], "sigmoid")

        #Validation Loss
        vA1, c1 = layer_forward(vX, parameters['W1'], parameters['b1'], "sigmoid")
        vA2, c2 = layer_forward(vA1, parameters['W2'], parameters['b2'], "sigmoid")

        # cost estimation
        ### CODE HERE
        cost = cost_estimate(A2, Y)
        vcost = cost_estimate(vA2, vY)

        # Backward Propagation
        ### CODE HERE
        dA2 = ((-Y/A2) + ((1-Y)/(1-A2)))/Y.shape[1]
        dA1, dW2, db2 = layer_backward(dA2, cache2, parameters['W2'], parameters['b2'], "sigmoid")
        d, dW1, db1 = layer_backward(dA1, cache1, parameters['W1'], parameters['b1'], "sigmoid")

        #update parameters
        ### CODE HERE
        parameters['W2'] = parameters['W2'] - learning_rate*dW2
        parameters['W1'] = parameters['W1'] - learning_rate*dW1
        parameters['b2'] = parameters['b2'] - learning_rate*db2
        parameters['b1'] = parameters['b1'] - learning_rate*db1

        costs.append(cost)
        val_costs.append(vcost)

        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
    
    return costs, val_costs, parameters

def accuracy(Pred, Y):
    x = np.sum(Y == Pred)/Y.shape[1]
    return x

def shuffle_sync(A, B):
    rng_state = np.random.get_state()
    np.random.shuffle(A)
    np.random.set_state(rng_state)
    np.random.shuffle(B)

def train(train_X, train_Y, val_X, val_Y, test_data, test_label, n_h, iters = 1001):
    np.random.seed(42)
    n_in, m = train_X.shape
    n_fin = 1
    n_h = n_h
    net_dims = [n_in, n_h, n_fin]
    # initialize learning rate and num_iterations
    learning_rate = 0.1
    num_iterations = iters

    costs, val_costs, parameters = two_layer_network(train_X, train_Y, val_X, val_Y, net_dims, \
            num_iterations=num_iterations, learning_rate=learning_rate)
    
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_X, parameters)
    val_Pred = classify(val_X, parameters)
    test_Pred = classify(test_data, parameters)
    
    # Test Loss
    #A1, cache1 = layer_forward(test_data, parameters['W1'], parameters['b1'], "sigmoid")
    #A2, cache2 = layer_forward(A1, parameters['W2'], parameters['b2'], "sigmoid")    
    #test_error = cost_estimate(A2, test_label)

    trAcc = accuracy(train_Pred, train_Y)
    valAcc = accuracy(val_Pred, val_Y)
    teAcc = accuracy(test_Pred, test_label)
    print("\nAccuracy after",num_iterations,"epochs with n_h=",n_h,"learning_rate=",learning_rate)
    print("Training: {0:0.3f}%".format(trAcc*100), "Validation: {0:0.3f}%".format(valAcc*100), "Testing: {0:0.3f}%".format(teAcc*100))

    return costs, val_costs, 1-teAcc

def main():
    os.system("mkdir graphs_1")
    # getting the subset dataset from MNIST
    # binary classification for digits 1 and 7
    digit_range = [1,7]
    train_X, train_Y, val_X, val_Y, test_data, test_label = \
            mnist(noTrSamples=2000, noValSamples=400, noTsSamples=1000,\
            digit_range=digit_range,\
            noTrPerClass=1000, noValPerClass=200, noTsPerClass=500)
    
    #convert to binary labels
    train_Y[train_Y==digit_range[0]] = 0
    train_Y[train_Y==digit_range[1]] = 1
    val_Y[val_Y==digit_range[0]] = 0
    val_Y[val_Y==digit_range[1]] = 1
    test_label[test_label==digit_range[0]] = 0
    test_label[test_label==digit_range[1]] = 1

    # #Separate Train and Validation Set
    # train_label = train_label.reshape(2400)
    # digit_0 = train_data.T[train_label == 0]
    # digit_1 = train_data.T[train_label == 1]
    # digit_0_label = train_label[train_label == 0]
    # digit_1_label = train_label[train_label == 1]
    # train_X = np.concatenate((digit_0[:1000], digit_1[:1000]), axis = 0)
    # train_Y = np.concatenate((digit_0_label[:1000], digit_1_label[:1000]), axis=0)
    # val_X = np.concatenate((digit_0[1000:], digit_1[1000:]), axis=0)
    # val_Y = np.concatenate((digit_0_label[1000:], digit_1_label[1000:]), axis=0)

    # #Shuffle Data (which was sorted originally)
    # shuffle_sync(train_X, train_Y)
    # shuffle_sync(val_X, val_Y)

    # # Shape data properly for training
    # train_X = train_X.T
    # val_X = val_X.T
    # train_Y = train_Y.reshape((1,train_Y.shape[0]))
    # val_Y = val_Y.reshape((1,val_Y.shape[0]))

    costs, val_costs, test_error = train(train_X, train_Y, val_X, val_Y, test_data, test_label, n_h=200, iters=1001)
    
    # CODE HERE TO PLOT costs vs iterations
    #### Q1 1)
    plt.clf()
    x_points = [i for i in range(0,len(costs))]
    plt.title("Error v/s Iterations")
    cost_line, = plt.plot(x_points, costs, 'b-', label="Training")
    val_line, = plt.plot(x_points, val_costs, 'r-', label="Validation")
    plt.legend(handles=[cost_line, val_line])
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.savefig('graphs_1/Error_vs_Iterations.png')
    # plt.show()
    ####

    #### Q1 2)
    print("Test Error aka Misclassification %: {0:0.3f}%".format(test_error*100))
    print("\n")
    ####

    #### Q1 3)
    c ,vc = [], []
    test_errors = []
    lrs, all_costs, all_v_costs = [], [], []
    for i in range(100, 501, 100):
        print("###")
        print("Hidden Layer Size: ",i)
        
        tc, tvc, t_e = train(train_X, train_Y, val_X, val_Y, test_data, test_label, n_h=i, iters=1001)
        x_points = [i for i in range(0,len(tc))]
        
        plt.clf()
        plt.title('Hidden Layer Size:'+str(i))
        plt.ylabel('Error')
        plt.xlabel('Iterations')
        cost_line, = plt.plot(x_points, tc, 'b-', label="Training")
        val_line, = plt.plot(x_points, tvc, 'r-', label="Validation")
        plt.legend(handles=[cost_line, val_line])
        plt.savefig('graphs_1/Error_Graph_HiddenLayerSize_'+str(i)+'.png')
        # plt.show()

        lrs.append(i)
        all_costs.append(tc)
        all_v_costs.append(tvc)
        test_errors.append(t_e)
        c.append(tc[-1])
        vc.append(tvc[-1])
        print("###")
        print("\n")
    
    # Error v/s Nodes
    # x_points = list(range(50, 1501, 50))
    # plt.clf()
    # plt.plot(x_points, c, 'b-', x_points, vc, 'r-')
    # plt.ylabel('Error')
    # plt.xlabel('Hidden Layer Size')
    # plt.savefig('graphs_1/Error_vs_HiddenLayerSize.png')
    # plt.show()
    ####

    # All lines together
    handles = []
    plt.clf()
    plt.title('All Learning Rates')
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y'])))
    for i in range(len(all_costs)):
        x_points = [i for i in range(0,len(all_costs[i]))]
        cost_line, = plt.plot(x_points, all_costs[i], label=str(lrs[i]))
        handles.append(cost_line)
    plt.legend(handles=handles)
    plt.savefig('graphs_1/Error for all architectures.png')

    #### Q1 4)
    ix = np.argmin(vc)
    neurons = list(range(100, 501, 100))[ix]
    print("Minimum validation error found when n_h="+str(neurons)+", Test Error during that run ="+str(test_errors[ix]*100)+"%")
    ####

if __name__ == "__main__":
    main()