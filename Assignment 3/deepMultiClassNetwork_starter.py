'''
This file implements a multi layer neural network for a multiclass classifier

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''
import numpy as np
from mod_load_mnist import mnist, one_hot
import matplotlib.pyplot as plt
from cycler import cycler
import pdb
import sys, ast
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

def relu(Z):
    '''
    computes relu activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    ### CODE HERE
    #softmax
    max_z = np.max(Z, axis=0, keepdims=True)
    
    # add e^-10 to prevent divide by zero error while taking log A?
    z_exp = np.exp(Z-max_z) #+ np.exp(-10)

    denominator = np.sum(z_exp, axis=0, keepdims=True)
    A = z_exp/denominator
    cache = {'A':A}

    

    #crossentropy
    if Y.size != 0:
        m = Y.shape[1]
        Y_ = one_hot(Y.astype(int),10).T
        # loss = -(1/Y.shape[1]) * np.sum( np.log( np.sum(A * Y_, axis = 0) ) )
        loss = -(1/Y.shape[1]) * np.sum( np.log( A[Y.astype(int),np.arange(m)] ) )   

    else:
        loss = None

    return A, cache, loss

def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs: 
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    ### CODE HERE
    Y_ = one_hot(Y.astype(int),10).T
    dZ = (cache['A'] - Y_)/Y.shape[1]

    return dZ

def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers-1):
        parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1], net_dims[l]) * 0.01
        parameters["b"+str(l+1)] = np.zeros((net_dims[l+1], 1)) #np.random.randn(net_dims[l+1], 1) * 0.01 
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
        cache - a dictionary containing the inputs A
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
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    elif activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    '''
    Forward propgation through the layers of the network

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = len(parameters)//2  
    A = X
    caches = []
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    A_prev = cache["A"]
    ## CODE HERE
    dA_prev = W.T.dot(dZ)
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
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
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs: 
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activation)
        activation = "relu"
    return gradients

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE
    # Forward propagate X using multi_layer_forward
    # Get predictions using softmax_cross_entropy_loss
    # Estimate the class labels using predictions
    AL, caches = multi_layer_forward(X, parameters)
    Af, cache, cost = softmax_cross_entropy_loss(AL)

    # maxp = np.max(Af, axis=0, keepdims=True)
    # Af = Af - maxp
    # Af[Af == 0] = 1
    # Af[Af < 0] = 0
    YPred = np.zeros(Af.shape)
    maxix = np.argmax(Af, axis=0)
    YPred[maxix, np.arange(len(maxix))] = 1
    
    return YPred

def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.0):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    alpha = learning_rate*(1/(1+decay_rate*epoch))
    L = len(parameters)//2
    for i in range(1,L+1):    
        parameters['W'+str(i)] = parameters['W'+str(i)] - alpha*gradients['dW'+str(i)]
        parameters['b'+str(i)] = parameters['b'+str(i)] - alpha*gradients['db'+str(i)]
    return parameters, alpha

def multi_layer_network(X, Y, vX, vY, net_dims, num_iterations=500, learning_rate=0.2, decay_rate=0.00):
    '''
    Creates the multilayer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labelsf
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    parameters = initialize_multilayer_weights(net_dims)
    A0 = X
    costs = []
    val_costs = []
    for ii in range(num_iterations):
        ### CODE HERE
        # Forward Prop

        ## call to multi_layer_forward to get activations
        ## call to softmax cross entropy loss
        AL, caches = multi_layer_forward(A0, parameters)
        Af, cache, cost = softmax_cross_entropy_loss(AL, Y)
        
        vAL, vcaches = multi_layer_forward(vX, parameters)
        vAf, vcache, vcost = softmax_cross_entropy_loss(vAL, vY)

        # Backward Prop
        ## call to softmax cross entropy loss der
        ## call to multi_layer_backward to get gradients
        ## call to update the parameters
        dZ = softmax_cross_entropy_loss_der(Y, cache)
        grads = multi_layer_backward(dZ, caches, parameters)
        parameters, alpha = update_parameters(parameters, grads, ii, learning_rate, decay_rate)

        
        costs.append(cost)
        val_costs.append(vcost)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, cost, alpha))
    
    return costs, val_costs, parameters

def accuracy(YPred, Y):
    
    Y_ = one_hot(Y.astype(int),10).T
    x = np.abs(YPred-Y_)
    x = np.sum(x)/(2*Y.shape[1])
    return (1-x)

def train(train_data, train_label, val_data, val_label, test_data, test_label, net_dims, learning_rate=0.1, decay_rate=0.00, iters = 501):
    np.random.seed(1)
    # initialize learning rate and num_iterations
    num_iterations = iters

    costs, val_costs, parameters = multi_layer_network(train_data, train_label, val_data, val_label, net_dims, \
            num_iterations, learning_rate, decay_rate)
    
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    val_Pred = classify(val_data, parameters)
    test_Pred = classify(test_data, parameters)

    #test loss
    # AL, caches = multi_layer_forward(test_data, parameters)
    # Af, cache, test_error = softmax_cross_entropy_loss(AL, test_label)

    trAcc = accuracy(train_Pred, train_label)
    valAcc = accuracy(val_Pred, val_label)
    teAcc = accuracy(test_Pred, test_label)
    print("\nAccuracy after",num_iterations,"decay_rate=",decay_rate)
    print("Training: {0:0.3f}%".format(trAcc*100), "Validation: {0:0.3f}%".format(valAcc*100), "Testing: {0:0.3f}%".format(teAcc*100))

    return costs, val_costs, 1-teAcc

def main():
    '''
    Trains a multilayer network for MNIST digit classification (all 10 digits)
    To create a network with 1 hidden layer of dimensions 800
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800]"
    The network will have the dimensions [784,800,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits

    To create a network with 2 hidden layers of dimensions 800 and 500
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800,500]"
    The network will have the dimensions [784,800,500,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits
    '''
    os.system("mkdir graphs_2")

    net_dims = ast.literal_eval( sys.argv[1] )
    net_dims.append(10) # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(net_dims))

    # getting the subset dataset from MNIST
    train_data, train_label, val_data, val_label, test_data, test_label = \
            mnist(noTrSamples=5000, noValSamples=1000, noTsSamples=1000,\
            digit_range=[0,1,2,3,4,5,6,7,8,9],\
            noTrPerClass=500, noValPerClass=100, noTsPerClass=100)
    
    # initialize learning rate and num_iterations
    learning_rate = 0.2
    num_iterations = 1001
    decay_rate = 0.01

    costs, val_costs, test_error = train(train_data, train_label, val_data, val_label, test_data, test_label, net_dims, learning_rate, decay_rate, num_iterations)
    
    ### CODE HERE to plot costs
    #### Q2 1)
    x_points = [i for i in range(0,len(costs))]

    plt.clf()
    plt.title("Error v/s Iterations")
    cost_line, = plt.plot(x_points, costs, 'b-', label="Training")
    val_line, = plt.plot(x_points, val_costs, 'r-', label="Validation")
    plt.legend(handles=[cost_line, val_line])
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.savefig('graphs_2/Error_vs_Iterations.png')
    # plt.show()
    ####

    #### Q2 2)
    print("Test Error or Misclassification %: "+str(test_error*100))
    print("\n")
    ####

    #### Q2 3)
    c ,vc = [], []
    all_costs = []
    all_v_costs = []
    test_errors = []
    lrs = []
    for i in [0.1,0.5,1,10]:
        print("###")
        print("Learning Rate: ",i)
        
        tc, tvc, t_e = train(train_data, train_label, val_data, val_label, test_data, test_label, net_dims, i, 0.01, iters=1001)
        x_points = [i for i in range(0,len(tc))]
        
        plt.clf()
        plt.title('Learning Rate:'+str(i))
        plt.ylabel('Error')
        plt.xlabel('Iterations')
        cost_line, = plt.plot(x_points, tc, 'b-', label="Training")
        val_line, = plt.plot(x_points, tvc, 'r-', label="Validation")
        plt.legend(handles=[cost_line, val_line])
        plt.savefig('graphs_2/Error_Graph_Learning_Rate_'+str(i)+'.png')
        # plt.show()

        lrs.append(i)
        all_costs.append(tc)
        all_v_costs.append(tvc)
        test_errors.append(t_e)
        c.append(tc[-1])
        vc.append(tvc[-1])
        print("###")
        print("\n")

    #All lines together
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
    plt.savefig('graphs_2/Error for all Learning Rates'+str(i*0.01)+'.png')



    # Error (After x iterations) v/s Learning Rate graph 
    # x_points = list(range(1, 21, 1))
    # x_points = [i*0.01 for i in x_points]
    # plt.clf()
    # plt.plot(x_points, c, 'b-', x_points, vc, 'r-')
    # plt.ylabel('Error')
    # plt.xlabel('Learning Rate')
    # plt.savefig('graphs_2/Error_vs_Learning_Rate.png')
    # plt.show()
    ####

    #### Q2 4)  
    ix = np.argmin(vc)
    lr = lrs[ix]
    print("Minimum validation error found when learning_rate="+str(lr)+", Test accuracy for that run has been printed above")
    ####



if __name__ == "__main__":
    main()