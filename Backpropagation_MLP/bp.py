import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weights, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.
        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network
        Returns:
            (nabla_b, nabla_w): tuple containing the gradient for all the biases
                and weights. nabla_b and nabla_w should be the same shape as 
                input biases and weights
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = []
    nabla_w = []

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###
    
    activations = [x] #store activations
    h = [] #store the z vectors 
    for i in range(num_layers-1):
        ai = biases[i] + np.dot(weights[i], activations[i])
        h.append(ai)
        activations.append(sigmoid(ai)) 
        
    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    g = (cost).delta(activations[-1], y) #delta for output layer for cross entropy cost
    nabla_w=[np.dot(g, np.transpose(activations[-2]))]
    nabla_b=[g]
    
    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###
    for i in range(2,num_layers):  
        g = np.dot(weights[-i+1].transpose(), g) * sigmoid_prime(h[-i])
        nabla_b = [g] + nabla_b
        nabla_w = [np.dot(g, np.transpose(activations[-i-1]))] + nabla_w

    return (nabla_b, nabla_w)
