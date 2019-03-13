#Stephanie Wankowicz
#3/13/2019
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.pyplot as plt

'''
 Implement a feed-forward, three-layer, neural network with standard sigmoidal units.
 Your program should allow for variation in the size of input layer, hidden layer, and output layer.
 Youwill need to write your code to support cross-validation.
 We expect that you will be able toproduce fast enough code to be of use in the learning task at hand.
 You will want to make surethat your code can learn the 8x3x8 encoder problem prior to attempting the Rap1learningtask.
'''
'''
first, we are going to develop a class to represent the neural network
'''
class neuralNetwork(object):
    def __init__(self,input_layer_size, hidden_layer_size, output_layer_size, Lambda): 
   '''
   NN has 3 layers(input, hidden, output). 
   INPUT (hyperparameters): 
   Input layer size- number of nodes for the input
   Hidden layer size- number of nodes in the hidden layer
   Output layer size- number of nodes in the output layer
   Lambda- Regularization Rate; preventing the weights from getting out of control. Pushes each weight and the mean of the weights towards zero.
   '''
        self.inputLayerSize = input_layer_size #setting the input layer size
        self.hiddenLayerSize = hidden_layer_size #setting the hidden layer size
        self.outputLayerSize = output_layer_size #setting the output layer size
        # Initialize the sinapses with random weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize) #setting the initial weights as random
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize) #setting the initial weights as random (This is also what is helping to set up our NN to be 8x3x8).
        self.Lambda = Lambda #setting the lambda value as your input lambda

    def forwardPropagate(self, input): 
     '''
     This function is going to move you forward through the NN. 
     '''
        self.z2 = np.dot(input, self.W1) #It is going to take the dot product of the input and the first set of weights.
        self.a2 = self.sigmoid(self.z2) #Perform the activation function on the middle (hidden) layer.
        self.z3 = np.dot(self.a2, self.W2) #Dot produce of the activated values in the middle (hidden) layer to produce the input for the third layer
        output = self.sigmoid(self.z3) #Perform the activation function on the third layer and produce the predicted output
        return output #return the third layer 

    def sigmoid(self, z):
    '''
    Function to perform sigmoid activation function; doing this on an array.
    This is a special case of the logistic function & will squash the output to be between 0 and 1.
    INPUT: array of weighted values
    OUTPUT: value between 0 and 1
    '''
        return 1 / (1 + np.exp(-z)) 

    def sigmoid_der(self, z):
    '''
    Function to perform the derivative of the sigmoid activation function; doing this on an array
    '''
        return np.exp(-z)/((1 + np.exp(-z)) ** 2) 

    def cost(self, input, known_output): 
     '''
    Measurement of goodness of fit of NN output data to the known output. 
    This will be used for our backpropagation.
    It based on the average of the output activation values.
    We also have our lambda value in here to regulaize 
    sum the squared difference between the predicted and known output
    ??lambda section
     '''
        self.h = self.forwardPropagate(input) #going to move forward through the neural network and get out some output value
        J = 0.5 * np.sum((known_output - self.h) ** 2) / input.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2)) #calculate the difference beween output of forward NN & known output
        return J

    def cost_der(self, X, y):
     '''
     
     '''
        self.h = self.forwardPropagate(X)
        delta3 = np.multiply(-(y - self.h), self.sigmoid_der(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda * self.W2
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_der(self.z2)
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda * self.W1

        return dJdW1, dJdW2

    def unrollweights(self):
        return np.concatenate((self.W1.ravel(), self.W2.ravel()))

    def set_weights(self, weights):
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(weights[0:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(weights[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def unrollGradients(self, X, y):
        dJdW1, dJdW2 = self.cost_der(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

# Class for train a neural network
class Trainer(object):
    def __init__(self, NN):
        self.N = NN

    def callback(self, weights):
        self.N.set_weights(weights)
        self.J.append(self.N.cost(self.X, self.y))

    def costFunctionWrapper(self, weights, X, y):
        self.N.set_weights(weights)
        #print('here2')
        cost = self.N.cost(X, y)
        #print('here3')
        grad = self.N.unrollGradients(X, y)
        #print('here4')
        #print(cost)
        #print(grad)
        return cost, grad

    def train(self, X, y):
        self.X = X
        self.y = y
        self.J = []
        weights_0 = self.N.unrollweights() #initial weights
        options = {'maxiter': 5000, 'disp': True}
        #print('here')
        opt = optimize.minimize(self.costFunctionWrapper, weights_0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callback)
        #print(opt)
        self.N.set_weithts(opt.x)
        self.optimizationResults = opt
