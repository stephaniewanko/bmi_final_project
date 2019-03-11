import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.pyplot as plt

'''
 Implement a feed-forward, three-layer, neural network with standard sigmoidal units.
 Yourprogram should allow for variation in the size of input layer, hidden layer, and output layer.
 Youwill need to write your code to support cross-validation.
 We expect that you will be able toproduce fast enough code to be of use in the learning task at hand.
 You will want to make surethat your code can learn the 8x3x8 encoder problem prior to attempting the Rap1learningtask.
'''

# Class representing the neural network
class neuralNetwork(object):
    def __init__(self,input_layer_size, hidden_layer_size, output_layer_size, Lambda):
        self.inputLayerSize = input_layer_size
        self.hiddenLayerSize = hidden_layer_size
        self.outputLayerSize = output_layer_size
        # Initialize the sinapses with random weights
        self.theta1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.theta2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.Lambda = Lambda

    def forwardPropagate(self, X):
        self.z2 = np.dot(X, self.theta1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.theta2)
        h = self.sigmoid(self.z3)
        return h

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z)/((1 + np.exp(-z)) ** 2)

    def cost(self, X, y):
        self.h = self.forwardPropagate(X)
        J = 0.5 * np.sum((y - self.h) ** 2) / X.shape[0] + (self.Lambda/2)*(np.sum(self.theta1**2)+np.sum(self.theta2**2))
        return J

    def costPrime(self, X, y):
        self.h = self.forwardPropagate(X)
        delta3 = np.multiply(-(y - self.h), self.sigmoidPrime(self.z3))
        dJdTheta2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda * self.theta2
        delta2 = np.dot(delta3, self.theta2.T) * self.sigmoidPrime(self.z2)
        dJdTheta1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda * self.theta1

        return dJdTheta1, dJdTheta2

    def unrollThetas(self):
        return np.concatenate((self.theta1.ravel(), self.theta2.ravel()))

    def setThetas(self, thetas):
        theta1_end = self.hiddenLayerSize * self.inputLayerSize
        self.theta1 = np.reshape(thetas[0:theta1_end], (self.inputLayerSize, self.hiddenLayerSize))
        theta2_end = theta1_end + self.hiddenLayerSize * self.outputLayerSize
        self.theta2 = np.reshape(thetas[theta1_end:theta2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def unrollGradients(self, X, y):
        dJdTheta1, dJdTheta2 = self.costPrime(X, y)
        return np.concatenate((dJdTheta1.ravel(), dJdTheta2.ravel()))

# Class for train a neural network
class Trainer(object):
    def __init__(self, NN):
        self.N = NN

    def callback(self, thetas):
        self.N.setThetas(thetas)
        self.J.append(self.N.cost(self.X, self.y))

    def costFunctionWrapper(self, thetas, X, y):
        self.N.setThetas(thetas)
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
        thetas_0 = self.N.unrollThetas()
        options = {'maxiter': 5000, 'disp': True}
        #print('here')
        opt = optimize.minimize(self.costFunctionWrapper, thetas_0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callback)
        #print(opt)
        self.N.setThetas(opt.x)
        self.optimizationResults = opt
