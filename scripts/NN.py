#3/15/2019
import numpy as np
from scipy import optimize

'''
 Implement a feed-forward, three-layer, neural network with standard sigmoidal units.
 Your program should allow for variation in the size of input layer, hidden layer, and output layer.
 Youwill need to write your code to support cross-validation.
 We expect that you will be able toproduce fast enough code to be of use in the learning task at hand.
 You will want to make surethat your code can learn the 8x3x8 encoder problem prior to attempting the Rap1learningtask.
'''

'''
   NN has 3 layers(input, hidden, output).
   INPUT (hyperparameters):
   Input layer size- number of nodes for the input
   Hidden layer size- number of nodes in the hidden layer
   Output layer size- number of nodes in the output layer
   Lambda- Regularization Rate; preventing the weights from getting out of control. Pushes each weight and the mean of the weights towards zero.
'''

class Neural_Network(object):
    def __init__(self,input_layer_size, hidden_layer_size, output_layer_size, Lambda):
        self.inputLayerSize = input_layer_size #setting the input layer size
        self.hiddenLayerSize = hidden_layer_size #setting the hidden layer size
        self.outputLayerSize = output_layer_size #setting the output layer size
        # Initialize with random weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize) #setting the initial weights as random
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize) #setting the initial weights as random (This is also what is helping to set up our NN to be 8x3x8).
        self.B1 = np.random.uniform(size=(1, self.hiddenLayerSize)) # bias for hidden layer, to prevent overfitting
        self.B2 = np.random.uniform(size=(1, self.outputLayerSize)) # bias for output layer
        self.Lambda = Lambda #setting the lambda value (regularization factor to make sure we are not overfitting. )
    def forward(self, input):
     '''
     This function is going to move you forward through the NN.
     '''
     self.hidden = np.dot(input, self.W1) #take the dot product of the input and the first set of weights.
     #print(self.hidden.shape)
     self.hid_act = self.sigmoid(self.hidden) +self.B1 #Perform the activation function on the middle (hidden) layer.
     #print(self.hid_act.shape)
     self.output = np.dot(self.hid_act, self.W2) +self.B2 #Dot produce of the activated values in the middle (hidden) layer to produce the input for the third layer
     #print(self.output.shape)
     self.pred_out = self.sigmoid(self.output) #Perform the activation function on the third layer and produce the predicted output
     return self.pred_out #return the third layer



    def back_prop(self, input,output,learning_rate):
        """
		Once we have a forward NN, we are going to see how bad our predictions are. 
		Optimize prediction by minimizing error with gradient descent. 
		INPUT: expected output and output activation
		OUTPUT: error of our original NN. We are also resetting weights and biases to the move more forward.
	"""
        error=(output-self.pred_out)#calculating the error between the output and the predicted output layer
        #calculate the derivative of the activation function to get the gradient slope
        grad_out=self.sigmoid_der(self.pred_out)
        grad_hid=self.sigmoid_der(self.hid_act) 
        delta_out=grad_out*error 
        #print(delta_out.shape)
        delta_hid=np.dot(delta_out,self.W2.T) * grad_hid
        #update weights by moving in the direction of the gradient. The size of the step is defined by the learning rate (hyperparameter).
        self.W2 += (np.dot(self.hid_act.T,delta_out)*learning_rate) #+ self.Lambda * self.W1
        self.W1 += (np.dot(input.T, delta_hid)*learning_rate) #+ self.Lambda * self.W1 #lambda serving as a regularization factor
        self.B2 += np.sum(delta_out,axis=0)*learning_rate 
        self.B1 += np.sum(delta_hid,axis=0)*learning_rate
        return error

    def train(self, input, known_output, iterations, learning_rate):
        '''
	To train, we are going to iteratively move forward and backwards through our NN, updating weights and biases.
	OUTPUT: We are going to keep track of the errors as we iterate through. We should see these errors drop and then plataeu. 
	'''
	#print('starting!')
        #i=0
	
        error_list=[]
        for i in range(iterations):
            self.forward(input)
            error=self.back_prop(input,known_output,learning_rate)
            average_error=np.average(error**2) #we are outputing the average error between all output nodes
            i+=1
        return average_error

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
	This is to perform the gradient movement through the NN. 
        '''
        return np.exp(-z)/((1 + np.exp(-z)) ** 2)
