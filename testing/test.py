
from scripts import NN

#testing the mathematical functions

assert NN.sigmoid_der(2)==0.5

#testing the autoencoder
x = np.identity(8)
y = np.identity(8)
NN = Neural_Network(input_layer_size=8, hidden_layer_size=3, output_layer_size=8, Lambda=2e-6)
NN.train(x,y,10000,0.45)
predict=NN.forward(x)
assert len(predict.shape[0]) == 8
