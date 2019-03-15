#RUN FUNCTIONS

#import packages
import numpy as np
import pandas as pd
import sklearn
from Bio import SeqIO
import re
from NN import Neural_Network
import matplotlib.pyplot as plt

np.random.seed(0) #just getting the same random numbers so we can evaluate our NN


#################__________________PART 1: Run autoencoder__________________###############
#create identity 8x8 matrix
x = np.identity(8)
y = np.identity(8)

NN = Neural_Network(input_layer_size=8, hidden_layer_size=3, output_layer_size=8, Lambda=2e-6)
#tested multiple lambda values. This lambda value gave me the best
#print(NN)
#T=trainer(NN)
#T.train(x, y)
#print(T)
#predict = NN.forward(x)
#NN.train(x,y,10000,0.45)
#T.train(x, y)
#print(d1)
#predict = NN.forward(x)

print('predict')
#rint(predict)
print('Now rounding Matrix')
# the original output is continous, but we want the output to be discrete
#print(predict.round())




#################__________________PART 2: DNA INPUT__________________###############

#Pre-process DNA for NN input

#1) remove some of the false negatives from the fasta file & create new fasta file.

###########________________________________________#################
pos_seq=pd.read_csv('data/rap1-lieb-positives.txt', sep='\t', header=None)
neg_seq=list(SeqIO.parse("data/yeast-upstream-1k-negative.fa", 'fasta'))
print(len(neg_seq))
remove_seq=[]
for y in range(len(neg_seq)):
    #find and remove exact matches
    for x in pos_seq[0]:
        if re.search(x, str(neg_seq[y].seq)):
            remove_seq.append(str(neg_seq[y].seq))
            #print('found')
        else:
            continue
remove_seq=set(remove_seq)

with open('New_Neg_seq.fa', "w") as f:
    for y in range(len(neg_seq)):
        if str(neg_seq[y].seq) in remove_seq:
            #print('no')
            continue
        else:
            SeqIO.write(neg_seq[y], f, "fasta")

def select_sequences(neg_seq_file, pos_seq_file, num_train):
    #2) Create inputs from positive and negative sequences
    #import positive
    pos_seq=[]
    pos_file= open(pos_seq_file, 'r')
    for line in pos_file.readlines():
        pos_seq.append(line.strip())
    pos_input = list(np.random.choice(pos_seq, num_train))
    #take random (consequetive) 17 bases of the neg seq (same size as positive seq)
    neg_seq = []
    neg_file = open('New_Neg_seq.fa', 'r')
    for line in neg_file.readlines():
        if (line[0] == '>'):
            continue
        else:
            neg_seq.append(line.strip()[0:17])
            #for each run through, only take 137 negative sequenes==# of positive sequences

            #remove neg_seq with n
    neg_input = list(np.random.choice(neg_seq, num_train))

    #create negative and positive outputs
    #one=positive
    #zero=negative
    pos_outputs = np.ones(len(pos_input)) #one
    neg_outputs = np.zeros(len(neg_input)) #zeros

    #combine and randomize input and outputs
    inputs=np.append(pos_input, neg_input) #np.random.shuffle()
    outputs=np.append(pos_outputs, neg_outputs)
    combined = list(zip(inputs, outputs))
    np.random.shuffle(combined)
    inputs[:], outputs[:] = zip(*combined)
    #for cross-validation, we are also going to select the testing postive and negative sequences
    pos_test=np.setdiff1d(pos_seq,pos_input)
    neg_test=np.setdiff1d(neg_seq,neg_input)
    pos_output_test = np.ones(len(pos_test)) #one
    neg_output_test = np.zeros(len(neg_test)) #zeros
    input_test=np.append(pos_test, neg_test)
    output_test=np.append(pos_output_test, neg_output_test)

    return inputs, outputs, input_test, output_test


#change DNA single letters into 4 letter matrix code
###########________________________________________#################
#A=0001
#C=1000
#G=0100
#T=0010

def DNA_input(inputs):
    s = (len(inputs), 4*len(inputs[0])) #create matrix w/num of sequences by 4*17(DNA input length)
    input_DNA = np.empty(s)
    #print(input_DNA)
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            index = j * 4
            if inputs[i][j] == "A":
                input_DNA[i,index]=0
                input_DNA[i,index+1]=0
                input_DNA[i,index+2]=0
                input_DNA[i,index+3]=1
            elif inputs[i][j] == "C":
                input_DNA[i,index]=1
                input_DNA[i,index+1]=0
                input_DNA[i,index+2]=0
                input_DNA[i,index+3]=0
            elif inputs[i][j] == "G":
                input_DNA[i,index]=0
                input_DNA[i,index+1]=1
                input_DNA[i,index+2]=0
                input_DNA[i,index+3]=0
            elif inputs[i][j] == "T":
                input_DNA[i,index]=0
                input_DNA[i,index+1]=0
                input_DNA[i,index+2]=1
                input_DNA[i,index+3]=0
    return input_DNA


#################__________________PART 3: Train NN on DNA Sequences__________________###############
#Train NN on positives and negatives (note: avoid overfitting)
#print(outputs.shape)
#print(input_DNA.shape)
input_train, output_train, input_test_seq, output_test=select_sequences('data/New_Neg_seq.fa','data/rap1-lieb-positives.txt', 90)
#print(output_train.shape)
#print(output_train.shape[0])
#print(output_train)
input_DNA_train=DNA_input(input_train)
#print(input_DNA_train)
print(input_DNA_train.shape[0])
print('starting DNA Neural Network')
NN_DNA = Neural_Network(input_layer_size=68, hidden_layer_size=3,output_layer_size=180, Lambda=2e-6)  # initialize NN #input_layer_size=input_DNA.shape[0], output_layer_size=outputs.shape[0], hidden_layer_size=10, Lambda=0.00000005
NN_DNA.train(input_DNA_train, output_train, iterations=50,learning_rate=0.45)
#initial prediction
print('training done')
predict = NN_DNA.forward(input_DNA_train)
#print(predict)
#print(predict.round()) #want this continous for the actual output
print('yay! something ran!!')

def learn_parms():
    error_list=pd.DataFrame()
    for i in [3,5,10,20,30,40,50,60,100]: #number of hidden nodes to test
        print(i)
        for learning_rate in [0.01,0.1,0.25,0.5,0.7,1,5]:
            #=k_folds
            input_train, output_train, input_test_seq, output_test=select_sequences('data/New_Neg_seq.fa','data/rap1-lieb-positives.txt', 90)
            input_DNA_train=DNA_input(input_train)
            NN_learn = Neural_Network(input_layer_size=68, hidden_layer_size=i,output_layer_size=180, Lambda=2e-6)  # initialize NN #input_layer_size=input_DNA.shape[0], output_layer_size=outputs.shape[0], hidden_layer_size=10, Lambda=0.00000005
            error=NN_learn.train(input_DNA_train, output_train, iterations=500,learning_rate=learning_rate)
            #print(error)
            #error_list.append(error)
            error_list.loc[i,learning_rate]=error
    print(error_list)
    return error_list
learning_errors=learn_parms()
learning_errors.to_csv('learn_parms.csv')


####______________________PART 4: CROSS VALIDATION______________________############
def cross_validation(pos_seq_file, neg_seq_file,num_train_runs, num_samples_training):
    """
    pick a subset of data for training, leave rest for testing
    We are going to select 180 sequences (90-positive, 90-negative) for training, leaving 74 (37-positive, 37-negative) for testing
    INPUT: Positive and Negative Sequences
    OUTPUT:
    We are going to look at the learning rate and the number of nodesin the hidden layer
    """
    for i in range(1,50,5): # we are going to train X times
        print(i)
        for lr in [0.01,0.5,0.05]:
            #=k_folds
            input_train, output_train, input_test_seq, output_test=select_sequences('data/New_Neg_seq.fa','data/rap1-lieb-positives.txt', 90)
        input_DNA_train=DNA_input(input_train)
        #avg_accuracy, avg_auc, tprs, fprs = k_folds(train_seq, train_exp, learn_rate, num_hidden)
        #output
    return x_training, y_training, x_validation, y_validation
