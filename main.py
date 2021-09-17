'''=======================================================================
 SPEACH RECOGNITION PROGRAM FOR 2 SECONDS PRE-RECORDED AUDIO
    DEVELOPED BY:
        - ABRIL BERENICE BAUTISTA ROMAN         ITESM CEM
        - JOSE ANGEL DEL ANGEL DOMINGUEZ        ITESM CEM
        - RAUL LOPEZ MUSITO                     ITESM CEM
        - LEONARDO JAVIER NAVA CASTELLANOS      ITESM CEM 
====================================================================='''


''' ________ COMONLY USED LIBRARIES ________ '''
import numpy as np                #numeric and matrix libary

'''___________ Neural Net LIBRARIES_____________'''
from nn_model import neuralNetwork
from numpy_matrix_list import numpy_matrix_list
from simple_test_data import *

''' ====================== WE TRAIN NEURAL NETWORK ====================== '''

print("\n============= TRAINING THE NEURAL NETWORK ==============\n")
simple_test_dataD = generate_simple_test_data()
Xdata = simple_test_dataD[0]
X_train = np.array(Xdata)
# Y_train is an array where we will store the answers that correspond to the training data
Y_train = simple_test_dataD[1]
Y_train = np.array(Y_train)
print("Y_train is :\n" + str(Y_train))
data_inputs = X_train
data_outputs = Y_train

bias_term = 0 #please change this line depending on the neral network model you import

# We define the neurons and layers that our neural network will have (network architecture) 
HL1_neurons = 9
input_HL1_weights = np.random.uniform(low=-1, high=1, size=(data_inputs.shape[1]+bias_term, HL1_neurons))
HL2_neurons = 9
HL1_HL2_weights = np.random.uniform(low=-1, high=1, size=(HL1_neurons+bias_term, HL2_neurons))
output_neurons = 2
HL2_output_weights = np.random.uniform(low=-1, high=1, size=(HL2_neurons+bias_term, output_neurons))
weights = numpy_matrix_list([input_HL1_weights,  HL1_HL2_weights, HL2_output_weights])
# we initialize and train our neural network with the corresponding training data 
neural_network = neuralNetwork(inital_weights = weights, trainX = data_inputs, trainY = data_outputs, learning_rate = 0.02)
neural_network.train_network(num_iterations = 200)

print("=================== CNN trained correctly ===================")

print("\n\n Let's predict training inputs, results must be : \n {f} \n\n".format(f = Y_train))
print("Prediction results for training are: \n {f} \n".format(f = neural_network.predict_outputs( X_train) ))


print("=================== AGORITHM TESTING ===================")
print("please present every row of the list in the format\n 'n m k' \n")

testing = True
while testing:
    test = []
    for i in range(3):
        row = input('row {n}: '.format(n=i))
        row = row.split()
        row = list(map(int, row))
        test += row
    test = np.array([test])
    print(  "\ntest matrix is: \n {f} \n".format(f=test) )
    print("Prediction results are: \n {f} \n".format(f = neural_network.predict_outputs( test ) ))
    testing = bool(int(input("Do you want to test again? (1=yes/0=no): ")))
