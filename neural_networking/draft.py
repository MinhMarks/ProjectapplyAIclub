# https://github.com/mnielsen/neural-networks-and-deep-learning
# https://www.geeksforgeeks.org/activation-functions-neural-networks/
# https://www.enjoyalgorithms.com/blog/activation-function-for-hidden-layers-in-neural-networks

import sys
import numpy as np 
import matplotlib.pyplot as plt 
import nnfs 
from nnfs.datasets import spiral_data 


nnfs.init() 
np.random.seed(0) 

'''
# pre_neuron = [ [1.4, 2.1, 3.3], 
#                [1.4, 3.3, 2.3]] 
# weight = [ [2.1, 3.2, 2.2],
#            [2.3, 4.0, 1.1] ]
# bias = [0.5, 9.1]  
# # output = pre_neuron[0] * weight[0] + pre_neuron[1] * weight[1] + pre_neuron[2] * weight[2] + bias 
# output = np.dot( weight, np.array(pre_neuron).T ) + bias 
# print(output)  
# test = [[1, 2, 3, 2.5],
#         [2.0, 5.0, -1.0, 2.0],
#         [-1.5, 2.7, 3.3, -0.8]]
# X, Y = spiral_data(100, 3)
# plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.show()
# print(X) 
''' 


class Layer_Dense: 
    def __init__(self, s_input, s_neuron ):
        self.weights = 0.01 * np.random.randn(s_input, s_neuron)
        self.bias = np.zeros((1,s_neuron)) 
    def forward(self, inputs ): 
        self.output = np.dot( inputs, self.weights ) + self.bias 
        
class Activation_ReLU: 
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) 

class Activation_Softmax: 
    def forward(self, inputs): 
        exp_value = np.exp(inputs - np.max(inputs, axis =1, keepdims=True)) 
        probabilities = exp_value / np.sum(exp_value, axis=1, keepdims=True) 
        self.output = probabilities 

class Loss: 
    def calculate( self, output, y ): 
        sample_losse = self.forward(output, y) 
        data_lose = np.mean( sample_losse ) 
        return data_lose  

class Loss_CategoricalCrossentropy(Loss): 
    def forward( self, y_pred, y_true):
        samples = len(y_pred) 
        y_pred_clip = np.clip(y_pred, 1e-7, 1- 1e-7) 
        if len(y_true.shape) == 1: 
            correct_confidences = y_pred_clip[range(samples), y_true] 
        elif len(y_true.shape) == 2: 
            correct_confidences = np.sum(y_pred_clip * y_true, axis=1 )
        
        negative_log_likelihood = -np.log(correct_confidences) 
        return negative_log_likelihood 


X, y = spiral_data(100, 3) 

dense1 = Layer_Dense(2, 3) 
activation1 = Activation_ReLU() 

dense2 = Layer_Dense(3, 3) 
activation2= Activation_Softmax()

dense1.forward(X) 
activation1.forward(dense1.output) 

dense2.forward(activation1.output) 
activation2.forward(dense2.output) 

loss_function = Loss_CategoricalCrossentropy() 
loss = loss_function.calculate(activation2.output, y) 

# navie upgrade 

lowest_loss = 999999 

best_dense1_weights = dense1.weights 
best_dense1_bias = dense1.bias
best_dense2_weights = dense2.weights 
best_dense2_bias = dense2.bias




for iteration in range(1,1000000): 
    dense1.weights = 0.05 * np.random.randn(2,3) 
    dense1.bias = 0.05 * np.random.randn(1,3) 
    dense2.weights = 0.05 * np.random.randn(3,3) 
    dense2.bias = 0.05 * np.random.randn(1,3) 

    dense1.forward(X) 
    activation1.forward(dense1.output)
    dense2.forward(activation1.output) 
    activation2.forward(dense2.output)
    
    loss = loss_function.calculate(activation2.output,y) 
    
    predictions = np.argmax(activation2.output, axis = 1) 
    accuracy = np.mean(predictions==y) 
    
    if ( loss < lowest_loss ): 
        with open('D:/Machine Learning/neural_networking/out.txt', 'a') as file:
            file.write(str(loss)+" ")
            file.write(str(accuracy)+" ")
            file.write("\n")
        lowest_loss = loss 
        best_dense1_bias = dense1.bias 
        best_dense1_weights = dense1.weights 
        best_dense2_bias = dense2.bias 
        best_dense2_weights = dense2.weights  
    

# with open('D:/Machine Learning/neural_networking/out.txt', 'w') as file:
#     # file.write(np.array2string(y)) 
#     # file.write(np.array2string(activation1.output))
#     # file.write(np.array2string(activation2.output)) 
#     file.write(str(loss)) 




 
        