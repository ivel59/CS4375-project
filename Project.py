#Vi Le , Yvonne Hsiao
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import requests

class customModel:
    def __init__(self, inputSize, hiddenSize, outputSize):
        # Store input and hidden dimensions
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        
        # Hidden state and cell state
        self.hiddenState = np.zeros((hiddenSize, 1))
        self.cellState = np.zeros((hiddenSize, 1))
       
        # Weights and biases for first stager
        self.weightFirst = np.random.randn(hiddenSize, inputSize)
        self.weightfirst = np.random.randn(hiddenSize, hiddenSize)
        self.biasFirst = np.zeros((hiddenSize, 1))
        
        # Weights and biases for sencond stage
        self.weightSecond = np.random.randn(hiddenSize, inputSize)
        self.weightsecond = np.random.randn(hiddenSize, hiddenSize)
        self.biasSecond = np.zeros((hiddenSize, 1))
        
        # Weights and biases for 3rd stage
        self.weightThird = np.random.randn(hiddenSize, inputSize)
        self.weightthird = np.random.randn(hiddenSize, hiddenSize)
        self.biasThird = np.zeros((hiddenSize, 1))
        
        # Weights and biases for cell
        self.wieghtCell = np.random.randn(hiddenSize, inputSize)
        self.wieghtcell = np.random.randn(hiddenSize, hiddenSize)
        self.biasCell = np.zeros((hiddenSize, 1))

        # Weights and biases for the output-Weight matrix
        self.wOut = np.random.randn(outputSize, hiddenSize)
        self.bOut = np.zeros((outputSize, 1))  # Bias term
        # Regularization
        self.dropout_prob = 0.2
    def sgm(self, x):
        return 1 / (1 + np.exp(-x))
    def fw(self, x):
        # Perform LSTM operations
        self.inputGate = self.sgm(np.dot(self.weightSecond, x) + np.dot(self.weightsecond, self.hiddenState) + self.biasSecond)
        self.forgetGate = self.sgm(np.dot(self.weightFirst, x) + np.dot(self.weightfirst, self.hiddenState) + self.biasFirst)
        self.outputGate = self.sgm(np.dot(self.weightThird, x) + np.dot(self.weightthird, self.hiddenState) + self.biasThird)
        self.cellGate = np.tanh(np.dot(self.wieghtCell, x) + np.dot(self.wieghtcell, self.hiddenState) + self.biasCell)
        
        # Update cell state and hidden state
        self.cellState = self.forgetGate * self.cellState + self.inputGate * self.cellGate
        self.hiddenState = self.outputGate * np.tanh(self.cellState)
        #output = self.linear_layer(self.hidden_state)
        output = np.dot(self.wOut, self.hiddenState) + self.bOut
        return output  # Return the output from the hidden state
    def backward(self, x, target, learning_rate):
        # Perform backpropagation
        output = self.fw(x)  # Output from the forward pass
        
        # Compute loss (assuming mean squared error)
        loss = 0.5 * np.square(output - target)
        
        # Gradient of the loss with respect to the output
        bwOutput = output - target
        
        # Gradient of the loss with respect to the output layer weights and biases
        bwW_out = np.dot(bwOutput, self.hiddenState.T)
        bwb_out = bwOutput
        
        # Gradient of the loss with respect to the hidden state
        bwHiddenState = np.dot(self.wOut.T, bwOutput)
        
        # Gradients through the gates and cell state
        bwOutputGate = bwHiddenState * np.tanh(self.cellState) * self.hiddenState * (1 - self.outputGate)
        bwCellState = bwHiddenState * self.outputGate * (1 - np.square(np.tanh(self.cellState)))
        bwInputGate = bwCellState * self.cellGate * (1 - self.inputGate) * self.inputGate
        bwForgetGate = bwCellState * self.cellState * (1 - self.forgetGate) * self.forgetGate
        
        # Biases and weights gradients
        gradients = {
            'bOut': bwb_out,
            'wOut': bwW_out,
            'biasCell': bwOutputGate,
            'weightThird': np.dot(bwOutputGate, x.T),
            'weightthird': np.dot(bwOutputGate, self.hiddenState.T),
            'biasCell': bwCellState,
            'wieghtCell': np.dot(bwCellState, x.T),
            'wieghtcell': np.dot(bwCellState, self.hiddenState.T),
            'biasSecond': bwInputGate,
            'weightSecond': np.dot(bwInputGate, x.T),
            'weightsecond': np.dot(bwInputGate, self.hiddenState.T),
            'biasFirst': bwForgetGate,
            'weightFirst': np.dot(bwForgetGate, x.T),
            'weightfirst': np.dot(bwForgetGate, self.hiddenState.T)
        }
        
        # Update weights and biases thur out
        for param_name, gradient in gradients.items():setattr(self, param_name, getattr(self, param_name) - learning_rate * gradient)
        
        return loss  # For keeping track of the behavior for training purpose

    
    def compute_loss(self, predicted, target):
        # Calculate mean squared error
        loss = np.mean((predicted - target) ** 2)
        return loss


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create a log file append mode
logFile = open('trials_log.txt', 'a')
# Get dataset link from github
github_link = 'https://raw.githubusercontent.com/ivel59/CS4375-HW1/main/coin_Bitcoin.csv'
# Download the raw data file
response = requests.get(github_link)
# Check if the download was successful
if response.status_code == 200:
    # Save the downloaded file locally
    with open('dataset.csv', 'wb') as file:
        file.write(response.content)    
    # Load the dataset into a DataFrame
    bitcoinData = pd.read_csv('dataset.csv')
else:
    print('Failed to download data.')

# Preprocess the data
scaler = MinMaxScaler()
bitcoinData[['High', 'Low', 'Open', 'Close']] = scaler.fit_transform(bitcoinData[['High', 'Low', 'Open', 'Close']])

# Split the data close will be target
features = bitcoinData[['High', 'Low', 'Open', 'Close']].values
targets = bitcoinData['Close'].values

# Split into 80 train and 20 test
trainFeatures, testFeatures, trainTargets, testTargets = train_test_split(features, targets, test_size=0.2, random_state=42)

# Define hyperparameters, tune these to analyze the performance
inputSize = 4 
hiddenSize = 8
learningRate = 0.1
numEpochs = 100

logFile.write(f"Hidden size: {hiddenSize}, Learning rate: {learningRate}, Epoch: {numEpochs}\n")
# Create an instance of training model
lstmModel = customModel(inputSize, hiddenSize, 1)

# Training, run the model and monitor loss
for epoch in range(numEpochs):
    trainLoss = 0.0
    # Iterate through the training dataset
    for i in range(len(trainFeatures)):
        # Perform forward pass
        inputData = trainFeatures[i].reshape(-1, 1) 
        predictedOutput = lstmModel.fw(inputData)
        
        # Compute training loss
        loss = lstmModel.compute_loss(predictedOutput, trainTargets[i])
        lstmModel.backward(inputData, targets[i], learningRate)
        trainLoss += loss
    
    # Calculate average training loss for the epoch
    averageTrainingLoss = trainLoss / len(trainFeatures)
   
    logFile.write(f'Epoch [{epoch+1}/{numEpochs}], Train Loss: {averageTrainingLoss}\n')

# Testing, use the model to predict output (closing price) and compare to actual price
totalError = 0.0
predictedOutput = []
for i in range(len(testFeatures)):
    inputDataTest = testFeatures[i].reshape(-1, 1)
    targetTest = testTargets[i]
    
    outputTest = lstmModel.fw(inputDataTest)
    predictedOutput.append(outputTest.item())
    errorTest = np.abs(outputTest - targetTest)
    totalError += errorTest

averageError = totalError / len(testFeatures)  
# Calculate average absolute error
averageErrorVal = np.mean(averageError)

# Print and log the average absolute error
print(f"Average Absolute Error on Test Set: {averageErrorVal:.4f}")
logFile.write(f'Average Absolute Error on Test Set: {averageErrorVal}\n')

# Plot Actual vs. Predicted Prices
plt.figure(figsize=(8, 6))
plt.plot(testTargets, label='Actual Output', marker='o')
plt.plot(predictedOutput, label='Predicted Output', marker='x')
plt.xlabel('Data Points')
plt.ylabel('Normalized Price')
plt.title('Actual vs. Predicted Output')
plt.legend()
plt.grid(True)
plt.show()

# Error Metrics on Test Set
errors = np.abs(predictedOutput - testTargets)  # Calculate errors
plt.figure(figsize=(8, 6))
plt.plot(errors, marker='o', linestyle='')
plt.xlabel('Samples')
plt.ylabel('Absolute Error')
plt.title('Absolute Errors on Test Set')
plt.show()