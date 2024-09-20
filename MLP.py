import pandas as pd
import numpy as np
from Processing import dataProcessing
from Main import Menu
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, n_inputs, n_neurons):
        # iniatlising variables
        lower_bound = -2 / n_inputs
        upper_bound = 2 / n_inputs

        self.weights_hidden = (upper_bound - lower_bound) * np.random.rand(n_inputs, n_neurons) + lower_bound
        self.biases_hidden = (upper_bound - lower_bound) * np.random.rand(1, n_neurons) + lower_bound
        
        self.weights_output = (upper_bound - lower_bound) * np.random.rand(n_neurons, 1 ) + lower_bound
        self.bias_output =  (upper_bound - lower_bound) * np.random.rand(1, 1) + lower_bound
        
        self.learning_rate = 0.05
        self.use_momentum = "y"
        self.momentum = 0.95

        self.m_weights_hidden =  self.weights_hidden
        self.m_biases_hidden =  self.biases_hidden
        
        self.m_weights_output =  self.weights_output
        self.m_bias_output = self.bias_output
 
        self.mse_values = [] # used for graphing 
        self.rmse_values = [] # used for graphing
        self.t_v_MSE = 0 #MSE of training and validation
        
    def hidden_activation_function(self, x):
        return 1 / (1 + np.exp(-x)) ##currently set to sigmoid activation

    def output_activation_function(self, x):
        return 1 / (1 + np.exp(-x)) # currently set to sigmoid activation

    def firstDiffSigmoid(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.inputs = inputs

        #inputs to hidden layer
        self.weighted_sum = np.dot(inputs, self.weights_hidden) + self.biases_hidden
        self.hidden_output = self.hidden_activation_function(self.weighted_sum)

        #hidden to output layer
        self.output_sum = np.dot(self.hidden_output, self.weights_output) + self.bias_output
        self.output = self.output_activation_function(self.output_sum)

    def backward(self, inputs, target, learning_rate): 
       # calculates error
        output_error = target - self.output
        
        # backpropagate error using the output error  to hidden layer
        output_delta = output_error * self.firstDiffSigmoid(self.output)
        hidden_error = np.dot(output_delta, self.weights_output.T)
        
        #checks if you shouold use momentum
        if self.use_momentum.lower() =="y":
            #print("using momenutm to speed up algo") ## debugging

            #update momentum terms
            self.m_weights_output = self.momentum * self.m_weights_output + learning_rate * np.dot(self.hidden_output.T, output_delta)
            self.m_biases_output = self.momentum * self.m_bias_output + learning_rate * np.sum(output_delta.to_numpy(), axis=0)

            #update weights and biases for output layer
            self.weights_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
            self.bias_output += learning_rate * np.sum(output_delta, axis=0)
            
            #update momentum terms for hidden layer
            self.m_weights_hidden = self.momentum  * self.m_weights_hidden + learning_rate * np.dot(inputs.T, hidden_error * self.firstDiffSigmoid(self.hidden_output))
            self.m_biases_hidden = self.momentum  * self.m_biases_hidden + learning_rate * np.sum(hidden_error * self.firstDiffSigmoid(self.hidden_output), axis=0)
            
            #update weights and biases hidden layer
            self.weights_hidden += self.m_weights_hidden
            self.biases_hidden += self.m_biases_hidden
        
        else:
            #update weights and biases output
            self.weights_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
            self.bias_output += learning_rate * np.sum(output_delta, axis=0)
            
            #update weights and biases hidden
            self.weights_hidden += learning_rate * np.dot(inputs.T, hidden_error * self.firstDiffSigmoid(self.hidden_output))
            self.biases_hidden += learning_rate * np.sum(hidden_error * self.firstDiffSigmoid(self.hidden_output), axis=0)
    
        
       

    def train(self, inputs, targets, epochs, learning_rate, v_inputs, v_targets): ## full batch training parameters
        validation_check_epochs = 500 ## checks dataset against validation after X number of epochs 
        current_validation_loss = 100000000000 ### high val

        for epoch in range(epochs):
            self.forward(inputs) # does forward pass with the inputs
            self.backward(inputs,targets, learning_rate) # does backwards pass 
            error = np.mean(np.square(targets - self.output)) # calculates MSE

            if epoch % 1000 == 0:### for debugging, checks current predicted of first row against target and fids mse
                print(f"in epoch {epoch}, for first row, predicted was {self.output[0][0]} actual output was {targets.iloc[0][0]} loss was {np.mean(np.square(targets.iloc[0][0] - self.output[0][0]))} ")
   
   
            self.mse_values.append(error.item()) ## adds MSE value to attribute
            self.rmse_values.append(np.sqrt(error.item())) # adds RMSE value to attributes

            
            if epoch % validation_check_epochs ==0:## after every x  epochs , test NN against validation dataset if loss is increasing stop testing
                validation_loss = ANN.validation(v_inputs,v_targets)
                if validation_loss > current_validation_loss:
                    print("the validtaion loss is increasing therferoe stop training")
                    break
                else:
                    current_validation_loss = validation_loss
            
            
    def train_mini_batch(self, inputs, targets, epochs, learning_rate, v_inputs, v_targets, batch_size): ## same inputs as full batch training but has batch size number specified
        validation_check_epochs = 1000 
        current_validation_loss = 100000000000

        for epoch in range(epochs):
            error = 0
            for batch_start in range(0, len(inputs), batch_size): # itereates through list in specifiec batch sizes 
                batch_inputs = inputs[batch_start:batch_start+batch_size] # sets input for this batch
                batch_targets = targets[batch_start:batch_start+batch_size] # sets targets for this batch


                self.forward(batch_inputs) # does forward pass on batch inputs 
                self.backward(batch_inputs, batch_targets, learning_rate) # back pass on batch

                batch_error = np.mean(np.square(batch_targets - self.output)) # finds MSE of batch
                error+= batch_error ## adds MSE of batch to current epoch MSE
                
            
            epoch_error = error/(len(inputs)/batch_size) # calculates MSE of epoch

            if epoch % 100 == 0:
                print(f"in epoch {epoch}, for first row, predicted was {self.output[0][0]} actual output was {targets.iloc[0][0]} loss was {np.mean(np.square(targets.iloc[0][0] - self.output[0][0]))} ")

            self.mse_values.append(epoch_error.item())
            self.rmse_values.append(np.sqrt(epoch_error.item()))
            
            
            if epoch % validation_check_epochs == 0:## after every x  epochs , test NN against validation dataset if loss is increasing stop testing
                validation_loss = ANN.validation(v_inputs, v_targets)
                if validation_loss > current_validation_loss:
                    print("the validation loss is increasing therefore stop training")
                    print("final epoch error was: ", epoch_error)
                    break
                else:
                    current_validation_loss = validation_loss
        
    def validation(self, validation_inputs, validation_targets):
        self.forward(validation_inputs)# does forward pass on inputs
        validation_loss = np.mean(np.square(validation_targets - self.output)) # MSE loss
        print(f"validation Loss: {validation_loss}")
        return validation_loss.item() # returns loss
    
    def plot_mse(self, mse_values):
        # Plotting RMSE against epoch number
        epochs = range(1, len(mse_values) + 1)
        plt.plot(epochs, mse_values, label="MSE", color="blue")
        plt.title("MSE against Epoch Number (using momentum)")
        plt.xlabel("Epoch Number")
        plt.ylabel("MSE")
        plt.legend()
        plt.plot(epochs, mse_values, label='MSE')
        plt.show()

    def plot_rmse(self, rmse_values):
        # Plotting RMSE against epoch number
        epochs = range(1, len(rmse_values) + 1)
        plt.plot(epochs, rmse_values, label='RMSE', color = "red")
        plt.title('RMSE against Epoch Number')
        plt.xlabel('Epoch Number')
        plt.ylabel('RMSE')
        plt.legend()
        plt.ylim(0,0.3)
        plt.yticks(np.arange(0,0.3,step= 0.03))
        plt.show()
    
    def SelectingNetwork(self,t_v_inputs, t_v_targets):# does forward pass on testing and validation dataset and calculates its loss
        self.forward(t_v_inputs) 
        self.t_v_MSE = np.mean(np.square(t_v_targets - self.output))
        print("-------")
        print(f"MSE score testing using training and validation data sets is {self.t_v_MSE.item()}")
        print("-------")
        
    
    def evaluation(self, e_inputs, e_targets):## called after training has finished and makes a forward pass on the testing data to receive a final score
        self.forward(e_inputs)# does forward pass on testing data
        #print(self.output) ## debugging 
        print("----")
        #print(e_targets) ## debugging 
        self.RMSE = np.sqrt(np.mean(np.square(self.output - e_targets))).item() # calculates RMSE
        self.MSRE = (1/len(e_targets))*np.sum(np.square((self.output-e_targets)/e_targets)).item() #calculates MSRE
        self.CE = 1-((np.sum(np.square(self.output-e_targets)))/np.sum(np.square(e_targets-np.mean(e_targets)))).item() # calculates CE
        print(f"Root Mean Squared Error: {self.RMSE}")
        print(f"Mean Squared Relative Error: {self.MSRE}")
        print(f"Coefficient of Efficiency: {self.CE}")
        
        print("   | predicted output | actual output") ## prints table of first 20 datapoints in testing with predicted vs actual
        for i in range(21):
           print(f"{i} |{self.output[i][0]}| {e_targets.iloc[i][0]}")
        

        plt.scatter(range(len(self.output)), self.output, color='blue', label='Predicted Outputs') #graph of actual vs predicted on y and datapoint number on x
        plt.scatter(range(len(e_targets)), e_targets, color='red', label='Target Outputs')
        plt.xlabel('Data Point') 
        plt.ylabel('standardised output')
        plt.title('Predicted vs Target Outputs')
        plt.legend()
        plt.show()

        ############
        plt.scatter(e_targets, self.output, color='blue', label='Predicted vs. Actual') # graph 2
        plt.plot(e_targets, e_targets, color='red', label='Perfect Prediction')  #diagonal line for perfect prediction 
        plt.xlabel('Actual Outputs')
        plt.ylabel('Predicted Outputs')
        plt.title('Scatter Plot of Predicted vs. Actual Outputs')
        plt.show()

   

    def saveNetwork(self): # asks user if they want to save the network and if they saves the important info to a file
        x = input("would you like to save this network: (y/n)")
        if x.lower() == "y":
            file = open("saved_mlps.txt", "a")
            file.write("Root Mean Squared Error: " +str(self.RMSE) + "\n")
            file.write("Mean Squared Relative Error: " +str(self.MSRE) + "\n")
            file.write("Coefficient of Efficiency: " +str(self.CE) + "\n")
            file.write("epochs: " + str(epochs) + "\n")
            file.write("inputs: " + str(Main_menu.no_inputs) + "\n" )
            file.write("hidden_nodes: " + str(Main_menu.neurons_in_hidden) + "\n")
            file.write("learning_rate: " + str(learning_rate) + "\n")
            #file.write("batch_size: " + str(batch_size) + "\n")
            file.write("weights_hidden: " + str(self.weights_hidden) + "\n")
            file.write("biases_hidden: " + str(self.biases_hidden) + "\n")
            file.write("weights_output: " + str(self.weights_output) + "\n")
            file.write("bias_output: " + str(self.bias_output) + "\n")
            file.write("final MSE" + str(self.t_v_MSE.item) + "\n")
            file.write("-------------------------------------" + "\n")
        

## calls class to process the data and standardise it each is a different dataset
data1 = dataProcessing("FEHDataStudent_processed_new.xlsx")
df_standardised_training = data1.Standardise("Training")

data2 = dataProcessing("FEHDataStudent_processed_new.xlsx")
df_standardised_validation = data2.Standardise("Validation")

data3 = dataProcessing("FEHDataStudent_processed_new.xlsx")
df_standardised_training_validation = data3.Standardise("Training+Validation")

data4  = dataProcessing("FEHDataStudent_processed_new.xlsx")
df_standardised_testing = data4.Standardise("Testing")

## seleting inputs nad targets of each dataset
inputs = df_standardised_training.loc[:,["AREA_logged", "BFIHOST", "FARL_cubed", "FPEXT", "LDP_logged", "SAAR", "RMED-1D"]]
targets = df_standardised_training.loc[:,["Index flood"]]

v_inputs = df_standardised_validation.loc[:,["AREA_logged", "BFIHOST", "FARL_cubed", "FPEXT", "LDP_logged", "SAAR",  "RMED-1D"]]
v_targets = df_standardised_validation.loc[:,["Index flood"]]

t_v_inputs = df_standardised_training_validation.loc[:,["AREA_logged", "BFIHOST", "FARL_cubed", "FPEXT", "LDP_logged", "SAAR",  "RMED-1D"]]
t_v_targets = df_standardised_training_validation.loc[:,["Index flood"]]

e_inputs =df_standardised_testing.loc[:,["AREA_logged", "BFIHOST", "FARL_cubed", "FPEXT", "LDP_logged", "SAAR",  "RMED-1D"]]
e_targets = df_standardised_testing.loc[:,["Index flood"]]

Main_menu = Menu() # calls menu to get user preferences
Main_menu.start()

ANN = NeuralNetwork(Main_menu.no_inputs,Main_menu.neurons_in_hidden) # initalises network
ANN.use_momentum = Main_menu.use_momentum 
epochs = Main_menu.epochs # sets parameters
batch_size = 64
learning_rate = Main_menu.learning_rate

ANN.train(inputs, targets, epochs, learning_rate,v_inputs,v_targets)
#ANN.train_mini_batch(inputs,targets,epochs,learning_rate,v_inputs,v_targets, batch_size)

ANN.SelectingNetwork(t_v_inputs,t_v_targets)

ANN.evaluation(e_inputs,e_targets) 

ANN.plot_mse(ANN.mse_values) 
ANN.plot_rmse(ANN.rmse_values)

ANN.saveNetwork() # asks user if they want to save 
