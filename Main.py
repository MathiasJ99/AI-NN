

class Menu:
    def __init__(self):
        self.epochs = 0
        self.no_inputs = 0 #req
        self.batch_size =0 
        self.learning_rate =0
        self.neurons_in_hidden = 0 #req
        
        self.weights_hidden_layer =None
        self.biases_hidden_layer =None

        self.weights_output_layer =None
        self.biases_hidden_layer =None
        self.RMSE_value  =0


    
    def start(self):# asks users question to get custom parametersS
        self.use_momentum = input("would you like to use momentum: (y/n)")
        self.epochs = int(input("how many epochs would you like to train the data for? "))
        #self.batch_size = int(input("what batch size would you like to choose? ")) ## implement if have time
        self.learning_rate = float(input("what would you like the learning rate to be? "))
        self.no_inputs = int(input("how mnay inputs would you like?: "))
        self.neurons_in_hidden = int(input("how many neurons in the hidden layer would you like there to be?:"))
            
          

