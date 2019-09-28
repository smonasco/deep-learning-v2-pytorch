import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # We're using the sigmoid function as the activation function for the hidden layer
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch. A 1D row array

        '''
        ### Forward pass ###
        # The inputs to the hidden layer are the features times their weights
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        # The outputs of the hidden layer are the inputs put through the activation funciton
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # signals into final output layer
        # The inputs of the output node are the hidden outputs times their weights
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        # The output of the output node is the inputs passed through the activation function of the output node
        # but we chose the output activation function to be f(x) = x, so it's just the inputs
        final_output = final_inputs 
        
        return final_output, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass (a scalar since we only have one output node)
            y: target (i.e. label) batch (also a scalar for the same reason as above)
            delta_weights_i_h: change in weights from input to hidden layers (a 3 X 2 matrix [inputs X hidden nodes])
            delta_weights_h_o: change in weights from hidden to output layers (a 2 X 1 matrix/column vector [hidden nodes X output node])

        '''
        ### Backward pass ###
        # Output layer error is the difference between desired target and actual output.
        error = y - final_outputs
        
        # error * derivative of activation function of the output node. Said derivative is 1 so leaving out for simplicity.
        output_error_term = error
        
        # A hidden node's contribution to the error is proportional to its weight going into the output
        # Here we consider the hidden_error to be the weights to output times the output error term
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        
        # The hidden_error_term is the hidden_error * the derivative of the activation function of the hidden layer
        # The hidden_error is a row vector and the derivative of sigmoid is f(x) * (1 - f(x))
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Weight step (input to hidden)
        # This step is equal to the learning rate times the hidden_error_term times the features
        # However, due to the distributive property and the fact that the learning rate remains constant
        # we can skip the multiplication for later (during the update_weights step)
        #
        # Since our delta weights are column vectors we'll flip one of our vectors to get the right shape
        delta_weights_i_h += hidden_error_term * X[:, None]
        
        # Weight step (hidden to output)
        # This step is equal to the learning rate times the output_error_term times the hidden_outputs.
        # Similarly, we can skip the learning rate and similarly we need to flip a vector to get the right shape
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # just average out the weights times the learning rate we skipped before
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        return self.forward_pass_train(features)[0]

#########################################################
# Set your hyperparameters here
##########################################################
iterations = 4000
learning_rate = 0.5
hidden_nodes = 7
output_nodes = 1
