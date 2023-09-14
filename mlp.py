import numpy as np

class MLP:
    def __init__(self, num_inputs = 3, num_hidden = [3,5], num_outputs = 2):
        """
        Arguments: 
        - number of inputs (int)
        - list of number of hidden layers
        - number of outputs (int)
        """

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        print("\nLayer Dimensions: {}".format(layers))

        #initalize with random
        weights = []
        for i in range(len(layers)-1):
            #weight matrix between layers
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights
        
        activations = []
        for i in range(len(layers)):
            #amount of zeros equal to num of neurons in each layer
            a = np.zeros(layers[i])
            #list of arrays representing activations for given layer
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1): #weights are number of layers - 1. (no open ended weights)
            # neurons in layer, and number in subsequent layer
            d = np.zeros((layers[i], layers[i+1]))
            #list of arrays representing activations for given layer
            derivatives.append(d)
        self.derivatives = derivatives
        
    def forward_propogation(self, inputs):
        activations = inputs
        self.activations[0] = inputs

        #loop through all layers in network
        for i, w in enumerate(self.weights):
            #calculate the net input for given layer
            net_inputs = np.dot(activations, w)

            #calculate the activation for given layer
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations

        #return output activation layer
        return(activations)
    
    def back_prop(self, error, verbose=False):
        
        # dE/dW_i     = (y - a_i+1) * sig'(h_i+1)) * a_i
        # sig'(h_i+1) = sig(h_i+1) * 1-sig(h_i+1)
        # sig(h_i+1)  = a_i+1

        #dE/dW_[i-1]  = (y - a_[i+1]) * sig'(h_[i+1]) * W_i * sig'(h_i) * a_[i-1]

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            #reshape to 2d array with single row array([data,data]) --> array([[data, data]])
            delta_reshape = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]
            #reshape array([data,data]) --> array([[data], [data]])
            current_activations_reshape = current_activations.reshape(current_activations.shape[0], -1) 

            # dot product for each layer of derivatives
            self.derivatives[i] = np.dot(current_activations_reshape, delta_reshape)
            
            #new error equal to (y - a_[i+1]) * sig'(h_[i+1]) * W_i
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("derivatives for W{}:{}".format(i, self.derivatives[i]))

        return(error)

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            #retrieve w and d for respctive layer
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            self.weights[i] += (derivatives*learning_rate)
    
    def train(self, inputs, targets, epochs, learning_rate):
        
        for i in range(epochs):
            sum_error = 0 
            for j,input in enumerate(inputs):
                
                target = targets[j]
                
                #perform forward prop
                output = self.forward_propogation(input)

                #calculate error
                error = target - output

                #back prop
                self.back_prop(error)

                #apply grad desc
                self.gradient_descent(learning_rate)

                #sum error for each epoch
                sum_error += self._mse(target, output)
            
            #report error
            print("Error : {} ({}%)".format((sum_error/len(inputs)), int((i/epochs)*100)))

    def _sigmoid(self, x):
        return( 1.0 / (1+np.exp(-x)))
    def _sigmoid_derivative(self, x):
        return (x * (1.0-x))
    def _mse(self, target, output):
        return np.average((target-output)**2)

#----------------------------------------------------------------------------------------#

if __name__ == "__main__":

    #create dataset to train network in sum operation
    items = np.array([[np.random.rand() / 2 for _ in range(2)] for _ in range(1000)]) # 1000 row, 2 col arr with random(0,0.5)
    targets = np.array([[i[0]+i[1]] for i in items]) # target array equals sum of i[0] and i[1] for each row

    #create MLP
    mlp = MLP(2,[8],1) #default, random weights

    mlp.train(items, targets, 100, 0.5)

    #create dummy data
    inputs = np.array([0.3, 0.5])
    target = np.array([0.8])

    output = mlp.forward_propogation(inputs)
    print("\nOur network believes that {} + {} is equal to {}\n".format(inputs[0], inputs[1], output))



    
    '''
    #perform forward prop
    outputs = mlp.forward_propogation(inputs)

    #calculate error
    error = target - outputs

    #back prop
    mlp.back_prop(error)

    #apply grad desc
    mlp.gradient_descent(learning_rate=0.01)
    '''


