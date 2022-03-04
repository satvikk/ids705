import numpy as np
class myNeuralNetwork(object):
    
    def __init__(self, n_in, n_layer1, n_layer2, n_out, learning_rate=0.01):
        '''__init__
        Class constructor: Initialize the parameters of the network including
        the learning rate, layer sizes, and each of the parameters
        of the model (weights, placeholders for activations, inputs, 
        deltas for gradients, and weight gradients). This method
        should also initialize the weights of your model randomly
            Input:
                n_in:          number of inputs
                n_layer1:      number of nodes in layer 1
                n_layer2:      number of nodes in layer 2
                n_out:         number of output nodes
                learning_rate: learning rate for gradient descent
            Output:
                none
        '''
        self.lr  = learning_rate
        n_params = n_in*n_layer1 + n_layer1*n_layer2 +n_layer2*n_out
        self.w1 = np.random.randn(n_in, n_layer1) / n_params
        self.w2 = np.random.randn(n_layer1, n_layer2) / n_params
        self.w3 = np.random.randn(n_layer2, n_out) / n_params
        
        self.w1_grad = np.zeros_like(self.w1)
        self.w2_grad = np.zeros_like(self.w2)
        self.w3_grad = np.zeros_like(self.w3)
        
        self.a1 = np.zeros(n_layer1)
        self.a2= np.zeros(n_layer2)
        self.a1_grad = np.zeros_like(self.a1)
        self.a2_grad = np.zeros_like(self.a2)
        self.a3_grad = np.zeros(n_out)

    def forward_propagation(self, x):
        '''forward_propagation
        Takes a vector of your input data (one sample) and feeds
        it forward through the neural network, calculating activations and
        layer node values along the way.
            Input:
                x: a vector of data representing 1 sample [n_in x 1]
            Output:
                y_hat: a vector (or scaler of predictions) [n_out x 1]
                (typically n_out will be 1 for binary classification)
        '''
        x = self.sigmoid(self.w1.T @ x)
        x = self.sigmoid(self.w2.T @ x)
        x = self.sigmoid(self.w3.T @ x)
        return x
    
    def compute_loss(self, X, y):
        '''compute_loss
        Computes the current loss/cost function of the neural network
        based on the weights and the data input into this function.
        To do so, it runs the X data through the network to generate
        predictions, then compares it to the target variable y using
        the cost/loss function
            Input:
                X: A matrix of N samples of data [N x n_in]
                y: Target variable [N x 1]
            Output:
                loss: a scalar measure of loss/cost
        '''
        y_hat = np.array([self.forward_propagation(X[i,:]) for i in range(X.shape[0]) ])
        return ((y-y_hat)**2).sum()*0.5
        #return (- y*np.log(y_hat) - (1-y)*np.log(1-y_hat)).sum()

    def backpropagate(self, x, y):
        '''backpropagate
        Backpropagate the error from one sample determining the gradients
        with respect to each of the weights in the network. The steps for
        this algorithm are:
            1. Run a forward pass of the model to get the activations 
               Corresponding to x and get the loss functionof the model 
               predictions compared to the target variable y
            2. Compute the deltas (see lecture notes) and values of the
               gradient with respect to each weight in each layer moving
               backwards through the network
    
            Input:
                x: A vector of 1 samples of data [n_in x 1]
                y: Target variable [scalar]
            Output:
                loss: a scalar measure of th loss/cost associated with x,y
                      and the current model weights
        '''
        x = x.reshape(-1,1)
        self.a1 = self.sigmoid(self.w1.T @ x).reshape(-1,1)
        self.a2 = self.sigmoid(self.w2.T @ self.a1).reshape(-1,1)
        self.y_hat = self.sigmoid(self.w3.T @ self.a2)
        ##L = - y*np.log(self.y_hat) - (1-y)*np.log(1-self.y_hat)
        L = ((y-self.y_hat)**2)*0.5
        self.a3_grad = self.y_hat - y
        ##self.a3_grad = (1-y)/(1-self.y_hat) - y/self.y_hat
        self.a2_grad = self.a3_grad * self.sigmoid_derivative(self.w3.T @ self.a2) * self.w3
        self.w3_grad = self.a3_grad * self.sigmoid_derivative(self.w3.T @ self.a2) * self.a2
        self.w2_grad = ( self.a1 @ self.a2_grad.T ) * self.sigmoid_derivative(self.w2 * self.a1)
        self.a1_grad = ( self.sigmoid_derivative(self.w2.T @ self.a1).reshape(1,-1) * self.w2 ) @ self.a2_grad
        self.w1_grad = x @ self.a1_grad.T * self.sigmoid_derivative(self.w1 * x)
        return L

    def stochastic_gradient_descent_step(self):
        '''stochastic_gradient_descent_step [OPTIONAL - you may also do this
        directly in backpropagate]
        Using the gradient values computed by backpropagate, update each
        weight value of the model according to the familiar stochastic
        gradient descent update equation.
        
        Input: none
        Output: none
        '''
        #self.w1 -= self.w1_grad*self.lr
        #self.w2 -= self.w2_grad*self.lr
        self.w3 -= self.w3_grad*self.lr
    
    def fit(self, X, y, max_epochs=1, val_X=None, val_y=None, record_rmsgrads=False):
        '''fit
            Input:
                X: A matrix of N samples of data [N x n_in]
                y: Target variable [N x 1]
            Output:
                training_loss:   Vector of training loss values at the end of each epoch
                validation_loss: Vector of validation loss values at the end of each epoch
                                 [optional output if get_validation_loss==True]
        '''
        training_loss = []
        val_loss = []
        if record_rmsgrads:
            rms_grads = []
        for e in range(max_epochs):
            for i in range(X.shape[0]):
                self.backpropagate(X[i,:], y[i])
                self.stochastic_gradient_descent_step()
                pass
            training_loss.append(self.compute_loss(X, y))
            if val_X is not None and val_y is not None:
                val_loss.append(self.compute_loss(val_X, val_y))
                pass
            if record_rmsgrads:
                rms_grads.append(self.rmsgrad())
            pass
        outdict =  {'training_loss': training_loss, 'val_loss': val_loss}
        if record_rmsgrads:
            outdict['rms_grads'] = rms_grads
        return outdict


    def predict_proba(self, X):
        '''predict_proba
        Compute the output of the neural network for each sample in X, with the last layer's
        sigmoid activation providing an estimate of the target output between 0 and 1
            Input:
                X: A matrix of N samples of data [N x n_in]
            Output:
                y_hat: A vector of class predictions between 0 and 1 [N x 1]
        '''
        return np.array([self.forward_propagation(X[i,:]) for i in range(X.shape[0])])

    def predict(self, X, decision_thresh=0.5):
        '''predict
        Compute the output of the neural network prediction for 
        each sample in X, with the last layer's sigmoid activation 
        providing an estimate of the target output between 0 and 1, 
        then thresholding that prediction based on decision_thresh
        to produce a binary class prediction
            Input:
                X: A matrix of N samples of data [N x n_in]
                decision_threshold: threshold for the class confidence score
                                    of predict_proba for binarizing the output
            Output:
                y_hat: A vector of class predictions of either 0 or 1 [N x 1]
        '''
        y_hat = self.predict_proba(X)
        return (y_hat > decision_thresh).astype(float)

    def sigmoid(self, X):
        '''sigmoid
        Compute the sigmoid function for each value in matrix X
            Input:
                X: A matrix of any size [m x n]
            Output:
                X_sigmoid: A matrix [m x n] where each entry corresponds to the
                           entry of X after applying the sigmoid function
        '''
        return 1 / (1 + np.exp(-X) )
    
    def sigmoid_derivative(self, X):
        '''sigmoid_derivative
        Compute the sigmoid derivative function for each value in matrix X
            Input:
                X: A matrix of any size [m x n]
            Output:
                X_sigmoid: A matrix [m x n] where each entry corresponds to the
                           entry of X after applying the sigmoid derivative function
        '''
        return self.sigmoid(X) * self.sigmoid(-X)

    def rmsgrad(self):
        return (self.w1_grad**2).mean() + (self.w2_grad**2).mean() + (self.w3_grad**2).mean()
