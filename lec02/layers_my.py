import numpy as np

class Linear:
    def __init__(self, input_size, output_size, optimize = None, reg = None, reg_lambda = None, silent = True):
        '''
        Creates weights and biases for linear layer.
        Dimention of inputs is *input_size*, of output: *output_size*.
        '''
        #### YOUR CODE HERE
        #### Create weights, initialize them with samples from N(0, 0.1).
        self.W = np.random.randn(input_size, output_size)*0.01
        self.b = np.zeros(output_size)
        self.reg = reg
        self.reg_lambda = reg_lambda
        
        self.silent = silent
        
        self.optimize = optimize
        self.iter_made = 0
        self.deltaW = 0
        self.deltab = 0
        
        if self.optimize == "Adam":
            self.mW = np.zeros([input_size, output_size])
            self.mb = np.zeros(output_size)
            self.gW = np.zeros([input_size, output_size])
            self.gb = np.zeros(output_size)

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        if self.silent == False:
            print('############ Linear start ############\n')
            print('X', X, '\n')
            print('y', X.dot(self.W)+self.b, '\n')
            print('W', self.W, '\n')
            print('############ Linear end ############')
        
        self.X = X
        return X.dot(self.W)+self.b

    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        self.dLdW = self.X.T.dot(dLdy)
        self.dLdb = dLdy.sum(0)
        self.dLdx = dLdy.dot(self.W.T)
        return self.dLdx

    def step(self, params):
        '''
        params:
        - No optimize - [learning_rate] - equal to NAG [learning_rate, 0]
        - Simple NAG - [learning_rate, NAG_rate]
        - Adam - [learning_rate, gamma1, gamma2, epsilon]
        '''
        
        '''
        1. Apply gradient dLdw to network:
        w <- w - learning_rate*dLdw
        '''
        
        if self.silent == False:
            print('############ Before step back ############')
            print("W", self.W, '\n')
            print("dLdW", self.dLdW, '\n')
            if self.optimize == "Adam":
                print("mW", self.mW, '\n')
                print("gW", self.gW, '\n')
            
        
        self.iter_made += 1
        
        # Regularization stuff
        
        if self.reg == "L1":
            self.W = self.W - self.reg_lambda * np.sign(self.W)
            self.b = self.b - self.reg_lambda * np.sign(self.b)
        elif self.reg == "L2":
            self.W = self.W - self.reg_lambda * self.W
            self.b = self.b - self.reg_lambda * self.b
            
        # Optimization stuff
        
        if self.optimize == "NAG":
            self.deltaW = params[1] * self.deltaW + params[0] * self.dLdW
            self.deltab = params[1] * self.deltab + params[0] * self.dLdb
        elif self.optimize == "Adam":
            self.mW = params[1] * self.mW + (1 - params[1]) * self.dLdW
            self.mb = params[1] * self.mb + (1 - params[1]) *  self.dLdb            
            self.gW = params[2] * self.gW + (1 - params[2]) * (self.dLdW ** 2)
            self.gb = params[2] * self.gb + (1 - params[2]) * (self.dLdb ** 2)
            
            mW_n = self.mW / (1 - (params[1] ** self.iter_made))
            mb_n = self.mb / (1 - (params[1] ** self.iter_made))
            gW_n = self.gW / (1 - (params[2] ** self.iter_made))
            gb_n = self.gb / (1 - (params[2] ** self.iter_made))
            
            self.deltaW = params[0] * mW_n / (((gW_n) ** (1/2)) + params[3])
            self.deltab = params[0] * mb_n / (((gb_n) ** (1/2)) + params[3])
        else:
            self.deltaW = params[0] * self.dLdW
            self.deltab = params[0] * self.dLdb
            
                        
        self.W = self.W - self.deltaW
        self.b = self.b - self.deltab
        
        if self.silent == False:
            print('############ After step back ############')
            print("W", self.W, '\n')
            print("dLdW", self.dLdW, '\n')
            if self.optimize == "Adam":
                print("mW", self.mW, '\n')
                print("gW", self.gW, '\n')
        
class Sigmoid:
    def __init__(self, silent=True):
        self.silent = silent
        pass
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        
        if self.silent == False:
            print('############ Sigmoid start ############\n')
            print('X', X, '\n')
            print('y', 1./(1+np.exp(-X)), '\n')
            print('############ Sigmoid end ############')
        
        self.s = 1./(1+np.exp(-X))
        return self.s
    
    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        return self.s*(1-self.s)*dLdy
    
    def step(self, learning_rate):
        pass

class NLLLoss:
    def __init__(self):
        '''
        Applies Softmax operation to inputs and computes NLL loss
        '''
        pass
    
    def forward(self, X, y):
        '''
        Passes objects through this layer.
        X is np.array of size (N, C), where C is the number of classes
        y is np.array of size (N), contains correct labels
        '''
        self.p = np.exp(X - np.max(X))
        self.p /= self.p.sum(1, keepdims=True)
        self.y = np.zeros((X.shape[0], X.shape[1]))
        self.y[np.arange(X.shape[0]), y] =1
        return -(np.log(self.p)*self.y).sum(1).mean(0)
    
    def backward(self):
        '''
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        '''
        return (self.p - self.y) / self.y.shape[0]


class MSE_Loss:
    def __init__(self):
        '''
        Applies Softmax operation to inputs and computes MSE loss
        '''
        pass
    
    def forward(self, X, y):
        '''
        Passes objects through this layer.
        X is np.array of size (N, C), where C is the number of classes
        y is np.array of size (N), contains correct labels
        '''
        
        self.p = np.exp(X - np.max(X))
        self.p /= self.p.sum(1, keepdims=True)
        self.y = np.zeros((X.shape[0], X.shape[1]))
        self.y[np.arange(X.shape[0]), y] = 1
        return np.sum((self.p - self.y) ** 2) / self.y.shape[0]
    
    def backward(self):
        '''
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        '''
        return 2. * self.p * (1. - self.p) * (self.p - self.y) / self.y.shape[0]
    
    
class NeuralNetwork:
    def __init__(self, modules):
        '''
        Constructs network with *modules* as its layers
        '''
        self.modules = modules
    
    def forward(self, X):
        y = X
        for i in range(len(self.modules)):
            y = self.modules[i].forward(y)
        return y
    
    def backward(self, dLdy):
        '''
        dLdy here is a gradient from loss function
        '''
        for i in range(len(self.modules))[::-1]:
            dLdy = self.modules[i].backward(dLdy)
    
    def step(self, learning_rate):
        for i in range(len(self.modules)):
            self.modules[i].step(learning_rate)