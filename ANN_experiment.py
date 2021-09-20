import numpy as np
class net:
    def __init__(self,size,batch_size,epochs,eta,training_data,test_data=None):
        self.training_data = training_data
        self.test_data = test_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.n_layers = len(size)
        self.size = size
        self.setup(self.size)
        self.process()
        
    def process(self):
        for i in range(self.epochs):
            self.cache_nabla_c_weights = [np.zeros((j,k)) for j,k in zip(self.size[1:],self.size[:-1])]
            self.cache_nabla_c_biases = [np.zeros((j,1)) for j in self.size[1:]]
            np.random.shuffle(self.training_data)
            '''The input below is set for tuples'''
            inp = self.training_data[0:self.batch_size]
            for m in range(self.batch_size):
                self.index = m+1
                self.stochastic_input(inp,test=False)
                zs = self.feedforward()
                self.SGD(inp,self.eta,zs)
                self.evaluate()
        if self.test_data:
            for i in range(len(self.test_data)):
                inp = self.test_data[i]
                self.stochastic_input(inp,test=True)
                self.feedforward()
                print('Input: ' + str(self.test_data[i]))
                '''Output adjusted for number guessing network:'''
                y = np.log(self.activations[-1]/(1-self.activations[-1]))
                print('Output: ' + str(y))
        
    def setup(self,size):
        self.weights = np.asarray([np.random.randn(j,k) for j,k in zip(size[1:],size[:-1])])
        self.biases = np.asarray([np.random.randn(j,1) for j in size[1:]])
        self.activations = np.asarray([np.zeros((j,1)) for j in size[:]])
              
    def stochastic_input(self,inp,test):
        if test == False:
            '''This might need to be changed to "inp[0][self.index-1]" depending on format of input (set of tuples or just two lists?)'''
            self.activations[0] = inp[self.index-1][0]
        if test == True:
            self.activations[0] = inp
            
    def feedforward(self):
        zs = np.array([np.zeros((j,1)) for j in self.size[1:]])
        for l in range(1,self.n_layers):
            z = np.dot(self.weights[l-1],self.activations[l-1]) + self.biases[l-1]
            zs[l-1] = z
            self.activations[l] = self.sigmoid(z)
        return zs
              
    def SGD(self,inp,eta,zs):
        nabla_c_weights, nabla_c_biases = self.backpropagate(inp,zs)
        for l in range(self.n_layers-1):
            self.cache_nabla_c_weights[l] += nabla_c_weights[l]
            self.cache_nabla_c_biases[l] += nabla_c_biases[l]
        if self.index == self.batch_size:
            for l in range(self.n_layers-1):
                self.weights[l] += -(eta/self.batch_size)*self.cache_nabla_c_weights[l]
                self.biases[l] += -(eta/self.batch_size)*self.cache_nabla_c_biases[l]
    
    def backpropagate(self,inp,zs):
        del_vec = self.del_vector(inp,zs)
        nabla_c_weights = [np.zeros((j,k)) for j,k in zip(self.size[1:],self.size[:-1])]
        nabla_c_biases = [np.zeros((j,1)) for j in self.size[1:]]
        for l in range(1,self.n_layers):
            nabla_c_weights[-l] = np.dot(del_vec[-l],self.activations[-(l+1)].T)
            nabla_c_biases[-l] = del_vec[-l]
        return nabla_c_weights, nabla_c_biases
    
    def del_vector(self,inp,zs):
        d = [np.zeros((j,1)) for j in self.size[1:]]
        '''The input below is set for tuples'''
        nabla_c_activation = self.cost_prime(self.activations[self.n_layers-1],inp[self.index-1][1])
        d[-1] = np.multiply(nabla_c_activation,self.sigmoid_prime(zs[-1]))
        for l in range(1,self.n_layers-1):
            d[-(l+1)] = np.multiply(np.dot(self.weights[-l].T,d[-l]),self.sigmoid_prime(zs[-(l+1)]))    
        return d
    
    def evaluate(self):
        return
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def sigmoid_prime(self,x):
        return np.exp(-x)/(1+np.exp(-x))**2
    def cost(self,x,y):
        return (1/2)*(x-y)**2
    def cost_prime(self,x,y):
        return x-y
