import numpy as np
import scipy.linalg as ln
class NeuralNetwork:
    def __init__(self):
        '''parameters of the network'''
        self.layers = 3
        self.input = 2
        self.hidden = 3
        self.output = 1

        self.W1 = np.random.rand(self.input+1, self.hidden)  # 3x3
        self.W2 = np.random.rand(self.hidden+1, self.output)  # 4x1

        '''setting universal constants'''
        self.LAMBDA = 0.0001

    def forward(self, x):
        '''
        :param x: numpy.multiarray.ndarray of 2d. Its dimensions will be (# of samples x # of features)
        :return: a3, or yhat, or guessed value of y
        '''

        m = x.shape[0] # m = number of samples
        x = np.insert(x, 0, 1, axis=1)
        self.a1 = x

        self.z2 = np.dot(self.a1, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.a2 = np.insert(self.a2, 0, 1, axis=1)

        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)

        return self.a3

    def costFunction(self, x, y):
        '''
        :param x: the inputs, in this case 5x2
        :param y: the outputs, in this case 5x1
        :return: the cost of the function given a3 guess and y
        '''
        self.forward(x)
        m = y.shape[0]
        cost = -y * np.log(self.a3) - (1-y) * np.log(1-self.a3)
        return np.sum(cost)/m

    def costFunctionReg(self, x, y):
        m = y.shape[0]

        cost = self.costFunction(x,y) + self.LAMBDA*(np.sum(self.W1[1:,:]**2) + np.sum(self.W2[1:,:]**2))/(2*m)
        return cost
    def costFunctionPrime(self, x, y):
        '''
        :param x: the inputs, in this case 5x2
        :param y: the outputs, in this case 5x1
        :return: the dJ/dW1 and dJ/dW2
        '''
        self.forward(x)

        m = y.shape[0]

        d3 = self.a3 - y
        d2 = np.dot(d3, (self.W2[1:]).T) * self.sigmoidPrime(self.z2)
        dJdW2 = np.dot(self.a2.T, d3)/m
        dJdW1 = np.dot(self.a1.T, d2)/m

        return [dJdW1, dJdW2]

    def costFunctionPrimeReg(self, x, y):
        dJdW1, dJdW2 = self.costFunctionPrime(x,y)

        m = y.shape[0]

        regularize2 = self.LAMBDA*(self.W2[1:,:])/m
        regularize1 = self.LAMBDA*(self.W1[1:,:])/m
        dJdW1[1:,:] = dJdW1[1:,:] + regularize1
        dJdW2[1:,:] = dJdW2[1:,:] + regularize2

        return [dJdW1, dJdW2]

    def numericalGradient(self, x, y):
        W = self.unrollWeights(self.W1, self.W2)
        numGrad = np.zeros(W.shape)
        perturb = np.zeros(W.shape)
        epsilon = 0.0001  # Variable for testing
        # Computing numGrad
        for i,num in enumerate(W):
            perturb[i] = epsilon
            self.rollWeights(W+perturb)
            cost1 = self.costFunction(x,y)
            self.rollWeights(W-perturb)
            cost2 = self.costFunction(x,y)

            numGrad[i] = (cost1-cost2)/(2*epsilon)
            perturb[i] = 0
        self.rollWeights(W)
        return numGrad

    def numericalGradientReg(self, x, y):
        W = self.unrollWeights(self.W1, self.W2)
        numGrad = np.zeros(W.shape)
        perturb = np.zeros(W.shape)
        epsilon = 0.0001  # Variable for testing
        # Computing numGrad
        for i,num in enumerate(W):
            perturb[i] = epsilon
            self.rollWeights(W+perturb)
            cost1 = self.costFunctionReg(x,y)
            self.rollWeights(W-perturb)
            cost2 = self.costFunctionReg(x,y)

            numGrad[i] = (cost1-cost2)/(2*epsilon)
            perturb[i] = 0
        self.rollWeights(W)
        return numGrad

    def sigmoid(self, z):
        '''
        :param z: the matrix or scalar that will be performed
        :rtype: np.multiarray.ndarray
        '''
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        '''
        :param z: the matrix or scalar that will be performed
        :rtype: np.multiarray.ndarray
        '''
        return (1-self.sigmoid(z))*self.sigmoid(z)

    def unrollWeights(self, W1, W2):
        '''
        Unrolls the weights into a 1 dimensional matrix
        :return: flattened 1d ndarray
        '''
        temp1 = W1.ravel()
        temp2 = W2.ravel()

        return np.concatenate((temp1, temp2))

    def rollWeights(self, W):
        '''
        Rolls the weights into their respective W1 and W2
        :param W: the long unrolled version of the 2 W's
        :return: W1, W2
        '''
        intersect1 = (self.input+1)*self.hidden
        W1 = W[0:intersect1]
        W2 = W[intersect1:]
        W1 = W1.reshape((self.input+1, self.hidden))
        W2 = W2.reshape((self.hidden+1, self.output))
        self.W1 = W1
        self.W2 = W2
        return [W1,W2]

MATRIX = np.array([[3,5,0],
          [8,10,1],
          [6,12,1],
          [1,4,0],
          [5,9,1]])

x = MATRIX[:,0:-1]
y = MATRIX[:,-1].reshape((-1,1))
nn = NeuralNetwork()

'''Testing the cost function and costFunctionPrimes initial step'''
cost = nn.costFunction(x, y)
dJdW1, dJdW2 = nn.costFunctionPrime(x, y)
print(cost)
print((dJdW1, dJdW2))

'''Numerical Gradient Checking'''
unrolledW = nn.unrollWeights(dJdW1, dJdW2)
numGrad = nn.numericalGradient(x, y)
print("Actual Gradient : " + str(unrolledW))
print("Numerical Gradient : " + str(numGrad))
print("Norm difference ratio : " + str(ln.norm(unrolledW - numGrad)/ln.norm(unrolledW+numGrad)))

'''Regularization checking'''
print(nn.costFunctionPrimeReg(x,y))

'''Numerical Gradient Regularized Checking'''
dJdW1, dJdW2 = nn.costFunctionPrimeReg(x, y)

unrolledW = nn.unrollWeights(dJdW1, dJdW2)
numGrad = nn.numericalGradientReg(x, y)
print("Actual Gradient : " + str(unrolledW))
print("Numerical Gradient : " + str(numGrad))
print("W1 : " + str(nn.W1))
print("W2 : " + str(nn.W2))
print("Norm difference ratio : " + str(ln.norm(unrolledW - numGrad)/ln.norm(unrolledW+numGrad)))

'''Standard Gradient Descent'''
iter = 10000
alpha = 0.05
for i in range(iter):
    dJdW1, dJdW2 = nn.costFunctionPrimeReg(x, y)
    nn.W1 -= dJdW1 * alpha
    nn.W2 -= dJdW2 * alpha
    cost = nn.costFunctionReg(x, y)
    print(cost)

print(nn.a3)

'''Testing new data'''
NEW_X = np.array([[5,8],  # 1
          [2,4], # 0
          [2,8], # 0.5
          [1,3], # 0
          [6,12], # 1
          [80,70], # ~1
          [-2, -3], # ~0
          [5,5], # ~0.5
          [1,1], # ~0
          [6,7]]) # 1
ans = nn.forward(NEW_X)
print(ans)