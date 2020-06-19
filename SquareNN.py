#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
X = np.array(([2, 9], [1, 5], [3, 6], [5, 10], [4, 2], [10, 10], [2, 6], [9, 5], [8, 8], [12,10], [4,7]), dtype=float)
y = np.array(([18], [5], [18], [50], [8], [100], [12], [45], [64], [120]), dtype=float)


# %%
X = X/np.max(X, axis = 0)
y = y/np.max(y, axis = 0)

# %%
X_train, X_test = np.split(X, [10])


# %%
class neural_network(object):
    def __init__(self):
        self.inputSize = 2
        self.hiddenSize = 3
        self.outputSize = 1

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def sigmoid(self, X):
        return 1/(1+np.exp(-X))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1)
        self.Z2 = self.sigmoid(self.Z1)
        self.Z3 = np.dot(self.Z2, self.W2)
        return self.sigmoid(self.Z3)

    def sigmoidPrime(self, s):
        return s*(1-s) 
           
    def backward(self, X, y, output):
        self.outpuEtError = y - output
        self.outputDelta = self.outpuEtError*self.sigmoidPrime(output)
        self.z2_error = self.outputDelta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.Z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.Z2.T.dot(self.outputDelta)

    def train(self, X, y):
        forward_output = self.forward(X)
        self.backward(X, y, forward_output)
    
    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(X_test))
        print("Output: \n" + str(self.forward(X_test)))


# %%
nn = neural_network()
count =[]
losses= []
for i in range(1000):
    print("Input: \n", str(X_train))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" + str(nn.forward(X_train)))
    loss = str(np.mean(np.square(y - nn.forward(X_train))))
    print("Loss: \n" + loss) # mean squared error
    print("\n")
    count.append(i)
    losses.append(np.round(float(loss), 6))
    plt.cla()
    plt.title("Loss over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(count, loss)
    plt.pause(.001)
    nn.train(X_train, y)

nn.saveWeights()
nn.predict()

# %%

