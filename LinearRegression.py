import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    # constructor called
    def __init__(self, LR=0.01, itr=1000):
        self.LR = LR
        self.itr = itr
        self.initial_m = 0
        self.initial_b = 0
        self.m = 0
        self.b = 0

    # scalling the data between -1 and 1 to make calucations eaiser
    def scaleData(self,X):
        j = 0
        arr = np.zeros(X.shape)
        for i in X.columns:
            mean = 0
            temp = X[i]
            mean = np.mean(temp)
            temp =  (temp - mean) / np.std(temp)
            arr[:,j] = temp
            j += 1
        return arr

    # funtion to calculate the loss of the line
    def calculateLoss(self,b, m , X, Y):
        loss = 0
        for i in range(0, len(X)):
            x = X[i]
            y = Y[i]
            loss += ((m * x + b) - y)**2

        return loss / (float)(len(X))


    # performing gradient descent for finding the best local minima
    # or the best m and b values for the best fitted line
    def performGD(self, b, m, X, Y):
        # initializing temp values for b and m
        b_temp, m_temp = 0, 0
        # length of the data
        N = float(len(X))
        # to calculate the partial derivative and finding the new values of b and m
        for i in range(0, len(X)):
            x = X[i]
            y = Y[i]
            # partial derivative formula
            m_temp += (-2/N) * x * (y - (m*x + b))
            b_temp += (-2/N) * (y - (m*x + b))
        
        newB = b - (self.LR * b_temp)
        newM = m - (self.LR * m_temp)

        return [newB, newM]
        

    def gradiantDescent(self,X, Y):
        b, m = self.initial_b, self.initial_m

        for i in range(self.itr):
            # update the values of the b and m using the gradiant descent method
            # to decrese the loss
            b, m = self.performGD(b, m, X, Y)
        
        return [b,m]


    # main function to fit the data 
    def fit(self, X, Y):

        print('Scalling the Data')
        X = self.scaleData(X)
        Y = self.scaleData(Y)
        print('Scalling Done')
        print('Starting with initial b = {}, m = {}, loss = {}'.format(
                                                                self.initial_b, self.initial_m, 
                                                                self.calculateLoss(self.initial_b,self.initial_m, X, Y)))

        [b, m] = self.gradiantDescent(X, Y)
        self.b = b
        self.m = m
        print('After {} iterations, b = {}, m = {}, loss = {}'.format(self.itr, b, m , self.calculateLoss(b, m, X, Y)))

        # plotting the best fitted line
        max_x = np.max(X) + 1
        min_x = np.min(X) - 1

        # generating a straight line with 100 points
        x = np.linspace(min_x, max_x, 100)
        # getting the line
        y = m * x + b

        #making a figure
        fig = plt.figure()
        # adding axes to the figure
        ax = fig.add_axes([0,0,1,1])
        # plotting line
        ax.plot(x, y, color='green', label = 'LR Line')
        # plotting data
        ax.scatter(X, Y, color='red', label='dataPoints')
        # adding title
        plt.title('Linear Regression')
        # displaying the lables
        ax.legend(loc='upper left')
        # showing the figure
        plt.show()

    def predict(self,test):
        predictions = []
        test = self.scaleData(test)
        for i in range(0,len(test)):
            y = (self.m * test[i]) + self.b
            predictions.append(y)
        
        return predictions
    
    def score(self, pred, y_test):
        y_test = self.scaleData(y_test)
        mean = np.mean(y_test)
        actual = np.sum((y_test - mean)**2)
        estimated = np.sum((pred - mean)**2)
        rsq = 1 - (estimated/actual)
        
        print(rsq)