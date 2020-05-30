import numpy as np

class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.u = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        print(self.u)
        print(self.std)
        self.Xe = self.__featureScaling__(X)
        return self.Xe
    
    def __featureScaling__(self, X):
       return (X - self.u) / self.std
   
    def computeBeta(self):
        Xe = self.Xe
        N = Xe.shape[0]
        Xe = np.c_[np.ones(N), Xe ]
        y = self.y
        self.theta =  np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)
        self.theta = self.theta.flatten()
        return self.theta
        
    def predictWithNormalEqua(self, X):
        Xe = self.__extendArray__(X)
        return (Xe.dot(self.theta))
    
    def predictWithGradient(self,X):
        Xe = self.__extendArray__(X)
        Xe = Xe.flatten()
        return np.sum(self.thetaGradient.flatten() * Xe, axis = 0)
    
    def __extendArray__(self,X):
        N = X.shape[0]
        X = self.__featureScaling__(X)
        return np.c_[np.ones(N), X]
    
    def getCost(self):
        N = self.X.shape[0]
        y = self.y.flatten()
        Xe = self.__featureScaling__(self.X)
        Xe = np.c_[np.ones(N), Xe ]
        j = np.dot(Xe, self.theta) - y
        self.cost = (j.T.dot(j)) / N
        return self.cost
    
    def computeGradient(self):
        Xe = self.__extendArray__(self.X)
        N = Xe.shape[0]
        cost = self.cost
        
        eta = .01
        iterations = 0
        theta = np.random.randn(Xe.shape[1],1)
        MSE = []
        while True:
            iterations +=1
            gradients = 2/N * Xe.T.dot((Xe.dot(theta) - self.y) )
            theta = theta - eta * gradients
            j = np.dot(Xe, theta) -  self.y
            J = (j.T.dot(j)) / N
            MSE.append(J[0][0])
            current = np.round(cost / (J[0]) *100, 2)
            if current == 99:
                break;
        self.thetaGradient = theta       
        return theta, MSE, iterations
