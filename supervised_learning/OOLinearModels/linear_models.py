import numpy as np
from base import BaseEstimator
from metrics import count_error, sum_squared_error, mean_squared_error, binary_crossentropy

class BasicClassifier(BaseEstimator):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.cost_func = None
        self.w_initialized = False

    def init_cost(self):
        raise NotImplementedError()

    def fit(self, X, y):
        self._setup_input(X, y)
        self.init_cost()

        self._initialize_weights(X.shape[1])
        self.cost_ = []
        X = self._add_intercept(X)       
 
        for _ in range(self.n_iter):
            self._fit(X, y)
        return self

    def _update_weights(self, X, y):
        output = self.activation(X)
        errors = (y - output)
        self.w_ += self.eta * X.T.dot(errors)
        cost = self.cost_func(y, output)   
        return cost
	
    @staticmethod
    def _add_intercept(X):
        b = np.ones([X.shape[0],1])
        return np.concatenate([b,X], axis=1)

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1+m)
        self.w_initialized = True

    def net_input(self,X):
        if X.shape[1]==self.w_.shape[0]:        # fit
            return np.dot(X, self.w_)
        elif X.shape[1]==self.w_.shape[0]-1:    # predict
            return np.dot(X, self.w_[1:]) + self.w_[0]

    def _fit(self, X, y):
        raise NotImplementedError()

    def activation(self,X):
        raise NotImplementedError()




class Perceptron(BasicClassifier):
    """Perceptron Classifier"""
    def init_cost(self):
        self.cost_func = count_error

    def _fit(self, X, y):
        errors = 0
        for xi, target in zip(X,y):
            errors += self._update_weights(xi,target)
        self.cost_.append(errors)

    def activation(self,X):
        return np.where(self.net_input(X)>=0.0, 1,-1)

    def _predict(self, X=None):
        return np.where(self.activation(X)>=0.0, 1,-1)

class AdalineGD(BasicClassifier):
    """ADAptive LInear NEuron classifier with gradient descent optimizer"""
    def init_cost(self):
        self.cost_func = sum_squared_error

    def _fit(self, X, y):
        self.cost_.append(self._update_weights(X,y))

    def activation(self,X):
        return self.net_input(X)

    def _predict(self, X=None):
        return np.where(self.activation(X)>=0.0, 1,-1)

class LogisticRegressionGD(BasicClassifier):
    """Binary logistic regression with gradient descent optimizer"""
    def init_cost(self):
        self.cost_func = binary_crossentropy 

    def _fit(self, X, y):
        self.cost_.append(self._update_weights(X,y))

    def activation(self,X):
        z = self.net_input(X)
        return self.sigmoid(z)

    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

    def _predict(self, X=None):
        return np.where(self.activation(X)>=0.5, 1,0)

class LinearRegressionGD(BasicClassifier):
    """Linear regression with gradient descent optimizer"""
    def init_cost(self):
        self.cost_func = mean_squared_error

    def _fit(self, X, y):
        self.cost_.append(self._update_weights(X,y))

    def activation(self,X):
        return self.net_input(X)

    def _predict(self,X=None):
        return np.where(self.activation(X)>=0.0, 1,-1)
