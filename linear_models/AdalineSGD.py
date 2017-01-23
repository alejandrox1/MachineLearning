import numpy as np
from numpy.random import seed

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier via stochastic gradient descent
    by Sebastian Raschka 
    (https://github.com/rasbt/python-machine-learning-book)

    Parameters:
    -----------
    eta : float
        Learning rate (or bias) [0,1]
    n_iter: int
        Iterations.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclasifications in every epoch.
    shuffle : bool (default: True)
        If True, shuffles training data set every epoch to prevent cycles.
    random_state: int (default: None)
        Set random state for shuffling and initializing the weights.
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """Fit training data

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors.
        y : array-like, shape = [n_samples]

        Returns
        -------
        slef : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def _initialize_weights(self, m):
        """Initialize weights to zero"""
        self.w_ = np.zeros(1+m)
        self.w_initialized = True

    def _shuffle(self, X, y):
        """Shuffle training set"""
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _update_weights(self, xi, target):
        """Apply ADALINE learning rule to update weights"""
        output = self.activation(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.T.dot(error) 
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
 
    def net_input(self,X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self,X):
        """Compute Linear Activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label"""
        return np.where(self.activation(X)>=0.0, 1,-1)

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(xi, target):
                self._update_weights(xi,target)
        else:
            self._update_weights(X, y)
        return self

