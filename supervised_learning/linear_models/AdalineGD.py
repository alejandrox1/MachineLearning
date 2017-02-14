import numpy as np

class AdalineGD(object):
    """ADAptive LInear NEuron classifier via gradient descent
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
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

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
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            self.cost_.append(self._update_weights(X,y))
        return self
   
    def _update_weights(self, X, y):
        """Apply ADALINE learning rule to update weights"""
        output = self.activation(X)
        errors = (y - output)
        self.w_[1:] += self.eta * X.T.dot(errors)
        self.w_[0] += self.eta * errors.sum()
        cost = 0.5 * (errors**2).sum()
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
