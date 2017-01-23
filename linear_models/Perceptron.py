import numpy as np

class Perceptron(object):
    """Perceptron Classifier.
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
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                errors += self._update_weights(xi, target)
            self.errors_.append(errors)
        return self
   
    def _update_weights(self, xi, target):
        """Apply ADALINE learning rule to update weights"""
        output = self.activation(xi)
        error = ( target - output )
        self.w_[1:] += self.eta * error * xi
        self.w_[0] += self.eta * error
        errors = int(error!=0.0)
        return errors
 
    def net_input(self,X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self,X):
        """Compute Linear Activation"""
        return np.where(self.net_input(X)>=0.0, 1,-1)

    def predict(self, X):
        """Return class label"""
        return np.where(self.net_input(X)>=0.0, 1,-1)
