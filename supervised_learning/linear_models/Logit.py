import numpy as np

class LogisticRegressionGD(object):
    """Logistic Regression classifier via gradient descent
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
        """Apply max. likelihood learning rule to update weights"""
        output = self.activation(X)
        errors = (y - output)
        self.w_[1:] += self.eta * X.T.dot(errors)
        self.w_[0] += self.eta * errors.sum()
        cost = -y.dot(np.log(output)) - (1-y).dot(np.log(1-output))
        return cost

    def net_input(self,X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self,X):
        """Compute Sigmoid Activation"""
        z = self.net_input(X)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return sigmoid

    def predict(self, X):
        """Return class label"""
        return np.where(self.activation(X)>=0.5, 1,0)
