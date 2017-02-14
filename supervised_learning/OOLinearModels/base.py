import numpy as np

class BaseEstimator(object):
    X = None
    y = None

    def _setup_input(self, X, y):
        """Ensure input to classifier is in the expected format.
    
        Ensure X and y are stored as numpy ndarrays.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors.
        y : array-like, shape = [n_samples]
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.shape[0] == 0:
            raise ValueError("Number of samples must be > 0")
        
        if X.ndim == 1:
            self.n_samples_, self.n_features_ = 1, X.shape[0]
        else:
            self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        self.X = X

        if y is None:
            raise ValueError("Missing class labels")

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if y.shape[0] == 0:
            raise ValueError("Number of labels must be > 0")

        self.y = y

    def fit(self, X, y):
        self._setup_input(X,y)

    def predict(self, X=None):
        if self.X is not None:
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            return self._predict(X)
        else:
            raise ValueError("You must call 'fit' method before 'predict'")

    def _predict(self, X=None):
        raise NotImplementedError()
