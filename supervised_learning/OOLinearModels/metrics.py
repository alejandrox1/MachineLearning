import numpy as np

def count_error(y, output):
    error = ( y - output )
    errors = int(error!=0.0)
    return errors

def squared_error(y, output):
    return (y - output)**2

def mean_squared_error(y, output):
    return np.mean(squared_error(y, output))

def sum_squared_error(y, output):
    errors = y - output
    cost = 0.5 * (errors**2).sum()
    return cost

def binary_crossentropy(y, output):
    cost = -y.dot(np.log(output)) - (1-y).dot(np.log(1-output))
    return cost
