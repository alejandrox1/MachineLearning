
# coding: utf-8

# * The last column of a Dataset always corresponds to the class label

# # Input

# In[9]:

from csv import reader

def load_csv(filename):
    with open(filename, 'r') as file:
        lines = reader(file)
        dataset = list(lines)
    dataset = [ row for row in dataset if row ]
    return dataset
    
def str_column_to_float(dataset, column):     
    """ Convert string numerical entries to floats. """
    for row in dataset:                                                     
        row[column] = float(row[column].strip())
                                         
def str_column_to_int(dataset, column):    
    """ Convert string label entries to ints. """
    class_values = [row[column] for row in dataset]                         
    unique = set(class_values)                                              
    lookup = dict()                                                         
    for i, value in enumerate(unique):                                      
        lookup[value] = i                                               
    for row in dataset:                                                     
        row[column] = lookup[row[column]]                               
    return lookup


# # Scaling
# 
# Scale of input and output to be equivalent.
# 
# Standardization assumes your data conforms to a normal distribution.
# Normalization is more sensitive to outliers.

# In[16]:

def dataset_minmax(dataset):
    minmax = []
    for i in range(len(dataset[0])): # for each feature
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append( [value_min, value_max] )
    return minmax

def normalize_dataset(dataset):
    """ Scaled value = value - min / max - min """
    minmax = dataset_minmax(dataset)
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i]-minmax[i][0]) / (minmax[i][1]-minmax[i][0])
    return dataset


# In[21]:

from math import sqrt

def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means

def column_stdevs(dataset):
    """ var = sum( value - mean )**2 / . """
    means = column_means(dataset)
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        valriance = [ (row[i]-means[i])**2.0 for row in dataset ]
        stdevs[i] = sum(valriance)
    stdevs = [ sqrt(x/float(len(dataset)-1)) for x in stdevs ]
    return stdevs, means

def standardize_dataset(dataset):
    stdevs, means = column_stdevs(dataset)
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]
    return dataset


# # Resampling Methods
# 
# If multiple algorithms are compared the same train test split should be used 
# for consistent comparison.
# 
# $k$-fold cross validation helps reduce noise of performance estimates.
# Here, the algorithm is trained and evaluated $k$ times and the performance is 
# summarized by taking the mean of the performance score.
# 
# Train on $k$-1 folds and evaluate on the $k$th one.
# Then repeat so that each of the $k$ groups is given an opportunity to be used 
# as a test set.
# 
# **A quick way to check if the fold sizes are representative is to calculate 
# summary statistics (i.e., mean and standard deviation) and see how much the 
# values differ from the statistics of the entire set.**
# 
# 

# In[24]:

import os
from time import time
from random import seed
from random import randrange

def train_test_split(dataset, split=0.60):
    seed( os.getpid() * time())
    train = []
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append( dataset_copy.pop(index) )
    return train, dataset_copy


# In[78]:

def cross_validation_split(dataset, n_folds=3):
    seed( os.getpid() * time())
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = len(dataset) // n_folds
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append( dataset_copy.pop(index) )
        dataset_split.append( fold )
    return dataset_split


# # Evaluation Metrics

# In[80]:

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))


# In[82]:

def confusion_matrix(actual, predicted):
    unique = list(set(actual))
    matrix = [[] for i in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for i in range(len(unique))]
        
    lookup = dict()
    for i, value in enumerate(unique): # assign id to each label
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[x][y] += 1
    return unique, matrix

def print_confusion_matrix(actual, predicted):
    unique, matrix = confusion_matrix(actual, predicted)
    print(' '*3, end='')
    for i in unique:
        print(i, end='  ')
    print()
    for i in range(len(matrix)):
        print(unique[i], matrix[i])


# In[84]:

def mae_metric(actual, predicted):    
    """ Mean Absolute Error"""
    sum_error = 0.0                                                         
    for i in range(len(actual)):                                            
        sum_error += abs(predicted[i] - actual[i])                      
    return sum_error / float(len(actual))

def rmse_metric(actual, predicted):                                             
    sum_error = 0.0                                                         
    for i in range(len(actual)):                                            
        prediction_error = predicted[i] - actual[i]                     
        sum_error += (prediction_error ** 2)                            
    mean_error = sum_error / float(len(actual))                             
    return sqrt(mean_error)


# # Baselines

# In[85]:

def random_algorithm(train, test):                                              
    output_values = [row[-1] for row in train]                              
    unique = list(set(output_values))                                       
    predicted = []                                           
    for row in test:                                                        
        index = randrange(len(unique))                                  
        predicted.append(unique[index])                                 
    return predicted                                                        
                                                                                
# In[86]:

def zero_rule_algorithm_classification(train, test):                            
    output_values = [row[-1] for row in train]                              
    prediction = max(set(output_values), key=output_values.count)           
    predicted = [prediction for i in range(len(train))]                     
    return predicted                                                        
                                                                                
# In[87]:

def zero_rule_algorithm_regression(train, test):                                
    output_values = [row[-1] for row in train]                              
    prediction = sum(output_values) / float(len(output_values))             
    predicted = [prediction for i in range(len(test))]                      
    return predicted                                                        
                                                                                

# # Test Harness

# In[88]:

def evaluate_algorithm_ttsplit(dataset, algorithm, split, *args):    
    # Train Test Split
    train, test = train_test_split(dataset, split)                          
    test_set = []                                                      
    for row in test:                                                        
        row_copy = list(row)     
        # delete class label
        row_copy[-1] = None                                             
        test_set.append(row_copy)
    # Fit
    predicted = algorithm(train, test_set, *args)                           
    actual = [row[-1] for row in test] 
    # Measure
    accuracy = accuracy_metric(actual, predicted)                           
    return accuracy 



# In[93]:

def evaluate_algorithm_kfold(dataset, algorithm, n_folds, *args):  
    # K folds
    folds = cross_validation_split(dataset, n_folds)                        
    scores = []                                                       
    for fold in folds:                                                      
        train_set = list(folds)                                         
        train_set.remove(fold)                                          
        train_set = sum(train_set, [])                                  
        test_set = list()
        # prep test set for each iteration
        for row in fold:                                                
            row_copy = list(row)                                    
            test_set.append(row_copy)                               
            row_copy[-1] = None           
        # fit
        predicted = algorithm(train_set, test_set, *args)               
        actual = [row[-1] for row in fold]                              
        accuracy = accuracy_metric(actual, predicted)                   
        scores.append(accuracy)                                         
    return scores


# In[94]:

# In[95]:

# Save ypour work


# In[ ]:
# get_ipython().system('jupyter nbconvert --to script config_template.ipynb')

