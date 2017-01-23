import numpy as np
from TreeCriterion import gini, entropy

class DecisionTree(object):
    def __init__(self, criterion="entropy", max_depth=1, min_size=1):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None
        if "gini" in criterion:
            self.criterion = gini
        elif "entropy" in criterion:
            self.criterion = entropy

    def fit(self, x, y):
        # put all labels on the last row of feature matrix
        Y = y[np.newaxis]
        X = np.concatenate([x,Y.T], 1)

        self.root = self._get_split(X)
        self._split(self.root, self.max_depth, self.min_size, 1)
        return self

    def _get_split(self, X):
        class_values = list(set(row[-1] for row in X)) 
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(X[0])-1):             
            for sample in X:                          
                groups = self._test_split(index, sample[index], X)
                score = self.criterion(groups, class_values)

                if score < b_score:
                    b_index, b_value, b_score, b_groups = \
                            index, sample[index], score, groups
        return {"index":b_index, "value":b_value, "groups":b_groups}

    def _test_split(self, index, value, X):
        left, right = list(), list()
        for sample in X:
            if sample[index] < value: 
                left.append(sample)
            else:
                right.append(sample)
        return left, right

    def _split(self, node, max_depth, min_size, depth):
        left, right = node["groups"]
        del(node["groups"])
        # check if left or right empty
        if not left or not right:
            node["left"] = node["right"] = self._toterminal(left+right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node["left"], node["right"] = self._toterminal(left), \
                    self._toterminal(right)
            return
        # process left child
        if len(left) > self.min_size:
            node["left"] = self._get_split(left)
            self._split(node["left"], self.max_depth, 
                        self.min_size, depth+1)
        else:
            node["left"] = self._toterminal(left)
        # process left child
        if len(right) > self.min_size:
            node["right"] = self._get_split(right)
            self._split(node["right"], self.max_depth, 
                        self.min_size, depth+1)
        else:
            node["right"] = self._toterminal(right)

    def _toterminal(self, group):
        outcomes = [ sample[-1] for sample in group ]
        return max(set(outcomes), key=outcomes.count)

    
    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append( self._predict(self.root, sample) )
        return predictions

    def _predict(self, node, row):
        if row[node["index"]] < node["value"]:
            if isinstance(node["left"],dict):
                return self._predict(node["left"], row)
            else:
                return node["left"]
        else:
            if isinstance(node["right"],dict):
                return self._predict(node["right"], row)
            else:
                return node["right"]




def print_tree(node, depth=0):
    if isinstance(node, dict):
        print( " {0}[X{1} < {2:.3f}] ".format(
            depth*' ' , (node[ "index" ]+1), node["value" ]) )
        print_tree(node[ "left" ], depth+1)
        print_tree(node[ "right" ], depth+1)
    else:
        print( " {0}[{1}] ".format(depth*' ' , node))

