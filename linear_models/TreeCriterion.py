from math import log

def gini(groups, class_values):
        gini = 0.0
        for label in class_values:
            for group in groups:
                size = float(len(group))
                if size!=0:
                    proportion = [ sample[-1] for sample in group
                                 ].count(label) / size
                    gini += proportion * (1-proportion)
        return gini

def entropy(groups, class_values):
    entropy = 0.0
    for label in class_values:
        for group in groups:
            size = float(len(group))
            if size!=0:
                proportion = [ sample[-1] for sample in group
                             ].count(label) / size
                entropy -= proportion * log(proportion, 2)
    return entropy
