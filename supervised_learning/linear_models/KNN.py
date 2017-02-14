from math import sqrt

class KNN(object):
    @staticmethod
    def euclid(sample1, sample2):
        dist = 0.0
        for i in range(len(sample1)-1):
            dist += (sample1[i] - sample2[i])**2
        return sqrt(dist)

    def predict(self, x, y, newSample, k, predictType=None):
        if isinstance(predictType, str):
            # put all labels on the last row of feature matrix
            Y = y[np.newaxis]
            X = np.concatenate([x,Y.T], 1)

            neighbors = self._get_neighbors(X, newSample, k)
            labels = [ row[-1] for row in neighbors ]

            if "classification" in predictType.lower():
                return prediction = max(set(labels), key=labels.count)
            elif "regression" in predictType.lower():
                return prediction = sum(labels) / float(len(labels))

    def _get_neighbors(X, newSample, k):
        distances = []
        neighbors = []
        for row in X:
            dist = euclid(newSample, row)
            distances.append((row, dist))
        distances.sort(key=lambda tup: tup[1])

        for n in range(k):
            neighbors.append(distances[n][0])
        return neighbors



