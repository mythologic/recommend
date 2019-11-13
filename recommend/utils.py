import numpy as np
from scipy.spatial.distance import cdist

class ItemVectorizer:
    def __init__(self, delimiter=',', max_items=None):
        '''
        Params:

        - delimiter:
        - max_items:
        '''
        self.delimiter = delimiter
        if max_items:
            self.max_items = max_items
        else:
            self.max_items = np.inf

    def __repr__(self):
        return f'ItemVectorizer(delimiter="{self.delimiter}", max_items={self.max_items})'

    def fit(self, X):
        '''
        Params:

        - X
        '''
        self.items = []
        for row in X:
            for thing in row.split(self.delimiter):
                if (thing not in self.items) and (len(self.items) < self.max_items):
                    self.items.append(thing)
        return self

    def transform(self, X):
        '''
        Params:

        - X:
        '''
        Xt = np.zeros((len(X), len(self.items)), dtype=int)
        for i, row in enumerate(X):
            for thing in row.split(self.delimiter):
                try:
                    idx = self.items.index(thing)
                    Xt[i, idx] = 1
                except ValueError:
                    pass
        return Xt

    def fit_transform(self, X):
        '''
        Params:

        - X:
        '''
        self.fit(X)
        Xt = self.transform(X)
        return Xt

class NearestNeighbors:
    def __init__(self, n=5):
        '''Fit an unsupervised nearest neighbors algorithm.
        Uses Eucliedien Distance.

        Params:
        - n: number of nearest neighbors
        '''
        self.n = n

    def __repr__(self):
        return f'NearestNeighbors(n={self.n})'

    def fit(self, X):
        '''
        Params:

        - X: the X array (2d)
        '''
        self.X = X
        return self

    def kneighbors(self, X):
        '''
        '''
        if isinstance(X, list):
            X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        distances = cdist(X, self.X)
        neighbors = np.argsort(distances)[:, :self.n]
        return neighbors
