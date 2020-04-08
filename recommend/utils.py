import pkg_resources
import numpy as np
from scipy.spatial.distance import cdist

def load_candy():
    CANDY_PATH = pkg_resources.resource_filename("recommend", "data/candy.csv")
    candy = np.genfromtxt(
        CANDY_PATH,
        delimiter=",",
        skip_header=1,
        dtype=[("user", "U100"), ("item", "U100"), ("rating", int)],
    )
    return candy

class ItemVectorizer:
    def __init__(self, delimiter=",", max_items=None):
        self.delimiter = delimiter
        if max_items:
            self.max_items = max_items
        else:
            self.max_items = np.inf

    def __repr__(self):
        return (
            f'ItemVectorizer(delimiter="{self.delimiter}", max_items={self.max_items})'
        )

    def fit(self, X):
        self.items = []
        for row in X:
            for thing in row.split(self.delimiter):
                if (thing not in self.items) and (len(self.items) < self.max_items):
                    self.items.append(thing)
        return self

    def transform(self, X):
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
        self.fit(X)
        Xt = self.transform(X)
        return Xt


class NearestNeighbors:
    def __init__(self, n=5):
        self.n = n

    def __repr__(self):
        return f"NearestNeighbors(n={self.n})"

    def fit(self, X):
        self.X = X
        return self

    def kneighbors(self, X):
        if isinstance(X, list):
            X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        distances = cdist(X, self.X)
        neighbors = np.argsort(distances)[:, : self.n]
        return neighbors
