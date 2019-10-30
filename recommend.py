import numpy as np
from scipy.spatial.distance import cdist

class NearestNeighbors:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def __repr__(self):
        return f'NearestNeighbors(n_neighbors={self.n_neighbors})'

    def fit(self, X):
        self.X = X
        return self

    def kneighbors(self, X, return_distance=False):
        distances = cdist(X, self.X)
        neighbors = np.argsort(distances)[:, :self.n_neighbors]
        if return_distance:
            return distances, neighbors
        return neighbors

class ItemVectorizer:
    def __init__(self, delimiter=',', max_features=None):
        self.delimiter = delimiter
        if max_features:
            self.max_features = max_features
        else:
            self.max_features = np.inf

    def __repr__(self):
        return f'ItemVectorizer(delimiter="{self.delimiter}", max_features={self.max_features})'

    def fit(self, X):
        self.items = []
        for row in X:
            if len(self.items) > self.max_features:
                break
            for item in row.split(self.delimiter):
                if item not in self.items:
                    self.items.append(item)
        return self

    def transform(self, X):
        Xt = np.zeros((len(X), len(self.items)), dtype=int)
        for i, row in enumerate(X):
            for item in row.split(self.delimiter):
                try:
                    idx = self.items.index(item)
                    Xt[i, idx] = 1
                except ValueError:
                    pass
        return Xt

    def fit_transform(self, X):
        self.fit(X)
        Xt = self.transform(X)
        return Xt

class NNRecommender:
    def __init__(self, n_neighbors=5, delimiter=',', max_features=None):
        self.iv = ItemVectorizer(delimiter, max_features)
        self.nn = NearestNeighbors(n_neighbors)

    def __repr__(self):
        return f'NNRecommender(n_neighbors={self.nn.n_neighbors}, delimiter="{self.iv.delimiter}", max_features={self.iv.max_features})'

    def fit(self, X):
        self.X = X
        X = self.iv.fit_transform(X)
        self.nn.fit(X)
        return self

    def predict(self, X):
        Xp = []
        for Xi in X:
            Xt = self.iv.transform([Xi])
            neighbors = self.nn.kneighbors(Xt)
            repos = []
            for n in neighbors[0]:
                r = self.X.iloc[int(n)].split(",")
                repos.extend(r)
            repos = list(set(repos))
            repos = [r for r in repos if r not in Xi.split(",")]
            Xp.append(repos)
        return Xp

# # TODO:
# predict returns things that the person actually likes
# predict returns numpy array
# try to pickle
# AUC and precision@k
# train test split
# documentation
# doc strings
# logo
# github releases
# travis ci
# better examples to include
# better tests
