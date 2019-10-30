import numpy as np
from scipy.spatial.distance import cdist

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# X = np.random.binomial(1, p=0.5, size=(1000, 5000))

X = np.array([
    [0, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 1],
])

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

nn = NearestNeighbors(n_neighbors=3)
nn.fit(X)
nn.kneighbors(X[0].reshape(1, -1))

np.array([[0, 2, 3]])




df = pd.DataFrame([
    [1, 'a,b,c,d'],
    [2, 'a,c,d,e'],
    [3, 'f,g,h,i'],
    [4, 'b,c,h,i'],
    [5, 'j,k,l'],
    [6, 'k,l'],
], columns=['user', 'items'])

X = df['items'].values.tolist()
X
X[0]

X = df['items']

max_features = 5
delimiter = ','
items = []
for row in X:
    if len(items) > max_features:
        break
    for item in row.split(delimiter):
        if item not in items:
            items.append(item)

Xt = np.zeros((len(X), len(items)), dtype=int)
for i, row in enumerate(X):
    for item in row.split(delimiter):
        try:
            idx = items.index(item)
            Xt[i, idx] = 1
        except ValueError:
            pass
Xt

#####

class ItemVectorizer:
    def __init__(self, delimiter=',', max_features=None):
        self.delimiter = delimiter
        if max_features:
            self.max_features = max_features
        else:
            self.max_features = np.inf

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


cv = CountVectorizer(tokenizer=lambda x: x.split(","))
X = cv.fit_transform(df['items'])
cv.
X.todense()

class NNRecommender:
    def __init__(self, n_neighbors=3, max_features=1000, tokenizer=lambda x: x.split(",")):
        self.cv = CountVectorizer(tokenizer=tokenizer, max_features=max_features)
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, X):
        self.X = X
        X = self.cv.fit_transform(X)
        self.nn.fit(X)
        return self

    def predict(self, X):
        Xp = []
        for Xi in X:
            Xt = self.cv.transform([Xi])
            neighbors = self.nn.kneighbors(Xt, return_distance=False)
            repos = []
            for n in neighbors[0]:
                r = self.X.iloc[int(n)].split(",")
                repos.extend(r)
            repos = list(set(repos))
            repos = [r for r in repos if r not in Xi.split(",")]
            Xp.append(repos)
        return Xp

r = NNRecommender(n_neighbors=3)
r.fit(df['items'])
r.predict(df['items'])
