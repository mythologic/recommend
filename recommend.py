import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

x = [0, 0, 1, 1, 0, 0, 1]
y = [0, 0, 1, 1, 1, 0, 1]

def dist(x, y):
    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

dist(x, y)

X = np.array([
    [0, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0]
])

%%timeit
euclidean_distances(X)

%%timeit
D = np.zeros((X.shape[0], X.shape[0]))
for x, xi in enumerate(X):
    for y, xj in enumerate(X):
        D[x, y] = dist(xi, xj)

def dist(x, y):
    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))


def euclidean_distances(X):
    distances = np.zeros((X.shape[0], X.shape[0]))
    for x, xi in enumerate(X):
        for y, xj in enumerate(X):
            distances[x, y] = dist(xi, xj)
    return distances



df = pd.DataFrame([
    [1, 'a,b,c,d'],
    [2, 'a,c,d,e'],
    [3, 'f,g,h,i'],
    [4, 'b,c,h,i'],
    [5, 'j,k,l'],
    [6, 'k,l'],
], columns=['user', 'items'])

cv = CountVectorizer(tokenizer=lambda x: x.split(","))
X = cv.fit_transform(df['items'])
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
