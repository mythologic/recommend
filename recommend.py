import numpy as np
from scipy.spatial.distance import cdist

class ThingVectorizer:
    def __init__(self, delimiter=',', max_things=None):
        self.delimiter = delimiter
        if max_things:
            self.max_things = max_things
        else:
            self.max_things = np.inf

    def __repr__(self):
        return f'ThingVectorizer(delimiter="{self.delimiter}", max_things={self.max_things})'

    def fit(self, X):
        self.things = []
        for row in X:
            for thing in row.split(self.delimiter):
                if (thing not in self.things) and (len(self.things) < self.max_things):
                    self.things.append(thing)
        return self

    def transform(self, X):
        Xt = np.zeros((len(X), len(self.things)), dtype=int)
        for i, row in enumerate(X):
            for thing in row.split(self.delimiter):
                try:
                    idx = self.things.index(thing)
                    Xt[i, idx] = 1
                except ValueError:
                    pass
        return Xt

    def fit_transform(self, X):
        self.fit(X)
        Xt = self.transform(X)
        return Xt

class AdjacentNeighbors:
    def __init__(self, n=5):
        self.n = n

    def __repr__(self):
        return f'AdjacentNeighbors(n={self.n})'

    def fit(self, X):
        self.X = X
        return self

    def kneighbors(self, X, return_distance=False):
        distances = cdist(X, self.X)
        neighbors = np.argsort(distances)[:, :self.n]
        if return_distance:
            return distances, neighbors
        return neighbors


class Recommend:
    def __init__(self, n=5, delimiter=',', max_things=None):
        self.tv = ThingVectorizer(delimiter, max_things)
        self.an = AdjacentNeighbors(n)

    def __repr__(self):
        return f'Recommend(n={self.an.n}, delimiter="{self.tv.delimiter}", max_things={self.tv.max_things})'

    def fit(self, X):
        self.X = X
        X = self.tv.fit_transform(X)
        self.an.fit(X)
        return self

    def predict(self, X):
        Xp = []
        for Xi in X:
            Xt = self.tv.transform([Xi])
            neighbors = self.an.kneighbors(Xt)
            things = []
            for n in neighbors[0]:
                t = self.X.iloc[int(n)].split(",")
                things.extend(t)
            things = list(set(things))
            things = [t for t in things if t not in Xi.split(",")]
            Xp.append(things)
        return Xp

if __name__ == '__main__':
    df = pd.DataFrame([
        ['Ryan', 'thai,mexican,indian,hawaiian'],
        ['Kavitha', 'thai,mexican,indian'],
        ['Hock', 'thai,sushi,ethiopian'],
        ['Benoit', 'thai,french,italian,ethiopian']
    ], columns=['name', 'food'])

    r = Recommend(n=2, delimiter=',')
    r.fit(df['food'])
    r.predict(df['food'])
    r.predict(['indian,sushi,italian,korean'])


# # TODO:
# remove scipy dependency -> numpy
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
