import pkg_resources
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances

def load_candy():
    """Load candy.csv as a numpy array.
    > Note: candy can and *should* be wrapped by `pd.DataFrame(candy)`
    Example:
    ```
    import pandas as pd
    raw_candy = load_candy()
    candy = pd.DataFrame(raw_candy)
    ```
    """
    CANDY_PATH = pkg_resources.resource_filename("recommend", "data/candy.csv")
    candy = np.genfromtxt(
        CANDY_PATH,
        delimiter=",",
        skip_header=1,
        dtype=[("user", "U100"), ("item", "U100"), ("rating", int)],
    )
    return candy

class LabelEncoder:
    def __init__(self):
        pass

    def fit(self, X):
        self.classes_ = []
        for xi in X:
            if xi not in self.classes_:
                self.classes_.append(xi)
        return self

    def transform(self, X):
        Xt = []
        for xi in X:
            if xi not in self.classes_:
                Xt.append(None)
            else:
                Xt.append(self.classes_.index(xi))
        return Xt

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        Xt = []
        for xi in X:
            try:
                Xt.append(self.classes_[xi])
            except IndexError:
                Xt.append(None)
        return Xt

class Interactions:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def __repr__(self):
        return 'Interactions()'

    @staticmethod
    def _remove(X, indices):
        return [v for i, v in enumerate(X) if i not in indices]

    def fit(self, items):
        self.item_encoder.fit(items)
        return self

    def transform(self, users, items, ratings=None):
        if ratings is None:
            ratings = np.array(len(users) * [1])
        uids = self.user_encoder.fit_transform(users)
        iids = self.item_encoder.transform(items)
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        # remove Nones
        indices = [i for i, value in enumerate(iids) if value is None]
        uids = self._remove(uids, indices)
        iids = self._remove(iids, indices)
        ratings = self._remove(ratings, indices)
        # create matrix
        n_users = len(np.unique(uids))
        matrix = sp.coo_matrix((ratings, (uids, iids)), shape=(n_users, n_items))
        return matrix

def train_test_split(matrix, test_size=0.2, random_state=None):
    # shuffle
    idx = np.arange(len(matrix.row))
    np.random.RandomState(random_state).shuffle(idx)
    uids = matrix.row[idx]
    iids = matrix.col[idx]
    ratings = matrix.data[idx]
    # split
    cut = int((1.0 - test_size) * len(uids))
    train = sp.coo_matrix((ratings[:cut], (uids[:cut], iids[:cut])), shape=matrix.shape)
    test = sp.coo_matrix((ratings[cut:], (uids[cut:], iids[cut:])), shape=matrix.shape)
    return train, test

class NearestNeighbors:
    def __init__(self, n=5):
        self.n = n

    def __repr__(self):
        return f"NearestNeighbors(n={self.n})"

    def fit(self, X):
        self.X = X
        return self

    def kneighbors(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        distances = euclidean_distances(X, self.X)
        neighbors = np.argsort(distances)[:, : self.n]
        return neighbors
