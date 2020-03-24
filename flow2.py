import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances

df = pd.DataFrame([
    ['max', 'skittles', 5],
    ['max', 'twix', 5],
    ['max', 'kitkat', 3],
    ['mc', 'twix', 4],
    ['mc', 'kitkat', 5],
    ['dex', 'ohenry', 4],
    ['dex', 'skittles', 5]
], columns=['user', 'item', 'rating'])

from utils import LabelEncoder

class Interactions:
    def __init__(self):
        self.item_encoder = LabelEncoder()

    def __repr__(self):
        return 'Interactions()'

    def fit(self, items):
        self.item_encoder.fit(items)
        self.n_items = len(self.item_encoder.classes_)
        return self

    def transform(self, users, items, ratings=None):
        if ratings is None:
            ratings = np.array(len(users) * [1])
        user_encoder = LabelEncoder()
        uids = user_encoder.fit_transform(users)
        iids = self.item_encoder.transform(items)
        n_users = len(user_encoder.classes_)
        # remove Nones
        ri = [i for i, value in enumerate(iids) if value is None]
        uids = [value for i, value in enumerate(uids) if i not in ri]
        iids = [value for i, value in enumerate(iids) if i not in ri]
        ratings = [value for i, value in enumerate(ratings) if i not in ri]
        # create matrix
        n_users = len(np.unique(uids))
        matrix = sp.coo_matrix((ratings, (uids, iids)), shape=(n_users, self.n_items))
        return matrix, user_encoder

interactions = Interactions()
interactions.fit(df['item'])
matrix, user_encoder = interactions.transform(df['user'], df['item'], df['rating'])

pd.DataFrame(matrix.todense(),
    index=user_encoder.classes_,
    columns=interactions.item_encoder.classes_
)

new = pd.DataFrame([
    ['kw', 'skittles', 4],
    ['kw', 'candycorn', 3]
], columns=['user', 'item', 'rating'])

matrix, user_encoder = interactions.transform(new['user'], new['item'], new['rating'])

pd.DataFrame(matrix.todense(),
    index=user_encoder.classes_,
    columns=interactions.item_encoder.classes_
)

new = pd.DataFrame([
    ['rich', 'skittles', 4],
], columns=['user', 'item', 'rating'])

matrix, user_encoder = interactions.transform(new['user'], new['item'], new['rating'])

pd.DataFrame(matrix.todense(),
    index=user_encoder.classes_,
    columns=interactions.item_encoder.classes_
)

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

train, test = train_test_split(matrix)

train.todense()
test.todense()

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

nn = NearestNeighbors()
nn.fit(train)
nn.kneighbors(train)
