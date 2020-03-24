import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.preprocessing import LabelEncoder

# load and slim down data

df = pd.read_csv('recommend/data/candy.csv')
# old = df[df['user'].isin([
#     'martinezdillon',
#     'bstark',
#     'hoerin',
#     'jessicapowers',
#     'doylelaura'
# ])].reset_index(drop=True)

df = pd.DataFrame([
    ['max', 'skittles', 5],
    ['max', 'twix', 5],
    ['max', 'kitkat', 3],
    ['maggie', 'twix', 4],
    ['maggie', 'kitkat', 5],
    ['dex', 'ohenry', 4],
    ['dex', 'skittles', 5]
], columns=['user', 'item', 'rating'])

df

# interaction machine

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

    def inverse_transform(self, X):
        Xt = []
        for xi in X:
            try:
                Xt.append(self.classes_[xi])
            except IndexError:
                Xt.append(None)
        return Xt

from unittest.mock import Mock
self = Mock()

# class Interactions:
#     def __init__(self):
self.user_encoder = LabelEncoder()
self.item_encoder = LabelEncoder()

def fit(self, items):
    self.item_encoder.fit(items)
    self.ni = len(self.item_encoder.classes_)
    return self

fit(self, df['item'])
self.item_encoder.classes_
self.ni



[i for i, value in enumerate(yt) if value is None]



def transform(self, users, items, ratings=None):
    if ratings is None:
        ratings = np.array(len(users) * [1])
    uids = self.user_encoder.fit_transform(users)
    iids = self.item_encoder.transform(items)
    nu = len(np.unique(uids))
    self.matrix = sp.coo_matrix((ratings, (uids, iids)), shape=(nu, self.ni))
    return self.matrix

interactions = Interactions()
interactions.fit(df['item'])
matrix = interactions.transform(df['user'], df['item'], df['rating'])

pd.DataFrame(matrix.todense(),
    index=interactions.user_encoder.classes_,
    columns=interactions.item_encoder.classes_
)






new = df[df['user'].isin([
    'darlene90',
    'taylordarlene',
    'aliciadennis'
])].reset_index(drop=True)

interactions.transform(new['user'], new['item'], new['review'])







from sklearn.preprocessing import MultiLabelBinarizer




mlb.classes_
array(['comedy', 'sci-fi', 'thriller'], dtype=object)






def train_test_split(interactions, test_size=0.2, random_state=None):
    matrix = interactions.matrix
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

train, test = train_test_split(interactions)

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
