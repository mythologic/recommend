import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

# load data

df = pd.read_csv('recommend/data/candy.csv')
df = df[df['user'].isin([
    'martinezdillon',
    'bstark',
    'hoerin',
    'jessicapowers',
    'doylelaura'
])].reset_index(drop=True)

# interaction machine 

class Interactions:
    def __init__(self, users, items, ratings=None):
        if ratings is None:
            ratings = np.array(len(users) * [1])
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        uids = self.user_encoder.fit_transform(users)
        iids = self.item_encoder.fit_transform(items)
        nu = len(np.unique(uids))
        ni = len(np.unique(iids))
        self.matrix = sp.coo_matrix((ratings, (uids, iids)), shape=(nu, ni))

interactions = Interactions(df['user'], df['item'], df['review'])
interactions = Interactions(df['user'], df['item'])

# def train_test_split(interactions, test_percentage=0.2, random_state=None):

random_state = 42
random_state = np.random.RandomState(random_state)

test_percentage=0.2

matrix = interactions.matrix

idx = np.arange(len(matrix.row))
random_state.shuffle(idx)

u = matrix.row[idx]
i = matrix.col[idx]
ratings = matrix.data[idx]
cut = int((1.0 - test_percentage) * len(uids))


train = sp.coo_matrix(
    (data[:cut], (uids[:cut], iids[:cut])),
    shape=shape,
)
test = sp.coo_matrix(
    (data[test_idx], (uids[test_idx], iids[test_idx])),
    shape=shape,
    dtype=interactions.dtype,
)

    return train, test




#
