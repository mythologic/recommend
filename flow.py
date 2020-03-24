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
# interactions = Interactions(df['user'], df['item'])

# from sklearn.model_selection import train_test_split
# train_test_split()

def train_test_split(interactions, test_size=0.2, random_state=None):
    # shuffle
    idx = np.arange(len(matrix.row))
    np.random.RandomState(random_state).shuffle(idx)
    uids = interactions.matrix.row[idx]
    iids = interactions.matrix.col[idx]
    ratings = interactions.matrix.data[idx]
    # split
    cut = int((1.0 - test_size) * len(uids))
    train = sp.coo_matrix((ratings[:cut], (uids[:cut], iids[:cut])), shape=matrix.shape)
    test = sp.coo_matrix((ratings[cut:], (uids[cut:], iids[cut:])), shape=matrix.shape)
    return train, test

train, test = train_test_split(interactions)

pd.DataFrame(
    train.todense(),
    index=interactions.user_encoder.classes_,
    columns=interactions.item_encoder.classes_
)

pd.DataFrame(
    test.todense(),
    index=interactions.user_encoder.classes_,
    columns=interactions.item_encoder.classes_
)


#
