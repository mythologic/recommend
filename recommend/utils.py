import pkg_resources
import numpy as np
from scipy.spatial.distance import cdist


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


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

df = pd.DataFrame([
    ['this is a sentence'],
    ['this is also a sentence'],
    ['this is not a sentence']
], columns=['sentences'])

X = df['sentences'].values

cvec = CountVectorizer(stop_words=None)
cvec.fit(df['sentences'])
pd.DataFrame(
    cvec.transform(df['sentences']).todense(),
    columns=cvec.get_feature_names()
)
dir(cvec)

# 1. split
# 2. map words to integers
# >>> Maybe. figure out size of vocab
# 3. figure out how to fill out zeroes and ones

X = df['sentences'].values

delimiter = ' '
max_items = 100

# this is the fitting step

items_ = []
for row in X:
    for item in row.split(delimiter):
        if (item not in items_) and (len(items_) < max_items):
            items_.append(item)

items_

# Xt = np.zeros((len(X), len(items)), dtype=int)
# for i, row in enumerate(X):
#     for thing in row.split(delimiter):
#         try:
#             idx = items.index(thing)
#             Xt[i, idx] = 1
#         except ValueError:
#             pass

from scipy.sparse import csr_matrix

total_users = len(X)
total_items = len(items_)

users = []
items = []
for user, item_list in enumerate(X):
    for item in item_list.split(delimiter):
        try:
            users.append(user)
            items.append(items_.index(item))
        except ValueError:
            pass

data = [1] * len(users)

pd.DataFrame(
    csr_matrix((data, (users, items)), shape=(total_users, total_items)).todense(),
    columns=items_
)

test = csr_matrix((data, (users, items)), shape=(total_users, total_items)).todense()

cdist(test, test)


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
        """Fit an unsupervised nearest neighbors algorithm.
        Uses Eucliedien Distance.

        Params:
        - n: number of nearest neighbors
        """
        self.n = n

    def __repr__(self):
        return f"NearestNeighbors(n={self.n})"

    def fit(self, X):
        """
        Params:

        - X: the X array (2d)
        """
        self.X = X
        return self

    def kneighbors(self, X):
        """
        """
        if isinstance(X, list):
            X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        distances = cdist(X, self.X)
        neighbors = np.argsort(distances)[:, : self.n]
        return neighbors
