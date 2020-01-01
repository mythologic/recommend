import pkg_resources
import numpy as np
from scipy.spatial.distance import cdist


def load_candy():
    """Load the candy data set as a numpy array.
    > Note: candy can and should be wrapped by `pd.DataFrame(candy)`
    Example:
    ```
    import pandas as pd
    raw_candy = load_candy()
    candy = pd.DataFrame(raw_candy)
    ```
    """
    CANDY_PATH = pkg_resources.resource_filename("recommend", "data/candy.csv")
    # f = "data/candy.csv"
    candy = np.genfromtxt(
        CANDY_PATH,
        delimiter=",",
        skip_header=1,
        dtype=[("user", "U100"), ("item", "U100"), ("rating", int)],
    )
    return candy


class ItemVectorizer:
    """Convert a collection of items into a matrix
    Example:
    ```
    X = ['a,b,c','b,c','c,d','a']
    iv = ItemVectorizer()
    iv.fit_transform(X)
    # array([
    #     [1, 1, 1, 0],
    #     [0, 1, 1, 0],
    #     [0, 0, 1, 1],
    #     [1, 0, 0, 0]
    # ])
    ```
    """

    def __init__(self, delimiter=",", max_items=None):
        """
        Params:
        - delimiter (str, ','): the list separator
        - max_items (int, None): maximum number of items to vectorize
        """
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
        """Learn a matrix representation of items
        Params:
        - X (list-like object): data to learn from
        """
        self.items = []
        for row in X:
            for thing in row.split(self.delimiter):
                if (thing not in self.items) and (len(self.items) < self.max_items):
                    self.items.append(thing)
        return self

    def transform(self, X):
        """Convert items into a matrix representation
        Params:
        - X (list-like object): data to transform
        """
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
        """See .fit and .transform
        Params:
        - X (list-like object): data to fit and transform
        """
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
