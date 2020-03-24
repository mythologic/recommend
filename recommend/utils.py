import pkg_resources
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

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

class ItemVectorizer:
    """Learn a vectorized representation of(similar to LabelBinarizer/CountVectorizer)

    Params:
    - delimiter (str, default=','): item separator
    - max_items (int, default=None): total number of items to vectorize

    ```
    import pandas as pd
    raw_candy = load_candy()
    candy = pd.DataFrame(raw_candy)
    ```
    """

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
        self.items_ = []
        for row in X:
            for item in row.split(self.delimiter):
                item = item.strip()
                if (item not in self.items_) and (len(self.items_) < self.max_items):
                    self.items_.append(item)
        return self

    def transform(self, X):
        users = []
        items = []
        for user, item_list in enumerate(X):
            for item in item_list.split(self.delimiter):
                item = item.strip()
                try:
                    users.append(user)
                    items.append(self.items_.index(item))
                except ValueError:
                    pass
        data = [1] * len(users)
        matrix = csr_matrix((data, (users, items)), shape=(len(X), len(self.items_)))
        return matrix

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
