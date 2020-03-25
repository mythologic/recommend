# from .utils import ItemVectorizer, NearestNeighbors
from recommend.utils import Interactions, NearestNeighbors
from recommend.utils import train_test_split
from recommend.utils import load_candy

import pandas as pd

df = pd.read_csv('recommend/data/candy.csv')

df.head(1)

interactions = Interactions()
interactions.fit(df['item'])
matrix = interactions.transform(df['user'], df['item'], df['review'])

train, test = train_test_split(matrix)

nn = NearestNeighbors()
nn.fit(train)
nn.kneighbors(train)

interactions.user_encoder.inverse_transform([0])

df[df['user'] == 'darlene90']

for u in [0, 1978, 2020, 15, 286]:
    name = interactions.user_encoder.inverse_transform([u])[0]
    print(df[df['user'] == name])





class Recommend:
    def __init__(self, n=5, delimiter=",", max_items=None):
        self.iv = ItemVectorizer(delimiter, max_items)
        self.nn = NearestNeighbors(n)

    def __repr__(self):
        return f'Recommend(n={self.nn.n}, delimiter="{self.iv.delimiter}", max_items={self.iv.max_items})'

    def fit(self, X):
        self.X = X
        X = self.iv.fit_transform(X)
        self.nn.fit(X)
        return self

    def predict(self, X):
        Xp = []
        for Xi in X:
            Xt = self.iv.transform([Xi])
            neighbors = self.nn.kneighbors(Xt)
            things = []
            for n in neighbors[0]:
                t = self.X.iloc[int(n)].split(",")
                things.extend(t)
            things = list(set(things))
            things = [t for t in things if t not in Xi.split(",")]
            Xp.append(things)
        return Xp
