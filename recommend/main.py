from .utils import ItemVectorizer, NearestNeighbors

class NNRecommender:
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
