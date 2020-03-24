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
