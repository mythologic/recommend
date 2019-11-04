import numpy as np
import pandas as pd
import pytest

from recommend import AdjacentNeighbors, ThingVectorizer
from recommend import Recommend

from scipy.spatial.distance import cdist

X = np.array([
    [0, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 1],
])

%%timeit
cdist(X, X)

xy1 = X
xy2 = X

def ndist(xy1, xy2):
    P = np.add.outer(np.sum(xy1**2, axis=1), np.sum(xy2**2, axis=1))
    N = np.dot(xy1, xy2.T)
    return np.sqrt(P - 2*N)

%%timeit
ndist(X, X)




@pytest.fixture
def X():
    X = np.array([
        [0, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 1],
    ])
    return X

def test_nearest_neighbors(X):
    an = AdjacentNeighbors(n=3)
    an.fit(X)
    an.kneighbors(X)
    r = an.kneighbors(X[0].reshape(1, -1))
    assert (r == np.array([[0, 2, 3]])).all()

@pytest.fixture
def df():
    df = pd.DataFrame([
        [1, 'a,b,c,d'],
        [2, 'a,c,d,e'],
        [3, 'f,g,h,i'],
        [4, 'b,c,h,i'],
        [5, 'j,k,l'],
        [6, 'k,l'],
    ], columns=['user', 'items'])
    return df

def test_item_vectorizer(df):
    tv = ThingVectorizer()
    tv.fit(df['items'])
    r = tv.transform(['c,h'])
    assert (r == np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]])).all()

@pytest.fixture
def food():
    food = pd.DataFrame([
        ['Ryan', 'thai,mexican,indian,hawaiian'],
        ['Kavitha', 'thai,mexican,indian'],
        ['Hock', 'thai,sushi,ethiopian'],
        ['Benoit', 'thai,french,italian,ethiopian']
    ], columns=['name', 'food'])
    return food

def test_nn_recommender(food):
    r = Recommend(n=2)
    r.fit(food['food'])
    results = r.predict(['ethiopian,italian'])
    assert set(results[0]) == {'french', 'sushi', 'thai'}
