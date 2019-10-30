import numpy as np
import pandas as pd
import pytest

from recommend import NearestNeighbors, ItemVectorizer
from recommend import NNRecommender

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
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(X)
    nn.kneighbors(X)
    r = nn.kneighbors(X[0].reshape(1, -1))
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
    iv = ItemVectorizer()
    iv.fit(df['items'])
    r = iv.transform(['c,h'])
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
    r = NNRecommender(n_neighbors=2)
    r.fit(food['food'])
    results = r.predict(['ethiopian,italian'])
    assert set(results[0]) == {'french', 'sushi', 'thai'}
