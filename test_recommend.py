import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import pytest

from recommend import AdjacentNeighbors, ThingVectorizer
from recommend import Recommend

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

@pytest.fixture
def Y(): #created another array to test with
    Y = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1],
    ])
    return Y

def test_an_works_like_cdist(X):
    an = AdjacentNeighbors(n=3)
    an.fit(X)
    dist, _ = an.kneighbors(X, return_distance=True)
    result = (dist == cdist(X, X))
    assert result.all()

def test_nearest_neighbors(X):
    an = AdjacentNeighbors(n=3)
    an.fit(X)
    r = an.kneighbors(X[0].reshape(1, -1))
    assert (r == np.array([[0, 2, 3]])).all()

def test_right_neighbors(X, Y, n=11):
    check = []
    for i in range(n+1):
        an = AdjacentNeighbors(n=i)
        an.fit(X)
        _, number_neighbors = an.kneighbors(Y[0:4]).shape
        number_neighbors
        if i > len(X): # to account for cases where i is more than number of neighbors available in X
            i = len(X)
            check.append(number_neighbors == i)
        else:
            check.append(number_neighbors == i)
    np.array(check)
    assert np.array(check).all()

def test_max_things(df):
    max_things = 2
    tv = ThingVectorizer(delimiter=',',max_things=max_things)
    tv.fit(df['items'])
    assert tv.max_things == max_things

def test_max_things_larger(df):
    max_things = 16
    tv = ThingVectorizer(delimiter=',',max_things=max_things)
    tv.fit(df['items'])
    assert tv.max_things == max_things

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

def test_thing_vectorizer(df):
    tv = ThingVectorizer()
    tv.fit(df['items'])
    r = tv.transform(['c,h'])
    assert (r == np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]])).all()

# Kavitha -> write a test for different delimiters like ' '
# write here

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
    anika = ['ethiopian,italian']
    results = r.predict(anika)
    assert set(results[0]) == {'french', 'sushi', 'thai'}

@pytest.fixture
def food():
    food = pd.DataFrame([
        ['Ryan', 'thai,mexican,indian,hawaiian'],
        ['Kavitha', 'thai,mexican,indian'],
        ['Hock', 'thai,sushi,ethiopian'],
        ['Benoit', 'thai,french,italian,ethiopian']
    ], columns=['name', 'food'])
    return food

def test_new_user(food):
    r = Recommend(n = 1)
    r.fit(food['food'])
    predictions = {}
    new_users = pd.DataFrame([
        ['Rowena', ['thai,viet,french']],
        ['Melissa', ['thai,korean,sushi']]
    ], columns = ['name', 'food'])
    i = 0
    for user in new_users['food']:
        prediction = r.predict(user)
        predictions.update({new_users['name'][i]: set(prediction[0])})
        i+= 1
    assert predictions == {'Rowena' : set(['italian','ethiopian']), 'Melissa' : set(['ethiopian'])}

def test_return_no_neighbors(food):
    r = Recommend(n=2)
    r.fit(food['food'])
    maxhumber = ['african,caribbean']
    results = r.predict(maxhumber)
    assert set(results[0]) == {}
