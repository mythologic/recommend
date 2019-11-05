import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import pytest

from recommend import AdjacentNeighbors

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
def Y():
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
