import numpy as np
import pytest

from recommend import NearestNeighbors


@pytest.fixture
def X():
    X = np.array(
        [
            [0, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 1],
        ]
    )
    return X


def test_nearest_neighbors(X):
    nn = NearestNeighbors(n=3)
    nn.fit(X)
    result = nn.kneighbors([1, 0, 0, 1, 1, 0, 1])
    assert (result == np.array([[6, 1, 2]])).all()
