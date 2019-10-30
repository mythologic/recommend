import numpy as np
import pytest

from recommend import NearestNeighbors

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
