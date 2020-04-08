import numpy as np
import pytest

from recommend import ItemVectorizer


@pytest.fixture
def X():
    X = np.array(["a,b,c,d", "a,c,d,e", "f,g,h,i", "b,c,h,i", "j,k,l", "k,l"])
    return X


def test_max_items(X):
    max_items = 2
    iv = ItemVectorizer(delimiter=",", max_items=max_items)
    iv.fit(X)
    assert iv.max_items == max_items


def test_max_items_larger(X):
    max_items = 16
    iv = ItemVectorizer(delimiter=",", max_items=max_items)
    iv.fit(X)
    assert iv.max_items == max_items


def test_thing_vectorizer(X):
    iv = ItemVectorizer()
    iv.fit(X)
    r = iv.transform(["c,h"])
    assert (r == np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]])).all()
