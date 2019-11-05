import numpy as np
import pandas as pd
import pytest

from recommend import ThingVectorizer

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

def test_thing_vectorizer(df):
    tv = ThingVectorizer()
    tv.fit(df['items'])
    r = tv.transform(['c,h'])
    assert (r == np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]])).all()
