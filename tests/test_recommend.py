import pandas as pd
import pytest

from recommend import Recommend


@pytest.fixture
def food():
    food = pd.DataFrame(
        [
            ["Ryan", "thai,mexican,indian,hawaiian"],
            ["Kavitha", "thai,mexican,indian"],
            ["Hock", "thai,sushi,ethiopian"],
            ["Benoit", "thai,french,italian,ethiopian"],
        ],
        columns=["name", "food"],
    )
    return food


def test_nn_recommender(food):
    r = Recommend(n=2)
    r.fit(food["food"])
    anika = ["ethiopian,italian"]
    results = r.predict(anika)
    assert set(results[0]) == {"french", "sushi", "thai"}
