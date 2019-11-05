import pandas as pd
import pytest

from recommend import Recommend

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

# # FIXME: need to get this working
# def test_return_no_neighbors(food):
#     r = Recommend(n=2)
#     r.fit(food['food'])
#     maxhumber = ['african,caribbean']
#     results = r.predict(maxhumber)
#     assert set(results[0]) == {}
