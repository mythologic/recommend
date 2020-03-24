import pandas as pd
from recommend import load_candy

def test_candy():
    candy = load_candy()
    assert candy[0][-1] == 5

def test_candy_df():
    candy = pd.DataFrame(load_candy())
    columns = candy.columns.values.tolist()
    assert columns == ["user", "item", "rating"]
