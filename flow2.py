import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances

from utils import LabelEncoder


interactions = Interactions()
interactions.fit(df['item'])
matrix = interactions.transform(df['user'], df['item'], df['rating'])

pd.DataFrame(matrix.todense(),
    index=interactions.user_encoder.classes_,
    columns=interactions.item_encoder.classes_
)



train, test = train_test_split(matrix)

train.todense()
test.todense()



nn = NearestNeighbors()
nn.fit(train)
nn.kneighbors(train)
