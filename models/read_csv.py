# Reads CSV file and returns train and test set


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_data(path):
    data = pd.read_csv(path, header=0, delimiter=' ')
    # shuffle data
    data.reindex(np.random.permutation(data.index))

    # prediction variables
    prediction_variables = ['age', 'sex', 'cp', 'threstbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal']

    # split into train and test dataset
    train, test = train_test_split(data, test_size=0.3)

    # train variables
    train_x = train[prediction_variables]
    # train output
    train_y = train.num

    # test variables
    test_x = test[prediction_variables]
    # test variables
    test_y = test.num

    return [train_x, train_y, test_x, test_y]







