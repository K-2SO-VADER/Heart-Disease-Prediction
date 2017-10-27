# Reads CSV file and returns train and test set


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# statistical data visualization library based on Matplotlib
# https://seaborn.pydata.org/
import seaborn as sns
import matplotlib.pyplot as plt

# prediction variables
prediction_variables = ['age', 'sex', 'cp', 'threstbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal']


def find_best_prediction_variables(path):
    data = pd.read_csv(path, header=0, delimiter=' ')
    # variables that are used to predict are in columns 0 - 12
    pred_variables = list(data.columns[0:12])
    # find correlation between all prediction variables
    corr = data[pred_variables].corr()
    # plot figure
    plt.figure(figsize=(50 , 40))
    # map features
    sns.set(font_scale=0.8)
    sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size': 10}, xticklabels=pred_variables,
                yticklabels=pred_variables, cmap='coolwarm')
    plt.show()

# A visual check on the graph shows that none of the 13 prediction variables are correlated
# find_best_prediction_variables('/home/zack/Desktop/ML/AI_CLASS/Data/reprocessedHungarianData')


def split_data(path, prediction_vars):
    data = pd.read_csv(path, header=0, delimiter=' ')
    # shuffle data
    data.reindex(np.random.permutation(data.index))

    # split into train and test dataset
    train, test = train_test_split(data, test_size=0.3)

    # train variables
    train_x = train[prediction_vars]
    # train output
    train_y = train.num

    # test variables
    test_x = test[prediction_vars]
    # test variables
    test_y = test.num

    return [train_x, train_y, test_x, test_y]







