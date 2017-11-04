# Reads CSV file and returns train and test set


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# statistical data visualization library based on Matplotlib
# https://seaborn.pydata.org/
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import Imputer


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
# find_best_prediction_variables('/home/zack/Desktop/ML/AI_CLASS/Data/reProcessedHungarianData')


# def split_data(path, prediction_vars):
#     data = pd.read_csv(path, header=0, delimiter=' ')
#     # shuffle data
#     data.reindex(np.random.permutation(data.index))
#
#     # split into train and test dataset
#     train, test = train_test_split(data, test_size=0.3)
#
#     # train variables
#     train_x = train[prediction_vars]
#     # train output
#     train_y = train.num
#
#     # test variables
#     test_x = test[prediction_vars]
#     # test variables
#     test_y = test.num
#
#     return [train_x, train_y, test_x, test_y]


def split_all_data(prediction_vars):
    # folder with all data: reProcessedCleveland, reProcessedHungarian, reProcessedSwitzerland, reProcessedVA
    data_dir = r'/home/zack/Desktop/ML/AI_CLASS/Data'
    os.chdir(data_dir)

    data_list = []

    # grab all files with part-name reProcessed
    # slip by ',' and ' '
    # Empty values = '?'
    # Combine all into one dataset
    for file in glob.glob('reProcessed*'):
        df = pd.read_csv(file, index_col=None, header=0, sep=' |,', engine='python', na_values=["?"])
        data_list.append(df)

    results = pd.concat(data_list)

    # split into train and test dataset
    train, test = train_test_split(results, test_size=0.3)

    # train variables
    train_x = train[prediction_vars]
    # train output
    train_y = train.num

    # test variables
    test_x = test[prediction_vars]
    # test variables
    test_y = test.num

    return [train_x, train_y, test_x, test_y]










