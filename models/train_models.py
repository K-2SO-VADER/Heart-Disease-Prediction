# import split data from read_csv
# Note: Ensure you have __init__.py in the same directory
# __init__.py makes it possible to import classes/functions/variables from other python files
from read_csv import split_all_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import pickle # Persist model in disk

# prediction variables
prediction_variables = ['age', 'sex', 'cp', 'threstbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal']

# path = '/home/zack/Desktop/ML/AI_CLASS/Data/reProcessedHungarianData'
data = split_all_data(prediction_variables)


train_x = data[0]
train_y = data[1]
test_x = data[2]
test_y = data[3]
cross_validation_data = data[4]


# fit a Decision Tree Classifier
def decision_tree_model():
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_x , train_y)
    s = pickle.dumps(clf)
    clf2 = pickle.loads(s)
    prediction = clf2.predict(test_x)
    print('Accuracy Score: ', accuracy_score(prediction, test_y))


# fit a Random Classifier
# Random Forest is an ensemble classifier: Combines multiple Decision Trees into one
# for better prediction
def predict_rf_model():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)

    saved = pickle.dump(model, open('rf_model_dumb', 'wb'))
    model2 = pickle.load(open('rf_model_dumb', 'rb'))
    prediction = model2.predict(test_x)

    rf_accuracy = accuracy_score(prediction, test_y)

    # optional: Print actual predictions vs Predicted results
    output = pd.DataFrame(data={"Actual": test_y, "predicted": prediction})
    output.to_csv("RFComparison.csv", index=False)

    return rf_accuracy


# fit SVM algorithm
# SVM finds an optimal hyperplane in the dataset then classifies data into into one of the dimensions
# A hyperplane divides a given dimension into multiple dimensions
def predict_svm():
    model = svm.SVC()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    accuracy = accuracy_score(prediction, test_y)
    return accuracy


# use grid search CV to find the parameters for prediction using decision trees
def model_grid_search_cv(model, param_grid, data_x, data_y):
    clf = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
    clf.fit(train_x, train_y)
    print("Best Parameters: ")
    print(clf.best_params_)
    print("Best Accuracy Score: ")
    print(clf.best_score_)

    return clf.best_params_


# Decision Trees Parameters: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
'''
    Criterion: Measure Quality of split
    Splitter:  The strategy used to choose split at each node. 
    Maximum depth: Maximum depth of the tree
    Min Samples Split: number of samples required to split an internal node
    .......
'''


def decision_trees_grid_search():
    model = DecisionTreeClassifier()
    parameters_grid = [
        {'criterion': ['gini', 'entropy'],
         'splitter': ['best' , 'random'],
         'max_depth': [5, 10, 15, 40],
         'min_samples_leaf': [1, 2, 3, 4, 5]}
    ]
    model_grid_search_cv(model, parameters_grid, train_x, train_y)


def random_forest_grid_search():
    model = RandomForestClassifier()
    parameters_grid = [
        {
            'n_estimators': [10, 50, 100, 150, 200],
            'criterion' : ['gini' , 'entropy'],
            'max_features' : ['auto' , 'sqrt' , 'log2'],
            'max_depth': [5, 10, 15, 40],
            'min_samples_leaf': [1, 2, 3, 4, 5]

        }
    ]
    model_grid_search_cv(model, parameters_grid, train_x, train_y)


print("----GridSearchCV Decision Trees----")
decision_tree_model()
decision_trees_grid_search()

print("----- Random Forest Classifier-----")
print(predict_rf_model())
random_forest_grid_search()


# Cross Validation of Models: Prevent over-fitting
def classification_model(model, data, prediction_input, output):
    # fit using training set
    model.fit(data[prediction_input], data[output])
    # predict based on training set
    predictions = model.predict(data[prediction_input])
    # accuracy
    accuracy = accuracy_score(predictions, data[output])
    print ("Accuracy: ", accuracy)

    # create 5 partitions
    kf = KFold(n_splits=5)

    error = []

    # Split dataset into 5 consecutive folds
    for train, test in kf.split(data):
        # rows & columns
        train_X = (data[prediction_input].iloc[train,:])
        # rows
        train_y = data[output].iloc[train]
        model.fit(train_X, train_y)

        # test data also
        test_X = data[prediction_input].iloc[test, :]
        test_y = data[output].iloc[test]
        error.append(model.score(test_X, test_y))

        # score
        print("Cross Validation score: ", np.mean(error))


def cross_validation_RF_Classifier():
    model = RandomForestClassifier(n_estimators=100)
    outcome_var = "num"
    classification_model(model, cross_validation_data, prediction_variables, outcome_var)

print("-----Cross Validation Results-----")
cross_validation_RF_Classifier()


'''
Final To do: 
Create a Pipeline for all the above. 
'''
'''
Initial Data: 
----GridSearchCV Decision Trees----
Accuracy Score:  0.595505617978
Best Parameters: 
{'criterion': 'gini', 'splitter': 'random', 'max_depth': 15, 'min_samples_leaf': 4}
Best Accuracy Score: 
0.673170731707

----- Random Forest Classifier-----
0.651685393258
Best Parameters: 
{'criterion': 'gini', 'max_depth': 15, 'max_features': 'sqrt', 'n_estimators': 50, 'min_samples_leaf': 3}
Best Accuracy Score: 
0.692682926829

-----SVM Classifier-----
0.685393258427
Best Parameters: 
{'C': 1, 'kernel': 'linear'}
Best Accuracy Score: 
0.653658536585

'''

'''
    To do:
    -1.Implement Imputer Class to fit and predict NaN values. 
   
   
      
'''

'''
Note: Increasing Data does not necessarily improve outcomes
1. This set uses all 3 'reprocessed' files with NaN rows removed

----GridSearchCV Decision Trees----
Accuracy Score:  0.550561797753
Best Parameters: 
{'min_samples_leaf': 5, 'criterion': 'gini', 'splitter': 'random', 'max_depth': 40}
Best Accuracy Score: 
0.642512077295
----- Random Forest Classifier-----
0.584269662921
Best Parameters: 
{'min_samples_leaf': 2, 'max_features': 'auto', 'criterion': 'gini', 'n_estimators': 150, 'max_depth': 10}
Best Accuracy Score: 
0.661835748792
-----SVM Classifier-----
0.550561797753

2. All Missing values replaced with mean of the column values
Accuracy Score:  0.416666666667
Best Parameters: 
{'criterion': 'gini', 'max_depth': 5, 'splitter': 'best', 'min_samples_leaf': 1}
Best Accuracy Score: 
0.534161490683
----- Random Forest Classifier-----
0.518115942029
...........................

Default: Remove all NaN rows

-----Cross Validation Results-----
Accuracy:  1.0
Cross Validation score:  0.621848739496
Cross Validation score:  0.609243697479
Cross Validation score:  0.606727436737
Cross Validation score:  0.622418458909
Cross Validation score:  0.638612733229

'''


