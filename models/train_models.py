# import split data from read_csv
# Note: Ensure you have __init__.py in the same directory
# __init__.py makes it possible to import classes/functions/variables from other python files

from models.read_csv import split_all_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import Imputer

# prediction variables
prediction_variables = ['age', 'sex', 'cp', 'threstbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal']


# path = '/home/zack/Desktop/ML/AI_CLASS/Data/reProcessedHungarianData'
data = split_all_data(prediction_variables)

train_x = data[0]
train_y = data[1]
test_x = data[2]
test_y = data[3]


# fit a Decision Tree Classifier
def decision_tree_model():
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_x , train_y)
    prediction = clf.predict(test_x)
    print('Accuracy Score: ', accuracy_score(prediction, test_y))


# fit a Random Classifier
# Random Forest is an ensemble classifier: Combines multiple Decision Trees into one
# for better prediction
def predict_rf_model():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    rf_accuracy = accuracy_score(prediction, test_y)
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


def svm_grid_search():
    model = svm.SVC()
    param_grid = [
        {'C': [1, 10, 100, 1000],
         'kernel': ['linear']
         },
        {'C': [1, 10, 100, 1000],
         'gamma': [0.001, 0.0001],
         'kernel': ['rbf']
         },
    ]
    model_grid_search_cv(model, param_grid, train_x , train_y)


print("----GridSearchCV Decision Trees----")
decision_tree_model()
decision_trees_grid_search()

print("----- Random Forest Classifier-----")
print(predict_rf_model())
random_forest_grid_search()

print("-----SVM Classifier-----")
print(predict_svm())
svm_grid_search()


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
    0. Figure out why the consistently low accuracy rates. 
    1. Play around with parameters to improve accuracy
    2. Hold out set to reduce overfitting.
    3. Find the best prediction variables from all >76 features/prediction variables
      
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

'''


