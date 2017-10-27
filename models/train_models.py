# import split data from read_csv
# Note: Ensure you have __init__.py in the same directory
# __init__.py makes it possible to import classes/functions/variables from other python files
from models.read_csv import split_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from models.read_csv import prediction_variables
from sklearn.ensemble import RandomForestClassifier

path = '/home/zack/Desktop/ML/AI_CLASS/Data/reprocessedHungarianData'
data = split_data(path, prediction_variables)

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
def predict_rf_model():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    rf_accuracy = accuracy_score(prediction, test_y)
    return rf_accuracy


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

decision_tree_model()

print("----GridSearchCV----")
decision_trees_grid_search()

print("----- Random Forest Classifier-----")
print(predict_rf_model())

'''
Accuracy Score:  0.584269662921

----- Random Forest Classifier------
0.707865168539 (Without Parameter Tuning)


----GridSearchCV----
Best Parameters: 
{'min_samples_leaf': 5, 'max_depth': 15, 'criterion': 'gini', 'splitter': 'random'}
Best Accuracy Score: 
0.70243902439
'''

'''
    To do:
    1. Play around with parameters to improve accuracy
    2. Hold out set to reduce overfitting.
    3. Find the best prediction variables from all >76 features/prediction variables
      
'''


