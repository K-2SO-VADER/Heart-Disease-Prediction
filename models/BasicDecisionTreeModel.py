# A Python library to help us read data easily
import pandas as pd
# Scikit Learn, A Python ML library that has an inbuilt dataset splitting function
from sklearn.model_selection import train_test_split
# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Import accuracy score function from Scikit Learn
from sklearn.metrics import accuracy_score

'''
    1. Open Hungarian Data using Excel (delimit using space) and add header row. (The last row is the predicted attribute)
    2. Use Pandas to read it to the Program
    3. Print the columns
'''
data = pd.read_csv('/home/zack/Desktop/ML/AI_CLASS/Data/reprocessedHungarianData', header=0, delimiter=' ')
data_cols = list(data.columns)

print(data_cols)

# split the data into train and test sets 60% and 40% respectively
train, test = train_test_split(data , test_size=0.4)

# prediction variables: Columns that we are using to predict data
# Note: The last column is not included because that is the prediction
prediction_variables =  ['age' , 'sex' , 'cp' , 'threstbps' , 'chol' , 'fbs' , 'restecg', 'thalach' , 'exang' , 'oldpeak' , 'slope' , 'ca' , 'thal']

# train a Decision Tree Classifier and use default parameters
# Parameters affect the performance aka accuracy of your Machine Learning algorithm
clf = DecisionTreeClassifier(random_state=0)
# Train the model/classifier by passing prediction variables and prediction from test data
# train.num is the num column in train set.
clf.fit(train[prediction_variables] , train.num)
# predict column num in our training dataset based on prediction variables
decision_tree_predictions = clf.predict(test[prediction_variables])
# check accuracy: Compare predicted results vs actual results
print('Accuracy Score is: ' , accuracy_score(decision_tree_predictions , test.num))

# optional: Print actual predictions vs Predicted results
output = pd.DataFrame(data={"Actual" : test.num, "predicted" : decision_tree_predictions})
output.to_csv("DecisionTreeComparison.csv", index=False)

'''
1. Accuracy Score: 0.550847457627 (Let me know what you guys get)
2. This implementation uses reprocessed Hungarian data which has no missing values. 

Future Improvements: 
1. Improve Accuracy through Parameter tuning -- finding the best parameters for DecisionTrees
2. Try out other algorithms 
3. Divide Dataset into 3: Test, Training, Validation to reduce bias in real world dataset
4. Clean the other data and combine them; could improve the outcome.
'''




