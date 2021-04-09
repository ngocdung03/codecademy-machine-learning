def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Every string has an extra space at the start
income_data = pd.read_csv("income.csv", header=0, delimiter = ', ')
#print(income_data.iloc[0])

# Separate label with data
labels = income_data[['income']]

# Random forests can’t use columns that contain Strings
income_data['sex-int'] = income_data['sex'].apply(lambda row: 0 if row =="Male" else 1)

#print(income_data['native-country'].value_counts())
income_data['country-int'] = income_data['native-country'].apply(lambda row:0 if row=='United-States' else 1)  #When mapping Strings to numbers like this, it is important to make sure that continuous numbers make sense. 
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

forest = RandomForestClassifier(random_state=1)
forest.fit(train_data, train_labels) # Random forests can’t use columns that contain Strings 
print(forest.score(test_data, test_labels))

# Create a tree.DecisionTreeClassifier, train it, test is using the same data, and compare the results to the random forest. When does the random forest do better than the single tree? When does a single tree do just as well as the forest?
# After calling .fit() on the forest, print forest.feature_importances_. This will show you a list of numbers where each number corresponds to the relevance of a column from the training data. Which features tend to be more relevant?
# Use some of the other columns that use continuous variables, or transform columns that use strings!




