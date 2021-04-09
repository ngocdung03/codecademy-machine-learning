##### Scikit-Learn cheatsheet - Introduction to ML
- Linear Regression:
```py
from sklearn.linear_model import LinearRegression
your_model = LinearRegression()
# Fit
your_model.fit(x_training_data, y_training_data)
# .coef_, .intercept_

# Predict
predictions = your_model.predict(your_x_data)
#.score(): return the coefficient of determination R squared
```
- Naive Bayes
```py
from sklearn.naive_bayes import MultinomialNB
your_model = MultinomialNB()
# Fit
your_model.fit(x_training_data, y_training_data)
# Predict
# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)
# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)
```
- K-nearest neighbors
```py
from sklearn.neigbors import KNeighborsClassifier
your_model = KNeighborsClassifier()
# Fit
your_model.fit(x_training_data, y_training_data)
# Predict
# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)
# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)
```
- K-means
```py
from sklearn.cluster import KMeans
your_model = KMeans(n_clusters=4, init='random')
# n_clusters: number of clusters to form and number of centroids to generate
# init: method for initialization. k-mean++:k-mean++[default], random: k-means
# random_state
# Fit
your_model.fit(x_training_data)
# Predict:
predictions = your_model.predict(your_x_data)
```
- Validating the model
```py
# Import and print accuracy, recall, precision, and F1 score:
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print(accuracy_score(true_labels, guesses))
print(recall_score(true_labels, guesses))
print(precision_score(true_labels, guesses))
print(f1_score(true_labels, guesses))
# Import and print the confusion matrix:
from sklearn.metrics import confusion_matrix
print(confusion_matrix(true_labels, guesses))
```
- Training sets and test sets
```py
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
# train_size: the proportion of the dataset to include in the train split
# test_size: the proportion of the dataset to include in the test split
# random_state
```
- We can scale the size of the step by multiplying the gradient by a *learning rate*.
- Convergence is when the loss stops changing (or changes very slowly) when parameters are changed.
- Finding the absolute best learning rate is not necessary for training a model. You just have to find a learning rate large enough that gradient descent converges with the efficiency you need, and not so large that convergence never happens.

##### Multiple Linear Regression
- StreetEasy dataset: https://www.codecademy.com/content-items/d19f2f770877c419fdbfa64ddcc16edc
    - manhattan.csv
    - brooklyn.csv
    - queens.csv
- When trying to evaluate the accuracy of our multiple linear regression model, one technique we can use is Residual Analysis.
- sklearn‘s linear_model.LinearRegression comes with a .score() method that returns the coefficient of determination R² of the prediction. 
- sklearn.tree: DecisionTreeClassifier class
    - when we built our tree from scratch, our data points contained strings like "vhigh" or "5more". When creating the tree using scikit-learn, it’s a good idea to map those strings to numbers
```py
classifier = DecisionTreeClassifier()
```
- Find accuracy:
```py
print(classifier.score(test_data, test_labels))
```
- Limitation:
    - Isn't always globally optimal
    - Greedy:  find the feature that will result in the largest information gain right now and split on that feature. We never consider the ramifications of that split further down the tree. 
    - potentially overfit -> prune the tree
- Flags dataset: http://archive.ics.uci.edu/ml/datasets/Flags

##### Random Forests
- Bagging: Every time a decision tree is made, it is created using a different subset, with replacement, of the points in the training set. 
    - Because we’re picking these rows with replacement, there’s no need to shrink our bagged training set from 1000 rows to 100. We can pick 1000 rows at random. 
    - Changing the features that we use: a randomly selected subset of features are considered as candidates for the best splitting feature.
    - If we have many features: A good rule of thumb is to randomly select the square root of the total number of features
- Random Forest in Scikit-learn: RandomForestClassifier()
- Data on income: https://archive.ics.uci.edu/ml/datasets/census%20income

##### K-means clustering
- Two questions arise:   
    - How many groups do we choose?
    - How do we define similarity? 
- The “Means” refers to the average distance of data to each cluster center, also known as the centroid, which we are trying to minimize.
- Iterative approach:
    - Place k random centroids for the initial clusters.
    - Assign data samples to the nearest centroid.
    - Update centroids based on the above-assigned data samples.
    - Repeat Steps 2 and 3 until convergence (when points don’t move between clusters and centroids stabilize).
- Argmin(distances) would return the index of the lowest corresponding distance
- Implementing K-Means: Scikit-Learn
```py
from sklearn.cluster import KMeans
model = KMeans(n_clusters = k)
model.fit(X)
model.predict(X)
```
- Evaluation:
    - Cross-tabulations (by Pandas library) enable you to examine relationships within the data that might not be readily apparent when analyzing total survey responses.
- Good clustering results in tight clusters, meaning that the samples in each cluster are bunched together. How spread out the clusters are is measured by inertia
    - Elbow method: choose an “elbow” in the inertia plot - when inertia begins to decrease more slowly.
```py
print(model.intertia_)
```
