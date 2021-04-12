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
##### K-Means++
- In the traditional K-Means algorithms, the starting postitions of the centroids are intialized completely randomly. This can result in *suboptimal clusters*.
-  K-Means++ algorithm replaces Step 1 of the K-Means algorithm and adds the following:
    - 1.1. The first cluster centroid is randomly picked from the data points.
    - 1.2 For each remaining data point, the distance from the point to its nearest cluster centroid is calculated.
    - 1.3 The next cluster centroid is picked according to a probability proportional to the distance of each point to its nearest cluster centroid. This makes it likely for the next cluster centroid to be far away from the already initialized centroids.
    - Repeat 1.2 - 1.3 until k centroids are chosen
- K-Means++ using Scikit-Learn: is actually default in sci-kit learn: `test=KMeans(init='k-means++, n_clusters=6)`

##### Perceptrons
- perceptron is an artificial neuron that can make a simple decision. Let’s implement one from scratch in Python!
- three main components:
    - Inputs: Each input corresponds to a feature.
    - Weights: Each input also has a weight which assigns a certain amount of importance to the input.
    - Output
- training error = actual label - predicted  label
- The Perceptron Algorithm: optimally tweak the weights and nudge the perceptron towards zero error: weight = weight + (error*input)
- Bias weight: there are times when a minor adjustment is needed for the perceptron to be more accurate. 
    - Consider 2 small changes: Add a 1 to the set of inputs (now there are 3 inputs instead of 2) OR Add a bias weight to the list of weights (now there are 3 weights instead of 2).
    ```py
    weighted sum = x1w1 + x2w2 + ... + xnwn + 1*wb
    ```
- Visualizing perceptron by a line:
    - slope = -self.weights[0]/self.weights[1]
    - intercept = -self.weights[2]/self.weights[1]
- Non-linear decision boundary: By increasing the number of features and perceptrons, we can give rise to the Multilayer Perceptrons, also known as Neural Networks, which can solve much more complicated problems.

##### Minimax
- Tic-tac-toe: An essential step in the minimax function is evaluating the strength of a leaf. If the game gets to a certain leaf, we want to know if that was a better outcome for player "X" or for player "O".
    - evaluation function: a leaf where player "X" wins evaluates to a 1, a leaf where player "O" wins evaluates to a -1, and a leaf that is a tie evaluates to 0.
    - First, we need to detect whether a board is a leaf — if either player has won, or if there are no more open spaces
    - State of the board: X won, O won, or tie
- ["Minimax algorithm - evaluating leaves.docx"]
- One of the central ideas behind the minimax algorithm is the idea of exploring future hypothetical board states.

##### Advanced Minimax
- There are games, like Chess, that have much larger trees. There are 20 different options for the first move in Chess, compared to 9 in Tic-Tac-Toe. On top of that, the number of possible moves often increases as a chess game progresses. Traversing to the leaves of a Chess tree simply takes too much computational power.
- Being able to stop before reaching the leaves is critically important for the efficiency of this algorithm. It could take literal days to reach the leaves of a game of chess. Stopping after only a few levels limits the algorithm’s understanding of the game, but it makes the runtime realistic.
- we’ll add another parameter to our function called depth. Every time we make a recursive call, we’ll decrease depth by 1 like so:
```py
def minimax(input_board, minimizing_player, depth):
  # Base Case
  if game_is over(input_bopard):
    return ...
  else:
    # …
    # Recursive Call
    hypothetical_value = minimax(new_board, True, depth - 1)[0]
 ```
 - Alpha-beta pruning: In order to traverse farther down the tree without dramatically increasing the runtime: ignore parts of the tree that we know will be dead ends.
    - alpha keeps track of the minimum score the maximizing player can possibly get. It starts at negative infinity and gets updated as that minimum score increases.
    - beta represents the maximum score the minimizing player can possibly get. It starts at positive infinity and will decrease as that maximum possible score decreases.
    - For any node, if alpha is greater than or equal to beta, that means that we can stop looking through that node’s children.