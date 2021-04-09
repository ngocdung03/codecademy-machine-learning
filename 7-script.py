### K-Means Clustering
## Iris dataset
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy
import pandas as pd

iris = datasets.load_iris()
print(iris.data)
# Since the datasets in sklearn datasets are used for practice, they come with the answers (target values) in the target key:
print(iris.target)  # Ground truth, the number corresponding to the flower that we are trying to learn.
# Description of data
print(iris.DESCR)

## Visualize before K-means
# Store iris.data
samples = iris.data
# Create x and y
x = samples[:, 0]
y = samples[:, 1]
# Plot x and y
plt.scatter(x, y, alpha=0.5)
# Show the plot
plt.show()

## Step 1: Place k random centroids for the initial clusters.
x = samples[:,0]
y = samples[:,1]

# Number of clusters
k = 3
# Generate random values in two lists:
# Create x coordinates of k random centroids
centroids_x = np.random.uniform(min(x), max(x), k)
# Create y coordinates of k random centroids
centroids_y = np.random.uniform(min(y), max(y), k)
# Create centroids array
centroids = np.array(list(zip(centroids_x, centroids_y)))
print(centroids)
# Make a scatter plot of x, y
plt.scatter(x, y)
# Make a scatter plot of the centroids
plt.scatter(centroids_x, centroids_y)
# Display plot
plt.show()

## Step 2: Assign data samples to the nearest centroid.
sepal_length_width = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

## Step 2: Assign samples to nearest centroid
# Distance formula
def distance(a, b):
  distance = 0
  for i in range(len(a)):
    distance += (a[i] - b[i])**2
  return distance**0.5
  
# Cluster labels for each point (either 0, 1, or 2)
labels = np.zeros(len(samples))
# Distances to each centroid
distances = np.zeros(k)
# Assign to the closest centroid
for i in range(len(samples)): 
  distances[0] = distance(sepal_length_width[i], centroids[0]) 
  distances[1] = distance(sepal_length_width[i], centroids[1]) 
  distances[2] = distance(sepal_length_width[i], centroids[2]) 
  cluster = np.argmin(distances)
  labels[i] = cluster
# Print labels
print(labels)

## Step 3: Update centroids
centroids_old = deepcopy(centroids)
# create an array named points where we get all the data points that have the cluster label i then get mean
for i in range(k):
  points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] ==i]
  centroids[i] = np.mean(points, axis=0) #the default is to compute the mean of the flattened array
print(centroids_old)
print(centroids)

## Step 4: Repeatedly execute Step 2 and 3 until the centroids stabilize (convergence).
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

def distance(a, b):
  one = (a[0] - b[0]) ** 2
  two = (a[1] - b[1]) ** 2
  distance = (one + two) ** 0.5
  return distance

# To store the value of centroids when it updates
centroids_old = np.zeros(centroids.shape)

# Cluster labeles (either 0, 1, or 2)
labels = np.zeros(len(samples))

distances = np.zeros(3)

## Repeatedly execute Step 2 and 3 until the centroids stabilize (convergence).
# Initialize error:
error = np.zeros(3)
for i in range(len(error)):
  error[i] = distance(centroids[i], centroids_old[i])
# Repeat Steps 2 and 3 until convergence:
while error.all() !=0:  #? while loop because don't know iterate how many time?
  # Step 2: Assign samples to nearest centroid
  for i in range(len(samples)):
    distances[0] = distance(sepal_length_width[i], centroids[0])
    distances[1] = distance(sepal_length_width[i], centroids[1])
    distances[2] = distance(sepal_length_width[i], centroids[2])
    cluster = np.argmin(distances)
    labels[i] = cluster
  # Step 3: Update centroids
  centroids_old = deepcopy(centroids)
  for i in range(3):
    points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
    centroids[i] = np.mean(points, axis=0)
  for i in range(len(error)):
    error[i] = distance(centroids[i], centroids_old[i])

colors = ['r', 'g', 'b']
for i in range(k):
  points = np.array([sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]) #without np.array(), TypeError: list indices must be integers or slices, not tuple
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)
# Visualizing all the points in each of the labels a different color
plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)
 
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
 
plt.show() 

## Implementing K-Means: Scikit-Learn
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

# From sklearn.cluster, import KMeans class
from sklearn.cluster import KMeans
iris = datasets.load_iris()
samples = iris.data

# Use KMeans() to create a model that finds 3 clusters
model = KMeans(n_clusters = 3)
# Use .fit() to fit the model to samples
model.fit(samples)
# Use .predict() to determine the labels of samples 
labels = model.predict(samples)
print(labels)

## Predict on new data
# Store the new Iris measurements
new_samples = np.array([[5.7, 4.4, 1.5, 0.4],
   [6.5, 3. , 5.5, 0.4],
   [5.8, 2.7, 5.1, 1.9]])
print(new_samples)
# Predict labels for the new_samples
new_labels = model.predict(new_samples)
print(new_labels)

# Make a scatter plot of x and y and using labels to define the colors
x = samples[:, 0]
y = samples[:, 1]
plt.scatter(x, y, colors=labels, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

## Evaluation:
species = np.chararray(target.shape, itemsize=150)
 
for i in range(len(samples)):
  if target[i] == 0:
    species[i] = 'setosa'
  elif target[i] == 1:
    species[i] = 'versicolor'
  elif target[i] == 2:
    species[i] = 'virginica'
df = pd.DataFrame({'labels': labels, 'species': species})
print(df)
# Perform cross-tabulation:
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

## Find out the optimal number of clusters:
num_clusters = list(range(1, 9))
inertias = []
for num_cluster in num_clusters:
  model = KMeans(n_clusters = num_cluster)
  model.fit(samples)
  inertias.append(model.inertia_)
plt.plot(num_clusters, inertias, '-o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

## Try it on your own
digits = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)
# Note that if you download the data like this, the data is already split up into a training and a test set, indicated by the extensions .tra and .tes. You’ll need to load in both files.