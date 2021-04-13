### Handwriting Recognition using K-Means
import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
print(digits)
print(digits.DESCR)
print(digits.data)
print(digits.target)

# Visualize the image at index 100:
plt.gray()
plt.matshow(digits.images[100])
plt.show()
print(digits.target[100])

# Visualize 64 sample images
# Figure size (width, height)
fig = plt.figure(figsize=(6, 6))
 
# Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
 
# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # Label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
plt.show()
# K-Mean clustering
from sklearn.cluster import KMeans
model = KMeans()
model.fit(digits.data)
# Visualize all the centroids. Because data samples live in a 64-dimensional space, the centroids have values so they can be images!
fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

#Scikit-learn sometimes calls centroids “cluster centers”.
# for i in range(10):  # ERROR: IndexError: index 8 is out of bounds for axis 0 with size 8
#   # Initialize subplots in a grid of 2X5, at 1+1th position
#   ax = fig.add_subplot(2, 5, 1 + i)
#   # Display images
#   ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap=plt.cm.binary) #The cluster centers should be a list with 64 values (0-16). Here, we are making each of the cluster centers into an 8x8 2D array.
# plt.show()

# If you want to see another example that visualizes the data clusters and their centers using K-means, check out the sklearn‘s own example: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

# Testing model:
new_samples = np.array([
[0.00,0.00,0.00,3.36,7.62,7.40,1.30,0.00,0.00,0.00,0.00,4.58,7.25,7.09,5.11,0.00,0.00,0.00,0.00,4.58,6.86,5.03,6.64,0.00,0.00,0.00,0.00,4.58,6.86,4.58,6.87,0.00,0.00,0.00,0.00,4.96,6.63,4.57,6.87,0.00,0.00,0.00,0.00,5.34,6.10,5.87,6.57,0.00,0.00,0.00,0.00,5.34,6.33,7.09,4.50,0.00,0.00,0.00,0.00,4.04,7.62,7.62,2.36,0.00],
[0.00,0.00,0.61,6.02,6.86,5.04,0.00,0.00,0.00,0.00,3.74,7.55,5.49,7.55,1.37,0.00,0.00,0.00,5.34,6.25,2.06,7.62,3.05,0.00,0.00,0.00,5.19,6.79,3.05,7.62,4.04,0.00,0.00,0.00,1.83,7.55,7.62,7.62,4.58,0.00,0.00,0.00,0.31,0.76,0.76,6.10,5.72,0.00,0.00,0.00,5.72,6.56,1.91,4.96,7.09,0.00,0.00,0.00,1.83,6.78,7.62,7.62,7.55,0.46],
[0.00,0.00,0.00,0.00,0.00,0.61,1.53,0.30,0.00,0.00,0.00,0.61,5.49,7.55,7.62,5.95,0.00,0.00,0.00,2.97,7.62,4.66,3.89,7.62,0.00,0.00,0.00,0.61,3.58,0.38,4.88,7.40,0.00,0.00,0.00,0.00,0.00,2.98,7.62,3.89,0.00,0.00,0.00,0.00,2.14,7.47,5.26,0.08,0.00,0.00,0.00,1.60,7.17,6.18,0.31,0.00,0.00,0.00,3.28,7.17,7.62,4.58,1.07,0.00],
[0.00,0.00,0.00,0.00,3.96,1.37,0.00,0.00,0.00,0.00,0.00,0.31,7.62,3.81,0.00,0.00,0.00,0.00,0.00,1.15,7.62,3.81,0.00,0.00,0.00,0.00,0.00,5.79,7.62,3.81,0.00,0.00,0.00,0.00,0.00,2.82,7.62,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.63,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.63,3.81,0.00,0.00,0.00,0.00,0.00,0.00,7.62,3.81,0.00,0.00]
])
new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')


