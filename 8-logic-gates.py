## Perceptron Logic Gates
# logic-gates.jpg
import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
# AND gate
data = [[0, 0], [1, 0], [0, 1], [1, 1]]
labels = [0 , 0, 0, 1]
x = [point[0] for point in data]
y = [point[1] for point in data]
# plt.scatter(x, y, c=labels)
# plt.show()
classifier = Perceptron(max_iter = 40)
classifier.fit(data, labels)
print(classifier.score(data, labels))  #1.00

# XOR gate
labels_x = [0 , 1, 1, 0]
# plt.scatter(x, y, c=labels_x)
# plt.show()
classifier_x = Perceptron(max_iter = 40)
classifier_x.fit(data, labels_x)
# print(classifier.score(data_x, labels_x))  #0.25

# OR gate
labels_o = [0 , 1, 1, 1]
# plt.scatter(x, y, c=labels_o)
# plt.show()
classifier_o = Perceptron(max_iter = 40)
classifier_o.fit(data, labels_o)
# print(classifier.score(data, labels_o))  #0.5

# Visualizing the perceptron
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))
# heat map that reveals the decision boundary.
x_values = np.linspace(0, 1, 100)  #list of 100 evenly spaced decimals between 0 and 1
y_values = np.linspace(0, 1, 100)
# Possible combinations for x and y values
point_grid = list(product(x_values, y_values))

distances = classifier.decision_function(point_grid)
print(distances)
abs_distances = [abs(value) for value in distances]
# Heatmap
# first turn abs_distances from 1x10000 into 100x100 array
distances_matrix = np.reshape(abs_distances, (100, 100))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap)  #put a legend on the heat map
plt.show()  #purple line is the decision boundary
# Change your labels back to representing an OR gate. Where does the decision boundary go?
# Change your labels to represent an XOR gate. Remember, this data is not linearly separable. Where does the decision boundary go?