import codecademylib3_seaborn
# Importing breast cancer data
from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])   #breast_cancer_data.data to see the data
print(breast_cancer_data.feature_names)

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)   #the first data point has a label of 0, so 0 is malignant
# Splitting the data
from sklearn.model_selection import train_test_split
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
print(len(training_data))
print(len(training_labels))

# Create KNeighborsClassifier and test for accuracy
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(training_data, training_labels)

# Find accuracy
print(classifier.score(validation_data, validation_labels))

# Accuracy with different k
accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracy = classifier.score(validation_data, validation_labels)
  accuracies.append(accuracy)

# Graph k versus accuracy
import matplotlib.pyplot as plt
k_list = range(1, 101)
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
