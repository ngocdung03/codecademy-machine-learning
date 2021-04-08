# Find the flag - decision tree
import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

flags = pd.read_csv("flags.csv", header=0)
print(flags.columns)
print(flags.head())

labels = flags[['Landmass']]
data = flags[["Red","Green","Blue","Gold","White","Black","Orange"]]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
tree = DecisionTreeClassifier(random_state=1)
tree.fit(train_data, train_labels)
print(tree.score(test_data, test_labels))

scores = []
for i in range(1,21):
  tree = DecisionTreeClassifier(random_state=1, max_depth=i)  #?why error without 'tree.' 
  tree.fit(train_data, train_labels)
  scores.append(tree.score(test_data, test_labels))
  print((tree.score(test_data, test_labels)))

# Graph accuracy
plt.plot(range(1,21), scores)
plt.show()
