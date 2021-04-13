### Predict Titanic Survival
import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
#print(passengers.head())

# Given the saying, “women and children first,” Sex and Age seem like good features to predict survival
# Update sex column to numerical
#passengers['Sex'] = 1 if passengers['Sex']=='female' else 0

passengers['Sex'] = passengers['Sex'].map({'female':1, 'male':0})

# Fill the nan values in the age column
print(passengers['Age'].values)
#passengers[passengers['Age'] == nan] = np.mean(passengers['Age'])
passengers['Age'].fillna(value = passengers['Age'].mean(), inplace=True)
print(passengers['Age'].value_counts())

# Create a first class column
passengers['FirstClass'] = [1 if i==1 else 0 for i in passengers['Pclass']] 
# Create a second class column
passengers['SecondClass'] = [1 if i==2 else 0 for i in passengers['Pclass']] 

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
training_data, testing_data, training_labels, testing_labels = train_test_split(features, survival)

# Since sklearn‘s Logistic Regression implementation uses Regularization, scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
training_data = scaler.fit_transform(training_data)
testing_data = scaler.transform(testing_data)

# Create and train the model
model = LogisticRegression()
model.fit(training_data, training_labels)

# Score the model on the train data
print(model.score(training_data,training_labels))

# Score the model on the test data
print(model.score(testing_data,testing_labels))

# Analyze the coefficients
print(model.coef_)

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([1.0,28,0.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
