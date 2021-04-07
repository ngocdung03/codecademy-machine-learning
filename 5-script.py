## Logistic regression
import numpy as np
from exam import hours_studied, calculated_coefficients, intercept

# Create your log_odds() function here
def log_odds(features, coefficients, intercept):
  return np.dot(features, coefficients) + intercept
# Calculate the log-odds for the Codecademy University data here
calculated_log_odds = log_odds(hours_studied, calculated_coefficients, intercept)

## Sigmoid function
import codecademylib3_seaborn
import numpy as np
from exam import calculated_log_odds

# Create your sigmoid function here
def sigmoid(z):
  denominator = 1+np.exp(-z)
  return 1/denominator
# Calculate the sigmoid of the log-odds here
probabilities = sigmoid(calculated_log_odds)

## Log Loss
import numpy as np
from exam import passed_exam, probabilities, probabilities_2

# Function to calculate log-loss
def log_loss(probabilities,actual_class):
  return np.sum(-(1/actual_class.shape[0])*(actual_class*np.log(probabilities) + (1-actual_class)*np.log(1-probabilities)))

# Print passed_exam here
print(passed_exam)

# Calculate and print loss_1 here
loss_1 = log_loss(probabilities, passed_exam)
print(loss_1)

# Calculate and print loss_2 here
# probabilities_2 contains the calculated probabilities of the students passing the exam with the coefficient for hours_studied set to 0
loss_2 = log_loss(probabilities_2, passed_exam)
print(loss_2)

## Classification thresholding
def predict_class(features, coefficients, intercept, threshold):
  calculated_log_odds = log_odds(features, coefficients, intercept)
  probabilities = sigmoid(calculated_log_odds)
  return np.where(probabilities >= threshold, 1, 0)  #Return 1 for all values within probabilities equal to or above threshold, and 0 for all values below threshold

final_results = predict_class(hours_studied, calculated_coefficients, intercept, 0.5)

## Practice with Codecademy data
import numpy as np
from sklearn.linear_model import LogisticRegression
from exam import hours_studied_scaled, passed_exam, exam_features_scaled_train, exam_features_scaled_test, passed_exam_2_train, passed_exam_2_test, guessed_hours_scaled
# Create and fit logistic regression model here
model = LogisticRegression()
model.fit(hours_studied_scaled, passed_exam)
# Save the model coefficients and intercept here
calculated_coefficients = model.coef_
intercept = model.intercept_
print(calculated_coefficients)
print(intercept)
# Predict the probabilities of passing for next semester's students here
passed_predictions = model.predict_proba(guessed_hours_scaled)
# Create a new model on the training data with two features here
model_2 = LogisticRegression()
model_2.fit(exam_features_scaled_train, passed_exam_2_train)
# Predict whether the students will pass here
passed_predictions_2 = model_2.predict(exam_features_scaled_test)

## Feature importance
# Assign and update coefficients
coefficients = model_2.coef_
coefficients = coefficients.tolist()[0]  #convert the array into a list 

# Plot bar graph
plt.bar([1,2],coefficients)
plt.xticks([1,2],['hours studied','math courses taken'])
plt.xlabel('feature')
plt.ylabel('coefficient')
plt.show()
