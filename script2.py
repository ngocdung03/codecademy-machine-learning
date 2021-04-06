## Multiple Linear Regression
import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")
df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]
y = df['rent']
# Make train-test sets:
random_state = 6
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Creat linear regression model:
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)

# Plug in the data from https://streeteasy.com/rental/2177438
# Sonny doesn't have an elevator so the 11th item in the list is a 0
sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
predict = mlr.predict(sonny_apartment)
print("Predicted rent: $%.2f" % predict)

# Scatter plot
lm = LinearRegression()
model=lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)
plt.scatter(y_test, y_predict)
plt.xlabel('Test outcome')
plt.ylabel('Predicted outcome')
plt.title('Test vs. predicted outcome')
plt.show()

# Print out the coefficients using
print(mlr.coef_)

# Use the .score() method from LinearRegression to find the mean squared error regression loss for the training and testing set.
print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))
 Print the coefficients again to see which ones are strongest.
print(lm.coef_)

# Print the coefficients again to see which ones are strongest.
print(lm.coef_)

# Remove some of the features that don’t have strong correlations and see if your scores improved!
x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]
y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
lm = LinearRegression()
model = lm.fit(x_train, y_train)
y_predict= lm.predict(x_test)
print("Train score:")
print(lm.score(x_train, y_train))
print("Test score:")
print(lm.score(x_test, y_test))