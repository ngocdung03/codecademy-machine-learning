# Linear regression
import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")
# print(df.head())

# For now, we care about the total production of honey per year. Use the .groupby() method provided by pandas to get the mean of totalprod per year.
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
# prod_per_year = pd.DataFrame(prod_per_year)
# print(prod_per_year.year)

# Create a variable called X that is the column of years in this prod_per_year DataFrame. We will need to reshape it
X = prod_per_year['year']
X = X.values.reshape(-1,1)
print(X)
y = prod_per_year['totalprod']
plt.scatter(X, y)

regr = linear_model.LinearRegression()
regr.fit(X,y)
print(regr.coef_)
print(regr.intercept_)
# y_predict = regr.predict(X)
# plt.plot(X, y_predict)
# plt.show()

# Let’s predict what the year until 2050 may look like in terms of honey production.
X_future = np.array(range(2013,2051))
# You can think of reshape() as rotating this array. Rather than one big row of numbers, X_future is now a big column of numbers — there’s one number in each row.
X_future = X_future.reshape(-1, 1)
# reshape() is a little tricky! It might help to print out X_future before and after reshaping.
print(X_future)

# Create a list called future_predict that is the y-values that your regr model would predict for the values of X_future.
future_predict = regr.predict(X_future)
plt.scatter(X_future, future_predict)
plt.show()