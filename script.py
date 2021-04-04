## Linear regression
x = [1, 2, 3]
y = [5, 1, 3]
# Calculate Gradient Descent for Intercept
def get_gradient_at_b(x, y, m, b):
  diff = 0
  for i in range(len(x)):
    diff0 = y[i] - (m*x[i]+b)
    diff += diff0
  b_gradient = -2/(len(x))*diff
  return b_gradient

  # Calculate Gradient Descent for Slope
def get_gradient_at_m(x, y, m, b):
    diff = 0
    N = len(x)
    for i in range(N):
      y_val = y[i]
      x_val = x[i]
      diff += x_val*(y_val - ((m * x_val) + b))
    m_gradient = -2/N * diff
    return m_gradient

# Learning rate: find the gradients at b_current and m_current, and then return new b and m values that have been moved in that direction.
def step_gradient(x, y, b_current, m_current, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

b, m = step_gradient(months, revenue, 0, 0)

# Convergence: This graph shows how much the parameter b changed with each iteration of a gradient descent runner
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from data import bs, bs_000000001, bs_01
# bs_000000001: 1400 iterations of gradient descent on b with a learning rate of 0.000000001
# bs_01: 100 iterations of gradient descent on b with a learning rate of 0.01

iterations = range(1400)
plt.plot(iterations, bs)
plt.xlabel("Iterations")
plt.ylabel("b value")
plt.show()

# Very small step - linear - no converence
iterations = range(1400)
plt.plot(iterations, bs_000000001)
plt.xlabel("Iterations")
plt.ylabel("b value")
plt.show()

# Very big step - 'goc vuong' - no converence
iterations = range(100)
plt.plot(iterations, bs_01)
plt.xlabel("Iterations")
plt.ylabel("b value")
plt.show()

# Putting together
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

def gradient_descent(x, y, learning_rate, num_iterations):
  b = 0
  m = 0
  for i in range(num_iterations):
    b, m = step_gradient(b, m, x, y, learning_rate)
  return [b, m]
#Uncomment the line below to run your gradient_descent function
b, m = gradient_descent(months, revenue, 0.01, 1000)

#Uncomment the lines below to see the line you've settled upon!
y = [m*x + b for x in months]
plt.plot(months, revenue, "o")
plt.plot(months, y)
plt.show()

# Use Your Functions on Real Data
import codecademylib3_seaborn
from gradient_descent_funcs import gradient_descent
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heights.csv")

X = df["height"]
y = df["weight"]

b, m = gradient_descent(X, y, num_iterations=1000, learning_rate=0.0001)
y_predictions = [x*m+b for x in X]
plt.plot(X, y, 'o')
#plot your line here:
plt.plot(X, y_predictions, 'x')
plt.show()

# Apply scikit-learn
import codecademylib3_seaborn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

temperature = np.array(range(60, 100, 2))
temperature = temperature.reshape(-1, 1)
sales = [65, 58, 46, 45, 44, 42, 40, 40, 36, 38, 38, 28, 30, 22, 27, 25, 25, 20, 15, 5]
line_fitter = LinearRegression()
line_fitter.fit(temperature, sales)
sales_predict = line_fitter.predict(temperature)
plt.plot(temperature, sales, 'o')
plt.plot(temperature, sales_predict, 'x')
plt.show()