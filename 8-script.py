### Perceptron
## Representing a Perceptron
class Perceptron:
  def __init__(self, num_inputs=2, weights=[1,1]):
    self.num_inputs = num_inputs
    self.weights = weights

cool_perceptron = Perceptron()
print(cool_perceptron)
    
## Finding the weighted sum of the inputs.
class Perceptron:
  def __init__(self, num_inputs=2, weights=[2,1]):
    self.num_inputs = num_inputs
    self.weights = weights
    
  def weighted_sum(self, inputs):
    # create variable to store weighted sum
    weighted_sum = 0
    for i in range(self.num_inputs):
      weighted_sum += inputs[i]*self.weights[i]   # why error with self.num_inputs: ‘int’ object is not subscriptable
      # complete this loop
    return weighted_sum  
cool_perceptron = Perceptron()
print(cool_perceptron.weighted_sum([24, 55]))