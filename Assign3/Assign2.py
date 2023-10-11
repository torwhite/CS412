import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp

# Define the function
def f(x, lmbda):
    if x < 0:
        return 0
    else:
        return (exp(-lmbda) * lmbda**x) / factorial(x)

# Values of lambda
lambdas = [0.5, 1.0, 2.0]

# Generate x values
x_values = np.arange(0, 10, 1)

# Create subplots for each lambda
plt.figure(figsize=(12, 6))
for lmbda in lambdas:
    y_values = [f(x, lmbda) for x in x_values]
    plt.plot(x_values, y_values, label=f'λ = {lmbda}')

plt.xlabel('x')
plt.ylabel('f(x; λ)')
plt.title('Probability Mass Function for Poisson Distribution')
plt.legend()
plt.grid(True)
plt.show()