import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples to generate
N = 100

# Generate a base feature (x1) uniformly distributed between -10 and 10
x1 = np.random.uniform(-10, 10, N)

# Generate x2 to be almost the same as x1 (highly collinear)
# Adds a very small amount of noise
x2 = x1 + np.random.normal(0, 0.01, N)

# Generate x3 as a linear combination of x1 with small noise (also collinear)
x3 = 2*x1 + np.random.normal(0, 0.01, N)

# Construct target variable y as a linear combination:
# 1.5*x1 - 2.0*x2 (with x3 having 0 weight, despite being collinear)
y = 1.5*x1 - 2.0*x2 + 0.0*x3 + np.random.normal(0, 0.1, N)

# Put the generated features and target into a DataFrame
df = pd.DataFrame({'x_1': x1, 'x_2': x2, 'x_3': x3, 'y': y})

# Save the dataset to CSV for use in testing sparsity and collinearity handling
df.to_csv('LassoHomotopy/datasets/collinear_dataset.csv', index=False)