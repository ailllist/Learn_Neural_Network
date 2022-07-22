import numpy as np

input_data = np.array([1, 1])
bias = 0.3
weights = np.array([0.5, 0])

signal_total = np.dot(input_data, weights) + bias

result = max(0, signal_total)
print(bool(result), result)