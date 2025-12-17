import numpy as np

def sigmoid(z):
	"""
	Sigmoid activation function
	"""
	z = np.clip(z,-500,500) # numerical stability
	return 1 / (1 + np.exp(-z))