import numpy as np

"""
Activation functions - each function assumes variable length and dimension inputs
"""

def sigmoid(features):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-features))

def relu(features):
    """ReLU activation function"""
    return np.maximum(0, features)
