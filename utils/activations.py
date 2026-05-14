import numpy as np

"""
Helper functions for activation functions and their derivatives.
"""

def sigmoid(features):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-features))

def sigmoid_gradient(features):
    return sigmoid(features) * (1 - sigmoid(features))

def relu(features):
    """ReLU activation function"""
    return np.maximum(0, features)
