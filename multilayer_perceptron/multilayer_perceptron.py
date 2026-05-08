import torch
import numpy as np

class MultilayerPerceptron:
    """
    Pure NumPy implementation of a single hidden layer MLP model, parameterized by thetas and bias.
    In other words, we keep track of layer weights instead of layers.
    
    params:
        data: dataset
        labels: dataset labels
        layers: configuration of neurons per layer e.g. [128, 256, 128] 3 layer MLP
        epsilon: non-linearity introduced to parameterization
        thetas: weights matrices
    """
    def __init__(self, data, labels, layers, epsilon):
        self.data = data
        self.labels = labels
        self.layers = layers
        self.epsilon = epsilon
        self.thetas = MultilayerPerceptron.initialize_thetas(self.layers)


    def train(self, unrolled_thetas):
        """
        Process:
            -> feedforward pass      (feedforward) 
            -> calculate gradients   (gradient_step)
            -> backpropagate         (backpropagate)
            -> repeat
        """

        return 


    def feedforward(self, ):
        """
        Runs data through all weights once.
        """

        

        return 


    @staticmethod
    def thetas_init(layers):
        """Randomly initialize thetas for each layer"""
        num_layers = len(layers)

        thetas = {}
        
        # no waits on output layer 
        for layer_idx in range(num_layers - 1):
            input_dim = layers[layer_idx]         # current layer
            output_dim = layers[layer_idx + 1]    # next layer 
            thetas[layer_idx] = np.random.rand(input_dim, output_dim + 1)  # +1 for bias neuron, where input layer has no bias

        return thetas
    

    @staticmethod
    def thetas_roll(unrolled_thetas, layers):
        """
        Unrolled thetas is a 1D array of variable length. Function rolls 1D array into len(layer)-D.
        """
        rolled_thetas = {}
        theta_shift = 0

        for layer_idx in range(len(layers) - 1):
            num_rows = layers[layer_idx]
            num_cols = layers[layer_idx + 1] + 1    # add 1 for bias

            total_thetas = num_rows * num_cols
            start_idx = theta_shift
            end_idx = theta_shift + total_thetas

            current_thetas = np.reshape(unrolled_thetas[start_idx:end_idx], (num_rows, num_cols), copy=False)
            rolled_thetas[layer_idx] = current_thetas

        return rolled_thetas


    @staticmethod
    def thetas_unroll(rolled_thetas):
        """
        Rolled thetas is an array of total layer length, and each element is an array of
        shape (N_in, N_out + 1) where N_in is the neurons of the previous layer, and N_out + 1 is current.
        """
        thetas = np.array([])

        for layer_idx in len(rolled_thetas):
            thetas.append(rolled_thetas[layer_idx].flatten())

        return