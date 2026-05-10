import torch
import numpy as np
import functio

from utils.activations import sigmoid

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
    def __init__(self, features, labels, layers, epsilon):
        self.features = features
        self.labels = labels
        self.layers = layers
        self.epsilon = epsilon
        self.thetas = MultilayerPerceptron.thetas_init(self.layers)


    def train(self, epochs, learning_rate):
        """
        Process:
            -> feedforward pass      (feedforward) 
            -> calculate gradients   (gradient_step)
            -> backpropagate         (backpropagate)
            -> repeat
        """

        cost_per_epoch = {}

        for epoch in epochs:
            pred_probs, layer_logits, layer_activations = MultilayerPerceptron.feedforward(
                self.features, 
                self.thetas, 
                self.layers
            ) 

            cost_per_epoch[epoch] = MultilayerPerceptron.get_cost(
                self.labels, 
                pred_probs
            )

            deltas = MultilayerPerceptron.backwards_pass(
                self.labels,
                self.thetas, 
                self.layers, 
                layer_logits, 
                layer_activations
            )

            unrolled_thetas = MultilayerPerceptron.thetas_unroll(self.thetas)
            unrolled_deltas = MultilayerPerceptron.thetas_unroll(deltas)

            self.thetas = MultilayerPerceptron.thetas_roll(
                (unrolled_thetas - learning_rate * unrolled_deltas), self.layers
            )

        return cost_per_epoch
    
    
    def predict(self, data):
        """
        Model inference using a single forward pass and pre-trained weights.
        """
        num_samples = data.shape[0]

        predictions = MultilayerPerceptron.feedforward(data, self.thetas, self.layers, train=False)

        return np.argmax(predictions, axis=1).reshape((num_samples, 1))


    @staticmethod
    def get_cost(labels, preds):
        """
        For the sake of simplicity, we're implementing binary cross-entropy loss
        """
        assert labels.shape == preds.shape

        running_cost = 0.
        for sample_idx in range(labels.shape[0]):
            true_label = labels[sample_idx]
            pred_label = np.argmax(preds[sample_idx])

            running_cost += true_label * np.log(pred_label) + (1 - true_label) * np.log(1 - pred_label)

        return -(running_cost / labels.shape[0])


    @staticmethod
    def feedforward(features, thetas, layers, train=False):
        """
        Runs data through all weights once.
        """
        num_layers = len(layers)
        num_samples = features.shape()[0]
        in_activations = features

        # Intermediate values used for backpropagation
        layer_logits = {}
        layer_activations = {}

        for layer_idx in range(num_layers - 1):
            theta = thetas[layer_idx]
            logits = in_activations @ theta.T
            activations = sigmoid(logits)
            
            if train:
                layer_logits[layer_idx] = layer_logits
                layer_activations[layer_idx] = activations

            out_activations = np.hstack(np.ones(num_samples, 1), activations)
            in_activations = out_activations

        return in_activations[:, 1:], layer_logits, layer_activations
    

    @staticmethod
    def backward_pass(labels, thetas, layers, layer_logits, layer_activations):
        """
        Backwards pass to update gradients for each parameter.
        """
        num_samples = labels.shape[0]
        num_layers = len(layers)

        deltas = {}

        for sample_idx in range(num_samples):
            delta = {}

            # getting gradient from model output a.k.a. last layer activation values
            last_layer_activation = layer_activations[-1][sample_idx]
            delta[num_layers] = last_layer_activation - labels[sample_idx] 

            # pass gradients backwards via chain rule :D
            for layer_idx in range(num_layers - 1, 0, -1):
                delta_next = delta[layer_idx + 1]
                delta[layer_idx] = (delta_next @ thetas[layer_idx][sample_idx]) * sigmoid_gradient(layer_logits[layer_idx][sample_idx])
            
        for layer_idx in range(num_layers - 1):
            layer_delta = delta[layer_idx + 1].T @ thetas[layer_idx]    # numpy will handle the multiplications along y-axis
            deltas[layer_idx] += layer_delta

        return deltas


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