import numpy as np
from utils import sigmoid, sigmoid_gradient

class MultilayerPerceptron:
    """
    Pure NumPy implementation of a single hidden layer MLP model, parameterized by thetas and bias.
    In other words, we keep track of layer weights instead of layers. As such, the model initializes
    with training data and labels.
    
    params:
        data: dataset
        labels: dataset labels
        layers: configuration of neurons per layer e.g. [128, 256, 128] 3 layer MLP
        epsilon: non-linearity introduced to parameterization
        thetas: weights matrices
    """

    def __init__(self, features, labels, layers, epsilon):
        self.features = np.asarray(features, dtype=np.float64)
        self.labels = np.asarray(labels).reshape(-1)
        self.layers = layers
        self.epsilon = epsilon
        self.thetas = MultilayerPerceptron.thetas_init(layers, epsilon)

    def train(self, epochs, learning_rate):
        """
        Process:
            -> feedforward pass      (feedforward) 
            -> calculate gradients   (gradient_step)
            -> backpropagate         (backpropagate)
            -> repeat
        """

        cost_per_epoch = {}
        num_classes = self.layers[-1]
        y_onehot = MultilayerPerceptron.one_hot_encode(self.labels, num_classes)

        for epoch in range(epochs):
            pred_probs, layer_logits, layer_activations = MultilayerPerceptron.feedforward(
                self.features, self.thetas, self.layers, train=True
            )

            cost_per_epoch[epoch] = MultilayerPerceptron.get_cost(y_onehot, pred_probs)

            theta_deltas = MultilayerPerceptron.backward_pass(
                y_onehot, self.thetas, self.layers, layer_logits, layer_activations
            )

            for layer_idx in self.thetas:
                self.thetas[layer_idx] -= learning_rate * theta_deltas[layer_idx]

            print("Cost ", cost_per_epoch[epoch])
        return cost_per_epoch


    def predict(self, data):
        """
        Model inference using a single forward pass and pre-trained weights.
        """
        num_samples = data.shape[0]

        class_probs, _, _ = MultilayerPerceptron.feedforward(
            data, self.thetas, self.layers, train=False
        )
        # reshapes [1, N] matrix (predicted class) into [N, 1] matrix 
        return np.argmax(class_probs, axis=1).reshape((num_samples, 1))


    @staticmethod
    def one_hot_encode(labels, num_classes):
        encoded_labels = []
        for label in labels:
            one_hot_label = np.zeros(num_classes)
            one_hot_label[label] = 1

            encoded_labels.append(one_hot_label)

        return np.array(encoded_labels)


    @staticmethod
    def get_cost(labels_onehot, preds):
        assert labels_onehot.shape == preds.shape
        eps = 1e-9
        p = np.clip(preds, eps, 1 - eps)
        term = -(labels_onehot * np.log(p) + (1 - labels_onehot) * np.log(1 - p))
        return float(np.mean(np.sum(term, axis=1)))


    @staticmethod
    def feedforward(features, thetas, layers, train=False):
        num_layers = len(layers)
        num_samples = features.shape[0]

        layer_logits = {}
        layer_activations = {}

        layer_activations[0] = features # input layer activations are always input features
        out_activations = features  # populate first layer activations to start feeding forward

        for layer_idx in range(num_layers - 1):
            theta = thetas[layer_idx]
            biased_in_activations = np.hstack((np.ones((num_samples, 1)), out_activations))
            logits = biased_in_activations @ theta.T
            out_activations = sigmoid(logits)

            if train:
                layer_logits[layer_idx + 1] = logits
                layer_activations[layer_idx + 1] =out_activations

        return out_activations, layer_logits, layer_activations

    @staticmethod
    def backward_pass(y_onehot, thetas, layers, layer_logits, layer_activations):
        num_samples = y_onehot.shape[0]
        last_layer_idx = len(layers) - 1

        deltas = {}
        layer_gradients = {}

        deltas[last_layer_idx] = layer_activations[last_layer_idx] - y_onehot

        for layer_idx in range(last_layer_idx - 1, -1, -1):
            biased_activations = np.hstack((np.ones((num_samples, 1)), layer_activations[layer_idx]))
            layer_gradients[layer_idx] = (deltas[layer_idx + 1].T @ biased_activations) / num_samples

            if layer_idx == 0:
                break

            
            deltas[layer_idx] = (deltas[layer_idx + 1] @ thetas[layer_idx][:, 1:]) * sigmoid_gradient(layer_logits[layer_idx])

        return layer_gradients


    @staticmethod
    def thetas_init(layers, epsilon):
        thetas = {}
        num_layers = len(layers)

        for layer_idx in range(num_layers - 1):
            input_dim = layers[layer_idx]
            output_dim = layers[layer_idx + 1]
            thetas[layer_idx] = (
                np.random.rand(output_dim, input_dim + 1) * 2 * epsilon - epsilon
            )

        return thetas

    @staticmethod
    def thetas_roll(unrolled_thetas, layers):
        """
        Unrolled thetas is a 1D array of variable length. Function rolls 1D array into len(layer)-D.
        """
        rolled_thetas = {}
        theta_shift = 0

        for layer_idx in range(len(layers) - 1):
            num_rows = layers[layer_idx + 1]    # we can't change theta for input layer so skip it
            num_cols = layers[layer_idx ] + 1 # add bias column
            rolled_thetas[layer_idx] = np.reshape(
                unrolled_thetas[theta_shift: theta_shift + num_rows * num_cols],
                (num_rows, num_cols),
                copy=False,
            )
            theta_shift += num_rows * num_cols

        return rolled_thetas

    @staticmethod
    def thetas_unroll(rolled_thetas):
        chunks = []
        for layer_idx in sorted(rolled_thetas.keys()):
            chunks.append(rolled_thetas[layer_idx].ravel())
        return np.concatenate(chunks, axis=0)
