import numpy as np
from typing import Sequence, Callable, Tuple

# Test comment

# Define activation functions.
def sigmoid(x: Sequence):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: Sequence):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x: Sequence):
    return np.tanh(x)

def tanh_derivative(x: Sequence):
    return 1 - tanh(x) ** 2

def relu(x: Sequence):
    return np.maximum(0, x)

def relu_derivative(x: Sequence):
    return np.where(x > 0, 1, 0)

def softmax(x: Sequence):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x: Sequence):
    return 1

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    return loss

def cross_entropy_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return y_pred - y_true

class Layer:
    def __init__(self, num_inputs:int, num_neurons:int, activation:str) -> None:
        # Define layer input and output
        self.input = None
        self.output = None
        
        # Initialise weights using He Initialisation
        self.weights = np.random.randn(num_neurons, num_inputs) * np.sqrt(1 / num_inputs)
        self.bias = np.zeros((num_neurons, 1))
        
        # Pull the activation function
        self.activation = activation
        self.activation_function, self.activation_function_derivative = self._get_activation(activation=self.activation)
    
    def _get_activation(self, activation:str) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        if activation == "sigmoid":
            return sigmoid, sigmoid_derivative
        elif activation == "relu":
            return relu, relu_derivative
        elif activation == "tanh":
            return tanh, tanh_derivative
        elif activation == "softmax":
            return softmax, softmax_derivative
        else:
            raise ValueError("Choose a proper activation function")

class NeuralNetwork:
    def __init__(self):
        self.layers : list = []
    
    def add_layer(self, num_neurons: int, activation: str, input_size: int = None) -> None:
        
        # Check if the layer is an input layer
        if not self.layers:
            if input_size is None:
                raise ValueError("No input size passed")

            # Add the layer to the list.
            self.layers.append(Layer(input_size, num_neurons, activation))
        
        else:
            # Get previous layers neuron number
            previous_layer_neurons : int = self.layers[-1].weights.shape[1]
            self.layers.append(Layer(previous_layer_neurons, num_neurons, activation))
    
    def forward_propagation(self, X: Sequence) -> None:
        # Define layer input
        input_ = X
        
        for layer in self.layers:
            layer.input = input_
            
            # Compute the z and activation
            z = np.dot(input_, layer.weights) + layer.bias
            layer.output = layer.activation_function(z)
            
            # Define input for the next layer
            input_ = layer.output
        
        return input_
    
    def backward_propagation(self, X: Sequence, y: Sequence, learning_rate: float) -> None:
        # Define some parameters first.
        num_examples = y.shape[0] # Can also use X.shape[1]
        last_layer = self.layers[-1]
        loss_gradient = cross_entropy_derivative(y_pred=last_layer.output, y=y)
        
        # Gradient descent algorithm.
        for l in reversed(len(self.layers)):
            layer = self.layers[l]
            dZ = loss_gradient
            dZ = dZ * layer.activation_function_derivative(layer.output)
            dW = np.dot(layer.input.T, dZ) / num_examples
            dB = np.sum(dZ, axis=1, keepdims=True)
            
            # Compute new loss.
            loss_gradient = np.dot(dZ, layer.weights.T)
            
            # Update the weights and biases.
            layer.weights -= learning_rate * dW
            layer.bias -= learning_rate * dB
    
    def train(self, X: Sequence, y: Sequence, epochs:int, learning_rate:float):
        
        for epoch in range(epochs):
            # Forward pass
            preds = self.forward_propagation(X=X)
            
            # Calculate loss.
            loss = cross_entropy_loss(y_pred=preds, y_true=y)
            
            # Backward pass
            self.backward_propagation(X=X, y=y, learning_rate=learning_rate)
            
            print(f"Epoch : {epoch}, Loss : {loss}")
    
    def show_weights(self):
        for layer in self.layers:
            print("="*100)
            print("Layer_")
            print(f"Weights :- {layer.weights}")
            print(f"Bias : {layer.bias}")
