import numpy as np
from typing import Callable, Tuple

# Activation functions and their derivatives
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Add small constant for numerical stability
    return loss

def cross_entropy_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return y_pred - y_true

class Layer:
    def __init__(self, num_inputs: int, num_neurons: int, activation: str):
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.1
        self.biases = np.zeros((1, num_neurons))
        self.activation_function, self.activation_derivative = self._get_activation(activation)
        self.output = None
        self.input = None

    def _get_activation(self, activation: str) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        if activation == "relu":
            return relu, relu_derivative
        elif activation == "softmax":
            return softmax, None
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, num_neurons: int, activation: str, input_size: int = None):
        if not self.layers:
            if input_size is None:
                raise ValueError("Input size must be specified for the first layer.")
            self.layers.append(Layer(input_size, num_neurons, activation))
        else:
            prev_layer_neurons = self.layers[-1].weights.shape[1]
            self.layers.append(Layer(prev_layer_neurons, num_neurons, activation))

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            layer.input = output
            z = np.dot(output, layer.weights) + layer.biases
            if layer.activation_function:
                layer.output = layer.activation_function(z)
            else:
                layer.output = z  # For layers like softmax (final layer)
            output = layer.output
        return output

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float):
        # Loss gradient for output layer
        m = y.shape[0]
        last_layer = self.layers[-1]
        loss_gradient = cross_entropy_derivative(last_layer.output, y)  # dL/dy_pred

        # Backpropagate
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            dZ = loss_gradient
            if i < len(self.layers) - 1:  # Skip softmax derivative, already handled
                dZ = dZ * layer.activation_derivative(layer.output)  # dL/dZ
            dW = np.dot(layer.input.T, dZ) / m  # dL/dW
            dB = np.sum(dZ, axis=0, keepdims=True) / m  # dL/dB
            loss_gradient = np.dot(dZ, layer.weights.T)  # Propagate to previous layer

            # Update weights and biases
            layer.weights -= learning_rate * dW
            layer.biases -= learning_rate * dB

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Compute loss
            loss = cross_entropy_loss(predictions, y)

            # Backward pass
            self.backward(X, y, learning_rate)

            # Print loss every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage for MNIST
if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Load MNIST dataset
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data
    y = mnist.target.astype(int)

    # Normalize and preprocess the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the neural network
    nn = NeuralNetwork()
    nn.add_layer(num_neurons=128, activation="relu", input_size=784)  # Input layer
    nn.add_layer(num_neurons=64, activation="relu")  # Hidden layer
    nn.add_layer(num_neurons=10, activation="softmax")  # Output layer

    # Train the network
    nn.train(X_train, y_train, epochs=100, learning_rate=0.01)