import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from NeuralNetwork import NeuralNetwork

mnist = fetch_openml(name='mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target
print(f"Raw Features Shape: {X.shape}")
print(f"Raw Labels Shape: {y.shape}") 

# Normalize X
X = X / 255.0
y = y.astype(int)

encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
X_train = np.reshape(X_train, shape=(784, 56000))
print(f"Train shape: {X_train.shape}, {y_train.shape}")
#print(f"Test shape: {X_test.shape}, {y_test.shape}")

nn = NeuralNetwork()
nn.add_layer(input_size=784, num_neurons=256, activation="relu")
nn.add_layer(num_neurons=128, activation="relu")
nn.add_layer(num_neurons=64, activation="relu")
nn.add_layer(num_neurons=10, activation="softmax")

nn.train(X=X_train, y=y_train, epochs=50, learning_rate=0.0001)