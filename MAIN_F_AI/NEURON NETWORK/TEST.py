import numpy as np
from  C_Network import NeuralNetwork
from C_Layer import Layer
from C_Network import train

# Exemple d'entrée

X = np.array([
    [0.6, -1.5],
    [1.0, 0.5]
    ])

y = np.array([1, 0])  # Valeurs cibles associées aux arrays d'entrée

nn = NeuralNetwork([
    Layer(n_inputs=2, n_neurons=3),  # Couche d'entrée avec 3 neurones et 2 entrées
    Layer(n_inputs=3, n_neurons=2),  # Couche cachée avec 2 neurones et 3 entrées
    Layer(n_inputs=2, n_neurons=1)   # Couche de sortie avec 1 neurone et 2 entrées
])

train(nn, X, y, lr=0.5, epochs=10000)

print("Sortie du réseau de neurones après entraînement :", nn.forward(X[1]))