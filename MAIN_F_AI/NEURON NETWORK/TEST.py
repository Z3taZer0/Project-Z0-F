import numpy as np
from  C_Network import NeuralNetwork
from C_Layer import Layer
from C_Network import train

# Exemple d'entrée

X = np.array([
    [[0.6, 2.5], [-1.5, 0.4]],
    [[1.0, -0.5], [0.3, -1.2]],
    [[-0.7, 0.9], [1.3, -0.6]],
    [[0.3, -1.4], [0.7, 1.2]]
    ])

y = np.array([1, 0, 1, 0])

nn = NeuralNetwork([
    Layer(2, 3),  # Couche d'entrée avec 2 neurones
    Layer(3, 2),  # Couche cachée avec 3 neurones
    Layer(2, 1)   # Couche de sortie avec 1 neurone
])

train(nn, X, y, lr=0.1, epochs=1000)

print("Sortie du réseau de neurones après entraînement :", nn.forward(X[0]))