import numpy as np
from  C_Network import NeuralNetwork
from C_Layer import Layer
from C_Network import train

# Exemple d'entrées pour entraîner le réseau de neurones :
X = np.array([
    [0.6, -1.5], #array 0
    [1.0, 0.5] #array 1
    ])

y = np.array([1, 0])  # Valeurs cibles associées aux arrays d'entrée (entre 0 et 1 pour sigmoïde)

# Création du réseau de neurones avec 3 couches, chaque neurone represente une entree pour la couche suivante
nn = NeuralNetwork([
    Layer(n_inputs=2, n_neurons=3),  # Couche d'entrée avec 3 neurones et 2 entrées
    Layer(n_inputs=3, n_neurons=2),  # Couche cachée avec 2 neurones et 3 entrées
    Layer(n_inputs=2, n_neurons=1)   # Couche de sortie avec 1 neurone et 2 entrées
])

train(nn, X, y, lr=0.5, epochs=10000) # Entraînement du réseau de neurones

print("Sortie du réseau de neurones après entraînement :", nn.forward(X[0])) # Test de la sortie du réseau après entraînement (array 0)
print("Sortie du réseau de neurones après entraînement :", nn.forward(X[1])) # Test de la sortie du réseau après entraînement (array 1)