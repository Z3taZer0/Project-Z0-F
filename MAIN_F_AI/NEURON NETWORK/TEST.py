import numpy as np
from C_Network import NeuralNetwork, train
from C_Layer import Layer
from SaveSystem import save_network, load_network
from Visual import visualize_network

# Exemple d'entrées pour entraîner le réseau de neurones :
X = np.array([
    [0.6, -1.5],   #array 0
    [1.0, 0.5]     #array 1
    ])

y = np.array([1, 0])  # Valeurs cibles associées aux arrays d'entrée (entre 0 et 1 pour sigmoïde)

# Création du réseau de neurones avec 3 couches, chaque neurone a une sortie et donc une entree pour la couche suivante
nn = NeuralNetwork([
    Layer(n_inputs=2, n_neurons=3),  # Couche d'entrée avec 2 entrées et 3 neurones
    Layer(n_inputs=3, n_neurons=2),  # Couche cachée avec 3 entrées et 2 neurones
    Layer(n_inputs=2, n_neurons=1)   # Couche de sortie avec 2 entrées et 1 neurone
])

load_network(nn, "network_data.npz")  # Chargement du réseau sauvegardé précédemment


train(nn, X, y, lr=0.5, epochs=1000000) # Entraînement du réseau de neurones

print("Sortie du réseau de neurones après entraînement :", nn.forward(X[0])) # Test de la sortie du réseau après entraînement (array 0)
print("Sortie du réseau de neurones après entraînement :", nn.forward(X[1])) # Test de la sortie du réseau après entraînement (array 1)

# Test de la sauvegarde et du chargement du réseau de neurones :

save_network(nn, "network_data.npz")  # Sauvegarde du réseau entraîné

nn2 = NeuralNetwork([
    Layer(n_inputs=2, n_neurons=3),
    Layer(n_inputs=3, n_neurons=2),
    Layer(n_inputs=2, n_neurons=1)
])

load_network(nn2, "network_data.npz")  # Chargement du réseau sauvegardé dans un nouveau réseau

print("Sortie du réseau de neurones chargé :", nn2.forward(X[0])) # Test de la sortie du réseau chargé (array 0)
print("Sortie du réseau de neurones chargé :", nn2.forward(X[1])) # Test de la sortie du réseau chargé (array 1)

visualize_network(nn2)  # Visualisation du réseau de neurones entraîné