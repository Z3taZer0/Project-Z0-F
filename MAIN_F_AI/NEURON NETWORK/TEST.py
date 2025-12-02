import numpy as np
from C_Neuron import NEURON

# Création d’un neurone avec 3 entrées :
neuron = NEURON(n_inputs=10, activation="sigmoid")

# Exemple d'entrée :
x = np.array([0.5, 1.2, -0.3, 0.7, -1.5, 2.0, 0.0, 1.1, -0.8, 0.4])

# Calcul de la sortie du neurone (le test quoi) :
output = neuron.forward(x)

##print("Sortie du neurone :", output)

# Test d'entraînement simple :

#Valeurs d'entree :
X_train = np.array([
    [0.5, 1.2, -0.3, 0.7, -1.5, 2.0, 0.0, 1.1, -0.8, 0.4],
    [1.0, -0.5, 0.3, -1.2, 0.8, -0.9, 1.5, -0.4, 0.6, -0.1],
    [-0.7, 0.9, 1.3, -0.6, 0.2, 1.0, -1.1, 0.5, -0.2, 0.8],
    [0.3, -1.4, 0.7, 1.2, -0.5, 0.6, -0.8, 1.4, -0.9, 0.2]
])

# Labels cibles :
y_train = np.array([1, 0, 1, 0])

from train import train

train(neuron, X_train, y_train, lr=0.3, epochs=5000)

# Vérification après entraînement

print("Sortie du neurone après entraînement :", neuron.forward(X_train[1]))