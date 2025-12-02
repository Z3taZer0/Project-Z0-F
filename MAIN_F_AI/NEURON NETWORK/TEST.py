import numpy as np
from C_Neuron import NEURON

# Création d’un neurone avec 10 entrées :
neuron = NEURON(n_inputs=10, activation="sigmoid")

# Exemple d'entrée :
x = np.array([0.5, 1.2, -0.3, 0.7, -1.5, 2.0, 0.0, 1.1, -0.8, 0.4])

# Calcul de la sortie du neurone (le test quoi) :
print("Output expected : 1")
print("Sortie du neurone :", neuron.forward(x))

# Test d'entraînement simple :

#Valeurs d'entree :
X_train = np.array([
    [0.5, 1.2, -0.3, 0.7, -1.5, 2.0, 0.0, 1.1, -0.8, 0.4], #array 0
    [1.0, -0.5, 0.3, -1.2, 0.8, -0.9, 1.5, -0.4, 0.6, -0.1], #array 1   
    [-0.7, 0.9, 1.3, -0.6, 0.2, 1.0, -1.1, 0.5, -0.2, 0.8], #array 2
    [0.3, -1.4, 0.7, 1.2, -0.5, 0.6, -0.8, 1.4, -0.9, 0.2] #array 3
])

# Labels cibles :
y_train = np.array([1, 0, 1, 0]) #valeurs cibles associées aux arrays d'entrée

neuron2 = NEURON(n_inputs=10, activation="sigmoid") # Second neurone pour test avec un learning rate plus élevé et un epochs plus grand

from train import train # Import de la fonction d'entraînement depuis le fichier train.py

train(neuron, X_train, y_train, lr=0.1, epochs=500)
train(neuron2, X_train, y_train, lr=0.5, epochs=50000)
# Vérification après entraînement

print("Sortie du neurone après entraînement :", neuron.forward(X_train[0]))
print("Sortie du neurone 2 apres entrainement :", neuron2.forward(X_train[0]))