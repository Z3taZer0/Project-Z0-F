import numpy as np
from C_Neuron import NEURON

def train(neuron, X, y, lr=0.1, epochs=1000): #epochs = nombre d'itérations d'entraînement, et lr = learning rate
    for _ in range(epochs):
        for inputs, target in zip(X, y):
            # on test la prédiction du neurone :
            z = np.dot(neuron.weights, inputs) + neuron.bias #calcul de z
            pred = neuron.activation(z) # application de la fonction d'activation (sigmoïde uniquement jsp pk ca marche pas avec les autres mais osef)
            # calcul de l'erreur :
            error = pred - target # différence entre la prédiction et la cible y
            
            # gradient (fonction sigmoide dérivée) :
            grad_pred = pred * (1 - pred)
            
            # update du neurone :
            neuron.weights -= lr * error * grad_pred * inputs # mise à jour des poids (plus d'aleatoire)
            neuron.bias -= lr * error * grad_pred # mise à jour du biais