import numpy as np
from C_Layer import Layer


class NeuralNetwork:
    def __init__(self, layers):
        
        self.layers = layers # Liste des couches du réseau

    def forward(self, inputs):
        output = inputs # Propagation avant à travers chaque couche
        for layer in self.layers:
            output = layer.forward(output) # Mise à jour de la sortie à chaque couche
        return output
    
def train(nn, X, y, lr=0.1, epochs=1000):
    for _ in range(epochs):
        for inputs, target in zip(X, y):

            # Forward pass
            output = nn.forward(inputs)

            # Erreur de la sortie
            error = output - target

            # Backpropagation
            delta = error * (output * (1 - output))  # dérivée de la fonction sigmoïde

            # On remonte couche par couche depuis la fin
            for layer in reversed(nn.layers):
                # gradient poids
                grad_w = np.outer(delta, layer.last_input)
                grad_b = delta

                # mise à jour
                layer.weights -= lr * grad_w
                layer.biases -= lr * grad_b

                # calcul delta pour la couche précédente
                delta = (layer.weights.T @ delta) * (layer.last_output * (1 - layer.last_output))