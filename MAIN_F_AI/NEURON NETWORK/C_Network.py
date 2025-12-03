import numpy as np
from C_Layer import Layer


class NeuralNetwork:
    def __init__(self, layers):
        
        self.layers = layers # Liste des couches du réseau

    def forward(self, inputs):
        self.activations = [inputs]
        output = inputs # Propagation avant à travers chaque couche
        for layer in self.layers:
            output = layer.forward(output) # Mise à jour de la sortie à chaque couche
            self.activations.append(output)
        return output
    
def train(nn, X, y, lr=0.1, epochs=1000):
    for _ in range(epochs):
        for inputs, target in zip(X, y):

            # Forward pass
            output = nn.forward(inputs)

            # Erreur de la sortie
            error = output - target
            delta = error * (output * (1 - output))  # dérivée de la fonction sigmoïde

             # Backprop couche par couche
            for i in reversed(range(len(nn.layers))):
                layer = nn.layers[i]

                # Activation de la couche précédente
                a_prev = nn.activations[i]      # shape = n_inputs
                a_curr = nn.activations[i+1]    # shape = n_neurons

                # gradients
                grad_w = np.outer(delta, a_prev)
                grad_b = delta

                # update
                layer.weights -= lr * grad_w
                layer.biases  -= lr * grad_b

                # delta pour la couche précédente (si pas input layer)
                if i > 0:
                    delta = (layer.weights.T @ delta) * (a_prev * (1 - a_prev))