import numpy as np

# Definition de la classe NEURON
class NEURON:
    def __init__(self, n_inputs, activation="sigmoid"): # Initialisation du neurone
        # Multiplicateurs aléatoires :
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
        
        # Choix de la fonction d'activation (sigmoïde de base):
        if activation == "sigmoid":
            self.activation = lambda x: 1 / (1 + np.exp(-x)) # Fonction Sigmoïde
        elif activation == "relu":
            self.activation = lambda x: np.maximum(0, x) # Fonction ReLU
        else:
            self.activation = lambda x: x  # Fonction Linéaire
    
    def forward(self, inputs):
        # Produit scalaire + biais, puis application de la fonction d'activation :
        z = np.dot(self.weights, inputs) + self.bias
        return self.activation(z)