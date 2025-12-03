import numpy as np

#Definition de la classe Layer, représentant une couche de un ou plusieurs neurones
class Layer:
    def __init__(self, n_inputs, n_neurons):   # Initialisation de la couche avec le nombre d'entrées et de neurones
        
        # Poids et biais aleatiores pour toute la couche de neurones :
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.random.randn(n_neurons)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) # Definition de la fonction Sigmoïde (utilisée comme fonction d'activation)

    def forward(self, inputs): # Propagation avant à travers la couche
        self.last_input = inputs #Sauvegarde les inputs pour le backpropagation
        self.last_z = np.dot(self.weights, inputs) + self.biases #Calcul de z pour chaque neurone de la couche
        self.last_output = self.sigmoid(self.last_z) #Application de la fonction d'activation sigmoïde sur tous les z de la couche
        return self.last_output