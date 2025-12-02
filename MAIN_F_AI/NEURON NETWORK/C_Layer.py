import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons):
        # Poids et biais pour toute la couche
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.random.randn(n_neurons)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) # Definition de la fonction Sigmoïde

    def forward(self, inputs):
        self.last_input = inputs #Sauvegarde les inputs pour le backpropagation
        self.last_z = np.dot(self.weights, inputs) + self.biases #Calcul de z pour toute la couche
        self.last_output = self.sigmoid(self.last_z) #Application de la fonction d'activation sigmoïde
        return self.last_output