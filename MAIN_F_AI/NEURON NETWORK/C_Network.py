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